from __future__ import annotations

import copy
import os
import sys
from collections.abc import Sequence
from types import ModuleType
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Callable, ClassVar, Protocol, cast

import numpy as np
import torch
from torch import Tensor

import streaming
from numpy.typing import NDArray
from omegaconf import DictConfig

if TYPE_CHECKING:
    from nemo.collections.asr.models import (
        EncDecHybridRNNTCTCBPEModel,
        EncDecRNNTModel,
    )
    from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
    from nemo.collections.asr.modules import ConformerEncoder
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    from omegaconf import OmegaConf as OmegaConfClass

Int16Array = NDArray[np.int16]
FloatArray = NDArray[np.float32]
OpenDictCallable = Callable[[DictConfig], AbstractContextManager[object]]


class GreedyDecodingConfig(Protocol):
    max_symbols: int


class DecodingConfigProtocol(Protocol):
    strategy: str
    preserve_alignments: bool
    greedy: GreedyDecodingConfig
    fused_batch_size: int


class PreprocessorConfigProtocol(Protocol):
    dither: float
    pad_to: int
    normalize: str
    features: int

class ASRStreamer:
    """ASR Streamer

    This class runs a streaming speech-recognition model, processing each new chunk
    with the context of previously-processed chunks.
    """

    ENCODER_STEP_LENGTH: ClassVar[int] = 80  # ms
    SAMPLE_RATE: ClassVar[int] = 16000 # Hz

    def __init__(
        self,
        model_name: str,
        lookahead_size: int,
        decoder_type: str = "rnnt",
        decoding_strategy: str = "greedy",
        verbose: bool = True,
    ) -> None:
        """
        Initialize the ASRStreamer class.

        Args:
            model_name (str): The name of the pre-trained ASR model to use.
            lookahead_size (int): The size of the lookahead window for streaming.
            decoder_type (str, optional): Type of decoder ("rnnt", "transformer", etc.). Defaults to "rnnt".
            decoding_strategy (str, optional): Decoding strategy ("greedy", "beam"). Defaults to "greedy".
            verbose (bool, optional): Whether to print detailed initialization messages. Defaults to True.
        """
        self.model_name: str = model_name
        self.lookahead_size: int = lookahead_size
        self.decoder_type: str = decoder_type
        self.decoding_strategy: str = decoding_strategy
        self.verbose: bool = verbose

        self.OmegaConf: type[OmegaConfClass] | None = None
        self.open_dict: OpenDictCallable | None = None
        self.EncDecCTCModelBPE: type[EncDecCTCModelBPE] | None = None
        self.Hypothesis: type[Hypothesis] | None = None
        self.nemo_asr: ModuleType | None = None
        self._encdec_rnnt_cls: type[EncDecRNNTModel] | None = None
        self._encdec_hybrid_cls: type[EncDecHybridRNNTCTCBPEModel] | None = None
        self._conformer_encoder_cls: type[ConformerEncoder] | None = None
        self.asr_model: EncDecHybridRNNTCTCBPEModel | None = None
        self.preprocessor: EncDecCTCModelBPE | None = None
        self.device: torch.device | None = None
        self.previous_hypotheses: Sequence[Hypothesis] | Sequence[Sequence[Hypothesis]] | None = None
        self.pred_out_stream: list[Tensor] = []
        self.cache_pre_encode: Tensor | None = None
        self.cache_last_channel: Tensor | None = None
        self.cache_last_time: Tensor | None = None
        self.cache_last_channel_len: Tensor | None = None
        self.pre_encode_cache_size: int = 0
        self.step_num: int = 0
        self.buffer: str = ""

        print(f"Loading NeMo ASR package and initializing {decoder_type} model with {decoding_strategy} decoding strategy...")
        self._import_nemo_packages()
        self._init_model()
        self._init_streaming_state()
        self._warm_up_model()
        print("...done initializing ASR model.")

    def _require_asr_model(self) -> EncDecHybridRNNTCTCBPEModel:
        if self.asr_model is None:
            raise RuntimeError("ASR model not initialized")
        return self.asr_model

    def _require_conformer_encoder_cls(self) -> type[ConformerEncoder]:
        if self._conformer_encoder_cls is None:
            raise RuntimeError("Conformer encoder class not initialized")
        return self._conformer_encoder_cls

    def _require_rnnt_cls(self) -> type[EncDecRNNTModel]:
        if self._encdec_rnnt_cls is None:
            raise RuntimeError("RNNT model class not initialized")
        return self._encdec_rnnt_cls

    @streaming.suppress_terminal_output
    def _import_nemo_packages(self) -> None:
        """Import required packages for NeMo ASR and store some of the imported
        class referenes for easy reference by other class methods.   This methods also
        serves to hide the messy terminal output that importing from nemo tend to generate.
        """
        from omegaconf import OmegaConf, open_dict
        self.OmegaConf = OmegaConf
        self.open_dict = open_dict
        import nemo.collections.asr as nemo_asr

        self.nemo_asr = nemo_asr
        from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE

        self.EncDecCTCModelBPE = EncDecCTCModelBPE
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

        self.Hypothesis = Hypothesis
        self._encdec_hybrid_cls = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
        self._encdec_rnnt_cls = nemo_asr.models.EncDecRNNTModel
        self._conformer_encoder_cls = nemo_asr.modules.ConformerEncoder

    def _init_model(self) -> None:
        """
        Initialize the ASR model using the provided model name.
        Adjusts the encoder settings if using a streaming model with a lookahead size other than default values.
        Sets the decoding strategy based on the specified decoding type and strategy.

        Raises:
            ValueError: If an invalid lookahead_size is provided for the streaming ConformerEncoder.
        """
        # set device intelligently
        if torch.cuda.is_available():
            map_location = torch.device('cuda:0')  # use 0th CUDA device
            print("...using NVIDIA GPU cuda:0...")
        elif torch.backends.mps.is_available():
            print("...using MPS (Apple Silicon acceleration)...")
            mps_fallback = os.getenv("PYTORCH_ENABLE_MPS_FALLBACK")
            if not mps_fallback:
                print("WARNING: Environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` might need to be set to avoid failures.")
            map_location = torch.device('mps')
        else:
            sys.exit("No GPU detected!  This probably won't run well on your cpu.  Exiting...")

        # load model
        hybrid_cls = self._encdec_hybrid_cls
        if hybrid_cls is None:
            raise RuntimeError("NeMo classes not initialized")
        asr_model = hybrid_cls.from_pretrained(model_name=self.model_name, map_location=map_location)
        self.asr_model = asr_model
        if self.model_name == "stt_en_fastconformer_hybrid_large_streaming_multi":
            encoder_cls = self._require_conformer_encoder_cls()
            assert isinstance(asr_model.encoder, encoder_cls)
            if self.lookahead_size not in [0, 80, 480, 1040]:
                raise ValueError(f"Invalid lookahead_size {self.lookahead_size}")
            left_context_size = asr_model.encoder.att_context_size[0]
            asr_model.encoder.set_default_att_context_size(
                [left_context_size, int(self.lookahead_size / ASRStreamer.ENCODER_STEP_LENGTH)]
            )
        # set decoding strategy
        asr_model.change_decoding_strategy(decoder_type=self.decoder_type)
        decoding_cfg = cast(DictConfig, asr_model.cfg.decoding)
        decoding_view = cast(DecodingConfigProtocol, cast(object, decoding_cfg))
        if self.open_dict is None:
            raise RuntimeError("OmegaConf open_dict helper not initialized")
        with self.open_dict(decoding_cfg):
            # save time by doing greedy decoding and not trying to record the alignments
            if self.decoding_strategy == "greedy":
                decoding_view.strategy = "greedy"
                decoding_view.preserve_alignments = False
                if hasattr(asr_model, 'joint'):  # if an RNNT model
                    # restrict max_symbols to make sure not stuck in infinite loop
                    decoding_view.greedy.max_symbols = 50
                    # sensible default parameter, but not necessary since batch size is 1
                    decoding_view.fused_batch_size = -1
                asr_model.change_decoding_strategy(decoding_cfg)
            elif self.decoding_strategy == "beam":
                decoding_view.strategy = "beam"
                asr_model.change_decoding_strategy(decoding_cfg)
            else:
                raise ValueError(f"Invalid decoding strategy {self.decoding_strategy} for {self.decoder_type} model!")

        # set model into inference mode (as opposed to training), and re
        _ = asr_model.eval()
        self.device = asr_model.device

        # initiallize audio preprocessor
        self.preprocessor = self._init_preprocessor()

    def _init_preprocessor(self) -> "EncDecCTCModelBPE":
        """
        Initialize the audio preprocessor using the configuration from the ASR model.

        This method sets up the preprocessing steps such as dithering, padding, and normalization,
        which are essential for preparing the audio data before inference.

        Returns:
            EncDecCTCModelBPE: The initialized audio preprocessor.
        """
        if self.EncDecCTCModelBPE is None or self.OmegaConf is None:
            raise RuntimeError("ASR model dependencies not initialized")
        asr_model = self._require_asr_model()
        cfg = copy.deepcopy(asr_model._cfg)
        raw_preprocessor_cfg = cast(DictConfig, cfg.preprocessor)
        preprocessor_cfg = cast(PreprocessorConfigProtocol, cast(object, raw_preprocessor_cfg))
        self.OmegaConf.set_struct(raw_preprocessor_cfg, False)
        preprocessor_cfg.dither = 0.0
        preprocessor_cfg.pad_to = 0
        preprocessor_cfg.normalize = "None"
        preprocessor = self.EncDecCTCModelBPE.from_config_dict(raw_preprocessor_cfg)
        if self.device is None:
            raise RuntimeError("Model device not initialized")
        _ = preprocessor.to(self.device)
        return preprocessor

    def reset_streaming_state(self) -> None:
        """Clear the transcription and assocated buffers in between transcription sessions,
        effectively applying the model anew on the next inference step."""
        rnnt_cls = self._require_rnnt_cls()
        encoder_cls = self._require_conformer_encoder_cls()
        asr_model = self._require_asr_model()
        assert isinstance(asr_model, rnnt_cls)
        assert isinstance(asr_model.encoder, encoder_cls)
        self.previous_hypotheses = None
        # ensure tensor memory is freed
        if self.pred_out_stream:
            for tensor in self.pred_out_stream:
                del tensor
            self.pred_out_stream.clear()
        self.step_num = 0
        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len
        ) = asr_model.encoder.get_initial_cache_state(batch_size=1)
        if self.cache_pre_encode is None:
            raise RuntimeError("Cache not initialized")
        _ = self.cache_pre_encode.zero_()

    def _init_streaming_state(self) -> None:
        """
        Initialize the streaming state required for speech recognition.

        This method initializes necessary variables and cache states that will be used
        to store intermediate results during the streaming inference process. These include
        previous hypotheses, prediction output stream, step number, and initial cache states
        for the encoder.
        """
        self.previous_hypotheses = None
        self.pred_out_stream = []
        rnnt_cls = self._require_rnnt_cls()
        encoder_cls = self._require_conformer_encoder_cls()
        asr_model = self._require_asr_model()
        assert isinstance(asr_model, rnnt_cls)
        assert isinstance(asr_model.encoder, encoder_cls)
        pre_encode_cache_sizes = asr_model.encoder.streaming_cfg.pre_encode_cache_size
        # hack for typechecking on nemo's erroneous CacheAwareStreamingConfig definition... (pre_encode_cache_size is actually of type list[int])
        if isinstance(pre_encode_cache_sizes, int):
            self.pre_encode_cache_size = pre_encode_cache_sizes
        else:
            self.pre_encode_cache_size = int(pre_encode_cache_sizes[1])
        preprocessor_cfg = cast(
            PreprocessorConfigProtocol,
            cast(object, asr_model.cfg.preprocessor),
        )
        num_channels = int(preprocessor_cfg.features)
        if self.device is None:
            raise RuntimeError("Model device not initialized")
        self.cache_pre_encode = torch.zeros(
            size=(1, num_channels, self.pre_encode_cache_size), device=self.device
        )
        self.reset_streaming_state()

    def preprocess_audio(self, audio: FloatArray) -> tuple[Tensor, Tensor]:
        """
        Preprocess the input audio chunk to be ready for inference by the ASR model.

        Args:
            audio (np.ndarray): The raw audio data chunk as a numpy array of int16 type.

        Returns:
            torch.Tensor: The preprocessed audio signal tensor.
            torch.Tensor: The length of the processed audio signal tensor.
        """
        device = self.device
        preprocessor = self.preprocessor
        if preprocessor is None or device is None:
            raise RuntimeError("Preprocessor not initialized")
        from_numpy = cast(Callable[[FloatArray], Tensor], torch.from_numpy)
        audio_signal = from_numpy(audio).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
        processed_signal, processed_signal_length = cast(
            tuple[Tensor, Tensor],
            preprocessor(input_signal=audio_signal, length=audio_signal_len),
        )
        del audio_signal
        del audio_signal_len
        return processed_signal, processed_signal_length

    def _extract_transcriptions(
        self,
        hyps: Sequence["Hypothesis"] | Sequence[Sequence["Hypothesis"]],
    ) -> list[str]:
        """Extract transcriptions from hypotheses based on decoding strategy.

        Args:
            hyps (list): List of hypotheses. Each hypothesis can be either a list or a single instance of Hypothesis.

        Returns:
            list: A list of transcribed texts.
        """
        if self.Hypothesis is None:
            raise RuntimeError("Hypothesis class not initialized")
        if not hyps:
            return []
        first_entry = hyps[0]
        if isinstance(first_entry, self.Hypothesis):
            greedy_hyps = cast(Sequence["Hypothesis"], hyps)
            return [(hyp.text or "") for hyp in greedy_hyps]
        # TODO return both dimensions for beam search
        beam_hyps = cast(Sequence[Sequence["Hypothesis"]], hyps)
        best_beam = beam_hyps[0]
        return [(hyp.text or "") for hyp in best_beam]

    def _extract_most_likely_transcription(
        self,
        hyps: Sequence["Hypothesis"] | Sequence[Sequence["Hypothesis"]],
    ) -> str:
        """Extract and return the best transcription among hypotheses based on the decoding strategy.

        Args:
            hyps (list): List of hypotheses. Each hypothesis can be either a list or a single instance of Hypothesis.

        Returns:
            str: The most likely transcribed text.
        """
        return self._extract_transcriptions(hyps)[0]

    def process_chunk(self, signal: Int16Array, limited_cache: bool = False) -> str:
        """
        Process an audio chunk and generate transcription using the streaming ASR model.

        Args:
            audio_chunk (np.ndarray): The raw audio data chunk as a numpy array of int16 type.
            limited_cache (bool, optional): Whether to use limited cache for processing. Defaults to False.

        Returns:
            str: The transcribed text from the current audio chunk.
        """
        # convert from np.int16
        _ = limited_cache  # reserved for future cache tuning hooks
        audio_data = signal.astype(np.float32) / 32768.0
        # get mel-spectrogram signal & length
        processed_signal, processed_signal_len = self.preprocess_audio(audio_data)

        # prepend with cache_pre_encode
        #print(f"cache length: {self.cache_pre_encode.shape[1]}")
        if self.cache_pre_encode is None:
            raise RuntimeError("Cache not initialized")
        processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
        processed_signal_len += self.cache_pre_encode.shape[1]

        # store for next time
        self.cache_pre_encode = processed_signal[:, :, -self.pre_encode_cache_size:]

        # Inference
        if self.asr_model is None:
            raise RuntimeError("ASR model not initialized")
        if self.cache_last_channel is None or self.cache_last_time is None or self.cache_last_channel_len is None:
            raise RuntimeError("Cache state not initialized")
        previous_hypotheses = self.previous_hypotheses
        previous_pred_out = self.pred_out_stream if previous_hypotheses else None
        with torch.no_grad():
            (
                self.pred_out_stream,
                transcribed_texts,
                self.cache_last_channel,
                self.cache_last_time,
                self.cache_last_channel_len,
                self.previous_hypotheses,
            ) = self.asr_model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_len,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=None,
                return_transcription=True,
                return_log_probs=False,
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=previous_pred_out,
            )
        hypotheses = transcribed_texts
        self.step_num += 1
        # extract transcription
        transcription = self._extract_most_likely_transcription(hypotheses)
        del processed_signal
        del processed_signal_len
        return transcription


    def _warm_up_model(self) -> None:
        """
        Warm up the ASR model by processing a chunk of silence.
        """
        num_frames = int(ASRStreamer.SAMPLE_RATE*(self.lookahead_size + ASRStreamer.ENCODER_STEP_LENGTH) / 1000) - 1
        silence = np.zeros(num_frames, dtype=np.int16)
        _ = self.process_chunk(silence)
