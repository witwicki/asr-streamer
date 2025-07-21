from typing import Tuple
import sys, os
import copy
import torch
import numpy as np
import streaming

class ASRStreamer:
    """ASR Streamer

    This class runs a streaming speech-recognition model, processing each new chunk
    with the context of previously-processed chunks.
    """

    ENCODER_STEP_LENGTH = 80  # ms
    SAMPLE_RATE = 16000 # Hz

    def __init__(self, model_name, lookahead_size, decoder_type="rnnt", decoding_strategy="greedy", verbose=True):
        """
        Initialize the ASRStreamer class.

        Args:
            model_name (str): The name of the pre-trained ASR model to use.
            lookahead_size (int): The size of the lookahead window for streaming.
            decoder_type (str, optional): Type of decoder ("rnnt", "transformer", etc.). Defaults to "rnnt".
            decoding_strategy (str, optional): Decoding strategy ("greedy", "beam"). Defaults to "greedy".
            verbose (bool, optional): Whether to print detailed initialization messages. Defaults to True.
        """
        self.model_name = model_name
        self.lookahead_size = lookahead_size
        self.decoder_type = decoder_type
        self.decoding_strategy = decoding_strategy
        self.verbose = verbose

        print(f"Loading NeMo ASR package and initializing {decoder_type} model with {decoding_strategy} decoding strategy...")
        self._import_nemo_packages()
        self._init_model()
        self._init_streaming_state()
        self._warm_up_model()
        self.buffer = "" # the buffer of recognized text
        print("...done initializing ASR model.")

    @streaming.show_terminal_output
    def _import_nemo_packages(self):
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

    def _init_model(self):
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
        self.asr_model = self.nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name, map_location=map_location)
        assert isinstance(self.asr_model, self.nemo_asr.models.EncDecHybridRNNTCTCBPEModel)
        if self.model_name == "stt_en_fastconformer_hybrid_large_streaming_multi":
            assert isinstance(self.asr_model.encoder, self.nemo_asr.modules.ConformerEncoder)
            if self.lookahead_size not in [0, 80, 480, 1040]:
                raise ValueError(f"Invalid lookahead_size {self.lookahead_size}")
            left_context_size = self.asr_model.encoder.att_context_size[0]
            self.asr_model.encoder.set_default_att_context_size(
                [left_context_size, int(self.lookahead_size / ASRStreamer.ENCODER_STEP_LENGTH)]
            )
        # set decoding strategy
        self.asr_model.change_decoding_strategy(decoder_type=self.decoder_type)
        decoding_cfg = self.asr_model.cfg.decoding
        with self.open_dict(decoding_cfg):
            # save time by doing greedy decoding and not trying to record the alignments
            if self.decoding_strategy == "greedy":
                decoding_cfg.strategy = "greedy"
                decoding_cfg.preserve_alignments = False
                if hasattr(self.asr_model, 'joint'):  # if an RNNT model
                    # restrict max_symbols to make sure not stuck in infinite loop
                    decoding_cfg.greedy.max_symbols = 50
                    # sensible default parameter, but not necessary since batch size is 1
                    decoding_cfg.fused_batch_size = -1
                self.asr_model.change_decoding_strategy(decoding_cfg)
            elif self.decoding_strategy == "beam":
                decoding_cfg.strategy = "beam"
                self.asr_model.change_decoding_strategy(decoding_cfg)
            else:
                raise ValueError(f"Invalid decoding strategy {self.decoding_strategy} for {self.decoder_type} model!")

        # set model into inference mode (as opposed to training), and re
        assert isinstance(self.asr_model, torch.nn.Module)
        self.asr_model.eval()
        self.device = self.asr_model.device

        # initiallize audio preprocessor
        self.preprocessor = self._init_preprocessor()

    def _init_preprocessor(self):
        """
        Initialize the audio preprocessor using the configuration from the ASR model.

        This method sets up the preprocessing steps such as dithering, padding, and normalization,
        which are essential for preparing the audio data before inference.

        Returns:
            EncDecCTCModelBPE: The initialized audio preprocessor.
        """
        assert isinstance(self.asr_model, self.nemo_asr.models.ASRModel)
        cfg = copy.deepcopy(self.asr_model._cfg)
        self.OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        preprocessor = self.EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        preprocessor.to(self.device)
        return preprocessor

    def _init_streaming_state(self):
        """
        Initialize the streaming state required for speech recognition.

        This method initializes necessary variables and cache states that will be used
        to store intermediate results during the streaming inference process. These include
        previous hypotheses, prediction output stream, step number, and initial cache states
        for the encoder.
        """
        assert isinstance(self.asr_model, self.nemo_asr.models.EncDecRNNTModel)
        assert isinstance(self.asr_model.encoder, self.nemo_asr.modules.ConformerEncoder)
        self.previous_hypotheses = None
        self.pred_out_stream = torch.Tensor()
        self.step_num = 0
        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len
        ) = self.asr_model.encoder.get_initial_cache_state(batch_size=1)
        pre_encode_cache_sizes = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size
        # hack for typechecking on nemo's erroneous CacheAwareStreamingConfig definition... (pre_encode_cache_size is actually of type list[int])
        self.pre_encode_cache_size : int = pre_encode_cache_sizes if isinstance(pre_encode_cache_sizes, int) else pre_encode_cache_sizes[1]
        num_channels = self.asr_model.cfg.preprocessor.features
        assert isinstance(self.device, torch.device)
        self.cache_pre_encode = torch.zeros(
            size=(1, num_channels, self.pre_encode_cache_size), device=self.device
        )

    def reset_streaming_state(self):
        """Clear the transcription and assocated buffers in between transcription sessions,
        effectively applying the model anew on the next inference step."""
        self._init_streaming_state()

    def preprocess_audio(self, audio) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the input audio chunk to be ready for inference by the ASR model.

        Args:
            audio (np.ndarray): The raw audio data chunk as a numpy array of int16 type.

        Returns:
            torch.Tensor: The preprocessed audio signal tensor.
            torch.Tensor: The length of the processed audio signal tensor.
        """
        processed_signal = torch.Tensor()
        processed_signal_length = torch.Tensor()
        assert isinstance(self.device, torch.device)
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(self.device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(self.device)
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length

    def _extract_transcriptions(self, hyps):
        """Extract transcriptions from hypotheses based on decoding strategy.

        Args:
            hyps (list): List of hypotheses. Each hypothesis can be either a list or a single instance of Hypothesis.

        Returns:
            list: A list of transcribed texts.
        """
        if self.decoding_strategy == "greedy":
            assert isinstance(hyps[0], self.Hypothesis)
            return [hyp.text for hyp in hyps]
        else:
            # TODO return both dimensions for beam search
            return [hyp.text for hyp in hyps[0]]
        #print(f"size={len(hyps)}, size of hyps[0]={len(hyps[0])}, hyps={hyps}")
        return ""

    def _extract_most_likely_transcription(self, hyps):
        """Extract and return the best transcription among hypotheses based on the decoding strategy.

        Args:
            hyps (list): List of hypotheses. Each hypothesis can be either a list or a single instance of Hypothesis.

        Returns:
            str: The most likely transcribed text.
        """
        return self._extract_transcriptions(hyps)[0]

    def process_chunk(self, audio_chunk, limited_cache=False):
        """
        Process an audio chunk and generate transcription using the streaming ASR model.

        Args:
            audio_chunk (np.ndarray): The raw audio data chunk as a numpy array of int16 type.
            limited_cache (bool, optional): Whether to use limited cache for processing. Defaults to False.

        Returns:
            str: The transcribed text from the current audio chunk.
        """
        # convert from np.int16
        audio_data = audio_chunk.astype(np.float32) / 32768.0
        # get mel-spectrogram signal & length
        processed_signal, processed_signal_len = self.preprocess_audio(audio_data)

        # prepend with cache_pre_encode
        #print(f"cache length: {self.cache_pre_encode.shape[1]}")
        processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
        processed_signal_len += self.cache_pre_encode.shape[1]

        # store for next time
        self.cache_pre_encode = processed_signal[:, :, -self.pre_encode_cache_size:]

        # Inference
        assert isinstance(self.asr_model, self.nemo_asr.models.EncDecHybridRNNTCTCBPEModel), "Only EncDecHybridRNNTCTCBPEModel is supported"
        # build kwargs dynamically
        kwargs = {
            "processed_signal": processed_signal,
            "processed_signal_length": processed_signal_len,
            "cache_last_channel": self.cache_last_channel,
            "cache_last_time": self.cache_last_time,
            "cache_last_channel_len": self.cache_last_channel_len,
            "keep_all_outputs": False,
            "drop_extra_pre_encoded": None,
            "return_transcription": True,
            "return_log_probs": False
        }
        if self.previous_hypotheses:
            kwargs["previous_hypotheses"] = self.previous_hypotheses
            kwargs["previous_pred_out"] = self.pred_out_stream
        # streaming inference step
        (
            self.pred_out_stream,
            transcribed_texts,
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
            self.previous_hypotheses
        ) = self.asr_model.conformer_stream_step(**kwargs)
        self.step_num += 1
        # extract transcription
        transcription = self._extract_most_likely_transcription(transcribed_texts)
        return transcription


    def _warm_up_model(self):
        """
        Warm up the ASR model by processing a chunk of silence.
        """
        num_frames = int(ASRStreamer.SAMPLE_RATE*(self.lookahead_size + ASRStreamer.ENCODER_STEP_LENGTH) / 1000) - 1
        silence = np.zeros(num_frames, dtype=np.int16)
        self.process_chunk(silence)
