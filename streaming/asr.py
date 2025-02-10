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

    #with contextlib.redirect_stdout(io.StringIO()):
    #    with contextlib.redirect_stderr(io.StringIO()):
    #        from omegaconf import OmegaConf, open_dict
    #        import nemo.collections.asr as nemo_asr
    #        from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
    #        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    #        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

    def __init__(self, model_name, lookahead_size, decoder_type="rnnt", decoding_strategy="greedy", verbose=True):
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
        print("...done initializing ASR model.")

    @streaming.show_terminal_output
    def _import_nemo_packages(self):
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
        self.asr_model = self.nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        
        if self.model_name == "stt_en_fastconformer_hybrid_large_streaming_multi":
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
        
        self.asr_model.eval()

        self.device = self.asr_model.device        
        self.preprocessor = self._init_preprocessor()

    def _init_preprocessor(self):
        cfg = copy.deepcopy(self.asr_model._cfg)
        self.OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        preprocessor = self.EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        preprocessor.to(self.device)
        return preprocessor

    def _init_streaming_state(self):
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.step_num = 0
        
        (self.cache_last_channel, 
         self.cache_last_time, 
         self.cache_last_channel_len) = self.asr_model.encoder.get_initial_cache_state(batch_size=1)
        
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        num_channels = self.asr_model.cfg.preprocessor.features
        self.cache_pre_encode = torch.zeros(
            (1, num_channels, self.pre_encode_cache_size), device=self.device
        )
        self.buffer = "" # the buffer of recognized text
    
    def reset_streaming_state(self):
        self._init_streaming_state()

    def preprocess_audio(self, audio):
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(self.device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(self.device)
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length

    def _extract_transcriptions(self, hyps):
        if self.decoding_strategy == "greedy" and isinstance(hyps[0], self.Hypothesis):
            return [hyp.text for hyp in hyps]
        else:
            # TODO return both dimensions for beam search
            return [hyp.text for hyp in hyps[0]]
        #print(f"size={len(hyps)}, size of hyps[0]={len(hyps[0])}, hyps={hyps}")
        return ""

    def _extract_most_likely_transcription(self, hyps):
        # TODO do this correctly for beam search
        # additional details
        #print(f"self._extract_transcriptions(transcribed_texts) = [{self._extract_transcriptions(transcribed_texts)}]")
        #for hyp in transcribed_texts[0]:
            #print(f"hyp={hyp}")
            #print(f"text={hyp.text}")
            #print(f"   score={hyp.score}")
            #print(f"   y_sequence={hyp.y_sequence}")

        return self._extract_transcriptions(hyps)[0]

    def process_chunk(self, audio_chunk, limited_cache=False):
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
        
        with torch.no_grad():
            (self.pred_out_stream,
             transcribed_texts,
             self.cache_last_channel,
             self.cache_last_time,
             self.cache_last_channel_len,
             self.previous_hypotheses) = self.asr_model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_len,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=self.previous_hypotheses,
                previous_pred_out=self.pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )
        
        self.step_num += 1

        transcription = self._extract_most_likely_transcription(transcribed_texts)

        return transcription


    def _warm_up_model(self):
            num_frames = int(ASRStreamer.SAMPLE_RATE*(self.lookahead_size + ASRStreamer.ENCODER_STEP_LENGTH) / 1000) - 1
            silence = np.zeros(num_frames, dtype=np.int16)
            self.process_chunk(silence)