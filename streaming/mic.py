import streaming
from streaming.asr import ASRStreamer

class AudioStreamManager:
    """ Manage audio stream from microphone

    Assumed to be invoked from an ASRChoreograoher, this class makes iterative
    calls back to the choreographer's speech recognition method for each new
    sample of audio received.
    """

    def __init__(self, asr_choreographer, chunk_size, verbose=True):
        print("Initializing audio stream managaer with the microphone as input...")
        self.verbose = verbose
        self._initialize_pyaudio()
        print(f"...using device: {self.p.get_default_input_device_info()['name']} ...")
        self.asr_choreographer = asr_choreographer
        self._pass_pyaudio_object_back_to_choreographer()
        self.chunk_size = chunk_size # ASR lookahead size + ENCODER_STEP_LENGTH
        self.stream = None
        print("...done initializing audio stream manager.")


    @streaming.suppress_terminal_output
    def _initialize_pyaudio(self):
        import pyaudio
        self.pyaudio = pyaudio
        self.p = pyaudio.PyAudio()
        self.input_device_index : int = int(self.p.get_default_input_device_info()['index'])

    def _pass_pyaudio_object_back_to_choreographer(self):
        self.asr_choreographer.setup_audio_output(self.pyaudio, self.p)

    def start_stream(self):
        self.stream = self.p.open(
            format=self.pyaudio.paInt16,
            channels=1,
            rate=ASRStreamer.SAMPLE_RATE,
            input=True,
            input_device_index=self.input_device_index,
            stream_callback=self.asr_choreographer.recognize_speech,
            frames_per_buffer=int(ASRStreamer.SAMPLE_RATE * self.chunk_size / 1000) - 1
        )
        self.stream.start_stream()

    def stop_stream(self):
        print("stopping...")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("termination pyaudio...")
        self.p.terminate()
        print("terminated!!!")
