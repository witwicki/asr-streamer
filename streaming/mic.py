from __future__ import annotations

import streaming
from choreography.sasrc import ASRChoreographer, PyAudioModule, PyAudioObject, PyAudioStream
from streaming.asr import ASRStreamer


class AudioStreamManager:
    """Manage audio stream from microphone."""

    def __init__(self, asr_choreographer: ASRChoreographer, chunk_size: float, verbose: bool = True):
        print("Initializing audio stream manager with the microphone as input...")
        self.pyaudio: PyAudioModule
        self.p: PyAudioObject
        self.input_device_index: int
        self.stream: PyAudioStream | None = None
        self.verbose: bool = verbose
        self.asr_choreographer: ASRChoreographer = asr_choreographer
        self.chunk_size: float = chunk_size  # ASR lookahead size + ENCODER_STEP_LENGTH
        self._initialize_pyaudio()
        print(f"...using device: {self.p.get_default_input_device_info()['name']} ...")
        self._pass_pyaudio_object_back_to_choreographer()
        print("...done initializing audio stream manager.")


    @streaming.suppress_terminal_output
    def _initialize_pyaudio(self) -> None:
        import pyaudio

        if not isinstance(pyaudio, PyAudioModule):
            raise TypeError("PyAudio module is missing the required interface")
        self.pyaudio = pyaudio
        pyaudio_instance = pyaudio.PyAudio()
        self.p = pyaudio_instance
        device_info = self.p.get_default_input_device_info()
        index_value = device_info['index']
        if not isinstance(index_value, int):
            raise TypeError("PyAudio default input device index was not an integer")
        self.input_device_index = index_value

    def _pass_pyaudio_object_back_to_choreographer(self) -> None:
        self.asr_choreographer.setup_audio_output(self.pyaudio, self.p)

    def start_stream(self) -> None:
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

    def stop_stream(self) -> None:
        print("stopping...")
        if self.stream is None:
            return
        stream = self.stream
        stream.stop_stream()
        stream.close()
        self.stream = None
        print("termination pyaudio...")
        self.p.terminate()
        print("terminated!!!")
