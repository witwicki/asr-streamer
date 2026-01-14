""" Streaming ASR Choreographer """

from __future__ import annotations

import os
import importlib.resources
import signal
import time
from deprecated import deprecated
import wave
from collections.abc import Mapping
from threading import Event, Thread
from types import FrameType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from streaming.asr import ASRStreamer


Int16Array = NDArray[np.int16]


@runtime_checkable
class PyAudioStream(Protocol):
    """Subset of a PyAudio stream used by the choreographer."""

    def write(self, data: bytes) -> None: ...

    def close(self) -> None: ...

    def start_stream(self) -> None: ...

    def stop_stream(self) -> None: ...


@runtime_checkable
class PyAudioObject(Protocol):
    """Protocol describing the PyAudio instance that opens streams."""

    def open(self, *args: object, **kwargs: object) -> PyAudioStream: ...

    def get_format_from_width(self, width: int) -> int: ...

    def get_default_input_device_info(self) -> Mapping[str, object]: ...

    def terminate(self) -> None: ...


@runtime_checkable
class PyAudioModule(Protocol):
    """Protocol for the imported PyAudio module reference."""

    PyAudio: type[PyAudioObject]
    paContinue: int
    paInt16: int


class TranscriptionServerProtocol(Protocol):
    """Interface required from the transcription server."""

    def send_transcription(self, transcription: str) -> None: ...

    def close_connections(self) -> None: ...


class ASRStreamerProtocol(Protocol):
    """Interface required from the ASR streamer implementation."""

    buffer: str

    def process_chunk(self, signal: Int16Array, limited_cache: bool = False) -> str: ...

    def reset_streaming_state(self) -> None: ...

class ASRChoreographer:
    """ Chroreograph the invocation of ASR and communication of speech recognition results

    This class is the glue that holds all of the pieces together and is ultimately responsible
    for managing user interaction.

    """

    MINIMUM_ACTIVE_DURATION: float = 1.0 # seconds

    def __init__(
        self, asr_streamer: ASRStreamerProtocol,
        transcription_server: TranscriptionServerProtocol,
        streaming_result_delay_silence_threshold: float = 0.5,
        silence_threshold: float = 1.0,
        exercise_output_channel: bool = True,
        verbose: bool = True
    ) -> None:
        self.asr_streamer: ASRStreamerProtocol = asr_streamer
        self.transcription_server: TranscriptionServerProtocol = transcription_server
        self.streaming_result_delay_silence_threshold: float = streaming_result_delay_silence_threshold
        self.silence_threshold: float = silence_threshold
        self.exercise_output_channel: bool = exercise_output_channel
        self.waiting_for_silence_to_deactivate: bool = False
        current_time = time.time()
        self.last_activity_time: float = current_time
        self.time_of_last_asr_result: float = current_time
        self.time_consumed_by_asr_processing: float|None = None
        self.time_since_last_asr_result: float = 0.0
        self.is_active: bool = False # active vs. passive (where ASR is still processing, just not sending the results)
        self.verbose: bool = verbose
        self.init_time: float = time.time()
        self.active_start_time: float = 0.0
        self.active_finish_time: float = 0.0

        # project path is one level up from this file
        self.project_path:str = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # handling of Ctrl-C
        self._exit_event: Event = Event()
        _ = signal.signal(signal.SIGINT, self._force_exit_handler)

        # state variables
        self.was_killed: bool = False
        self.frames_processed: int = 0

        # pyaudio variables (including background noise to keep pulseaudio sink awake)
        self.pyaudio: PyAudioModule | None = None
        self._pyaudio: PyAudioObject | None = None
        self._background_noise_thread: Thread|None = None
        self._background_stream: PyAudioStream | None = None
        self._activate_sound_stream: PyAudioStream | None = None
        self._deactivate_sound_stream: PyAudioStream | None = None
        self._silence_audio_data: Int16Array | None = None
        self._activate_audio_data: Int16Array | None = None
        self._deactivate_audio_data: Int16Array | None = None


    def setup_audio_output(
        self,
        pyaudio_package: PyAudioModule,
        pyaudio_object: PyAudioObject,
    ) -> None:
        self.pyaudio = pyaudio_package
        self._pyaudio = pyaudio_object
        if self.exercise_output_channel:
            # keep pyaudio output awake using silent stream
            self._background_noise_thread = Thread(target=self._background_thread_worker)
            self._background_noise_thread.start()
        # initialize sound effects
        self._init_toggling_waveforms()

    def toggle_asr(self) -> None:
        # TODO shouldn't the elif condition go here instead?
        if not self.is_active:
            if self.verbose:
                print(f"ON {time.time()-self.init_time}")
            self.set_asr_state(active=True)
            self.active_start_time = time.time()
        elif not self.waiting_for_silence_to_deactivate:
            if self.verbose:
                print(f"OFF {time.time()-self.init_time}")
            self.active_finish_time = time.time()
            self.play_deactivate_sound() # TODO: this looks like it might not make a sound
            self.waiting_for_silence_to_deactivate = True
            if self.verbose:
                print(f"*** ASR Deactivation process started. Waiting for {self.streaming_result_delay_silence_threshold} seconds of silence before cutting transcription. ***")

    def set_asr_state(self, active: bool = True) -> None:
        """Set the state of the ASR system to either active or passive."""
        if active and not self.is_active:
            print("ACTIVATING ASR")
            self.is_active = True
            self.last_activity_time = time.time()
            self.play_activate_sound()
        elif not active:
            if self.is_active:
                print("DEACTIVATING ASR")
            self.is_active = False
        else:
            print("Warning: set_asr_state(active=True) called when ASR is already active. Ignoring.")
        # a reset is really only necessary when the buffer contains recognized speech
        if self.asr_streamer.buffer:
            self.asr_streamer.reset_streaming_state()

    @deprecated
    def send_final_asr_result(self):
        print(f"SENDING LATEST RESULT: [{self.asr_streamer.buffer}]")
        self.transcription_server.send_transcription_state(
            transcription = self.asr_streamer.buffer,
            user_activated = True,
            final = True
        )

    def recognize_speech(
        self,
        in_data: bytes,
        frame_count: int,
        _time_info: Mapping[str, float] | None,
        _status: int,
    ) -> tuple[bytes, int]:
        start_time = time.time() # profile compute/runtime delta between callbacks
        t_chunk_start = self.frames_processed / ASRStreamer.SAMPLE_RATE
        self.frames_processed += frame_count
        t_chunk_end = self.frames_processed / ASRStreamer.SAMPLE_RATE
        signal = np.frombuffer(in_data, dtype=np.int16)
        text = self.asr_streamer.process_chunk(signal)
        self.time_consumed_by_asr_processing = (time.time() - start_time)
        self.time_since_last_asr_result = (time.time() - self.time_of_last_asr_result)
        self.time_of_last_asr_result = time.time()
        if self.verbose:
            log_message = "/ wall={wall}, mic=[{chunk_start}, {chunk_end}], offset={offset}, compute_time={compute}/".format(
                wall=start_time - self.init_time,
                chunk_start=t_chunk_start,
                chunk_end=t_chunk_end,
                offset=t_chunk_end - (start_time - self.init_time),
                compute=self.time_since_last_asr_result,
            )
            print(log_message)
            print(f"ASR_result: [{text}]") #, end='\r')
            print(f"  previous was {self.asr_streamer.buffer}")
            print("time taken for recognize_speech: {:.2f}s, ".format(self.time_consumed_by_asr_processing), end="")
            print("time since last activity: {:.2f}s, ".format(self.time_since_last_speech_activity_detected()), end="")
            print("time since last asr_result: {:.2f}s, ".format(self.time_since_last_asr_result), end="")
            print("effective detected silence: {:.2f}s".format(self.time_since_last_speech_activity_detected() - self.time_since_last_asr_result))
        # was no new speech detected?
        complete_buffer = False
        asr_result_changed = False
        was_user_activated = self.is_active

        if text == self.asr_streamer.buffer:
            # has enough silence passed that we should consider resetting asr?
            if self.is_active:
                if self.waiting_for_silence_to_deactivate:
                    # in this case we are just looking for any late asr results followed by silence...
                    if self.deactivate_if_silence_threshold_exceeded(self.streaming_result_delay_silence_threshold):
                        if (self.active_finish_time - self.active_start_time) > self.MINIMUM_ACTIVE_DURATION:
                            complete_buffer = True
            else:
                # ...whereas here we are looking for natural gaps in between utterances
                #      that serve to seed our next active asr session with an appropriate beginning
                if self.deactivate_if_silence_threshold_exceeded(self.silence_threshold):
                    # because asr is in passive mode, simply throw away the latest result
                    self.asr_streamer.buffer = ""
                    complete_buffer = True
        else:
            self.asr_streamer.buffer = text
            self.update_last_activity()
            asr_result_changed = True

        if text and not self.verbose: # otherwise we would have already printed it above
            print(f"ASR_result: [{text}]", end=('\n\n' if complete_buffer else '\r'))

        if text and (asr_result_changed or complete_buffer):
            self.transcription_server.send_transcription_state(
                transcription = text,
                user_activated = was_user_activated,
                final = complete_buffer
            )


        if self.pyaudio is None:
            raise RuntimeError("PyAudio module is not initialized")
        return (in_data, self.pyaudio.paContinue)


    def time_since_last_speech_activity_detected(self) -> float:
        current_time = time.time()
        return (current_time - self.last_activity_time)

    def deactivate_if_silence_threshold_exceeded(self, threshold: float) -> bool:
        """Check if no speech has been detected for the threshold duration."""
        if self.verbose:
            print(f"ASRChoreographer.deactivate_if_silence_threshold_exceeded({threshold})")
        if (self.time_since_last_speech_activity_detected() - self.time_since_last_asr_result) >= threshold:
            if self.verbose:
                print("\nAutomatically deactivating ASR due to silence.")
            self.set_asr_state(active=False)
            self.waiting_for_silence_to_deactivate = False # state reset
            return True
        return False

    def update_last_activity(self) -> None:
        """Update the last activity timestamp when speech is detected."""
        self.last_activity_time = time.time()

    def _force_exit_handler(self, _sig: int, _frame: FrameType | None) -> None:
        print("\nCtrl-C pressed!")
        self._exit_event.set()
        self.transcription_server.close_connections()
        self.close_sound_streams()
        if self.exercise_output_channel:
            if self._background_noise_thread is not None:
                self._background_noise_thread.join()
        self.was_killed = True

    def _background_thread_worker(self) -> None:
        # prepare to loop silence
        self._init_silence_waveform()
        # play until exit event occurs
        print("Silence-playing background thread started.")
        while not self._exit_event.is_set():
            if self._background_stream is None or self._silence_audio_data is None:
                raise RuntimeError("Background audio stream was not initialized")
            self._background_stream.write(self._silence_audio_data.tobytes())
            time.sleep(0.01)
        if self._background_stream is not None:
            self._background_stream.close()
        print("Finished silence-playing background thread.")

    def _init_silence_waveform(self) -> None:
        # load dummy wave file (need not be silence)
        with importlib.resources.files('choreography.ui_sound_effects').joinpath('beep-open.wav').open('rb') as wav_file:
            with wave.open(wav_file, 'rb') as wf:
                data = wf.readframes(1000000) # the whole wave file
                # reduce volume to near zero in waveform
                data_readonly = np.frombuffer(data, dtype=np.int16).reshape(-1, wf.getnchannels())
                self._silence_audio_data = np.copy(data_readonly)
                _ = np.multiply(
                    self._silence_audio_data,
                    0.00001,
                    out=self._silence_audio_data,
                    casting="unsafe",
                )
                # start a stream and close file
                if self._pyaudio is None:
                    raise RuntimeError("PyAudio instance is not initialized")
                self._background_stream = self._pyaudio.open(
                    format=self._pyaudio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                )

    def _init_toggling_waveforms(self) -> None:
        # load sounds from wave files
        with (
            importlib.resources.files('choreography.ui_sound_effects').joinpath('beep-open.wav').open('rb') as wav_activate,
            importlib.resources.files('choreography.ui_sound_effects').joinpath('beep-close.wav').open('rb') as wav_deactivate
        ):
            wf_activate = wave.open(wav_activate, 'rb')
            wf_deactivate = wave.open(wav_deactivate, 'rb')
            data = wf_activate.readframes(1000000) # the whole wave file
            self._activate_audio_data = np.frombuffer(data, dtype=np.int16).reshape(-1, wf_activate.getnchannels())
            data = wf_deactivate.readframes(1000000)
            self._deactivate_audio_data = np.frombuffer(data, dtype=np.int16).reshape(-1, wf_deactivate
            .getnchannels())
            # start streams and close files
            if self._pyaudio is None:
                raise RuntimeError("PyAudio instance is not initialized")
            self._activate_sound_stream = self._pyaudio.open(
                format=self._pyaudio.get_format_from_width(wf_activate.getsampwidth()),
                channels=wf_activate.getnchannels(),
                rate=wf_activate.getframerate(),
                output=True,
            )
            wf_activate.close()
            self._deactivate_sound_stream = self._pyaudio.open(
                format=self._pyaudio.get_format_from_width(wf_deactivate.getsampwidth()),
                channels=wf_deactivate.getnchannels(),
                rate=wf_deactivate.getframerate(),
                output=True,
            )
            wf_deactivate.close()

    def play_activate_sound(self) -> None:
        if self._activate_sound_stream is None or self._activate_audio_data is None:
            raise RuntimeError("Activation sound not initialized")
        self._activate_sound_stream.write(self._activate_audio_data.tobytes())

    def play_deactivate_sound(self) -> None:
        if self._deactivate_sound_stream is None or self._deactivate_audio_data is None:
            raise RuntimeError("Deactivation sound not initialized")
        self._deactivate_sound_stream.write(self._deactivate_audio_data.tobytes())

    def close_sound_streams(self) -> None:
        if self._activate_sound_stream is not None:
            self._activate_sound_stream.close()
            self._activate_sound_stream = None
        if self._deactivate_sound_stream is not None:
            self._deactivate_sound_stream.close()
            self._deactivate_sound_stream = None
