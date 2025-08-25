""" Streaming ASR Choreographer """

import os
import importlib.resources
import signal
import time
from deprecated import deprecated
import numpy as np
from threading import Event, Thread
import wave
from streaming.asr import ASRStreamer

class ASRChoreographer:
    """ Chroreograph the invocation of ASR and communication of speech recognition results

    This class is the glue that holds all of the pieces together and is ultimately responsible
    for managing user interaction.

    """
    def __init__(self, asr_streamer, transcription_server, streaming_result_delay_silence_threshold=0.5, silence_threshold=1.0, exercise_output_channel=True, verbose=True):
        self.asr_streamer = asr_streamer
        self.transcription_server = transcription_server
        self.streaming_result_delay_silence_threshold = streaming_result_delay_silence_threshold
        self.silence_threshold = silence_threshold
        self.exercise_output_channel = exercise_output_channel
        self.waiting_for_silence_to_deactivate = False
        current_time = time.time()
        self.last_activity_time = current_time
        self.time_of_last_asr_result = current_time
        self.time_consumed_by_asr_processing = None
        self.time_since_last_asr_result = 0.0
        self.is_active = False # active vs. passive (where ASR is still processing, just not sending the results)
        self.verbose = verbose
        self.init_time = time.time()
        self.active_start_time = 0.0
        self.active_finish_time = 0.0
        self.MINIMUM_ACTIVE_DURATION = 1.0 # seconds

        # project path is one level up from this file
        self.project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # handling of Ctrl-C
        self._exit_event = Event()
        signal.signal(signal.SIGINT, self._force_exit_handler)

        # state variables
        self.was_killed = False
        self.frames_processed = 0

    def setup_audio_output(self, pyaudio_package, pyaudio_object):
        self.pyaudio = pyaudio_package
        self._pyaudio = pyaudio_object
        if self.exercise_output_channel:
            # keep pyaudio output awake using silent stream
            self._background_noise_thread = Thread(target=self._background_thread_worker)
            self._background_noise_thread.start()
        # initialize sound effects
        self._init_toggling_waveforms()

    def toggle_asr(self):
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

    def set_asr_state(self, active=True):
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

    def recognize_speech(self, in_data, frame_count, time_info, status):
            start_time = time.time() # profile the runtime of this method, and track time in between results
            t_chunk_start = self.frames_processed / ASRStreamer.SAMPLE_RATE
            self.frames_processed += frame_count
            t_chunk_end = self.frames_processed / ASRStreamer.SAMPLE_RATE
            signal = np.frombuffer(in_data, dtype=np.int16)
            text = self.asr_streamer.process_chunk(signal)
            self.time_consumed_by_asr_processing = (time.time() - start_time)
            self.time_since_last_asr_result = (time.time() - self.time_of_last_asr_result)
            self.time_of_last_asr_result = time.time()
            if self.verbose:
                print(
                    f"/ wall={start_time-self.init_time}, mic=[{t_chunk_start}, {t_chunk_end}]"
                    f", offset={t_chunk_end-(start_time-self.init_time)}, compute_time={self.time_since_last_asr_result}/"
                )

            if self.verbose:
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


            return (in_data, self.pyaudio.paContinue)


    def time_since_last_speech_activity_detected(self):
        current_time = time.time()
        return (current_time - self.last_activity_time)

    def deactivate_if_silence_threshold_exceeded(self, threshold):
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

    def update_last_activity(self):
        """Update the last activity timestamp when speech is detected."""
        self.last_activity_time = time.time()

    def _force_exit_handler(self, sig, frame):
        print("\nCtrl-C pressed!")
        self._exit_event.set()
        self.transcription_server.close_connections()
        self.close_sound_streams()
        if self.exercise_output_channel:
            self._background_noise_thread.join()
        self.was_killed = True

    def _background_thread_worker(self):
        # prepare to loop silence
        self._init_silence_waveform()
        # play until exit event occurs
        print("Silence-playing background thread started.")
        while not self._exit_event.is_set():
            self._background_stream.write(self._silence_audio_data.tobytes())
            time.sleep(0.01)
        self._background_stream.close()
        print("Finished silence-playing background thread.")

    def _init_silence_waveform(self):
        # load dummy wave file (need not be silence)
        with importlib.resources.files('choreography.ui_sound_effects').joinpath('beep-open.wav').open('rb') as wav_file:
            with wave.open(wav_file, 'rb') as wf:
                data = wf.readframes(1000000) # the whole wave file
                # reduce volume to near zero in waveform
                data_readonly = np.frombuffer(data, dtype=np.int16).reshape(-1, wf.getnchannels())
                self._silence_audio_data = np.copy(data_readonly)
                np.multiply(self._silence_audio_data, 0.00001, out=self._silence_audio_data, casting="unsafe")
                # start a stream and close file
                self._background_stream = self._pyaudio.open(format = self._pyaudio.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)

    def _init_toggling_waveforms(self):
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
            self._activate_sound_stream = self._pyaudio.open(format = self._pyaudio.get_format_from_width(wf_activate.getsampwidth()),
                channels = wf_activate.getnchannels(),
                rate = wf_activate.getframerate(),
                output = True)
            wf_activate.close()
            self._deactivate_sound_stream = self._pyaudio.open(format = self._pyaudio.get_format_from_width(wf_deactivate.getsampwidth()),
                channels = wf_deactivate.getnchannels(),
                rate = wf_deactivate.getframerate(),
                output = True)
            wf_deactivate.close()

    def play_activate_sound(self):
        self._activate_sound_stream.write(self._activate_audio_data.tobytes())

    def play_deactivate_sound(self):
        self._deactivate_sound_stream.write(self._deactivate_audio_data.tobytes())

    def close_sound_streams(self):
        self._activate_sound_stream.close()
        self._deactivate_sound_stream.close()
