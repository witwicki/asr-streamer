""" Streaming ASR Choreographer """


import signal
import time
import sys
import numpy as np
from threading import Event, Thread
import wave
from pathlib import Path

from communication.server import TranscriptionServer
from communication.rc import RemoteControl
from streaming.asr import ASRStreamer

class ASRChoreographer:
    """ Chroreograph the invocation of ASR and communication of speech recognition results

    This class is the glue that holds all of the pieces together and is ultimately responsible
    for managing user interaction.

    """
    def __init__(self, asr_streamer, transcription_server, silence_threshold=5.0, project_path=".", verbose=True):
        self.asr_streamer = asr_streamer
        self.transcription_server = transcription_server
        self.silence_threshold = silence_threshold  # seconds
        self.waiting_for_silence_to_deactivate = False
        current_time = time.time()
        self.last_activity_time = current_time
        self.time_of_last_asr_result = current_time
        self.time_consumed_by_asr_processing = None
        self.time_since_last_asr_result = None
        self.is_active = False
        self.project_path = project_path
        self.verbose = verbose

        # handling of Ctrl-C
        self._exit_event = Event()
        signal.signal(signal.SIGINT, self._force_exit_handler)

        # basic state variable
        self.was_killed = False

    def setup_audio_output(self, pyaudio_package, pyaudio_object):
        self.pyaudio = pyaudio_package
        self._pyaudio = pyaudio_object
        # keep pyaudio output awake using silent stream
        self._background_noise_thread = Thread(target=self._background_thread_worker)
        self._background_noise_thread.start()
        # initialize sound effects
        self._init_toggling_waveforms()


        
    def toggle_asr(self):
        if not self.is_active:
            self.activate_asr()
        else:
            self.play_deactivate_sound()
            self.waiting_for_silence_to_deactivate = True
            if self.verbose:
                print(f"*** ASR Deactivation process started. Waiting for {self.silence_threshold} seconds of silence before cutting transcription. ***")
            #self.deactivate_asr()

    def send_latest_asr_result(self):
        self.transcription_server.send_transcription(f"[{self.asr_streamer.buffer}]")


    def activate_asr(self):
        """Activate the ASR streamer and reset last activity time."""
        if not self.is_active:
            print("\nACTIVATING ASR...")
            self.asr_streamer.reset_streaming_state()
            self.is_active = True
            self.last_activity_time = time.time()
            self.play_activate_sound()
            # TODO Start silence monitoring thread if needed
    
    def deactivate_asr(self):
        """Deactivate the ASR streamer and send any buffered transcription."""
        if self.is_active:
            print(f"\n...SENDING LATEST RESULT: [{self.asr_streamer.buffer}]")
            self.send_latest_asr_result()
            self.asr_streamer.reset_streaming_state()
            self.is_active = False
            # TODO Stop silence monitoring thread if running
    
    def recognize_speech(self, in_data, frame_count, time_info, status):
            # TODO accumulate a THRESHOLD-seconds buffer of speech recognition results
            if self.is_active:
                start_time = time.time() # profile the runtime of this method, and track time in between results
                signal = np.frombuffer(in_data, dtype=np.int16)
                text = self.asr_streamer.process_chunk(signal)
                self.time_consumed_by_asr_processing = (time.time() - start_time)
                self.time_since_last_asr_result = (time.time() - self.time_of_last_asr_result)
                self.time_of_last_asr_result = time.time()
                if self.verbose:
                    print(f"ASR_result: [{text}]") #, end='\r')
                    print("time taken for recognize_speech: {:.2f}s, ".format(self.time_consumed_by_asr_processing), end="")
                    print("time since last activity: {:.2f}s, ".format(self.time_since_last_speech_activity_detected()), end="")
                    print("time since last asr_result: {:.2f}s, ".format(self.time_since_last_asr_result), end="")
                    print("effective detected silence: {:.2f}s".format(self.time_since_last_speech_activity_detected() - self.time_since_last_asr_result))

                # was no new speech detected?
                if text == self.asr_streamer.buffer:
                    # has enough silence passed that we should consider terminating asr?
                    self.check_silence()
                else:
                    if not self.verbose: # otherwise we would have already printed it above
                        print(f"ASR_result: [{text}]") #, end='\r')
                    self.asr_streamer.buffer = text
                    self.update_last_activity()                        

            return (in_data, self.pyaudio.paContinue)


    def time_since_last_speech_activity_detected(self):
        current_time = time.time()
        return (current_time - self.last_activity_time)

    def check_silence(self):
        """Check if no speech has been detected for the threshold duration."""
        if self.is_active:
            if (self.time_since_last_speech_activity_detected() - self.time_since_last_asr_result) > self.silence_threshold:
                if self.waiting_for_silence_to_deactivate:
                    if self.verbose:
                        print(f"\nAutomatically deactivating ASR due to silence.")
                    self.deactivate_asr()
                    self.waiting_for_silence_to_deactivate = False # state reset
                return True
        return False

    def update_last_activity(self):
        """Update the last activity timestamp when speech is detected."""
        if self.is_active:
            self.last_activity_time = time.time()

    def _force_exit_handler(self, sig, frame):
        print("\nCtrl-C pressed!")
        self._exit_event.set()
        self._background_noise_thread.join()
        self.was_killed = True
        #sys.exit("\n...program terminated by user.")


    def _background_thread_worker(self):
        # prepare to loop silence
        self._init_silence_waveform()
        # play until exit event occurs
        print("Silence-playing background thread started.")
        # TODO instead listen to self.was_killed and forgo the sys.exit()
        while not self._exit_event.is_set():
            self._background_stream.write(self._silence_audio_data.tobytes())
        self._background_stream.close()
        print("Finished silence-playing background thread.")
        sys.exit() # this line intentionally throws an exception to bring the program down

    def _init_silence_waveform(self):
        # load dummy wave file (need not be silence)
        with wave.open(str(self.project_path.joinpath("assets/ui_sound_effects/beep-open.wav")), 'rb') as wf:
            data = wf.readframes(1000000) # the whole wave file
            # reduce volume to near zero in waveform
            data_readonly = np.frombuffer(data, dtype=np.int16).reshape(-1, wf.getnchannels())
            self._silence_audio_data = np.copy(data_readonly)
            np.multiply(self._silence_audio_data, 0.0002, out=self._silence_audio_data, casting="unsafe")
            # start a stream and close file
            self._background_stream = self._pyaudio.open(format = self._pyaudio.get_format_from_width(wf.getsampwidth()),  
                channels = wf.getnchannels(),  
                rate = wf.getframerate(),  
                output = True)

    def _init_toggling_waveforms(self):
        # load sounds from wave files
        wf_activate = wave.open(str(self.project_path.joinpath("assets/ui_sound_effects/beep-open.wav")), 'rb')
        wf_deactivate = wave.open(str(self.project_path.joinpath("assets/ui_sound_effects/beep-close.wav")), 'rb')
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
