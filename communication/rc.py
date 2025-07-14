import socket
import time
from pynput import keyboard
from choreography.sasrc import ASRChoreographer

class RemoteControl:
    """ Base class for controlling an ASR session remotely """

    def __init__(self, asr_choreographer: ASRChoreographer, mode="toggle"):
        self.asr_choreographer = asr_choreographer
        self.mode = mode
        # assume that the current state of control is the initial state of the choreographer
        self.current_state = asr_choreographer.is_active
        self.was_killed = False

    def run(self):
        pass

    def switch_mode(self):
        if self.mode == "toggle":
            self.mode = "press_and_hold_to_talk"
        else:
            self.mode = "toggle"

    def end(self):
        self.was_killed = True

    def _handle_control_event(self, is_pressed):
        new_control_state = is_pressed

        if self.mode == "toggle":
            # Detect transition from False to True
            if self.current_state == False and new_control_state == True:
                self.asr_choreographer.toggle_asr()
        elif self.mode == "press_and_hold_to_talk":
            if self.current_state == False and new_control_state == True:
                if not self.asr_choreographer.is_active:
                    self.asr_choreographer.toggle_asr()
            elif self.current_state == True and new_control_state == False:
                if self.asr_choreographer.is_active:
                    self.asr_choreographer.toggle_asr()

        # Update previous boolean state for next iteration
        self.current_state = new_control_state


class RemoteControlByKeyboard(RemoteControl):
    """ Control an asr session using keyboard events
    In the simplest form, key presses and releases start and stop the transcription-gathering choreographer.
    """

    def __init__(self, asr_choreographer: ASRChoreographer, mode="toggle"):
        super().__init__(asr_choreographer, mode)

    def run(self):
        with keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release) as listener:
            while not self.was_killed:
                time.sleep(0.01)
            listener.stop()

    def _on_key_press(self, key):
        try:
            if key == keyboard.Key.space:  # Use alt as the control key
                self._handle_control_event(True)
        except AttributeError:
            pass

    def _on_key_release(self, key):
        try:
            if key == keyboard.Key.space:  # Use alt as the control key
                self._handle_control_event(False)
        except AttributeError:
            pass

class RemoteControlByUDP(RemoteControl):
    """ Control an asr session remotely

    In the simplest form, button presses and releases start and stop the transciption-gathering
    choroegrapher.
    """

    def __init__(self, asr_choreographer: ASRChoreographer, udp_host: str = "0.0.0.0", udp_port: int = 5656, mode="toggle"):
        super().__init__(asr_choreographer, mode)
        self.udp_host = udp_host
        self.udp_port = udp_port

    def run(self):
        self._listen_to_boolean_signal()

    def _listen_to_boolean_signal(self):
        """
        Listens for boolean updates over UDP and detects when bool_state toggles from False to True.

        Args:
            target_ip (str): IP address to bind to and listen on.
            target_port (int): Port number to listen on.
        """
        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            # Bind the socket to the target IP and port
            sock.bind((self.udp_host, self.udp_port))
            sock.setblocking(False)

            print(f"Starting remote control interface, listening on UDP:{self.udp_host}:{self.udp_port}")

            while not self.was_killed:
                try:
                    # Receive data (max 1 byte in this case)
                    data, addr = sock.recvfrom(1)

                    if len(data) == 0:
                        continue

                    # Convert bytes back to boolean
                    new_control_state = bool(int.from_bytes(data, byteorder='big'))
                    #print(f"Received: {bool_state} from {addr}")

                    self._handle_control_event(new_control_state)
                except BlockingIOError:
                    pass
                except ValueError as e:
                    print(f"Error converting data: {e}")
                finally:
                    time.sleep(0.01)

        except socket.error as e:
            print(f"RemoteControl: Socket error occurred: {e}")
        finally:
            # Close the socket
            sock.close()
