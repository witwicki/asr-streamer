from __future__ import annotations

import socket
import time
from types import TracebackType
from typing import Callable, Protocol, runtime_checkable

from typing_extensions import override

from choreography.sasrc import ASRChoreographer


class KeyboardKeySet(Protocol):
    """Subset of pynput.keyboard.Key used by the remote control."""

    space: object


class KeyboardListenerProtocol(Protocol):
    """Context-managed keyboard listener."""

    def __enter__(self) -> KeyboardListenerProtocol: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...

    def stop(self) -> None: ...


@runtime_checkable
class KeyboardModuleProtocol(Protocol):
    """Interface exposed by pynput.keyboard that we rely on."""

    Key: KeyboardKeySet

    def Listener(
        self,
        *,
        on_press: Callable[[object], None] | None = None,
        on_release: Callable[[object], None] | None = None,
    ) -> KeyboardListenerProtocol: ...

class RemoteControl:
    """ Base class for controlling an ASR session remotely """

    def __init__(
        self,
        asr_choreographer: ASRChoreographer,
        mode: str = "toggle",
        udp_port: int = 5656,
    ) -> None:
        self.asr_choreographer: ASRChoreographer = asr_choreographer
        self.mode: str = mode
        # udp port for sending and receiving
        self.udp_port: int = udp_port
        # assume that the current state of control is the initial state of the choreographer
        self.current_state: bool = asr_choreographer.is_active
        self.was_killed: bool = False

    def run(self) -> None:
        raise NotImplementedError

    def switch_mode(self) -> None:
        if self.mode == "toggle":
            self.mode = "press_and_hold_to_talk"
        else:
            self.mode = "toggle"

    def end(self) -> None:
        self.was_killed = True

    def _handle_control_event(self, is_pressed: bool) -> None:
        new_control_state: bool = is_pressed

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

    def __init__(
        self,
        asr_choreographer: ASRChoreographer,
        mode: str = "toggle",
        udp_port: int = 5656,
    ) -> None:
        super().__init__(asr_choreographer, mode, udp_port+1) 
        # hack: a bit of a magic number in order to conform to an external listener defaulting to 5657 = 5656+1
        from pynput import keyboard as pynput_keyboard

        if not isinstance(pynput_keyboard, KeyboardModuleProtocol):
            raise TypeError("pynput.keyboard module is missing listener interface")
        self.keyboard: KeyboardModuleProtocol = pynput_keyboard
        # udp socket for sending events
        self.udp_destination: str = "localhost"
        self.sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _send_signal_over_udp(self, on: bool) -> None:
        value = on.to_bytes(1, byteorder='big')
        _ = self.sock.sendto(value, (self.udp_destination, self.udp_port))

    @override
    def run(self) -> None:
        with self.keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release) as listener:
            while not self.was_killed:
                time.sleep(0.01)
            listener.stop()

    def _on_key_press(self, key: object) -> None:
        try:
            if key == self.keyboard.Key.space:  # Use alt as the control key
                self._handle_control_event(True)
                self._send_signal_over_udp(on=True)
        except AttributeError:
            pass

    def _on_key_release(self, key: object) -> None:
        try:
            if key == self.keyboard.Key.space:  # Use alt as the control key
                self._handle_control_event(False)
                self._send_signal_over_udp(on=False)

        except AttributeError:
            pass

class RemoteControlByUDP(RemoteControl):
    """ Control an asr session remotely

    In the simplest form, button presses and releases start and stop the transciption-gathering
    choroegrapher.
    """

    def __init__(
        self,
        asr_choreographer: ASRChoreographer,
        udp_host: str = "0.0.0.0",
        udp_port: int = 5656,
        mode: str = "toggle",
    ) -> None:
        super().__init__(asr_choreographer, mode, udp_port)
        self.udp_host: str = udp_host

    @override
    def run(self) -> None:
        self._listen_to_boolean_signal()

    def _listen_to_boolean_signal(self) -> None:
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
                    data = sock.recvfrom(1)[0]

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
