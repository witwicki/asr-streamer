from __future__ import annotations

import socket
import threading
import time
from typing import cast
import json

class TranscriptionServer:
    """TranscriptionServer
    This class is responsible for sending live transcriptions to exactly one listener.

    """
    def __init__(self) -> None:
        self.server_socket: socket.socket | None = None
        self.client_socket: socket.socket | None = None
        self.asr_active: bool = False
        self.connections_closed: bool = False
        self.buffer: str = ""
        self.client_thread: Thread | None = None

    def start_server(self, host: str = '0.0.0.0', port: int = 27400) -> None:
        """Start TCP server and wait for a single client connection."""
        print("Starting transcription server...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.setblocking(False)
        self.server_socket.listen(1)
        print(f"...advertising on TCP:{host}:{port}")

        # Wait for client connection in separate thread
        self.client_thread = threading.Thread(target=self.accept_client)
        self.client_thread.start()

    def accept_client(self) -> None:
        """Accept client connection or reconnection"""
        if self.client_socket is None:
            print("Waiting for a TCP listener to connect...")
        while not self.connections_closed:
            try:
                if self.server_socket:
                    accept_result: tuple[socket.socket, object] = self.server_socket.accept()
                    client_socket = accept_result[0]
                    addr_info_raw: object = accept_result[1]
                    addr_info = cast(tuple[str, int] | tuple[str, int, int, int], addr_info_raw)
                    self.client_socket = client_socket
                    client_addr = cast(tuple[str, int], addr_info)
                    print(f"Connected by {client_addr}.  Socket={self.client_socket}")
            except BlockingIOError:
                pass
            finally:
                time.sleep(1.0)

    def send_transcription(self, transcription: str) -> None:
        """Send buffered transcription over TCP connection."""
        if self.client_socket:
            try:
                # Send as JSON string or another format
                msg = transcription.encode('utf-8')
                self.client_socket.sendall(msg)
                print(f'Sending over TCP: "{transcription}"')
            except ConnectionResetError:
                print("Client disconnected unexpectedly.")
                self.close_connections()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Error sending transcription: {exc}")

    def send_transcription_state(self, transcription: str, user_activated: bool, final: bool):
        """Send latest transcription result, whether or not user has marked this transcription as active,
        and whether or not this is the final result before buffer is cleared.
        """
        # compose message
        message = {}
        message['transcription'] = transcription
        message['user_activated'] = user_activated
        message['final'] = final
        encoded_message = json.dumps(message).encode('utf-8')
        # send message
        if self.client_socket:
            try:
                self.client_socket.sendall(encoded_message)
            except ConnectionResetError:
                print("Client disconnected unexpectedly.")
                self.close_connections()
            except Exception as e:
                print(f"Error sending transcription: {e}")

    def close_connections(self):
        """Close TCP connections."""
        print("Closing TCP connections...")
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        self.connections_closed = True
        if self.client_thread is not None:
            self.client_thread.join()
