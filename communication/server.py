from __future__ import annotations

import socket
import threading
import time
from typing import cast
import json

class TranscriptionServer:
    """TranscriptionServer
    This class is responsible for sending live transcriptions to multiple listeners across multiple ports.

    """
    def __init__(self) -> None:
        self.server_sockets: list[socket.socket] = []
        self.client_sockets: list[socket.socket] = []
        self.asr_active: bool = False
        self.connections_closed: bool = False
        self.buffer: str = ""
        self.client_threads: list[threading.Thread] = []
        self.lock = threading.Lock()

    def start_server(self, host: str = '0.0.0.0', ports: list[int] = [27400]) -> None:
        """Start TCP servers on multiple ports and wait for client connections."""
        print("Starting transcription server...")
        for port in ports:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.setblocking(False)
            server_socket.listen(1)
            self.server_sockets.append(server_socket)
            print(f"...advertising on TCP:{host}:{port}")

            # Wait for client connection in separate thread
            client_thread = threading.Thread(target=self.accept_client, args=(server_socket,))
            client_thread.start()
            self.client_threads.append(client_thread)

    def accept_client(self, server_socket: socket.socket) -> None:
        """Accept client connection or reconnection"""
        print("Waiting for a TCP listener to connect...")
        while not self.connections_closed:
            try:
                accept_result: tuple[socket.socket, object] = server_socket.accept()
                client_socket = accept_result[0]
                addr_info_raw: object = accept_result[1]
                addr_info = cast(tuple[str, int] | tuple[str, int, int, int], addr_info_raw)
                with self.lock:
                    self.client_sockets.append(client_socket)
                client_addr = cast(tuple[str, int], addr_info)
                print(f"Connected by {client_addr}.  Socket={client_socket}")
            except BlockingIOError:
                pass
            finally:
                time.sleep(1.0)

    def send_transcription(self, transcription: str) -> None:
        """Send buffered transcription over TCP connection."""
        msg = transcription.encode('utf-8')
        with self.lock:
            for client_socket in self.client_sockets[:]:
                try:
                    client_socket.sendall(msg)
                    print(f'Sending over TCP: "{transcription}"')
                except (ConnectionResetError, BrokenPipeError):
                    print("Client disconnected unexpectedly.")
                    self.client_sockets.remove(client_socket)
                    client_socket.close()
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
        with self.lock:
            for client_socket in self.client_sockets[:]:
                try:
                    client_socket.sendall(encoded_message)
                except (ConnectionResetError, BrokenPipeError):
                    print("Client disconnected unexpectedly.")
                    self.client_sockets.remove(client_socket)
                    client_socket.close()
                except Exception as e:
                    print(f"Error sending transcription: {e}")

    def close_connections(self):
        """Close TCP connections."""
        print("Closing TCP connections...")
        self.connections_closed = True
        with self.lock:
            for client_socket in self.client_sockets:
                client_socket.close()
            self.client_sockets.clear()
        for server_socket in self.server_sockets:
            server_socket.close()
        for client_thread in self.client_threads:
            client_thread.join()
