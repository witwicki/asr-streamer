import socket
import threading
import time
import json

class TranscriptionServer:
    """TranscriptionServer
    This class is responsible for sending live transcriptions to exactly one listener.

    """
    def __init__(self):
        self.server_socket = None
        self.client_socket = None
        self.asr_active = False
        self.connections_closed = False
        self.buffer = ""

    def start_server(self, host='0.0.0.0', port=27400):
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

    def accept_client(self):
        """Accept client connection or reconnection"""
        while not self.connections_closed:
            try:
                if self.server_socket:
                    self.client_socket, addr = self.server_socket.accept()
                    print(f"Connected by {addr}.  Socket={self.client_socket}")
            except BlockingIOError:
                pass
            finally:
                if self.client_socket is None:
                    print("...waiting for a TCP listener to connect...")
                time.sleep(1.0)

    def send_transcription(self, transcription):
        """Send buffered transcription over TCP connection."""
        if self.client_socket:
            try:
                # Send as JSON string or another format
                msg = ''.join(transcription).encode('utf-8')
                self.client_socket.sendall(msg)
                print(f'Sending over TCP: "{transcription}"')
            except ConnectionResetError:
                print("Client disconnected unexpectedly.")
                self.close_connections()
            except Exception as e:
                print(f"Error sending transcription: {e}")

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
                print(f'Sending over TCP: "{transcription}"')
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
        self.client_thread.join()
