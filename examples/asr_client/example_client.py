#!/usr/bin/env python3

import socket

""" Example TCP client set up to receive live transcription results """

def main():
    host = 'localhost'
    port = 27400

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        while True:
            data = s.recv(1024)
            decoded_data = ""
            if data:
                decoded_data = data.decode('utf-8')
            print(f"Received: {decoded_data}")

if __name__ == "__main__":
    main()
