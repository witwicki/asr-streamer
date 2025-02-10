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
            if not data:
                break
            print(f"Received: {data.decode('utf-8')}")

if __name__ == "__main__":
    main()
