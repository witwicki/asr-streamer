#!/usr/bin/env python3

import time
import sys
import select
import termios
import tty
from threading import Thread
from pathlib import Path

from streaming.asr import ASRStreamer
from streaming.mic import AudioStreamManager
from communication.server import TranscriptionServer
from communication.rc import RemoteControl
from choreography.sasrc import ASRChoreographer

import click

def run_stream_manager(stream_manager):
    stream_manager.start_stream()

def run_remote_control(remote_control):
    remote_control.run()

@click.command()
@click.option('--tcp_server_port', default=27400, help='The port over which to serve TCP messages, e.g., 27400')
@click.option('--lookahead', type=click.Choice(['0', '80', '480', '1040']), default='80', help='Lookahead size for streaming ASR in ms')
@click.option('--decoder_type', type=click.Choice(['rnnt']), default='rnnt') #, 'ctc'])
@click.option('--decoding_strategy', type=click.Choice(['greedy', 'beam']), default='greedy')
@click.option('--rc_udp_host', default='0.0.0.0', help='The address (e.g., 0.0.0.0) over which to listen to UDP packets sent from the Remote-Control')
@click.option('--rc_udp_port', default=5656, help='The port of the remote-control UDP host')
@click.option('--toggle_button_control', is_flag=True, help='Initialize control mode to: button press turns on and off ASR, as opposed to the default press-and-hold-to-talk controls')
@click.option('--verbose', '-v', is_flag=True, help='Show debug info and warnings on the terminal')
def main(lookahead, decoder_type, decoding_strategy, tcp_server_port, rc_udp_host, rc_udp_port, toggle_button_control, verbose):
    # Model configuration
    model_name = "stt_en_fastconformer_hybrid_large_streaming_multi"
    lookahead_size = int(lookahead)
    streaming_result_delay_silence_threshold = 0.5 # seconds
    silence_threshold = 0.5 # seconds

    if(toggle_button_control):
        control_mode = "toggle"
    else:
        control_mode = "press_and_hold_to_talk"

    # Initialize components
    asr_streamer = ASRStreamer(
        model_name,
        lookahead_size,
        decoder_type=decoder_type,
        decoding_strategy=decoding_strategy,
        verbose=verbose
    )
    server = TranscriptionServer()

    # Create choreographer instance
    asr_choreographer = ASRChoreographer(
        asr_streamer=asr_streamer,
        transcription_server=server,
        streaming_result_delay_silence_threshold=streaming_result_delay_silence_threshold,
        silence_threshold=silence_threshold,
        verbose=verbose
    )

    # Start audio stream in separate thread
    stream_manager = AudioStreamManager(asr_choreographer, lookahead_size + ASRStreamer.ENCODER_STEP_LENGTH, verbose=verbose)
    thread_stream = Thread(target=run_stream_manager, args=(stream_manager,))
    thread_stream.start()

    # Start transcription server, which takes care of its own threading
    server.start_server(port=tcp_server_port)

    # start remote control interface in another thread
    remote_control_interface = RemoteControl(rc_udp_host, rc_udp_port, asr_choreographer, mode=control_mode)
    thread_remote = Thread(target=run_remote_control, args=(remote_control_interface,))
    thread_remote.start()

    # Configure terminal for non-blocking input
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        print("\n\nStreaming ASR Model loaded. Press 's' to manually toggle state {active, paused}, and 'm' to switch to toggle mode\n")
        while not asr_choreographer.was_killed:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 's':
                    asr_choreographer.toggle_asr()
                elif key == 'm':
                    remote_control_interface.switch_mode()
                elif key == 'z':
                    asr_choreographer.play_activate_sound()
                elif key == 'x':
                    asr_choreographer.play_deactivate_sound()
                elif key == 'q':
                    break
            # Check for silence timeout
            #asr_choreographer.check_silence()
            time.sleep(0.01)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        stream_manager.stop_stream()
        thread_stream.join()
        server.close_connections()
        remote_control_interface.end()
        thread_remote.join()


if __name__ == "__main__":
    main()
