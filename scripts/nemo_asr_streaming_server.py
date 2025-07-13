#!/usr/bin/env python3

import time
import sys
from threading import Thread

from streaming.asr import ASRStreamer
from streaming.mic import AudioStreamManager
from communication.server import TranscriptionServer
from communication.rc import RemoteControl, RemoteControlByUDP, RemoteControlByKeyboard
from choreography.sasrc import ASRChoreographer

from pynput import keyboard

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
@click.option('--toggle_button_control', is_flag=True, help='Set control mode to: button press turns on and off ASR, as opposed to the default press-and-hold-to-talk controls')
@click.option('--keyboard', is_flag=True, help='Use the keyboard as your press-to-talk controller')
@click.option('--verbose', '-v', is_flag=True, help='Show debug info and warnings on the terminal')
def main(lookahead, decoder_type, decoding_strategy, tcp_server_port, rc_udp_host, rc_udp_port, toggle_button_control, keyboard, verbose):

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
        exercise_output_channel=(sys.platform!="darwin"), # patch sound problem unless on macos
        verbose=verbose
    )

    # Start audio stream in separate thread
    stream_manager = AudioStreamManager(asr_choreographer, lookahead_size + ASRStreamer.ENCODER_STEP_LENGTH, verbose=verbose)
    thread_stream = Thread(target=run_stream_manager, args=(stream_manager,))
    thread_stream.start()

    # Start transcription server, which takes care of its own threading
    server.start_server(port=tcp_server_port)

    # start remote control interface in another thread
    remote_control_interface : RemoteControl
    if keyboard:
        remote_control_interface = RemoteControlByKeyboard(asr_choreographer, mode=control_mode)
    else:
        remote_control_interface = RemoteControlByUDP(asr_choreographer, udp_host=rc_udp_host, udp_port=rc_udp_port, mode=control_mode)
    thread_remote = Thread(target=run_remote_control, args=(remote_control_interface,))
    thread_remote.start()

    # loop until
    while not asr_choreographer.was_killed:
        time.sleep(0.01)

    # cleanup
    # TODO: more graceful cleanup
    stream_manager.stop_stream()
    server.close_connections()
    remote_control_interface.end()
    thread_remote.join()
    thread_stream.join()






if __name__ == "__main__":
    main()
