import socket
import time

try:
    from choreography.sasrc import ASRChoreographer
except ImportError:
    # avoiding circular dependency
    ASRChoreographer = None

class RemoteControl:
    """ Control an asr session remotely
    
    In the simplest form, button presses and releases start and stop the transciption-gathering
    choroegrapher.
    """    

    def __init__(self, udp_host: str, udp_port: int, asr_choreographer: ASRChoreographer, mode="toggle"):
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.asr_choreographer = asr_choreographer
        self.mode = mode
        # assume that the current state of control is the initial state of the choreographer
        self.current_state = asr_choreographer.is_active 
        self.was_killed = False
    
    def run(self):
        self._listen_to_boolean_signal()

    def switch_mode(self):
        if self.mode == "toggle":
            self.mode = "press_and_hold_to_talk"
        else:
            self.mode = "toggle"
    
    def end(self):
        self.was_killed = True

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

                    if self.mode == "toggle":
                        # Detect transition from False to True
                        if self.current_state == False and new_control_state == True:
                            self.asr_choreographer.toggle_asr()
                    elif self.mode == "press_and_hold_to_talk":
                        if self.current_state == False and new_control_state == True:
                            self.asr_choreographer.activate_asr()
                        elif self.current_state == True and new_control_state == False:
                            self.asr_choreographer.deactivate_asr()
                        
                    # Update previous boolean state for next iteration
                    self.current_state = new_control_state
                except BlockingIOError as e:
                    #print(f"{e}")
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


