import socket
import struct
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("SV2_Client")

# Stratum V2 Protocol Constants
SV2_VERSION = 2
SV2_PORT_DEFAULT = 34255

# Message Types
MSG_SETUP_CONNECTION = 0x00
MSG_SETUP_CONNECTION_SUCCESS = 0x01
MSG_OPEN_STANDARD_MINING_CHANNEL = 0x10
MSG_OPEN_STANDARD_MINING_CHANNEL_SUCCESS = 0x11

class StratumV2Client:
    """
    Stratum V2 Native Client
    =======================
    Implements the binary framing and handshake protocol for Stratum V2.
    SV2 improves security (Noise protocol), reduces bandwidth (binary), 
    and allows for decentralized job selection.
    """

    def __init__(self, host: str, port: int = SV2_PORT_DEFAULT):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False
        self.noise_handshake_complete = False

    def _create_frame(self, msg_type: int, payload: bytes) -> bytes:
        """
        Create an SV2 Binary Frame:
        - Extension Type (2 bytes, usually 0)
        - Message Type (1 byte)
        - Payload Length (3 bytes, little-endian)
        - Payload (N bytes)
        """
        ext_type = 0
        payload_len = len(payload)
        
        # Pack header: H (unsigned short), B (unsigned char), 3s (3 bytes for length)
        # Using a workaround for 3-byte length in struct
        header = struct.pack('<HB', ext_type, msg_type)
        header += payload_len.to_bytes(3, byteorder='little')
        
        return header + payload

    def connect(self) -> bool:
        """Initiate connection and SV2 Handshake."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.host, self.port))
            
            # 1. Send SetupConnection
            # protocol (2), min_version (2), max_version (2), flags (4)
            # protocol=0 (Mining), versions=2, flags=0
            payload = struct.pack('<HHHI', 0, 2, 2, 0)
            frame = self._create_frame(MSG_SETUP_CONNECTION, payload)
            
            self.socket.sendall(frame)
            logger.info(f"Sent SV2 SetupConnection to {self.host}")
            
            # 2. Receive Response (Header is 6 bytes)
            header = self.socket.recv(6)
            if len(header) < 6:
                return False
                
            ext, msg_type, l1, l2, l3 = struct.unpack('<HBBBB', header)
            p_len = l1 | (l2 << 8) | (l3 << 16)
            
            if msg_type == MSG_SETUP_CONNECTION_SUCCESS:
                logger.info("SV2 Connection Successful (Unauthenticated)")
                self.is_connected = True
                return True
            else:
                logger.error(f"SV2 Setup Failed. Msg Type: {msg_type}")
                return False
                
        except Exception as e:
            logger.error(f"SV2 Connection Error: {e}")
            return False

    def open_channel(self, user: str) -> bool:
        """Open a standard mining channel."""
        if not self.is_connected:
            return False
            
        try:
            # dummy request_id (4), flags (4), nominal_hash_rate (4), min_diff (8), user_len (1), user
            user_bytes = user.encode()
            payload = struct.pack('<IIIfB', 0, 0, 0, 1.0, len(user_bytes)) + user_bytes
            frame = self._create_frame(MSG_OPEN_STANDARD_MINING_CHANNEL, payload)
            
            self.socket.sendall(frame)
            logger.info(f"SV2 Opening Channel for {user}...")
            return True
        except Exception as e:
            logger.error(f"SV2 Channel Error: {e}")
            return False

if __name__ == "__main__":
    # Test stub (requires an SV2 proxy like SRI or Braiins)
    logging.basicConfig(level=logging.INFO)
    client = StratumV2Client("127.0.0.1", 34255)
    if client.connect():
        client.open_channel("HCE_Worker_01")
