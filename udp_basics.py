"""Raw UDP in a few lines.
Terminal 1:  python udp_basics.py recv
Terminal 2:  python udp_basics.py send
"""
import socket, struct, sys, time

ADDR = ("127.0.0.1", 9000)
FMT = "<Ifff"   # binary packet, little-endian: uint32 sequence + 3 float32 (x, y, heading)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # SOCK_DGRAM = UDP

if sys.argv[1] == "send":
    for seq in range(100):
        sock.sendto(struct.pack(FMT, seq, seq * 0.1, 0.0, 0.0), ADDR)   # fire and forget
        time.sleep(0.05)                                                # 20 Hz 
    print("sent 100 packets")
else:
    sock.bind(ADDR)
    sock.settimeout(1.0)           # lets Ctrl+C work on Windows 
    last = -1
    while True:
        try:
            data, sender = sock.recvfrom(1024)
        except socket.timeout:
            continue
        seq, x, y, h = struct.unpack(FMT, data)
        if seq != last + 1:
            print(f"  gap: lost or reordered packets before #{seq}")
        last = seq
        print(f"#{seq:3d}  x={x:5.2f}  ({len(data)} bytes from {sender})")
