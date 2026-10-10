"""Stream boat pos. into Unreal Engine over UDP, as OSC messages.
   python stream_to_unreal.py            - uses the zip file that contains trained neural net.
   python stream_to_unreal.py scripted   - my manually coded controller (no training needed)
"""
import math, sys, time
from pythonosc.udp_client import SimpleUDPClient
from boat_dock_env import BoatDockEnv
from evaluate import scripted

client = SimpleUDPClient("127.0.0.1", 8000)   # must match the port 8000 Unreal


def to_unreal(x, y, psi):
    """Sim frame: meters, x forward, y LEFT, heading counter-clockwise (right-handed).
    - Unreal: in cm, (hence the x100) X forward, Y RIGHT, Z up (left-handed), positive yaw turns toward +Y."""
    return x * 100.0, -y * 100.0, -math.degrees(psi)


mode = sys.argv[1] if len(sys.argv) > 1 else "ppo"
model = None
if mode == "ppo":
    from stable_baselines3 import PPO
    model = PPO.load("ppo_boat")

env = BoatDockEnv()
for ep in range(5):
    obs, _ = env.reset(seed=2000 + ep)
    while True:
        action = model.predict(obs, deterministic=True)[0] if model else scripted(env)
        obs, r, term, trunc, info = env.step(action)
        client.send_message("/boat/pose", [float(v) for v in to_unreal(env.x, env.y, env.psi)])
        time.sleep(env.dt)                         # real-time playback at 10 Hz
        if term or trunc:
            print(f"episode {ep}: {'docked' if info['success'] else 'did not dock'}")
            time.sleep(1.0)
            break
