"""Compare PPO vs a random policy vs a classical controller, then plot PPO's paths."""
import numpy as np
from boat_dock_env import BoatDockEnv, wrap


def scripted(env):
    """My scripted 98% baseline. baseline: aim at a point 4 m before the slip, then @ slip.
    Rudder = PD control on bearing error. Throttle = P control + target speed
    that shrinks as the boat gets close (to reduce vel. down near the dock)."""
    tx = 0.0 if env.x > -4 else -4.0
    bearing = np.arctan2(-env.y, tx - env.x)
    err = wrap(bearing - env.psi)
    rudder = np.clip(3 * err - 0.5 * env.r, -1, 1)
    target_u = np.clip(0.25 * np.hypot(env.x, env.y), 0.4, 2.0)
    throttle = np.clip(1.5 * (target_u - env.u) + 0.3 * target_u, -1, 1)
    return np.array([throttle, rudder])


def run(name, n=100, model=None, **env_kwargs):
    env = BoatDockEnv(**env_kwargs)
    successes, returns, paths = 0, [], []
    for ep in range(n):
        obs, _ = env.reset(seed=1000 + ep)          # start positions never used in training
        total, path = 0.0, [(env.x, env.y)]
        while True:
            if name == "ppo":
                action, _ = model.predict(obs, deterministic=True)
            elif name == "scripted":
                action = scripted(env)
            else:
                action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            total += r
            path.append((env.x, env.y))
            if term or trunc:
                break
        successes += info["success"]
        returns.append(total)
        paths.append(np.array(path))
    print(f"  {name:9s} success {successes / n:5.0%}   mean return {np.mean(returns):7.1f}")
    return paths


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO

    model = PPO.load("ppo_boat")
    print("Calm water (training conditions):")
    paths = run("ppo", model=model)
    run("scripted")
    run("random")
    print("With a sideways current the agent never saw (robustness test):")
    run("ppo", model=model, current=0.12)
    run("scripted", current=0.12)

    for p in paths[:25]:
        plt.plot(p[:, 0], p[:, 1], alpha=0.6)
    plt.scatter([0], [0], marker="s", s=120, c="k", label="dock slip")
    plt.axis("equal"); plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title("PPO docking paths (25 unseen starts)"); plt.legend()
    plt.savefig("trajectories.png", dpi=150)
    print("saved trajectories.png")
    plt.show()
