"""Train PPO on the boat-docking envr (takes ~1:20.) 8x boats; each reports after 1024, until 400k steps."""
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from boat_dock_env import BoatDockEnv

if __name__ == "__main__":
    env = make_vec_env(BoatDockEnv, n_envs=8, seed=0)
    model = PPO(
        "MlpPolicy", env,            
        n_steps=1024,                
        batch_size=256,
        gamma=0.99,                
        gae_lambda=0.95,            ##
        learning_rate=3e-4,         ##
        clip_range=0.2,              
        verbose=1, seed=0,
    )
    model.learn(total_timesteps=400_000, progress_bar=True)
    model.save("ppo_boat")
    print("saved ppo_boat.zip")

