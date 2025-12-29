#./.venv/bin/python sac/sac.py
import gymnasium as gym
import os
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

def train_sac():
    env_name = "BipedalWalker-v3"
    log_dir = "sac_logs/"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(env_name)
    env = Monitor(env, log_dir)
    
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
    )

    print("Training SAC Model...")
    model.learn(total_timesteps=100000)
    model.save("sac_bipedal_model")
    print("Model Saved.")

def test_and_record_sac():
    env_name = "BipedalWalker-v3"
    video_folder = "sac_test_videos/"
    num_test_episodes = 5
    
    def make_env():
        return Monitor(gym.make(env_name, render_mode="rgb_array"))

    test_venv = DummyVecEnv([make_env])
    test_venv = VecVideoRecorder(
        test_venv, 
        video_folder, 
        record_video_trigger=lambda x: x == 0, 
        video_length=2000, 
        name_prefix="sac_eval"
    )

    model = SAC.load("sac_bipedal_model")
    
    print("\n" + "="*40)
    print("SAC OFFICIAL TEST SCOREBOARD")
    print("="*40)
    
    all_scores = []

    for ep in range(num_test_episodes):
        obs = test_venv.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_venv.step(action)
            
            if done:
                final_score = info[0]['episode']['r']
                all_scores.append(final_score)
                print(f"Episode {ep+1}: Score = {final_score:.2f}")

    test_venv.close()
    
    print("-" * 40)
    print(f"AVERAGE TEST SCORE: {np.mean(all_scores):.2f}")
    print(f"MAX TEST SCORE:     {np.max(all_scores):.2f}")
    print(f"Videos saved to: {video_folder}")

if __name__ == "__main__":
    train_sac()
    test_and_record_sac()