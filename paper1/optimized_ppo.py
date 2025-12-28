# ./.venv/bin/python paper1/optimized_ppo.py

import gymnasium as gym
import torch.nn as nn
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# 1. MINIMAL REWARD BIASING
# Based on your finding that simple shaping > complex height penalties
class BipedalOptimizedWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # Focus on "smooth reward biasing" rather than aggressive penalties
        vel_x = self.env.unwrapped.hull.linearVelocity[0]
        forward_bonus = 0.1 * max(0, vel_x) 
        return reward + forward_bonus

class TrainingLogger(BaseCallback):
    def __init__(self, verbose=1):
        super(TrainingLogger, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            if 'episode' in info.keys():
                self.episode_count += 1
                print(f"EP {self.episode_count} | Score: {info['episode']['r']:.2f}")
        return True

def make_env():
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    env = BipedalOptimizedWrapper(env)
    return Monitor(env)

# --- TRAINING PHASE ---
video_folder = "logs/ppo_ablation_optimized"
os.makedirs(video_folder, exist_ok=True)

venv = DummyVecEnv([make_env])

# RESEARCH OPTIMIZATION 6 & 7: Observation Normalization and Clipping [cite: 814, 816]
# Your ablation found this was the "dominant factor for successful learning."
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

# RESEARCH OPTIMIZATION 3 & 8: Orthogonal Init & Tanh [cite: 811, 818]
policy_kwargs = dict(
    activation_fn=nn.Tanh,
    ortho_init=True,
    net_arch=dict(pi=[64, 64], vf=[64, 64])
)

model = PPO(
    "MlpPolicy",
    venv,
    verbose=1,
    learning_rate=3e-4, 
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    # RESEARCH OPTIMIZATION 9: Global Gradient Clipping 
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs
)

print("Training Optimized PPO (Input Stabilization Focus)...")
model.learn(total_timesteps=100000, callback=TrainingLogger())

model.save("ppo_optimized_final")
venv.save("normalization_stats.pkl")
venv.close()

# --- TESTING PHASE ---
print("\n" + "="*50 + "\nTESTING PHASE\n" + "="*50)

test_venv = DummyVecEnv([make_env])
test_venv = VecNormalize.load("normalization_stats.pkl", test_venv)
test_venv.training = False 
test_venv.norm_reward = False 

model = PPO.load("ppo_optimized_final") 

for i in range(5):
    eval_env = VecVideoRecorder(
        test_venv, video_folder,
        record_video_trigger=lambda x: x == 0, 
        video_length=2000, name_prefix=f"optimized-test-{i+1}"
    )
    obs = eval_env.reset() 
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = eval_env.step(action) 
        if done:
            print(f"Test Episode {i+1}: Official Score = {info[0]['episode']['r']:.2f}")
    eval_env.close()

test_venv.close()