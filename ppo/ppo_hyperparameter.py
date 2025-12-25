#Modified hyperparameters of basic PPO implementation
# Run: ./.venv/bin/python ppo/ppo_hyperparameter.py

# Experiments:
# 1. Default Run: highest 116.36
# 2. Learning Rate = 0.0007 (# --- Episode Finished | Final Score: 167.29 ---)
# 3. Learning Rate = 0.001 (# --- Episode Finished | Final Score: 123.23 ---)
# 4. learning_rate=0.00001, AND default total_timesteps (# --- Episode Finished | Final Score: -27.05 ---)
# 5. # learning_rate=0.0001 AND total_timesteps=200,000 (# --- Episode Finished | Final Score: 146.44 ---)

#observations: 
# Higher learning rates (0.0007) significantly speed up progress toward the 300-point goal, 
# but pushing too high (0.001) causes instability and lower peak scores. Extremely low learning rates (0.00001)
#  fail to learn basic movement within the time budget.


import gymnasium as gym
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os

class PointTrackerCallback(BaseCallback):
    def __init__(self, check_freq_steps=400, verbose=1):
        super(PointTrackerCallback, self).__init__(verbose)
        self.check_freq_steps = check_freq_steps
        self.current_episode_reward = 0
        self.episode_steps = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward
        self.episode_steps += 1

        if self.episode_steps % self.check_freq_steps == 0:
            print(f"-> Step {self.episode_steps}: Current Points = {self.current_episode_reward:.2f}")

        if self.locals['dones'][0]:
            print(f"--- Episode Finished | Final Score: {self.current_episode_reward:.2f} ---")
            self.current_episode_reward = 0
            self.episode_steps = 0
        return True

def make_env():
    env = gym.make("BipedalWalker-v3", render_mode="none") 
    return Monitor(env)

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=0.0001, #0.0003 default lr
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
)

print("Starting training...")
model.learn(total_timesteps=100000, callback=PointTrackerCallback()) #default 100000

model.save("ppo_bipedal_walker_model")
env.close()

#Default Run:  116.36
# -> Step 400: Current Points = 25.06
# -> Step 800: Current Points = 65.27
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 887         |
# |    ep_rew_mean          | -62.9       |
# | time/                   |             |
# |    fps                  | 2736        |
# |    iterations           | 48          |
# |    time_elapsed         | 35          |
# |    total_timesteps      | 98304       |
# | train/                  |             |
# |    approx_kl            | 0.010081818 |
# |    clip_fraction        | 0.0832      |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -4.38       |
# |    explained_variance   | 0.846       |
# |    learning_rate        | 0.0003      |
# |    loss                 | 1.07        |
# |    n_updates            | 470         |
# |    policy_gradient_loss | -0.00699    |
# |    std                  | 0.724       |
# |    value_loss           | 4.22        |
# -----------------------------------------
# -> Step 1200: Current Points = 102.81
# -> Step 1600: Current Points = 116.36
# --- Episode Finished | Final Score: 116.36 ---

#Learning Rate: 0.0007 = 167.29
# -> Step 400: Current Points = 39.47
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 1.18e+03    |
# |    ep_rew_mean          | -56.9       |
# | time/                   |             |
# |    fps                  | 46          |
# |    iterations           | 47          |
# |    time_elapsed         | 2050        |
# |    total_timesteps      | 96256       |
# | train/                  |             |
# |    approx_kl            | 0.020758614 |
# |    clip_fraction        | 0.244       |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -3.55       |
# |    explained_variance   | 0.86        |
# |    learning_rate        | 0.0007      |
# |    loss                 | 0.157       |
# |    n_updates            | 460         |
# |    policy_gradient_loss | -0.0123     |
# |    std                  | 0.584       |
# |    value_loss           | 0.395       |
# -----------------------------------------
# -> Step 800: Current Points = 83.47
# -> Step 1200: Current Points = 130.83
# -> Step 1600: Current Points = 167.29
# --- Episode Finished | Final Score: 167.29 ---

#Learning Rate: 0.001
# -> Step 400: Current Points = 24.28
# -> Step 800: Current Points = 60.60
# -> Step 1200: Current Points = 94.50
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 1.04e+03    |
# |    ep_rew_mean          | -62.6       |
# | time/                   |             |
# |    fps                  | 42          |
# |    iterations           | 43          |
# |    time_elapsed         | 2053        |
# |    total_timesteps      | 88064       |
# | train/                  |             |
# |    approx_kl            | 0.030335896 |
# |    clip_fraction        | 0.313       |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -3.78       |
# |    explained_variance   | 0.871       |
# |    learning_rate        | 0.001       |
# |    loss                 | 0.114       |
# |    n_updates            | 420         |
# |    policy_gradient_loss | -0.0194     |
# |    std                  | 0.624       |
# |    value_loss           | 0.241       |
# -----------------------------------------
# -> Step 1600: Current Points = 123.23
# --- Episode Finished | Final Score: 123.23 ---

#learning_rate=0.00001, AND default total_timesteps=100,000
# -> Step 400: Current Points = -1.84
# -> Step 800: Current Points = -11.01
# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 932         |
# |    ep_rew_mean          | -91.4       |
# | time/                   |             |
# |    fps                  | 2737        |
# |    iterations           | 47          |
# |    time_elapsed         | 35          |
# |    total_timesteps      | 96256       |
# | train/                  |             |
# |    approx_kl            | 0.008247184 |
# |    clip_fraction        | 0.0724      |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -5.06       |
# |    explained_variance   | 0.265       |
# |    learning_rate        | 0.0001      |
# |    loss                 | 0.0513      |
# |    n_updates            | 460         |
# |    policy_gradient_loss | -0.0087     |
# |    std                  | 0.854       |
# |    value_loss           | 0.14        |
# -----------------------------------------
# -> Step 1200: Current Points = -18.05
# -> Step 1600: Current Points = -27.05
# --- Episode Finished | Final Score: -27.05 ---

# learning_rate=0.0001 AND total_timesteps=200,000
# -> Step 400: Current Points = 39.79
# -> Step 800: Current Points = 65.78
# -> Step 1200: Current Points = 101.77
# ------------------------------------------
# | rollout/                |              |
# |    ep_len_mean          | 941          |
# |    ep_rew_mean          | -25.1        |
# | time/                   |              |
# |    fps                  | 2785         |
# |    iterations           | 95           |
# |    time_elapsed         | 69           |
# |    total_timesteps      | 194560       |
# | train/                  |              |
# |    approx_kl            | 0.0043804296 |
# |    clip_fraction        | 0.0302       |
# |    clip_range           | 0.2          |
# |    entropy_loss         | -4.67        |
# |    explained_variance   | 0.402        |
# |    learning_rate        | 0.0001       |
# |    loss                 | 2.63         |
# |    n_updates            | 940          |
# |    policy_gradient_loss | -0.00473     |
# |    std                  | 0.778        |
# |    value_loss           | 32.7         |
# ------------------------------------------
# -> Step 1600: Current Points = 146.44
# --- Episode Finished | Final Score: 146.44 ---