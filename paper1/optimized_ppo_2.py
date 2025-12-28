# ./.venv/bin/python paper1/optimized_ppo_2.py

import gymnasium as gym
import torch.nn as nn
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Learning Rate Annealing (optimization 4 from paper - https://arxiv.org/pdf/2005.12729)
#linear schedule that decreases from 100% to 0%  of the initial value over the course of training.
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

class BipedalRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        obs = self.env.unwrapped.hull.angle 
        vel_x = self.env.unwrapped.hull.linearVelocity[0]
        stability_penalty = -0.05 * abs(obs) 
        forward_bonus = 0.1 * max(0, vel_x)
        return reward + stability_penalty + forward_bonus

class TrainingLogger(BaseCallback):
    def __init__(self, verbose=1):
        super(TrainingLogger, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            if 'episode' in info.keys():
                self.episode_count += 1
                print(f"EP {self.episode_count} | Official Score: {info['episode']['r']:.2f}")
        return True

def make_env():
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    env = BipedalRewardWrapper(env)
    return Monitor(env)

video_folder = "logs/ppo_lr_annealing"
os.makedirs(video_folder, exist_ok=True)

venv = DummyVecEnv([make_env])
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

policy_kwargs = dict(
    activation_fn=nn.Tanh,
    ortho_init=True,
    net_arch=dict(pi=[64, 64], vf=[64, 64])
)

model = PPO(
    "MlpPolicy",
    venv,
    verbose=1,
    learning_rate=linear_schedule(3e-4), 
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs
)

print("Starting training with Learning Rate Annealing...")
model.learn(total_timesteps=100000, callback=TrainingLogger())

model.save("ppo_lr_annealed_model")
venv.save("lr_stats.pkl")
venv.close()

print("\n" + "="*50 + "\nTESTING PHASE\n" + "="*50)

test_venv = DummyVecEnv([make_env])
test_venv = VecNormalize.load("lr_stats.pkl", test_venv)
test_venv.training = False 
test_venv.norm_reward = False 

model = PPO.load("ppo_lr_annealed_model") 

for i in range(5):
    eval_env = VecVideoRecorder(
        test_venv, video_folder,
        record_video_trigger=lambda x: x == 0, 
        video_length=2000, name_prefix=f"lr-test-ep-{i+1}"
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