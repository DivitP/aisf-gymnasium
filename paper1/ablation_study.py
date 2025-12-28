# ./.venv/bin/python paper1/ablation_study.py
import gymnasium as gym
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class SimpleRewardWrapper(gym.RewardWrapper):
    """Hull angle and velocity"""
    def __init__(self, env):
        super().__init__(env)
    def reward(self, reward):
        obs = self.env.unwrapped.hull.angle
        vel_x = self.env.unwrapped.hull.linearVelocity[0]
        return reward - (0.1 * abs(obs)) + (0.05 * max(0, vel_x))

class ResearchRewardWrapper(gym.RewardWrapper):
    """height safety and control cost added"""
    def __init__(self, env):
        super().__init__(env)
        self.current_action = np.zeros(env.action_space.shape)

    def step(self, action):
        self.current_action = action
        return super().step(action)

    def reward(self, reward):
        env = self.env.unwrapped
        hull_angle = env.hull.angle
        hull_height = env.hull.position[1]
        vel_x = env.hull.linearVelocity[0]
        
        stability_penalty = -0.2 * abs(hull_angle)
        height_penalty = -2.0 * max(0, 0.9 - hull_height)
        forward_bonus = 0.1 * max(0, vel_x)
        control_cost = -0.01 * np.sum(np.square(self.current_action))
        
        return reward + stability_penalty + height_penalty + forward_bonus + control_cost

class AblationLoggerCallback(BaseCallback):
    """training metrics for graphs"""
    def __init__(self):
        super().__init__()
        self.history = {'entropy': [], 'value_loss': []}

    def _on_step(self) -> bool:
        ent = self.logger.name_to_value.get('train/entropy_loss', np.nan)
        vf_loss = self.logger.name_to_value.get('train/value_loss', np.nan)
        self.history['entropy'].append(ent)
        self.history['value_loss'].append(vf_loss)
        return True

class EntropyDecayCallback(BaseCallback):
    def __init__(self, start=0.01, end=0.001, total_steps=100000):
        super().__init__()
        self.start, self.end, self.total_steps = start, end, total_steps
    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self.total_steps)
        self.model.ent_coef = self.start * (1 - progress) + self.end * progress
        return True

def run_ablation():
    STEPS = 100000
    experiments = [
        ("1_Baseline", None, False, False),
        ("2_Shaping", SimpleRewardWrapper, False, False),
        ("3_Normalization", SimpleRewardWrapper, True, False),
        ("4_Height_Penalty", ResearchRewardWrapper, True, False),
        ("5_Full_Research", ResearchRewardWrapper, True, True),
    ]

    all_histories = {}
    base_log_dir = "ablation_study"
    os.makedirs(base_log_dir, exist_ok=True)

    for name, wrap, norm, decay in experiments:
        log_dir = f"{base_log_dir}/{name}_train"
        os.makedirs(log_dir, exist_ok=True)
        
        def make_env():
            env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
            if wrap: env = wrap(env)
            return Monitor(env, log_dir)

        venv = DummyVecEnv([make_env])
        if norm: venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

        model = PPO("MlpPolicy", venv, verbose=0, learning_rate=3e-4,
                    n_steps=4096 if decay else 2048,
                    batch_size=128 if decay else 64,
                    policy_kwargs=dict(activation_fn=nn.Tanh, ortho_init=True),
                    max_grad_norm=0.5, ent_coef=0.01)

        logger_cb = AblationLoggerCallback()
        callbacks = [logger_cb]
        if decay: callbacks.append(EntropyDecayCallback(total_steps=STEPS))

        print(f"Training {name}...")
        model.learn(total_timesteps=STEPS, callback=callbacks)
        
        model.save(f"{base_log_dir}/model_{name}")
        if norm: venv.save(f"{base_log_dir}/stats_{name}.pkl")
        all_histories[name] = logger_cb.history
        venv.close()

    print("\n" + "="*50)
    print("STARTING TESTING PHASE & SCOREBOARD")
    print("="*50)
    
    final_scoreboard = []

    for name, wrap, norm, decay in experiments:
        test_dir = f"{base_log_dir}/{name}_test_videos"
        os.makedirs(test_dir, exist_ok=True)

        def make_test_env():
            # Use raw env to get the official, un-shapped game score
            return Monitor(gym.make("BipedalWalker-v3", render_mode="rgb_array"))

        test_venv = DummyVecEnv([make_test_env])
        if norm:
            test_venv = VecNormalize.load(f"{base_log_dir}/stats_{name}.pkl", test_venv)
            test_venv.training = False
            test_venv.norm_reward = False

        test_venv = VecVideoRecorder(test_venv, test_dir, 
                                     record_video_trigger=lambda x: x == 0, 
                                     video_length=2000, name_prefix=f"test_{name}")

        model = PPO.load(f"{base_log_dir}/model_{name}")
        
        model_scores = []
        for ep in range(5):
            obs = test_venv.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = test_venv.step(action)
                if done:
                    score = info[0]['episode']['r']
                    model_scores.append(score)
                    print(f"Model: {name:20} | Episode {ep+1} | Score: {score:.2f}")
        
        final_scoreboard.append({
            "Experiment": name,
            "Average Score": np.mean(model_scores),
            "Max Score": np.max(model_scores)
        })
        test_venv.close()

    # --- FINAL SUMMARY TABLE ---
    print("\n" + "="*50)
    print("FINAL ABLATION SUMMARY")
    print("="*50)
    summary_df = pd.DataFrame(final_scoreboard)
    print(summary_df.to_string(index=False))
    
    generate_graphs(base_log_dir, experiments, all_histories)

def generate_graphs(base_dir, experiments, histories):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    
    for name, _, _, _ in experiments:
        df = pd.read_csv(f"{base_dir}/{name}_train/monitor.csv", skiprows=1)
        axs[0, 0].plot(df['l'].cumsum(), df['r'].rolling(10).mean(), label=name)
        axs[0, 1].plot(histories[name]['value_loss'], label=name)
        axs[1, 0].plot(histories[name]['entropy'], label=name)

    axs[0, 0].set_title("Training Reward (Smoothed)")
    axs[0, 1].set_title("Value Function Loss (Log Scale)")
    axs[0, 1].set_yscale('log')
    axs[1, 0].set_title("Policy Entropy (Exploration Decay)")
    
    axs[0, 0].legend(); axs[0, 1].legend(); axs[1, 0].legend()
    plt.tight_layout()
    plt.savefig(f"{base_dir}/ablation_graphs.png")
    print(f"Results saved in {base_dir}/")

if __name__ == "__main__":
    run_ablation()