#modified ppo w/ reward crafting
# Run: ./.venv/bin/python ppo/ppo_reward_crafting.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os

class BipedalRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        obs = self.env.unwrapped.hull.angle #obs[0] = hull angle, obs[2] = horizontal velocity
        vel_x = self.env.unwrapped.hull.linearVelocity[0]

        stability_penalty = -0.1 * abs(obs)
        forward_bonus = 0.05 * max(0, vel_x)
        return reward + stability_penalty + forward_bonus

#make sure you get the crafted reward and the official score of the orginal enviroment
class PointTrackerCallback(BaseCallback):
    def __init__(self, check_freq_steps=400, verbose=1):
        super(PointTrackerCallback, self).__init__(verbose)
        self.check_freq_steps = check_freq_steps
        self.crafted_reward_sum = 0
        self.episode_steps = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.crafted_reward_sum += reward
        self.episode_steps += 1

        if self.episode_steps % self.check_freq_steps == 0:
            print(f"-> Step {self.episode_steps}: Crafted Points = {self.crafted_reward_sum:.2f}")

        # When the episode ends, pull the OFFICIAL score from the Monitor info
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            if 'episode' in info.keys():
                official_score = info['episode']['r']
                official_length = info['episode']['l']
                print("\n" + "="*40)
                print(f"EPISODE FINISHED")
                print(f"Official Game Score: {official_score:.2f}  <-- (Target: 300)")
                print(f"Crafted AI Score:   {self.crafted_reward_sum:.2f}")
                print(f"Steps Taken:        {official_length}")
                print("="*40 + "\n")
            
            self.crafted_reward_sum = 0
            self.episode_steps = 0
        return True

video_folder = "logs/ppo_reward_crafting/250"
video_length = 500
os.makedirs(video_folder, exist_ok=True)

def make_env():
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    env = BipedalRewardWrapper(env)
    return Monitor(env)

env = DummyVecEnv([make_env])

env = VecVideoRecorder(
    env, 
    video_folder,
    record_video_trigger=lambda x: x % 10000 == 0, 
    video_length=video_length,
    name_prefix="ppo-bipedal-walker"
)

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=0.0003, 
    n_steps=2048,
    batch_size=64, 
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
)

print("Starting training with Reward Crafting...")
model.learn(total_timesteps=150000, callback=PointTrackerCallback())

model.save("ppo_bipedal_walker_model")
env.close()

#Testing
print("\n" + "="*50)
print("TESTING PHASE")
print("="*50)

model = PPO.load("ppo_bipedal_walker_model") 

test_episodes = 5
all_test_scores = []

for i in range(test_episodes):
    obs = env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if done:
            final_info = info[0].get('episode')
            if final_info:
                score = final_info['r']
                length = final_info['l']
                all_test_scores.append(score)
                print(f"Test Episode {i+1}: Official Score = {score:.2f} | Steps = {length}")

if all_test_scores:
    avg_score = sum(all_test_scores) / len(all_test_scores)
    print("-" * 30)
    print(f"AVERAGE TEST SCORE: {avg_score:.2f}")
    if avg_score >= 300:
        print("RESULT: ENVIRONMENT SOLVED!")
    else:
        print(f"RESULT: {300 - avg_score:.2f} points away from solving.")
print("=" * 50)