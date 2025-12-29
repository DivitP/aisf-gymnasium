import gymnasium as gym
import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
# ./.venv/bin/python sac/normal_sac.py

ENV_ID = "BipedalWalker-v3"
TRAIN_STEPS = 100000
VIDEO_FOLDER = "sac_improved_videos"
MODEL_PATH = "sac_bipedal_final"
STATS_PATH = "vec_normalize_stats.pkl"

def train_agent():
    print(f"--- Starting Training: {ENV_ID} ---")
    
    def make_env():
        env = gym.make(ENV_ID)
        return Monitor(env)

    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = SAC(
        "MlpPolicy",
        venv,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto'
    )

    model.learn(total_timesteps=TRAIN_STEPS)
    
    model.save(MODEL_PATH)
    venv.save(STATS_PATH)
    print("Training complete. Model and Stats saved.")

def test_and_record():
    print(f"\n--- Starting Evaluation: {ENV_ID} ---")
    os.makedirs(VIDEO_FOLDER, exist_ok=True)

    def make_test_env():
        return Monitor(gym.make(ENV_ID, render_mode="rgb_array"))

    test_venv = DummyVecEnv([make_test_env])
    
    test_venv = VecNormalize.load(STATS_PATH, test_venv)
    test_venv.training = False
    test_venv.norm_reward = False

    test_venv = VecVideoRecorder(
        test_venv, 
        VIDEO_FOLDER,
        record_video_trigger=lambda x: x == 0, 
        video_length=2000,
        name_prefix="sac_final_eval"
    )

    model = SAC.load(MODEL_PATH)
    
    scores = []
    for ep in range(5):
        obs = test_venv.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_venv.step(action)
            
            if done:
                final_score = info[0]['episode']['r']
                scores.append(final_score)
                print(f"Episode {ep+1} | Score: {final_score:.2f}")

    test_venv.close()
    
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Average: {np.mean(scores):.2f}")
    print(f"Max:     {np.max(scores):.2f}")
    print(f"Videos:  {VIDEO_FOLDER}/")

if __name__ == "__main__":
    train_agent()
    test_and_record()