## Project Structure
```

aisf-gymnasium/
├── ablation_study/                # Ablation study train vids, test vids, and graphs
│ 
├── ppo/                          # PPO algorithm implementations
│   ├── BipedalWalkerPPO.py      # Basic PPO implementation
│   ├── ppo_hyperparameter.py    # Hyperparameter tuning experiments
│   └── ppo_reward_crafting.py   # Reward shaping experiments
│
├── sac/                          # SAC algorithm implementations
│   ├── sac.py                    # Basic SAC implementation
│   ├── normalized_sac.py         # SAC with observation normalization
│   ├── sac_logs/                 # SAC training logs
│   ├── sac_test_videos/          # sac.py test vids
│   ├── sac_improved_videos/      # normalized_sac.py test vids
│   └── sac_results.txt           # SAC experiment results
│
├── paper1/                       # Research paper 1 experiments
│   ├── optimized_ppo.py          # Optimized PPO implementation
│   ├── optimized_ppo_2.py        # Second version of optimized PPO (best performance)
│   ├── ablation_study.py         # Ablation study implementation
│   ├── optimized_ppo_results.txt # Results from optimized PPO
│   ├── ablation_results.txt      # Ablation study reward results
│
├── setup/                       
│   └── blackjack.py              # Blackjack environment setup (ignore)
│
├── logs/                         # Training logs and evaluation videos
│   ├── basic_PPO/                # Basic PPO experiment logs
│   ├── ppo_ablation_optimized/   # Optimized ablation study videos
│   ├── ppo_hyperparameter/       # Hyperparameter tuning results
│   ├── ppo_lr_annealing/        # Learning rate annealing experiments
│   ├── ppo_reward_crafting/      # Reward crafting experiments
│   ├── ppo_v3_refined/           # Refined PPO v3 experiments (ignore)
│   └── ppo_v4/                    # PPO v4 experiments (ignore)
│
├── models_and_stats/             # Saved models and training statistics│

```
## Write Up
Link: https://docs.google.com/document/d/1JI9Ye1AJBlLBe0nOMo7xtEMwxvq-Opsuc7mdksRVxgE/edit?usp=sharing

## Breakdown
Baseline Proximal Policy Optimization (PPO) and Initial Observations:
- Code: ppo/BipedalWalkerPPO.py
- Videos: logs/basic_PPO

Learning Rate Experiment 
- Code: ppo/ppo_hyperparameter.py
- Videos: logs/ppo_hyperparameter

Reward Crafting 
- Code: ppo/ppo_reward_crafting.py
- Videos: logs/ppo_reward_crafting/100k AND logs/ppo_reward_crafting/150k

Literature Guided Improvement and Ablation Study
- Code: paper1/ablation_study.py
- Results: ablation_results.txt
- Videos: ablation_study/…
- Graph: ablation_study/ablation_graphs.png

Alternate Algorithms and Future Directions
- Code: sac/sac.py and sac/normalized_sac.py
- Videos: sac/test_videos (for sac.py) and sac/improved_videos (for normalized_sac.py)
- Results: sac/sac_results.txt

## Environment

- **Primary Environment**: BipedalWalker-v3 (Gymnasium)
- **Framework**: Stable Baselines3
- **Language**: Python
