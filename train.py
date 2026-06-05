"""
HuggingFace Deep RL Course — Unit 1
=====================================
Train PPO on LunarLander-v2, evaluate, and upload to HF Hub.
Matches the official course notebook exactly (local Windows adaptation).

Install (run once in your .venv terminal):
    pip install swig
    pip install stable-baselines3[extra] gymnasium[box2d] huggingface-sb3 shimmy

Certification requirement: mean_reward - std_reward >= 200
Leaderboard: https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import package_to_hub

# ══════════════════════════════════════════════════════════════════
# CONFIG — set your token here or via:  $env:HF_TOKEN="hf_..."
# ══════════════════════════════════════════════════════════════════

HF_TOKEN = os.environ.get("HF_TOKEN", "")   # or paste your token: "hf_xxxxx"

env_id             = "LunarLander-v3"
model_architecture = "PPO"
model_name         = "ppo-LunarLander-v3"
repo_id            = "roky51/ppo-LunarLander-v3"
commit_message     = "Upload PPO LunarLander-v2 trained agent"

# ══════════════════════════════════════════════════════════════════
# STEP 1 — TRAIN
# ══════════════════════════════════════════════════════════════════

# 16 parallel environments — matches the course notebook exactly
env = make_vec_env(env_id, n_envs=16)

model = PPO(
    policy     = "MlpPolicy",
    env        = env,
    n_steps    = 1024,
    batch_size = 64,
    n_epochs   = 4,
    gamma      = 0.999,
    gae_lambda = 0.98,
    ent_coef   = 0.01,
    verbose    = 1,
)

model.learn(total_timesteps=1_000_000)
model.save(model_name)
print(f"\n[Train] Model saved -> {model_name}.zip")

env.close()

# ══════════════════════════════════════════════════════════════════
# STEP 2 — EVALUATE
# ══════════════════════════════════════════════════════════════════

eval_env = Monitor(gym.make(env_id))

mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes = 10,
    deterministic   = True,
)
print(f"\n[Eval] mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
print(f"[Eval] Certification score = {mean_reward - std_reward:.2f}  (need >= 200)")

eval_env.close()

# ══════════════════════════════════════════════════════════════════
# STEP 3 — UPLOAD TO HF HUB
# ══════════════════════════════════════════════════════════════════

if not HF_TOKEN:
    print("\n[Upload] Set HF_TOKEN to upload. Skipping.")
else:
    # DummyVecEnv with rgb_array render mode is required by package_to_hub
    upload_env = DummyVecEnv(
        [lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))]
    )

    package_to_hub(
        model              = model,
        model_name         = model_name,
        model_architecture = model_architecture,
        env_id             = env_id,
        eval_env           = upload_env,
        repo_id            = repo_id,
        commit_message     = commit_message,
        token              = HF_TOKEN,
    )
    print(f"\n[Upload] Done -> https://huggingface.co/{repo_id}")
