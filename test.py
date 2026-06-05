"""
HuggingFace Deep RL Course — Unit 1  |  Test / Evaluation
===========================================================
Loads the trained PPO model and evaluates it on LunarLander-v3.

Usage:
    # Evaluate local saved model
    python test.py

    # Evaluate a model downloaded from HF Hub
    python test.py --from_hub roky51/ppo-LunarLander-v3

    # Watch the agent live (opens a window)
    python test.py --render

    # Record a video
    python test.py --record
"""

import argparse
import os
import json
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

MODEL_PATH = "ppo-LunarLander-v3"   # local .zip (no extension needed)
ENV_ID     = "LunarLander-v3"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

# ══════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default=MODEL_PATH, help="Path to local .zip model")
    p.add_argument("--env",       default=ENV_ID)
    p.add_argument("--episodes",  type=int, default=10, help="Number of eval episodes")
    p.add_argument("--from_hub",  default="",  help="Load from HF Hub: user/repo-name")
    p.add_argument("--render",    action="store_true", help="Render live in a window")
    p.add_argument("--record",    action="store_true", help="Save a video to ./videos/")
    return p.parse_args()


def load_model(args: argparse.Namespace) -> PPO:
    if args.from_hub:
        from huggingface_sb3 import load_from_hub
        checkpoint = load_from_hub(
            repo_id    = args.from_hub,
            filename   = f"{args.from_hub.split('/')[-1]}.zip",
        )
        model = PPO.load(checkpoint)
        print(f"[Load] Model pulled from Hub: {args.from_hub}")
    else:
        model = PPO.load(args.model)
        print(f"[Load] Model loaded from: {args.model}.zip")
    return model


def evaluate(model: PPO, args: argparse.Namespace) -> dict:
    eval_env = Monitor(gym.make(args.env))

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes = args.episodes,
        deterministic   = True,
    )
    eval_env.close()

    score = mean_reward - std_reward
    print("\n" + "═" * 45)
    print(f"  Environment   : {args.env}")
    print(f"  Episodes      : {args.episodes}")
    print(f"  Mean reward   : {mean_reward:.2f}")
    print(f"  Std reward    : {std_reward:.2f}")
    print(f"  Cert score    : {score:.2f}  (need >= 200)")
    status = "PASS ✓" if score >= 200 else "not yet"
    print(f"  Certification : {status}")
    print("═" * 45 + "\n")

    return {"mean_reward": round(mean_reward, 2),
            "std_reward":  round(std_reward,  2),
            "cert_score":  round(score, 2),
            "certified":   score >= 200}


def render_live(model: PPO, args: argparse.Namespace, n_episodes: int = 3) -> None:
    """Watch the agent play in a pop-up window."""
    env = gym.make(args.env, render_mode="human")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"[Render] Episode {ep+1}: reward = {total_reward:.2f}")
    env.close()


def record_video(model: PPO, args: argparse.Namespace) -> None:
    """Record a short video and save to ./videos/."""
    video_folder = "./videos"
    os.makedirs(video_folder, exist_ok=True)

    vec_env = DummyVecEnv(
        [lambda: Monitor(gym.make(args.env, render_mode="rgb_array"))]
    )
    vec_env = VecVideoRecorder(
        vec_env,
        video_folder,
        record_video_trigger = lambda step: step == 0,
        video_length         = 1000,
        name_prefix          = f"ppo-{args.env}",
    )

    obs = vec_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env.step(action)
        if done[0]:
            obs = vec_env.reset()

    vec_env.close()
    print(f"[Record] Video saved → {video_folder}/")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args  = parse_args()
    model = load_model(args)

    metrics = evaluate(model, args)

    # Save metrics to file
    with open("test_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Test] Results saved → test_results.json")

    if args.render:
        render_live(model, args)

    if args.record:
        record_video(model, args)
