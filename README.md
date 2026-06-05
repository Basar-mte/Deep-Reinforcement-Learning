# Deep Reinforcement Learning — Unit 1

Training a PPO agent on **LunarLander-v3** using Stable-Baselines3.  
Part of the [HuggingFace Deep RL Course](https://huggingface.co/deep-rl-course/unit1/introduction).

---

## Results

| Metric | Value |
|---|---|
| Algorithm | PPO (Proximal Policy Optimization) |
| Environment | LunarLander-v3 |
| Mean Reward | 256.64 |
| Std Reward | 13.70 |
| Certification Score | **242.95** ✓ (need ≥ 200) |

**Trained model on HF Hub:** [roky51/ppo-LunarLander-v3](https://huggingface.co/roky51/ppo-LunarLander-v3)

---

## Setup

```bash
# Requires Python 3.12
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install swig
pip install stable-baselines3 "gymnasium[box2d]" huggingface-sb3 shimmy moviepy
```

---

## Train

```bash
python train.py
```

Trains PPO for 1,000,000 timesteps with 16 parallel environments, then evaluates and uploads to HF Hub.

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Policy | MlpPolicy |
| n_steps | 1024 |
| batch_size | 64 |
| n_epochs | 4 |
| gamma | 0.999 |
| gae_lambda | 0.98 |
| ent_coef | 0.01 |

---

## Test

```bash
# Evaluate the local saved model (10 episodes)
python test.py

# Watch the agent play live
python test.py --render

# Record a video to ./videos/
python test.py --record

# Load directly from HF Hub
python test.py --from_hub roky51/ppo-LunarLander-v3
```

---

## Upload to HF Hub

Set your token and run:

```bash
# Windows PowerShell
$env:HF_TOKEN = "hf_your_token_here"
python train.py
```

The script automatically calls `package_to_hub()` after training, which uploads the model weights, a demo video, and a model card.

---

## Leaderboard

Submit your model at the [Deep RL Leaderboard](https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard) to get certified.
