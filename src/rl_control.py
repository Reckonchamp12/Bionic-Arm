"""
Toy RL loop: a PPO policy learns to smooth/compensate noisy decoded commands.
This does NOT control real hardware; it's a pedagogical demo.

- The decoder emits one of {open, close, supinate, pronate, rest}.
- We map that to a continuous target (e.g., joint velocity).
- PPO learns to track the (noisy) target smoothly with minimal jerk.

"""
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from pathlib import Path
from models import TransformerDecoder, RNNDecoder

CLASS_TO_VEL = {
    0: -1.0,   # open
    1:  1.0,   # close
    2:  0.7,   # supinate
    3: -0.7,   # pronate
    4:  0.0,   # rest
}

class SmoothArmEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, max_steps=400):
        super().__init__()
        self.observation_space = spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)  # [pos, vel]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)       # acc command
        self.max_steps = max_steps
        self.reset(seed=123)

    def step(self, action):
        acc = float(np.clip(action[0], -1, 1))
        # simple physics
        self.vel = 0.95*self.vel + 0.05*acc
        self.pos = self.pos + self.vel
        self.t += 1
        # reward = -|vel - target_vel| - small jerk
        jerk = abs(acc - self.last_acc)
        self.last_acc = acc
        err = abs(self.vel - self.target_vel)
        reward = - (err + 0.05*jerk)
        terminated = False
        truncated = self.t >= self.max_steps
        obs = np.array([self.pos, self.vel], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = 0.0
        self.vel = 0.0
        self.last_acc = 0.0
        self.t = 0
        # will be set externally per-episode with decoder
        self.target_vel = 0.0
        return np.array([self.pos, self.vel], dtype=np.float32), {}

def load_decoder(ckpt_path):
    ckpt = Path(ckpt_path)
    if "transformer" in ckpt.name:
        model = TransformerDecoder(in_ch=4)
    else:
        model = RNNDecoder(in_ch=4)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state); model.eval()
    return model

def simulate_decoder_target(decoder, T=400):
    # Use class schedule and add noise to emulate decoding uncertainty
    rng = np.random.default_rng(7)
    classes = rng.integers(0, 5, T)
    # add a little stickiness to look realistic
    for i in range(1, T):
        if rng.random() < 0.8:
            classes[i] = classes[i-1]
    target = np.array([CLASS_TO_VEL[c] for c in classes], dtype=np.float32)
    target += 0.05 * rng.normal(0, 1, T)
    return target

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder_ckpt", default="results/transformer_best.pt")
    args = ap.parse_args()

    if not Path(args.decoder_ckpt).exists():
        print("Decoder checkpoint not found. Train first.")
        return

    decoder = load_decoder(args.decoder_ckpt)
    env = SmoothArmEnv(max_steps=400)
    model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=256, learning_rate=3e-4)
    target = simulate_decoder_target(decoder, T=env.max_steps)

    # train with per-step target injection
    rewards = []
    for it in range(10):  # 10 episodes for demo
        obs, _ = env.reset()
        env.target_vel = target[0]
        ep_rew = 0.0
        for t in range(env.max_steps):
            env.target_vel = target[t]
            action, _ = model.predict(obs, deterministic=False)
            obs, r, term, trunc, _ = env.step(action)
            ep_rew += r
            if term or trunc:
                break
        rewards.append(ep_rew)
        # short learning burst each episode
        model.learn(total_timesteps=env.max_steps, progress_bar=False, reset_num_timesteps=False)

    # plot rewards
    import matplotlib.pyplot as plt
    import os
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(rewards, marker="o")
    plt.title("RL Episode Reward (higher = smoother tracking)")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.savefig("results/rl_training_curve.png", dpi=150)
    print("Saved results/rl_training_curve.png")

if __name__ == "__main__":
    main()
