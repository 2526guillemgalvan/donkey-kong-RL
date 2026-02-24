"""
Entrenamiento continuo de Donkey Kong con PPO.

Carga el checkpoint guardado por donkey_kong_ppo.ipynb y continua
entrenando, mostrando el progreso visual periodicamente.

Uso:
    python train_donkey_kong.py                        # 100 updates mas
    python train_donkey_kong.py --updates 200          # 200 updates mas
    python train_donkey_kong.py --no-render             # sin ventana visual
    python train_donkey_kong.py --height-reward          # bonus por subir de altura
    python train_donkey_kong.py --eval-interval 10      # evaluar cada 10 updates
"""

import argparse
import sys
import time
from collections import deque

import cv2
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "donkey_kong_ppo.pth"

# ── Estos valores por defecto se sobreescriben con los del checkpoint ──
FRAME_STACK = 4
FRAME_SIZE = 84


# ---------------------------------------------------------------------------
# Wrapper y Red (identicos al notebook donkey_kong_ppo.ipynb)
# ---------------------------------------------------------------------------

class DonkeyKongWrapper(gym.Wrapper):
    """Preprocessing Atari para Donkey Kong:
       - Escala de grises + redimensionar a 84x84
       - Apilar 4 frames consecutivos
       - Recortar recompensas a [-1, 1]
    """
    def __init__(self, env, frame_stack=4, frame_size=84):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frames = deque(maxlen=frame_stack)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(frame_stack, frame_size, frame_size),
            dtype=np.uint8,
        )
        # Tracking de episodio (recompensa total real, sin shaping)
        self.episode_reward = 0.0
        self.episode_steps = 0

    def _preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            gray, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA
        )
        return resized

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._preprocess(obs)
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        self.episode_reward = 0.0
        self.episode_steps = 0
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._preprocess(obs)
        self.frames.append(frame)
        stacked = np.stack(self.frames, axis=0)
        self.episode_reward += reward
        self.episode_steps += 1
        reward = np.clip(reward, -1.0, 1.0)
        if terminated or truncated:
            info["episode"] = {"r": self.episode_reward, "l": self.episode_steps}
        return stacked, reward, terminated, truncated, info


class DonkeyKongHeightWrapper(gym.Wrapper):
    """Wrapper adicional que da reward bonus por subir de altura.

    Detecta la posicion Y de Mario en el frame RGB (pixeles rojos)
    y da un bonus cuando sube (Y decrece en coordenadas de pantalla).

    Se aplica ENCIMA del DonkeyKongWrapper base para no romper
    la compatibilidad del checkpoint (misma observation space).
    """
    HEIGHT_BONUS = 0.005  # bonus por pixel de altura ganada

    def __init__(self, env):
        super().__init__(env)
        self.prev_mario_y = None

    def _find_mario_y(self, raw_obs):
        """Encuentra la Y media de Mario en el frame RGB."""
        r, g, b = raw_obs[:, :, 0], raw_obs[:, :, 1], raw_obs[:, :, 2]
        mario_mask = (r > 150) & (g < 80) & (b < 80)
        ys = np.where(mario_mask)[0]
        if len(ys) > 5:
            return float(np.mean(ys))
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        raw = self.env.unwrapped.ale.getScreenRGB()
        self.prev_mario_y = self._find_mario_y(raw)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            raw = self.env.unwrapped.ale.getScreenRGB()
            mario_y = self._find_mario_y(raw)
            if mario_y is not None and self.prev_mario_y is not None:
                delta_y = self.prev_mario_y - mario_y  # positivo = subir
                if delta_y > 2:  # umbral para evitar ruido
                    reward += delta_y * self.HEIGHT_BONUS
            if mario_y is not None:
                self.prev_mario_y = mario_y
        else:
            self.prev_mario_y = None
        return obs, reward, terminated, truncated, info


def make_env(env_id, seed=0, render_mode=None, frame_stack=4, frame_size=84,
             height_reward=False):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = DonkeyKongWrapper(env, frame_stack, frame_size)
        if height_reward:
            env = DonkeyKongHeightWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPONetwork(nn.Module):
    """CNN compartida con cabezas Actor y Critic separadas."""
    def __init__(self, in_channels, n_actions, frame_size=84):
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, frame_size, frame_size)
            cnn_out = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            layer_init(nn.Linear(cnn_out, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        x = self.cnn(x.float() / 255.0)
        x = self.fc(x)
        return self.actor(x), self.critic(x)

    def get_value(self, x):
        _, value = self.forward(x)
        return value

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ---------------------------------------------------------------------------
# Cargar checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    """Carga un checkpoint y reconstruye modelo + optimizer."""
    print(f"Cargando checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hp = checkpoint["hyperparams"]

    model = PPONetwork(hp["FRAME_STACK"], hp["n_actions"], hp["FRAME_SIZE"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    optimizer = optim.Adam(model.parameters(), lr=hp["LR"], eps=1e-5)
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(f"  Update:       {checkpoint['update']}")
    print(f"  Global step:  {checkpoint['global_step']:,}")
    print(f"  Episodios:    {len(checkpoint['episode_rewards'])}")
    if checkpoint["episode_rewards"]:
        print(f"  Mejor reward: {max(checkpoint['episode_rewards']):.1f}")
        last_20 = checkpoint["episode_rewards"][-20:]
        print(f"  Media ult.20: {np.mean(last_20):.2f}")

    return model, optimizer, checkpoint


# ---------------------------------------------------------------------------
# Evaluacion visual
# ---------------------------------------------------------------------------

def evaluate_visual(model, env_id, frame_stack=4, frame_size=84, n_episodes=1):
    """Ejecuta episodios con render_mode='human' para ver el progreso."""
    env = gym.make(env_id, render_mode="human")
    env = DonkeyKongWrapper(env, frame_stack, frame_size)
    model.eval()
    scores = []

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total = 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.uint8).unsqueeze(0).to(device)
                action, _, _, _ = model.get_action_and_value(obs_t)
                obs, reward, terminated, truncated, _ = env.step(action.item())
                total += reward
                done = terminated or truncated
            scores.append(total)
            print(f"    Eval episodio {ep + 1}: score = {total:.1f}")

    env.close()
    model.train()
    return scores


# ---------------------------------------------------------------------------
# Entrenamiento continuo
# ---------------------------------------------------------------------------

def train(args):
    # Cargar checkpoint
    model, optimizer, checkpoint = load_checkpoint(args.checkpoint)
    hp = checkpoint["hyperparams"]

    # Restaurar estado
    start_update = checkpoint["update"]
    global_step = checkpoint["global_step"]
    episode_rewards = checkpoint["episode_rewards"]
    update_losses = checkpoint.get("update_losses", [])
    prev_total_updates = checkpoint.get("total_updates", start_update)
    total_updates = prev_total_updates + args.updates

    # Hiperparametros del checkpoint
    env_id = hp["ENV_ID"]
    num_envs = hp["NUM_ENVS"]
    frame_stack = hp["FRAME_STACK"]
    frame_size = hp["FRAME_SIZE"]
    rollout_steps = hp["ROLLOUT_STEPS"]
    num_epochs = hp["NUM_EPOCHS"]
    minibatch_size = hp["MINIBATCH_SIZE"]
    gamma = hp["GAMMA"]
    gae_lambda = hp["GAE_LAMBDA"]
    clip_epsilon = hp["CLIP_EPSILON"]
    entropy_coef = hp["ENTROPY_COEF"]
    value_coef = hp["VALUE_COEF"]
    max_grad_norm = hp["MAX_GRAD_NORM"]
    base_lr = hp["LR"]
    lr_anneal = hp["LR_ANNEAL"]
    n_actions = hp["n_actions"]
    batch_size = num_envs * rollout_steps

    # Crear entornos (sin render para velocidad, con reward por altura)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed=i, frame_stack=frame_stack, frame_size=frame_size,
                  height_reward=args.height_reward)
         for i in range(num_envs)]
    )
    obs_shape = envs.single_observation_space.shape

    # Buffers
    obs_buf = torch.zeros((rollout_steps, num_envs) + obs_shape, dtype=torch.uint8).to(device)
    actions_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.long).to(device)
    logprobs_buf = torch.zeros((rollout_steps, num_envs)).to(device)
    rewards_buf = torch.zeros((rollout_steps, num_envs)).to(device)
    dones_buf = torch.zeros((rollout_steps, num_envs)).to(device)
    values_buf = torch.zeros((rollout_steps, num_envs)).to(device)

    # Reset
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.uint8).to(device)
    next_done = torch.zeros(num_envs).to(device)

    print(f"\nContinuando entrenamiento: {args.updates} updates mas")
    print(f"  Updates previos:  {start_update}")
    print(f"  Updates objetivo: {start_update + args.updates}")
    print(f"  Eval visual cada: {args.eval_interval} updates")
    print(f"  Render:           {'Si' if args.render else 'No'}")
    print(f"  Height reward:    {'Si' if args.height_reward else 'No'}")
    print()

    start_time = time.time()
    session_rewards = []

    for update_i in range(1, args.updates + 1):
        current_update = start_update + update_i

        # LR annealing
        if lr_anneal:
            frac = 1.0 - (current_update - 1) / total_updates
            frac = max(frac, 0.0)
            for param_group in optimizer.param_groups:
                param_group["lr"] = frac * base_lr

        # -- Rollout --
        for step in range(rollout_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs)
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value.flatten()

            next_obs_np, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            rewards_buf[step] = torch.tensor(reward).to(device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.uint8).to(device)
            next_done = torch.tensor(done, dtype=torch.float32).to(device)

            # Tracking de episodios terminados (gymnasium 1.x format)
            if "_episode" in info:
                for i, done_flag in enumerate(info["_episode"]):
                    if done_flag:
                        ep_rew = float(info["episode"]["r"][i])
                        episode_rewards.append(ep_rew)
                        session_rewards.append(ep_rew)

        # -- GAE --
        with torch.no_grad():
            next_value = model.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards_buf).to(device)
            last_gae = 0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalue = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalue = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalue * nextnonterminal - values_buf[t]
                last_gae = delta + gamma * gae_lambda * nextnonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + values_buf

        # -- Flatten --
        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # -- PPO update --
        indices = np.arange(batch_size)
        clip_fracs = []
        epoch_losses = []

        for epoch in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]

                _, new_logprob, entropy, new_value = model.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )

                log_ratio = new_logprob - b_logprobs[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    clip_fracs.append(
                        ((ratio - 1).abs() > clip_epsilon).float().mean().item()
                    )

                mb_adv = b_advantages[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - clip_epsilon, 1 + clip_epsilon
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.flatten()
                v_clipped = b_values[mb_idx] + torch.clamp(
                    new_value - b_values[mb_idx], -clip_epsilon, clip_epsilon
                )
                vf_loss = 0.5 * torch.max(
                    (new_value - b_returns[mb_idx]).pow(2),
                    (v_clipped - b_returns[mb_idx]).pow(2),
                ).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + value_coef * vf_loss - entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                epoch_losses.append(loss.item())

        update_losses.append(np.mean(epoch_losses))

        # -- Logging --
        if update_i % 5 == 0 or update_i == args.updates:
            elapsed = time.time() - start_time
            sps = int((update_i * batch_size) / elapsed) if elapsed > 0 else 0
            mean_rew = np.mean(session_rewards[-20:]) if session_rewards else 0.0
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Update {current_update:>5} ({update_i}/{args.updates})  "
                f"Steps {global_step:>10,}  "
                f"SPS {sps:>5,}  "
                f"MeanRew {mean_rew:>7.2f}  "
                f"Loss {update_losses[-1]:.4f}  "
                f"LR {cur_lr:.2e}"
            )

        # -- Evaluacion visual --
        if args.render and update_i % args.eval_interval == 0:
            print(f"\n  --- Evaluacion visual (update {current_update}) ---")
            evaluate_visual(model, env_id, frame_stack, frame_size)
            print()

    envs.close()

    # -- Guardar checkpoint actualizado --
    final_update = start_update + args.updates
    torch.save(
        {
            "update": final_update,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "episode_rewards": episode_rewards,
            "update_losses": update_losses,
            "total_updates": total_updates,
            "hyperparams": hp,
        },
        args.checkpoint,
    )

    print(f"\nCheckpoint actualizado: {args.checkpoint}")
    print(f"  Update total: {final_update}")
    print(f"  Global steps: {global_step:,}")
    print(f"  Episodios totales: {len(episode_rewards)}")
    if session_rewards:
        print(f"  Reward medio (esta sesion): {np.mean(session_rewards):.2f}")
        print(f"  Mejor reward (esta sesion): {max(session_rewards):.1f}")
    if episode_rewards:
        print(f"  Mejor reward (total): {max(episode_rewards):.1f}")

    # Evaluacion final visual
    if args.render:
        print("\n--- Evaluacion final ---")
        evaluate_visual(model, env_id, frame_stack, frame_size)


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento continuo de Donkey Kong con PPO"
    )
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH, help="Ruta al checkpoint (default: donkey_kong_ppo.pth)"
    )
    parser.add_argument(
        "--updates", type=int, default=100, help="Updates adicionales de entrenamiento (default: 100)"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=25, help="Evaluar visualmente cada N updates (default: 25)"
    )
    parser.add_argument(
        "--no-render", dest="render", action="store_false",
        help="Desactivar evaluacion visual"
    )
    parser.add_argument(
        "--height-reward", action="store_true",
        help="Activar reward bonus por subir de altura"
    )
    parser.set_defaults(render=True)
    args = parser.parse_args()

    print("=" * 60)
    print("  DONKEY KONG PPO - Entrenamiento Continuo")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train(args)


if __name__ == "__main__":
    main()
