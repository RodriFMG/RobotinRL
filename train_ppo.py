"""
train_ppo.py - entrena la roombita con PPO (Stable-Baselines3).

  # estado interno (rapido, sin render)
  python train_ppo.py --track=simple_line --obs_mode=state --total_timesteps=100000

  # mascara directa del entorno / RGB crudo
  python train_ppo.py --track=simple_line --obs_mode=mask --total_timesteps=100000
  python train_ppo.py --track=simple_line --obs_mode=rgb  --total_timesteps=100000

  # tu segmentador .pth congelado:  RGB -> seg.pth -> mascara -> PPO
  python train_ppo.py --track=oval --obs_mode=seg --seg_model=seg_dice.pth --total_timesteps=100000

  # reanudar desde una politica ya guardada (NO entrena desde 0)
  python train_ppo.py --track=oval --obs_mode=seg --seg_model=seg_dice.pth --model=models/policy_model.zip

Politica segun obs_mode:
  state -> MlpPolicy
  mask/rgb/seg -> CnnPolicy

Guarda la politica en models/policy_model.zip
Al terminar guarda los plots de entrenamiento en plots/<fecha>/
"""

import os
import argparse
import warnings
import datetime
import json
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


# ---------------------------------------------------------------------
# Wrapper: RGB del entorno -> segmentacion predicha por tu modelo .pth
# ---------------------------------------------------------------------
class SegmenterObsWrapper(gym.ObservationWrapper):
    """RoombitaEnv(obs_mode=rgb) -> VisionSegmenter(seg.pth) -> mascara 84x84x3 -> PPO."""

    def __init__(self, env, seg_model_path, cam_w=84, cam_h=84, seg_device="auto"):
        super().__init__(env)
        from vision_segmenter import VisionSegmenter

        self.seg_model_path = seg_model_path
        self.cam_w = cam_w
        self.cam_h = cam_h

        if not os.path.exists(seg_model_path):
            raise FileNotFoundError(f"No existe el modelo de segmentacion: {seg_model_path}")

        try:
            self.segmenter = VisionSegmenter(seg_model_path, out_size=(cam_h, cam_w), device=seg_device)
        except TypeError:
            self.segmenter = VisionSegmenter(seg_model_path, out_size=(cam_h, cam_w))

        self.observation_space = spaces.Box(low=0, high=255, shape=(cam_h, cam_w, 3), dtype=np.uint8)
        print(f"[seg] SegmenterObsWrapper activo | modelo: {seg_model_path}")

    def observation(self, obs):
        rgb = np.asarray(obs)
        if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        seg = np.asarray(self.segmenter.predict_seg_map(rgb))
        if seg.ndim == 2:
            seg = np.stack([seg, seg, seg], axis=-1)
        if seg.ndim == 3 and seg.shape[0] == 3 and seg.shape[-1] != 3:
            seg = np.transpose(seg, (1, 2, 0))
        if seg.dtype != np.uint8:
            if seg.max() <= 1.0:
                seg = seg * 255.0
            seg = np.clip(seg, 0, 255).astype(np.uint8)
        return seg


# ---------------------------------------------------------------------
# Callback: captura los factores de entrenamiento de SB3 (la tabla)
# ---------------------------------------------------------------------
def _make_history_callback():
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import KVWriter

    class TrainingHistoryCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.records = []

        def _on_training_start(self) -> None:
            recs = self.records

            class _Grab(KVWriter):
                def write(self, key_values, key_excluded, step=0):
                    rec = {k: float(v) for k, v in key_values.items()
                           if isinstance(v, (int, float, np.integer, np.floating))}
                    if rec:
                        rec["_step"] = step
                        recs.append(rec)

                def close(self):
                    pass

            self.model.logger.output_formats.append(_Grab())

        def _on_step(self) -> bool:
            return True

    return TrainingHistoryCallback()


# ---------------------------------------------------------------------
# Plot de los factores principales de entrenamiento
# ---------------------------------------------------------------------
def plot_training(records, out_dir, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        print("[plot] sin registros para graficar")
        return None

    xkey = "time/total_timesteps"
    xs = [r.get(xkey, r.get("_step", i)) for i, r in enumerate(records)]

    metrics = [
        "rollout/ep_rew_mean", "rollout/ep_len_mean", "time/fps",
        "train/approx_kl", "train/clip_fraction", "train/entropy_loss",
        "train/explained_variance", "train/loss", "train/policy_gradient_loss",
        "train/value_loss", "train/std", "train/learning_rate",
    ]
    metrics = [m for m in metrics if any(m in r for r in records)]

    cols = 3
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.7))
    axes = np.array(axes).reshape(-1)

    for ax, m in zip(axes, metrics):
        ys = [r.get(m, np.nan) for r in records]
        ax.plot(xs, ys, lw=1.6)
        ax.set_title(m, fontsize=9)
        ax.grid(alpha=.3)
        ax.tick_params(labelsize=7)
    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle(title or "PPO training", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%H-%M-%S")
    png = os.path.join(out_dir, f"training_{stamp}.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)

    with open(os.path.join(out_dir, f"history_{stamp}.json"), "w") as f:
        json.dump(records, f, indent=2)

    print(f"[plot] guardado: {png}")
    return png


# ---------------------------------------------------------------------
# Crear env
# ---------------------------------------------------------------------
def make_env_fn(args, seed):
    from roombita_gym_env import RoombitaEnv

    def _f():
        env_obs_mode = "rgb" if args.obs_mode == "seg" else args.obs_mode
        cam_w = args.seg_res if args.obs_mode == "seg" else args.cam_w
        cam_h = args.seg_res if args.obs_mode == "seg" else args.cam_h

        env = RoombitaEnv(
            track=args.track, obs_mode=env_obs_mode,
            obstacle_slots=args.obstacle_slots, obstacle_prob=args.obstacle_prob,
            obstacle_count_mode=args.obstacle_count_mode, time_max=args.time_max,
            random_brightness=args.random_brightness,
            cam_w=cam_w, cam_h=cam_h, seed=seed,
        )

        if args.obs_mode == "seg":
            env = SegmenterObsWrapper(env, seg_model_path=args.seg_model,
                                      cam_w=args.cam_w, cam_h=args.cam_h, seg_device=args.seg_device)
        from stable_baselines3.common.monitor import Monitor
        return Monitor(env)

    return _f


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Entorno
    ap.add_argument("--track", type=str, default="simple_line")
    ap.add_argument("--obs_mode", type=str, default="state", choices=["state", "mask", "rgb", "seg"])
    ap.add_argument("--n_envs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    # Segmentador
    ap.add_argument("--seg_model", type=str, default="seg_dice.pth")
    ap.add_argument("--seg_device", type=str, default="cuda")
    # Camara
    ap.add_argument("--cam_w", type=int, default=84)
    ap.add_argument("--cam_h", type=int, default=84)
    ap.add_argument("--seg_res", type=int, default=256,
                    help="resolucion ALTA a la que se renderiza el RGB y segmenta; la mascara sale a cam_w/cam_h para PPO")
    # Obstaculos / episodio
    ap.add_argument("--obstacle_slots", type=int, default=4)
    ap.add_argument("--obstacle_prob", type=float, default=0.6)
    ap.add_argument("--obstacle_count_mode", type=str, default="probabilistic")
    ap.add_argument("--time_max", type=float, default=45.0)
    ap.add_argument("--random_brightness", action="store_true")
    # PPO
    ap.add_argument("--total_timesteps", type=int, default=100000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--ent_coef", type=float, default=0.005)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--rollout_steps", type=int, default=2048)
    # Guardado / reanudar
    ap.add_argument("--save_path", type=str, default=os.path.join("models", "policy_model.zip"),
                    help="donde se guarda la politica entrenada")
    ap.add_argument("--model", type=str, default="",
                    help="ruta a una politica .zip existente: si se pasa, NO entrena desde 0, sigue desde ahi")
    ap.add_argument("--plots_dir", type=str, default="plots",
                    help="directorio raiz de plots; se crea un subdir con la fecha")

    args = ap.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError:
        raise SystemExit("Falta Stable-Baselines3:  pip install stable-baselines3")

    if args.obs_mode == "seg":
        if not args.seg_model or not os.path.exists(args.seg_model):
            raise FileNotFoundError(f"Con --obs_mode=seg necesitas un --seg_model valido (no existe: {args.seg_model})")
        if args.n_envs > 1:
            warnings.warn("obs_mode=seg con n_envs>1: cada proceso carga su propia copia del segmentador.")

    VecEnv = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecEnv([make_env_fn(args, args.seed + i) for i in range(args.n_envs)])

    policy = "MlpPolicy" if args.obs_mode == "state" else "CnnPolicy"
    n_steps = max(64, args.rollout_steps // max(args.n_envs, 1))

    out = args.save_path
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # --- cargar politica existente o crear nueva ---
    load_from = args.model if (args.model and os.path.exists(args.model)) else None
    if args.model and not load_from:
        warnings.warn(f"--model={args.model} no existe; se entrena desde 0.")

    if load_from:
        print(f"[resume] cargando politica desde: {load_from}")
        model = PPO.load(load_from, env=venv, device=args.device)
        print("[resume] politica cargada, continuo entrenamiento")
    else:
        print("[new] creando PPO nuevo")
        model = PPO(policy, venv, verbose=1, seed=args.seed, device=args.device,
                    n_steps=n_steps, batch_size=args.batch_size, gae_lambda=args.gae_lambda,
                    gamma=args.gamma, ent_coef=args.ent_coef, learning_rate=args.learning_rate)

    print(f"[train] track={args.track} | obs_mode={args.obs_mode} | policy={policy} | "
          f"n_envs={args.n_envs} | n_steps={n_steps} | total_timesteps={args.total_timesteps}")
    if args.obs_mode == "seg":
        print(f"[train] flujo visual: RGB -> {args.seg_model} -> mascara -> PPO")

    hist_cb = _make_history_callback()
    try:
        model.learn(total_timesteps=args.total_timesteps, progress_bar=True,
                    reset_num_timesteps=load_from is None, callback=hist_cb)
    except ImportError:
        # falta tqdm/rich para la barra: entrena igual sin progress_bar
        model.learn(total_timesteps=args.total_timesteps, progress_bar=False,
                    reset_num_timesteps=load_from is None, callback=hist_cb)

    model.save(out)
    venv.close()
    print(f"[ok] politica guardada en {out}")

    # --- plots al finalizar el training ---
    date_dir = os.path.join(args.plots_dir, datetime.datetime.now().strftime("%Y-%m-%d"))
    plot_training(hist_cb.records, date_dir,
                  title=f"PPO  {args.track}  obs={args.obs_mode}  steps={args.total_timesteps}")


if __name__ == "__main__":
    main()