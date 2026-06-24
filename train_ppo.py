"""
train_ppo.py - entrena la roombita con PPO (Stable-Baselines3).

  python train_ppo.py --track=simple_line --total_timesteps=100000 --obs_mode=mask
  python train_ppo.py --track=oval --obs_mode=state --total_timesteps=200000 --n_envs=4

Politica segun obs_mode:
  state -> MlpPolicy  (rapido, sin render; recomendado para empezar)
  mask  -> CnnPolicy  (segmentacion 84x84x3)
  rgb   -> CnnPolicy  (camara 84x84x3)

Requiere: pip install stable-baselines3
Guarda el modelo en models/<track>_<obs_mode>.zip
"""
import os, argparse


def make_env_fn(args, seed):
    from roombita_gym_env import RoombitaEnv

    def _f():
        env = RoombitaEnv(track=args.track, obs_mode=args.obs_mode,
                          obstacle_slots=args.obstacle_slots, obstacle_prob=args.obstacle_prob,
                          obstacle_count_mode=args.obstacle_count_mode, time_max=args.time_max,
                          random_brightness=args.random_brightness, cam_w=84, cam_h=84, seed=seed)
        from stable_baselines3.common.monitor import Monitor
        return Monitor(env)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=str, default="simple_line")
    ap.add_argument("--total_timesteps", type=int, default=100000)
    ap.add_argument("--obs_mode", type=str, default="state", choices=["state", "mask", "rgb"])
    ap.add_argument("--n_envs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacle_slots", type=int, default=6)
    ap.add_argument("--obstacle_prob", type=float, default=0.6)
    ap.add_argument("--obstacle_count_mode", type=str, default="probabilistic")
    ap.add_argument("--time_max", type=float, default=45.0)
    ap.add_argument("--random_brightness", action="store_true")
    ap.add_argument("--save_path", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError:
        raise SystemExit("Falta Stable-Baselines3.  Instalalo con:  pip install stable-baselines3")

    VecEnv = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecEnv([make_env_fn(args, args.seed+i) for i in range(args.n_envs)])

    policy = "MlpPolicy" if args.obs_mode == "state" else "CnnPolicy"
    model = PPO(policy, venv, verbose=1, seed=args.seed, device=args.device,

                # a cada env parallel le da un trozo uniforme del num stepts
                n_steps=2048 // max(args.n_envs, 1), batch_size=256,
                gae_lambda=0.95, gamma=0.995, ent_coef=0.005, learning_rate=3e-4)
    
    print(f"[train] track={args.track} obs_mode={args.obs_mode} policy={policy} "
          f"n_envs={args.n_envs} steps={args.total_timesteps}")
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    out = args.save_path or os.path.join("models", f"{args.track}_{args.obs_mode}.zip")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model.save(out)
    venv.close()
    print(f"[ok] modelo guardado en {out}")


if __name__ == "__main__":
    main()
