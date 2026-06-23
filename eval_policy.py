"""
eval_policy.py - carga un modelo PPO y lo corre en el entorno.

  python eval_policy.py --model models/simple_line_state.zip --track=simple_line --obs_mode=state --episodes=5
  python eval_policy.py --model models/oval_mask.zip --track=oval --obs_mode=mask --episodes=3 --record=True

Con --record=True guarda clips .npz (rgb/seg/obstacle/reward/done) en
runs_eval/<fecha>/ para reproducir con view_clip.py.
"""
import os, json, argparse, datetime
import numpy as np


def str2bool(v):
    return str(v).lower() in ("1", "true", "t", "yes", "y", "si")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--track", type=str, default="simple_line")
    ap.add_argument("--obs_mode", type=str, default="state", choices=["state", "mask", "rgb"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--obstacle_slots", type=int, default=6)
    ap.add_argument("--obstacle_prob", type=float, default=0.6)
    ap.add_argument("--time_max", type=float, default=45.0)
    ap.add_argument("--record", type=str2bool, default=False)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise SystemExit("Falta Stable-Baselines3.  Instalalo con:  pip install stable-baselines3")
    from roombita_gym_env import RoombitaEnv

    env = RoombitaEnv(track=args.track, obs_mode=args.obs_mode, obstacle_slots=args.obstacle_slots,
                      obstacle_prob=args.obstacle_prob, time_max=args.time_max,
                      cam_w=84, cam_h=84, seed=args.seed,
                      render_mode="rgb_array" if args.record else None)
    model = PPO.load(args.model, device="auto")

    sess = None
    if args.record:
        from arena_env import Vision
        if env.vis is None:
            env.vis = Vision(env.m, 120, 160)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sess = os.path.join("runs_eval", stamp); os.makedirs(sess, exist_ok=True)

    for e in range(args.episodes):
        obs, _ = env.reset(seed=args.seed+e)
        done = False; R = 0.0
        rgb_l, seg_l, obs_l, rew_l, done_l = [], [], [], [], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action); R += r
            done = term or trunc
            if args.record:
                rgb = env.vis.rgb(env.d)
                rgb_l.append(rgb); seg_l.append(env.vis.seg_map(env.d))
                obs_l.append(env.vis.overlay(rgb, env.d)); rew_l.append(r); done_l.append(done)
        print(f"  ep {e}: outcome={info.get('outcome')} R={R:.2f} t={info.get('sim_time')}s")
        if args.record and rgb_l:
            fn = os.path.join(sess, f"episode_{e+1:04d}.npz")
            np.savez_compressed(fn, rgb=np.asarray(rgb_l, np.uint8), seg=np.asarray(seg_l, np.uint8),
                                obstacle=np.asarray(obs_l, np.uint8), reward=np.asarray(rew_l, np.float32),
                                done=np.asarray(done_l, bool),
                                info=json.dumps(info), track_name=args.track, config=json.dumps(vars(args)))
    if sess:
        print(f"[ok] clips de evaluacion en {sess}")
    env.close()


if __name__ == "__main__":
    main()
