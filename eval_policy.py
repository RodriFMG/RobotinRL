"""
eval_policy.py - carga una política PPO entrenada y la ejecuta en RoombitaEnv.

Ejemplos:

  # Evaluar política entrenada con estado interno
  python eval_policy.py --model models/policy_model.zip --track=simple_line --obs_mode=state --episodes=5 --show=true

  # Evaluar política entrenada con máscara directa del simulador
  python eval_policy.py --model models/policy_model.zip --track=simple_line --obs_mode=mask --episodes=5 --show=true

  # Evaluar política entrenada con RGB crudo
  python eval_policy.py --model models/policy_model.zip --track=simple_line --obs_mode=rgb --episodes=5 --show=true

  # Evaluar política entrenada con segmentador propio:
  # RGB del entorno -> seg_dice.pth -> máscara predicha -> PPO
  python eval_policy.py --model models/policy_model.zip --track=simple_line --obs_mode=seg --seg_model=seg_dice.pth --episodes=5 --show=true

  # Guardar clips de evaluación
  python eval_policy.py --model models/policy_model.zip --track=simple_line --obs_mode=seg --seg_model=seg_dice.pth --episodes=3 --record=true --show=true

Modos:

  state -> la política recibe estado interno.
  mask  -> la política recibe máscara directa del entorno.
  rgb   -> la política recibe RGB crudo.
  seg   -> el entorno entrega RGB, tu segmentador .pth genera la máscara, y PPO recibe esa máscara.

Con --show=true abre una ventana de OpenCV para ver la ejecución.
Con --record=true guarda clips .npz en runs_eval/<fecha>/.
"""

import os
import cv2
import json
import time
import argparse
import datetime
import numpy as np


def str2bool(v):
    return str(v).lower() in ("1", "true", "t", "yes", "y", "si", "sí")


def ensure_uint8_img(x):
    x = np.asarray(x)

    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))

    if x.dtype != np.uint8:
        if x.max() <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)

    return x


def resize_for_panel(img, h=240, w=320):
    img = ensure_uint8_img(img)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)


def put_title(img, title):
    out = img.copy()
    cv2.putText(
        out,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default="models/policy_model.zip",
                    help="Ruta al modelo PPO .zip entrenado.")
    ap.add_argument("--track", type=str, default="simple_line")
    ap.add_argument("--obs_mode", type=str, default="seg",
                    choices=["state", "mask", "rgb", "seg"])

    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--obstacle_slots", type=int, default=4)
    ap.add_argument("--obstacle_prob", type=float, default=0.6)
    ap.add_argument("--obstacle_count_mode", type=str, default="probabilistic")
    ap.add_argument("--time_max", type=float, default=45.0)

    ap.add_argument("--seg_model", type=str, default="seg_dice.pth",
                    help="Ruta al .pth del segmentador. Solo se usa con --obs_mode=seg.")
    ap.add_argument("--seg_device", type=str, default="cuda",
                    help="cpu/cuda. Si falla cuda, usa cpu.")

    ap.add_argument("--cam_w", type=int, default=84)
    ap.add_argument("--cam_h", type=int, default=84)

    ap.add_argument("--show", type=str2bool, default=True,
                    help="Muestra ventana OpenCV durante evaluación.")
    ap.add_argument("--record", type=str2bool, default=False,
                    help="Guarda clips .npz para ver luego.")
    ap.add_argument("--fps", type=float, default=20.0)

    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"No existe el modelo PPO: {args.model}")

    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise SystemExit("Falta Stable-Baselines3. Instálalo con: pip install stable-baselines3")

    from roombita_gym_env import RoombitaEnv

    # ---------------------------------------------------------
    # Segmentador: solo si obs_mode=seg
    # ---------------------------------------------------------
    segmenter = None

    if args.obs_mode == "seg":
        if not os.path.exists(args.seg_model):
            raise FileNotFoundError(f"No existe el segmentador: {args.seg_model}")

        from vision_segmenter import VisionSegmenter

        try:
            segmenter = VisionSegmenter(
                args.seg_model,
                out_size=(args.cam_h, args.cam_w),
                device=args.seg_device,
            )
        except TypeError:
            segmenter = VisionSegmenter(
                args.seg_model,
                out_size=(args.cam_h, args.cam_w),
            )

        env_obs_mode = "rgb"
        print(f"[seg] usando segmentador: {args.seg_model}")
        print("[seg] flujo: RGB del entorno -> segmentador .pth -> máscara -> PPO")

    else:
        env_obs_mode = args.obs_mode

    # ---------------------------------------------------------
    # Crear entorno
    # ---------------------------------------------------------
    env = RoombitaEnv(
        track=args.track,
        obs_mode=env_obs_mode,
        obstacle_slots=args.obstacle_slots,
        obstacle_prob=args.obstacle_prob,
        obstacle_count_mode=args.obstacle_count_mode,
        time_max=args.time_max,
        cam_w=args.cam_w,
        cam_h=args.cam_h,
        seed=args.seed,
        render_mode="rgb_array",
    )

    # Para grabar/mostrar visualmente
    from arena_env import Vision

    if getattr(env, "vis", None) is None:
        env.vis = Vision(env.m, 120, 160)

    # ---------------------------------------------------------
    # Cargar política PPO
    # ---------------------------------------------------------
    model = PPO.load(args.model, device="auto")

    print(f"[eval] modelo PPO: {args.model}")
    print(f"[eval] track={args.track} | obs_mode={args.obs_mode} | episodes={args.episodes}")

    def policy_obs(raw_obs):
        """
        Convierte la observación del entorno en la observación que espera PPO.
        """
        if segmenter is None:
            return raw_obs

        rgb = ensure_uint8_img(raw_obs)
        pred = segmenter.predict_seg_map(rgb)
        pred = ensure_uint8_img(pred)
        return pred

    # ---------------------------------------------------------
    # Carpeta de grabación
    # ---------------------------------------------------------
    sess = None
    if args.record:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sess = os.path.join("runs_eval", stamp)
        os.makedirs(sess, exist_ok=True)
        print(f"[record] guardando clips en: {sess}")

    # ---------------------------------------------------------
    # Evaluación
    # ---------------------------------------------------------
    all_rewards = []
    all_lengths = []
    all_outcomes = []

    delay = max(int(1000 / args.fps), 1)

    for e in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + e)

        done = False
        total_reward = 0.0
        step_count = 0

        rgb_l = []
        sim_seg_l = []
        pred_seg_l = []
        obstacle_l = []
        reward_l = []
        done_l = []

        info = {}

        while not done:
            obs_for_policy = policy_obs(obs)

            action, _ = model.predict(obs_for_policy, deterministic=True)

            obs, r, term, trunc, info = env.step(action)

            total_reward += float(r)
            step_count += 1
            done = bool(term or trunc)

            # Visual real del simulador
            rgb_view = env.vis.rgb(env.d)
            sim_seg_view = env.vis.seg_map(env.d)
            obstacle_view = env.vis.overlay(rgb_view, env.d)

            if segmenter is not None:
                pred_seg_view = segmenter.predict_seg_map(rgb_view)
            else:
                pred_seg_view = obs_for_policy

            pred_seg_view = ensure_uint8_img(pred_seg_view)

            # Mostrar ventana
            if args.show:
                p1 = put_title(resize_for_panel(rgb_view), "RGB")
                p2 = put_title(resize_for_panel(sim_seg_view), "SIM MASK")
                p3 = put_title(resize_for_panel(pred_seg_view), "POLICY INPUT")
                p4 = put_title(resize_for_panel(obstacle_view), "OVERLAY")

                top = np.hstack([p1, p2])
                bottom = np.hstack([p3, p4])
                canvas = np.vstack([top, bottom])

                text = f"ep={e+1}/{args.episodes} step={step_count} R={total_reward:.2f} r={r:.3f}"
                cv2.putText(
                    canvas,
                    text,
                    (10, canvas.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Roombita PPO Evaluation", canvas)

                key = cv2.waitKey(delay) & 0xFF
                if key in (ord("q"), 27):
                    print("[eval] evaluación interrumpida por usuario.")
                    done = True
                    break

            # Guardar frames
            if args.record:
                rgb_l.append(rgb_view)
                sim_seg_l.append(sim_seg_view)
                pred_seg_l.append(pred_seg_view)
                obstacle_l.append(obstacle_view)
                reward_l.append(float(r))
                done_l.append(done)

        outcome = info.get("outcome", "unknown")
        sim_time = info.get("sim_time", None)

        all_rewards.append(total_reward)
        all_lengths.append(step_count)
        all_outcomes.append(outcome)

        print(
            f"ep {e+1:03d}/{args.episodes} | "
            f"outcome={outcome} | "
            f"R={total_reward:.2f} | "
            f"steps={step_count} | "
            f"sim_time={sim_time}"
        )

        if args.record and rgb_l:
            fn = os.path.join(sess, f"episode_{e+1:04d}.npz")
            np.savez_compressed(
                fn,
                rgb=np.asarray(rgb_l, np.uint8),
                seg=np.asarray(sim_seg_l, np.uint8),
                pred_seg=np.asarray(pred_seg_l, np.uint8),
                obstacle=np.asarray(obstacle_l, np.uint8),
                reward=np.asarray(reward_l, np.float32),
                done=np.asarray(done_l, bool),
                info=json.dumps(info),
                track_name=args.track,
                config=json.dumps(vars(args)),
            )
            print(f"[record] clip guardado: {fn}")

    # ---------------------------------------------------------
    # Resumen final
    # ---------------------------------------------------------
    print("\n========== RESUMEN EVALUACIÓN ==========")
    print(f"modelo: {args.model}")
    print(f"track: {args.track}")
    print(f"obs_mode: {args.obs_mode}")
    print(f"episodes: {args.episodes}")
    print(f"reward mean: {np.mean(all_rewards):.2f}")
    print(f"reward max : {np.max(all_rewards):.2f}")
    print(f"reward min : {np.min(all_rewards):.2f}")
    print(f"len mean   : {np.mean(all_lengths):.1f}")
    print("outcomes   :", {o: all_outcomes.count(o) for o in set(all_outcomes)})

    if sess:
        print(f"[ok] clips de evaluación en: {sess}")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()