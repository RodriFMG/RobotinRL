"""
make_paper_figures.py - genera TODAS las figuras y tablas del paper a partir de
artefactos REALES del proyecto (sin reentrenar):

  1) figures/formacion_ppo.pdf   curvas de entrenamiento (history JSON de 1M pasos)
  2) figures/ablation_quant.pdf  ablacion cuantitativa:
        (a) barrido de checkpoints (100k..1M) en oval/seg  -> reward y tasa de meta vs pasos
        (b) ablacion de percepcion mask(oraculo) vs seg(aprendida) en el modelo 1M
        (c) generalizacion del modelo 1M en varias pistas (seg)
     + figures/ablation_results.json / .csv con los numeros
  3) figures/qualitative.pdf     pipeline de percepcion + trayectoria meta vs flip

Uso:
  MUJOCO_GL=egl python make_paper_figures.py --stage curves
  MUJOCO_GL=egl python make_paper_figures.py --stage quant --episodes 10
  MUJOCO_GL=egl python make_paper_figures.py --stage qual
  MUJOCO_GL=egl python make_paper_figures.py --stage all --episodes 10
"""
import os, csv, json, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "figures"
HIST = "plots/2026-07-01/history_01-43-15.json"
SEG_PTH = "models/seg_model/seg_dice.pth"
CKPT = "models/ppo_ckpt_{}_steps.zip"
FINAL = "models/ppo_ckpt_1000000_steps.zip"
os.makedirs(FIG, exist_ok=True)


# ============================ 1) CURVAS DE ENTRENAMIENTO ============================
def _series(h, key):
    xs, ys = [], []
    for r in h:
        if key in r:
            xs.append(r.get("time/total_timesteps", r.get("_step", 0)))
            ys.append(r[key])
    return np.array(xs, float), np.array(ys, float)


def fig_curves():
    h = json.load(open(HIST))
    panels = [
        ("rollout/ep_rew_mean",     "Mean episode reward",        False),
        ("rollout/ep_len_mean",     "Mean episode length (steps)", False),
        ("train/explained_variance","Value explained variance",   False),
        ("train/approx_kl",         "Approx. KL divergence",       True),
        ("train/clip_fraction",     "Clip fraction",               False),
        ("train/std",               "Policy action std",           False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4))
    axes = axes.reshape(-1)
    for ax, (k, title, logy) in zip(axes, panels):
        x, y = _series(h, k)
        if len(x) == 0:
            ax.set_visible(False); continue
        ax.plot(x / 1e3, y, lw=1.8, color="#1f4e79")
        if k == "rollout/ep_rew_mean":
            ax.axhline(0, ls="--", lw=0.9, color="0.5")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Timesteps (x$10^3$)", fontsize=8)
        ax.grid(alpha=.3)
        ax.tick_params(labelsize=8)
        if logy:
            ax.set_yscale("log")
    fig.tight_layout()
    for name in ("formacion_ppo.pdf", "formación_ppo.pdf"):
        fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("[curves] figures/formacion_ppo.pdf (+ alias con tilde)")


# ============================ EVALUACION LEAN (sin render por paso) ============================
def _build(mode, track, seg_res):
    from roombita_gym_env import RoombitaEnv
    if mode == "seg":
        env = RoombitaEnv(track=track, obs_mode="rgb", obstacle_slots=4, obstacle_prob=0.6,
                          time_max=45.0, cam_w=seg_res, cam_h=seg_res, seed=123,
                          render_mode="rgb_array")
    else:  # mask (oraculo del simulador)
        env = RoombitaEnv(track=track, obs_mode="mask", obstacle_slots=4, obstacle_prob=0.6,
                          time_max=45.0, cam_w=84, cam_h=84, seed=123)
    return env


def eval_cell(model, mode, track, episodes, segmenter, seed0=123):
    from eval_policy import FrameStacker, _infer_frame_stack, _to_model_layout, ensure_uint8_img
    env = _build(mode, track, seg_res=256)
    mshape = tuple(model.observation_space.shape)
    nstack = _infer_frame_stack(model, (84, 84, 3))
    outc = {}; rews = []; lens = []
    for e in range(episodes):
        stk = FrameStacker(nstack, (84, 84, 3))
        obs, _ = env.reset(seed=seed0 + e); first = True; done = False; R = 0.0; L = 0; info = {}
        while not done:
            if mode == "seg":
                base = ensure_uint8_img(segmenter.predict_seg_map(ensure_uint8_img(obs)))
            else:
                base = ensure_uint8_img(obs)
            o = stk.reset(base) if first else stk.append(base); first = False
            o = _to_model_layout(o, mshape)
            a, _ = model.predict(o, deterministic=True)
            obs, r, term, trunc, info = env.step(a); R += float(r); L += 1; done = term or trunc
        oc = info.get("outcome", "?"); outc[oc] = outc.get(oc, 0) + 1
        rews.append(R); lens.append(L)
    env.close()
    n = max(episodes, 1)
    return dict(track=track, mode=mode, episodes=episodes,
                reward_mean=float(np.mean(rews)), reward_std=float(np.std(rews)),
                len_mean=float(np.mean(lens)),
                goal_rate=outc.get("goal", 0) / n, flip_rate=outc.get("flip", 0) / n,
                lost_rate=outc.get("lost", 0) / n, timeout_rate=outc.get("timeout", 0) / n,
                outcomes=outc)


def fig_quant(episodes=10):
    from stable_baselines3 import PPO
    from vision_segmenter import VisionSegmenter
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter = VisionSegmenter(SEG_PTH, out_size=(84, 84), device=dev)

    results = {"checkpoint_sweep": [], "perception": [], "track_gen": []}

    # (a) barrido de checkpoints en oval/seg
    steps = [100000, 300000, 500000, 700000, 900000, 1000000]
    for s in steps:
        m = PPO.load(CKPT.format(s), device="auto")
        r = eval_cell(m, "seg", "oval", episodes, segmenter)
        r["steps"] = s; results["checkpoint_sweep"].append(r)
        print(f"[quant] ckpt {s:>7} oval/seg  goal={r['goal_rate']:.2f} flip={r['flip_rate']:.2f} R={r['reward_mean']:.1f}")

    # (b) percepcion: modelo final mask (oraculo) vs seg (aprendida) en oval
    mfin = PPO.load(FINAL, device="auto")
    for mode in ("mask", "seg"):
        r = eval_cell(mfin, mode, "oval", episodes, segmenter)
        results["perception"].append(r)
        print(f"[quant] 1M oval/{mode:4s}  goal={r['goal_rate']:.2f} flip={r['flip_rate']:.2f} R={r['reward_mean']:.1f}")

    # (c) generalizacion del modelo 1M (seg) en varias pistas
    for tr in ("straight", "oval", "s_curve", "zigzag"):
        r = eval_cell(mfin, "seg", tr, episodes, segmenter)
        results["track_gen"].append(r)
        print(f"[quant] 1M {tr:9s}/seg  goal={r['goal_rate']:.2f} flip={r['flip_rate']:.2f} R={r['reward_mean']:.1f}")

    json.dump(results, open(os.path.join(FIG, "ablation_results.json"), "w"), indent=2)
    # CSV plano
    with open(os.path.join(FIG, "ablation_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "steps", "track", "mode", "episodes", "reward_mean", "reward_std",
                    "len_mean", "goal_rate", "flip_rate", "lost_rate", "timeout_rate"])
        for g, rows in results.items():
            for r in rows:
                w.writerow([g, r.get("steps", ""), r["track"], r["mode"], r["episodes"],
                            f"{r['reward_mean']:.2f}", f"{r['reward_std']:.2f}", f"{r['len_mean']:.1f}",
                            f"{r['goal_rate']:.2f}", f"{r['flip_rate']:.2f}", f"{r['lost_rate']:.2f}",
                            f"{r['timeout_rate']:.2f}"])

    # ---- figura ----
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
    sw = results["checkpoint_sweep"]
    xs = [r["steps"] / 1e3 for r in sw]
    ax[0].plot(xs, [r["goal_rate"] for r in sw], "-o", label="goal rate", color="#2e7d32")
    ax[0].plot(xs, [r["flip_rate"] for r in sw], "-s", label="flip rate", color="#c62828")
    ax[0].set_xlabel("Training timesteps (x$10^3$)"); ax[0].set_ylabel("Rate")
    ax[0].set_title("(a) Learning-progress ablation (oval, seg)"); ax[0].set_ylim(-.03, 1.03)
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

    per = {r["mode"]: r for r in results["perception"]}
    modes = ["mask", "seg"]; labels = ["mask\n(oracle sim)", "seg\n(learned MiniUNet)"]
    gr = [per[m]["goal_rate"] for m in modes]; fr = [per[m]["flip_rate"] for m in modes]
    x = np.arange(2)
    ax[1].bar(x - .2, gr, .4, label="goal", color="#2e7d32")
    ax[1].bar(x + .2, fr, .4, label="flip", color="#c62828")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_title("(b) Perception gap (1M model, oval)"); ax[1].set_ylim(0, 1.03)
    ax[1].grid(alpha=.3, axis="y"); ax[1].legend(fontsize=8)

    tg = results["track_gen"]
    tks = [r["track"] for r in tg]; x = np.arange(len(tks))
    ax[2].bar(x - .2, [r["goal_rate"] for r in tg], .4, label="goal", color="#2e7d32")
    ax[2].bar(x + .2, [r["flip_rate"] for r in tg], .4, label="flip", color="#c62828")
    ax[2].set_xticks(x); ax[2].set_xticklabels(tks, fontsize=8, rotation=15)
    ax[2].set_title("(c) Track generalization (1M model, seg)"); ax[2].set_ylim(0, 1.03)
    ax[2].grid(alpha=.3, axis="y"); ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation_quant.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[quant] figures/ablation_quant.pdf + ablation_results.{json,csv}")
    return results


# ============================ 3) FIGURA CUALITATIVA ============================
def _run_capture(model, segmenter, track, seed, want=None, max_keep=400):
    from eval_policy import FrameStacker, _infer_frame_stack, _to_model_layout, ensure_uint8_img
    from arena_env import Vision
    env = _build("seg", track, seg_res=256)
    if getattr(env, "vis", None) is None:
        env.vis = Vision(env.m, 240, 320)
    mshape = tuple(model.observation_space.shape)
    stk = FrameStacker(_infer_frame_stack(model, (84, 84, 3)), (84, 84, 3))
    obs, _ = env.reset(seed=seed); first = True; done = False; info = {}
    frames = []
    while not done:
        rgb = env.vis.rgb(env.d)
        pred = ensure_uint8_img(segmenter.predict_seg_map(ensure_uint8_img(obs)))
        frames.append(dict(rgb=rgb, simseg=env.vis.seg_map(env.d),
                           overlay=env.vis.overlay(rgb, env.d), pred=pred))
        o = stk.reset(pred) if first else stk.append(pred); first = False
        o = _to_model_layout(o, mshape)
        a, _ = model.predict(o, deterministic=True)
        obs, r, term, trunc, info = env.step(a); done = term or trunc
        if len(frames) > max_keep:
            frames = frames[::2]
    env.close()
    return frames, info.get("outcome", "?")


def fig_qual():
    from stable_baselines3 import PPO
    from vision_segmenter import VisionSegmenter
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seg = VisionSegmenter(SEG_PTH, out_size=(84, 84), device=dev)
    model = PPO.load(FINAL, device="auto")

    goal_f = flip_f = None
    for sd in range(123, 145):
        frames, oc = _run_capture(model, seg, "oval", sd)
        print(f"[qual] seed {sd}: {oc} ({len(frames)} frames)")
        if oc == "goal" and goal_f is None:
            goal_f = frames
        if oc == "flip" and flip_f is None:
            flip_f = frames
        if goal_f and flip_f:
            break

    def pick(frames, n=4):
        idx = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in idx]

    fig, axes = plt.subplots(3, 4, figsize=(11, 8))
    # fila A: pipeline de percepcion (primer frame de la meta o del flip)
    src = (goal_f or flip_f)
    f0 = src[len(src) // 3]
    for ax, key, ttl in zip(axes[0], ["rgb", "simseg", "pred", "overlay"],
                            ["RGB camera", "Sim segmentation", "Predicted seg (MiniUNet)", "Obstacle overlay"]):
        ax.imshow(f0[key]); ax.set_title(ttl, fontsize=10); ax.axis("off")
    # fila B: episodio meta (RGB en el tiempo)
    if goal_f:
        for ax, fr in zip(axes[1], pick(goal_f)):
            ax.imshow(fr["rgb"]); ax.axis("off")
        axes[1][0].set_ylabel("GOAL episode", fontsize=10)
        axes[1][0].axis("on"); axes[1][0].set_xticks([]); axes[1][0].set_yticks([])
    # fila C: episodio flip (RGB en el tiempo)
    if flip_f:
        for ax, fr in zip(axes[2], pick(flip_f)):
            ax.imshow(fr["rgb"]); ax.axis("off")
        axes[2][0].set_ylabel("FLIP (failure) episode", fontsize=10)
        axes[2][0].axis("on"); axes[2][0].set_xticks([]); axes[2][0].set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "qualitative.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[qual] figures/qualitative.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["curves", "quant", "qual", "all"], default="all")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    t0 = time.time()
    if args.stage in ("curves", "all"):
        fig_curves()
    if args.stage in ("quant", "all"):
        fig_quant(args.episodes)
    if args.stage in ("qual", "all"):
        fig_qual()
    print(f"[done] {time.time()-t0:.1f}s")
