"""
run_play.py - corre/graba episodios sobre cualquier pista del catalogo, con el
sistema de reward completo.

Ejemplos:
  python run_play.py
  python run_play.py --track=oval --save=True --episode_count=10
  python run_play.py --track=hairpin --obstacle_slots=8 --obstacle_prob=0.75 --world_speed=10 --save=True
  python run_play.py --track=random --seed=42 --obstacle_count_mode=fixed --headless=True --save=True

Slots de obstaculos: --obstacle_slots=8 NO significa 8 fijos; cada slot aparece
con --obstacle_prob (modo probabilistic). En modo fixed aparecen todos.

Al grabar crea runs/<fecha-hora>/ con:
  config.json, episode_summary.json, episode_summary.csv y un .npz por episodio
  (claves: rgb, seg, obstacle, reward, done, info, track_name, config).
Reproducir:  python view_clip.py --clip runs/.../episode_0001.npz
"""
import os, sys, json, csv, time, argparse, datetime
import numpy as np
import mujoco
import mujoco.viewer

from tracks import get_track
from arena_build import build_arena
from arena_env import Vision, ObstacleField, Lighting, Episode, Dynamics, Rewards

CTRL_EVERY = 12
WHEEL_R, HALF_AXLE, KP = 0.028, 0.085, 2.6


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "t", "yes", "y", "si")


def diff_drive(v, w):
    return (v - w*HALF_AXLE)/WHEEL_R, (v + w*HALF_AXLE)/WHEEL_R


def follow(mask, h, w):
    # franja CERCANA al robot (parte baja-media de la imagen): ahi la linea es
    # grande y estable, y se tapa menos con gente/curvas lejanas.
    band = mask.copy()
    band[:int(h*0.45)] = False          # descarta lo muy lejano (arriba)
    band[int(h*0.95):] = False          # descarta el borde inferior
    n = band.sum()
    if n < 20:
        return None
    cx = (band.sum(0) * np.arange(w)).sum() / n
    return (cx - w/2) / (w/2)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str2bool, nargs="?", const=True, default=False)
    ap.add_argument("--time_max", type=float, default=45.0)
    ap.add_argument("--track", type=str, default="simple_line")
    ap.add_argument("--seed", type=int, default=-1)                 # -1 = aleatoria
    ap.add_argument("--robot_speed", type=float, default=28.0)      # cm/s
    ap.add_argument("--world_speed", type=float, default=1.0)       # dilatacion temporal
    ap.add_argument("--obstacle_slots", type=int, default=8)
    ap.add_argument("--obstacle_prob", type=float, default=0.6)
    ap.add_argument("--obstacle_count_mode", type=str, default="probabilistic",
                    choices=["probabilistic", "fixed"])
    ap.add_argument("--brightness", type=float, default=1.0)
    ap.add_argument("--random_brightness", type=str2bool, nargs="?", const=True, default=False)
    ap.add_argument("--record_fps", type=float, default=15.0)
    ap.add_argument("--camera_width", type=int, default=160)
    ap.add_argument("--camera_height", type=int, default=120)
    ap.add_argument("--episode_count", type=int, default=5)
    ap.add_argument("--headless", type=str2bool, nargs="?", const=True, default=False)
    ap.add_argument("--show_viewer", type=str2bool, nargs="?", const=True, default=True)
    ap.add_argument("--show_robot_view", type=str2bool, nargs="?", const=True, default=True)
    a = ap.parse_args()
    if a.headless:
        a.show_viewer = False; a.show_robot_view = False
    if a.seed < 0:
        a.seed = int(time.time()) % 100000
    return a


def main():
    a = parse()
    track = get_track(a.track, seed=a.seed)
    build_arena(track, n_slots=max(a.obstacle_slots, 1), path="arena.xml", seed=a.seed)
    m = mujoco.MjModel.from_xml_path("arena.xml")
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    vis = Vision(m, a.camera_height, a.camera_width)
    field = ObstacleField(m, d, track)
    light = Lighting(m)
    dyn = Dynamics(m, d, seed=a.seed)
    ep = Episode(m, d, track, seed=a.seed)
    rew = Rewards(m, d, track, field)
    lid = m.actuator("left_motor").id; rid = m.actuator("right_motor").id
    pan = m.actuator("pan_servo").id;  tilt = m.actuator("tilt_servo").id
    v_robot = a.robot_speed/100.0
    CAP_EVERY = max(int(round((1.0/a.record_fps)/dt)), 1)            # cadencia FIJA de captura
    fps = 1.0/(CAP_EVERY*dt)

    sess = None
    if a.save:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sess = os.path.join("runs", stamp); os.makedirs(sess, exist_ok=True)
        with open(os.path.join(sess, "config.json"), "w") as f:
            json.dump(vars(a), f, indent=2)
        print(f"[grabando] {sess}  (track={track.name}, {fps:.1f} fps/clip)")
    cfg_json = json.dumps(vars(a))

    viewer = cv2 = None
    if a.show_viewer:
        viewer = mujoco.viewer.launch_passive(m, d)
    if a.show_robot_view:
        import cv2 as _cv2; cv2 = _cv2
        cv2.namedWindow("robot view", cv2.WINDOW_NORMAL); cv2.resizeWindow("robot view", 760, 200)

    summary = []
    for ei in range(a.episode_count):
        ep.reset(field=field, obstacles=(a.obstacle_slots > 0), prob=a.obstacle_prob,
                 mode=a.obstacle_count_mode, lighting=light,
                 brightness=a.brightness, random_brightness=a.random_brightness)
        dyn.reset(); rew.reset()
        rgb_l, seg_l, obs_l, rew_l, done_l = [], [], [], [], []
        sim_t = 0.0; last_w = 0.0; stuck = 0; recovery = 0.0; lost = 0
        prev = rew._xy(); outcome = "running"; gstep = 0; done = False
        while not done:
            err = follow(vis.mask(d, "line"), a.camera_height, a.camera_width)
            if recovery > 0:
                v, w = -0.5*v_robot, 1.6
            elif err is not None:
                w = -KP*err; v = v_robot*(1-0.55*min(abs(err), 1)); last_w = w; lost = 0
            else:
                lost += 1
                if lost <= 12:                      # perdida breve (gente/curva): seguir derecho
                    w = float(np.clip(last_w, -0.6, 0.6)); v = 0.55*v_robot
                else:                               # perdida sostenida: arco LENTO, sin trompo
                    w = np.sign(last_w+1e-6)*1.0; v = 0.18*v_robot
            d.ctrl[lid], d.ctrl[rid] = diff_drive(v, w)
            d.ctrl[pan] = 0.0; d.ctrl[tilt] = 0.0

            nsub = max(int(round(a.world_speed)), 1) * CTRL_EVERY
            for _ in range(nsub):
                dyn.update(dt); mujoco.mj_step(m, d); gstep += 1; sim_t += dt
                if gstep % CAP_EVERY == 0:
                    r, terminal, oc = rew.step(CAP_EVERY*dt)
                    pos = rew._xy(); moved = np.hypot(*(pos-prev)); prev = pos
                    if recovery > 0: recovery -= CAP_EVERY*dt
                    elif v > 0.1*v_robot and moved < 0.2*abs(v)*CAP_EVERY*dt:
                        stuck += 1
                        if stuck >= 3: recovery = 0.4; stuck = 0
                    else: stuck = 0

                    end = False
                    if oc == "goal":
                        r += rew.arrival_reward(sim_t, a.time_max); outcome = "goal"; end = True
                    elif oc == "reverse_goal":
                        r += rew.reverse_reward(); outcome = "reverse_goal"; end = True
                    elif oc in ("flip", "lost"):
                        outcome = oc; end = True
                    elif sim_t >= a.time_max:
                        tr, _ = rew.timeout_reward(); r += tr; outcome = "timeout"; end = True

                    if a.save:
                        rgb = vis.rgb(d)
                        rgb_l.append(rgb); seg_l.append(vis.seg_map(d)); obs_l.append(vis.overlay(rgb, d))
                        rew_l.append(r); done_l.append(end)
                    if end:
                        done = True; break

            if a.show_robot_view:
                strip = cv2.cvtColor(np.hstack([vis.rgb(d), vis.seg_map(d)]), cv2.COLOR_RGB2BGR)
                cv2.putText(strip, f"{track.name} ep{ei} t{sim_t:4.1f} R{rew.total:6.1f} x{int(a.world_speed)} {outcome}",
                            (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imshow("robot view", strip)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    a.episode_count = 0; done = True
            if a.show_viewer:
                viewer.sync()

        s = rew.summary()
        print(f"  ep {ei:3d} | {outcome:12s} | t={sim_t:5.1f}s | "
              f"R={s['total']:7.2f}  [adv {s['advance']:.1f} | arr {s['arrival']+s['speed_bonus']:.1f} | "
              f"col {s['collision']:.1f} | avoid {s['avoid']:.1f} | off {s['offtrack']:.1f} | "
              f"ret {s['ret']:.1f} | rev {s['wrong_dir']:.1f} | to {s['timeout']:.1f} | flip {s['flip']:.1f}]")
        row = dict(episode=ei, track=track.name, outcome=outcome, sim_time=round(sim_t, 2), **s)
        summary.append(row)

        if a.save and rgb_l:
            info = dict(outcome=outcome, sim_time=round(sim_t, 2), fps=round(fps, 3),
                        world_speed=a.world_speed, robot_speed=a.robot_speed, reward=s)
            fn = os.path.join(sess, f"episode_{ei+1:04d}.npz")
            np.savez_compressed(
                fn,
                rgb=np.asarray(rgb_l, np.uint8), seg=np.asarray(seg_l, np.uint8),
                obstacle=np.asarray(obs_l, np.uint8),
                reward=np.asarray(rew_l, np.float32), done=np.asarray(done_l, bool),
                info=json.dumps(info), track_name=track.name, config=cfg_json)
        if a.episode_count == 0:
            break

    if a.save and sess:
        with open(os.path.join(sess, "episode_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if summary:
            with open(os.path.join(sess, "episode_summary.csv"), "w", newline="") as f:
                wcsv = csv.DictWriter(f, fieldnames=list(summary[0].keys())); wcsv.writeheader()
                wcsv.writerows(summary)
        print(f"[ok] {len(summary)} episodios en {sess}")
    if a.show_viewer and viewer is not None:
        viewer.close()
    if a.show_robot_view and cv2 is not None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
