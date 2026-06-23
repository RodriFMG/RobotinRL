================================================================
 ROOMBITA RL ARENA  -  line-follower + obstaculos en MuJoCo
================================================================

Robot tipo roombita que sigue una pista dentro de un laboratorio simulado,
con obstaculos, segmentacion, grabacion de clips y entorno Gymnasium para PPO.
Todo 100% MuJoCo (sin ROS2).

----------------------------------------------------------------
 ARCHIVOS
----------------------------------------------------------------
  tracks.py            Catalogo de pistas + clase Track (progreso por arco).
  arena_build.py       Construye arena.xml para una pista (mismo lab) + slots.
  arena_env.py         Vision, ObstacleField, Lighting, Episode, Dynamics, Rewards.
  run_play.py          Corre/graba episodios (seguidor scriptado). Config por CLI.
  view_clip.py         Reproductor de clips .npz (RGB|seg|obstaculos + reward).
  roombita_gym_env.py  Entorno Gymnasium (obs_mode state/mask/rgb; accion = ruedas).
  train_ppo.py         Entrena PPO (Stable-Baselines3).
  eval_policy.py       Evalua un modelo entrenado, con grabacion opcional.
  arena.xml            Ultima arena generada (se regenera al elegir pista).
  textures/ meshes/    Assets procedurales (se autogeneran si faltan).

----------------------------------------------------------------
 ESTRUCTURA DE CARPETAS
----------------------------------------------------------------
  proyecto/
    *.py, arena.xml, textures/, meshes/
    runs/<fecha-hora>/            <- generado por run_play.py --save=True
        config.json
        episode_summary.json / .csv
        episode_0001.npz ...      claves: rgb seg obstacle reward done info
                                          track_name config
    models/<track>_<obs_mode>.zip <- generado por train_ppo.py
    runs_eval/<fecha-hora>/       <- generado por eval_policy.py --record=True

----------------------------------------------------------------
 PISTAS (--track=)
----------------------------------------------------------------
  straight  simple_line  oval  zigzag  s_curve  eight  hairpin  complex  random
  (random usa --seed). Mismo laboratorio para todas; la pista va a ras del piso.

----------------------------------------------------------------
 COMANDOS
----------------------------------------------------------------
  # correr sin grabar (visor + camara)
  python run_play.py --track=oval

  # grabar 10 episodios, obstaculos probabilisticos, rapido y sin ventanas
  python run_play.py --track=hairpin --save=True --episode_count=10 \
      --obstacle_slots=8 --obstacle_prob=0.75 --world_speed=10 --headless=True

  # reproducir un clip
  python view_clip.py --clip runs/2026-06-17_22-35-12/episode_0001.npz

  # entrenar PPO
  python train_ppo.py --track=simple_line --total_timesteps=100000 --obs_mode=state
  python train_ppo.py --track=oval --obs_mode=mask --total_timesteps=200000

  # evaluar y grabar
  python eval_policy.py --model models/simple_line_state.zip --track=simple_line \
      --obs_mode=state --episodes=5 --record=True

  Headless / sandbox sin display:  anteponer  MUJOCO_GL=egl

----------------------------------------------------------------
 ARGUMENTOS DE run_play.py
----------------------------------------------------------------
  --save (False) --time_max (45) --track (simple_line) --seed (-1=random)
  --robot_speed (28 cm/s) --world_speed (1) --obstacle_slots (8)
  --obstacle_prob (0.6) --obstacle_count_mode (probabilistic|fixed)
  --brightness (1.0) --random_brightness (False) --record_fps (15)
  --camera_width (160) --camera_height (120) --episode_count (5)
  --headless (False) --show_viewer (True) --show_robot_view (True)

  Slots != cantidad: --obstacle_slots=8 son 8 candidatos; cada uno aparece con
  --obstacle_prob. Con --obstacle_count_mode=fixed aparecen los 8.

----------------------------------------------------------------
 REWARD ACTUAL (pesos por defecto en arena_env.Rewards.DEFAULT_W)
----------------------------------------------------------------
  avance correcto sobre pista : progress    = +5.0  por metro avanzado
  paso seguro sin choque      : safe        = +0.03 por paso
  llegada correcta            : arrival     = +20.0
  bonus por rapidez           : speed_bonus = +20.0 * (time_max-t)/time_max
  llegada en direccion contraria: wrong_dir = -20.0   (= -arrival)
  timeout (base)              : timeout     = -5.0
  distancia restante en pista : timeout_per_m = -1.5 por metro (sobre la pista)
  choque con obstaculo        : collision   = -3.0  (evento, cooldown 1 s)
  esquive de obstaculo        : avoid       = +3.0  (1 vez por obstaculo)
  salida de pista             : offtrack    = -0.10 por paso
  retorno a pista             : ret         = +1.5
  vuelco / inestabilidad      : flip        = -8.0  (termina episodio)
  estancamiento               : stuck       = -1.0  por segundo (acotado)
