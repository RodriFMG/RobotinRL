================================================================
 ROOMBITA RL ARENA  -  line-follower + obstaculos en MuJoCo
================================================================

Robot tipo roombita que sigue una pista dentro de un laboratorio simulado,
con obstaculos, segmentacion, grabacion de clips y entorno Gymnasium para PPO.
Todo 100% MuJoCo (sin ROS2).

----------------------------------------------------------------
 ARCHIVOS
----------------------------------------------------------------
x  tracks.py            Catalogo de pistas + clase Track (progreso por arco).
x  arena_build.py       Construye arena.xml para una pista (mismo lab) + slots.
x  arena_env.py         Vision, ObstacleField, Lighting, Episode, Dynamics, Rewards.
o  run_play.py          Corre/graba episodios (seguidor scriptado). Config por CLI.
x  view_clip.py         Reproductor de clips .npz (RGB|seg|obstaculos + reward).
o   .py  Entorno Gymnasium (obs_mode state/mask/rgb; accion = ruedas).
o  train_ppo.py         Entrena PPO (Stable-Baselines3).
x  eval_policy.py       Evalua un modelo entrenado, con grabacion opcional.
x  arena.xml            Ultima arena generada (se regenera al elegir pista).
x  textures/ meshes/    Assets procedurales (se autogeneran si faltan).

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
  paso seguro sin choque      : safe        = +0.05 por paso
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
  
  avance contrario sobre pista: progress    = -5.0 por metro avanzado
  paso contrario sin choque   : sage        = -0.05 por paso
----------------------------------------------------------------

ACCIONES (action_space)

numero de acciones : 2
tipo : Box(-1, 1, shape=(2,))
archivo : roombita_gym_env.py
significado : [rueda_izquierda, rueda_derecha]

Cada valor controla la velocidad de una rueda del robot.
Internamente se escala a velocidad angular, aproximadamente hasta +/-18 rad/s.

action[0] : rueda izquierda
- valor positivo -> gira hacia adelante
- valor negativo -> gira hacia atras
- valor cercano 0 -> rueda casi detenida

action[1] : rueda derecha
- valor positivo -> gira hacia adelante
- valor negativo -> gira hacia atras
- valor cercano 0 -> rueda casi detenida

Ejemplos:
[ 1.0, 1.0] -> avanzar recto
[-1.0, -1.0] -> retroceder recto
[ 1.0, -1.0] -> girar sobre su eje
[ 0.5, 1.0] -> curva hacia un lado
[ 1.0, 0.5] -> curva hacia el otro lado

----------------------------------------------------------------

ESTADOS / OBSERVACIONES (observation_space)

archivo : roombita_gym_env.py
modos : obs_mode=state | obs_mode=mask | obs_mode=rgb

El entorno permite 3 tipos de observacion para entrenar PPO:

obs_mode=state

state: vector de 10 elementos

[lat, sin(err), cos(err), v, w, on_track, rem, d_obs, sin(b_obs), cos(b_obs)]

lat:
Error lateral respecto al centro de la pista.
Indica que tan lejos esta el robot de la linea/ruta ideal.

sin(err):
Seno del error angular entre la orientacion del robot y la direccion
correcta de avance sobre la pista.

cos(err):
Coseno del error angular.
Se usa junto con sin(err) para evitar saltos bruscos de angulo.

v:
Velocidad lineal del robot.

w:
Velocidad angular del robot.

on_track:
Indica si el robot esta dentro de la pista.
Normalmente 1 = en pista, 0 = fuera de pista.

rem:
Distancia restante hasta la meta, medida sobre la pista.
No es distancia en linea recta, es distancia siguiendo el recorrido.

d_obs:
Distancia al obstaculo mas cercano o mas relevante.

sin(b_obs):
Seno del angulo hacia el obstaculo.

cos(b_obs):
Coseno del angulo hacia el obstaculo.

Politica usada:
obs_mode=state -> MlpPolicy

obs_mode=mask

mask: imagen de segmentacion 84x84x3 uint8

Representa la vista procesada de la camara del robot.
Es una observacion mas limpia que RGB porque separa elementos importantes
como pista, linea, obstaculos y fondo.

Uso:
Sirve para entrenar al agente con informacion visual segmentada.
Es mas facil que RGB porque reduce el ruido visual del laboratorio.

Politica usada:
obs_mode=mask -> CnnPolicy

obs_mode=rgb

rgb: imagen RGB 84x84x3 uint8

Representa directamente lo que ve la camara del robot.

Incluye:
- pista
- linea
- obstaculos
- suelo
- mesas
- personas
- cambios de iluminacion
- ruido visual del laboratorio

Uso:
Es el modo mas realista, pero tambien el mas dificil de entrenar.

Politica usada:
obs_mode=rgb -> CnnPolicy

MODELO DE VISION / DEEP LEARNING

Actualmente no hay un modelo de vision propio creado manualmente por nosotros.

Cuando se usa obs_mode=mask o obs_mode=rgb, el procesamiento visual lo hace
Stable-Baselines3 mediante CnnPolicy.

Internamente CnnPolicy usa una CNN llamada NatureCNN, basada en la arquitectura
clasica usada en Atari DQN.

Ubicacion dentro del entorno virtual:

  venv/Lib/site-packages/stable_baselines3/common/torch_layers.py

Clase:

  NatureCNN

En train_ppo.py la seleccion de politica se hace con:

  policy = "MlpPolicy" if obs_mode == "state" else "CnnPolicy"

Entonces:

  obs_mode=state -> MlpPolicy, red densa para el vector de 10 estados.
  obs_mode=mask  -> CnnPolicy, CNN para procesar la mascara.
  obs_mode=rgb   -> CnnPolicy, CNN para procesar el frame RGB.

Si mas adelante se quiere usar una CNN propia, se debe modificar train_ppo.py
usando policy_kwargs y un features_extractor_class personalizado.
