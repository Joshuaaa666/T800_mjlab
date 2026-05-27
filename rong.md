docker build -t mjlab .


docker run -it --rm --gpus '"device=6"' \
  -v ./unitree_rl_mjlab/:/workspace/unitree_rl_mjlab/ \
  -v ./logs:/workspace/logs \
  mjlab bash
  
uv pip install --python /app/.venv/bin/python --no-cache "warp-lang<1.13"

WANDB_MODE=disabled python unitree_rl_mjlab/scripts/train.py T800-1307-Stage-I \
  --motion-file data/act01_attack_uppercut.npz \
  --env.scene.num-envs 4096 \
  --env.commands.motion.sampling-mode adaptive \
  --video True --video-interval 10000

WANDB_MODE=disabled python scripts/train.py T800-1307-Stage-II \
  --motion-file data/act01_attack_uppercut.npz \
  --env.scene.num-envs 4096 \
  --env.commands.motion.sampling-mode adaptive \
  --agent.resume=True \
  --agent.load-run 2026-05-26_09-57-09 \
  --agent.load-checkpoint model_31000.pt \
  --video True \
  --video-interval 10000

WANDB_MODE=disabled python scripts/train.py T800-1307-Stage-III \
  --motion-file data/act01_attack_uppercut.npz \
  --env.scene.num-envs 4096 \
  --env.commands.motion.sampling-mode adaptive \
  --agent.resume=True \
  --agent.load-run <Stage-II的日期文件夹> \
  --agent.load-checkpoint <Stage-II的最新model.pt> \
  --video True \
  --video-interval 10000

python3 scripts/play.py T800-1307-Stage-I \
  --motion-file data/act01_attack_uppercut.npz \
  --checkpoint-file logs/rsl_rl/t800_tracking/2026-05-26_09-57-09/model_31000.pt \
  --video-length 689


python scripts/play.py T800-1307-Stage-I --motion-file data/act01_attack_uppercut.npz --viewer native --checkpoint-file