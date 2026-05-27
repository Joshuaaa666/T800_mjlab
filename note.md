# t800

WANDB_MODE=disabled python scripts/train.py T800-1307-Stage-I \
  --motion-file data/act01_attack_uppercut.npz \
  --env.scene.num-envs 8192 \
  --env.commands.motion.sampling-mode adaptive \
  --video True \
  --video-interval 10000


python scripts/play.py T800-1307-Stage-I \
  --motion-file data/act01_attack_uppercut.npz \
  --checkpoint-file 


python -m tensorboard.main --logdir 

## test
WANDB_MODE=disabled python scripts/train.py T800-1307-Stage-I \
  --motion-file data/act01_attack_uppercut.npz \
  --env.scene.num-envs 2 \
  --env.commands.motion.sampling-mode adaptive \
  --video True \
  --video-interval 1000



uv run train Mjlab-Velocity-Flat-Unitree-G1 \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 10000 \
    --agent.algorithm.learning-rate 3e-4 \
    --env.decimation 2

WANDB_MODE=disabled uv run train Mjlab-Velocity-Flat-Unitree-G1 \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 10000 \
    --agent.algorithm.learning-rate 3e-4 \
    --env.decimation 2 \
    --video True \
    --video-interval 10000

WANDB_MODE=disabled python scripts/train.py Unitree-G1-1307-Stage-I \
    --motion_file=src/assets/motions/g1/1307.npz \
    --env.scene.num-envs=8192 \
    --env.commands.motion.sampling-mode=adaptive \
    --video True \
    --video-interval 10000 \
    --agent.resume True \
    --agent.load-run 2026-05-25_11-11-20 \
    --agent.load-checkpoint model_4500.pt


WANDB_MODE=disabled python scripts/train.py Unitree-G1-1307-Stage-I --motion_file=src/assets/motions/g1/1307.npz --env.scene.num-envs=8192 --env.commands.motion.sampling-mode=adaptive --video True --video-interval 10000

WANDB_MODE=disabled python scripts/train.py Unitree-G1-1307-Stage-II --motion_file=src/assets/motions/g1/1307.npz --env.scene.num-envs=8192 --env.commands.motion.sampling-mode=adaptive --agent.resume=True --video True --video-interval 10000



python scripts/play.py Unitree-G1-1307-Stage-I \
    --motion_file=src/assets/motions/g1/1307.npz \
    --checkpoint_file=


