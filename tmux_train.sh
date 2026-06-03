#!/bin/bash
# T800 训练 TMUX 管理脚本 (宿主机版)
# 用法:
#   ./tmux_train.sh [start|train_B|attach|kill|list|logs|status]

# 训练B配置（加躯干起身奖励）
TRAIN_B_GPU="6"
TRAIN_B_RUN="2026-06-02_06-42-18"
TRAIN_B_CKPT="model_17000.pt"

SESSION="mjlab_train"
PROJECT_DIR="/home/wangyirong/projects/倒地起身V1"
DOCKER_IMAGE="mjlab"
DOCKER_GPUS="device=1,6"
CUDA_DEVICES="1,6"

start() {
    tmux kill-session -t $SESSION 2>/dev/null
    tmux new-session -d -s $SESSION -n "train"

    # ======== 窗格 1: 训练 (宿主机启动 Docker, Stage I 全新训练) ========
    tmux send-keys -t $SESSION:0.0 \
        "cd $PROJECT_DIR && \
         CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
         docker run -it --rm --gpus all \
           -e CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
           -v ./unitree_rl_mjlab/:/workspace/unitree_rl_mjlab/ \
           -v ./logs:/workspace/logs \
           $DOCKER_IMAGE bash -c '
             cd /workspace/unitree_rl_mjlab && \
             WANDB_MODE=disabled python scripts/train.py T800-1307-Stage-I \
               --motion-file data/act01_attack_uppercut.npz \
               --env.scene.num-envs 16384 \
               --env.commands.motion.sampling-mode adaptive \
               --gpu-ids \"[0, 1]\" \
               --video True --video-interval 20000
           '" C-m

    # ======== 窗格 2: TensorBoard ========
    tmux split-window -h -t $SESSION:0
    tmux send-keys -t $SESSION:0.1 \
        "python3 -m tensorboard.main --logdir $PROJECT_DIR/unitree_rl_mjlab/logs/rsl_rl/t800_tracking/ --port 6006" C-m

    # ======== 窗格 3: GPU 监控 ========
    tmux split-window -v -t $SESSION:0.1
    tmux send-keys -t $SESSION:0.2 "watch -n 2 nvidia-smi" C-m

    echo "✅ 训练已启动 (TMUX: $SESSION)"
    echo "   连接: tmux attach -t $SESSION"
    echo "   TensorBoard: http://localhost:6006"
}

train_B() {
    local GPU="$TRAIN_B_GPU"
    local RUN_DIR="$TRAIN_B_RUN"
    local CKPT="$TRAIN_B_CKPT"

    tmux kill-session -t train_B 2>/dev/null
    tmux new-session -d -s train_B -n "train_B"

    tmux send-keys -t train_B:0 \
        "cd $PROJECT_DIR && \
         docker run -it --rm --gpus all \
           -e CUDA_VISIBLE_DEVICES=$GPU \
           -v ./unitree_rl_mjlab/:/workspace/unitree_rl_mjlab/ \
           -v ./logs:/workspace/logs \
           -e RUN_DIR=$RUN_DIR \
           -e CKPT=$CKPT \
           $DOCKER_IMAGE bash -c '
             cd /workspace/unitree_rl_mjlab && \
             WANDB_MODE=disabled python scripts/train.py T800-1307-Stage-I-WithReward \
               --motion-file data/act01_attack_uppercut.npz \
               --env.scene.num-envs 8192 \
               --env.commands.motion.sampling-mode adaptive \
               --gpu-ids \"[0]\" \
               --video True --video-interval 20000 \
               --agent.resume True \
               --agent.load-run \"\$RUN_DIR\" \
               --agent.load-checkpoint \"\$CKPT\"
           '" C-m

    echo "✅ train_B 已启动 (GPU $GPU, 加躯干起身奖励)"
    echo "   从 $RUN_DIR/$CKPT 继续"
    echo "   连接: tmux attach -t train_B"
}

fresh() {
    local GPU="$TRAIN_B_GPU"

    tmux kill-session -t train_B 2>/dev/null
    tmux new-session -d -s train_B -n "train_B"

    tmux send-keys -t train_B:0 \
        "cd $PROJECT_DIR && \
         docker run -it --rm --gpus all \
           -e CUDA_VISIBLE_DEVICES=$GPU \
           -v ./unitree_rl_mjlab/:/workspace/unitree_rl_mjlab/ \
           -v ./logs:/workspace/logs \
           $DOCKER_IMAGE bash -c '
             cd /workspace/unitree_rl_mjlab && \
             WANDB_MODE=disabled python scripts/train.py T800-1307-Stage-I-WithReward \
               --motion-file data/act01_attack_uppercut.npz \
               --env.scene.num-envs 8192 \
               --env.commands.motion.sampling-mode adaptive \
               --gpu-ids \"[0]\" \
               --video True --video-interval 20000
           '" C-m

    echo "✅ 全新训练已启动 (GPU $GPU, 加躯干起身奖励)"
    echo "   连接: tmux attach -t train_B"
}

attach() {
    tmux attach -t $SESSION
}

kill() {
    tmux kill-session -t $SESSION 2>/dev/null && echo "✅ 已终止" || echo "❌ session 不存在"
}

list() {
    tmux ls 2>/dev/null || echo "❌ 没有活跃的 tmux session"
}

logs() {
    local logdir="$PROJECT_DIR/unitree_rl_mjlab/logs/rsl_rl/t800_tracking"
    local latest=$(ls -td "$logdir"/*/ 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "最新日志: $latest"
        ls -lh "$latest"/*.pt 2>/dev/null | tail -5
    else
        echo "❌ 未找到日志"
    fi
}

status() {
    echo "=== TMUX 会话 ==="
    tmux ls 2>/dev/null | grep -E "$SESSION|wbc" || echo "❌ 无训练会话"
    echo ""
    echo "=== GPU 状态 ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | head -10
    echo ""
    echo "=== 最新模型 ==="
    ls -lt "$PROJECT_DIR/unitree_rl_mjlab/logs/rsl_rl/t800_tracking/"*/model_*.pt 2>/dev/null | head -3
}

case "${1:-attach}" in
    start)    start ;;
    train_B)  train_B ;;
    fresh)    fresh ;;
    attach|a) attach ;;
    kill|k)   kill ;;
    list|l)   list ;;
    logs)     logs ;;
    status|s) status ;;
    *)
        echo "用法: $0 {start|train_B|fresh|attach|kill|list|logs|status}"
        echo "  start     全新训练（双卡，无额外奖励）"
        echo "  train_B   GPU$TRAIN_B_GPU 从 checkpoint 继续（加躯干奖励）"
        echo "  fresh     GPU$TRAIN_B_GPU 全新训练（加躯干奖励）"
        echo "  attach/a  连接 tmux 会话"
        echo "  kill/k    终止训练"
        echo "  list/l    列出 tmux 会话"
        echo "  logs      查看最新日志"
        echo "  status/s  查看状态"
        exit 1
        ;;
esac
