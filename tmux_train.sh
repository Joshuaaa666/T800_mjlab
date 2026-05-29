#!/bin/bash
# T800 训练 TMUX 管理脚本 (宿主机版)
# 用法:
#   ./tmux_train.sh [start|attach|kill|list|logs|status]

SESSION="wbc"
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
               --video True --video-interval 10000
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
    attach|a) attach ;;
    kill|k)   kill ;;
    list|l)   list ;;
    logs)     logs ;;
    status|s) status ;;
    *)
        echo "用法: $0 {start|attach|kill|list|logs|status}"
        exit 1
        ;;
esac
