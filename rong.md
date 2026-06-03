# T800 训练笔记

## 训练管理
```bash
./tmux_train.sh start     # 启动训练（自动用 Docker 双卡）
./tmux_train.sh attach    # 连接看实时输出 (Ctrl+B d 断开)
./tmux_train.sh kill      # 停止训练
./tmux_train.sh status    # 查看状态
```

## 查看进度
```bash
python3 unitree_rl_mjlab/scripts/check_training.py
```

## 查看可视化
```bash
python3 -m tensorboard.main --logdir unitree_rl_mjlab/logs/rsl_rl/t800_tracking/2026-05-29_02-30-53/
# 浏览器: http://localhost:6006
```

## 各阶段训练命令

### Stage I (全新训练, 从随机权重开始)
```bash
WANDB_MODE=disabled python scripts/train.py T800-1307-Stage-I \
  --motion-file data/act01_attack_uppercut.npz \
  --env.scene.num-envs 16384 \
  --env.commands.motion.sampling-mode adaptive \
  --gpu-ids "[0, 1]" \
  --video True --video-interval 10000
```

### Stage II (从 Stage I resume)
```bash
WANDB_MODE=disabled python scripts/train.py T800-1307-Stage-II \
  --motion-file data/act01_attack_uppercut.npz \
  --agent.resume True \
  --agent.load-run <Stage-I的日期文件夹> \
  --agent.load-checkpoint model_xxxxx.pt \
  --env.scene.num-envs 16384 \
  --env.commands.motion.sampling-mode adaptive \
  --gpu-ids "[0, 1]" \
  --video True --video-interval 10000
```

### Stage III (从 Stage II resume)
```bash
WANDB_MODE=disabled python scripts/train.py T800-1307-Stage-III \
  --motion-file data/act01_attack_uppercut.npz \
  --agent.resume True \
  --agent.load-run <Stage-II的日期文件夹> \
  --agent.load-checkpoint <Stage-II的最新model.pt> \
  --env.scene.num-envs 16384 \
  --env.commands.motion.sampling-mode adaptive \
  --gpu-ids "[0, 1]" \
  --video True --video-interval 10000
```

## Play 测试
```bash
# 无头录制视频
python3 scripts/play.py T800-1307-Stage-III \
  --motion-file data/act01_attack_uppercut.npz \
  --checkpoint-file logs/rsl_rl/t800_tracking/<run>/model_xxxxx.pt \
  --video True --video-length 689

# 交互式测试 (需要显示器)
python3 scripts/play_interactive.py T800-1307-Stage-I \
  --motion-file data/act01_attack_uppercut.npz \
  --viewer native \
  --checkpoint-file logs/rsl_rl/t800_tracking/<run>/model_xxxxx.pt \
  --push-strength 10.0
```

## 三阶段课程设计

| 阶段 | 终止条件 | 特殊事件 | 奖励 |
|------|---------|---------|------|
| **Stage I** | 宽松: anchor_z=0.8, ori=1.2, ee_z=0.8；容错8s | 域随机化小扰动、GRSI、随机推力 | 基础跟踪奖励+起身奖励 |
| **Stage II** | 收紧: ori=0.6；移除z条件；加hip_dof限制 | 同Stage I | +COM reward (1.0) |
| **Stage III** | 同Stage II | +随机地形、更强推力、更强速度扰动 | 同Stage II |

## 起身奖励 (Recovery Rewards)
```python
penalty_relative_shoulder_high      weight=-2.0   # 肩膀高度偏差惩罚
penalty_xy_rate_before_stand        weight=-1.0   # 倒地时原地起身不横移
penalty_relative_root_orientation   weight=-0.5   # 躯干朝向偏差
reward_center_of_mass               weight=1.0    # 重心控制 (Stage II+)
```

## GRSI (倒地姿态数据集)
- 文件: `data/t800_init_states_8192.pth`
- 来源: `data/robot_data.csv` (4096种倒地姿态，通过 GRSI init_file 加载)
- 转换脚本: `scripts/csv_to_gksi.py`
- 配置: `tracking_standing_env_cfg.py` → `init_pos_file`

## 关键参考
- 论文: A Kung Fu Athlete Bot (arXiv 2602.13656)
- 官方代码: `/home/wangyirong/projects/KungFuAthleteBot/`
- Motion文件: `data/act01_attack_uppercut.npz` (13.8s, 站立上勾拳)

## 打包本地测试
```bash
cd /home/wangyirong/projects/倒地起身V1 && \
tar -czvf t800_stage3_play.tar.gz \
  unitree_rl_mjlab/logs/rsl_rl/t800_tracking/<run>/events.out.tfevents.* \
  unitree_rl_mjlab/logs/rsl_rl/t800_tracking/<run>/params/ \
  unitree_rl_mjlab/logs/rsl_rl/t800_tracking/<run>/model_xxxxx.pt \
  unitree_rl_mjlab/logs/rsl_rl/t800_tracking/<run>/policy.onnx \
  unitree_rl_mjlab/data/act01_attack_uppercut.npz

scp wangyirong@xtellar-6U-GPU-Server:/home/wangyirong/projects/倒地起身V1/t800_stage3_play.tar.gz ./
```