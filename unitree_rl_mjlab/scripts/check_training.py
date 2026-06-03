#!/usr/bin/env python3
"""T800 Training Status Checker
Usage:
  python3 scripts/check_training.py                          # latest run
  python3 scripts/check_training.py 2026-06-02_10-28-58      # specific run
"""
import os, sys, glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT = "/home/wangyirong/projects/倒地起身V1/unitree_rl_mjlab"
LOGDIR = f"{PROJECT}/logs/rsl_rl/t800_tracking"

def find_latest():
    runs = sorted(os.listdir(LOGDIR))
    if not runs:
        print("❌ No training runs found")
        sys.exit(1)
    return runs[-1]

def get(tag, ea):
    if tag not in ea.Tags()['scalars']: return None, None
    ev = ea.Scalars(tag)
    vals = [e.value for e in ev]
    return vals, ev[-1].step

def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else find_latest()
    logpath = f"{LOGDIR}/{run_name}"
    if not os.path.exists(logpath):
        print(f"❌ Run not found: {logpath}")
        sys.exit(1)

    ea = EventAccumulator(logpath)
    ea.Reload()
    tags = ea.Tags()['scalars']

    print("=" * 55)
    print(f"  T800 Training Report: {run_name}")
    print("=" * 55)

    # 1. Progress
    rew_vals, step = get('Train/mean_reward', ea)
    ep_vals = None
    if rew_vals:
        ep_vals, _ = get('Train/mean_episode_length', ea)
        hrs = step * 5.09 / 3600 if step else 0
        print(f"\n📍 Progress: {step} iterations ({hrs:.1f}h elapsed)")

    # 2. Reward
    if rew_vals:
        n = len(rew_vals)
        last = np.mean(rew_vals[-10:]) if n >= 10 else rew_vals[-1]
        first = np.mean(rew_vals[:10]) if n >= 10 else rew_vals[0]
        peak = max(rew_vals)
        mid = n // 2
        recent_trend = np.mean(rew_vals[-n//5:]) - np.mean(rew_vals[mid:-n//5]) if n > 20 else 0
        print(f"  Total Reward: {rew_vals[-1]:.2f} (last10={last:.2f}, peak={peak:.2f})")
        print(f"  Trend: {first:.2f} → ... → {last:.2f} ({'+' if recent_trend > 0 else ''}{recent_trend:+.2f})")

    # 3. Episode
    if ep_vals:
        print(f"  Episode Len: {ep_vals[-1]:.0f}/500 (last10={np.mean(ep_vals[-10:]):.0f})")

    # 4. COM (Stage II indicator)
    com_vals, _ = get('Episode_Reward/reward_center_of_mass', ea)
    if com_vals:
        print(f"  COM Reward: {com_vals[-1]:.4f} (last20={np.mean(com_vals[-20:]):.4f})")
        if len(com_vals) > 40:
            trend = np.mean(com_vals[-20:]) - np.mean(com_vals[-40:-20])
            arrow = "⬆️" if trend > 0.005 else ("➡️" if abs(trend) <= 0.005 else "⬇️")
            print(f"  COM Trend: {arrow} ({trend:+.4f})")

    # 5. Terminations
    for t in ['Episode_Termination/time_out', 'Episode_Termination/tracking_failure']:
        v, _ = get(t, ea)
        if v:
            print(f"  {t.split('/')[-1]}: {np.mean(v[-10:]):.2f}%")

    # 6. Tracking errors
    for t in ['Metrics/motion/error_body_pos', 'Metrics/motion/error_body_rot',
              'Metrics/motion/error_joint_pos']:
        v, _ = get(t, ea)
        if v:
            print(f"  {t.split('/')[-1]}: {v[-1]:.4f}")

    # 7. Training health
    std_vals, _ = get('Policy/mean_std', ea)
    ent_vals, _ = get('Loss/entropy', ea)
    val_vals, _ = get('Loss/value', ea)
    if std_vals and ent_vals:
        print(f"\n🧠 Health: std={std_vals[-1]:.3f}, entropy={ent_vals[-1]:.1f}, value_loss={val_vals[-1]:.3f}")
        if ent_vals[-1] < 1.0:
            print("  ⚠️  Entropy very low! Policy may be stuck.")
        if val_vals[-1] > 1.0:
            print("  ⚠️  Value loss high! Critic struggling.")

    # 8. 检查训练阶段 (修复: TensorBoard tag是完整路径, 用模糊匹配)
    print(f"\n📋 Stage: ", end="")
    has_com = any('reward_center_of_mass' in t for t in tags)  # 匹配 Episode_Reward/reward_center_of_mass
    if has_com:
        print(f"II/III (has COM reward)")
    else:
        print(f"I (no COM reward)")

    # 9. Check if improving or plateaued
    if rew_vals and len(rew_vals) > 30:
        last30 = np.mean(rew_vals[-30:])
        prev30 = np.mean(rew_vals[-60:-30]) if len(rew_vals) >= 60 else np.mean(rew_vals[:30])
        delta = last30 - prev30
        print(f"\n💡 Verdict: ", end="")
        if delta > 0.5:
            print(f"Still improving (last30={last30:.1f}, +{delta:.2f}) ✅")
        elif delta > -0.5:
            print(f"Plateaued (last30={last30:.1f}, {delta:+.2f}) ➡️")
            if com_vals and np.mean(com_vals[-20:]) < 0.1:
                print(f"   Consider switching to harder Stage")
        else:
            print(f"Degrading (last30={last30:.1f}, {delta:+.2f}) ⚠️")

    # 10. Latest model files
    models = sorted(glob.glob(f"{logpath}/model_*.pt"))
    if models:
        latest = models[-1]
        size_mb = os.path.getsize(latest) / 1e6
        print(f"\n📦 Latest: {os.path.basename(latest)} ({size_mb:.0f}MB)")

    print("=" * 55)

if __name__ == '__main__':
    main()
