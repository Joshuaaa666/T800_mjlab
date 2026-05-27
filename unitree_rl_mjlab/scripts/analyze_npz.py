import numpy as np
import os

def analyze_motion(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    print(f"--- 正在分析动作文件: {os.path.basename(file_path)} ---")
    data = np.load(file_path)
    
    # 打印所有的键值
    print(f"包含键值: {list(data.keys())}")
    
    # 分析根节点高度 (Body Index 0 通常是 Root/Pelvis)
    if 'body_pos_w' in data:
        # body_pos_w 形状通常是 (frames, num_bodies, 3)
        # index 0 是根节点, index 2 是 Z 轴 (高度)
        root_heights = data['body_pos_w'][:, 0, 2]
        
        min_h = np.min(root_heights)
        max_h = np.max(root_heights)
        mean_h = np.mean(root_heights)
        
        print(f"根节点高度范围 (Z-axis): {min_h:.4f}m 到 {max_h:.4f}m")
        print(f"平均高度: {mean_h:.4f}m")
        
        if min_h < 0.4:
            print("结论: 该动作包含【低高度/倒地】状态。")
        else:
            print("结论: 该动作机器人【始终处于站立/高位】。")
    
    # 分析帧数和时长
    if 'joint_pos' in data:
        num_frames = data['joint_pos'].shape[0]
        fps = data['fps'] if 'fps' in data else 50
        duration = num_frames / float(fps)
        print(f"总帧数: {num_frames}, FPS: {fps}, 总时长: {duration:.2f}秒")

    print("-" * 50)

if __name__ == "__main__":
    # 分析负责人说的那个“倒地起身”文件
    path1 = "unitree_rl_mjlab/data/act01_attack_uppercut.npz"
    analyze_motion(path1)
    
    # 分析 G1 的那个 1307 文件进行对比
    path2 = "unitree_rl_mjlab/src/assets/motions/g1/1307.npz"
    analyze_motion(path2)
