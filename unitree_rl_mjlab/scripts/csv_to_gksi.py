#!/usr/bin/env python3
"""Convert T800 falling CSV to GRSI .pth format (matching robot_init_states_8192.pth)"""
import csv
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

INPUT = 'data/robot_data.csv'
OUTPUT = 'data/t800_init_states_8192.pth'

with open(INPUT) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'CSV rows: {len(rows)}')

# Joint columns: J00_HIP_PITCH_L ... J24_HEAD_YAW
joint_cols = [c for c in rows[0].keys() if c.startswith('J')]
n_joints = len(joint_cols)
print(f'Joints: {n_joints}')
print(f'Joint names: {joint_cols}')

n = len(rows)
dof_pos = np.zeros((n, n_joints), dtype=np.float32)
root_states_xyzw = np.zeros((n, 13), dtype=np.float32)

for i, row in enumerate(rows):
    # Joint positions
    for j, col in enumerate(joint_cols):
        dof_pos[i, j] = float(row[col])
    
    # Root position (x, y, z)
    root_states_xyzw[i, 0] = float(row['base_x'])
    root_states_xyzw[i, 1] = float(row['base_y'])
    root_states_xyzw[i, 2] = float(row['base_z'])
    
    # Euler angles (roll, pitch, yaw) → quaternion (xyzw)
    roll = float(row['roll'])
    pitch = float(row['pitch'])
    yaw = float(row['yaw'])
    r = R.from_euler('xyz', [roll, pitch, yaw])
    root_states_xyzw[i, 3:7] = r.as_quat()  # returns xyzw
    
    # Velocities: not in CSV, set to zero
    root_states_xyzw[i, 7:13] = 0.0

# Also create wxyz version
root_states_wxyz = root_states_xyzw.copy()
root_states_wxyz[:, 3:7] = root_states_xyzw[:, [3, 4, 5, 6]]  # same ordering for wxyz

# Verify
print(f'\nVerification:')
print(f'  base_z: {root_states_xyzw[:,2].min():.3f} ~ {root_states_xyzw[:,2].max():.3f} m')
print(f'  dof_pos[0,0]: {dof_pos[0,0]:.4f}')
print(f'  quat[0]: {root_states_xyzw[0,3:7].tolist()}')
print(f'  unique (ep, robot) pairs: {len(set((r["episode_idx"], r["robot_id"]) for r in rows))}')

data = {
    'num_envs': n,
    'dof_pos': torch.from_numpy(dof_pos),
    'robot_root_states_wxyz': torch.from_numpy(root_states_wxyz),
    'robot_root_states_xyzw': torch.from_numpy(root_states_xyzw),
}

torch.save(data, OUTPUT)
print(f'\n✅ Saved: {OUTPUT}')
print(f'   dof_pos: {dof_pos.shape}')
print(f'   root_states: {root_states_xyzw.shape}')
