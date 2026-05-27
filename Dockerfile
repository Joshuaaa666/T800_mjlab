# =============================================================================
# 基于 mjlab 预构建镜像，只需添加自定义代码
# ghcr.io/mujocolab/mjlab 已包含 CUDA、Python、mjlab 及全部系统依赖
# =============================================================================

FROM ghcr.io/mujocolab/mjlab

LABEL description="KungfuAthlete T800 — custom robot on top of mjlab"

WORKDIR /workspace

COPY unitree_rl_mjlab/ /workspace/unitree_rl_mjlab/
# data/ 实际在 unitree_rl_mjlab/data/ 下，映射到 /workspace/data/ 方便训练命令引用
COPY unitree_rl_mjlab/data/ /workspace/data/

# 基础镜像的 venv 路径（含 mjlab、tyro 等）
ENV PATH="/app/.venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV PYTHONPATH="/workspace/unitree_rl_mjlab"

# 升级到项目需要的 mjlab 版本
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --python /app/.venv/bin/python --no-cache mjlab==1.2.0 \
    && python -c "import mjlab; print('mjlab upgraded OK')"

# 锁定 warp-lang 到兼容版本
RUN uv pip install --python /app/.venv/bin/python --no-cache "warp-lang<1.13"

CMD ["bash"]
