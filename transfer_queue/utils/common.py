# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from contextlib import contextmanager
from typing import Optional

import psutil
import ray
import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

DEFAULT_TORCH_NUM_THREADS = torch.get_num_threads()

# Ensure logger has a handler，handler决定日志输出到那里。
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


"""
创建一个 Ray 放置组（Placement Group），用于控制 Ray actor 在集群中的分布位置。

核心概念：

Bundle：资源单位，这里每个 bundle 占 num_cpus_per_actor 个 CPU
Placement Group：一组 bundle 的集合，Ray 调度器将其作为一个整体来管理
SPREAD 策略：把各个 bundle 分散到不同节点，而非集中在同一台机器
代码流程：

定义单个 bundle 的资源需求（CPU 数量）
创建包含 num_ray_actors 个 bundle 的放置组，策略为 SPREAD
ray.get(placement_group.ready()) 阻塞等待放置组就绪（资源分配完成）
返回放置组对象
为什么用 SPREAD？

在分布式训练中，TransferQueue 的 Controller、StorageUnit 等 actor 需要分散部署：


节点 A          节点 B          节点 C
┌────────┐    ┌────────┐    ┌────────┐
│Actor 0 │    │Actor 1 │    │Actor 2 │
│ 2 CPU  │    │ 2 CPU  │    │ 2 CPU  │
└────────┘    └────────┘    └────────┘
   ↑               ↑               ↑
         SPREAD 策略分散部署
如果全挤在一台机器上，单机故障会导致所有 actor 丢失；分散部署提高可用性和网络带宽利用。

使用方式：


pg = get_placement_group(num_ray_actors=4, num_cpus_per_actor=2)
actor = MyActor.options(
    placement_group=pg,
    placement_group_bundle_index=0  # 指定用哪个 bundle
).remote()
这样 actor 会被调度到放置组预分配好的节点上，避免资源竞争。
"""
def get_placement_group(num_ray_actors: int, num_cpus_per_actor: int = 1):
    """
    Create a placement group with SPREAD strategy for Ray actors.

    Args:
        num_ray_actors (int): Number of Ray actors to create.
        num_cpus_per_actor (int): Number of CPUs to allocate per actor.

    Returns:
        placement_group: The created placement group.
    """
    bundle = {"CPU": num_cpus_per_actor}
    placement_group = ray.util.placement_group([bundle for _ in range(num_ray_actors)], strategy="SPREAD")
    ray.get(placement_group.ready())
    return placement_group


"""
limit_pytorch_auto_parallel_threads — 上下文管理器

问题：PyTorch 的 torch.stack() 等操作会自动多线程并行。在多进程分布式训练中，每个进程都开大量线程会导致线程数爆炸（进程数 × 线程数 >> CPU 核数），反而拖慢速度。

解决方案：临时限制线程数，用完恢复。


调用前：torch.get_num_threads() = 64（PyTorch 默认占满）
    ↓  进入 with 块
设为 16（或物理核数）
    ↓  执行 torch.stack() 等操作
    ↓  退出 with 块
恢复为 DEFAULT_TORCH_NUM_THREADS
线程数选择逻辑：

target_num_threads=None → 自动决定：物理核 ≥ 16 则用 16，否则用物理核数
超过物理核数 → 警告并降为物理核数
为什么用 try/finally？：即使 with 块内抛异常，也能恢复线程数，避免影响后续操作。

典型使用（在 client.py 中）：


with limit_pytorch_auto_parallel_threads(info="stack_data"):
    data = torch.stack(tensors)  # 这一步不会开太多线程

"""
@contextmanager
def limit_pytorch_auto_parallel_threads(target_num_threads: Optional[int] = None, info: str = ""):
    """Prevent PyTorch from overdoing the automatic parallelism during torch.stack() operation"""
    pytorch_current_num_threads = torch.get_num_threads()
    physical_cores = psutil.cpu_count(logical=False)
    pid = os.getpid()
    if target_num_threads is None:
        # auto determine target_num_threads
        if physical_cores >= 16:
            target_num_threads = 16
        else:
            target_num_threads = physical_cores

    if target_num_threads > physical_cores:
        logger.warning(
            f"target_num_threads {target_num_threads} should not exceed total "
            f"physical CPU cores {physical_cores}. Setting to {physical_cores}."
        )
        target_num_threads = physical_cores

    try:
        torch.set_num_threads(target_num_threads)
        logger.debug(
            f"{info} (pid={pid}): torch.get_num_threads() is {pytorch_current_num_threads}, "
            f"setting to {target_num_threads}."
        )
        yield
    finally:
        # Restore the original number of threads
        torch.set_num_threads(DEFAULT_TORCH_NUM_THREADS)
        logger.debug(
            f"{info} (pid={pid}): torch.get_num_threads() is {torch.get_num_threads()}, "
            f"restoring to {DEFAULT_TORCH_NUM_THREADS}."
        )


"""
get_env_bool — 安全读取环境变量布尔值
"""
def get_env_bool(env_key: str, default: bool = False) -> bool:
    """Robustly get a boolean from an environment variable."""
    env_value = os.getenv(env_key)

    if env_value is None:
        return default

    env_value_lower = env_value.strip().lower()

    true_values = {"true", "1", "yes", "y", "on"}
    return env_value_lower in true_values
