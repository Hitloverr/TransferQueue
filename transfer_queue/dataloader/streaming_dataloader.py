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

"""
StreamingDataLoader 的核心价值
1. 与原生 DataLoader 对比

原生 DataLoader:
  数据在本地磁盘 → Dataset.__getitem__() 逐条读 → collate 凑批 → 返回
  ❌ 不支持流式数据源
  ❌ 不支持分布式 DP 组协调

StreamingDataLoader:
  数据在远程 TransferQueue → StreamingDataset 整批拉 → 直接返回
  ✅ 流式数据源
  ✅ DP 组协调
2. 解决的三个关键问题
问题 1：训练数据是实时产生的


# RLHF / 在线学习场景：Actor 生成数据 → 训练 Reward Model
# 数据不是预先存在磁盘上的，而是持续产生的

# 原生 DataLoader：只能读本地文件，做不到
# StreamingDataLoader：
for batch, meta in dataloader:  # 持续轮询新数据
    train(batch)
问题 2：分布式训练中同一 DP 组需要相同数据


4 块 GPU 做张量并行（TP），2 组数据并行（DP）：

原生 DataLoader:
  每个 rank 独立读文件 → 需要手动协调 → 容易出错

StreamingDataLoader + RankAwareSampler:
  rank 0, rank 1 (DP组0) → 自动拿到相同样本
  rank 2, rank 3 (DP组1) → 自动拿到相同样本
问题 3：GPU 等待数据


原生 DataLoader:
  GPU 空闲 ← 等待磁盘 IO ← 逐条读取 → collate 凑批

StreamingDataLoader:
  DataLoader Worker 1: get() → 预取 batch A
  DataLoader Worker 2: get() → 预取 batch B   ← 并行预取
  GPU: 直接拿预取的 batch A → 训练            ← 无等待
3. 关键设计优势
特性	说明
batch_size=None	批次管理交给 StreamingDataset，不重复做
_identity_collate_fn	数据已经在远端凑好批了，不需要 PyTorch 再 collate
num_workers + prefetch_factor	多 worker 并行预取，GPU 不等数据
reset() / step()	支持多轮训练和分区切换
4. 实际场景

# 场景：RLHF 训练
# Actor 持续生成 rollout 数据 → 写入 TransferQueue
# Critic/Reward Model 从 TransferQueue 流式消费

dataloader = StreamingDataLoader(
    dataset=StreamingDataset(
        partition_id="rollout",
        should_check_consumption_status=False,  # 无限流
        ...
    ),
    num_workers=2,
    prefetch_factor=2,  # 预取 4 个 batch
)

# 训练循环：数据持续产生，训练不停
for batch, meta in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
一句话总结
当你需要从远程、实时、流式的数据源训练模型，尤其是在分布式环境下，原生 DataLoader 搞不定，StreamingDataLoader 是为此设计的解决方案。
"""

import logging
import os
from typing import Optional

import torch
from tensordict import TensorDict

from transfer_queue.dataloader.streaming_dataset import StreamingDataset
from transfer_queue.metadata import BatchMeta

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


def _identity_collate_fn(data: tuple[TensorDict, BatchMeta]) -> tuple[TensorDict, BatchMeta]:
    """恒等 collate 函数，直接返回输入数据。

    本函数作为透传函数，保持 `StreamingDataset` 产出的
    `(TensorDict, BatchMeta)` 结构不变。它防止 PyTorch 尝试
    对已经在远端凑好批的数据进行堆叠或修改。

    数据流::
        StreamingDataset.__iter__() → (TensorDict, BatchMeta)
                                           ↓
                              _identity_collate_fn (透传)
                                           ↓
                                    返回原数据

    为什么需要这个函数？
        原生 PyTorch DataLoader 默认会调用 collate 函数将多条数据
        堆叠成一个批次。但 StreamingDataset 已经从 TransferQueue
        获取了整批数据（TensorDict 本身就是批量的），不需要再次处理。

    Args:
        data: StreamingDataset 产出的 (TensorDict, BatchMeta) 元组。

    Returns:
        与输入完全相同的 (TensorDict, BatchMeta) 元组。

    """
    return data


class StreamingDataLoader(torch.utils.data.DataLoader):
    """TransferQueue 的流式数据加载器。

    本类封装 StreamingDataset，提供 PyTorch DataLoader 接口，
    用于分布式训练场景下的流式数据访问。

    核心价值：
        与原生 DataLoader 相比，StreamingDataLoader 解决了以下关键问题：

        1. **流式数据源支持**：训练数据由远端实时产生（如 RLHF 中的 Actor 生成
           rollout 数据），而非预先存储在本地磁盘。原生 DataLoader 只能读取本地
           文件，而 StreamingDataLoader 可以持续从 TransferQueue 轮询新数据。

        2. **分布式 DP 组协调**：在混合并行场景（DP + TP/PP）下，同一 DP 组
           的多个 rank 必须接收相同的数据样本。StreamingDataLoader 配合
           RankAwareSampler 自动保证这一点，无需手动协调。

        3. **GPU 零等待**：通过 ``num_workers`` 和 ``prefetch_factor`` 实现
           多 worker 并行预取，GPU 直接消费已预取的批次数据，无需等待 IO。

    关键设计决策：
        ============================  ================================================
        设计                          原因
        ============================  ================================================
        ``batch_size=None``           批次管理交给 StreamingDataset，避免重复凑批
        ``_identity_collate_fn``      数据已在远端凑好批，PyTorch 不需要再 collate
        ``num_workers + prefetch``   多 worker 并行预取，GPU 不等数据
        ``reset() / step()``         支持多轮训练和分区切换
        ============================  ================================================

    数据流::

        TransferQueue (远程)
              │
              ▼ StreamingDataset.__iter__() 获取整批数据
        (TensorDict, BatchMeta)
              │
              ▼ _identity_collate_fn (透传)
        (TensorDict, BatchMeta)
              │
              ▼ DataLoader 预取队列 (num_workers × prefetch_factor)
        (TensorDict, BatchMeta)
              │
              ▼ 训练循环 for batch, meta in dataloader

    适用场景：
        - RLHF / 在线学习：Actor 持续生成数据，Critic/Reward Model 流式消费
        - 大规模分布式训练：需要 DP 组协调的混合并行场景
        - 流式推理+训练：数据实时产生，训练不停

    Example:
        RLHF 训练场景::

            dataset = StreamingDataset(
                config=config,
                batch_size=8,
                micro_batch_size=4,
                data_fields=["input_ids", "attention_mask"],
                partition_id="rollout",
                task_name="reward_training",
                dp_rank=0,
                should_check_consumption_status=False,  # 无限流
            )

            dataloader = StreamingDataLoader(
                dataset,
                num_workers=2,       # 2 个 worker 并行取数据
                prefetch_factor=2,   # 每个 worker 预取 2 个 batch
            )

            # 训练循环：数据持续产生，训练不停
            for batch, batch_meta in dataloader:
                loss = model(batch)
                loss.backward()
                optimizer.step()

    ---

    StreamingDataLoader interface for TransferQueue.

    This DataLoader wraps StreamingDataset and provides a PyTorch DataLoader
    interface for distributed training with streaming data access.

    Key Features:
    - Compatible with PyTorch training loops (for loop iteration)
    - Works with StreamingDataset for streaming data access
    - Supports distributed training via RankAwareSampler coordination


    Note:
        This DataLoader is typically used with StreamingDataset which manages
        batch size internally. The standard PyTorch DataLoader batch_size
        parameter is set to None because batching is handled by the dataset
        in coordination with TransferQueue's sampling logic.

    Example:
        >>> dataset = StreamingDataset(
        ...     config=config,
        ...     micro_batch_size=4,
        ...     required_fields=["input_ids", "attention_mask"],
        ...     partition_id="train",
        ...     task_name="update_actor",
        ...     data_replica_group=0,
        ...     data_replica_rank=0,
        ...     data_replica_world_size=1,
        ... )
        >>> dataloader = StreamingDataLoader(dataset, num_workers=0)
        >>> for batch, batch_meta in dataloader:
        ...     # batch: TensorDict with requested fields
        ...     # batch_meta: Metadata for TransferQueue coordination
        ...     loss = model(batch)
        ...     loss.backward()
    """

    def __init__(
        self,
        dataset: StreamingDataset,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        worker_init_fn=None,
        multiprocessing_context=None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        """初始化 StreamingDataLoader。

        参数：
            dataset: StreamingDataset 实例，负责从 TransferQueue 获取数据。
            num_workers: 数据加载子进程数量。每个子进程会创建独立的
                TransferQueue 客户端（通过 ZMQ 直连，fork 安全），实现
                并行数据获取。默认为 0（主进程加载）。
            collate_fn: 样本整理函数。默认为 ``_identity_collate_fn``，
                因为 StreamingDataset 已返回整批数据，不需要 PyTorch
                再次 collate。仅在需要自定义后处理时修改此参数。
            pin_memory: 是否将数据固定在内存中以加速 GPU 传输。
            worker_init_fn: Worker 子进程初始化函数。
            multiprocessing_context: 多进程上下文。
            prefetch_factor: 每个 worker 预取的批次数量。总预取数为
                ``num_workers × prefetch_factor``。例如 ``num_workers=2,
                prefetch_factor=2`` 时共有 4 个批次在预取队列中。
            persistent_workers: 是否在 epoch 之间保持 worker 存活。
            pin_memory_device: 固定内存的目标设备。

        注意：
            本 DataLoader 的 ``batch_size`` 固定为 ``None``，因为批次管理
            由 StreamingDataset 通过 ``micro_batch_size`` 参数内部处理，
            与 RankAwareSampler 协调工作。

        Initialize the StreamingDataLoader.

        Args:
            dataset: StreamingDataset instance.
            num_workers: Number of subprocesses for data loading.
            collate_fn: Function to collate samples into batches.
            pin_memory: If True, pin memory for GPU transfer.
            worker_init_fn: Worker initialization function.
            multiprocessing_context: Multiprocessing context.
            prefetch_factor: Number of batches to prefetch per worker.
            persistent_workers: Keep workers alive between epochs.
            pin_memory_device: Device for pin_memory.

        Note:
            This DataLoader is designed to work with StreamingDataset which handles
            batch size internally via the micro_batch_size parameter. The batch_size
            parameter in PyTorch DataLoader is set to None because batching is managed
            by the StreamingDataset in coordination with RankAwareSampler.
        """
        self.dataset: StreamingDataset = dataset

        if collate_fn is None:
            # use identical collate function to directly return the self-defined
            # [TensorDict, BatchMeta] output of StreamingDataset
            final_collate_fn = _identity_collate_fn
        else:
            final_collate_fn = collate_fn

        super().__init__(
            dataset=dataset,
            batch_size=None,  # Batch size is handled by the dataset
            shuffle=None,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=final_collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            timeout=0,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    def reset(self):
        """重置迭代器到缓存开头。

        将底层 StreamingDataset 的 batch_index 重置为 0，允许
        重新遍历缓存的批次数据，无需从 TransferQueue 重新获取。
        适用于多轮训练场景。

        示例::
            for epoch in range(num_epochs):
                dataloader.reset()
                for batch, meta in dataloader:
                    train(batch)

        ---

        Reset the dataset iterator to the beginning.

        Clears the buffer and resets the batch index for a fresh iteration.
        """
        self.dataset.reset()

    def step(self, partition_id):
        """切换到新分区并重置数据集状态。

        清空缓存、重置批次索引，并更新 partition_id 以从不同分区
        获取数据。用于在训练阶段之间切换（如 "train" → "val"）。

        与 reset() 不同，本方法会清除缓存数据，因为不同分区包含
        不同的样本。

        参数：
            partition_id: 要切换到的新分区 ID（如 "val"、"test"）。

        示例::
            # 训练阶段
            for batch, meta in dataloader:
                train(batch)

            # 切换到验证集
            dataloader.step(partition_id="val")
            for batch, meta in dataloader:
                validate(batch)

        ---

        Switch to a new partition and reset the dataset state.

        This method clears the buffer, resets the batch index, and updates the partition_id
        to fetch data from a different partition (e.g., switching from "train" to "val").

        Args:
            partition_id: The new partition ID to switch to.
        """
        self.dataset.step(partition_id)

    def get_buffer(self):
        """获取底层 dataset 的当前缓存。

        返回 StreamingDataset 维护的批次缓存，该缓存存储了
        已预取的批次数据，用于高效数据访问。

        用途：
            - 调试：检查已获取的数据
            - 分析：统计缓存使用情况
            - 重放：手动遍历缓存数据

        返回：
            list: 包含预取的 (TensorDict, BatchMeta) 元组的缓存列表。

        ---

        Get the current buffer from the underlying dataset.

        Returns the batch buffer maintained by StreamingDataset, which stores
        pre-fetched batches for efficient data access.

        Returns:
            list: Buffer containing pre-fetched (TensorDict, BatchMeta) tuples.
        """
        return self.dataset.buffer
