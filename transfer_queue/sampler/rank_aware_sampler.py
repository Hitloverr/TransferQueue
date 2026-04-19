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

from typing import Any

from transfer_queue.sampler import BaseSampler


class RankAwareSampler(BaseSampler):
    """分布式训练的 Rank 感知采样器。

    本采样器专为分布式数据并行训练场景设计，每个 rank 独立检索数据。
    它保证同一数据并行组内的所有 rank 接收相同的样本索引。

    为什么需要 Rank 感知？
        在分布式训练中（尤其是混合并行：DP + TP/PP），多个 rank 可能需要处理
        同一批数据的不同模型分片。例如：

        - Rank 0, Rank 1：张量并行的两个 rank，处理同一批数据
        - Rank 2, Rank 3：张量并行的两个 rank，处理另一批数据

        如果每个 rank 独立采样，Rank 0 和 Rank 1 可能拿到不同的数据，导致训练错误。
        RankAwareSampler 确保同组 rank 拿到相同索引。

    工作原理：
        采样器维护内部状态来协调各 rank 的采样：

        - 同一 DP 组内**第一个**调用 :meth:`sample` 的 rank 执行实际采样，
          并将结果缓存供同组其他 rank 使用
        - 同组**后续** rank 直接获取缓存的索引
        - 如果缓存不可用，重新采样并缓存

    状态缓存结构::

        self._states = {
            "partition_id": {
                "task_name": {
                    dp_rank: {
                        batch_index: [sampled_indexes]
                    }
                }
            }
        }

    调用时序图:: 相同batch_index拿到的数据就相同了

        时间轴 ──────────────────────────────────────────────────→

        Rank 0 (dp_rank=0):
            sample(ready_indexes, batch_size=8, dp_rank=0, batch_index=0)
                │
                ├── 检查缓存: 无
                ├── 执行采样: ready_indexes[:8]
                ├── 缓存结果: _states[...][0][0] = [0,1,2,3,4,5,6,7]
                └── 返回: [0,1,2,3,4,5,6,7]

        Rank 1 (dp_rank=0):  ← 同一 dp_rank
            sample(ready_indexes, batch_size=8, dp_rank=0, batch_index=0)
                │
                ├── 检查缓存: 有！
                └── 返回: [0,1,2,3,4,5,6,7]  ← 与 Rank 0 相同

    使用示例::

        from transfer_queue import StreamingDataset, RankAwareSampler

        dataset = StreamingDataset(
            config=config,
            batch_size=8,
            data_fields=["input_ids"],
            partition_id="train",
            task_name="training",
            dp_rank=local_rank,  # 每个 rank 传入自己的 dp_rank
        )

    更多详情请参考：
        [Roadmap] StreamingDataLoader for task-separated RL post-training
        https://github.com/Ascend/TransferQueue/issues/1

    ---

    Rank-aware sampler for distributed training with TransferQueue.

    This sampler is designed for distributed data parallel training scenarios
    where each rank retrieves data independently.

    This sampler guarantees that all ranks within the same data replica group receive
    the same sample indices.

    The sampler maintains inner state to coordinate sampling across ranks:

    - First rank in a data replica group to call :meth:`sample` performs actual sampling from
      ``ready_indexes`` and caches the result for other ranks in the same group
    - Subsequent ranks in the same group retrieve the cached indices.
    - If no cached indices are available, sampling is performed again and cached for others.


    Please refer to our roadmap for more details:
    [Roadmap] StreamingDataLoader for task-separated RL post-training
    https://github.com/Ascend/TransferQueue/issues/1
    """

    def __init__(self):
        """初始化 Rank 感知采样器。

        采样器维护内部状态来协调同一数据并行组内各 rank 的采样。
        该状态追踪哪些样本已被采样以及被获取了多少次。

        ---

        Initialize the RankAwareSampler.

        The sampler maintains internal state to coordinate sampling across ranks
        within the same data replica group. This state tracks which samples have been sampled
        and how many times they have been fetched.
        """

        super().__init__()

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        dp_rank: int,
        batch_index: int,
        task_name: str,
        partition_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """为当前 rank 采样索引，与其他数据并行 rank 协调。

        本方法实现分布式训练的协调采样。同一 dp_rank 组内第一个调用此方法
        的 rank 从 ``ready_indexes`` 执行实际采样并缓存结果。同组后续 rank
        直接获取缓存的索引。

        内部状态结构 (self._states)::

            self._states = {
                "partition_id": {
                    "task_name": {
                        dp_rank: {
                            batch_index: [sampled_indexes]
                        }
                    }
                }
            }

        状态生命周期：
            1. 第一个 rank 从 ``ready_indexes`` 采样，缓存结果供其他 rank 使用
            2. 其他 rank 直接获取缓存的索引

        参数：
            ready_indexes: 全局索引列表，对应样本的所有必需字段已产生，
                且未被对应任务标记为已消费。

            batch_size: 要选择的样本数量。如果大于可用样本数，
                返回空列表。

            dp_rank: 当前 worker 所属的数据并行 rank ID。
                同一 dp_rank 的 rank 接收相同的数据样本。

            batch_index: 当前批次索引，用于追踪消费进度。
            task_name: 任务标识符。
            partition_id: 分区 ID，用于数据管理。
            *args: 额外位置参数（被忽略）。
            **kwargs: 额外关键字参数（被忽略）。

        返回：
            tuple[list[int], list[int]]: 包含两个列表的元组：
                - 采样的全局索引列表。通常长度为 ``batch_size``，
                  样本不足时为空。

                - 要标记为已消费的全局索引列表（排除其他数据并行组
                  未来检索的样本）。

        异常：
            ValueError: 当 ``dp_rank`` 无效时抛出。

        ---

        Sample indices for the current rank, coordinating with other data replica ranks.

        This method implements coordinated sampling for distributed training.
        The first rank in each data replica group to call this method performs actual sampling
        from ``ready_indexes`` and caches the result. Subsequent ranks in the same
        data replica group receive the cached indices directly.

        Internal state structure (self._states):

        .. code-block:: python

            self._states = {
                "partition_id": {
                    "task_name": {
                        dp_rank: {
                            "batch_index": [sampled_indexes]
                        }
                    }
                }
            }

        State lifecycle:
        1. First rank samples from ``ready_indexes``, caches results for other ranks
        2. Other ranks pop and retrieve the cached indices

        Args:
            ready_indexes: List of global indices for which all required fields of the
                corresponding samples have been produced, and the samples are not labeled
                as consumed in the corresponding task.
            batch_size: Number of samples to select. If larger than available
                ready samples, no samples are returned and both lists are empty.
            dp_rank: Data parallel rank ID that this worker belongs to
                The same Ranks receive the same data samples.
            batch_index: Current batch index for tracking consumption progress.
            task_name: Identifier for the task.
            partition_id: Partition ID for data management.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple of two lists:
            - List of sampled global indices. Typically, has length ``batch_size``,
              or empty if samples are insufficient.
            - List of global indices to mark as consumed (excluded from future
              retrieval by other data_replica_groups).

        Raises:
            ValueError: If ``data_replica_rank`` or ``data_replica_world_size`` is invalid.

        """

        if dp_rank < 0:
            raise ValueError(f"dp_rank {dp_rank} must be greater than or equal to 0")

        # 构建缓存
        if partition_id not in self._states:
            self._states[partition_id] = {}

        if task_name not in self._states[partition_id]:
            self._states[partition_id][task_name] = {}

        if dp_rank not in self._states[partition_id][task_name]:
            self._states[partition_id][task_name][dp_rank] = {}

        if batch_index not in self._states[partition_id][task_name][dp_rank]:
            # Select first batch_size indices from ready_indexes
            sampled_indexes = ready_indexes[:batch_size]

            if len(sampled_indexes) < batch_size:
                return [], []

            consumed_indexes = sampled_indexes
            # 保存缓存
            self._states[partition_id][task_name][dp_rank][batch_index] = sampled_indexes
        else:
            # Return the cached indices (identical to what first rank received)
            sampled_indexes = self._states[partition_id][task_name][dp_rank][batch_index]
            consumed_indexes = sampled_indexes

        return sampled_indexes, consumed_indexes
