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

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseSampler(ABC):
    """采样器基类，控制 TransferQueue 的数据消费策略。

    采样器定义了从可用样本中选择哪些样本进行检索的逻辑，以及哪些样本应该被
    标记为已消费（未来不再检索）。基于此抽象，用户可以实现各种数据消费策略，
    以适应不同的训练场景，如顺序采样、强化学习的分组采样或自定义采样模式。

    核心概念：
        - **就绪索引 (ready_indexes)**: 所有必需字段已产生且未被消费的样本索引
        - **采样索引 (sampled_indexes)**: 本次要检索的样本索引
        - **消费索引 (consumed_indexes)**: 本次要标记为已消费的样本索引

    设计理念：
        采样器接口在数据生产状态（由 TransferQueueController 管理）和数据消费策略
        （由采样器实现）之间提供了清晰的分离。这允许用户自定义数据消费行为，
        而无需修改 TransferQueue 的核心代码。

    可用的采样器：
        =====================  =========================================================
        采样器                  描述
        =====================  =========================================================
        SequentialSampler      默认采样器，顺序选择样本，无放回
        GRPOGroupNSampler      分组采样器，仅在连续 N 个样本全部就绪时才进行采样
                              （假设同一 prompt 关联的 N 个样本连续存储）
        RankAwareSampler       分布式训练的 rank 感知采样器，保证同一 DP 组的
                              多个 rank 消费相同的样本
        =====================  =========================================================

    采样流程::

        TransferQueueController
               │
               ▼ 调�用 sampler.sample(ready_indexes, batch_size)
        ┌──────────────────────────────────────┐
        │           BaseSampler                │
        │  输入: ready_indexes, batch_size    │
        │  输出: (sampled_indexes, consumed_indexes) │
        └──────────────────────────────────────┘
               │
               ▼ 返回给 Controller
        Controller 分配 global_indexes 给 Client

    注意：
        sample() 方法始终返回采样索引和消费索引两个列表（可能相同）。

    ---

    Base class for samplers that control how data is consumed from TransferQueue.

    A sampler defines the logic for selecting which samples to retrieve from the
    available samples, and which should be labeled as consumed (will never be retrieved in the future).
    Based on this abstraction, users can implement various data consumption strategies
    for different training scenarios, such as sequential sampling, grouped sampling for
    reinforcement learning, or custom sampling patterns.

    The sampler interface provides a clean separation between data production status
    (handled by TransferQueueController) and data consumption strategy (implemented by samplers).
    This allows users to customize data consumption behavior without modifying the TransferQueue codes.

    Available Samplers:
    - **SequentialSampler**: Default sampler, selects samples sequentially without replacement
    - **GRPOGroupNSampler**: A sampler that performs sampling on continuous N samples only when all of them are ready.
                            It assumes the N samples associated with the same prompt are stored contiguously
    - **RankAwareSampler**: Rank-aware sampling for distributed training where each rank retrieves data independently.
                            This sampler will guarantee ranks of the same DP group consume identical samples.

    NOTE: Always return both sampled and consumed indexes (may be identical).
    """

    def __init__(self):
        """
        partition_id  → 哪个数据分区（train/val）
        task_name → 哪个训练任务（不同任务独立消费）
        dp_rank → 哪个 DP rank（同组共享结果）
        batch_index → 第几个批次（避免批次间混淆）
            核心目的：保证同一 (分区, 任务, rank, 批次) 组合的采样结果确定且可复用。
        """
        self._states: dict[Any, Any] = {}

    @abstractmethod
    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """从就绪索引中采样一批索引。

        参数：
            ready_indexes: 全局索引列表，对应样本的所有必需字段已产生，
                且未被对应任务标记为已消费。
            batch_size: 要选择的样本数量。
            *args: 特定采样器实现的额外位置参数。
            **kwargs: 特定采样器实现的额外关键字参数。

        返回：
            tuple[list[int], list[int]]: 包含两个列表的元组：
                - 第一个列表：采样的全局索引，长度为 batch_size
                - 第二个列表：应标记为已消费的全局索引，长度为 batch_size
                  （未来不再检索）。采样索引和消费索引可能相同。

        异常：
            ValueError: 当 batch_size 无效或 ready_indexes 不足时抛出。

        ---

        Sample a batch of indices from the ready indices.

        Args:
            ready_indexes: List of global indices for which all required fields of the
            corresponding samples have been produced, and the samples are not labeled as
            consumed in the corresponding task.
            batch_size: Number of samples to select
            *args: Additional positional arguments for specific sampler implementations
            **kwargs: Additional keyword arguments for specific sampler implementations

        Returns:
            List of sampled global indices of length batch_size

            List of global indices of length batch_size that should be labeled as consumed
            (will never be retrieved in the future)

        Raises:
            ValueError: If batch_size is invalid or ready_indexes is insufficient
        """
        raise NotImplementedError("Subclasses must implement sample")

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[list[int], list[int]]:
        return self.sample(*args, **kwargs)

    def has_cached_result(
        self,
        partition_id: str,
        task_name: str,
        sampling_config: Optional[dict[str, Any]] = None,
    ) -> bool:
        """检查采样器是否缓存了指定上下文的采样结果。

        用于 Controller 在轮询模式下判断是否已有之前计算的采样结果。
        如果有缓存，Controller 可以跳过等待更多数据，直接返回缓存结果。

        缓存查找基于 ``_states`` 的层级结构：
        ``_states[partition_id][task_name][dp_rank][batch_index]``。

        参数：
            partition_id: 分区标识符。
            task_name: 消费者任务名称。
            sampling_config: 可选的采样配置字典，可能包含用于定位
                缓存结果的 ``dp_rank`` 和 ``batch_index`` 键。

        返回：
            bool: 如果指定 ``(partition_id, task_name, dp_rank, batch_index)``
            组合的缓存结果存在则返回 True，否则返回 False。
            当 ``sampling_config`` 中未提供 ``dp_rank`` 时也返回 False。

        ---

        Check whether the sampler has a cached sampling result for the given context.

        This is used by the controller in polling mode to determine if a previously
        computed sampling result is already available, so that it can skip waiting
        for more data and directly proceed to return the cached result.

        The check is based on the ``_states`` cache structure:
        ``_states[partition_id][task_name][dp_rank][batch_index]``.

        Args:
            partition_id: The partition identifier.
            task_name: The consumer task name.
            sampling_config: Optional sampling configuration dict that may contain
                ``dp_rank`` and ``batch_index`` keys used to locate the cached result.

        Returns:
            True if a cached result exists for the specified
            ``(partition_id, task_name, dp_rank, batch_index)`` combination,
            False otherwise. Also returns False if ``dp_rank`` is not provided
            in ``sampling_config``.
        """
        sampling_config = sampling_config or {}
        dp_rank = sampling_config.get("dp_rank", None)
        batch_index = sampling_config.get("batch_index", None)

        if dp_rank is None:
            return False

        states = self._states.get(partition_id, {}).get(task_name, {})
        return dp_rank in states and batch_index in states[dp_rank]

    def clear_cache(self, partition_id: str):
        """清除指定分区的缓存状态。

        本方法移除包含指定全局索引的缓存采样结果，确保未来的采样操作
        不会引用过时的数据。通常在分区数据被清除时调用。

        参数：
            partition_id: 与任务关联的分区 ID。

        ---

        Clear cached states.

        This method removes any cached sampling results that include the specified
        global indexes, ensuring that future sampling operations do not reference
        stale data.

        Args:
            partition_id: The partition ID associated with the task.
        """
        if partition_id in self._states.keys():
            self._states.pop(partition_id)
