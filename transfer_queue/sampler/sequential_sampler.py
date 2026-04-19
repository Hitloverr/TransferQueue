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


class SequentialSampler(BaseSampler):
    """顺序采样器，用于基本的数据消费模式。

    本采样器实现无放回的顺序采样，从 ready_indexes 列表头部按顺序选择样本。
    它是 TransferQueueController 的默认采样策略，提供简单、确定性的数据消费，
    且开销最小。

    工作原理::

        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 3

        采样结果:
            sampled_indexes  = [0, 1, 2]   ← 取前 3 个
            consumed_indexes = [0, 1, 2]   ← 与采样索引相同

        下一次调用:
            ready_indexes = [3, 4, 5, 6, 7]  ← 已消费的被移除
            sampled_indexes  = [3, 4, 5]
            consumed_indexes = [3, 4, 5]

    特点：
        - **无放回**：每个样本只被消费一次，不会重复
        - **顺序性**：按就绪顺序消费，行为可预测
        - **零额外状态**：不需要维护内部状态，开销最小
        - **确定性**：相同输入始终产生相同输出

    适用场景：
        - 标准监督学习
        - 数据预处理流水线
        - 任何需要有序、可预测数据消费的场景

    默认使用方式::

        # SequentialSampler 是默认采样器
        controller = TransferQueueController.remote()
        # 或显式指定：
        controller = TransferQueueController.remote(sampler=SequentialSampler())

    ---

    Sequential sampler for basic data consumption patterns.

    This sampler implements sequential sampling without replacement, selecting samples
    from the beginning of the ready_indexes list in order. It's the default sampling
    strategy for TransferQueueController and provides simple, deterministic data consumption
    with minimal overhead.

    The sampler is ideal for standard supervised learning scenarios, data preprocessing
    pipelines, and any use case where ordered, predictable data consumption is preferred.
    It ensures each sample is consumed exactly once, maintaining a clean progression through
    the available data.

    This sampler is typically used as the default sampler in TransferQueueController:

    ```python
    # Default usage (SequentialSampler is the default)
    controller = TransferQueueController.remote()
    # or explicitly:
    controller = TransferQueueController.remote(sampler=SequentialSampler)
    ```
    """

    def __init__(
        self,
    ):
        """初始化顺序采样器。

        SequentialSampler 不需要初始化参数，仅维护最小的内部状态
        以保证最佳性能。

        ---

        Initialize the SequentialSampler.

        SequentialSampler requires no initialization parameters and maintains
        minimal internal state for optimal performance.
        """
        super().__init__()

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """从 ready_indexes 头部选择 batch_size 个元素。

        简单地取列表前 N 个元素，实现顺序、无放回的采样。
        采样索引与消费索引相同，即每个被采样的样本立即标记为已消费。

        参数：
            ready_indexes: 可用的样本索引列表。
            batch_size: 要选择的样本数量。如果大于可用样本数，
                则返回所有可用样本。
            *args: 额外位置参数（被忽略）。
            **kwargs: 额外关键字参数（被忽略）。

        返回：
            tuple[list[int], list[int]]: 包含两个相同列表的元组：
                - sampled_indexes: 采样的索引
                - consumed_indexes: 消费的索引（与采样索引相同）

        示例::
            >>> sampler = SequentialSampler()
            >>> ready_indexes = [0, 1, 2, 3, 4, 5]
            >>> sampler.sample(ready_indexes, batch_size=3)
            ([0, 1, 2], [0, 1, 2])

        ---

        Select first batch_size elements from ready_indexes.

        Args:
            ready_indexes: Available sample indices.
            batch_size: Number of samples to select. If larger than available ready samples,
                all available samples will be returned.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple of (sampled_indexes, consumed_indexes), where consumed_indexes = sampled_indexes.
        """
        sampled_indexes = ready_indexes[:batch_size]
        consumed_indexes = sampled_indexes

        return sampled_indexes, consumed_indexes
