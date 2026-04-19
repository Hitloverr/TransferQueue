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


class GRPOGroupNSampler(BaseSampler):
    """分组采样器，用于强化学习和多样本生成工作流。

    本采样器实现无放回的分组采样，专为需要从同一输入 prompt 生成多个样本
    或需要分组采样的场景设计。它确保属于同一 prompt 的所有样本要么一起被
    选中，要么全部不被选中，从而在整个训练过程中保持 prompt 分组的完整性。

    典型应用场景 —— GRPO (Group Relative Policy Optimization)：
        在 GRPO 训练中，需要从同一个 prompt 生成多个响应，然后基于所有响应
        计算相对奖励来训练策略。这要求同一 prompt 的所有响应必须作为一个整体
        被采样和处理。

    工作原理::

        假设 n_samples_per_prompt = 3，每个 prompt 生成 3 个样本

        数据组织（连续存储）:
            ready_indexes = [0, 1, 2,    ← prompt A 的 3 个样本
                            3, 4, 5,    ← prompt B 的 3 个样本
                            6, 7,       ← prompt C 的 2 个样本（不完整）
                            9, 10, 11]  ← prompt D 的 3 个样本

        采样 batch_size = 6:

            分组检测:
                [0, 1, 2] → 连续 ✓ → 完整组
                [3, 4, 5] → 连续 ✓ → 完整组
                [6, 7, 9] → 不连续 ✗ → 跳过
                [9, 10, 11] → 连续 ✓ → 完整组

            结果:
                sampled_indexes  = [0, 1, 2, 3, 4, 5]  ← 取前 2 个完整组
                consumed_indexes = [0, 1, 2, 3, 4, 5]

    核心逻辑：
        1. 对 ready_indexes 排序
        2. 扫描寻找连续的 N 个索引（完整组）
        3. 只返回完整组，不完整组被跳过
        4. 如果完整组数量不足，返回空列表

    数据组织要求：
        用户必须将同一 prompt 的多个样本连续存储。例如：
        ``[prompt1_sample1, prompt1_sample2, prompt2_sample1, prompt2_sample2, ...]``

        即 ready_indexes 应类似：
        ``[prompt1_sample1, prompt1_sample2, prompt1_sample3, prompt1_sample4,
          prompt2_sample1, prompt2_sample2, prompt2_sample3, prompt2_sample4, ...]``

    使用示例::

        from transfer_queue import TransferQueueController, GRPOGroupNSampler

        # 初始化：每个 prompt 生成 4 个样本
        controller = TransferQueueController.remote(
            sampler=GRPOGroupNSampler(n_samples_per_prompt=4)
        )

        # 获取元数据
        meta = await client.async_get_meta(
            data_fields=["input_ids", "generated_text", "reward"],
            batch_size=16,  # 16 个样本 = 4 个 prompt × 4 个样本/prompt
            partition_id="train",
            task_name="grpo_training",
        )

    ---

    Group-based sampler for reinforcement learning and multi-sample generation workflows.

    This sampler implements grouped sampling without replacement, specifically designed
    for scenarios where multiple samples need to be generated from the same input prompt
    or where grouped sampling is required. It ensures that all samples belonging to the
    same prompt are either selected together or not at all, maintaining the integrity
    of prompt groups throughout the training process.

    The sampler is commonly used in GRPO (Group Relative Policy Optimization)
    training scenarios where you need to generate multiple responses from the same
    prompt and train the policy on all of them together.

    The sampler is configured through TransferQueueController and receives parameters
    via the sampling_config in get_meta calls:

    ```python
    # Initialize controller with GRPO sampler
    from transfer_queue import TransferQueueController, GRPOGroupNSampler, AsyncTransferQueueClient

    controller = TransferQueueController.remote(sampler=GRPOGroupNSampler(n_samples_per_prompt=4))
    controller_info = process_zmq_server_info(controller)

    client = AsyncTransferQueueClient(
        client_id="rl_client",
        controller_info=controller_info,
    )

    # Get metadata with grouped sampling configuration
    meta = await client.async_get_meta(
        data_fields=["input_ids", "attention_mask", "generated_text", "reward"],
        batch_size=16,  # Total samples requested
        partition_id="train_0",
        task_name="rl_training",
    )
    # This will return 16 samples organized as 4 groups of 4 samples each
    ```

    Data Organization:
    This sampler assumes the user puts the prompts in consecutive orders, such as
    [prompt1_sample1, prompt1_sample2, prompt2_sample1, prompt2_sample2, ...]
    belong to the same prompt group:
    ```
    ready_indexes = [prompt1_sample1, prompt1_sample2, prompt1_sample3, prompt1_sample4,
                    prompt2_sample1, prompt2_sample2, prompt2_sample3, prompt2_sample4, ...]
    ```
    """

    def __init__(
        self,
        n_samples_per_prompt: int = 1,
    ):
        """初始化 GRPO 分组采样器。

        参数：
            n_samples_per_prompt: 每个 prompt 的样本数量。必须大于 0。
                例如设为 4 表示每个 prompt 生成 4 个响应样本，
                采样时会确保这 4 个样本要么全部选中，要么全部不选。

        示例::
            >>> # 每个 prompt 生成 8 个响应
            >>> sampler = GRPOGroupNSampler(n_samples_per_prompt=8)

        ---

        Initialize the GRPOGroupNSampler.

        The sampler maintains minimal internal state and relies on runtime
        configuration through the sampling_config parameter.
        Args:
            n_samples_per_prompt: Number of samples per prompt group. Must be > 0.

        """
        super().__init__()

        # Basic validation
        if n_samples_per_prompt <= 0:
            raise ValueError(f"n_samples_per_prompt must be positive, got {n_samples_per_prompt}")
        self.n_samples_per_prompt = n_samples_per_prompt

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = "",
        partition_id: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """从就绪索引中按分组方式采样。

        本方法实现分组完整性验证，确保只有完整的组才会被采样。
        如果完整组数量不足，返回空列表。

        采样流程::

            1. 检查缓存：如果已有该 (partition_id, task_name, dp_rank, batch_index) 的结果，直接返回
            2. 参数校验：batch_size 必须是 n_samples_per_prompt 的倍数
            3. 排序扫描：对 ready_indexes 排序后扫描寻找连续组
            4. 完整性检查：检查 N 个索引是否连续（差值均为 1）
            5. 结果缓存：将采样结果缓存到 _states 中，确保分布式训练的一致性

        参数：
            ready_indexes: 所有必需字段已产生且未被消费的样本全局索引列表。
                这些索引应按 prompt 分组连续组织。
            batch_size: 要选择的样本总数。必须是 n_samples_per_prompt 的倍数。
                例如 n_samples_per_prompt=4，batch_size=16 会选择 4 个完整组。
            task_name: 训练任务的唯一标识符，用于状态缓存和追踪消费样本。
            partition_id: 分区 ID，用于数据版本管理和状态组织。
            *args: 额外位置参数（当前实现中被忽略）。
            **kwargs: 额外关键字参数，关键参数包括：
                - dp_rank: 数据并行 rank，用于多 GPU 训练的状态缓存组织。
                - batch_index: 当前批次索引，用于追踪消费进度。

        返回：
            tuple[list[int], list[int]]: 包含两个列表的元组：
                - sampled_indexes: 选中的全局索引列表，长度为 batch_size；
                  如果完整组不足则返回空列表。
                - consumed_indexes: 要标记为已消费的索引列表，与 sampled_indexes 相同
                  （无放回语义）。

        异常：
            ValueError: 当 batch_size 不是 n_samples_per_prompt 的倍数时抛出。

        示例::
            >>> sampler = GRPOGroupNSampler(n_samples_per_prompt=3)
            >>> # 无完整组的情况
            >>> ready_indexes = [0, 1, 3, 4, 6, 7]
            >>> sampled, consumed = sampler.sample(ready_indexes, 6)
            >>> sampled
            []

            >>> # 有完整组的情况
            >>> ready_indexes = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]
            >>> sampled, consumed = sampler.sample(ready_indexes, 6)
            >>> sampled
            [3, 4, 5, 9, 10, 11]  # 两个完整组

        ---

        Sample groups of indices from the ready indices.

        This method implements group completeness validation and ensures that only complete
        groups are sampled. It returns empty lists if insufficient complete groups are available.

        Args:
            ready_indexes: List of global indices for which all required fields have been
                produced and samples are not labeled as consumed. These should be organized
                such that consecutive indices belong to the same prompt group.
            batch_size: Total number of samples to select. Must be divisible by n_samples_per_prompt.
            task_name: Unique identifier for the training task. Used for state caching and
                tracking consumed samples.
            partition_id: Partition ID for data versioning. Used for state organization.
            *args: Additional positional arguments (ignored in current implementation).
            **kwargs: Additional keyword arguments, key ones are:
                - dp_rank: Data parallel rank for multi-GPU training. Used for state cache organization.
                - batch_index: Current batch index for tracking consumption progress.

        Returns:
            Tuple of (sampled_indexes, consumed_indexes):
            - sampled_indexes: List of selected global indices, length = batch_size or empty if
              insufficient complete groups are available.
            - consumed_indexes: List of indices to mark as consumed, identical to sampled_indexes
              (without replacement semantics).

        Raises:
            ValueError: batch_size is not divisible by n_samples_per_prompt.

        Examples:
            >>> sampler = GRPOGroupNSampler(n_samples_per_prompt=3)
            >>> ready_indexes = [0, 1, 3, 4, 6, 7]  # No complete groups after sorting
            >>> sampled, consumed = sampler.sample(ready_indexes, 6)
            >>> sampled
            []
            >>> consumed
            []

            >>> ready_indexes = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]  # Has complete groups after sorting
            >>> sampled, consumed = sampler.sample(ready_indexes, 6)
            >>> sampled
            [3, 4, 5, 9, 10, 11]
            >>> consumed
            [3, 4, 5, 9, 10, 11]
        """
        states = self._states.get(partition_id, {}).get(task_name, {})
        dp_rank = kwargs.get("dp_rank", None)
        batch_index = kwargs.get("batch_index", None)

        # Return cached result if available
        if dp_rank in states.keys() and batch_index in states[dp_rank].keys():
            return states[dp_rank][batch_index]

        if batch_size % self.n_samples_per_prompt != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of n_samples_per_prompt ({self.n_samples_per_prompt})"
            )

        required_groups = batch_size // self.n_samples_per_prompt
        sorted_ready_indexes = sorted(ready_indexes)

        complete_group_indices = []
        found_groups = 0

        # Scan for consecutive groups
        i = 0
        while i <= len(sorted_ready_indexes) - self.n_samples_per_prompt and found_groups < required_groups:
            potential_group = sorted_ready_indexes[i : i + self.n_samples_per_prompt]
            # Check if this forms a complete group (consecutive indices)
            is_consecutive = all(
                potential_group[j + 1] - potential_group[j] == 1 for j in range(len(potential_group) - 1)
            )
            if is_consecutive:
                complete_group_indices.extend(potential_group)
                found_groups += 1
                i += self.n_samples_per_prompt
            else:
                i += 1

        # Return empty if insufficient complete groups
        if found_groups < required_groups:
            return [], []

        sampled_indexes = complete_group_indices
        consumed_indexes = sampled_indexes.copy()

        # Cache the sampling result for deterministic future calls
        if dp_rank is not None:
            if dp_rank not in states:
                states[dp_rank] = {}
                states[dp_rank][batch_index] = (sampled_indexes, consumed_indexes)
            elif batch_index not in states[dp_rank]:
                states[dp_rank][batch_index] = (sampled_indexes, consumed_indexes)
            if partition_id not in self._states:
                self._states[partition_id] = {}
            self._states[partition_id][task_name] = states  # 每个分区，每个任务： dp_rank, batch_index
        return sampled_indexes, consumed_indexes
