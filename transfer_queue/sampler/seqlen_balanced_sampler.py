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

import heapq
import logging
import os
from typing import Any

from transfer_queue.sampler.grpo_group_n_sampler import GRPOGroupNSampler

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))


class SeqlenBalancedSampler(GRPOGroupNSampler):
    """序列长度均衡采样器，继承自 GRPOGroupNSampler。

    本采样器先使用 GRPO 分组逻辑选择完整的 prompt 组（保证分组完整性），
    然后使用 Karmarkar-Karp 均衡分区算法将选中的样本重新分配到各 DP rank，
    使每个 rank 获得大致相同的 token 总数。

    为什么需要序列长度均衡？
        在分布式训练中，如果按顺序分配样本，各 rank 可能收到长度差异很大的数据：

        不均衡分配（SequentialSampler）::
            Rank 0: [长文本, 长文本, 长文本] → 3000 tokens → 计算耗时 3s
            Rank 1: [短文本, 短文本, 短文本] → 500 tokens  → 计算耗时 0.5s
            → Rank 1 空等 Rank 2.5s，GPU 利用率低

        均衡分配（SeqlenBalancedSampler）::
            Rank 0: [长文本, 短文本, 短文本] → 1700 tokens → 计算耗时 1.7s
            Rank 1: [长文本, 短文本, 短文本] → 1800 tokens → 计算耗时 1.8s
            → rank 间等待时间最小化，GPU 利用率高

    工作流程：
        每个 DP rank 独立调用 ``sample()`` 并传入自己的 ``dp_rank``。
        对于给定 ``(partition_id, task_name, batch_index)`` 的**第一次**调用，
        采样器执行以下步骤：

        1. **全局 GRPO 采样**：调用 ``GRPOGroupNSampler.sample()`` 获取
           ``global_batch_size = batch_size × dp_size`` 的完整 prompt 组。
        2. **查询序列长度**：从分区的 ``custom_meta`` 中获取每个样本的
           ``total_lengths``（在数据插入时填充）。
        3. **Karmarkar-Karp 分区**：运行 KK 算法（``get_seqlen_balanced_partitions``）
           将样本按组级别均衡分配到 ``dp_size`` 个 rank。
        4. **缓存分配结果**：将各 rank 的分配结果缓存，后续调用直接返回。

        流程图::

            第一次调用 (dp_rank=0):
            ┌──────────────────────────────────────────────────┐
            │ Step 1: GRPO 分组采样 (global_batch_size)       │
            │   → 选中完整 prompt 组: [组0, 组1, 组2, 组3]    │
            ├──────────────────────────────────────────────────┤
            │ Step 2: 查询 total_lengths                      │
            │   → 组0: 500 tokens, 组1: 200 tokens,           │
            │     组2: 300 tokens, 组3: 400 tokens             │
            ├──────────────────────────────────────────────────┤
            │ Step 3: Karmarkar-Karp 均衡分区                  │
            │   → Rank 0: [组0, 组1] = 700 tokens              │
            │   → Rank 1: [组2, 组3] = 700 tokens              │
            ├──────────────────────────────────────────────────┤
            │ Step 4: 缓存 + 返回 dp_rank=0 的部分             │
            └──────────────────────────────────────────────────┘
            后续调用 (dp_rank=1, 相同 cache_key):
            → 直接返回缓存的 Rank 1 分配结果

    前置要求：
        - 每个样本的 ``custom_meta`` 必须包含 ``{"total_lengths": <int>}``。
        - Controller 调用采样器时必须在 kwargs 中传入
          ``partition=<DataPartitionStatus>``。
        - 传入的 ``batch_size`` 是**每个 DP rank** 的批次大小；采样器内部
          会乘以 ``dp_size`` 得到全局批次大小用于初始 GRPO 采样。

    ---

    Sequence-length balanced sampler that extends GRPOGroupNSampler.

    This sampler first uses the GRPO group-N logic to select complete prompt
    groups (ensuring group integrity), then redistributes the selected
    samples across DP ranks using Karmarkar-Karp balanced partitioning so
    that each rank receives approximately the same total token count.

    Each DP rank independently calls ``sample()`` with its own ``dp_rank``.
    On the **first** call for a given ``(partition_id, task_name, batch_index)``,
    the sampler:

    1. Delegates to ``GRPOGroupNSampler.sample()`` with the full
       ``global_batch_size`` to select complete prompt groups.
    2. Looks up per-sample ``total_lengths`` from the partition's
       ``custom_meta`` (populated during data insertion).
    3. Runs the Karmarkar-Karp algorithm (``get_seqlen_balanced_partitions``)
       to partition samples across ``dp_size`` ranks.
    4. Caches the per-DP assignments.

    Subsequent calls for the same key return the cached assignment for the
    requested ``dp_rank``.

    Requires:
    - ``custom_meta`` for each sample must contain ``{"total_lengths": <int>}``.
    - The controller must pass ``partition=<DataPartitionStatus>`` in kwargs
      when calling the sampler.
    - ``batch_size`` passed in is the **per-DP** batch size; the sampler
      internally multiplies by ``dp_size`` to get the global batch size for
      the initial GRPO selection.
    """

    def __init__(self, n_samples_per_prompt: int = 1, dp_size: int = 1):
        """初始化序列长度均衡采样器。

        参数：
            n_samples_per_prompt: 每个 prompt 的样本数量（继承自 GRPOGroupNSampler）。
                例如设为 4 表示每个 prompt 生成 4 个响应样本。
            dp_size: 数据并行规模（DP rank 数量）。采样器会按此数量
                将样本均衡分配到各 rank。例如设为 2 表示 2 个 DP rank。

        示例::
            >>> # 4 个样本/prompt，2 个 DP rank
            >>> sampler = SeqlenBalancedSampler(n_samples_per_prompt=4, dp_size=2)

        初始化后会创建 ``_balanced_cache`` 用于缓存均衡分区结果。

        ---

        Initialize the sequence-length balanced sampler.
        """
        super().__init__(n_samples_per_prompt=n_samples_per_prompt)
        if dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {dp_size}")
        self.dp_size = dp_size
        # 缓存: (partition_id, task_name, batch_index) -> list[list[int]]
        self._balanced_cache: dict[tuple, list[list[int]]] = {}

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = "",
        partition_id: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """为指定 DP rank 采样索引，并进行序列长度均衡。

        本方法在 GRPO 分组采样的基础上，使用 Karmarkar-Karp 算法将样本
        按序列长度均衡分配到各 DP rank，使每个 rank 的 token 总数近似相等。

        采样流程（首次调用时）::

            1. 缓存检查 → 如有缓存直接返回
            2. GRPO 全局采样 → batch_size × dp_size 个样本
            3. 查询 total_lengths → 从 custom_meta 获取序列长度
            4. 分组聚合 → 将同 prompt 组的 token 数求和作为组权重
            5. KK 均衡分区 → 按组权重将 prompt 组分配到各 rank
            6. 展开回样本索引 → 组索引 → 样本索引
            7. 缓存所有 rank 的分配结果

        参数：
            ready_indexes: 就绪的全局索引列表。
            batch_size: **每个 DP rank** 请求的批次大小（非全局）。
                采样器内部会乘以 dp_size 得到全局批次大小。
            task_name: 任务标识符。
            partition_id: 分区标识符。
            **kwargs: 必须包含以下参数：
                - dp_rank: 当前 DP rank 编号
                - batch_index: 当前批次索引
                - partition: Controller 传入的 ``DataPartitionStatus`` 对象，
                  用于获取 custom_meta 中的 total_lengths

        返回：
            tuple[list[int], list[int]]: 包含两个列表的元组：
                - sampled_indexes: 当前 dp_rank 分配到的全局索引
                - consumed_indexes: 要标记为已消费的索引（与 sampled_indexes 相同）

        ---

        Sample indices for a specific DP rank with seqlen balancing.

        Args:
            ready_indexes: List of ready global indices.
            batch_size: **Per-DP** batch size requested by this rank.
            task_name: Task identifier.
            partition_id: Partition identifier.
            **kwargs: Must include ``dp_rank``, ``batch_index``, and
                ``partition`` (the ``DataPartitionStatus`` object from the
                controller).

        Returns:
            Tuple of (sampled_indexes, consumed_indexes).
        """
        dp_rank = kwargs.get("dp_rank", 0)
        batch_index = kwargs.get("batch_index", 0)
        partition = kwargs.get("partition", None)

        cache_key = (partition_id, task_name, batch_index)

        if cache_key in self._balanced_cache:
            # Return cached assignment for this dp_rank
            partitions = self._balanced_cache[cache_key]
            if dp_rank < len(partitions):
                sampled = partitions[dp_rank]
            else:
                sampled = []
            return sampled, sampled.copy()

        # --- First call: do global sampling + balancing ---

        # Step 1: Use GRPO logic to select complete groups for the full
        # global batch (batch_size * dp_size).
        global_batch_size = batch_size * self.dp_size
        grpo_sampled, grpo_consumed = super().sample(
            ready_indexes,
            global_batch_size,
            task_name=task_name,
            partition_id=partition_id,
        )

        if not grpo_sampled:
            return [], []

        # Step 2: Get total_lengths from custom_meta
        if partition is None:
            logger.warning(
                "SeqlenBalancedSampler: no partition object provided, falling back to equal-split without balancing."
            )
            # Fallback: equal split
            chunk_size = len(grpo_sampled) // self.dp_size
            partitions = []
            for i in range(self.dp_size):
                start = i * chunk_size
                end = start + chunk_size if i < self.dp_size - 1 else len(grpo_sampled)
                partitions.append(grpo_sampled[start:end])
        else:
            custom_meta = partition.get_custom_meta(grpo_sampled)
            total_lengths = []
            for idx in grpo_sampled:
                meta = custom_meta.get(idx, {})
                tl = meta.get("total_lengths", 1)
                total_lengths.append(tl)

            # Step 3: Karmarkar-Karp balanced partitioning at the GROUP
            # level.  Each prompt group consists of ``n_samples_per_prompt``
            # consecutive samples.  We aggregate their total_lengths into a
            # single group weight so that the KK algorithm keeps groups
            # intact, preserving the invariant that every DP rank receives
            # complete groups (required by pass@k metrics and GRPO
            # advantage normalisation).
            group_size = self.n_samples_per_prompt
            num_groups = len(total_lengths) // group_size
            remainder = len(total_lengths) % group_size

            if num_groups > 0 and remainder == 0:
                # Aggregate per-group total token counts
                group_lengths = [sum(total_lengths[g * group_size : (g + 1) * group_size]) for g in range(num_groups)]
                # Balance groups across DP ranks
                balanced_group_partitions = get_seqlen_balanced_partitions(group_lengths, self.dp_size, equal_size=True)
                # Expand group indices back to sample indices
                partitions = []
                for group_indices in balanced_group_partitions:
                    sample_indices = []
                    for g in group_indices:
                        for s in range(group_size):
                            sample_indices.append(grpo_sampled[g * group_size + s])
                    partitions.append(sample_indices)
            else:
                # Fallback: no valid grouping — balance at sample level
                balanced_partitions = get_seqlen_balanced_partitions(total_lengths, self.dp_size, equal_size=False)
                partitions = [[grpo_sampled[i] for i in part_indices] for part_indices in balanced_partitions]

        # Cache the result
        self._balanced_cache[cache_key] = partitions

        # Populate the inherited _states cache for ALL dp_ranks so that
        # the controller's polling check (which looks at self.sampler._states)
        # works correctly even when ready_indexes < batch_size for later ranks
        # (because earlier ranks already consumed their portion).
        if partition_id not in self._states:
            self._states[partition_id] = {}
        if task_name not in self._states[partition_id]:
            self._states[partition_id][task_name] = {}
        states = self._states[partition_id][task_name]
        for rank_i in range(self.dp_size):
            if rank_i not in states:
                states[rank_i] = {}
            rank_sampled = partitions[rank_i] if rank_i < len(partitions) else []
            states[rank_i][batch_index] = (rank_sampled, rank_sampled.copy())

        # Return this dp_rank's portion
        sampled = partitions[dp_rank] if dp_rank < len(partitions) else []
        # All samples are consumed (without replacement)
        return sampled, sampled.copy()

    def clear_cache(self, partition_id: str):
        """清除指定分区的缓存状态。

        同时清除父类的 _states 缓存和本类的 _balanced_cache 缓存，
        确保未来采样操作不会引用过时的数据。

        参数：
            partition_id: 要清除缓存的分区 ID。

        ---

        Clear cached states for the given partition."""
        super().clear_cache(partition_id)
        keys_to_remove = [k for k in self._balanced_cache if k[0] == partition_id]
        for k in keys_to_remove:
            del self._balanced_cache[k]


# Copied from https://github.com/volcengine/verl/blob/468adf22c43b744348051fccd7a5d830c6c3c36a/verl/utils/seqlen_balancing.py
def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """使用 Karmarkar-Karp 最大差分法将项目均衡分区。

    本算法将一组带权重的项目划分为 k 个分区，使各分区的权重之和
    尽可能接近。基于堆的贪心合并策略，优先合并差异最大的分区对。

    算法原理（以 2 路分区为例）::
        输入: [8, 7, 6, 5, 4]

        Step 1: 排序配对
            (8,7) → 差值 1 → 合并为 {8,7}
            (6,5) → 差值 1 → 合并为 {6,5}
            (4)   → 单独一个

        Step 2: 继续合并差异最大的对
            {8,7} 与 {6,5} → 差值 |15-11|=4
            合并: {8,5} vs {7,6} → 和分别为 13, 13

        结果: 分区0=[8,5]=13, 分区1=[7,6]=13 ✓ 完美均衡

    参数：
        seqlen_list: 要分区的序列长度（或权重）列表。
        k_partitions: 要创建的分区数量。
        equal_size: 如果为 True，强制每个分区包含相同数量的项目
            （要求 ``len(seqlen_list) % k_partitions == 0``）。
            如果为 False，只考虑均衡权重之和，每个分区可以有
            不同数量的项目。

    返回：
        list[list[int]]: k 个分区的列表，每个分区是原始索引的列表。

    参考：
        https://en.wikipedia.org/wiki/Largest_differencing_method

    ---

    Partition items into k groups with balanced sums using the Karmarkar-Karp largest differencing method.

    See: https://en.wikipedia.org/wiki/Largest_differencing_method

    Args:
        seqlen_list: List of sequence lengths (or weights) to partition.
        k_partitions: Number of partitions to create.
        equal_size: If True, enforce that all partitions have exactly the same number of items
            (requires ``len(seqlen_list) % k_partitions == 0``).

    Returns:
        A list of k partitions, where each partition is a list of original indices.
    """

    class Set:
        """带权重的集合，用于追踪项目和累积和以支持分区。"""

        def __init__(self) -> None:
            """带权重的集合，用于追踪项目和累积和以支持分区。

            Attributes:
                sum: 集合中所有项目的权重之和
                items: 项目列表，每个元素为 (原始索引, 权重) 元组

            ---

            A weighted set that tracks items and their cumulative sum for partitioning."""
            self.sum = 0
            self.items: list[tuple[int, int]] = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        """K 路分区状态，用于 Karmarkar-Karp 堆合并过程。

        维护 k 个 Set 集合，支持合并操作以逐步均衡分区。

        ---

        A k-way partition state used in the Karmarkar-Karp heap-based merge process."""

        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq: list[State] = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for _i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """获取序列长度的均衡分区顺序，用于平衡 DP rank 和微批次间的序列长度之和。

    本函数是 Karmarkar-Karp 算法的外层封装，在返回结果前会进行验证和排序，
    确保所有索引都被恰好分配一次。

    参数：
        seqlen_list (list[int]): 每个项目的序列长度（权重）列表。
        k_partitions (int): 要创建的分区数量。
        equal_size (bool):
            - True: 每个分区必须包含相同数量的项目
            - False: 只考虑均衡权重之和，每个分区可以有不同数量的项目

    返回：
        list[list[int]]: k 个分区的列表，每个分区包含原始项目的索引。

    示例::
        >>> lengths = [100, 200, 300, 50, 150, 250]
        >>> partitions = get_seqlen_balanced_partitions(lengths, k_partitions=2, equal_size=True)
        >>> # 结果类似: [[1, 3, 5], [0, 2, 4]]
        >>> # 分区0的 token 总和: 200+50+250=500
        >>> # 分区1的 token 总和: 100+300+150=550

    ---

    get order of seq lengths to make partitions balanced, this is
        used in balancing sum of seqlength across dp ranks and microbatches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions: list[list[int]] = [[] for _ in range(k_partitions)]
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)
