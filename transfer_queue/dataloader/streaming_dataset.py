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

"""流式数据集模块，用于分布式训练场景下的 TransferQueue 数据消费。

本模块提供了 StreamingDataset 类，它是 PyTorch IterableDataset 的实现，
可与 TransferQueue 集成，支持流式数据加载和分布式训练。
"""

import logging
import os
import time
import uuid
from typing import Callable, Iterator

from omegaconf import DictConfig
from tensordict import TensorDict
from torch.utils.data import IterableDataset

from transfer_queue.client import TransferQueueClient
from transfer_queue.metadata import BatchMeta

TQ_STREAMING_DATASET_EMPTY_BATCH_SLEEP_INTERVAL = float(
    os.environ.get("TQ_STREAMING_DATASET_EMPTY_BATCH_SLEEP_INTERVAL", 1)
)  # 单位：秒

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


class StreamingDataset(IterableDataset):
    """用于分布式训练的 TransferQueue 流式数据集。

    本数据集与 TransferQueue 集成，为分布式训练场景提供高效的流式数据加载能力。
    它与 ``RankAwareSampler`` 配合使用，确保同一数据并行（DP）组内的所有 rank
    接收到相同的样本。

    核心特性：
        - **流式模式**：无限迭代，适用于在线/持续训练场景
        - **有限模式**：基于消费状态控制的迭代，适用于固定数据集
        - **批次缓存**：缓存已获取的批次，支持重放和多轮训练
        - **微批次支持**：将大批次拆分为微批次，支持梯度累积
        - **Fork 安全客户端**：在子进程中直接创建 ZMQ 客户端，绕过 Ray API

    架构概览::

        ┌─────────────────────────────────────────────────────────────┐
        │                    StreamingDataset                          │
        │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐   │
        │  │   buffer    │  │ batch_index  │  │   _tq_client       │   │
        │  │ (缓存队列)   │  │ (迭代索引)    │  │ (ZMQ直连客户端)    │   │
        │  └─────────────┘  └──────────────┘  └────────────────────┘   │
        └─────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
            TransferQueue Controller     StorageUnit (数据)
            (元数据分配)                  (实际数据)

    两阶段数据获取流程：
        1. **get_meta()**：从 Controller 获取批次元数据（global_indexes）
        2. **get_data()**：根据索引从 StorageUnit 获取实际张量数据

    这种分离设计支持：
        - 部分字段读取（只获取需要的字段）
        - 跨 DP rank 的一致性采样（相同索引 = 相同数据）
        - 高效的元数据追踪和消费状态管理

    示例：
        基础用法::

            from transfer_queue import StreamingDataset, StreamingDataLoader

            dataset = StreamingDataset(
                config=config,
                batch_size=8,
                micro_batch_size=4,
                data_fields=["input_ids", "attention_mask"],
                partition_id="train",
                task_name="training_task",
                dp_rank=0,
                should_check_consumption_status=False,  # 流式模式
            )

            dataloader = StreamingDataLoader(
                dataset,
                num_workers=2,       # 并行数据获取
                prefetch_factor=2,   # 每个 worker 预取 2 个批次
            )

            for batch, batch_meta in dataloader:
                # batch: TensorDict，包含请求的字段
                # batch_meta: BatchMeta，包含 global_indexes 等元数据
                train_step(batch)

        多轮训练（缓存复用）::

            dataset = StreamingDataset(..., should_check_consumption_status=True)

            for epoch in range(num_epochs):
                dataset.reset()  # 复用缓存数据
                for batch, meta in dataset:
                    train(batch)

        切换分区::

            # 训练阶段
            for batch, meta in dataset:
                train(batch)

            # 切换到验证集
            dataset.step(partition_id="val")
            for batch, meta in dataset:
                validate(batch)

    注意：
        当 DataLoader 使用 ``num_workers > 0`` 时，本类直接通过 ZMQ 创建
        TransferQueue 客户端（绕过 Ray API），以确保在 PyTorch DataLoader
        worker 子进程中的 fork 安全性。

    另见：
        - :class:`~transfer_queue.RankAwareSampler`: DP 组协调采样器
        - :class:`~transfer_queue.StreamingDataLoader`: 带预取功能的 DataLoader 封装
        - :class:`~transfer_queue.BatchMeta`: 批次元数据容器
    """

    def __init__(
        self,
        config: DictConfig,
        batch_size: int,
        micro_batch_size: int,
        data_fields: list[str],
        partition_id: str,
        task_name: str,
        dp_rank: int,
        should_check_consumption_status: bool = False,
        fetch_batch_fn: Callable | None = None,
        process_batch_fn: Callable | None = None,
    ):
        """初始化 StreamingDataset。

        参数：
            config: TransferQueue 配置，包含：
                - ``controller.zmq_info``: TransferQueueController 的 ZMQServerInfo
                - ``backend.storage_backend``: 存储后端类型（如 "SimpleStorage"）
                - ``backend.<backend_name>``: 后端特定配置
            batch_size: 每次从 TransferQueue 请求的样本数量。决定单次网络调用
                拉取的数据量。
            micro_batch_size: 每个微批次包含的样本数量。用于梯度累积场景：
                大小为 ``batch_size`` 的批次会被拆分为
                ``ceil(batch_size / micro_batch_size)`` 个微批次。
                例如：``batch_size=8, micro_batch_size=4`` 会产出 2 个微批次，
                每个包含 4 个样本，等效于 2 步梯度累积。
            data_fields: 要从存储中获取的字段名列表。只获取指定的字段，
                减少网络开销。例如：``["input_ids", "attention_mask", "labels"]``
            partition_id: 分区标识符，用于数据版本管理。不同分区可存储不同的
                数据集划分（如 ``"train"``、``"val"``、``"test"``）。
                使用 :meth:`step` 方法切换分区。
            task_name: 训练任务的唯一标识符。Controller 用它追踪哪些样本
                被哪个任务消费，以支持消费状态报告。
            dp_rank: 数据并行 rank/组 ID。具有相同 ``dp_rank`` 的所有 rank
                从 TransferQueue 接收相同的样本。对于混合并行（DP + TP/PP）
                场景至关重要，多个 rank 处理同一个批次的不同模型分片。
            should_check_consumption_status: 控制迭代终止行为。

                - ``False``（默认）：**流式模式** —— 迭代器无限运行，
                  无数据时休眠等待。适用于生产者持续推送数据的在线训练。

                - ``True``：**有限模式** —— 当分区内所有样本消费完毕且
                  缓存批次都已产出时，迭代器终止。适用于固定数据集的离线训练。

            fetch_batch_fn: 自定义批次获取函数。接收参数
                ``(tq_client, data_fields, batch_size, partition_id, task_name,
                sampling_config, batch_index)``，返回 ``(TensorDict | None,
                BatchMeta)``。默认为 :func:`default_fetch_batch_fn`。
            process_batch_fn: 自定义批次处理函数，用于将批次拆分为微批次。
                接收参数 ``(TensorDict, BatchMeta, micro_batch_size)``，
                产出 ``(TensorDict, BatchMeta)`` 元组。
                默认为 :func:`chunk_batch_fn`。

        异常：
            ValueError: 当 ``micro_batch_size < 1``、``data_fields`` 为空
                或 ``dp_rank < 0`` 时抛出。

        示例：
            >>> dataset = StreamingDataset(
            ...     config=config,
            ...     batch_size=16,
            ...     micro_batch_size=4,  # 4 步梯度累积
            ...     data_fields=["input_ids", "labels"],
            ...     partition_id="train",
            ...     task_name="llm_finetune",
            ...     dp_rank=0,
            ... )
        """

        if micro_batch_size < 1:
            raise ValueError(f"micro_batch_size 必须 >= 1，当前值: {micro_batch_size}")

        if len(data_fields) < 1:
            raise ValueError(f"data_fields 必须包含至少一个字段名，当前值: {data_fields}")

        if dp_rank < 0:
            raise ValueError(f"dp_rank 必须 >= 0，当前值: {dp_rank}")

        self.config = config
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.data_fields = data_fields
        self.partition_id = partition_id
        self.task_name = task_name
        self.dp_rank = dp_rank
        self.should_check_consumption_status = should_check_consumption_status
        self.fetch_batch_fn = fetch_batch_fn if fetch_batch_fn else default_fetch_batch_fn
        self.process_batch_fn = process_batch_fn if process_batch_fn else chunk_batch_fn

        # 构建 Controller 采样配置
        self.sampling_config = {
            "dp_rank": self.dp_rank,
            "task_name": self.task_name,
        }

        self._tq_client = None

        # 批次缓存队列，用于：
        # 1) 缓存从 TransferQueue/存储获取的训练批次，便于日志、调试和批次重放
        # 2) 支持多轮训练场景 —— 使用 batch_index 多次遍历缓存批次，
        #    避免重复从远程存储获取，减少网络/存储开销
        # 3) 与 reset()/step() 配合，干净地管理迭代状态，避免丢失未消费的批次
        self.buffer: list[tuple[TensorDict, BatchMeta]] = []
        self.batch_index = 0

        super().__init__()

    def _create_client(self):
        """直接从配置创建 TransferQueue 客户端（Fork 安全）。

        本方法使用 ``self.config`` 中的 ZMQ 地址创建 ``TransferQueueClient``，
        而不调用 ``tq.init()``。这对于在 PyTorch DataLoader worker 子进程中
        的 fork 安全性至关重要。

        为什么不使用 ``tq.init()``？
            ``tq.init()`` 内部调用 Ray API（``ray.get_actor()``、``ray.get()``），
            这些在 fork 的子进程中是**不安全的**。当 PyTorch DataLoader 使用
            ``num_workers > 0`` 启动 worker 时，它使用 ``fork()``，这会复制
            文件描述符和锁状态。Ray 的内部状态在这些 fork 的子进程中会损坏。

        解决方案：
            使用从父进程传入的配置，直接通过 ZMQ 创建客户端。这完全绕过 Ray，
            在 fork 的子进程中是安全的。

        安全性对比：
            =====================  ============  ====================================
            依赖                    Fork 安全?    原因
            =====================  ============  ====================================
            ``self.config``        是            纯 Python 对象，fork 后正常工作
            ``TransferQueueClient`` 是            纯 ZMQ 连接，不依赖 Ray
            ``ray.get_actor()``    否            Ray 内部状态在 fork 后损坏
            ``ray.get()``          否            同上
            =====================  ============  ====================================
        """
        client_id = f"StreamingDataset_{uuid.uuid4().hex[:8]}"

        controller_info = self.config.controller.zmq_info
        storage_backend = self.config.backend.storage_backend
        backend_config = self.config.backend[storage_backend]

        self._tq_client = TransferQueueClient(client_id, controller_info)
        self._tq_client.initialize_storage_manager(manager_type=storage_backend, config=backend_config)

    def __iter__(self) -> Iterator[tuple[TensorDict, BatchMeta]]:
        """迭代数据集，产出 (batch, batch_meta) 元组。

        迭代行为取决于 ``should_check_consumption_status``：

        流式模式（``should_check_consumption_status=False``）：
            迭代器作为无限流无限运行。当没有数据时，休眠
            ``TQ_STREAMING_DATASET_EMPTY_BATCH_SLEEP_INTERVAL`` 秒（默认：1）
            后重试。这是生产者持续推送数据的在线训练的标准模式。

        有限模式（``should_check_consumption_status=True``）：
            当满足以下条件时迭代器终止：(1) 分区内所有样本已消费完毕，
            且 (2) 所有缓存的批次都已产出。适用于固定大小数据集的离线训练。

        迭代流程::

            while (流式模式 OR 还有数据 OR 缓存非空):
                if 缓存有数据:
                    yield from buffer[batch_index]  # 无网络调用
                else:
                    batch = 从 TransferQueue 获取()
                    if batch 为空:
                        sleep(1)  # 轮询模式：等待后重试
                    else:
                        buffer.append(batch)
                        yield batch

        产出：
            tuple[TensorDict, BatchMeta]: 包含以下内容的元组：
                - **TensorDict**: 包含请求的 ``data_fields`` 的批次数据
                - **BatchMeta**: 包含 ``global_indexes``、``partition_ids`` 等元数据

        注意：
            对于需要准确消费状态的流式模式，请将环境变量 ``TQ_PRE_ALLOC_SAMPLE_NUM``
            设置为全局批次大小。这允许消费者在生产者生成所有样本之前就能确定消费状态。
        """
        if self._tq_client is None:
            self._create_client()

        assert self._tq_client is not None, "创建 TransferQueue 客户端失败"

        while (
            not self.should_check_consumption_status  # 流式模式：一直处理，不检查消费状态
            or not self._tq_client.check_consumption_status(self.task_name, self.partition_id)  # 还有数据未消费完
            or self.batch_index <= len(self.buffer) - 1  # 缓存里还有数据
        ):
            try:
                # 从缓存取数据，不走网络
                if self.batch_index <= len(self.buffer) - 1:
                    current_data = self.buffer[self.batch_index]
                    self.batch_index += 1
                    logger.debug(f"StreamDataloader 当前批次索引: {self.batch_index}/{len(self.buffer)}")
                    # 按索引取，可能拆成多个 micro_batch
                    yield from self.process_batch_fn(*current_data, micro_batch_size=self.micro_batch_size)

                else:
                    batch_data, batch_meta = self.fetch_batch_fn(
                        tq_client=self._tq_client,
                        data_fields=self.data_fields,
                        batch_size=self.batch_size,
                        partition_id=self.partition_id,
                        task_name=self.task_name,
                        sampling_config=self.sampling_config,
                        batch_index=self.batch_index,
                    )
                    if batch_data is not None:
                        self.buffer.append((batch_data, batch_meta))
                    else:
                        time.sleep(1)  # 休息一秒，然后继续重试

            except Exception as e:
                logger.error(f"[StreamingDataset]: 数据迭代错误: {e}")
                raise

    def reset(self):
        """将迭代器重置到缓存开头。

        本方法将 ``batch_index`` 重置为 0，允许重新遍历缓存的批次，
        而无需从 TransferQueue 重新获取。缓存被保留以支持对同一数据
        进行多轮训练。

        使用场景：
            多轮训练，每轮遍历相同的缓存样本::

                for epoch in range(num_epochs):
                    dataset.reset()
                    for batch, meta in dataset:
                        train(batch)

        注意：
            本方法只重置索引；缓存数据保留在 ``self.buffer`` 中。
            如需清除缓存数据并获取新样本，请使用 :meth:`step`。
        """
        self.batch_index = 0

    def step(self, partition_id):
        """切换到新分区并重置数据集状态。

        本方法清空缓存、重置批次索引，并更新 ``partition_id`` 以从
        不同分区获取数据。用于在训练阶段之间切换（如 "train" → "val"）。

        与 :meth:`reset` 不同，本方法会清除缓存数据，因为不同分区
        包含不同的样本。

        参数：
            partition_id: 要切换到的新分区 ID（如 "val"、"test"）。

        示例：
            >>> # 训练阶段
            >>> for batch, meta in dataset:
            ...     train(batch)
            >>> # 切换到验证集
            >>> dataset.step(partition_id="val")
            >>> for batch, meta in dataset:
            ...     validate(batch)
        """
        self.buffer = []
        self.batch_index = 0
        self.partition_id = partition_id


def default_fetch_batch_fn(tq_client, data_fields, batch_size, partition_id, task_name, sampling_config, batch_index):
    """从 TransferQueue 获取批次数据（默认实现）。

    本函数实现标准的两阶段数据获取：
        1. **get_meta()**: 从 Controller 请求批次元数据（global_indexes）
        2. **get_data()**: 从 StorageUnit 获取实际张量数据

    这种分离设计支持：
        - 部分字段读取（减少网络开销）
        - 跨 DP rank 的一致性采样
        - 正确的消费追踪

    参数：
        tq_client: 用于数据获取的 TransferQueueClient 实例。
        data_fields: 要获取的字段名列表（如 ["input_ids", "labels"]）。
        batch_size: 请求的样本数量。
        partition_id: 要从中获取数据的分区（如 "train"）。
        task_name: 用于消费追踪的任务标识符。
        sampling_config: 包含 ``dp_rank`` 的采样配置。
        batch_index: 用于进度追踪的当前批次索引。

    返回：
        tuple: 包含 ``(TensorDict | None, BatchMeta)`` 的元组：
            - ``TensorDict``: 获取的数据，无数据时为 ``None``
            - ``BatchMeta``: 包含 ``global_indexes``、``field_names`` 等的元数据

    注意：
        当 Controller 设置 ``polling_mode=True`` 时，如果可用数据不足，
        本函数可能返回 ``(None, BatchMeta(size=0))``。调用者应通过休眠
        和重试来处理这种情况。
    """
    config = {**sampling_config, "batch_index": batch_index, "partition_id": partition_id}
    batch_meta = tq_client.get_meta(
        data_fields=data_fields,
        batch_size=batch_size,
        partition_id=partition_id,
        task_name=task_name,
        sampling_config=config,
    )

    if batch_meta.size == 0:
        logger.debug(
            f"[StreamingDataset]: 收到空批次，等待更多数据... "
            f"请求 batch_size={batch_size}, data_fields={data_fields}, "
            f"partition_id={partition_id}, task_name={task_name}。"
        )
        return None, batch_meta
    else:
        batch = tq_client.get_data(batch_meta)
        return batch, batch_meta


def chunk_batch_fn(td, batch_meta, micro_batch_size=1):
    """将批次拆分为微批次以支持梯度累积（默认实现）。

    本函数将 TensorDict 及其对应的 BatchMeta 拆分为较小的微批次。
    当 ``batch_size`` 大于 ``micro_batch_size`` 时使用，支持内存高效
    的梯度累积。

    拆分策略::

        输入: batch_size=8, micro_batch_size=3

        输出: 3 个微批次
        ├── [0:3] → 3 个样本
        ├── [3:6] → 3 个样本
        └── [6:8] → 2 个样本（余数）

    TensorDict 和 BatchMeta 会被一致地拆分，确保每个微批次都有正确的
    ``global_indexes`` 用于消费追踪。

    参数：
        td: 具有非空 ``batch_size`` 的输入 TensorDict。
        batch_meta: 与 TensorDict 一起拆分的 BatchMeta。
        micro_batch_size: 每个微批次的目标样本数（默认：1）。

    返回：
        list[tuple[TensorDict, BatchMeta]]: (micro_batch, micro_meta) 元组列表。

    异常：
        TypeError: 当 ``td`` 不是 TensorDict 时抛出。
        ValueError: 当 ``micro_batch_size <= 0``、batch_size 为空或
            ``micro_batch_size > 总批次大小`` 时抛出。

    示例：
        >>> td = TensorDict({"input_ids": torch.randn(8, 16)}, batch_size=8)
        >>> meta = BatchMeta(global_indexes=[0, 1, 2, 3, 4, 5, 6, 7], ...)
        >>> chunks = chunk_batch_fn(td, meta, micro_batch_size=4)
        >>> len(chunks)
        2
        >>> chunks[0][0].batch_size
        torch.Size([4])
    """
    if not isinstance(td, TensorDict):
        raise TypeError(f"期望 TensorDict，得到: {type(td).__name__}")

    if not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size 必须为正整数，当前值: {micro_batch_size}")

    if len(td.batch_size) == 0:
        raise ValueError("输入 TensorDict 的 batch_size 不能为空")

    total_size = td.batch_size[0]
    if micro_batch_size > total_size:
        raise ValueError(f"micro_batch_size ({micro_batch_size}) 超过总批次大小 ({total_size})")

    # 计算拆分数量（处理不能整除的情况）
    num_splits = (total_size + micro_batch_size - 1) // micro_batch_size
    splits = []
    batch_meta_list = batch_meta.chunk(num_splits)

    # 拆分 TensorDict 并与对应的元数据块配对
    for i in range(num_splits):
        start = i * micro_batch_size
        end = min(start + micro_batch_size, total_size)
        splits.append((td[start:end], batch_meta_list[i]))

    return splits
