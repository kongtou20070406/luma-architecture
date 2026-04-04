"""
PackedPretrainDataset — 序列打包数据集
========================================
把多个短文本紧密打包进一条固定长度的序列，彻底消灭 padding。

打包规则：
  [BOS, text1_tokens, EOS, BOS, text2_tokens, EOS, ...]
  直到剩余空间不足时停止。

labels 处理：
  - 每段文档内：正常 next-token 预测（EOS 作为最后一个目标 token）
  - 跨文档边界：每段文档的第一个 BOS token 的 label 设为 -100
    （原因：该位置的上文来自上一篇文档，跨文档预测没有意义）
  - 填充不足 max_length 时，剩余位置用 pad_token_id，label=-100

兼容性说明：
  - Mamba 层天然无绝对位置编码，打包对它友好
  - 压缩区 / 推理区的 SWA（滑动窗口注意力）是局部注意力，
    跨文档泄漏只在边界 ±window_size 范围内，可以接受
  - 无需传 position_ids 重置（Mamba 是序列依赖，不是位置依赖）
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class PackedPretrainDataset(Dataset):
    """
    Parameters
    ----------
    data_path : str
        jsonl 文件路径，每行 {"text": "..."}
    tokenizer : transformers.PreTrainedTokenizer
    max_length : int
        打包后的最大序列长度（即 context window 大小）
    shuffle_docs : bool
        是否在构建 pack 之前打乱文档顺序（每次 __init__ 时打乱一次）
    min_doc_tokens : int
        过滤掉 token 数少于此阈值的文档（避免噪声极短文本）
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        shuffle_docs: bool = True,
        min_doc_tokens: int = 4,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # 读取并 tokenize 所有文档
        raw = load_dataset("json", data_files=data_path, split="train")
        all_docs: List[List[int]] = []
        for sample in raw:
            text = str(sample.get("text", "")).strip()
            if not text:
                continue
            ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            ).input_ids
            if len(ids) < min_doc_tokens:
                continue
            # 每篇文档: [BOS] + tokens + [EOS]
            doc = [self.bos_id] + ids + [self.eos_id]
            all_docs.append(doc)

        if shuffle_docs:
            random.shuffle(all_docs)

        # 打包：把多个文档拼成 max_length 的序列
        self.packed_sequences: List[Tuple[List[int], List[int]]] = []
        self._pack(all_docs)

    # ------------------------------------------------------------------
    def _pack(self, docs: List[List[int]]) -> None:
        """Greedy first-fit packing."""
        buf_tokens: List[int] = []
        # 记录每篇文档在 buf 中的起始位置（用于标记 label=-100 的边界）
        doc_starts: List[int] = []

        for doc in docs:
            # 如果当前 doc 加进去还能放下，就追加
            if len(buf_tokens) + len(doc) <= self.max_length:
                doc_starts.append(len(buf_tokens))
                buf_tokens.extend(doc)
            else:
                # 当前 buf 凑满（或尽量接近），生成一个样本
                if buf_tokens:
                    self._finalize(buf_tokens, doc_starts)
                # 开新 buf
                # 如果 doc 本身就超过 max_length，截断后单独成一个样本
                doc_starts = [0]
                buf_tokens = doc[: self.max_length]

        # 最后剩余的 buf
        if buf_tokens:
            self._finalize(buf_tokens, doc_starts)

    def _finalize(self, tokens: List[int], doc_starts: List[int]) -> None:
        """把一个打包 buffer 转成 (input_ids, labels)，并 padding 到 max_length。"""
        seq_len = len(tokens)
        pad_len = self.max_length - seq_len

        input_ids = tokens + [self.pad_id] * pad_len
        labels = list(tokens) + [-100] * pad_len

        # 跨文档边界处理：每段文档（第 2 篇起）的 BOS 位置 label=-100
        # 理由：BOS[N] 预测 first_token[N]，但 BOS[N] 的上文是前一篇文档，污染上下文
        for start in doc_starts[1:]:  # 第 0 篇文档的 BOS 不需要屏蔽
            if start < len(tokens):
                labels[start] = -100

        self.packed_sequences.append((input_ids, labels))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.packed_sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.packed_sequences[index]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
