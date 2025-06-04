#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import json
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer


def mcq(idx: int) -> None:
    with open("questions.json", "r") as f:
        D = json.load(f)
    display_quiz(D[str(idx)])


def setup() -> torch.device:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device


def make_dataloader(dataset, tokenizer: PreTrainedTokenizer, label_pad_token_id: int = -100, batch_size: int = 1, num_workers: int = 0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn(tokenizer, label_pad_token_id),
        shuffle=True,
    )


def collate_fn(tokenizer: PreTrainedTokenizer, label_pad_token_id: int):

    def wrapped(batch):

        input_ids = []
        attention_mask = []
        labels = []
        for sample in batch:
            input_ids += [sample["best_input_ids"], sample["worst_input_ids"]]
            attention_mask += [sample["best_attention_mask"], sample["worst_attention_mask"]]
            labels += [sample["best_labels"], sample["worst_labels"]]

        max_len = max(len(x) for x in input_ids)
        for idx in range(len(batch) * 2):
            input_ids[idx] = [tokenizer.pad_token_id] * (max_len - len(input_ids[idx])) + input_ids[idx]
            attention_mask[idx] = [0] * (max_len - len(attention_mask[idx])) + attention_mask[idx]
            labels[idx] = [label_pad_token_id] * (max_len - len(labels[idx])) + labels[idx]

        input_ids = torch.as_tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        return input_ids[..., :-1], attention_mask[..., :-1], labels[..., 1:]

    return wrapped


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()