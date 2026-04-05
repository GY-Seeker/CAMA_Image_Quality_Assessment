from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
from ...extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


def compute_accuracy(eval_preds: "EvalPrediction") -> Dict[str, float]:
    preds, labels = eval_preds.predictions, eval_preds.label_ids
    if preds.shape != labels.shape:
        return {}
    accuracies = []
    for i in range(len(preds)):
        pred, label = preds[i, :-1], labels[i, 1:]
        label_mask = label != IGNORE_INDEX
        accuracies.append(np.mean(pred[label_mask] == label[label_mask]))

    return {"accuracy": float(np.mean(accuracies))}


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)