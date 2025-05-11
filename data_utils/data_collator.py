import copy
from dataclasses import dataclass, field, fields, asdict
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import sys
import torch

import transformers
import gc

from PIL import Image
import numpy as np
import os

def PIOCollator(features):
    import torch

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    tasks = []
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None:
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
                batch[k] = batch[k].to(dtype=torch.bfloat16)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(np.stack([f[k] for f in features]))
            elif isinstance(v, str):
                tasks = [each[k] for each in features]
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    batch["task"] = tasks
    return batch
