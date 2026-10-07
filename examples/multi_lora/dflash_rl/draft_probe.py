"""SGLang ``--forward-hooks`` factory that records the loaded DFlash draft model's parameter dtypes.

The hook manager calls the factory for each model runner after loading and CUDA-graph capture. On TP rank 0 it
finds the draft model (quantized by then under --speculative-draft-model-quantization) and writes the parameter
count per dtype to ``config["path"]``. It registers no hook.
"""

import gc
import json
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist


def record_draft_dtypes(config: dict) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return None
    for obj in gc.get_objects():
        if isinstance(obj, torch.nn.Module) and type(obj).__name__ == "DFlashDraftModel":
            numel = Counter()
            for parameter in obj.parameters():
                numel[str(parameter.dtype)] += parameter.numel()
            Path(config["path"]).write_text(json.dumps(numel))
    return None
