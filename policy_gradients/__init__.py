# Policy Gradient Methods for Language Model Training — vendored into research/
#
# Original implementation by Zafir Stojanovski (@zafstojano) — https://github.com/zafstojano/policy-gradients (Apache 2.0)
# Adapted for RLHF Book (https://rlhfbook.com) by Nathan Lambert.
#
# Vendored (copied) from /project/inniang/vibe/code/policy_gradients/ so the whole post-training
# stack is self-contained under research/. Only change: removed the `mlrunx` dependency — the
# standalone `train.py::main()` is gone (training entry point is research/harness/unified_trainer.py),
# and the `mlrunx_*` config fields are gone (logging is Weights & Biases — research/harness/wandb_logging.py).
# `loss.py`, `buffer.py`, `utils.py` are verbatim; `config.py` and `train.py` carry the trims noted above.

from .buffer import Experience, ReplayBuffer
from .config import Config, load_config
from .loss import CISPOLoss, GRPOLoss, GSPOLoss, PPOLoss, ReinforceLoss


__all__ = [
    "Experience",
    "ReplayBuffer",
    "Config",
    "load_config",
    "GRPOLoss",
    "GSPOLoss",
    "PPOLoss",
    "ReinforceLoss",
    "CISPOLoss",
]