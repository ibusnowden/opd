"""Convenience re-exports from the vendored `policy_gradients` package (under research/).

`policy_gradients/` lives next to `harness/` in `research/`, so this is a plain eager import — no
sys.path tricks, no lazy imports. Everything the harness uses from the reference policy-gradient
code is surfaced here in one place; `harness.*` modules import from `harness._pg`.

(The vendored `train.py` has no module-level `mlrunx` import — that was stripped along with its
standalone `main()`; see `policy_gradients/__init__.py`. So importing it here is safe.)
"""

from __future__ import annotations

from policy_gradients import buffer as pg_buffer
from policy_gradients import config as pg_config
from policy_gradients import loss as pg_loss
from policy_gradients import train as pg_train

# --- losses / KL utils -------------------------------------------------------
approx_kl = pg_loss.approx_kl
masked_mean = pg_loss.masked_mean
GRPOLoss = pg_loss.GRPOLoss
GSPOLoss = pg_loss.GSPOLoss
ReinforceLoss = pg_loss.ReinforceLoss
CISPOLoss = pg_loss.CISPOLoss
PPOLoss = pg_loss.PPOLoss
get_loss_objective = pg_train.get_loss_objective

# --- buffer ------------------------------------------------------------------
Experience = pg_buffer.Experience
ReplayBuffer = pg_buffer.ReplayBuffer
join_experiences_batch = pg_buffer.join_experiences_batch

# --- config ------------------------------------------------------------------
Config = pg_config.Config
DataConfig = pg_config.DataConfig
DatasetSpec = pg_config.DatasetSpec

# --- training helpers (from policy_gradients.train) --------------------------
rollout = pg_train.rollout
compute_log_probs = pg_train.compute_log_probs
compute_values = pg_train.compute_values
compute_rewards = pg_train.compute_rewards
apply_reward_kl = pg_train.apply_reward_kl
compute_advantages = pg_train.compute_advantages
compute_gae = pg_train.compute_gae
compute_standardized_advantages = pg_train.compute_standardized_advantages
compute_nonstandardized_advantages = pg_train.compute_nonstandardized_advantages
compute_loo_advantages = pg_train.compute_loo_advantages
load_model = pg_train.load_model
get_ref_model = pg_train.get_ref_model
get_val_model = pg_train.get_val_model
create_dataset = pg_train.create_dataset
iter_training_batches = pg_train.iter_training_batches
setup_distributed = pg_train.setup_distributed
cleanup_distributed = pg_train.cleanup_distributed
unwrap_model = pg_train.unwrap_model
get_model_device = pg_train.get_model_device
DistEnv = pg_train.DistEnv
distributed_sum = pg_train.distributed_sum
distributed_mean = pg_train.distributed_mean
distributed_max = pg_train.distributed_max
seed_everything = pg_train.seed_everything
get_attn_implementation = pg_train.get_attn_implementation
get_gpu_metrics = pg_train.get_gpu_metrics
filter_numeric_metrics = pg_train.filter_numeric_metrics

__all__ = [
    "pg_buffer", "pg_config", "pg_loss", "pg_train",
    "approx_kl", "masked_mean", "GRPOLoss", "GSPOLoss", "ReinforceLoss", "CISPOLoss", "PPOLoss",
    "get_loss_objective",
    "Experience", "ReplayBuffer", "join_experiences_batch",
    "Config", "DataConfig", "DatasetSpec",
    "rollout", "compute_log_probs", "compute_values", "compute_rewards", "apply_reward_kl",
    "compute_advantages", "compute_gae", "compute_standardized_advantages",
    "compute_nonstandardized_advantages", "compute_loo_advantages",
    "load_model", "get_ref_model", "get_val_model", "create_dataset", "iter_training_batches",
    "setup_distributed", "cleanup_distributed", "unwrap_model", "get_model_device",
    "DistEnv", "distributed_sum", "distributed_mean", "distributed_max",
    "seed_everything", "get_attn_implementation", "get_gpu_metrics", "filter_numeric_metrics",
]
