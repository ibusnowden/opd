"""harness — unified (alpha, lambda, pi_T) post-training trainer (scaffold).

Backbone for research proposals #6-#14 (see ../SKILLS.md). Builds on the policy-gradient code
vendored under research/policy_gradients/ (copied from vibe/code/policy_gradients/, mlrunx removed;
see that package's __init__.py) and surfaced via harness._pg — plain eager imports, no sys.path
tricks, no lazy imports; the whole stack lives under research/. Adds an SFT loop, a teacher-logprob
/ reverse-KL term, a teacher registry, memory/fit knobs, and W&B logging.
"""

__all__ = ["config", "teachers", "distill_losses", "wandb_logging", "unified_trainer"]
