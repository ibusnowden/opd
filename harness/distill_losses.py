"""Loss pieces the meta-algorithm adds on top of the reference policy-gradient losses.

The meta-algorithm's per-token policy gradient (../meta-algorithm-alpha-lambda.md, §7):

    grad J(theta) = E_{x, y_hat ~ pi_theta(.|x)} [ sum_t A_t * grad log pi_theta(y_hat_t | y_hat_<t) ]
    A_t          = lam * ( log pi_T(y_hat_t | y_hat_<t) - log pi_theta(y_hat_t | y_hat_<t) )   # teacher reverse-KL
                 + (1 - lam) * A^outcome_t                                                      # sequence-level reward

Corners:
  lam = 0                       -> A_t = A^outcome_t                -> RL  (reuse policy_gradients.loss.*)
  lam = 1, pi_T = same-family   -> A_t = log pi_T - log pi_theta    -> OPD
  lam = 1, pi_T = self+answer   -> same, + per-token KL clip         -> OPSD
  lam = 1, pi_T = delta_data, alpha = 0 -> reduces to cross-entropy -> SFT (handled by `sft_ce_loss`)

This module provides:
  * `sft_ce_loss`                  — the SFT corner, written explicitly (NLL on the dataset tokens).
  * `reverse_kl_distill_advantage` — the lam-weighted teacher term, returned as a per-token "advantage".
  * `per_token_kl_clip`            — OPSD-style point-wise KL clamp on the teacher term.
  * `UnifiedTokenLoss`             — combines the teacher term with an outcome `Experience.advantages`
                                     branch and produces the final scalar loss (REINFORCE-style on the
                                     combined advantage; clipping/IS-correction for alpha<1 is TODO).

Reuses `approx_kl` (k3 estimator) and `masked_mean` from policy_gradients.loss verbatim.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._pg import Experience, approx_kl, masked_mean  # noqa: F401  (approx_kl re-exported for callers)


# --- SFT corner --------------------------------------------------------------

def sft_ce_loss(student_logprobs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Plain supervised fine-tuning loss: mean negative log-likelihood over demonstrated tokens.

    `student_logprobs` is (B, S-1) = log pi_theta(target_t | prefix_<t) where the targets ARE the
    demonstration tokens (i.e. the "rollout" is the dataset completion, alpha == 0). This is exactly
    the lam=1, pi_T=delta_data, alpha=0 corner of the meta-algorithm.
    """
    nll = -student_logprobs
    return masked_mean(nll, mask=action_mask, dim=-1).mean(dim=0)


# --- teacher (reverse-KL) term ----------------------------------------------

def per_token_kl_clip(teacher_minus_student: torch.Tensor, clip: float | None) -> torch.Tensor:
    """OPSD-style point-wise clip on the per-token (log pi_T - log pi_theta) term.

    Zhao et al. 2026 cap the per-position divergence contribution so a few stylistic / "pivot"
    tokens don't dominate the update (without it they report collapse within ~100 steps). Here we
    apply a symmetric clamp to the per-token log-ratio; `clip=None` disables it.

    NOTE: the paper clips the *per-vocabulary-entry* KL contribution, which needs the full teacher
    and student distributions at each position, not just the log-prob of the realized token. Doing
    that faithfully requires returning full log-softmax rows from the teacher/student forward passes
    — see ../per-token-kl-pivot-vs-style.md. This token-level clamp is the cheap approximation;
    TODO: add the faithful per-vocab-entry version.
    """
    if clip is None:
        return teacher_minus_student
    return teacher_minus_student.clamp(min=-abs(clip), max=abs(clip))


def prm_importance_weights(
    g: torch.Tensor,                  # (B, S-1) per-token process-importance (answer-info-gain); action tokens only
    action_mask: torch.Tensor,        # (B, S-1)
    *,
    fn: str = "softmax",
    temp: float = 1.0,
    ceiling: float | None = None,
) -> torch.Tensor:
    """Map per-token process-importance g_t to a MASS-PRESERVING per-token weight w_t (PRM-reweighted
    OPSD, prms-as-teachers variant (c)).

    Mass-preserving := mean(w_t) over each sequence's action tokens == 1, so the weight REDISTRIBUTES
    the teacher reverse-KL step toward high-importance (content/pivot) tokens without changing its
    overall magnitude — keeping arm-vs-arm comparisons about *where* the KL mass lands, not *how much*.
    Non-action positions get w=0.

      fn="softmax": w = n_act * softmax(g / temp) over the row's action tokens (the proposal's
                    "softmax-over-importance"). Higher temp -> flatter (-> uniform == plain OPSD).
      fn="linear":  standardize g over action tokens, w = clamp(1 + z, >=0), renormalized to mean 1
                    (the proposal's "linear" reweighting).
    `ceiling` (optional) caps w_t after normalization — itself a soft clip, so leave None for the
    "does reweighting REPLACE the blunt clip?" arm.
    """
    m = action_mask.to(g.dtype)                          # 1.0 on action tokens, 0 elsewhere
    n = m.sum(dim=-1, keepdim=True)                       # (B, 1) action-token count per row
    g = g * m                                            # defensive: ignore non-action positions
    if fn == "softmax":
        logits = (g / max(float(temp), 1e-6)).masked_fill(m == 0, float("-inf"))
        sm = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)   # sums to 1 over action tokens (0 if none)
        w = sm * n                                       # sum_t w = n  ->  mean over action tokens = 1
    elif fn == "linear":
        mean_g = g.sum(dim=-1, keepdim=True) / n.clamp_min(1.0)
        var_g = (((g - mean_g) * m) ** 2).sum(dim=-1, keepdim=True) / n.clamp_min(1.0)
        z = (g - mean_g) / (var_g.sqrt() + 1e-6)
        w = torch.clamp(1.0 + z, min=0.0) * m
        w = w * (n / w.sum(dim=-1, keepdim=True).clamp_min(1e-6))       # renormalize to mean 1
    else:
        raise ValueError(f"prm_weight_fn must be 'softmax' or 'linear', got {fn!r}")
    if ceiling is not None:
        w = w.clamp(max=abs(ceiling))
    return w * m


def reverse_kl_distill_advantage(
    student_logprobs: torch.Tensor,   # (B, S-1) log pi_theta(y_hat_t | y_hat_<t)  -- requires grad
    teacher_logprobs: torch.Tensor,   # (B, S-1) log pi_T(y_hat_t | y_hat_<t)      -- no grad
    action_mask: torch.Tensor,        # (B, S-1)
    *,
    clip: float | None = None,
    prm_weights: torch.Tensor | None = None,   # (B, S-1) per-token PRM reweight (variant c); no grad
) -> torch.Tensor:
    """The per-token teacher term  A^teacher_t = w_t * clip(log pi_T - log pi_theta), detached, masked.

    Returned as a per-token "advantage" so it can be linearly combined with an outcome advantage and
    fed through the same REINFORCE-style estimator. (Treated as a constant w.r.t. theta — the policy
    gradient multiplies it by grad log pi_theta downstream, matching the formula at the top of file.)

    `prm_weights` (PRM-reweighted OPSD, variant c): a per-token multiplicative weight applied AFTER the
    optional blunt clip, redistributing the reverse-KL mass toward causally-important tokens. None = off.
    Clip-then-reweight so the "clip + reweight" arm stacks them and the "reweight only" arm (clip=None)
    isolates whether learned importance can REPLACE the blunt per-token clip.
    """
    adv = (teacher_logprobs.detach() - student_logprobs.detach())
    adv = per_token_kl_clip(adv, clip)
    if prm_weights is not None:
        adv = adv * prm_weights.detach()
    if action_mask is not None:
        adv = adv * action_mask
    return adv


# --- unified combiner --------------------------------------------------------

class UnifiedTokenLoss(nn.Module):
    """Combine the teacher reverse-KL term and the outcome-reward term into one scalar loss.

    A_t = lam * A^teacher_t + (1 - lam) * A^outcome_t   ;   loss = - mean_t [ A_t * log pi_theta_t ]

    * `lam == 0`: pure outcome RL — you should just use the corresponding `policy_gradients.loss`
      class directly (GRPO/RLOO/...); this combiner falls back to plain REINFORCE on the outcome
      advantage, which is only correct for the rloo/reinforce case. (Trainer wires the proper class.)
    * `0 < lam < 1`: "expert RL + OPD" (../expert-rl-plus-opd.md) — both terms live at once.
    * `lam == 1`: OPD / OPSD — outcome branch unused.

    TODO (alpha < 1): off-policy mixing needs an importance-sampling correction
    ratio = exp(log pi_theta - log_probs_old) with PPO-style clipping (`is_clip_lo/hi`); not done here.
    """

    def __init__(self, lam: float, per_token_kl_clip: float | None = None) -> None:
        super().__init__()
        if not (0.0 <= lam <= 1.0):
            raise ValueError("lam must be in [0, 1]")
        self.lam = lam
        self.kl_clip = per_token_kl_clip

    def forward(
        self,
        student_logprobs: torch.Tensor,       # (B, S-1) requires grad
        experience: Experience,               # carries action_mask, advantages (outcome), log_probs_old, ...
        teacher_logprobs: torch.Tensor | None = None,  # (B, S-1) no grad; required if lam > 0
    ) -> torch.Tensor:
        action_mask = experience.action_mask

        if self.lam > 0.0:
            if teacher_logprobs is None:
                raise ValueError("lam > 0 requires teacher_logprobs")
            a_teacher = reverse_kl_distill_advantage(
                student_logprobs, teacher_logprobs, action_mask, clip=self.kl_clip
            )
        else:
            a_teacher = torch.zeros_like(student_logprobs)

        if self.lam < 1.0:
            if experience.advantages is None:
                raise ValueError("lam < 1 requires experience.advantages (the outcome term)")
            a_outcome = experience.advantages
        else:
            a_outcome = torch.zeros_like(student_logprobs)

        a_t = self.lam * a_teacher + (1.0 - self.lam) * a_outcome
        # REINFORCE on the combined advantage. NOTE: at 0 < lam < 1 this combines two terms WITHOUT the
        # policy-ratio clipping that the configured outcome loss (e.g. GRPO) would normally provide; the
        # trainer's `_run_distill_loop` instead uses this class ONLY at lam=1 (pure teacher REINFORCE)
        # and at 0<lam<1 blends `lam * L_teacher_REINFORCE + (1-lam) * L_outcome_clipped` (the proper
        # clipped objective from `policy_gradients.loss`) at the loss level.  This combiner is kept for
        # the OPSD-style clip ablation and as a reference for the unified-advantage form.
        loss = -(a_t.detach() * student_logprobs)
        return masked_mean(loss, mask=action_mask, dim=-1).mean(dim=0)


__all__ = [
    "sft_ce_loss",
    "per_token_kl_clip",
    "prm_importance_weights",
    "reverse_kl_distill_advantage",
    "UnifiedTokenLoss",
    "approx_kl",
    "masked_mean",
]
