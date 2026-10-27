"""Budgeted adaptive refinement using gsplat 1.5.3 strategy operations.

The strategy subclasses gsplat's gradient-statistics implementation. Keep the
gsplat dependency pinned: the overridden growth hook is a version-specific API.
"""

import torch
from gsplat import DefaultStrategy
from gsplat.strategy.ops import duplicate, reset_opa, split

from .config import RefinementConfig


class BudgetedStrategy(DefaultStrategy):
    def __init__(self, config: RefinementConfig, training_views: int):
        super().__init__(
            prune_opa=config.prune_opacity,
            grow_grad2d=config.grow_gradient,
            grow_scale3d=config.grow_scale_fraction,
            prune_scale3d=config.prune_scale_fraction,
            refine_start_iter=config.start_step,
            refine_stop_iter=config.stop_step,
            refine_every=config.every,
            reset_every=config.reset_every,
            pause_refine_after_reset=min(training_views, config.reset_every - 1),
            absgrad=config.absgrad,
        )
        self.max_gaussians = config.max_gaussians

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        count = len(params["means"])
        available = self.max_gaussians - count
        if available <= 0:
            return 0, 0
        scores = state["grad2d"] / state["count"].clamp_min(1)
        candidates = torch.where(scores > self.grow_grad2d)[0]
        if len(candidates) > available:
            candidates = candidates[torch.topk(scores[candidates], available).indices]
        small = params["scales"].exp().amax(dim=-1) <= self.grow_scale3d * state["scene_scale"]
        grow = torch.zeros(count, dtype=torch.bool, device=scores.device)
        grow[candidates] = True
        clone_mask, split_mask = grow & small, grow & ~small
        clones, splits = int(clone_mask.sum()), int(split_mask.sum())

        # Split first: 2 children replace one parent (+1). Peak allocation is bounded
        # by max_gaussians + splits before the operation removes those parents.
        if splits:
            split(params=params, optimizers=optimizers, state=state, mask=split_mask)
            clone_mask = torch.cat((clone_mask[~split_mask], clone_mask.new_zeros(2 * splits)))
        if clones:
            duplicate(params=params, optimizers=optimizers, state=state, mask=clone_mask)
        return clones, splits

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        # Use gsplat's projection-gradient accumulation and opacity/scale pruning.
        # Explicit reset scheduling also avoids the bitwise-comparison reset bug
        # in gsplat 1.5.3's DefaultStrategy.step_post_backward.
        if step >= self.refine_stop_iter:
            return
        self._update_state(params, state, info, packed=packed)
        ready = (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        )
        if ready:
            self._grow_gs(params, optimizers, state, step)
            self._prune_gs(params, optimizers, state, step)
            if not len(params["means"]):
                raise RuntimeError("Refinement pruned every Gaussian; inspect geometry and losses")
            state["grad2d"].zero_()
            state["count"].zero_()
        if step > 0 and step % self.reset_every == 0:
            reset_opa(params=params, optimizers=optimizers, state=state, value=self.prune_opa * 2)
