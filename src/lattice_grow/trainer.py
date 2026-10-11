"""Training loop: integrates working memory, router, growth controller, and cache."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cache import ExpertCache
from .compression import CompressedExpert
from .growth_controller import GrowthController
from .router import HierarchicalRouter
from .working_memory import WorkingMemory


@dataclass
class TrainStepMetrics:
    """Per-step training metrics."""

    loss: float
    routing_confidence: float
    experts_active: int
    tokens_processed: int
    step_time_ms: float


class LatticeGrowTrainer:
    """End-to-end trainer for the Lattice-Grow architecture.

    Coordinates:
      - WorkingMemory (frozen base + LoRA) for context encoding
      - HierarchicalRouter for expert selection
      - ExpertCache for VRAM management
      - GrowthController for lifecycle decisions
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        router: HierarchicalRouter,
        growth_controller: GrowthController,
        expert_cache: ExpertCache,
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 3e-4,
        ewc_lambda: float = 5000.0,
        expert_output_dim: int = 768,
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        lr_warmup_steps: int = 100,
        lr_total_steps: int = 10000,
    ):
        self.wm = working_memory.to(device)
        self.router = router.to(device)
        self.gc = growth_controller
        self.cache = expert_cache
        self.device = device
        self.expert_output_dim = expert_output_dim
        self.grad_accum_steps = max(1, grad_accum_steps)

        # Freeze WM base, train only LoRA + router + lm_head
        self.wm.freeze_base()
        self.wm.unfreeze_lora()
        # LM head must be trainable for text generation
        for p in self.wm.lm_head.parameters():
            p.requires_grad = True

        # Optimizer covers LoRA params + lm_head + router params
        trainable_params = [
            p for p in self.wm.parameters() if p.requires_grad
        ] + [
            p for p in self.router.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        # Stove schedule: fast flame early (linear warmup), slow cool-down
        # late (cosine decay). Starts at 10% LR, ramps to full over warmup,
        # then eases toward zero across total steps. Steady hands beat one heat.
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=max(1, lr_warmup_steps)
        )
        decay_steps = max(1, lr_total_steps - lr_warmup_steps)
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=decay_steps
        )
        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, decay], milestones=[max(1, lr_warmup_steps)]
        )

        # Mixed precision: autocast on CUDA devices, GradScaler for FP16 stability.
        # On CPU or when disabled, these are no-ops so the same code path works everywhere.
        self.use_amp = use_amp and device.type == "cuda"
        self.amp_dtype = torch.float16 if self.use_amp else torch.float32
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.router.set_ewc_lambda(ewc_lambda)
        self._step_count = 0
        self._total_tokens = 0
        self._accum_step = 0

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> TrainStepMetrics:
        """Execute a single training step.

        Args:
            batch: Dict with 'input_ids' [batch, seq_len] and 'labels' [batch].

        Returns:
            TrainStepMetrics for this step.
        """
        t0 = time.time()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        bsz, seq_len = input_ids.shape

        # 1. Encode context via working memory
        routing_hidden = self.wm.get_routing_hidden(input_ids)  # [batch, hidden]

        # 2. Route to experts
        expert_ids, confidences, domain_indices = self.router(routing_hidden)

        # 3. Load experts from cache and batch execution by expert ID
        # Groups samples routing to the same expert into a single forward pass
        # instead of running batch-of-1 tensors in a Python loop.
        top_k = expert_ids.size(1)

        # Collect unique (expert_id, k_idx) pairs and their sample indices
        expert_groups: dict[str, list[tuple[int, int]]] = {}  # eid -> [(b_idx, k_idx)]
        for k_idx in range(top_k):
            for b_idx in range(bsz):
                eid = expert_ids[b_idx, k_idx].item()
                if eid < 0:
                    continue
                eid_str = str(eid)
                if eid_str not in expert_groups:
                    expert_groups[eid_str] = []
                expert_groups[eid_str].append((b_idx, k_idx))

        # Load all unique experts once, then run batched forward passes
        loaded_experts: dict[str, nn.Module] = {}
        for eid_str in expert_groups:
            cached = self.cache.get(eid_str)
            if cached is not None:
                loaded_experts[eid_str] = cached
            else:
                expert = self._load_expert(eid_str)
                if expert is not None:
                    self.cache.put(expert)
                    loaded_experts[eid_str] = expert

        # 4. Aggregate expert outputs weighted by confidence
        # Reuse routing_hidden as fallback so gradient flows through WM LoRA
        # even when no experts are active (fresh model). Avoids a second full
        # transformer forward pass which doubled step latency.
        fallback_proj = routing_hidden
        if fallback_proj.size(-1) != self.expert_output_dim:
            fallback_proj = fallback_proj[:, : self.expert_output_dim]

        aggregated = fallback_proj.clone()
        weight_sum = torch.ones(bsz, 1, device=self.device)

        for eid_str, sample_pairs in expert_groups.items():
            expert = loaded_experts.get(eid_str)
            if expert is None:
                continue
            # Gather sample indices for this expert
            b_indices = [p[0] for p in sample_pairs]
            k_indices = [p[1] for p in sample_pairs]
            # Batch forward: [n_samples, hidden_dim] → [n_samples, expert_out]
            batch_input = routing_hidden[b_indices]
            batch_out = expert(batch_input)  # [n_samples, expert_out]
            # Project if needed
            if batch_out.size(-1) != self.expert_output_dim:
                batch_out = batch_out[:, : self.expert_output_dim]
            # Scatter weighted outputs back
            for i, (b_idx, k_idx) in enumerate(sample_pairs):
                w = confidences[b_idx, k_idx]
                aggregated[b_idx] += w * batch_out[i]
                weight_sum[b_idx] += w

        # Normalize only the expert-contributed portion; fallback stays as-is
        # to preserve gradient connection through WM LoRA params.
        expert_mask = weight_sum > 1.0 + 1e-8
        safe_weight = weight_sum.clamp(min=1.0)
        aggregated = torch.where(
            expert_mask,
            aggregated / safe_weight,
            aggregated,
        )

        # 5. Compute loss (cross-entropy against labels via LM head)
        # Route aggregated hidden states through lm_head → [batch, vocab_size].
        # This connects gradients to lm_head.weight so it learns alongside LoRA.
        if labels.dim() == 2:
            labels_flat = labels.reshape(-1)
        else:
            labels_flat = labels

        # Project through LM head: [batch, hidden_dim] → [batch, vocab_size]
        logits = self.wm.lm_head(aggregated)  # [batch, vocab_size]

        # Expand logits to match flattened label count if needed
        if labels.dim() == 2 and labels.size(1) > 1:
            # Repeat each sample's logits for each token position
            logits = logits.unsqueeze(1).expand(-1, labels.size(1), -1).reshape(-1, logits.size(-1))

        ce_loss = F.cross_entropy(logits, labels_flat, reduction="mean")

        # 6. Add EWC penalty
        ewc_penalty = self.router.compute_ewc_penalty()
        total_loss = ce_loss + ewc_penalty

        # 7. Backward + optimize (with AMP and gradient accumulation support)
        # Scale loss by accum steps so effective LR stays constant.
        scaled_loss = total_loss / self.grad_accum_steps
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._accum_step += 1
        if self._accum_step >= self.grad_accum_steps:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.wm.parameters()) + list(self.router.parameters()),
                max_norm=1.0,
            )
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self._accum_step = 0
            self.lr_scheduler.step()

        # 8. Update growth controller stats
        avg_conf = confidences.mean().item()
        routed_ids_flat = expert_ids.flatten().tolist()
        conf_flat = confidences.flatten().tolist()
        # Router stores integer hashes in expert_ids_map, but _active_experts
        # uses UUID strings. Resolve via bidirectional mapping so activation
        # stats are recorded against the correct expert.
        valid_routed: list[str] = []
        valid_confs: list[float] = []
        for eid, c in zip(routed_ids_flat, conf_flat):
            if eid < 0:
                continue
            global_id = int(eid)
            uuid_key = self.gc._global_id_to_uuid.get(global_id)
            if uuid_key is not None:
                valid_routed.append(uuid_key)
                valid_confs.append(c)
        self.gc.update_activation_stats(valid_routed, bsz * seq_len, valid_confs)

        # 8b. Centroid EMA: pull each routed expert's centroid toward the
        # hidden states that actually routed to it. Without this, centroids
        # stay at their random spawn values forever and unreachable experts
        # (0 tokens) never get picked no matter how much data flows.
        with torch.no_grad():
            per_expert_hiddens: dict[int, list[int]] = {}
            for b_idx in range(bsz):
                for k_idx in range(expert_ids.size(1)):
                    gid = expert_ids[b_idx, k_idx].item()
                    if gid < 0:
                        continue
                    per_expert_hiddens.setdefault(int(gid), []).append(b_idx)
            for gid, rows in per_expert_hiddens.items():
                mean_h = routing_hidden[rows].detach().float().mean(dim=0)
                self.router.update_centroid(gid, mean_h)
            # Second-opinion forwarding: when the top pick fumbles (conf < 0.3)
            # but a sibling was second choice, nudge the sibling toward that
            # sample too. Misdelivered mail gets re-sent — specialists stay
            # specialist, but nobody stays blind to what they missed.
            if expert_ids.size(1) > 1:
                with torch.no_grad():
                    top1_conf = confidences[:, 0]
                    second_ids = expert_ids[:, 1]
                    for b_idx in range(bsz):
                        if float(top1_conf[b_idx].item()) < 0.3:
                            sid = int(second_ids[b_idx].item())
                            if sid >= 0:
                                self.router.update_centroid(
                                    sid, routing_hidden[b_idx].detach().float(), lr=0.02
                                )
            # Rescue pull: experts at 0 tokens are never routed, so EMA above
            # never reaches them. Drift them gently toward live traffic so the
            # router can discover them instead of starving them forever.
            batch_mean = routing_hidden.detach().float().mean(dim=0)
            for uuid_key, gid in self.gc._uuid_to_global_id.items():
                exp = self.gc._active_experts.get(uuid_key)
                if exp is None:
                    continue
                seen = exp.metadata.activation_stats.total_tokens_processed
                if seen == 0:
                    self.router.update_centroid(gid, batch_mean, lr=0.01)
                elif seen < 500:
                    # Underdog boost: small experts get a stronger pull so the
                    # leader/distributor can't hoard all traffic forever.
                    # Rich-get-richer ends here — pups grow into the room.
                    self.router.update_centroid(gid, batch_mean, lr=0.03)
            # Veteran drift: the whole room creeps toward live traffic so
            # unrouted veterans don't freeze while data shifts under them.
            self.router.drift_all_toward(batch_mean, lr=0.005)
            # Leader hub: the leader sees ALL traffic mean each step with the
            # strongest pull — it speaks, seats, and teaches. Its centroid
            # becomes the room's best map; specialists orient around it.
            # (Legacy distributor experts, if present in old saves, train on
            # as seniors with the standard underdog rules above.)
            for uuid_key, exp in self.gc._active_experts.items():
                if getattr(exp.metadata, "role", "") == "leader":
                    gid = self.gc._uuid_to_global_id.get(uuid_key)
                    if gid is not None:
                        self.router.update_centroid(gid, batch_mean, lr=0.04)
                    break

        # 9. Check growth triggers per domain
        self.gc.record_step()
        self.gc.note_loss(total_loss.item())
        self.gc.note_domain_traffic(domain_indices)
        # Refresh the EWC shield every 200 steps so frozen routing knowledge
        # stays protected as new domains arrive. First call arms the penalty
        # (it was silently 0 before — lambda set, Fisher never computed).
        if self._step_count % 200 == 0:
            self.router.update_ewc_state(routing_hidden.detach(), self.device)
            # Auto-prune sleepers: zero-token experts past grace get folded
            # away so seats free for real specialists. Keeps the kitchen lean.
            pruned = self.prune_underutilized()
            if pruned:
                print(f"[lattice] pruned {len(pruned)} sleeper expert(s): {pruned}")
            # Distributor merges: twins fuse on its watch, seats freed.
            merged = self.consolidate_if_needed()
            if merged:
                print(f"[lattice] distributor merged {merged} twin pair(s)")
        for d in range(self.router.num_domains):
            d_count = self.router.domain_expert_counts[d].item()
            if self.gc.should_spawn(avg_conf, d, d_count, total_loss.item()):
                new_expert = self.gc.spawn_expert(d, device=self.device)
                self.cache.put(new_expert)
                break  # Only one spawn per training step

        self._step_count += 1
        self._total_tokens += bsz * seq_len
        elapsed = (time.time() - t0) * 1000

        return TrainStepMetrics(
            loss=total_loss.item(),
            routing_confidence=avg_conf,
            experts_active=self.gc.active_expert_count,
            tokens_processed=bsz * seq_len,
            step_time_ms=elapsed,
        )

    def run_epoch(
        self,
        dataloader: DataLoader,
        max_steps: Optional[int] = None,
    ) -> List[TrainStepMetrics]:
        """Run one epoch of training.

        Args:
            dataloader: Training data loader.
            max_steps: Optional cap on steps per epoch.

        Returns:
            List of per-step metrics.
        """
        self.wm.train()
        self.router.train()
        metrics_log: List[TrainStepMetrics] = []

        for step_idx, batch in enumerate(dataloader):
            if max_steps and step_idx >= max_steps:
                break
            m = self.train_step(batch)
            metrics_log.append(m)

        return metrics_log

    def consolidate_if_needed(self) -> int:
        """Run consolidation pass across all domains.

        Returns:
            Number of consolidations performed.
        """
        count = 0
        for d in range(self.router.num_domains):
            candidates = self.gc.find_consolidation_candidates(d)
            for id_a, id_b, sim in candidates:
                merged = self.gc.consolidate_experts(id_a, id_b)
                if merged is not None:
                    self.cache.put(merged)
                    count += 1
        return count

    def prune_underutilized(self) -> List[str]:
        """Remove experts below activation threshold past grace period.

        Returns:
            List of pruned expert IDs.
        """
        underused = self.gc.check_underutilized(self._total_tokens)
        for eid in underused:
            if eid in self.gc._active_experts:
                del self.gc._active_experts[eid]
        return underused

    def _load_expert(self, expert_id: str) -> Optional[nn.Module]:
        """Callback for cache misses — loads from active experts or returns None.

        The router stores integer hashes in its expert_ids_map buffer, but
        _active_experts uses UUID strings as keys. We resolve via the
        bidirectional mapping maintained by GrowthController.
        """
        # Try direct UUID lookup first (for manual spawns or string IDs)
        if expert_id in self.gc._active_experts:
            return self.gc._active_experts[expert_id].to(self.device)
        # Try integer-to-UUID resolution (router returns hashed int IDs)
        try:
            global_id = int(expert_id)
            uuid_key = self.gc._global_id_to_uuid.get(global_id)
            if uuid_key and uuid_key in self.gc._active_experts:
                return self.gc._active_experts[uuid_key].to(self.device)
        except (ValueError, TypeError):
            pass
        return None
