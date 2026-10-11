"""Growth controller: expert spawning, freezing, and consolidation logic."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .expert import ExpertMetadata, ExpertModule, ExpertState
from .compression import CompressionPipeline, CompressedExpert
from .router import HierarchicalRouter


@dataclass
class GrowthMetrics:
    """Tracks growth-related statistics for adaptive decisions."""

    total_experts_spawned: int = 0
    total_experts_frozen: int = 0
    total_experts_consolidated: int = 0
    avg_routing_confidence: float = 0.0
    low_confidence_streak: int = 0
    tokens_since_last_growth: int = 0
    steps_since_last_spawn: int = 0
    total_training_steps: int = 0
    last_growth_timestamp: float = field(default_factory=time.time)


class GrowthController:
    """Orchestrates expert lifecycle: spawn → train → validate → freeze → archive.

    Growth triggers:
      - Low average routing confidence over a window
      - High domain-cluster load imbalance
      - Explicit user/scheduler signal

    Consolidation triggers:
      - Two experts in same domain with cosine similarity > threshold
      - Expert activation count below minimum after grace period
    """

    def __init__(
        self,
        router: HierarchicalRouter,
        hidden_dim: int = 768,
        expert_dim: int = 1024,
        ffn_dim: int = 4096,
        confidence_threshold: float = 0.4,
        low_confidence_window: int = 10,
        consolidation_similarity: float = 0.92,
        min_activation_count: int = 50,
        grace_period_tokens: int = 10000,
        max_experts_per_domain: int = 256,
        min_steps_before_spawn: int = 5,
        max_total_experts: int = 8,
        spawn_cooldown_tokens: int = 5000,
    ):
        self.router = router
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.ffn_dim = ffn_dim
        self.confidence_threshold = confidence_threshold
        self.low_confidence_window = low_confidence_window
        self.consolidation_similarity = consolidation_similarity
        self.min_activation_count = min_activation_count
        self.grace_period_tokens = grace_period_tokens
        self.max_experts_per_domain = max_experts_per_domain
        self.min_steps_before_spawn = min_steps_before_spawn
        self.max_total_experts = max_total_experts
        self.spawn_cooldown_tokens = spawn_cooldown_tokens
        self.metrics = GrowthMetrics()
        self._active_experts: Dict[str, ExpertModule] = {}
        self._confidence_history: List[float] = []
        self._domain_traffic: Dict[int, int] = {}
        self._compression_pipeline = CompressionPipeline()
        # Bidirectional mapping between router's integer IDs and UUID strings.
        # Router buffers are torch.long tensors, but _active_experts uses UUID keys.
        self._global_id_to_uuid: Dict[int, str] = {}
        self._uuid_to_global_id: Dict[str, int] = {}
        # Leader-health tracking: the distributor only births when the leader
        # is stable or improving. Loss spiking >25% above its running average
        # means the kitchen is unstable — hold births until it settles.
        self._loss_ema: Optional[float] = None

    def note_loss(self, loss: float) -> None:
        """Feed each step's loss so birth decisions follow leader health."""
        if self._loss_ema is None:
            self._loss_ema = float(loss)
        else:
            self._loss_ema = 0.95 * self._loss_ema + 0.05 * float(loss)

    def leader_stable(self, loss: float) -> bool:
        """True unless the leader is spiking hard right now."""
        if self._loss_ema is None:
            return True
        return float(loss) <= self._loss_ema * 1.25

    @property
    def active_expert_count(self) -> int:
        return len(self._active_experts)

    def record_step(self) -> None:
        """Record one training step. Call once per train_step, NOT per domain."""
        self.metrics.total_training_steps += 1
        self.metrics.steps_since_last_spawn += 1

    def note_domain_traffic(self, domain_indices) -> None:
        """Track per-domain routing traffic; warn when lanes sit empty.

        Call once per training step with the router's domain_indices tensor.
        Every 100 steps, reports how many of the domain lanes saw traffic —
        if fewer than half are active, routing has collapsed and whole
        domains starve regardless of centroids.
        """
        try:
            for d in domain_indices.flatten().tolist():
                self._domain_traffic[int(d)] = self._domain_traffic.get(int(d), 0) + 1
        except Exception:
            return
        if self.metrics.total_training_steps % 100 == 0 and self._domain_traffic:
            import torch as _torch

            n_domains = int(self.router.num_domains)
            active = len(self._domain_traffic)
            if active < max(2, n_domains // 2):
                print(
                    f"[lattice] domain-collapse watch: only {active}/{n_domains} "
                    f"domain lanes saw traffic in the last 100 steps "
                    f"(counts={dict(sorted(self._domain_traffic.items()))}). "
                    f"Consider a load-balance nudge before more experts starve."
                )
            self._domain_traffic = {}

    def should_spawn(
        self,
        avg_confidence: float,
        domain_idx: int,
        current_domain_count: int,
        current_loss: Optional[float] = None,
    ) -> bool:
        """Decide whether to spawn a new expert based on routing signals.

        IMPORTANT: Call record_step() once per training step before calling
        this method. This method should be called per-domain but only
        increments confidence history (not step counters).
        """
        # Update confidence history
        self._confidence_history.append(avg_confidence)
        if len(self._confidence_history) > self.low_confidence_window:
            self._confidence_history.pop(0)

        # Check domain capacity
        if current_domain_count >= self.max_experts_per_domain:
            return False

        # Speaker-first: the very first expert is always born into domain_0
        # (conversation) — the voice of the kitchen. Specialists come after.
        if self.metrics.total_experts_spawned == 0 and domain_idx != 0:
            return False

        # Kitchen cap: no new births past max_total_experts. Merge or prune
        # a sleeper first to free a seat, then birth.
        if len(self._active_experts) >= self.max_total_experts:
            return False

        # Birth cooldown: 5000 tokens of real work since last birth.
        # Stops panic-births from one bad batch.
        if self.metrics.tokens_since_last_growth < self.spawn_cooldown_tokens:
            return False

        # Leader-health gate: births happen on the distributor's feelings
        # (confidence), but only when the leader is stable. Spiking leader =
        # unstable kitchen = hold births until it settles. Distributing skill
        # grows as the leader grows — the distributor's choice, earned.
        if current_loss is not None and not self.leader_stable(current_loss):
            return False

        # Enforce minimum training steps before any spawning is allowed.
        # Prevents cold-start deadlock where untrained routers produce
        # borderline confidence that triggers runaway expert creation.
        if self.metrics.total_training_steps < self.min_steps_before_spawn:
            return False

        # Only allow one spawn per training step across all domains.
        # Without this, the first eligible step spawns experts in ALL
        # empty domains simultaneously.
        if self.metrics.steps_since_last_spawn == 0:
            return False

        # Trigger on sustained low confidence with cooldown
        if len(self._confidence_history) >= self.low_confidence_window:
            window_avg = sum(self._confidence_history) / len(self._confidence_history)
            if window_avg < self.confidence_threshold:
                self.metrics.low_confidence_streak += 1
                min_steps_gap = max(5, self.low_confidence_window)
                if (
                    self.metrics.low_confidence_streak >= 1
                    and self.metrics.steps_since_last_spawn >= min_steps_gap
                ):
                    return True
            else:
                self.metrics.low_confidence_streak = 0
        else:
            self.metrics.low_confidence_streak = 0

        return False

    def spawn_expert(
        self,
        domain_idx: int,
        parent_id: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ) -> ExpertModule:
        """Create and register a new expert module."""
        metadata = ExpertMetadata(
            parent_id=parent_id,
            domain_cluster=f"domain_{domain_idx}",
            created_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        expert = ExpertModule(
            hidden_dim=self.hidden_dim,
            expert_dim=self.expert_dim,
            ffn_dim=self.ffn_dim,
            metadata=metadata,
        ).to(device)

        # If child of existing expert, initialize from parent weights
        if parent_id and parent_id in self._active_experts:
            parent = self._active_experts[parent_id]
            self._initialize_from_parent(expert, parent)

        # Role assignment: 1st born = leader (voice, teacher, distributor —
        # speaks, seats traffic, teaches specialists). The middleman
        # distributor role is retired: leader does it all directly.
        # (Old saves may still carry a distributor; it trains on as senior.)
        birth_order = self.metrics.total_experts_spawned
        if birth_order == 0:
            metadata.role = "leader"
            metadata.display_name = "leader-voice-1"
        centroid = self._compute_initial_centroid(domain_idx, expert, device)
        global_id = hash(metadata.expert_id) % (2**31)
        self.router.register_expert(domain_idx, global_id, centroid)

        # Maintain bidirectional mapping so cache lookups by integer ID
        # can resolve to the UUID string used as _active_experts key.
        self._global_id_to_uuid[global_id] = metadata.expert_id
        self._uuid_to_global_id[metadata.expert_id] = global_id

        self._active_experts[metadata.expert_id] = expert
        self.metrics.total_experts_spawned += 1
        self.metrics.last_growth_timestamp = time.time()
        self.metrics.tokens_since_last_growth = 0
        self.metrics.steps_since_last_spawn = 0
        self.metrics.low_confidence_streak = 0

        return expert

    def _initialize_from_parent(
        self, child: ExpertModule, parent: ExpertModule
    ) -> None:
        """Copy parent weights to child with small noise injection."""
        with torch.no_grad():
            for (cn, cp), (pn, pp) in zip(
                child.named_parameters(), parent.named_parameters()
            ):
                noise = torch.randn_like(pp.data) * 0.01 * pp.data.abs().mean()
                cp.data.copy_(pp.data + noise)

    def _compute_initial_centroid(
        self, domain_idx: int, expert: ExpertModule, device: torch.device
    ) -> torch.Tensor:
        """Compute initial routing centroid for a new expert.

        The centroid must live in the router's hidden space (hidden_dim).
        output_proj.weight is [hidden_dim, expert_dim], so mean over dim=1
        yields a [hidden_dim] vector suitable for router registration.
        """
        with torch.no_grad():
            # output_proj: Linear(expert_dim, hidden_dim) → weight [hidden_dim, expert_dim]
            # Mean across expert_dim gives one vector per hidden_dim position.
            centroid = expert.output_proj.weight.mean(dim=1)
            return centroid.to(device)

    def transition_to_validating(self, expert_id: str) -> bool:
        """Move an expert from training to validation phase."""
        if expert_id not in self._active_experts:
            return False
        expert = self._active_experts[expert_id]
        if expert.state != ExpertState.TRAINING:
            return False
        expert.transition(ExpertState.VALIDATING)
        return True

    def freeze_expert(
        self,
        expert_id: str,
        parent_blobs: Optional[Dict[str, bytes]] = None,
    ) -> Optional[CompressedExpert]:
        """Freeze and compress an expert, transitioning it to archived state."""
        if expert_id not in self._active_experts:
            return None

        expert = self._active_experts[expert_id]
        if expert.state != ExpertState.VALIDATING:
            return None

        # Validation gate: refuse to freeze experts that barely trained.
        # They stay VALIDATING and can retry later instead of getting
        # archived forever as duds.
        if expert.metadata.activation_stats.total_tokens_processed < self.min_activation_count:
            return None

        # Transition through freezing to archived
        expert.transition(ExpertState.FREEZING)
        compressed = self._compression_pipeline.compress_expert(
            expert,
            expert_id=expert.metadata.expert_id,
            parent_blobs=parent_blobs,
        )
        expert.transition(ExpertState.ARCHIVED)

        # Update metadata
        expert.metadata.freeze_timestamp = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )

        self.metrics.total_experts_frozen += 1
        return compressed

    def find_consolidation_candidates(
        self,
        domain_idx: int,
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of experts eligible for consolidation within a domain.

        Returns list of (expert_id_a, expert_id_b, similarity) tuples.
        """
        candidates = []
        domain_experts = [
            (eid, exp)
            for eid, exp in self._active_experts.items()
            if exp.metadata.domain_cluster == f"domain_{domain_idx}"
            and exp.state == ExpertState.ARCHIVED
        ]

        for i in range(len(domain_experts)):
            for j in range(i + 1, len(domain_experts)):
                eid_a, exp_a = domain_experts[i]
                eid_b, exp_b = domain_experts[j]

                sim = self._expert_similarity(exp_a, exp_b)
                if sim >= self.consolidation_similarity:
                    candidates.append((eid_a, eid_b, sim))

        return sorted(candidates, key=lambda x: x[2], reverse=True)

    def consolidate_experts(
        self, id_a: str, id_b: str
    ) -> Optional[ExpertModule]:
        """Merge two similar experts into one consolidated expert."""
        if id_a not in self._active_experts or id_b not in self._active_experts:
            return None

        exp_a = self._active_experts[id_a]
        exp_b = self._active_experts[id_b]

        if exp_a.state != ExpertState.ARCHIVED or exp_b.state != ExpertState.ARCHIVED:
            return None

        # Create consolidated expert by averaging weights
        consolidated = ExpertModule(
            hidden_dim=self.hidden_dim,
            expert_dim=self.expert_dim,
            ffn_dim=self.ffn_dim,
            metadata=ExpertMetadata(
                domain_cluster=exp_a.metadata.domain_cluster,
                created_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )

        with torch.no_grad():
            for (cn, cp), (an, ap), (bn, bp) in zip(
                consolidated.named_parameters(),
                exp_a.named_parameters(),
                exp_b.named_parameters(),
            ):
                cp.data.copy_((ap.data + bp.data) / 2.0)

        # Mark originals as consolidated
        exp_a.transition(ExpertState.CONSOLIDATED)
        exp_b.transition(ExpertState.CONSOLIDATED)

        self._active_experts[consolidated.metadata.expert_id] = consolidated
        self.metrics.total_experts_consolidated += 1

        # Register the merged expert with the router so it stays reachable.
        # Before this, consolidations vanished from routing (registered ids
        # still pointed at the retired originals) — merged twins could never
        # be picked and disk savings never turned into usable knowledge.
        try:
            dom = int(exp_a.metadata.domain_cluster.split("_")[-1])
        except (ValueError, IndexError, AttributeError):
            dom = 0
        import hashlib as _hashlib

        new_gid = int(_hashlib.md5(consolidated.metadata.expert_id.encode()).hexdigest()[:8], 16) % (2**31)
        with torch.no_grad():
            new_centroid = consolidated.output_proj.weight.mean(dim=1).detach()
        try:
            self.router.register_expert(dom, new_gid, new_centroid)
        except RuntimeError:
            pass  # domain at capacity — merged expert stays cached, routed later
        self._global_id_to_uuid[new_gid] = consolidated.metadata.expert_id
        self._uuid_to_global_id[consolidated.metadata.expert_id] = new_gid

        return consolidated

    def check_underutilized(
        self, current_token_count: int
    ) -> List[str]:
        """Return IDs of experts that are underutilized past their grace period."""
        underutilized = []
        for eid, expert in self._active_experts.items():
            stats = expert.metadata.activation_stats
            if (
                stats.total_tokens_processed < self.min_activation_count
                and expert.state == ExpertState.ARCHIVED
                and (current_token_count - self.metrics.tokens_since_last_growth)
                > self.grace_period_tokens
            ):
                underutilized.append(eid)
        return underutilized

    def update_activation_stats(
        self,
        expert_ids: List[str],
        token_count: int,
        confidences: List[float],
    ) -> None:
        """Record activation statistics for routed experts.

        Token credit is weighted by routing confidence so shared batches
        don't triple-count: three experts splitting one batch share the
        credit instead of each claiming the full batch.
        """
        for eid, conf in zip(expert_ids, confidences):
            if eid in self._active_experts:
                stats = self._active_experts[eid].metadata.activation_stats
                stats.total_tokens_processed += int(token_count * float(conf))
                stats.avg_confidence = (
                    stats.avg_confidence * 0.95 + conf * 0.05
                )
                stats.last_access_timestamp = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )

        self.metrics.tokens_since_last_growth += token_count

    @staticmethod
    def _expert_similarity(a: ExpertModule, b: ExpertModule) -> float:
        """Cosine similarity between flattened parameter vectors."""
        vec_a = torch.cat([p.data.flatten() for p in a.parameters()])
        vec_b = torch.cat([p.data.flatten() for p in b.parameters()])
        cos = torch.nn.functional.cosine_similarity(
            vec_a.unsqueeze(0), vec_b.unsqueeze(0)
        )
        return cos.item()
