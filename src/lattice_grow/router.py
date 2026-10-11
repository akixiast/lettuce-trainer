"""Hierarchical router: two-stage expert selection with EWC protection."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalRouter(nn.Module):
    """Two-stage router: domain classification → within-cluster kNN.

    Stage 1 selects a tier-1 domain cluster via softmax.
    Stage 2 retrieves top-k experts from that cluster using centroid similarity.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_domains: int = 16,
        max_experts_per_domain: int = 256,
        top_k: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.max_experts_per_domain = max_experts_per_domain
        self.top_k = top_k

        # Stage 1: domain classifier
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_domains),
        )

        # Stage 2: expert centroids buffer (updated externally)
        # Shape: [num_domains, max_experts_per_domain, hidden_dim]
        self.register_buffer(
            "expert_centroids",
            torch.zeros(num_domains, max_experts_per_domain, hidden_dim),
        )
        self.register_buffer(
            "expert_ids_map",
            torch.full((num_domains, max_experts_per_domain), -1, dtype=torch.long),
        )
        self.register_buffer(
            "domain_expert_counts",
            torch.zeros(num_domains, dtype=torch.long),
        )
        # Load-balance state: per-domain traffic counts (persistent). Forward
        # subtracts a small log-bonus so overused lanes get gently penalized
        # and unexplored lanes get tried. Prevents total lane collapse.
        self.register_buffer(
            "domain_traffic_counts",
            torch.zeros(num_domains),
        )
        self.balance_strength: float = 0.15

        # EWC state
        self._ewc_fisher: dict[str, torch.Tensor] = {}
        self._ewc_optimal_params: dict[str, torch.Tensor] = {}
        self._ewc_lambda: float = 0.0

    def forward(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route to top-k experts.

        Args:
            hidden: [batch, hidden_dim] representation from working memory.

        Returns:
            expert_ids: [batch, top_k] global expert indices
            confidences: [batch, top_k] routing weights (sum to 1)
            domain_indices: [batch] selected domain for each sample
        """
        # Stage 1: domain selection (with load-balance nudge: overused lanes
        # pay a small log penalty so unexplored lanes get tried — prevents
        # total collapse where every batch lands in one domain)
        domain_logits = self.domain_head(hidden)  # [batch, num_domains]
        with torch.no_grad():
            bonus = self.balance_strength * torch.log1p(
                self.domain_traffic_counts.to(domain_logits.device, dtype=domain_logits.dtype)
            )
            domain_logits = domain_logits - bonus.unsqueeze(0)
            # count this batch's winners toward traffic (use detached argmax)
            winners = domain_logits.detach().argmax(dim=-1)
            for d in winners.tolist():
                self.domain_traffic_counts[int(d)] += 1.0
        domain_probs = F.softmax(domain_logits, dim=-1)
        domain_indices = domain_probs.argmax(dim=-1)  # [batch]

        # Stage 2: within-domain expert retrieval (vectorized)
        batch_size = hidden.size(0)
        device = hidden.device

        # Pre-compute per-sample expert counts and identify empty domains
        sample_counts = self.domain_expert_counts[domain_indices]  # [batch]
        no_expert_mask = sample_counts == 0  # [batch]

        # Initialize output tensors
        raw_scores = torch.zeros(batch_size, self.top_k, device=device)
        expert_ids_out = torch.full(
            (batch_size, self.top_k), -1, dtype=torch.long, device=device
        )

        # Handle samples with no experts: use domain prob as pseudo-confidence
        if no_expert_mask.any():
            ne_indices = no_expert_mask.nonzero(as_tuple=True)[0]
            ne_domains = domain_indices[ne_indices]
            ne_probs = domain_probs[ne_indices, ne_domains].clamp(min=1e-6)
            # softmax(x * 5) ≈ p  =>  x ≈ log(p) / 5
            pseudo_logits = torch.log(ne_probs) / 5.0  # [n_no_expert]
            raw_scores[ne_indices] = pseudo_logits.unsqueeze(-1).expand(-1, self.top_k)
            expert_ids_out[ne_indices] = 0

        # Vectorized cosine similarity for samples WITH experts
        has_expert_mask = ~no_expert_mask
        if has_expert_mask.any():
            he_indices = has_expert_mask.nonzero(as_tuple=True)[0]
            he_domains = domain_indices[he_indices]
            he_hidden = hidden[he_indices]  # [n_has, hidden_dim]
            he_counts = sample_counts[he_indices]  # [n_has]
            max_count = int(he_counts.max().item())

            # Gather centroids into a padded batch tensor [n_has, max_count, hidden_dim]
            # Use arange masking to handle variable counts without Python loops
            pos_arange = torch.arange(max_count, device=device).unsqueeze(0)  # [1, max_count]
            valid_pos = pos_arange < he_counts.unsqueeze(1)  # [n_has, max_count]

            # Index gather: expand domain indices to select from expert_centroids
            # expert_centroids shape: [num_domains, max_experts_per_domain, hidden_dim]
            domain_expanded = he_domains.unsqueeze(1).unsqueeze(2).expand(
                -1, max_count, self.hidden_dim
            )  # [n_has, max_count, hidden_dim]
            pos_expanded = pos_arange.unsqueeze(2).expand(
                he_indices.size(0), max_count, self.hidden_dim
            )  # [n_has, max_count, hidden_dim]

            # Gather centroids: use per-sample slicing to guarantee 3D output
            # Fancy indexing with 3D index tensors can collapse dims when n_has=1.
            # Build [n_has, max_count, hidden_dim] via stack of per-row gathers.
            n_has = he_indices.size(0)
            gathered_centroids = torch.zeros(
                n_has, max_count, self.hidden_dim, device=device
            )
            for i in range(n_has):
                d_i = he_domains[i].item()
                c_i = he_counts[i].item()
                gathered_centroids[i, :c_i] = self.expert_centroids[d_i, :c_i]

            # Masked cosine similarity via batched matmul
            he_hidden_norm = F.normalize(he_hidden, dim=-1)  # [n_has, hidden_dim]
            centroids_norm = F.normalize(
                gathered_centroids, dim=-1
            )  # [n_has, max_count, hidden_dim]
            sim_matrix = torch.bmm(
                he_hidden_norm.unsqueeze(1), centroids_norm.transpose(1, 2)
            ).squeeze(1)  # [n_has, max_count]

            # Mask invalid positions to -inf so they never appear in topk
            sim_matrix = sim_matrix.masked_fill(~valid_pos, float("-inf"))

            # Top-k selection across the padded dimension
            # Clamp k to at most max_count to avoid "index k out of range" when
            # every sample in this sub-batch belongs to a domain with < top_k experts.
            k_vals = torch.clamp(he_counts, max=self.top_k)
            safe_k = min(self.top_k, max_count)
            top_scores_raw, top_local_raw = sim_matrix.topk(
                safe_k, dim=-1
            )  # [n_has, safe_k]
            # Pad back to [n_has, top_k] if safe_k < top_k
            if safe_k < self.top_k:
                pad_w = self.top_k - safe_k
                top_scores = torch.cat([
                    top_scores_raw,
                    torch.full((n_has, pad_w), float("-inf"), device=device),
                ], dim=-1)
                top_local = torch.cat([
                    top_local_raw,
                    torch.zeros(n_has, pad_w, dtype=torch.long, device=device),
                ], dim=-1)
            else:
                top_scores = top_scores_raw
                top_local = top_local_raw

            # Map local indices to global expert IDs
            # Gather from expert_ids_map using domain + local index
            local_clamped = top_local.clamp(min=0)  # avoid negative indexing on inf rows
            top_global = self.expert_ids_map[
                he_domains.unsqueeze(1).expand_as(local_clamped), local_clamped
            ]  # [n_has, top_k]

            # Zero-out padded positions (where count < top_k)
            pad_mask = (
                torch.arange(self.top_k, device=device).unsqueeze(0) >= k_vals.unsqueeze(1)
            )  # [n_has, top_k]
            top_scores = top_scores.masked_fill(pad_mask, 0.0)
            top_global = top_global.masked_fill(pad_mask, -1)

            raw_scores[he_indices] = top_scores
            expert_ids_out[he_indices] = top_global

        # Normalize confidences
        confidences = F.softmax(raw_scores * 5.0, dim=-1)  # temperature scaling

        # For samples with no experts, use entropy-based confidence:
        #   High entropy (uniform/untrained) → low confidence → triggers spawning
        #   Low entropy (peaked/learned) → high confidence → suppresses spawning
        # Entropy of uniform dist over N domains = log(N); we normalize by that.
        # This makes the signal independent of num_domains.
        #   - untrained router: norm_entropy ≈ 1.0 → conf ≈ 0.25 + 0.0 = 0.25 < 0.4 ✓
        #   - mildly peaked (max_prob=0.2, N=8): norm_entropy ≈ 0.85 → conf ≈ 0.38 < 0.4 ✓
        #   - confident (max_prob=0.5): norm_entropy ≈ 0.5 → conf ≈ 0.55 > 0.4 ✓
        # Once experts exist, normal cosine-similarity confidence takes over.
        # IMPORTANT: Build a new tensor instead of in-place modification to avoid
        # autograd version mismatch on the softmax output during backward().
        if no_expert_mask.any():
            n_d = float(self.num_domains)
            max_entropy = torch.log(torch.tensor(n_d, device=hidden.device))
            # Per-sample entropy of domain distribution
            safe_probs = domain_probs[no_expert_mask].clamp(min=1e-8)
            entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)  # [n_no_expert]
            norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0)
            # Map: high entropy → low confidence, low entropy → high confidence
            scaled = (0.25 + 0.55 * (1.0 - norm_entropy)).clamp(0.15, 0.90)
            # Use where() to avoid in-place write on softmax output tensor
            expanded_scaled = scaled.unsqueeze(-1).expand_as(confidences[no_expert_mask])
            confidences = confidences.clone()
            confidences[no_expert_mask] = expanded_scaled

        return expert_ids_out, confidences, domain_indices

    def register_expert(self, domain_idx: int, expert_id: int, centroid: torch.Tensor) -> None:
        """Register a new expert's centroid in the routing table."""
        count = self.domain_expert_counts[domain_idx].item()
        if count >= self.max_experts_per_domain:
            raise RuntimeError(f"Domain {domain_idx} at capacity ({self.max_experts_per_domain})")
        self.expert_centroids[domain_idx, count] = centroid.to(self.expert_centroids.device)
        self.expert_ids_map[domain_idx, count] = expert_id
        self.domain_expert_counts[domain_idx] += 1

    @torch.no_grad()
    def update_centroid(self, global_id: int, target: torch.Tensor, lr: float = 0.05) -> bool:
        """EMA-pull an expert's centroid toward observed routing hidden states.

        Centroids are random at spawn and never followed learning before this —
        starved experts stayed unreachable forever. Call once per training step
        per routed expert with the mean hidden vector that routed to it.
        Returns False when the id is not registered (safe to ignore).
        """
        hit = (self.expert_ids_map == int(global_id)).nonzero(as_tuple=False)
        if hit.numel() == 0:
            return False
        d_idx, slot = int(hit[0, 0].item()), int(hit[0, 1].item())
        cur = self.expert_centroids[d_idx, slot]
        self.expert_centroids[d_idx, slot] = cur + lr * (
            target.to(cur.device, dtype=cur.dtype) - cur
        )
        return True

    @torch.no_grad()
    def drift_all_toward(self, target: torch.Tensor, lr: float = 0.005) -> int:
        """Creep EVERY registered centroid slightly toward recent traffic.

        Veterans that receive no routings would otherwise freeze while the
        kitchen moves on (data shift strands them). Tiny lr: follows the
        room without wrecking specializations. Returns slots moved.
        """
        moved = 0
        t = target.to(self.expert_centroids.device, dtype=self.expert_centroids.dtype)
        for d in range(self.num_domains):
            n = int(self.domain_expert_counts[d].item())
            for s in range(n):
                cur = self.expert_centroids[d, s]
                self.expert_centroids[d, s] = cur + lr * (t - cur)
                moved += 1
        return moved

    def compute_ewc_penalty(self) -> torch.Tensor:
        """Compute EWC regularization penalty for router parameters."""
        if not self._ewc_fisher:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, param in self.named_parameters():
            if name in self._ewc_fisher:
                fisher = self._ewc_fisher[name].to(param.device)
                optimal = self._ewc_optimal_params[name].to(param.device)
                penalty = penalty + (fisher * (param - optimal) ** 2).sum()
        return self._ewc_lambda * penalty

    def update_ewc_state(self, hidden_batch, device: torch.device) -> None:
        """Accumulate Fisher information for the domain head.

        Call periodically with a batch of routing hidden states
        ([batch, hidden_dim], detached). Uses sensitivity of the domain
        logits as the Fisher signal — no labels needed. Before this fix
        the method forwarded raw inputs through the router (wrong
        signature) so it could never run; the EWC penalty silently stayed 0
        and old routing knowledge was unprotected.
        """
        self.eval()
        h = hidden_batch.to(device)
        self.zero_grad()
        domain_logits = self.domain_head(h)
        loss = domain_logits.pow(2).mean()
        loss.backward()
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_sq = param.grad.data.detach().clone() ** 2
                if name in self._ewc_fisher:
                    self._ewc_fisher[name] = self._ewc_fisher[name] + grad_sq
                else:
                    self._ewc_fisher[name] = grad_sq
        for name, param in self.named_parameters():
            self._ewc_optimal_params[name] = param.data.detach().clone()
        self.train()

    def set_ewc_lambda(self, lam: float) -> None:
        self._ewc_lambda = lam
