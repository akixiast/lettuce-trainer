"""Expert module: self-contained FFN block with lifecycle management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn


class ExpertState(Enum):
    TRAINING = "training"
    VALIDATING = "validating"
    FREEZING = "freezing"
    ARCHIVED = "archived"
    CONSOLIDATED = "consolidated"


@dataclass
class CompressionInfo:
    method: str = "nanoquant-v1"
    bit_width: float = 0.65
    scale_factors_path: Optional[str] = None
    delta_base_id: Optional[str] = None
    delta_encoding: str = "zipllm-bitx-v1"


@dataclass
class ActivationStats:
    total_tokens_processed: int = 0
    last_access_timestamp: Optional[str] = None
    avg_confidence: float = 0.0


# Descriptive name templates per domain cluster.
# Used to generate human-readable expert names instead of raw UUIDs.
_DOMAIN_NAME_TEMPLATES: dict[str, list[str]] = {
    "domain_0": ["conversation", "dialogue", "chat", "interaction", "exchange"],
    "domain_1": ["reasoning", "logic", "analysis", "deduction", "inference"],
    "domain_2": ["knowledge", "facts", "memory", "recall", "information"],
    "domain_3": ["creativity", "imagination", "generation", "synthesis", "expression"],
    "domain_4": ["planning", "strategy", "organization", "structure", "sequencing"],
    "domain_5": ["language", "grammar", "syntax", "vocabulary", "linguistics"],
    "domain_6": ["context", "situation", "environment", "setting", "background"],
    "domain_7": ["emotion", "sentiment", "tone", "empathy", "feeling"],
}

# Counter for generating unique descriptive names within each domain.
_domain_name_counters: dict[str, int] = {}


def _generate_descriptive_name(domain_cluster: str) -> str:
    """Generate a human-readable expert name based on domain cluster.

    Returns names like 'expert-of-conversation-1', 'expert-of-reasoning-2', etc.
    Falls back to 'expert-general-N' for unknown domains.
    """
    templates = _DOMAIN_NAME_TEMPLATES.get(domain_cluster)
    if not templates:
        key = "general"
        idx = _domain_name_counters.get(key, 0)
        _domain_name_counters[key] = idx + 1
        return f"expert-general-{idx + 1}"

    # Cycle through templates and add counter for uniqueness
    idx = _domain_name_counters.get(domain_cluster, 0)
    _domain_name_counters[domain_cluster] = idx + 1
    template = templates[idx % len(templates)]
    suffix = (idx // len(templates)) + 1
    return f"expert-of-{template}-{suffix}"


@dataclass
class ExpertMetadata:
    expert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    display_name: str = ""
    parent_id: Optional[str] = None
    domain_cluster: str = "general"
    role: str = "specialist"  # specialist | leader | distributor
    created_timestamp: Optional[str] = None
    freeze_timestamp: Optional[str] = None
    version: int = 1
    compression: CompressionInfo = field(default_factory=CompressionInfo)
    routing_centroid: Optional[torch.Tensor] = None
    activation_stats: ActivationStats = field(default_factory=ActivationStats)
    training_samples_seen: int = 0
    consolidation_eligible: bool = False

    def __post_init__(self) -> None:
        """Auto-generate a descriptive display name if not provided."""
        if not self.display_name:
            self.display_name = _generate_descriptive_name(self.domain_cluster)


class ExpertModule(nn.Module):
    """Self-contained FFN expert block (~10M params at default dims)."""

    def __init__(
        self,
        hidden_dim: int = 768,
        expert_dim: int = 1024,
        ffn_dim: int = 4096,
        dropout: float = 0.1,
        metadata: Optional[ExpertMetadata] = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(hidden_dim, expert_dim)
        self.ffn = nn.Sequential(
            nn.Linear(expert_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, expert_dim),
        )
        self.output_proj = nn.Linear(expert_dim, hidden_dim)
        self.metadata = metadata or ExpertMetadata()
        self._state = ExpertState.TRAINING

    @property
    def state(self) -> ExpertState:
        return self._state

    def transition(self, new_state: ExpertState) -> None:
        valid_transitions = {
            ExpertState.TRAINING: {ExpertState.VALIDATING},
            ExpertState.VALIDATING: {ExpertState.FREEZING, ExpertState.TRAINING},
            ExpertState.FREEZING: {ExpertState.ARCHIVED},
            ExpertState.ARCHIVED: {ExpertState.CONSOLIDATED},
            ExpertState.CONSOLIDATED: set(),
        }
        if new_state not in valid_transitions.get(self._state, set()):
            raise ValueError(
                f"Invalid transition {self._state.value} -> {new_state.value}"
            )
        self._state = new_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.ffn(h)
        return self.output_proj(h)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
