"""Working memory: fixed 12-layer transformer decoder with LoRA adapters."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation (rank=16 default)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA weights to zero so base model starts unchanged
        nn.init.zeros_(self.lora_b.weight)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))

        # Freeze base linear during LoRA training
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.dropout(self.lora_b(self.lora_a(x))) * self.scaling


class TransformerBlock(nn.Module):
    """Single decoder-only transformer block with LoRA on attention projections."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        lora_rank: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # LoRA wrappers around Q/V projections (applied post-attn via hook is complex;
        # here we use a simpler approach: LoRA on the output projection)
        self.attn_out = LoRALinear(hidden_dim, hidden_dim, rank=lora_rank)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            LoRALinear(hidden_dim, ffn_dim, rank=lora_rank),
            nn.GELU(),
            nn.Dropout(dropout),
            LoRALinear(ffn_dim, hidden_dim, rank=lora_rank),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(self.attn_out(attn_out))

        # FFN with pre-norm
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class WorkingMemory(nn.Module):
    """Fixed-size transformer decoder (~100M params) serving as the compute core.

    Architecture:
      - Token embedding + positional encoding
      - 12 transformer blocks with LoRA adapters (rank=16)
      - Output head projecting to expert routing space

    The base transformer weights are frozen after initial pretraining;
    only LoRA parameters and the router interface are updated during
    Lattice-Grow training.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        max_seq_len: int = 2048,
        lora_rank: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.embed_pos = nn.Embedding(max_seq_len, hidden_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                lora_rank=lora_rank,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output projection used by router for expert selection
        self.output_head = nn.Linear(hidden_dim, hidden_dim)

        # Language model head for text generation (hidden → vocab)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        self._max_seq_len = max_seq_len
        self._causal_mask_cache: dict[int, torch.Tensor] = {}

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len].to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through working memory.

        Args:
            input_ids: [batch, seq_len] token indices.
            positions: [batch, seq_len] position indices (auto-generated if None).

        Returns:
            hidden: [batch, seq_len, hidden_dim] contextual representations.
                    Last token's representation is used for expert routing.
        """
        bsz, seq_len = input_ids.shape

        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embed_tokens(input_ids) + self.embed_pos(positions)
        x = self.embed_dropout(x)

        causal_mask = self._get_causal_mask(seq_len, x.device)

        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)

        x = self.final_norm(x)
        x = self.output_head(x)
        return x

    def get_routing_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the last-token hidden state for expert routing.

        Returns:
            [batch, hidden_dim] representation of the most recent token.
        """
        hidden = self.forward(input_ids)
        return hidden[:, -1, :]  # Last token

    def freeze_base(self) -> None:
        """Freeze all non-LoRA parameters."""
        for name, param in self.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    def unfreeze_lora(self) -> None:
        """Ensure all LoRA parameters are trainable."""
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def lora_param_count(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if "lora_" in n and p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: int = 50256,
    ) -> torch.Tensor:
        """Autoregressive text generation using the LM head.

        Args:
            input_ids: [1, seq_len] prompt token indices.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            top_k: Top-k filtering for sampling.
            eos_token_id: Token ID that signals end of generation.

        Returns:
            Generated token IDs [1, seq_len + n_generated].
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            if generated.size(1) > self._max_seq_len:
                gen_input = generated[:, -self._max_seq_len:]
            else:
                gen_input = generated

            # Forward pass through transformer (no output_head projection needed for LM)
            bsz, seq_len = gen_input.shape
            positions = torch.arange(seq_len, device=gen_input.device).unsqueeze(0)
            x = self.embed_tokens(gen_input) + self.embed_pos(positions)
            x = self.embed_dropout(x)
            causal_mask = self._get_causal_mask(seq_len, x.device)

            for layer in self.layers:
                x = layer(x, causal_mask=causal_mask)

            x = self.final_norm(x)

            # Project last token to vocab space via LM head
            logits = self.lm_head(x[:, -1, :])  # [1, vocab_size]

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop on EOS
            if next_token.item() == eos_token_id:
                break

        return generated
