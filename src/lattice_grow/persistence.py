"""Model persistence: save/load Lattice-Grow models to/from disk.

Directory layout per model:
    ~/.lattice_grow/models/<model_name>/
        config.json          # Constructor kwargs for WM, Router, GC, Trainer
        wm_state.pt          # WorkingMemory state_dict
        router_state.pt      # HierarchicalRouter state_dict
        router_ewc.pt        # EWC fisher + optimal params + lambda
        gc_state.pt          # GrowthController metrics, confidence_history, config
        experts/<id>.pt      # One file per ExpertModule state_dict
        optimizer_state.pt   # AdamW optimizer state_dict
        trainer_meta.json    # _step_count, _total_tokens
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .cache import ExpertCache
from .expert import ExpertMetadata, ExpertModule, ExpertState, CompressionInfo, ActivationStats
from .growth_controller import GrowthController, GrowthMetrics
from .router import HierarchicalRouter
from .trainer import LatticeGrowTrainer
from .working_memory import WorkingMemory

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path.home() / ".lattice_grow" / "models"


def get_default_base_dir() -> Path:
    """Return the default base directory for saved models."""
    return DEFAULT_BASE_DIR


def list_saved_models(base_dir: Optional[Path] = None) -> List[str]:
    """List names of saved models in the base directory.

    Returns:
        Sorted list of model directory names that contain a config.json.
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    if not base.exists():
        return []
    models = []
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and (entry / "config.json").exists():
            models.append(entry.name)
    return models


def delete_model(model_name: str, base_dir: Optional[Path] = None) -> bool:
    """Delete a saved model from disk.

    Returns:
        True if deleted, False if not found.
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    model_dir = base / model_name
    if not model_dir.exists():
        return False
    shutil.rmtree(model_dir)
    logger.info("Deleted model: %s", model_name)
    return True


def save_metrics_history(
    metrics_history: list,
    model_name: str,
    base_dir: Optional[Path] = None,
) -> None:
    """Persist dashboard metrics history to disk as JSON.

    Args:
        metrics_history: List of TrainStepMetrics (or dicts) to save.
        model_name: Name of the model directory.
        base_dir: Override default storage location.
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    model_dir = base / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    from .trainer import TrainStepMetrics
    serializable = []
    for m in metrics_history:
        if isinstance(m, TrainStepMetrics):
            serializable.append({
                "loss": m.loss,
                "routing_confidence": m.routing_confidence,
                "experts_active": m.experts_active,
                "tokens_processed": m.tokens_processed,
                "step_time_ms": m.step_time_ms,
            })
        elif isinstance(m, dict):
            serializable.append(m)

    with open(model_dir / "metrics_history.json", "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved %d metrics entries for model '%s'", len(serializable), model_name)


def load_metrics_history(
    model_name: str,
    base_dir: Optional[Path] = None,
) -> list:
    """Load persisted metrics history from disk.

    Returns:
        List of TrainStepMetrics objects, or empty list if not found.
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    path = base / model_name / "metrics_history.json"
    if not path.exists():
        return []

    from .trainer import TrainStepMetrics
    try:
        with open(path) as f:
            data = json.load(f)
        return [
            TrainStepMetrics(
                loss=float(d.get("loss", 0.0)),
                routing_confidence=float(d.get("routing_confidence", 0.0)),
                experts_active=int(d.get("experts_active", 0)),
                tokens_processed=int(d.get("tokens_processed", 0)),
                step_time_ms=float(d.get("step_time_ms", 0.0)),
            )
            for d in data
        ]
    except Exception as e:
        logger.warning("Failed to load metrics history for '%s': %s", model_name, e)
        return []


def save_model(
    trainer: LatticeGrowTrainer,
    model_name: str,
    base_dir: Optional[Path] = None,
    metrics_history: Optional[list] = None,
) -> Path:
    """Save a complete Lattice-Grow model to disk.

    Args:
        trainer: The trainer instance to serialize.
        model_name: Name for the model directory (filesystem-safe).
        base_dir: Override default storage location.

    Returns:
        Path to the saved model directory.
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    model_dir = base / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    experts_dir = model_dir / "experts"
    experts_dir.mkdir(exist_ok=True)

    wm = trainer.wm
    router = trainer.router
    gc = trainer.gc

    # --- Extract WorkingMemory config ---
    # WorkingMemory doesn't store all constructor kwargs as instance attrs,
    # so we extract what we can and infer the rest from submodules.
    wm_config = {
        "vocab_size": wm.embed_tokens.num_embeddings,
        "hidden_dim": wm.hidden_dim,
        "num_layers": wm.num_layers,
        "num_heads": wm.layers[0].attn.num_heads if wm.layers else 12,
        "ffn_dim": wm.layers[0].ffn[0].base.out_features if wm.layers else 3072,
        "max_seq_len": wm._max_seq_len,
        "lora_rank": wm.layers[0].attn_out.lora_a.out_features if wm.layers else 16,
        "dropout": wm.embed_dropout.p,
    }

    # --- Extract Router config ---
    router_config = {
        "hidden_dim": router.hidden_dim,
        "num_domains": router.num_domains,
        "max_experts_per_domain": router.max_experts_per_domain,
        "top_k": router.top_k,
    }

    # --- Extract GrowthController config ---
    gc_config = {
        "hidden_dim": gc.hidden_dim,
        "expert_dim": gc.expert_dim,
        "ffn_dim": gc.ffn_dim,
        "confidence_threshold": gc.confidence_threshold,
        "low_confidence_window": gc.low_confidence_window,
        "consolidation_similarity": gc.consolidation_similarity,
        "min_activation_count": gc.min_activation_count,
        "grace_period_tokens": gc.grace_period_tokens,
        "max_experts_per_domain": gc.max_experts_per_domain,
    }

    # --- Extract Trainer config ---
    trainer_config = {
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "ewc_lambda": router._ewc_lambda,
        "expert_output_dim": trainer.expert_output_dim,
        "device": str(trainer.device),
    }

    # --- Save config.json ---
    config = {
        "wm": wm_config,
        "router": router_config,
        "gc": gc_config,
        "trainer": trainer_config,
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Save WorkingMemory state ---
    torch.save(wm.state_dict(), model_dir / "wm_state.pt")

    # --- Save Router state (weights/buffers only) ---
    torch.save(router.state_dict(), model_dir / "router_state.pt")

    # --- Save Router EWC state separately ---
    ewc_state = {
        "fisher": {k: v.cpu() for k, v in router._ewc_fisher.items()},
        "optimal_params": {k: v.cpu() for k, v in router._ewc_optimal_params.items()},
        "lambda": router._ewc_lambda,
    }
    torch.save(ewc_state, model_dir / "router_ewc.pt")

    # --- Save GrowthController state ---
    gc_state = _serialize_gc(gc)
    torch.save(gc_state, model_dir / "gc_state.pt")

    # --- Save each expert ---
    for eid, expert in gc._active_experts.items():
        safe_eid = eid.replace("/", "_").replace("\\", "_")
        expert_path = experts_dir / f"{safe_eid}.pt"
        torch.save(expert.state_dict(), expert_path)

    # --- Save optimizer state ---
    torch.save(trainer.optimizer.state_dict(), model_dir / "optimizer_state.pt")

    # --- Save trainer meta ---
    trainer_meta = {
        "step_count": trainer._step_count,
        "total_tokens": trainer._total_tokens,
    }
    with open(model_dir / "trainer_meta.json", "w") as f:
        json.dump(trainer_meta, f, indent=2)

    # --- Save metrics history (if provided) ---
    if metrics_history is not None:
        save_metrics_history(metrics_history, model_name, base_dir=base_dir)

    logger.info(
        "Saved model '%s' to %s (%d experts, %d steps)",
        model_name, model_dir, len(gc._active_experts), trainer._step_count,
    )
    return model_dir


def load_model(
    model_name: str,
    base_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> LatticeGrowTrainer:
    """Load a complete Lattice-Grow model from disk.

    Args:
        model_name: Name of the model directory.
        base_dir: Override default storage location.
        device: Target device (defaults to cpu).

    Returns:
        Fully reconstructed LatticeGrowTrainer.

    Raises:
        FileNotFoundError: If model directory or config missing.
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    model_dir = base / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found at {model_dir}")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json missing in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    if device is None:
        device_str = config["trainer"].get("device", "cpu")
        device = torch.device(device_str)

    # 1. Reconstruct WorkingMemory
    wm_cfg = config["wm"]
    wm = WorkingMemory(
        vocab_size=wm_cfg["vocab_size"],
        hidden_dim=wm_cfg["hidden_dim"],
        num_layers=wm_cfg["num_layers"],
        num_heads=wm_cfg["num_heads"],
        ffn_dim=wm_cfg["ffn_dim"],
        max_seq_len=wm_cfg["max_seq_len"],
        lora_rank=wm_cfg["lora_rank"],
        dropout=wm_cfg["dropout"],
    ).to(device)
    # strict=False handles backward compatibility: older checkpoints saved
    # before the lm_head was added won't contain "lm_head.weight". The freshly
    # constructed WorkingMemory initializes it with default (random) weights,
    # which will be trained alongside LoRA on subsequent training steps.
    wm_result = wm.load_state_dict(
        torch.load(model_dir / "wm_state.pt", map_location=device),
        strict=False,
    )
    if wm_result.missing_keys:
        logger.info(
            "WorkingMemory loaded with %d missing key(s) (expected for pre-lm_head checkpoints): %s",
            len(wm_result.missing_keys), wm_result.missing_keys,
        )
    if wm_result.unexpected_keys:
        logger.warning(
            "WorkingMemory loaded with %d unexpected key(s): %s",
            len(wm_result.unexpected_keys), wm_result.unexpected_keys,
        )

    # 2. Reconstruct Router
    r_cfg = config["router"]
    router = HierarchicalRouter(
        hidden_dim=r_cfg["hidden_dim"],
        num_domains=r_cfg["num_domains"],
        max_experts_per_domain=r_cfg["max_experts_per_domain"],
        top_k=r_cfg["top_k"],
    ).to(device)
    router.load_state_dict(torch.load(model_dir / "router_state.pt", map_location=device))

    # Load EWC state
    ewc_path = model_dir / "router_ewc.pt"
    if ewc_path.exists():
        ewc_state = torch.load(ewc_path, map_location=device)
        router._ewc_fisher = {k: v.to(device) for k, v in ewc_state.get("fisher", {}).items()}
        router._ewc_optimal_params = {
            k: v.to(device) for k, v in ewc_state.get("optimal_params", {}).items()
        }
        router._ewc_lambda = ewc_state.get("lambda", 0.0)

    # 3. Reconstruct GrowthController
    gc_cfg = config["gc"]
    gc = GrowthController(
        router=router,
        hidden_dim=gc_cfg["hidden_dim"],
        expert_dim=gc_cfg["expert_dim"],
        ffn_dim=gc_cfg["ffn_dim"],
        confidence_threshold=gc_cfg["confidence_threshold"],
        low_confidence_window=gc_cfg["low_confidence_window"],
        consolidation_similarity=gc_cfg["consolidation_similarity"],
        min_activation_count=gc_cfg["min_activation_count"],
        grace_period_tokens=gc_cfg["grace_period_tokens"],
        max_experts_per_domain=gc_cfg["max_experts_per_domain"],
    )

    # Load GC state (metrics, confidence history, expert metadata)
    gc_state_path = model_dir / "gc_state.pt"
    if gc_state_path.exists():
        gc_state = torch.load(gc_state_path, map_location=device)
        _deserialize_gc(gc, gc_state, device)

    # Load expert weights
    experts_dir = model_dir / "experts"
    if experts_dir.exists():
        for eid, expert in gc._active_experts.items():
            safe_eid = eid.replace("/", "_").replace("\\", "_")
            expert_path = experts_dir / f"{safe_eid}.pt"
            if expert_path.exists():
                expert.load_state_dict(
                    torch.load(expert_path, map_location=device)
                )
                expert.to(device)

    # 4. Rebuild ExpertCache
    cache = ExpertCache(max_experts=8, device=device)
    for eid, expert in gc._active_experts.items():
        cache.put(expert)

    # 5. Reconstruct Trainer
    t_cfg = config["trainer"]
    trainer = LatticeGrowTrainer(
        working_memory=wm,
        router=router,
        growth_controller=gc,
        expert_cache=cache,
        device=device,
        learning_rate=t_cfg.get("learning_rate", 3e-4),
        ewc_lambda=t_cfg.get("ewc_lambda", 5000.0),
        expert_output_dim=t_cfg.get("expert_output_dim", 768),
    )

    # Restore optimizer state
    opt_path = model_dir / "optimizer_state.pt"
    if opt_path.exists():
        try:
            trainer.optimizer.load_state_dict(
                torch.load(opt_path, map_location=device)
            )
        except Exception as e:
            logger.warning("Could not restore optimizer state: %s", e)

    # Restore trainer meta
    meta_path = model_dir / "trainer_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        trainer._step_count = meta.get("step_count", 0)
        trainer._total_tokens = meta.get("total_tokens", 0)

    # Ensure correct freeze/unfreeze state
    wm.freeze_base()
    wm.unfreeze_lora()

    logger.info(
        "Loaded model '%s': %d experts, %d steps, %d tokens",
        model_name, len(gc._active_experts), trainer._step_count, trainer._total_tokens,
    )
    return trainer


# ---------------------------------------------------------------------------
# Internal serialization helpers
# ---------------------------------------------------------------------------

def _serialize_gc(gc: GrowthController) -> dict:
    """Serialize GrowthController state to a dict suitable for torch.save."""
    metrics = asdict(gc.metrics)

    experts_data: Dict[str, dict] = {}
    for eid, expert in gc._active_experts.items():
        md = expert.metadata
        # Convert tensor centroid to list for JSON compatibility
        centroid_list = None
        if md.routing_centroid is not None:
            centroid_list = md.routing_centroid.cpu().tolist()

        experts_data[eid] = {
            "metadata": {
                "expert_id": md.expert_id,
                "parent_id": md.parent_id,
                "domain_cluster": md.domain_cluster,
                "role": md.role,
                "created_timestamp": md.created_timestamp,
                "freeze_timestamp": md.freeze_timestamp,
                "version": md.version,
                "compression": asdict(md.compression),
                "routing_centroid": centroid_list,
                "activation_stats": asdict(md.activation_stats),
                "training_samples_seen": md.training_samples_seen,
                "consolidation_eligible": md.consolidation_eligible,
            },
            "state": expert.state.value,
            "hidden_dim": expert.input_proj.in_features,
            "expert_dim": expert.input_proj.out_features,
            "ffn_dim": expert.ffn[0].out_features,
            "dropout": expert.ffn[2].p if hasattr(expert.ffn[2], "p") else 0.1,
        }

    return {
        "metrics": metrics,
        "confidence_history": list(gc._confidence_history),
        "experts": experts_data,
    }


def _deserialize_gc(
    gc: GrowthController,
    state: dict,
    device: torch.device,
) -> None:
    """Restore GrowthController state from serialized dict."""
    # Restore metrics
    m = state.get("metrics", {})
    gc.metrics = GrowthMetrics(
        total_experts_spawned=m.get("total_experts_spawned", 0),
        total_experts_frozen=m.get("total_experts_frozen", 0),
        total_experts_consolidated=m.get("total_experts_consolidated", 0),
        avg_routing_confidence=m.get("avg_routing_confidence", 0.0),
        low_confidence_streak=m.get("low_confidence_streak", 0),
        tokens_since_last_growth=m.get("tokens_since_last_growth", 0),
        steps_since_last_spawn=m.get("steps_since_last_spawn", 0),
        last_growth_timestamp=m.get("last_growth_timestamp", 0.0),
    )

    # Restore confidence history
    gc._confidence_history = list(state.get("confidence_history", []))

    # Restore experts
    experts_data = state.get("experts", {})
    for eid, edata in experts_data.items():
        md_raw = edata.get("metadata", {})

        # Reconstruct nested dataclasses
        comp_raw = md_raw.get("compression", {})
        compression = CompressionInfo(
            method=comp_raw.get("method", "nanoquant-v1"),
            bit_width=comp_raw.get("bit_width", 0.65),
            scale_factors_path=comp_raw.get("scale_factors_path"),
            delta_base_id=comp_raw.get("delta_base_id"),
            delta_encoding=comp_raw.get("delta_encoding", "zipllm-bitx-v1"),
        )

        stats_raw = md_raw.get("activation_stats", {})
        activation_stats = ActivationStats(
            total_tokens_processed=stats_raw.get("total_tokens_processed", 0),
            last_access_timestamp=stats_raw.get("last_access_timestamp"),
            avg_confidence=stats_raw.get("avg_confidence", 0.0),
        )

        centroid_raw = md_raw.get("routing_centroid")
        centroid_tensor = None
        if centroid_raw is not None:
            centroid_tensor = torch.tensor(centroid_raw, dtype=torch.float32)

        metadata = ExpertMetadata(
            expert_id=md_raw.get("expert_id", eid),
            parent_id=md_raw.get("parent_id"),
            domain_cluster=md_raw.get("domain_cluster", "general"),
            role=md_raw.get("role", "specialist"),
            created_timestamp=md_raw.get("created_timestamp"),
            freeze_timestamp=md_raw.get("freeze_timestamp"),
            version=md_raw.get("version", 1),
            compression=compression,
            routing_centroid=centroid_tensor,
            activation_stats=activation_stats,
            training_samples_seen=md_raw.get("training_samples_seen", 0),
            consolidation_eligible=md_raw.get("consolidation_eligible", False),
        )

        hidden_dim = edata.get("hidden_dim", gc.hidden_dim)
        expert_dim = edata.get("expert_dim", gc.expert_dim)
        ffn_dim = edata.get("ffn_dim", gc.ffn_dim)
        dropout = edata.get("dropout", 0.1)

        expert = ExpertModule(
            hidden_dim=hidden_dim,
            expert_dim=expert_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            metadata=metadata,
        ).to(device)

        # Restore state enum
        state_str = edata.get("state", "training")
        try:
            expert._state = ExpertState(state_str)
        except ValueError:
            logger.warning("Unknown expert state '%s' for %s, defaulting to TRAINING", state_str, eid)
            expert._state = ExpertState.TRAINING

        gc._active_experts[eid] = expert
