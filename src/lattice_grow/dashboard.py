"""Gradio web dashboard for Lattice-Grow prototype.

Provides:
  - Model management (create, load, save, delete)
  - Live training metrics (loss, routing confidence, active experts)
  - Expert lifecycle visualization (spawn/freeze/consolidate controls)
  - Compression statistics and expert tree view
  - Real-time Plotly charts for training curves
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import plotly.graph_objects as go
import torch

logger = logging.getLogger(__name__)

from .cache import ExpertCache
from .compression import CompressionPipeline
from .growth_controller import GrowthController
from .ingestion import TextIngestionPipeline
from .persistence import (
    delete_model,
    get_default_base_dir,
    list_saved_models,
    load_metrics_history,
    load_model,
    save_model,
)
from .router import HierarchicalRouter
from .trainer import LatticeGrowTrainer, TrainStepMetrics
from .working_memory import WorkingMemory
from .crawler import AutonomousCrawler, CrawlerConfig


@dataclass
class DashboardState:
    """Shared mutable state between training loop and UI callbacks."""

    trainer: Optional[LatticeGrowTrainer] = None
    ingestion: Optional[TextIngestionPipeline] = None
    crawler: Optional["AutonomousCrawler"] = None
    is_training: bool = False
    stop_requested: bool = False
    training_status_msg: str = ""
    _training_target_steps: int = 0
    metrics_history: List[TrainStepMetrics] = field(default_factory=list)
    pending_texts: List[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    autonomous: bool = False

    def append_metrics(self, m: TrainStepMetrics) -> None:
        with self.lock:
            self.metrics_history.append(m)
            # Keep last 500 points to avoid memory bloat
            if len(self.metrics_history) > 500:
                self.metrics_history = self.metrics_history[-500:]

    def get_loss_series(self) -> Tuple[List[int], List[float]]:
        with self.lock:
            steps = list(range(len(self.metrics_history)))
            losses = [m.loss for m in self.metrics_history]
        return steps, losses

    def get_confidence_series(self) -> Tuple[List[int], List[float]]:
        with self.lock:
            steps = list(range(len(self.metrics_history)))
            confs = [m.routing_confidence for m in self.metrics_history]
        return steps, confs

    def get_experts_series(self) -> Tuple[List[int], List[int]]:
        with self.lock:
            steps = list(range(len(self.metrics_history)))
            counts = [m.experts_active for m in self.metrics_history]
        return steps, counts


def _make_loss_fig(state: DashboardState) -> go.Figure:
    steps, losses = state.get_loss_series()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=losses, mode="lines+markers", name="Loss", line=dict(color="#636EFA")))
    fig.update_layout(
        title="Training Loss",
        xaxis_title="Step",
        yaxis_title="Loss",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def _make_confidence_fig(state: DashboardState) -> go.Figure:
    steps, confs = state.get_confidence_series()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=confs, mode="lines+markers", name="Confidence", line=dict(color="#EF553B")))
    fig.update_layout(
        title="Routing Confidence",
        xaxis_title="Step",
        yaxis_title="Avg Confidence",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def _make_experts_fig(state: DashboardState) -> go.Figure:
    steps, counts = state.get_experts_series()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=counts, mode="lines+markers", name="Active Experts", line=dict(color="#00CC96")))
    fig.update_layout(
        title="Active Experts Over Time",
        xaxis_title="Step",
        yaxis_title="Count",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def _get_stats_markdown(state: DashboardState) -> str:
    if not state.metrics_history:
        return "*No training data yet. Click **Start Training** to begin.*"
    with state.lock:
        latest = state.metrics_history[-1]
        total_steps = len(state.metrics_history)
        avg_loss = sum(m.loss for m in state.metrics_history[-50:]) / min(50, total_steps)
        avg_conf = sum(m.routing_confidence for m in state.metrics_history[-50:]) / min(50, total_steps)
    lines = [
        f"### Current Status",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total Steps | {total_steps} |",
        f"| Latest Loss | {latest.loss:.4f} |",
        f"| Avg Loss (last 50) | {avg_loss:.4f} |",
        f"| Routing Confidence | {latest.routing_confidence:.3f} |",
        f"| Avg Confidence (last 50) | {avg_conf:.3f} |",
        f"| Active Experts | {latest.experts_active} |",
        f"| Tokens Processed | {latest.tokens_processed:,} |",
        f"| Step Time | {latest.step_time_ms:.1f} ms |",
    ]
    return "\n".join(lines)


def _training_loop(
    state: DashboardState,
    num_steps: int,
    batch_size: int,
    seq_len: int,
) -> None:
    """Background training loop that feeds metrics into DashboardState.

    Only trains on real ingested content (uploaded files or autonomous crawler
    output). If no pending texts are available, the loop exits immediately.

    Args:
        state: Shared dashboard state.
        num_steps: Max training steps.
        batch_size: Batch size per step.
        seq_len: Sequence length per sample.
    """
    if state.trainer is None:
        return

    if not state.pending_texts or state.ingestion is None:
        logger.warning("No pending texts to train on; skipping training loop.")
        return

    device = state.trainer.device
    state.is_training = True
    state.stop_requested = False

    try:
        dataloader = state.ingestion.create_dataloader(state.pending_texts)
    except Exception as e:
        logger.error("Failed to create dataloader from ingested texts: %s", e)
        state.is_training = False
        return

    data_iter = iter(dataloader)

    try:
        for step in range(num_steps):
            if state.stop_requested:
                break

            try:
                batch = next(data_iter)
            except StopIteration:
                # Data exhausted — restart from the beginning so training
                # continues for the full num_steps instead of stopping early.
                logger.info("Data exhausted at step %d/%d; restarting dataloader.", step, num_steps)
                data_iter = iter(dataloader)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Dataset is genuinely empty (no valid chunks)
                    logger.warning("Dataloader produced no batches; stopping training.")
                    break

            # Move tensors to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            metrics = state.trainer.train_step(batch)
            state.append_metrics(metrics)

            # Update live progress message for the dashboard timer tick
            state.training_status_msg = f"🏋️ Training... ({step + 1}/{num_steps} steps)"

            # Periodic consolidation check every 50 steps
            if (step + 1) % 50 == 0:
                state.trainer.consolidate_if_needed()
                # Autonomous switch: also prune sleepers mid-run so the room
                # self-cleans while training, no babysitting.
                if getattr(state, "autonomous", False):
                    state.trainer.prune_underutilized()

    finally:
        # Don't clear is_training here — the wrapper in start_training
        # holds it True briefly so the timer tick can capture final metrics.
        # If training was started without the wrapper (e.g., direct call),
        # this is safe because the timer only reads metrics_history.
        pass


def create_dashboard(
    hidden_dim: int = 768,
    num_domains: int = 8,
    max_experts_per_domain: int = 32,
    device_str: str = "cpu",
) -> gr.Blocks:
    """Build and return the Gradio Blocks dashboard.

    Args:
        hidden_dim: Working memory hidden dimension.
        num_domains: Number of routing domains.
        max_experts_per_domain: Cap on experts per domain.
        device_str: Torch device string ('cpu' or 'cuda').

    Returns:
        gr.Blocks app ready to launch.
    """
    # State starts empty — model must be created or loaded via UI
    state = DashboardState()

    # --- Callbacks ---

    def start_training(num_steps: int, batch_size: int, seq_len: int):
        if state.trainer is None:
            return "❌ No model loaded. Create or load a model in the Model Management tab first."
        if state.is_training:
            return "⚠️ Training already running."
        if not state.pending_texts or state.ingestion is None:
            return "⚠️ No training data available. Upload text or start autonomous learning first."

        # Track target steps and initial status for live progress display
        state._training_target_steps = int(num_steps)
        state.training_status_msg = f"🏋️ Training... (0/{num_steps} steps)"

        # Wrap training loop to keep is_training True briefly after completion
        # so the 2s timer tick catches the final metrics before the flag clears.
        def _run_and_hold():
            _training_loop(state, int(num_steps), int(batch_size), int(seq_len))
            # Auto-save model and metrics after training completes
            try:
                model_name = getattr(state, '_current_model_name', '')
                if state.trainer is not None and model_name:
                    save_model(
                        state.trainer,
                        model_name,
                        metrics_history=state.metrics_history,
                    )
                    logger.info("Auto-saved model '%s' after training", model_name)
            except Exception as e:
                logger.error("Failed to auto-save model after training: %s", e)
            # Mark completion in the status message
            state.training_status_msg = "✅ Training complete! Model saved."
            # Hold is_training=True for up to 3s so the next timer tick
            # (which fires every 2s) sees the completed metrics.
            import time as _time
            _time.sleep(3)
            state.is_training = False

        t = threading.Thread(target=_run_and_hold, daemon=True)
        t.start()
        return state.training_status_msg

    def stop_training():
        if not state.is_training:
            return "ℹ️ No training in progress."
        state.stop_requested = True
        return "🛑 Stop requested. Finishing current step..."

    def get_training_status():
        """Return current training status message for live UI updates."""
        if state.is_training:
            return state.training_status_msg or "🏋️ Training..."
        # If not training but we have a completion message, show it
        if state.training_status_msg and "complete" in state.training_status_msg.lower():
            return state.training_status_msg
        # Idle state — only show if we previously had a status
        if state.training_status_msg:
            return state.training_status_msg
        return ""

    def refresh_charts():
        return _make_loss_fig(state), _make_confidence_fig(state), _make_experts_fig(state), _get_stats_markdown(state)

    def spawn_expert_manual(domain_idx: int):
        if state.trainer is None:
            return "❌ No model loaded. Create or load a model first."
        d = int(domain_idx)
        gc = state.trainer.gc
        cache = state.trainer.expert_cache
        device = state.trainer.device
        num_d = state.trainer.router.num_domains
        if d < 0 or d >= num_d:
            return f"❌ Invalid domain {d}. Must be 0–{num_d - 1}."
        expert = gc.spawn_expert(d, device=device)
        cache.put(expert)
        return f"✅ Spawned expert `{expert.metadata.expert_id}` in domain_{d}"

    def consolidate_now():
        if state.trainer is None:
            return "❌ No model loaded. Create or load a model first."
        count = state.trainer.consolidate_if_needed()
        return f"🔗 Consolidation pass complete: {count} merges performed."

    def prune_now():
        if state.trainer is None:
            return "❌ No model loaded. Create or load a model first."
        pruned = state.trainer.prune_underutilized()
        if not pruned:
            return "ℹ️ No underutilized experts found."
        return f"🗑️ Pruned {len(pruned)} experts: {', '.join(pruned[:5])}{'...' if len(pruned) > 5 else ''}"

    def get_expert_table():
        if state.trainer is None:
            return "*No model loaded.*"
        gc = state.trainer.gc
        rows = []
        for eid, exp in gc._active_experts.items():
            stats = exp.metadata.activation_stats
            rows.append({
                "ID": eid[:12],
                "Domain": exp.metadata.domain_cluster,
                "State": exp.state.value,
                "Tokens": stats.total_tokens_processed,
                "Avg Conf": f"{stats.avg_confidence:.3f}",
            })
        if not rows:
            return "*No active experts yet.*"
        # Build markdown table
        header = "| " + " | ".join(rows[0].keys()) + " |"
        sep = "|" + "|".join(["---"] * len(rows[0])) + "|"
        body = "\n".join("| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows)
        return f"{header}\n{sep}\n{body}"

    # --- Chat callback ---

    def chat_respond(message: str, history: list) -> tuple[list, str]:
        """Temporary chat that routes through the expert architecture.

        This is a session-only conversational interface — history disappears
        when the browser tab is closed or refreshed. It uses the trained
        WorkingMemory + Router to select a domain/expert, then generates
        a lightweight contextual reply acknowledging the user's message.
        No persistent memory or external LLM is used.

        Uses Gradio 6.x messages format: list of dicts with 'role' and 'content'.
        """
        if not message or not message.strip():
            return history, ""

        if state.trainer is None:
            response = "❌ No model loaded. Create or load a model in the Model Management tab first."
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
            return history, ""

        gc = state.trainer.gc
        router = state.trainer.router
        wm = state.trainer.wm
        device = state.trainer.device
        num_d = router.num_domains

        # --- Real routing through WM + Router ---
        try:
            # Tokenize message into token IDs (simple char-level fallback)
            msg_text = message.strip()
            # Use ingestion tokenizer if available, else simple char-to-int mapping
            if state.ingestion is not None and hasattr(state.ingestion, '_tokenizer'):
                toks = state.ingestion._tokenizer.encode(msg_text, add_special_tokens=False)
                if not toks:
                    toks = [ord(c) % 50257 for c in msg_text[:64]]
            else:
                # Fallback: map chars to vocab range
                toks = [ord(c) % 50257 for c in msg_text[:64]]

            # Pad/truncate to seq_len=64
            seq_len = 64
            if len(toks) < seq_len:
                toks = toks + [0] * (seq_len - len(toks))
            else:
                toks = toks[:seq_len]

            input_ids = torch.tensor([toks], dtype=torch.long, device=device)

            with torch.no_grad():
                routing_hidden = wm.get_routing_hidden(input_ids)  # [1, hidden_dim]
                expert_ids, confidences, domain_indices = router(routing_hidden)

            domain_idx = domain_indices[0].item()
            top_eid = expert_ids[0, 0].item()
            top_conf = confidences[0, 0].item()

            # Resolve expert UUID for display
            uuid_key = gc._global_id_to_uuid.get(top_eid, None)
            expert_short = uuid_key[:12] if uuid_key else f"#{top_eid}"

        except Exception as e:
            logger.warning("Chat routing failed, falling back to hash: %s", e)
            domain_idx = hash(msg_text) % num_d
            active_ids = list(gc._active_experts.keys())
            expert_short = active_ids[0][:12] if active_ids else "none"
            top_conf = 0.0

        # --- Autonomous text generation via LM head ---
        total_tokens = state.trainer._total_tokens
        domain_label = f"domain_{domain_idx}"

        try:
            # Build prompt tokens (no padding — generate uses raw sequence)
            if state.ingestion is not None and hasattr(state.ingestion, '_tokenizer'):
                prompt_toks = state.ingestion._tokenizer.encode(msg_text, add_special_tokens=False)
            else:
                prompt_toks = [ord(c) % 50257 for c in msg_text[:64]]

            if not prompt_toks:
                prompt_toks = [0]

            prompt_ids = torch.tensor([prompt_toks], dtype=torch.long, device=device)

            with torch.no_grad():
                generated_ids = wm.generate(
                    prompt_ids,
                    max_new_tokens=48,
                    temperature=0.8,
                    top_k=40,
                    eos_token_id=50256,
                )

            # Decode only the newly generated tokens (skip prompt)
            new_tokens = generated_ids[0, len(prompt_toks):]
            if state.ingestion is not None and hasattr(state.ingestion, '_tokenizer'):
                reply = state.ingestion._tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True).strip()
            else:
                reply = "".join(chr(t.item()) if 32 <= t.item() < 127 else "" for t in new_tokens).strip()

            if not reply:
                reply = "(generated empty response — model may need more training)"

        except Exception as e:
            logger.warning("Chat generation failed: %s", e)
            reply = f"(generation error: {e})"

        response = (
            f"{reply}\n\n"
            f"---\n"
            f"🧠 Expert `{expert_short}` ({domain_label}) | "
            f"Confidence: {top_conf:.2f} | "
            f"Tokens: {total_tokens:,}\n"
            f"*Session-only — disappears on tab close.*"
        )

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        return history, ""

    # --- Model Management Callbacks ---

    def create_new_model(
        model_name: str,
        h_dim: int, n_layers: int, n_heads: int, ffn_d: int, lora_r: int,
        n_domains: int, max_exp: int, lr: float, ewc_l: float,
    ) -> str:
        name = (model_name or "").strip()
        if not name:
            return "❌ Model name is required."
        if not re.match(r'^[A-Za-z0-9_\-]+$', name):
            return "❌ Invalid name. Use only letters, digits, hyphens, and underscores."
        if state.trainer is not None:
            return "⚠️ A model is already loaded. Save or discard it before creating a new one."

        try:
            device = torch.device(device_str)
            wm = WorkingMemory(
                vocab_size=50257,
                hidden_dim=int(h_dim),
                num_layers=int(n_layers),
                num_heads=int(n_heads),
                ffn_dim=int(ffn_d),
                max_seq_len=64,
                lora_rank=int(lora_r),
                dropout=0.1,
            ).to(device)
            wm.freeze_base()
            wm.unfreeze_lora()

            router = HierarchicalRouter(
                hidden_dim=int(h_dim),
                num_domains=int(n_domains),
                max_experts_per_domain=int(max_exp),
                top_k=2,
            ).to(device)

            gc = GrowthController(
                router=router,
                hidden_dim=int(h_dim),
                expert_dim=int(h_dim),
                ffn_dim=int(ffn_d),
                confidence_threshold=0.4,
                low_confidence_window=10,
                min_steps_before_spawn=5,
            )

            cache = ExpertCache(max_experts=8, device=device)

            trainer = LatticeGrowTrainer(
                working_memory=wm,
                router=router,
                growth_controller=gc,
                expert_cache=cache,
                device=device,
                learning_rate=float(lr),
                ewc_lambda=float(ewc_l),
                expert_output_dim=int(h_dim),
            )

            ingestion = TextIngestionPipeline(seq_len=64, batch_size=4)

            state.trainer = trainer
            state.ingestion = ingestion
            state.metrics_history.clear()
            state.pending_texts.clear()
            state._current_model_name = name  # type: ignore[attr-defined]

            # Auto-save so the model appears in the saved models list immediately
            try:
                save_path = save_model(trainer, name)
                save_msg = f"  • Auto-saved to `{save_path}`"
            except Exception as save_err:
                logger.warning("Auto-save on create failed: %s", save_err)
                save_msg = f"  • ⚠️ Auto-save failed: {save_err}. Click 💾 Save Current manually."

            return (
                f"✅ Created model `{name}`\n"
                f"  • hidden={int(h_dim)}, layers={int(n_layers)}, heads={int(n_heads)}\n"
                f"  • domains={int(n_domains)}, max_experts/domain={int(max_exp)}\n"
                f"  • lr={float(lr):.1e}, ewc_lambda={float(ewc_l):.0f}\n"
                f"{save_msg}\n\n"
                f"Go to the **Training & Monitoring** tab to start teaching it."
            )
        except Exception as e:
            logger.exception("Failed to create model")
            return f"❌ Error creating model: {e}"

    def load_selected_model(model_name: str) -> str:
        name = (model_name or "").strip()
        if not name:
            return "⚠️ Select a model from the dropdown first."
        if state.trainer is not None:
            return "⚠️ A model is already loaded. Save or discard it before loading another."
        try:
            device = torch.device(device_str)
            trainer = load_model(name, device=device)
            ingestion = TextIngestionPipeline(seq_len=64, batch_size=4)
            state.trainer = trainer
            state.ingestion = ingestion
            # Restore persisted metrics history so charts show growth across sessions
            restored_metrics = load_metrics_history(name)
            state.metrics_history.clear()
            state.metrics_history.extend(restored_metrics)
            state.pending_texts.clear()
            state._current_model_name = name  # type: ignore[attr-defined]
            return (
                f"✅ Loaded model `{name}`\n"
                f"  • Steps: {trainer._step_count:,}\n"
                f"  • Tokens: {trainer._total_tokens:,}\n"
                f"  • Experts: {len(trainer.gc._active_experts)}\n"
                f"  • Chart points restored: {len(restored_metrics)}"
            )
        except Exception as e:
            logger.exception("Failed to load model")
            return f"❌ Error loading model: {e}"

    def save_current_model(save_name: str) -> str:
        if state.trainer is None:
            return "❌ No model loaded to save."
        name = (save_name or "").strip()
        if not name:
            # Reuse the current model name if available; only generate
            # a timestamped name when saving a brand-new unsaved model.
            existing = getattr(state, '_current_model_name', '')
            if existing and existing != 'unsaved':
                name = existing
            else:
                import datetime
                name = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not re.match(r'^[A-Za-z0-9_\-]+$', name):
            return "❌ Invalid name. Use only letters, digits, hyphens, and underscores."
        try:
            path = save_model(state.trainer, name, metrics_history=state.metrics_history)
            state._current_model_name = name  # type: ignore[attr-defined]
            return f"✅ Saved model `{name}` to `{path}`"
        except Exception as e:
            logger.exception("Failed to save model")
            return f"❌ Error saving model: {e}"

    def discard_current_model() -> str:
        if state.trainer is None:
            return "ℹ️ Nothing loaded — clean already."
        name = getattr(state, '_current_model_name', 'unsaved')
        state.trainer = None
        state.ingestion = None
        state.metrics_history.clear()
        state.pending_texts.clear()
        state._current_model_name = ''  # type: ignore[attr-defined]
        try:
            state.cache = None  # type: ignore[attr-defined]
        except Exception:
            pass
        return f"🗑️ Discarded session `{name}` (unsaved work dropped). Clean — make or load next."

    def delete_selected_model(model_name: str) -> str:
        name = (model_name or "").strip()
        if not name:
            return "⚠️ Select a model from the dropdown first."
        try:
            deleted = delete_model(name)
            if deleted:
                return f"🗑️ Deleted model `{name}`."
            else:
                return f"⚠️ Model `{name}` not found on disk."
        except Exception as e:
            logger.exception("Failed to delete model")
            return f"❌ Error deleting model: {e}"

    def refresh_model_dropdown():
        models = list_saved_models()
        return gr.update(choices=models, value=None)

    def get_model_info_md() -> str:
        if state.trainer is None:
            return "*No model loaded. Create or load a model above.*"
        t = state.trainer
        name = getattr(state, '_current_model_name', 'unsaved')
        wm = t.wm
        r = t.router
        lines = [
            f"### Current Model: `{name}`",
            f"| Parameter | Value |",
            f"|---|---|",
            f"| Hidden Dim | {wm.hidden_dim} |",
            f"| Layers | {wm.num_layers} |",
            f"| Heads | {wm.layers[0].attn.num_heads if wm.layers else 'N/A'} |",
            f"| LoRA Rank | {wm.layers[0].attn_out.lora_a.out_features if wm.layers else 'N/A'} |",
            f"| Domains | {r.num_domains} |",
            f"| Max Experts/Domain | {r.max_experts_per_domain} |",
            f"| Active Experts | {len(t.gc._active_experts)} |",
            f"| Training Steps | {t._step_count:,} |",
            f"| Total Tokens | {t._total_tokens:,} |",
            f"| Device | {t.device} |",
        ]
        return "\n".join(lines)

    # --- Layout with tabs ---

    with gr.Blocks(title="Lattice-Grow Dashboard") as demo:
        gr.Markdown("# 🧬 Lattice-Grow Prototype Dashboard")
        header_md = gr.Markdown(f"Device: **{device_str}** | *Create or load a model to begin*")

        with gr.Tabs(selected="model_mgmt_tab"):
            # ---- Model Management Tab (FIRST, default) ----
            with gr.TabItem("🧠 Model Management", id="model_mgmt_tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ✨ Create New Model")
                        create_name_input = gr.Textbox(
                            label="Model Name",
                            placeholder="my_first_model",
                            info="Letters, digits, hyphens, underscores only.",
                        )
                        with gr.Accordion("Advanced Configuration", open=False):
                            create_h_dim = gr.Number(value=768, label="Hidden Dim", minimum=64)
                            create_n_layers = gr.Number(value=12, label="Num Layers", minimum=1)
                            create_n_heads = gr.Number(value=12, label="Num Heads", minimum=1)
                            create_ffn_d = gr.Number(value=3072, label="FFN Dim", minimum=64)
                            create_lora_r = gr.Number(value=16, label="LoRA Rank", minimum=1)
                            create_n_domains = gr.Number(value=8, label="Num Domains", minimum=1)
                            create_max_exp = gr.Number(value=32, label="Max Experts/Domain", minimum=1)
                            create_lr = gr.Number(value=3e-4, label="Learning Rate", minimum=1e-6)
                            create_ewc_l = gr.Number(value=5000.0, label="EWC Lambda", minimum=0.0)
                        create_btn = gr.Button("🏗️ Create Model", variant="primary")
                        create_output = gr.Textbox(label="Status", interactive=False)
                        gr.Markdown("---")
                        gr.Markdown("### 📂 Load / Save / Delete")
                        load_dropdown = gr.Dropdown(
                            choices=list_saved_models(),
                            label="Saved Models",
                            interactive=True,
                        )
                        refresh_dd_btn = gr.Button("🔄 Refresh List")
                        refresh_dd_btn.click(refresh_model_dropdown, outputs=[load_dropdown])

                        with gr.Row():
                            load_btn = gr.Button("📥 Load", variant="secondary")
                            save_btn = gr.Button("💾 Save Current", variant="secondary")
                            delete_btn = gr.Button("🗑️ Delete", variant="stop")
                        with gr.Row():
                            discard_btn = gr.Button("🧹 Discard Session (unload, start clean)")

                        save_name_input = gr.Textbox(
                            label="Save As (optional)",
                            placeholder="Leave empty for timestamped name",
                        )
                        mgmt_status = gr.Textbox(label="Management Status", interactive=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### ℹ️ Model Info")
                        model_info_md = gr.Markdown(get_model_info_md())
                        gr.Markdown("### 📊 Experts")
                        mgmt_experts_plot = gr.Plot(_make_experts_fig(state))
                        mgmt_expert_table = gr.Markdown(get_expert_table())

                    def _refresh_mgmt_panel():
                        return get_model_info_md(), _make_experts_fig(state), get_expert_table()

                    def _create_and_refresh(*args):
                        msg = create_new_model(*args)
                        dd = refresh_model_dropdown()
                        info, fig, tbl = _refresh_mgmt_panel()
                        return msg, dd, info, fig, tbl

                    def _load_and_refresh(model_name):
                        msg = load_selected_model(model_name)
                        info, fig, tbl = _refresh_mgmt_panel()
                        return msg, info, fig, tbl

                    def _delete_and_refresh(model_name):
                        msg = delete_selected_model(model_name)
                        dd = refresh_model_dropdown()
                        info, fig, tbl = _refresh_mgmt_panel()
                        return msg, dd, info, fig, tbl

                    # Wire callbacks AFTER model_info_md is defined
                    # NOTE: domain_input is defined later in Training tab; use string reference
                    def _update_domain_max_after_load():
                        if state.trainer is None:
                            return gr.update()
                        max_d = max(0, state.trainer.router.num_domains - 1)
                        return gr.update(maximum=max_d)

                    load_btn.click(
                        _load_and_refresh, [load_dropdown], [mgmt_status, model_info_md, mgmt_experts_plot, mgmt_expert_table]
                    )
                    load_btn.click(lambda: f"Device: **{device_str}** | *Model loaded*", outputs=[header_md])

                    save_btn.click(save_current_model, [save_name_input], [mgmt_status])
                    save_btn.click(refresh_model_dropdown, outputs=[load_dropdown])

                    delete_btn.click(
                        _delete_and_refresh, [load_dropdown], [mgmt_status, load_dropdown, model_info_md, mgmt_experts_plot, mgmt_expert_table]
                    )

                    discard_btn.click(
                        lambda: (discard_current_model(),) + _refresh_mgmt_panel(),
                        outputs=[mgmt_status, model_info_md, mgmt_experts_plot, mgmt_expert_table],
                    )

                    create_btn.click(
                        _create_and_refresh,
                        [
                            create_name_input,
                            create_h_dim, create_n_layers, create_n_heads, create_ffn_d, create_lora_r,
                            create_n_domains, create_max_exp, create_lr, create_ewc_l,
                        ],
                        [create_output, load_dropdown, model_info_md, mgmt_experts_plot, mgmt_expert_table],
                    )
                    create_btn.click(lambda: f"Device: **{device_str}** | *Model loaded*", outputs=[header_md])

            # ---- Chat / Talk Tab ----
            with gr.TabItem("💬 Choose & Talk", id="chat_tab"):
                gr.Markdown(
                    "### 💬 Choose a Model and Talk to It\n"
                    "*Select a saved model (or use the currently loaded one) and chat with it. "
                    "This prototype routes your message through the expert architecture.*"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        chat_model_dropdown = gr.Dropdown(
                            choices=list_saved_models(),
                            label="Select Model",
                            interactive=True,
                            info="Pick a saved model to load, or leave blank to use the current session.",
                        )
                        chat_refresh_btn = gr.Button("🔄 Refresh Models")
                        chat_load_btn = gr.Button("📥 Load Selected Model", variant="secondary")
                        chat_load_status = gr.Textbox(label="Load Status", interactive=False)

                        def _chat_load_model(model_name):
                            if not model_name:
                                return "ℹ️ No model selected. Using currently loaded model (if any)."
                            try:
                                device = torch.device(device_str)
                                trainer = load_model(model_name, device=device)
                                state.trainer = trainer
                                # Restore persisted metrics history so charts show growth across sessions
                                restored_metrics = load_metrics_history(model_name)
                                state.metrics_history.clear()
                                state.metrics_history.extend(restored_metrics)
                                # Re-init ingestion pipeline for the new model
                                try:
                                    from src.lattice_grow.ingestion import TextIngestionPipeline
                                    state.ingestion = TextIngestionPipeline(seq_len=64, batch_size=4)
                                except Exception as ing_err:
                                    logger.warning("Ingestion init failed in chat load: %s", ing_err)
                                    state.ingestion = None
                                state._current_model_name = model_name  # type: ignore[attr-defined]
                                return (
                                    f"✅ Loaded model '{model_name}' successfully.\n"
                                    f"  • Chart points restored: {len(restored_metrics)}"
                                )
                            except Exception as e:
                                return f"❌ Failed to load '{model_name}': {e}"

                        def _chat_refresh_models():
                            return gr.update(choices=list_saved_models())

                        chat_refresh_btn.click(_chat_refresh_models, outputs=[chat_model_dropdown])
                        chat_load_btn.click(
                            _chat_load_model,
                            [chat_model_dropdown],
                            [chat_load_status],
                        ).then(get_model_info_md, outputs=[model_info_md])

                        gr.Markdown("---")
                        gr.Markdown("**Current Model Info:**")
                        chat_model_info = gr.Markdown(get_model_info_md())
                        # Keep chat model info in sync when loading from this tab
                        chat_load_btn.click(get_model_info_md, outputs=[chat_model_info])

                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=450,
                        )
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="Your Message",
                                placeholder="Type something to talk to the model...",
                                lines=2,
                                scale=5,
                            )
                            chat_send_btn = gr.Button("📨 Send", variant="primary", scale=1)
                        with gr.Row():
                            chat_clear_btn = gr.Button("🗑️ Clear Chat", variant="stop")

                        chat_send_btn.click(
                            chat_respond,
                            [chat_input, chatbot],
                            [chatbot, chat_input],
                        )
                        chat_input.submit(
                            chat_respond,
                            [chat_input, chatbot],
                            [chatbot, chat_input],
                        )
                        chat_clear_btn.click(
                            lambda: ([], ""),
                            outputs=[chatbot, chat_input],
                        )

            # ---- Training & Monitoring Tab ----
            with gr.TabItem("📊 Training & Monitoring", id="training_tab"):
                gr.Markdown(
                    "### 🗺️ Workflow Guide\n"
                    "**Step 1:** Add data (upload files or crawl the web) → "
                    "**Step 2:** Train the model on that data → "
                    "**Step 3:** Monitor metrics and manage experts."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("📚 Step 1: Add Training Data", open=True):
                            gr.Markdown(
                                "*Upload documents or fetch web pages to build the training corpus.*"
                            )
                            file_upload = gr.File(
                                label="Upload Documents (.txt, .md, .pdf, .json, .csv)",
                                file_count="multiple",
                                file_types=[".txt", ".md", ".pdf", ".json", ".csv"],
                            )
                            url_input = gr.Textbox(
                                label="Web URLs (one per line)",
                                placeholder="https://example.com/article\nhttps://arxiv.org/abs/...",
                                lines=2,
                            )
                            ingest_btn = gr.Button("🚀 Ingest Data", variant="primary")
                            ingest_output = gr.Textbox(label="Ingestion Status", interactive=False)

                        def ingest_data(files, urls_text):
                            # Auto-initialize ingestion if trainer is loaded but ingestion wasn't set
                            if state.ingestion is None and state.trainer is not None:
                                try:
                                    from src.lattice_grow.ingestion import TextIngestionPipeline
                                    state.ingestion = TextIngestionPipeline(seq_len=64, batch_size=4)
                                    logger.info("Auto-initialized ingestion pipeline for training tab")
                                except Exception as ing_err:
                                    logger.warning("Auto-init ingestion failed: %s", ing_err)
                            if state.ingestion is None:
                                return "❌ Ingestion pipeline not available. Load a model first (Model Management or Chat tab), then try again."

                            results = []
                            texts_added = 0

                            # Process uploaded files
                            if files:
                                for f in files:
                                    path = getattr(f, "name", None)
                                    if not path:
                                        continue
                                    try:
                                        text = state.ingestion.read_file(path)
                                        if text and text.strip():
                                            with state.lock:
                                                state.pending_texts.append(text)
                                            texts_added += 1
                                            chars = min(len(text), state.ingestion.max_text_per_source)
                                            results.append(f"📄 Read `{Path(path).name}` ({chars:,} chars)")
                                        else:
                                            results.append(f"⚠️ Empty or unsupported: `{Path(path).name}`")
                                    except Exception as e:
                                        logger.error("Failed to read %s: %s", path, e)
                                        results.append(f"❌ Error reading `{Path(path).name}`: {e}")

                            # Process URLs
                            if urls_text and urls_text.strip():
                                for line in urls_text.strip().splitlines():
                                    url = line.strip()
                                    if not url:
                                        continue
                                    try:
                                        text = state.ingestion.fetch_url(url)
                                        if text and text.strip():
                                            with state.lock:
                                                state.pending_texts.append(text)
                                            texts_added += 1
                                            chars = min(len(text), state.ingestion.max_text_per_source)
                                            results.append(f"🌐 Fetched `{url}` ({chars:,} chars)")
                                        else:
                                            results.append(f"⚠️ No content extracted from `{url}`")
                                    except Exception as e:
                                        logger.error("Failed to fetch %s: %s", url, e)
                                        results.append(f"❌ Error fetching `{url}`: {e}")

                            if texts_added == 0 and not results:
                                return "⚠️ No files or URLs provided."

                            total_sources = len(state.pending_texts)
                            header = f"✅ Added {texts_added} source(s). Total sources ready: {total_sources}.\n\n"
                            header += "Next: Open **Step 2: Train** below and click **▶ Start Training**."
                            return header + "\n" + "\n".join(results)

                        ingest_btn.click(
                            ingest_data,
                            [file_upload, url_input],
                            ingest_output,
                        )

                        with gr.Accordion("🌐 Step 1b: Autonomous Web Learning (Optional)", open=False):
                            gr.Markdown(
                                "*Crawl the web to automatically gather training data from seed URLs.*"
                            )
                            crawl_seeds_input = gr.Textbox(
                                label="Seed URLs (one per line)",
                                placeholder="https://en.wikipedia.org/wiki/Machine_learning",
                                lines=2,
                            )
                            with gr.Row():
                                crawl_max_pages = gr.Number(value=20, label="Max Pages", minimum=1, maximum=500)
                                crawl_depth = gr.Number(value=1, label="Depth", minimum=0, maximum=5)
                                crawl_delay = gr.Number(value=3.0, label="Delay (s)", minimum=0.5)
                            crawl_same_domain = gr.Checkbox(value=True, label="Same domain only")
                            with gr.Row():
                                crawl_start_btn = gr.Button("🕷️ Start Crawler", variant="primary")
                                crawl_stop_btn = gr.Button("⏹ Stop", variant="stop")
                            crawl_status = gr.Textbox(label="Crawler Status", interactive=False)

                        def start_autonomous_learning(seeds_text, max_pages, depth, delay, same_domain):
                            if state.trainer is None:
                                return "❌ No model loaded. Create or load a model first."
                            if state.ingestion is None:
                                return "❌ Ingestion pipeline not available."
                            if not seeds_text or not seeds_text.strip():
                                return "❌ Provide at least one seed URL."

                            # Stop existing crawler if running
                            if state.crawler is not None and state.crawler.is_running:
                                state.crawler.stop()
                                time.sleep(0.5)

                            seed_urls = [
                                line.strip()
                                for line in seeds_text.strip().splitlines()
                                if line.strip()
                            ]
                            if not seed_urls:
                                return "❌ No valid seed URLs found."

                            config = CrawlerConfig(
                                max_pages=int(max_pages),
                                max_depth=int(depth),
                                delay_between_requests=float(delay),
                                same_domain_only=bool(same_domain),
                            )
                            crawler = AutonomousCrawler(config)

                            def text_callback(text: str) -> None:
                                with state.lock:
                                    state.pending_texts.append(text)

                            def status_callback(msg: str) -> None:
                                # Status updates are picked up by the timer tick
                                pass

                            crawler.start(seed_urls, text_callback, status_callback)
                            state.crawler = crawler
                            return (
                                f"🕷️ Crawler started: {len(seed_urls)} seed(s), "
                                f"max {config.max_pages} pages, depth {config.max_depth}, "
                                f"delay {config.delay_between_requests}s"
                            )

                        def stop_autonomous_learning():
                            if state.crawler is None or not state.crawler.is_running:
                                return "ℹ️ No crawler is currently running."
                            state.crawler.stop()
                            return "⏹ Stop signal sent to crawler."

                        def get_crawler_status():
                            if state.crawler is None:
                                return "Idle — no crawler configured."
                            running = state.crawler.is_running
                            pages = state.crawler.pages_fetched
                            status = state.crawler.status
                            prefix = "🟢 Running" if running else "⚪ Stopped"
                            pending = len(state.pending_texts)
                            return f"{prefix} | {status} | Pages fetched: {pages} | Pending texts: {pending}"

                        crawl_start_btn.click(
                            start_autonomous_learning,
                            [crawl_seeds_input, crawl_max_pages, crawl_depth, crawl_delay, crawl_same_domain],
                            [crawl_status],
                        )
                        crawl_stop_btn.click(stop_autonomous_learning, outputs=[crawl_status])

                        with gr.Accordion("🏋️ Step 2: Train the Model", open=True):
                            gr.Markdown(
                                "*Run training steps on all ingested data. Adjust hyperparameters as needed.*"
                            )
                            with gr.Row():
                                num_steps_input = gr.Number(value=100, label="Steps", minimum=1)
                                batch_size_input = gr.Number(value=4, label="Batch", minimum=1)
                                seq_len_input = gr.Number(value=64, label="Seq Len", minimum=1)
                            with gr.Row():
                                start_btn = gr.Button("▶ Start Training", variant="primary")
                                stop_btn = gr.Button("⏹ Stop", variant="stop")
                            status_text = gr.Textbox(label="Training Status", interactive=False)

                            start_btn.click(start_training, [num_steps_input, batch_size_input, seq_len_input], status_text)
                            stop_btn.click(stop_training, outputs=status_text)

                        with gr.Accordion("🔧 Step 3: Expert Management (Advanced)", open=False):
                            gr.Markdown(
                                "*Manually spawn, consolidate, or prune experts. Usually handled automatically during training.*"
                            )
                            with gr.Row():
                                domain_input = gr.Number(value=0, label="Domain Index", minimum=0, maximum=7)
                                spawn_btn = gr.Button("➕ Spawn Expert")
                            spawn_output = gr.Textbox(label="Result", interactive=False)
                            spawn_btn.click(spawn_expert_manual, domain_input, spawn_output)

                            # Wire dynamic domain max updates now that domain_input exists
                            load_btn.click(_update_domain_max_after_load, outputs=[domain_input])
                            create_btn.click(_update_domain_max_after_load, outputs=[domain_input])

                            with gr.Row():
                                consolidate_btn = gr.Button("🔗 Consolidate")
                                prune_btn = gr.Button("🗑️ Prune")
                            mgmt_output = gr.Textbox(label="Result", interactive=False)
                            consolidate_btn.click(consolidate_now, outputs=mgmt_output)
                            prune_btn.click(prune_now, outputs=mgmt_output)

                        refresh_btn = gr.Button("🔄 Refresh All", variant="secondary")

                    with gr.Column(scale=2):
                        stats_md = gr.Markdown(_get_stats_markdown(state))
                        loss_plot = gr.Plot(_make_loss_fig(state))
                        with gr.Row():
                            conf_plot = gr.Plot(_make_confidence_fig(state))
                            experts_plot = gr.Plot(_make_experts_fig(state))

                        gr.Markdown("### 📋 Active Experts")
                        expert_table = gr.Markdown(get_expert_table())

                # Auto-refresh every 2 seconds while training
                timer = gr.Timer(value=2)
                timer.tick(
                    refresh_charts,
                    outputs=[loss_plot, conf_plot, experts_plot, stats_md],
                )
                # Live training status update (progress / completion)
                timer.tick(get_training_status, outputs=[status_text])
                refresh_btn.click(
                    refresh_charts,
                    outputs=[loss_plot, conf_plot, experts_plot, stats_md],
                )
                # Also refresh expert table on timer
                timer.tick(get_expert_table, outputs=expert_table)
                refresh_btn.click(get_expert_table, outputs=expert_table)

                # Also refresh crawler status on timer
                timer.tick(get_crawler_status, outputs=[crawl_status])

            # ---- Self-Improve Tab (FOURTH): memory strengthen + online learning ----
            with gr.TabItem("🧠 Self-Improve", id="self_tab"):
                gr.Markdown(
                    "### 🧠 Teach Itself, Then Learn Online\n"
                    "*Strengthen memory (merge twins, retire sleepers) and let the model "
                    "fetch its own answers from the web about questions it can't answer.*"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 💪 Strengthen Memory")
                        gr.Markdown(
                            "*Leader-led reasoning pass: distributor merges twin experts, "
                            "sleepers pruned, room recentered. Whole model improves itself.*"
                        )
                        strengthen_btn = gr.Button("💪 Strengthen Now", variant="primary")
                        strengthen_out = gr.Textbox(label="Result", interactive=False)

                        def strengthen_memory():
                            if state.trainer is None:
                                return "❌ No model loaded. Create or load one first."
                            merged = state.trainer.consolidate_if_needed()
                            pruned = state.trainer.prune_underutilized()
                            n = len(state.trainer.gc._active_experts)
                            return (
                                f"✅ Strengthened: {merged} twin merge(s), "
                                f"{len(pruned)} sleeper(s) retired, {n} experts standing. "
                                f"Memory tighter, room sharper."
                            )

                        strengthen_btn.click(strengthen_memory, outputs=[strengthen_out])

                        gr.Markdown("### 🤖 Autonomous Switch")
                        auto_switch = gr.Checkbox(
                            value=False,
                            label="Self-clean while training (prune sleepers every 50 steps)",
                        )

                        def _flip_autonomous(on: bool):
                            state.autonomous = bool(on)
                            return "🟢 Autonomous ON — room self-cleans mid-training." if on else "⚪ Autonomous OFF."

                        auto_status = gr.Textbox(label="Autonomous", interactive=False)
                        auto_switch.change(_flip_autonomous, [auto_switch], [auto_status])

                    with gr.Column(scale=1):
                        gr.Markdown("### 🌐 Online Learning — Feed Its Curiosity")
                        gr.Markdown(
                            "*Type what it wonders about + seed pages. It crawls, reads, "
                            "and queues the knowledge for the next training run — by itself.*"
                        )
                        wonder_input = gr.Textbox(
                            label="What does it wonder about?",
                            placeholder="e.g. how do black holes evaporate",
                            lines=2,
                        )
                        wonder_seeds = gr.Textbox(
                            label="Seed URLs to learn from (one per line)",
                            placeholder="https://en.wikipedia.org/wiki/Hawking_radiation",
                            lines=3,
                        )
                        wonder_btn = gr.Button("🧠 Go Learn Online", variant="primary")
                        wonder_out = gr.Textbox(label="Online Learning", interactive=False)

                        def _wonder_learn(question, seeds):
                            if state.trainer is None:
                                return "❌ No model loaded. Create or load one first."
                            if not seeds or not seeds.strip():
                                return "❌ Give at least one seed URL to learn from."
                            q = (question or "").strip()
                            msg = start_autonomous_learning(seeds, 20, 1, 3.0, True)
                            if q:
                                return f"🔎 Curiosity logged: “{q}”\n{msg}\nTexts land in pending queue by themselves."
                            return msg

                        wonder_btn.click(_wonder_learn, [wonder_input, wonder_seeds], [wonder_out])

    return demo


def launch_dashboard(**kwargs) -> None:
    """Convenience entry point: build and launch the dashboard."""
    demo = create_dashboard(**kwargs)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
