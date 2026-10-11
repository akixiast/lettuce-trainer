"""Data ingestion pipeline: tokenization and web fetching for Lattice-Grow.

Converts raw text (from uploaded files or fetched URLs) into tokenized
tensors that the trainer can consume. Designed to run on old laptops —
uses a small tokenizer, streams data instead of loading everything into
memory, and fetches URLs sequentially with timeouts.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import requests
import torch
import trafilatura
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Default small tokenizer that works well on low-resource machines
DEFAULT_TOKENIZER = "gpt2"  # ~50MB, fast, no special hardware needed
DEFAULT_VOCAB_SIZE = 50257
DEFAULT_SEQ_LEN = 64
DEFAULT_BATCH_SIZE = 4
FETCH_TIMEOUT = 15  # seconds per URL
MAX_TEXT_PER_SOURCE = 500_000  # chars (~100k tokens), prevents OOM on huge books


class TextIngestionPipeline:
    """Converts raw text sources into tokenized training batches.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier.
        seq_len: Sequence length for training chunks.
        batch_size: Batch size for the DataLoader.
        max_text_per_source: Max characters to read from a single source.
    """

    def __init__(
        self,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        seq_len: int = DEFAULT_SEQ_LEN,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_text_per_source: int = MAX_TEXT_PER_SOURCE,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_text_per_source = max_text_per_source

        logger.info("Loading tokenizer: %s", tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size

    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize raw text into a flat tensor of token IDs.

        Truncates to max_seq_len * 16 tokens (a generous upper bound for
        chunking) to prevent sequences longer than the model's context
        window from reaching the trainer. The chunk_tokens() method
        further splits this into seq_len-sized pieces.
        """
        # Cap tokenization length to avoid transformer indexing errors.
        # We tokenize more than seq_len because chunk_tokens splits into
        # multiple seq_len chunks, but we don't need unbounded length.
        max_tokens = self.seq_len * 16  # e.g., 64*16 = 1024 tokens max
        encoded = self.tokenizer(
            text[: self.max_text_per_source],
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
        )
        return encoded["input_ids"].squeeze(0)

    def chunk_tokens(
        self, token_ids: torch.Tensor
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Split a flat token tensor into (input_ids, labels) pairs.

        Labels are the next-token prediction target (shifted by 1).
        Yields chunks of exactly seq_len tokens.
        """
        # Need seq_len + 1 tokens for input/label pairing
        effective_len = self.seq_len + 1
        n_chunks = (len(token_ids) - 1) // self.seq_len
        for i in range(n_chunks):
            start = i * self.seq_len
            chunk = token_ids[start : start + effective_len]
            if len(chunk) < effective_len:
                break
            input_ids = chunk[:-1]
            labels = chunk[1:]
            yield input_ids, labels

    def read_file(self, file_path: str | Path) -> str:
        """Read text content from a file. Supports .txt, .md, .json, .csv.

        PDF support would require pdfminer or PyMuPDF — skipped for now
        to keep dependencies light for old laptops.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            logger.warning("PDF reading not yet supported: %s", path.name)
            return ""

        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("Failed to read %s: %s", path, e)
            return ""

    def fetch_url(self, url: str) -> str:
        """Fetch and extract main text content from a URL.

        Uses trafilatura for clean article extraction. Falls back to
        raw response text if extraction fails.
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text:
                    return text

            # Fallback: raw requests with timeout
            resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={
                "User-Agent": "LatticeGrow/0.1 (research prototype)"
            })
            resp.raise_for_status()
            return resp.text[: self.max_text_per_source]
        except Exception as e:
            logger.error("Failed to fetch %s: %s", url, e)
            return ""

    def build_dataset_from_texts(
        self, texts: List[str]
    ) -> IterableDataset:
        """Create an IterableDataset from a list of raw text strings."""
        return _TokenStreamDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            max_text_per_source=self.max_text_per_source,
        )

    def create_dataloader(self, texts: List[str]) -> DataLoader:
        """Build a DataLoader from raw text sources ready for training."""
        dataset = self.build_dataset_from_texts(texts)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # IterableDataset doesn't support shuffle
            drop_last=True,
        )


class _TokenStreamDataset(IterableDataset):
    """Streams tokenized chunks from multiple text sources.

    Memory-efficient: tokenizes one source at a time, yields chunks,
    then moves to the next source. Never holds all tokens in memory.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        seq_len: int,
        max_text_per_source: int,
    ):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_text_per_source = max_text_per_source

    def __iter__(self):
        effective_len = self.seq_len + 1
        for text in self.texts:
            # Defensive: skip non-string entries (e.g. dicts from malformed JSON)
            if not isinstance(text, str) or not text.strip():
                continue
            truncated = text[: self.max_text_per_source]
            # Cap tokenization length to avoid transformer indexing errors.
            # We tokenize more than seq_len because chunk_tokens splits into
            # multiple seq_len chunks, but we don't need unbounded length.
            max_tokens = self.seq_len * 16  # e.g., 64*16 = 1024 tokens max
            encoded = self.tokenizer(
                truncated,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
                add_special_tokens=False,
            )
            token_ids = encoded["input_ids"].squeeze(0)
            n_chunks = (len(token_ids) - 1) // self.seq_len
            for i in range(n_chunks):
                start = i * self.seq_len
                chunk = token_ids[start : start + effective_len]
                if len(chunk) < effective_len:
                    break
                yield {
                    "input_ids": chunk[:-1],
                    "labels": chunk[1:],
                }
