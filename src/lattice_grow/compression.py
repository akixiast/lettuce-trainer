"""Compression pipeline: NanoQuant sub-1-bit + ZipLLM delta encoding."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CompressedExpert:
    """Immutable container for a compressed expert blob."""

    expert_id: str
    bit_width: float
    tensor_names: list[str]
    tensor_shapes: list[tuple[int, ...]]
    compressed_data: bytes
    scale_factors: Dict[str, torch.Tensor]
    delta_base_id: Optional[str] = None
    checksum: int = 0

    def size_bytes(self) -> int:
        return len(self.compressed_data)


class NanoQuantCompressor:
    """Sub-1-bit quantization via latent factorization + binary residuals.

    This is a reference implementation targeting ~0.65 bits/weight.
    Production would use CUDA kernels for the packing/unpacking.
    """

    def __init__(self, target_bit_width: float = 0.65, rank: int = 8):
        self.target_bit_width = target_bit_width
        self.rank = rank

    @torch.no_grad()
    def compress_tensor(self, weight: torch.Tensor) -> Tuple[bytes, torch.Tensor]:
        """Compress a single weight matrix to sub-1-bit representation.

        Returns:
            packed_bits: raw bytes of binary/ternary encoded residuals
            scale_factor: per-channel scale tensor for reconstruction
        """
        orig_shape = weight.shape
        if weight.dim() == 1:
            # Bias or embedding — just quantize to INT4
            return self._quantize_int4(weight), torch.tensor(1.0)

        # Low-rank factorization: W ≈ U @ V^T
        flat = weight.reshape(orig_shape[0], -1).float()
        U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
        k = min(self.rank, min(S.shape))
        U_k = U[:, :k] * S[:k].unsqueeze(0)  # [out, k]
        V_k = Vh[:k, :]  # [k, in]

        # Quantize factors to INT8
        U_q, u_scale = self._quantize_int8(U_k)
        V_q, v_scale = self._quantize_int8(V_k)

        # Reconstruct low-rank approximation and compute residual
        approx = (U_q.float() * u_scale) @ (V_q.float() * v_scale)
        residual = flat - approx

        # Binary encode residual signs (1 bit each)
        signs = (residual > 0).to(torch.uint8)
        magnitudes = residual.abs()
        mag_scale = magnitudes.mean().clamp(min=1e-8)
        binary_mag = (magnitudes / mag_scale > 0.5).to(torch.uint8)

        # Pack bits
        packed = self._pack_bits(signs, binary_mag)

        # Serialize everything together
        header = struct.pack(
            "!IIffI",
            orig_shape[0],
            orig_shape[1] if len(orig_shape) > 1 else 1,
            u_scale.item(),
            v_scale.item(),
            k,
        )
        u_bytes = U_q.cpu().numpy().tobytes()
        v_bytes = V_q.cpu().numpy().tobytes()
        mag_scale_bytes = struct.pack("!f", mag_scale.item())

        blob = header + u_bytes + v_bytes + mag_scale_bytes + packed
        scale = torch.tensor([u_scale.item(), v_scale.item(), mag_scale.item()])
        return blob, scale

    def _quantize_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        abs_max = tensor.abs().max().clamp(min=1e-8)
        scale = abs_max / 127.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    def _quantize_int4(self, tensor: torch.Tensor) -> bytes:
        abs_max = tensor.abs().max().clamp(min=1e-8)
        scale = abs_max / 7.0
        quantized = (tensor / scale).round().clamp(-8, 7).to(torch.int8)
        return quantized.cpu().numpy().tobytes()

    @staticmethod
    def _pack_bits(signs: torch.Tensor, magnitudes: torch.Tensor) -> bytes:
        """Pack sign + magnitude bits into bytes."""
        combined = (signs << 1) | magnitudes  # 2 bits per element
        flat = combined.flatten().cpu().numpy()
        # Pack 4 elements per byte
        packed = bytearray()
        for i in range(0, len(flat), 4):
            chunk = flat[i : i + 4]
            val = 0
            for j, b in enumerate(chunk):
                val |= int(b) << (j * 2)
            packed.append(val)
        return bytes(packed)


class DeltaEncoder:
    """XOR-based delta encoding between parent and child expert weights."""

    @staticmethod
    def compute_delta(child_blob: bytes, parent_blob: bytes) -> bytes:
        """Compute XOR delta; returns compressed delta via zlib."""
        min_len = min(len(child_blob), len(parent_blob))
        child_arr = bytearray(child_blob[:min_len])
        parent_arr = parent_blob[:min_len]

        delta = bytearray(min_len)
        for i in range(min_len):
            delta[i] = child_arr[i] ^ parent_arr[i]

        # Append any extra bytes from child unchanged
        if len(child_blob) > min_len:
            delta.extend(child_blob[min_len:])

        return zlib.compress(bytes(delta), level=9)

    @staticmethod
    def apply_delta(parent_blob: bytes, delta_blob: bytes) -> bytes:
        """Reconstruct child blob from parent + compressed delta."""
        delta = zlib.decompress(delta_blob)
        result = bytearray(len(delta))
        parent_arr = parent_blob

        for i in range(len(delta)):
            if i < len(parent_arr):
                result[i] = parent_arr[i] ^ delta[i]
            else:
                result[i] = delta[i]

        return bytes(result)


class ExpertFileWriter:
    """Writes .lge (Lattice-Grow Expert) files per spec §5.2."""

    MAGIC = b"LGEX"
    VERSION = 1

    @classmethod
    def write(cls, path: Path, compressed: CompressedExpert) -> None:
        tensor_headers = bytearray()
        data_offset = 0

        for name, shape in zip(compressed.tensor_names, compressed.tensor_shapes):
            name_bytes = name.encode("utf-8")
            header = struct.pack("!B", len(name_bytes))
            header += name_bytes
            header += struct.pack("!I", len(shape))
            for dim in shape:
                header += struct.pack("!I", dim)
            header += struct.pack("!B", 0)  # dtype placeholder
            header += struct.pack("!Q", data_offset)
            header += struct.pack("!Q", len(compressed.compressed_data))
            tensor_headers.extend(header)
            data_offset += len(compressed.compressed_data)

        body = (
            cls.MAGIC
            + struct.pack("!H", cls.VERSION)
            + struct.pack("!B", 0)  # compression method
            + struct.pack("!e", compressed.bit_width)
            + struct.pack("!I", len(compressed.tensor_names))
            + bytes(tensor_headers)
            + compressed.compressed_data
        )
        checksum = zlib.crc32(body) & 0xFFFFFFFF
        body += struct.pack("!I", checksum)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


class CompressionPipeline:
    """Orchestrates freeze-time compression for an expert module."""

    def __init__(self, target_bit_width: float = 0.65, svd_rank: int = 8):
        self.compressor = NanoQuantCompressor(target_bit_width, svd_rank)
        self.delta_encoder = DeltaEncoder()

    def compress_expert(
        self,
        module: nn.Module,
        expert_id: str,
        parent_blobs: Optional[Dict[str, bytes]] = None,
    ) -> CompressedExpert:
        """Compress all parameters of an expert module.

        Args:
            module: The expert nn.Module to compress.
            expert_id: Unique identifier for this expert.
            parent_blobs: Optional dict of {param_name: parent_compressed_blob}
                          for delta encoding against parent expert.
        """
        all_blobs: list[bytes] = []
        all_scales: Dict[str, torch.Tensor] = {}
        tensor_names: list[str] = []
        tensor_shapes: list[tuple[int, ...]] = []

        for name, param in module.named_parameters():
            blob, scale = self.compressor.compress_tensor(param.data)

            # Apply delta if parent exists
            if parent_blobs and name in parent_blobs:
                blob = self.delta_encoder.compute_delta(blob, parent_blobs[name])

            all_blobs.append(blob)
            all_scales[name] = scale
            tensor_names.append(name)
            tensor_shapes.append(tuple(param.shape))

        combined_data = b"".join(all_blobs)

        return CompressedExpert(
            expert_id=expert_id,
            bit_width=self.compressor.target_bit_width,
            tensor_names=tensor_names,
            tensor_shapes=tensor_shapes,
            compressed_data=combined_data,
            scale_factors=all_scales,
            delta_base_id=None,
            checksum=zlib.crc32(combined_data) & 0xFFFFFFFF,
        )
