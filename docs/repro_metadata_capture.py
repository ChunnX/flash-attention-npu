#!/usr/bin/env python3
"""Minimal reproduction: get_scheduler_metadata cannot be captured into an NPUGraph.

Two capture attempts that differ only in where the metadata is built:

  A. metadata built outside the capture region -- what the existing graph tests do
  B. metadata built inside it -- what a caller with device-side sequence lengths needs

A passes and B fails with runtime 207000 at aclrtRecordEvent, because
GetSchedulerMetadataImpl launches the AICPU kernel on a stream from the pool and
joins it to the current stream with a pair of events, which a capturing stream
rejects.

Usage:
    python repro_metadata_capture.py            # flash_attn_npu_4
    python repro_metadata_capture.py --version v3

B is attempted last and nothing follows it, because a failed capture leaves the
stream in capture mode for the rest of the process -- anything after it fails for a
reason that is not its own.
"""

import argparse
import importlib

import torch
import torch_npu  # noqa: F401  # registers the NPU device

# Small and flash-decode free (max KV under 1024), so nothing else is in play.
BATCH, Q_LEN = 1, 4
NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE = 8, 2, 128
BLOCK_SIZE, BLOCKS_PER_SEQ = 128, 4
KV_LEN = 256
# The metadata call derives the block-table row stride from this, so it has to be the
# page capacity the block table was allocated at, not the actual maximum KV length.
MAX_SEQLEN_K = BLOCK_SIZE * BLOCKS_PER_SEQ
SCALE = HEAD_SIZE**-0.5


def build(version):
    module_name = {"v3": "flash_attn_npu_3", "v4": "flash_attn_npu_4"}[version]
    module = importlib.import_module(module_name)
    # Say which build this is. Running from a source tree that shadows site-packages
    # picks up a different .so, and the only symptom is the operator behaving like
    # another version of itself.
    print(f"module      : {module.__file__}")

    dev = "npu"
    tensors = {
        "q": torch.randn(BATCH * Q_LEN, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=dev),
        "k": torch.randn(
            BATCH * BLOCKS_PER_SEQ, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=dev
        ),
        "page_table": torch.arange(BATCH * BLOCKS_PER_SEQ, dtype=torch.int32, device=dev).reshape(
            BATCH, BLOCKS_PER_SEQ
        ),
        "cu_seqlens_q": torch.tensor([0, Q_LEN], dtype=torch.int32, device=dev),
        "cache_seqlens": torch.tensor([KV_LEN], dtype=torch.int32, device=dev),
    }
    tensors["v"] = torch.randn_like(tensors["k"])

    def metadata():
        return module.get_scheduler_metadata(
            batch_size=BATCH,
            max_seqlen_q=Q_LEN,
            max_seqlen_k=MAX_SEQLEN_K,
            num_heads_q=NUM_HEADS,
            num_heads_kv=NUM_KV_HEADS,
            headdim=HEAD_SIZE,
            cache_seqlens=tensors["cache_seqlens"],
            qkv_dtype=torch.bfloat16,
            cu_seqlens_q=tensors["cu_seqlens_q"],
            page_size=BLOCK_SIZE,
            causal=False,
            window_size=(-1, -1),
            softmax_scale=SCALE,
        )

    def forward(meta):
        if version == "v4":
            return module.flash_attn_varlen_func(
                tensors["q"],
                tensors["k"],
                tensors["v"],
                cu_seqlens_q=tensors["cu_seqlens_q"],
                seqused_k=tensors["cache_seqlens"],
                page_table=tensors["page_table"],
                max_seqlen_q=Q_LEN,
                max_seqlen_k=MAX_SEQLEN_K,
                softmax_scale=SCALE,
                causal=False,
                window_size=(-1, -1),
                scheduler_metadata=meta,
                num_splits=0,
                return_lse=False,
            )
        # v3 reaches the paged AICPU-metadata path through flash_attn_with_kvcache;
        # its flash_attn_varlen_func takes neither page_table nor scheduler_metadata.
        # It derives and validates the KV bound itself, so max_seqlen_k is not passed.
        return module.flash_attn_with_kvcache(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            cache_seqlens=tensors["cache_seqlens"],
            page_table=tensors["page_table"],
            cu_seqlens_q=tensors["cu_seqlens_q"],
            max_seqlen_q=Q_LEN,
            softmax_scale=SCALE,
            causal=False,
            window_size=(-1, -1),
            rotary_interleaved=False,
            scheduler_metadata=meta,
            num_splits=0,
            return_softmax_lse=False,
        )

    return metadata, forward


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", choices=("v3", "v4"), default="v4")
    args = parser.parse_args()

    if torch.npu.device_count() == 0:
        print("no NPU visible")
        return 1
    print(f"device      : {torch_npu.npu.get_device_name()}")
    print(f"api         : flash_attn_npu {args.version}")

    metadata, forward = build(args.version)

    forward(metadata())
    torch.npu.synchronize()
    print("eager       : OK\n")

    print("A. metadata built OUTSIDE the capture region (what the graph tests do)")
    graph_a = torch.npu.NPUGraph()
    meta = metadata()
    with torch.npu.graph(graph_a):
        forward(meta)
    graph_a.replay()
    torch.npu.synchronize()
    print("   capture + replay: OK\n")

    print("B. metadata built INSIDE the capture region")
    print("   (expected to fail: aclrtRecordEvent -> runtime 207000, ascendc_fa_metadata)")
    graph_b = torch.npu.NPUGraph()
    with torch.npu.graph(graph_b):
        forward(metadata())
    graph_b.replay()
    torch.npu.synchronize()
    print("   capture + replay: OK -- the bug is fixed in this build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
