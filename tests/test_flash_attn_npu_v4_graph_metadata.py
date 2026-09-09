# Copyright (c) 2026, Minghua Shen.
"""Does the AICPU scheduler-metadata path survive NPUGraph capture?

`test_flash_attn_npu_v4_graph.py` captures only the forward: `get_scheduler_metadata`
runs outside `torch.npu.graph`, so the tiling in the captured buffer is frozen at
whatever the KV lengths were during capture. A parallel-drafting draft cannot work
that way -- its KV lengths are written by the verify kernel on device and change
every step, so the metadata op has to be *inside* the graph and re-read them on
replay. That is what the FIA sink operator does today, and it is the one property
this backend has to match to replace it.

So: capture metadata + forward together, then replay after mutating the device-side
`cache_seqlens` in place. Three outcomes, each attributable to one test:

1. capture raises -> the AICPU launch is not capturable. This is what the first run
   of this test found: `GetSchedulerMetadataImpl` ran the kernel on a pool stream
   joined to the current one by two `thread_local` events, and under capture the
   first `aclrtRecordEvent` failed with runtime 207000, `capture_end` naming
   `ascendc_fa_metadata`. That fork/join produced a full serialization anyway, so it
   was replaced by a plain launch on the current stream; this test is now the
   regression guard for that.
2. capture succeeds but replay is stale -> `test_replay_tracks_device_seqlens` fails
   on the B comparison. This is the dangerous outcome: no error, just capture-time
   tiling applied to new lengths.
3. replay tracks -> the path is a viable drop-in for the sink operator.

`test_eager_metadata_matches_reference` runs first and pins the eager result against
a CPU golden, so a graph failure cannot be blamed on the metadata path being wrong
to begin with.

The KV bound deserves a note. `get_scheduler_metadata` derives the block-table row
stride as `ceil(max_seqlen_k / page_size)` (`flash_api.cpp`, `args.maxNumBlocksPerBatch`)
while the kernel uses that value to offset into the block table. So `max_seqlen_k`
here is the page capacity -- `MAX_BLOCKS_PER_SEQ * BLOCK_SIZE` -- not the actual
maximum KV length. Passing the actual max would silently mis-address paged KV.
"""

import pytest
import torch
import torch_npu

_device_name = torch_npu.npu.get_device_name() if torch_npu.npu.device_count() > 0 else ""
if "Ascend910" not in _device_name:
    pytest.skip("flash_attn_varlen_func / get_scheduler_metadata only on Ascend910", allow_module_level=True)

from flash_attn_npu_4 import flash_attn_varlen_func, get_scheduler_metadata
from tests.common.attention_ref import ref_flash_attention_pair
from tests.common.compare import assert_fa_close
from tests.common.test_utils import gather_paged_kv, make_random_tensor


DATA_TYPE = torch.bfloat16
BATCH_SIZE = 4
NUM_HEADS = 8
NUM_KV_HEADS = 2
# Uniform query per request, as a parallel-drafting draft produces.
Q_TOKENS_PER_REQ = 4
BLOCK_SIZE = 128
MAX_BLOCKS_PER_SEQ = 16
MAX_SEQLEN_K = MAX_BLOCKS_PER_SEQ * BLOCK_SIZE  # page capacity, see module docstring

# Two length sets that differ per request, both within the page capacity. Used by
# the replay test, where what matters is only that A and B give different output.
SEQLENS_A = [1024, 512, 900, 128]
SEQLENS_B = [2048, 1500, 64, 777]

# The eager test runs both flash-decode modes. Whether the tiling turns flash
# decode on is decided by, among other things, `maxKvSeqlen >= 1024`, and at this
# batch/head shape that is the only term that varies -- so these two sets differ
# in nothing else. Splitting them apart is what tells a wrong answer in the split
# KV path from a wrong answer everywhere.
EAGER_SEQLENS = {
    "fd_on": [1024, 512, 900, 128],
    "fd_off": [900, 512, 700, 128],
}

HEAD_SIZES = [128, 256]


def _flash_decode_predicted(kv_lens):
    """Mirror the fdFlag predicate the tiling computes, so the log names the mode.

    Reproduced from the host and AICPU tiling rather than read back out of the
    metadata blob: decoding that blob means hand-computing the C++ alignment of
    FAInferTilingData, which cannot be checked here. This is only a label -- a
    wrong prediction mislabels a case, it does not change what runs.
    """
    try:
        cube = torch.npu.get_stream_limit(torch.npu.current_stream())["cube_core_num"]
    except Exception:  # the label is a convenience, not the test
        return None
    group = NUM_HEADS // NUM_KV_HEADS
    num_tasks = BATCH_SIZE * NUM_KV_HEADS
    max_kv = max(kv_lens)
    long_seq = num_tasks <= 0.8 * cube and max_kv >= cube * 512
    short_seq = num_tasks <= 0.4 * cube and max_kv >= 1024
    return (
        Q_TOKENS_PER_REQ * group <= 128
        and Q_TOKENS_PER_REQ <= 16
        and max_kv >= 1024
        and min(kv_lens) > 0
        and (long_seq or short_seq)
    )


def _make_inputs(head_size, kv_lens):
    total_q = BATCH_SIZE * Q_TOKENS_PER_REQ
    num_blocks = BATCH_SIZE * MAX_BLOCKS_PER_SEQ
    query = make_random_tensor((total_q, NUM_HEADS, head_size), DATA_TYPE, device="npu")
    key_cache = make_random_tensor(
        (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_size), DATA_TYPE, device="npu"
    )
    value_cache = make_random_tensor(
        (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_size), DATA_TYPE, device="npu"
    )
    page_table = (
        torch.arange(num_blocks, dtype=torch.int32)
        .reshape(BATCH_SIZE, MAX_BLOCKS_PER_SEQ)
        .npu()
    )
    cu_seqlens_q = (
        torch.arange(BATCH_SIZE + 1, dtype=torch.int32) * Q_TOKENS_PER_REQ
    ).npu()
    # One stable device buffer, mutated in place -- the address the graph captures.
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32).npu()
    return query, key_cache, value_cache, page_table, cu_seqlens_q, cache_seqlens


def _run(query, key_cache, value_cache, page_table, cu_seqlens_q, cache_seqlens, head_size):
    """Metadata + forward, exactly as a backend would call them per layer."""
    scale = 1.0 / (head_size**0.5)
    scheduler_metadata = get_scheduler_metadata(
        batch_size=BATCH_SIZE,
        max_seqlen_q=Q_TOKENS_PER_REQ,
        max_seqlen_k=MAX_SEQLEN_K,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=head_size,
        cache_seqlens=cache_seqlens,
        qkv_dtype=DATA_TYPE,
        cu_seqlens_q=cu_seqlens_q,
        page_size=BLOCK_SIZE,
        causal=False,
        window_size=(-1, -1),
        softmax_scale=scale,
    )
    return flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        page_table=page_table,
        max_seqlen_q=Q_TOKENS_PER_REQ,
        softmax_scale=scale,
        causal=False,
        window_size=(-1, -1),
        scheduler_metadata=scheduler_metadata,
        num_splits=0,
        return_lse=False,
    )


def _reference(query, key_cache, value_cache, page_table, seqlens, head_size):
    """Per-request non-causal golden over the gathered paged KV."""
    scale = 1.0 / (head_size**0.5)
    key_cache_cpu = key_cache.detach().cpu()
    value_cache_cpu = value_cache.detach().cpu()
    page_table_cpu = page_table.cpu()
    query_cpu = query.detach().cpu()

    outs_ref, outs_pt = [], []
    for batch_idx, kv_seqlen in enumerate(seqlens):
        key_cpu, value_cpu = gather_paged_kv(
            key_cache_cpu, value_cache_cpu, page_table_cpu[batch_idx], kv_seqlen, BLOCK_SIZE
        )
        start = batch_idx * Q_TOKENS_PER_REQ
        out_ref, _, out_pt, _ = ref_flash_attention_pair(
            query_cpu[start : start + Q_TOKENS_PER_REQ].unsqueeze(0),
            key_cpu.unsqueeze(0),
            value_cpu.unsqueeze(0),
            scale,
            None,  # non-causal, full KV
            DATA_TYPE,
            0.0,
        )
        outs_ref.append(out_ref.squeeze(0))
        outs_pt.append(out_pt.squeeze(0))
    return torch.cat(outs_ref, dim=0), torch.cat(outs_pt, dim=0)


def _assert_matches(actual, expected, *, name, diagnosis):
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    max_diff = (actual - expected).abs().max().item()
    assert max_diff <= 1e-3, f"{name}: max|graph - eager| = {max_diff}. {diagnosis}"


@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("fd_mode", sorted(EAGER_SEQLENS))
def test_eager_metadata_matches_reference(fd_mode, head_size):
    """Eager AICPU-metadata path is correct before any graph question is asked."""
    kv_lens = EAGER_SEQLENS[fd_mode]
    print(f"\n  kv_lens={kv_lens} flash_decode_predicted={_flash_decode_predicted(kv_lens)}")
    query, key_cache, value_cache, page_table, cu_seqlens_q, cache_seqlens = _make_inputs(head_size, kv_lens)

    with torch.no_grad():
        output_npu = _run(
            query, key_cache, value_cache, page_table, cu_seqlens_q, cache_seqlens, head_size
        )
    torch.npu.synchronize()

    golden_ref, golden_pt = _reference(query, key_cache, value_cache, page_table, kv_lens, head_size)
    assert_fa_close(output_npu, golden_ref, golden_pt, name=f"eager out ({fd_mode})")


# Not parametrized over HEAD_SIZES, unlike the eager test. Capture is head-size
# independent, and a failed capture leaves the stream stuck in capture mode for the
# rest of the process -- a second case then dies in the autouse seeding fixture with
# "set_current_seed can be called during stream capture only if...", which reads like
# a second, unrelated problem. One case, one answer.
CAPTURE_HEAD_SIZE = 256


def test_replay_tracks_device_seqlens():
    """Capture metadata + forward, then replay against different device lengths."""
    head_size = CAPTURE_HEAD_SIZE
    query, key_cache, value_cache, page_table, cu_seqlens_q, cache_seqlens = _make_inputs(head_size, SEQLENS_A)
    seqlens_a = torch.tensor(SEQLENS_A, dtype=torch.int32).npu()
    seqlens_b = torch.tensor(SEQLENS_B, dtype=torch.int32).npu()

    # Eager baselines for both length sets, taken before capture so the graph
    # pool cannot recycle them.
    with torch.no_grad():
        cache_seqlens.copy_(seqlens_a)
        eager_a = _run(
            query, key_cache, value_cache, page_table, cu_seqlens_q, cache_seqlens, head_size
        ).clone()
        cache_seqlens.copy_(seqlens_b)
        eager_b = _run(
            query, key_cache, value_cache, page_table, cu_seqlens_q, cache_seqlens, head_size
        ).clone()
    torch.npu.synchronize()

    # Without this the replay comparison proves nothing: if A and B produced the
    # same output, a graph frozen at capture-time tiling would pass anyway.
    assert not torch.allclose(eager_a.float(), eager_b.float(), atol=1e-3), (
        "SEQLENS_A and SEQLENS_B produce indistinguishable output; pick lengths "
        "that actually change the result before trusting the replay assertions"
    )

    cache_seqlens.copy_(seqlens_a)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.no_grad():
        with torch.npu.graph(graph):
            graph_out = _run(
                query, key_cache, value_cache, page_table, cu_seqlens_q, cache_seqlens, head_size
            )

    # B first: capture ran with A, so this is the assertion that distinguishes a
    # graph that re-reads the device lengths from one replaying stale tiling.
    cache_seqlens.copy_(seqlens_b)
    graph.replay()
    torch.npu.synchronize()
    _assert_matches(
        graph_out,
        eager_b,
        name="replay(B) out",
        diagnosis=(
            "The graph replayed capture-time tiling instead of re-reading "
            "cache_seqlens: the AICPU metadata kernel is not being re-executed on "
            "replay, or it is but the forward reads a tiling buffer the replay did "
            "not refresh. This backend cannot replace the FIA sink operator until "
            "this passes."
        ),
    )

    # And back to A, so a graph that somehow recomputes but ignores the buffer
    # does not slip through the B check alone.
    cache_seqlens.copy_(seqlens_a)
    graph.replay()
    torch.npu.synchronize()
    _assert_matches(
        graph_out,
        eager_a,
        name="replay(A) out",
        diagnosis=(
            "Replay(B) passed but replay(A) did not, so the graph is producing "
            "something other than a function of the current cache_seqlens."
        ),
    )
