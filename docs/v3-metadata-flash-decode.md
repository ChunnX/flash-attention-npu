# v3: wrong results when the scheduler-metadata path meets flash decode

Internal note, not filed upstream. Written 2026-09-09 against
`fa4-aicpu-graph-d256` (base `6124c6c`), measured on Ascend910B4.

## Summary

The 910 v3 backend returns zeros or garbage when `flash_attn_with_kvcache` is
given a `scheduler_metadata` blob *and* the tiling turns flash decode on. No
error is raised. The forward simply reads a workspace region that only the host
tiling branch ever initialises.

v4 is correct in both modes on the same shapes. v4 carries the same missing
initialisation, so the bug is latent there rather than absent.

## Symptom

Two length sets differing only in `max(kv_lens)`, which is the one term of the
flash-decode predicate that varies at this shape:

| shape | mode | v3 | v4 |
| --- | --- | --- | --- |
| head 128, kv ≤ 900 | fd off | pass | pass |
| head 256, kv ≤ 900 | fd off | pass | pass |
| head 128, kv ≤ 1024 | **fd on** | **fail** | pass |
| head 256, kv ≤ 1024 | **fd on** | **fail** | pass |

The failure looks different by head size, which is what points at uninitialised
memory rather than a wrong index:

- head 128: garbage of the wrong magnitude — `[226.0, 20.625, -31.875, 520.0,
  -41.25, -176.0, -65.0]` where the reference is inside ±5.
- head 256: all zeros.

Same bug, differing in what the allocator happened to leave behind.

## Confirmed at the shape that matters

The table above is the synthetic shape the tests use. The standalone comparison
harness reproduces it independently at Qwen3.6-27B's own topology -- 32 query
heads over 8 KV heads, head_dim 256, paged KV, non-causal -- with every backend
on the same inputs against the same fp32 golden:

| batch | kv lengths | flash decode | fa_v3 max err | fa_v4 max err | noise floor |
| --- | --- | --- | --- | --- | --- |
| 1 | 2048 | **on** | **7.043e-02** | 1.669e-04 | 1.952e-04 |
| 2 | 256, 2048 | off | 5.977e-04 | 5.977e-04 | 6.045e-04 |
| 4 | 256 … 2048 | off | 5.683e-04 | 5.683e-04 | 4.723e-04 |
| 8 | 256 … 2048 | off | 6.142e-04 | 6.142e-04 | 4.782e-04 |

v3's error in the flash-decode row is 360x the noise floor. Every other row has
v3, v4, the sink operator and `npu_fused_infer_attention_score` agreeing to the
last digit.

Note which row it is. Flash decode needs `numTasks = batch * kvHeads` under
`0.4 * blockDim`, and at 8 KV heads on a 910B4 that admits batch 1 and nothing
else. The single failing configuration is single-request long-context -- the
latency case speculative decoding exists to serve.

## Reproduction

```
pytest -q -s tests/test_flash_attn_npu_v3_graph_metadata.py
```

Shape: batch 4, 4 query tokens per request, 8/2 heads (GQA), paged KV with
block size 128 and 16 blocks per request, non-causal, TND varlen, bf16.
`EAGER_SEQLENS` in that file holds the two length sets.

## Root cause

Flash decode splits the KV range and writes per-split partial LSE and O into the
workspace, then combines them. A split that gets no work never writes its
partial, so the combine has to see an initialised value: `-inf` for LSE, `0` for
O.

The host tiling branch does that initialisation:

```
csrc/ascend910/flash_attn_npu_3/flash_api.cpp:562  workspace_tensor = at::empty({workSpaceSize}, ...)
csrc/ascend910/flash_attn_npu_3/flash_api.cpp:565  if (flashDecodeFlag && (splitLseTotalSize > 0 || splitOTotalSize > 0)) { ... }
```

The scheduler-metadata branch does not:

```
csrc/ascend910/flash_attn_npu_3/flash_api.cpp:407  workspace_tensor = at::empty({wsBase + wsSplit}, ...)
                                                   // no init
```

It cannot simply be copied across. The split region's layout is not host-knowable
on that path: the kernel takes its per-core offsets from the tiling struct,

```
csrc/ascend910/flash_attn_npu_3/mha_fwd_kvcache.cpp:282  fATilingData->coreInfo[coreIdx].firstSplitKVTaskLseOffset
csrc/ascend910/flash_attn_npu_3/mha_fwd_kvcache.cpp:283  fATilingData->coreInfo[coreIdx].firstSplitKVTaskOOffset
```

and those are computed by `splitBN2S1GS2` on AICPU, inside `get_scheduler_metadata`.
The host allocating the workspace in the later `mha_fwd` call would have to read
them back to know where the LSE region ends and the O region begins — which is
the device-to-host sync the scheduler-metadata path exists to remove. The host
only has an upper bound (`wsSplit`), and a single fill value cannot serve a
region that needs `-inf` in one part and `0` in another.

The two branches also disagree on how many blocks run, which is the other place
flash decode is special:

```
host:     if (flashDecodeFlag) { splitBN2S1GS2(...); if (needCoreNum) launchBlockDim = needCoreNum; }
metadata: launchBlockDim = blockDim;   // always full
```

Not yet established whether that second divergence contributes; the workspace
one is sufficient to explain the symptom.

## Why the suite never caught it

No case in `tests/test_flash_attn_npu_v3_metadata.py` can turn flash decode on.
The predicate is

```
fdFlag = pagedKV && isVarlen && maskType != MASK_BAND
      && maxQ * groupSize <= 128 && maxQ <= 16
      && maxKvSeqlen >= 1024 && minQ > 0
      && (  (numTasks <= 0.8 * blockDim && maxKvSeqlen >= blockDim * 512)
         || (numTasks <= 0.4 * blockDim && maxKvSeqlen >= 1024) )

numTasks = batch * numHeadsK
```

- `KV_CACHE_TND_CASES` use `q_seqlen` 1024 and 512, so `maxQ <= 16` is false.
- `KV_CACHE_BSND_CASES` are not varlen, so `isVarlen` is false.
- `FLASH_ATTN_VARLEN_CASES` are not paged, so `pagedKV` is false.

The gap is the decode-shaped call — short query, long KV, paged, varlen — which
is exactly what a speculative-decoding draft issues, and nothing else does.

## Blast radius

`maxQ <= 16` and `numTasks <= 0.4 * blockDim` together mean flash decode is a
small-batch phenomenon. On 910B4, `0.4 * blockDim` is around 8, so with two KV
heads it needs batch ≤ 4, plus context ≥ 1024.

That is off in throughput serving and on in low-concurrency latency runs. It is
not a corner nobody reaches: single-request long-context is how speculative
decoding is usually benchmarked.

v4 has the same missing initialisation at `flash_api.cpp:339` and passes anyway
on these shapes — presumably no empty split arises, or one of its flash-decode
fixes removed the dependency. Either way the same hole is open there.

## Options

**A. Suppress flash decode on the scheduler-metadata path.** Add a field to
`FAMetadataArgs` that the AICPU kernel honours when setting `flashDecodeFlag`.
Small and safe; costs whatever flash decode was buying at those shapes, which we
have not measured.

**B. Make the kernel not depend on a pre-initialised workspace** — have each
split write its partial unconditionally, or have the combine skip splits that
did no work. The real fix, and it would close the latent v4 hole too. Kernel
work, and it needs someone who owns that code.

## Also established

v3's scheduler-metadata path does capture into an NPUGraph, and replay does track
the device-side sequence lengths -- `test_replay_tracks_device_seqlens[fd_off]`
passes. The capture fix on this branch works for v3 as well as v4. Flash decode
is the only thing wrong with v3 here, and it is worth separating the two: an
earlier run had all three v3 cases failing, which looked like a capture problem
until the flash-decode axis was split out and the replay case was given a shape
it could reach.

## Our position

Parked. We are using v4, which is correct in both modes, so this is not on our
critical path. Worth raising upstream when someone can act on it — B is the fix
we would argue for, since A leaves v4's hole open.

Reported by the tests added on this branch, which cover both modes for both
generations.
