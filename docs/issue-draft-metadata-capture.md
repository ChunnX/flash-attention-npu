# DRAFT — issue for MinghuasLab/flash-attention-npu (not filed)

**Title**

```
[BUG]: get_scheduler_metadata cannot be captured into an NPUGraph — AICPU kernel is launched on a pool stream
```

---

## Environment

```
- Device: Ascend 910B4
- CANN version: 9.0.1
- PyTorch version: 2.10.0
- torch_npu version: 2.10.0.post2
- Python version: 3.12.13
- flash_attn_npu version: built from 6124c6c (v3 and v4, FLASH_ATTN_BUILD_NPU=910)
```

## Bug Description

`get_scheduler_metadata` cannot be called inside a `torch.npu.graph` capture region.
Capture fails at `capture_end()` with runtime error `207000`, reported against
`ascendc_fa_metadata`.

The cause is in `GetSchedulerMetadataImpl`. The AICPU metadata kernel is not launched
on the current stream. It goes to a stream taken from the pool, with the two streams
joined by a pair of events:

```cpp
c10_npu::NPUStream aicpuStream = c10_npu::getNPUStreamFromPool();
...
ACL_CHECK(aclrtRecordEvent(inputReady, curHandle));        // (1) fails under capture
ACL_CHECK(aclrtStreamWaitEvent(aicpuHandle, inputReady));
ComputeFAMetadata<<<1, nullptr, aicpuHandle>>>(&metaArgs, sizeof(metaArgs));
ACL_CHECK(aclrtRecordEvent(metadataDone, aicpuHandle));
ACL_CHECK(aclrtStreamWaitEvent(curHandle, metadataDone));
```

A capturing stream rejects that cross-stream event synchronisation, so step (1) returns
`207000` and the whole capture is lost.

The same construction is present in four backends:

| file | line (at 6124c6c) |
| --- | --- |
| `csrc/ascend910/flash_attn_npu/flash_api.cpp` | 124 |
| `csrc/ascend910/flash_attn_npu_3/flash_api.cpp` | 103 |
| `csrc/ascend910/flash_attn_npu_4/flash_api.cpp` | 103 |
| `csrc/ascend950/flash_attn_npu_3/flash_api.cpp` | 33 |

We verified `flash_attn_npu_3` and `flash_attn_npu_4` on 910 with hardware. The other
two are the same code by inspection, not tested.

### Why this combination matters

The forward alone captures fine, and that is what `tests/test_flash_attn_npu_v3_graph.py`
and `tests/test_flash_attn_npu_v4_graph.py` exercise — they build the metadata outside
the graph and capture only the forward. So the combination in this report is not covered
by the suite.

It is, however, the combination a decode-side caller needs. The reason to use
`get_scheduler_metadata` at all is that the sequence lengths live on device and change
every step. If the metadata has to be built outside the graph, the tiling captured in
the graph is frozen at capture time and replay silently uses stale tiling. For a
speculative-decoding draft — where the KV length is the scheduled length minus the
tokens the verify kernel just rejected — reading those lengths on the host is exactly
the device-to-host sync this API exists to remove.

## Steps to Reproduce

Two capture attempts differing only in where the metadata is built. A passes, B fails.

```python
import torch
import torch_npu  # noqa: F401
from flash_attn_npu_4 import flash_attn_varlen_func, get_scheduler_metadata

B, Q, H, HKV, D = 1, 4, 8, 2, 128
BLOCK, NBLK = 128, 4
MAX_K = BLOCK * NBLK          # page capacity == page_size * page_table.shape[1]
scale = D**-0.5

q = torch.randn(B * Q, H, D, dtype=torch.bfloat16, device="npu")
k = torch.randn(B * NBLK, BLOCK, HKV, D, dtype=torch.bfloat16, device="npu")
v = torch.randn_like(k)
page_table = torch.arange(B * NBLK, dtype=torch.int32, device="npu").reshape(B, NBLK)
cu_seqlens_q = torch.tensor([0, Q], dtype=torch.int32, device="npu")
cache_seqlens = torch.tensor([256], dtype=torch.int32, device="npu")

def metadata():
    return get_scheduler_metadata(
        batch_size=B, max_seqlen_q=Q, max_seqlen_k=MAX_K,
        num_heads_q=H, num_heads_kv=HKV, headdim=D,
        cache_seqlens=cache_seqlens, qkv_dtype=torch.bfloat16,
        cu_seqlens_q=cu_seqlens_q, page_size=BLOCK,
        causal=False, window_size=(-1, -1), softmax_scale=scale)

def forward(meta):
    return flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_seqlens_q, seqused_k=cache_seqlens,
        page_table=page_table, max_seqlen_q=Q, max_seqlen_k=MAX_K,
        softmax_scale=scale, causal=False, window_size=(-1, -1),
        scheduler_metadata=meta, num_splits=0, return_lse=False)

forward(metadata()); torch.npu.synchronize()      # eager: fine

# A. metadata OUTSIDE the capture region -- what the existing graph tests do
graph_a = torch.npu.NPUGraph()
meta = metadata()
with torch.npu.graph(graph_a):
    forward(meta)
graph_a.replay(); torch.npu.synchronize()
print("A: OK")

# B. metadata INSIDE the capture region -- fails
graph_b = torch.npu.NPUGraph()
with torch.npu.graph(graph_b):
    forward(metadata())
print("B: OK")        # never reached
```

`flash_attn_npu_3` + `flash_attn_with_kvcache` reproduces the same failure. A script
covering both generations is attached.

B is attempted last on purpose: a failed capture leaves the stream in capture mode for
the rest of the process, so anything after it fails for a reason that is not its own.

## Error output

```
[E] aclrtRecordEvent(inputReady, curHandle) failed
Exception raised from operator() at csrc/ascend910/flash_attn_npu_4/flash_api.cpp:131

RuntimeError: The Inner error is reported as above. The process exits for this inner
error, and the current working operator name is ascendc_fa_metadata.
EH9999: Inner Error!
EH9999 record event failed, runtime result = 207000[FUNC:ReportCallError]

  File ".../torch_npu/npu/graphs.py", line 533, in __exit__
    self.npu_graph.capture_end()
  File ".../torch_npu/npu/graphs.py", line 370, in capture_end
    super().capture_end()
```

Once capture has failed the stream stays in capture mode, so later code fails in ways
that look unrelated — a pytest fixture calling `torch.manual_seed`, for instance, dies
with `NPUGeneratorImpl::set_current_seed can be called during stream capture only if new
seed is the same as the original seed`. Worth knowing when reading a failing run.

## Additional Context

### Suggested fix

Launch the kernel on the current stream, and drop the pool stream and both events:

```diff
-    c10_npu::NPUStream aicpuStream = c10_npu::getNPUStreamFromPool();
-    ... event creation ...
+    aclrtStream curHandle = c10_npu::getCurrentNPUStream().stream(false);
+
     FAMetadataArgs metaArgs = args;
-    auto metadata_task = [curHandle, aicpuHandle, inputReady, metadataDone, metaArgs]() mutable -> int {
-        ACL_CHECK(aclrtRecordEvent(inputReady, curHandle));
-        ACL_CHECK(aclrtStreamWaitEvent(aicpuHandle, inputReady));
-        ComputeFAMetadata<<<1, nullptr, aicpuHandle>>>(&metaArgs, sizeof(metaArgs));
-        ACL_CHECK(aclrtRecordEvent(metadataDone, aicpuHandle));
-        ACL_CHECK(aclrtStreamWaitEvent(curHandle, metadataDone));
+    auto metadata_task = [curHandle, metaArgs, meta, seqlensK, cuSeqlensQ]() mutable -> int {
+        ComputeFAMetadata<<<1, nullptr, curHandle>>>(&metaArgs, sizeof(metaArgs));
         return 0;
     };
     at_npu::native::OpCommand::RunOpApiV2("ascendc_fa_metadata", metadata_task);
-
-    c10_npu::NPUCachingAllocator::recordStream(meta.storage().data_ptr(), aicpuStream);
-    ...
```

Notes on the shape:

- `RunOpApiV2` is kept. The comment above the original code still holds — a direct
  host-side launch races the forward on the first call in a fresh process and leaves the
  tiling uninitialised. This keeps the metadata and the forward on one ordered queue.
- The ordering is unchanged. The two events fork and join around a single kernel, so
  they already produced the serial order a plain launch on the current stream gives.
- Tensors are captured by value, for the same reason `launch_fa_infer` keeps
  `seqlenk_gpu_tensor`: the deferred task can outlive the enclosing scope and holds raw
  pointers into them. `recordStream` is dropped because it only means something when the
  memory is in flight on a second stream.

We are running this locally on 910 v3 and v4. Capture succeeds, and replay picks up new
device-side `cache_seqlens` — verified by replaying one graph against two different
length sets and comparing each against an eager run.

Happy to send a PR if this looks right to you.
