# AMD ROCm / HIP (Linux)

Native HIP backend for this fork. **ZLUDA is not supported**
([FlashML-org/FreeToken#60](https://github.com/FlashML-org/FreeToken/issues/60)).

First-class target: **Linux x86_64, 2× AMD Radeon AI PRO R9700, RDNA 4, `gfx1201`**.
One-card DSV4-Flash HIP+Triton already landed. This slice adds **dual-card TP /
RCCL wiring** (`--gpu 0,1`). Dual-card e2e on the R9700 box is leftover.

This cloud / CI environment does **not** have an R9700. The checks below that
need a GPU must be run on the workstation. Nothing in this document invents
tok/s or unpublished numbers. **R9700 tok/s is not published.**

## What this PR does vs leftover

| Works in-tree (additive; NVIDIA CUDA path unchanged) | Leftover / on-box only |
|---|---|
| Detect AMD vs NVIDIA (`ft gpu`); hide Granite Ridge iGPU so TP ranks are the two R9700s | Dual-card **e2e** on 2× R9700 (this VM has no AMD GPU) |
| HIP JIT of tvm-ffi kernels (`index`, `store`, `fast_index_copy`, …) for `gfx1201` | Vulkan (community: the hard problem) |
| HIP `host_register` / `device_ptr` (#122 TDR fix) | Full one-card e2e on R9700 (must run locally) |
| HIP attention: `triton` aliases `dsv4_sparse` / `dsa` / `m3_sparse` / `qsa_sparse` (already Triton; no flashinfer/sgl cubins) | Native HIP FP8 tensor cores (e4m3 **emulated** unless `FREETOKEN_HIP_E4M3_NATIVE=1`) |
| GGUF HIP compile (`-DUSE_ROCM`, `--offload-arch`) + rocThrust shim | gfx1201 HIP **graph replay** (TDR on Windows #82) |
| MoE `offload` / `fused` / `cpu` with Triton experts; DSV4 `ds_fp4` host banks | `--moe-backend hybrid`; Windows ROCm |
| `--gpu 0,1` infers TP=2; HIP uses **RCCL** (`torch.distributed` backend `rccl` or the ROCm `nccl` alias). PyNCCL / NCCL 2.27 is NVIDIA-only and fails loudly on HIP. Dual-card DSV4 generate **proven on 2× R9700** (2026-09-02, checklist below) | Rank-0-only fill; RCCL P2P stability under long serves is only a ~2 h sample |
| Shared expert banks on TP>1: one **shmem (`memfd`) pool** created by rank 0, mapped `MAP_SHARED` by every rank (paths handed over on the gloo CPU group). Needs `amdgpu no_system_mem_limit=1` (below) | Rank-0-only bank fill (both ranks still read the checkpoint); FTW `HostBank()` still anonymous-mmaps **2×137 GiB** on TP=2 |
| Dense models that already call `LinearOProj` / `LinearRowParallel` all-reduce via `TorchDistributedImpl` on HIP | DSV4 `Linear` / MLA head **sharding** (today: replicate resident weights + full Triton `n_heads`) |

Default on `gfx1201`: `--cuda-graph-max-bs 0` unless you set
`FREETOKEN_HIP_GRAPH_REPLAY=1`. That is conservative — Linux replay on this SKU
is unverified. ROCm has also hung on similar RDNA4 bring-up
([ROCm#6630](https://github.com/ROCm/ROCm/issues/6630), 2026-08-28); if HIP
init hangs, collect `rocminfo` / `dmesg` and stay on eager + `triton`.

## Install (Linux ROCm, gfx1201)

Do **not** install the default `freetoken[accel]` extra: that pins
`torch` from the PyTorch **cu130** index.

1. Install [ROCm](https://rocm.docs.amd.com/) (or the
   [TheRock nightly](https://rocm.nightlies.amd.com/whl-multi-arch/) stack
   the community ports used). You need `hipcc` and `libamdhip64.so`.
2. Install a **ROCm** PyTorch wheel whose HIP version matches that toolkit
   (`torch.version.hip` must be set; `torch.version.cuda` should be unset).
3. Triton with an AMD backend (Linux `triton` 3.6+ or the matching ROCm build).
4. Optional: [rocWMMA](https://github.com/ROCm/rocWMMA) headers on
   `ROCM_HOME/include` — Maxritz noted they are needed for some gfx1201 GEMMs.
5. Install this tree:

```bash
git clone https://github.com/LoneExile/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
# Use your ROCm torch; do not let uv pull cu130.
FREETOKEN_GFX_ARCH=gfx1201 uv pip install -e .
```

Environment (also set automatically by `ft gpu` / `ft serve` when the vendor
is AMD and the vars are unset):

| Variable | Example | Purpose |
|---|---|---|
| `FREETOKEN_GFX_ARCH` | `gfx1201` | ISA for JIT / docs |
| `TVM_FFI_ROCM_ARCH_LIST` | `gfx1201` | tvm-ffi `--offload-arch` (default gfx906 is a silent footgun) |
| `TRITON_OVERRIDE_ARCH` | `gfx1201` | Triton codegen |
| `PYTORCH_ROCM_ARCH` | `gfx1201` | torch `cpp_extension` GGUF kernels |
| `ROCM_HOME` / `HIP_PATH` | `/opt/rocm` | hipcc + headers |
| `FREETOKEN_INCLUDE_IGPU` | `1` | keep Granite Ridge iGPU visible |
| `FREETOKEN_HIP_GRAPH_REPLAY` | `1` | try CUDA-graph replay on gfx1201 |
| `FREETOKEN_SKIP_CUDA_EXT` | `1` | skip install-time C++ ext; HIP ctypes fallback still registers host memory |
| `FREETOKEN_GPU_VENDOR` | `amd` | force vendor in tests / odd wheels |
| `FREETOKEN_HIP_E4M3_NATIVE` | `1` | try Triton native fp8e4nv on HIP (default: emulate) |
| `NCCL_IB_DISABLE` | `1` (auto on HIP TP>1 if unset) | RCCL still reads `NCCL_*` names. Dual-card desktops have no InfiniBand; leaving IB on can stall init. Override if you actually have IB. |
| `FREETOKEN_BANK_SHARE` | `0` | Disable the TP>1 shared pool (each rank anonymous-mmaps the full pool — likely OOM on 192 GB) |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | **Not set on HIP by default** (NVIDIA runs get it automatically). See the handle cap below before turning it on. |

```bash
ft gpu
# vendor: amd
# gcn_arch: gfx1201
#   gpu0: AMD Radeon AI PRO R9700 ...
```

A Ryzen 9 9950X iGPU (`AMD Radeon Graphics` / `gfx1036`) is hidden so it does
not become device 0. Strix Halo is **not** hidden.

A **CUDA PyTorch wheel + NVIDIA dGPU + that AMD iGPU** stays the NVIDIA CUDA
path (`Vendor.NVIDIA`). `CUDA_ON_AMD` is only when the CUDA wheel would run on
a **discrete** AMD GPU after iGPU hide.

## DeepSeek-V4-Flash-0731 (one R9700, experts in host RAM)

The in-tree DSV4 path is already Triton (`dsv4_sparse` + `kernel/triton/dsv4/*`).
On HIP, `--attention-backend triton` remaps to `dsv4_sparse`. Do **not** use
flashinfer / sgl / trtllm.

Expert pool (from `_BANK_BYTES_PER_EXPERT["ds_fp4"]`, not a benchmark):

- Shape: `H=4096`, `I=2048`, 43 layers, 256 experts, 6 active.
- Bytes per expert-layer: `2*I*(H/2 + H/32) + H*(I/2 + I/32)` = 13 369 344 (~12.75 MiB).
- Full host banks: `43 × 256 × 13 369 344` = **147 169 738 752 B ≈ 137.1 GiB**.
- Fits ~192 GB system RAM with headroom for OS + KV + activations. One 32 GB
  R9700 holds the resident (non-expert) path + a GPU expert slot cache.

```bash
export FREETOKEN_GFX_ARCH=gfx1201
# Hide the 9950X iGPU (default). One card:
ft serve --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --moe-backend offload \
  --attention-backend triton \
  --cuda-graph-max-bs 0
```

Optional one-card trade: `--kv-reserve-tokens` raises the KV floor versus the
GPU expert-slot cache (`--moe-cache-auto`). That reserve is **per card**, not a
system-wide window.

`--gpu 0` if `ft gpu` still lists the iGPU. If HIP init hangs, see
[ROCm#6630](https://github.com/ROCm/ROCm/issues/6630) (2026-08-28 RDNA4 bring-up);
collect `rocminfo` / `dmesg` and stay on eager + `triton`.

## Dual-card TP / RCCL (2× R9700)

`--gpu 0,1` with default `--tp-size 1` now means **TP=2**. The parent process
hides Granite Ridge (`gfx1036`) before workers spawn, so those indices are the
two discrete cards (`ft gpu` / `tp_discrete`), not the 9950X iGPU.

HIP never loads PyNCCL (`pynccl.cu` / `-lnccl` / NCCL 2.27
`ncclMemAlloc` / `ncclCommWindowRegister`). That plugin is NVIDIA-only and
raises if a HIP process reaches it. AMD TP uses `torch.distributed` with
backend **`rccl`** when the ROCm wheel exposes it, otherwise the ROCm alias
still named **`nccl`** (that build links RCCL). The CPU group stays **gloo**
(memory sync / scheduler). NVIDIA `--disable-pynccl` (real NCCL world + gloo)
and the default PyNCCL path are unchanged.

RCCL install: it ships with the **ROCm PyTorch wheel**. There is no separate
`pip install rccl`. Optional debug (not set by FreeToken): `NCCL_DEBUG=INFO`.
If PCIe P2P misbehaves on this SKU, you may try `NCCL_P2P_DISABLE=1` (shared
memory fallback). This tree only auto-sets `NCCL_IB_DISABLE=1` when unset.

Expert banks: TP>1 maps **one shmem pool** (`memfd_create`; rank 0 creates every
bank and broadcasts its `/proc/<pid>/fd/<n>` paths over the gloo CPU group, the
other ranks open them `MAP_SHARED`). The pool is RAM: `MemAvailable` must cover
it (~137.1 GiB for DSV4-Flash-0731) and it stays resident once pinned.

Five things that do **not** work, all measured on 2×R9700 / ROCm 7.14:

* A **disk-backed** shared pool (`MAP_SHARED` file on ext4 + `hipHostRegister`).
  On ROCm the registration is a KFD SVM range; the writeback flusher then
  write-protects the dirty pages, every `page_mkclean` is an MMU-notifier
  invalidation, KFD evicts the range and **quiesces the process's GPU queues**,
  the restore re-faults the pages dirty, and the cycle never converges
  (`dmesg`: `svm_range_restore_work [amdgpu] hogged CPU`, counts doubling). The
  DSV4 load wedged at shard 9/43 with both ranks in `futex_do_wait`. shmem never
  goes through writeback, so the storm cannot start. `/dev/shm` is not used
  either: its mount is capped at 50% of RAM.
* The default **KFD resident-system-memory budget** (`≈ MemTotal − MemTotal/64`,
  ~169 GiB on 172 GiB). Every rank's `hipHostRegister` of the pool is charged
  against that one global counter, so 2 × 137 GiB fails the second rank's pins
  partway through the load (`dmesg`: `SVM mapping failed, exceeds resident
  system memory limit`). The pages are physically shared; the accounting is
  double counting. Disable the limit before a dual-card DSV4 serve — the engine
  preflight refuses to start otherwise:

  ```bash
  echo Y | sudo tee /sys/module/amdgpu/parameters/no_system_mem_limit     # now
  echo 'options amdgpu no_system_mem_limit=1' | sudo tee /etc/modprobe.d/freetoken-kfd.conf  # after reboot
  ```
* PyTorch **`expandable_segments`** on ROCm. The allocator maps VRAM in 20 MiB
  (large pool) / 2 MiB (small pool) physical handles, and ROCr caps a process at
  **~1016 handles** (fixed `mem_handle_aperture`, no `HSA_*` knob): a 32 GiB R9700
  tops out near **20 GiB**, or **~2 GiB** when the tensors are small. DSV4's resident
  weights are hundreds of small tensors, so after loading them *every* further
  allocation failed with 22 GiB free (the MoE slot cache, at any size). The engine
  therefore leaves `expandable_segments` off on HIP; `PYTORCH_ALLOC_CONF` still
  wins if you set it yourself. Measured: 20 MiB × 1016 = 19.8 GiB, 256 MiB × 79 =
  19.8 GiB, 1 MiB × 2032 = 1.98 GiB; without the setting, 30 GiB of any size.
* Triton 3.8 on gfx1201 **drops a `tl.atomic_add` whose mask is derived from a
  `tl.histogram` result** (`mask=(le < E) & (h > 0)` compiles to no store; the
  same kernel with `mask=le < E` is correct; `hist+store`, `hist+atomic` without
  the derived predicate, `tl.range` fills and `tl.arange` scatters all pass).
  `moe_align_block_size` had exactly that shape in `_fill_and_count`, and the
  fused single-CTA kernel took a `tl.where(m_e & (nblk > 0), ...)` on a histogram
  count, so every prompt of ≥ 128 tokens (T × top_k ≥ the 768-route grouped
  prefill crossover, where the aligned layout is consumed) got `expert_ids`
  uninitialised: garbage tokens or `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on
  both ranks. Neither the NVIDIA build nor the CPU tests could see it. The kernels
  now write bins unconditionally (adding 0 is harmless) and reload block counts
  from the cumsum scratch; `tests/moe/test_moe_align_hip.py` runs the kernel
  against a torch reference on the real GPU (16/16 red on the old kernel).
* A Triton **`@constexpr_function` that calls back into Python** (the e4m3
  emulation gate read `os.environ` / the driver at compile time). ROCm Triton's
  compile-time evaluator rejects it (`Unsupported function referenced:
  _hip_forces_e4m3_emu`) on the *first kernel launch* — i.e. 12 minutes after a
  load that succeeded. The gate is now resolved once at import
  (`e4m3_compat.HIP_EMU`); a drift guard raises if the vendor/env flips after
  import instead of silently compiling the wrong branch.

Both ranks still *read* the checkpoint (rank-0-only fill is leftover). FTW
`HostBank()` still anonymous-mmaps **2×137 GiB** on TP=2.

DSV4 / MLA in this slice: **replicated** resident `Linear` + full Triton
`n_heads`. KV is a **per-rank** `DSV4PagedKVCache` — dual-card does **not**
shard KV across 64 GB. 128k context is **not** enabled by `--gpu 0,1`. Each
32 GB card must still hold its own KV. Models that already use `LinearOProj` /
`LinearRowParallel` (Llama, Qwen, …) all-reduce through `TorchDistributedImpl`
on HIP.

```bash
export FREETOKEN_GFX_ARCH=gfx1201
# TP>1 shared expert pool is shmem (RAM); needs amdgpu no_system_mem_limit=1 (see above).
# Optional: NCCL_DEBUG=INFO
ft gpu   # two R9700s; Granite Ridge hidden; tp_discrete: 2
ft serve --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --moe-backend offload \
  --attention-backend triton \
  --cuda-graph-max-bs 0 \
  --gpu 0,1 \
  --moe-cache-auto
```

Do **not** add `--max-seq-len-override 131072 --kv-reserve-tokens 131072` to
this dual-card command expecting a TP-split 128k window. Those flags are a
**per-card** KV floor; 128k still has to fit on one 32 GB R9700.

If RCCL or a second discrete GPU is missing, launch exits with a precise list
(no CUDA-symbol crash). On this cloud VM that list is expected.

### Dual-card checklist — run on the box 2026-09-02 (2× R9700, ROCm 7.14, torch 2.11.0+rocm7.14.1)

1. `ft gpu`: two `gfx1201` R9700s; iGPU hidden; `tp_discrete: 2`. **Done.**
2. Command above reaches READY (RCCL init, no PyNCCL/NCCL 2.27, no flashinfer). **Done**: 43/43 shards in ~12 min, READY, KV pool 64 128 tokens per card, 31.8 GiB VRAM used per card after init.
3. `chat/completions` returns text. **Done**, and beyond `max_tokens=16`: correct answers on real prompts through 2 015 prompt tokens (needle-in-haystack at 564 tokens, 1 515-token prompts, `ignore_eos` 128-token decodes), multi-turn with radix-cache hits (128–1 024 cached tokens). Both P2P and the shmem banks held across ~2 h of requests.
4. Host RAM: one ~137 GiB pool, not two (shmem 137 GiB, `MemAvailable` 161 → 30 GiB during load). Free RAM ≳ 160 GB before load — the serve script gates on it.
5. tok/s: the numbers live in the harbor-sandboc `docs/model-speed.md` grid, measured client-side over streaming (this server returns no `timings`). They are bandwidth-bound by the RAM-resident expert design (TTFT ≈ 11 s at 165 prompt tokens, ≈ 2.8 tok/s decode at conc 1) and are **not** a GPU-compute figure for the R9700.
6. RCCL P2P did not hang; `NCCL_P2P_DISABLE=1` was not needed.

### One-card checklist (R9700 — not faked in CI)

1. `ft gpu`: one R9700 `gfx1201`; Granite Ridge iGPU hidden.
2. Host free RAM ≳ 160 GB before load (137 GiB banks + workspace).
3. Command above reaches READY (no CUDA-symbol crash; no flashinfer/sgl).
4. One `chat/completions` with `max_tokens=16` returns text.
5. Optional: `FREETOKEN_HIP_GRAPH_REPLAY=1` — leftover, report Linux stability.
6. Do **not** record tok/s as a published number.

## Minimal generate / serve (on the R9700)

Use a **small** dense HF or GGUF checkpoint first (not Qwen3.8-27B FP8):

```bash
export FREETOKEN_GFX_ARCH=gfx1201
ft serve --model <small-model> --attention-backend triton --cuda-graph-max-bs 0
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<id>","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

MoE offload (host expert banks) needs the HIP register/`device_ptr` path:

```bash
ft serve --model <moe-or-gguf> --moe-backend offload --attention-backend triton --cuda-graph-max-bs 0
```

### What you must run on the R9700 (not faked in CI)

1. `ft gpu` lists the R9700 as `gfx1201`; iGPU absent or marked hidden.
2. `ft serve` on a small dense model reaches READY and returns a completion.
3. Same with `--moe-backend offload` on a small/supported MoE or GGUF (no TDR).
4. DSV4-Flash-0731 one-card checklist above.
5. Optional: `FREETOKEN_HIP_GRAPH_REPLAY=1` — report whether graph replay is
   stable on **Linux** gfx1201 (Windows #82 said no for MoE).
6. Dual-card checklist above (`--gpu 0,1`, `no_system_mem_limit=1`, shared shmem banks). KV is per-card; dual TP does not unlock 128k.

## Adding another gfx

1. Append the ISA to `SUPPORTED_GFX_ARCHES` in
   `python/freetoken/runtime/gpu.py`.
2. Map the marketing name in `_arch_from_name` if you want `ft gpu` autodetection.
3. `export FREETOKEN_GFX_ARCH=gfxXXXX` (and the three `*_ARCH*` vars).
4. Rebuild: HIP JIT and GGUF `cpp_extension` both pass `--offload-arch=`.

RDNA2 (`gfx1030`/`gfx1031`): Maxritz noted wave64 + `dot4`; that is untested here.

## Community work (hints, not copied blindly)

- [Maxritz/FreeToken-ROCm](https://github.com/Maxritz/FreeToken-ROCm) — Windows
  gfx1201 bring-up; hip_compat, PTX gates, skip CUDA ext.
- [PialGhosh2233/FreeToken-rocm-gfx1200](https://github.com/PialGhosh2233/FreeToken-rocm-gfx1200)
  and [issue #122](https://github.com/FlashML-org/FreeToken/issues/122) —
  `hipHostRegister(Portable\|Mapped)` + `hipHostGetDevicePointer`.
- [issue #82](https://github.com/FlashML-org/FreeToken/issues/82) — gfx1201
  graph-replay TDR.

Those forks are older than this tree and Windows-first. This port re-verified
the same failure modes against **this** source and implements Linux HIP here.

## NVIDIA

Unchanged. `uv pip install -e ".[accel]"` still resolves cu130 torch. A CUDA
wheel on a box whose only AMD PCI device is the 9950X iGPU stays
`Vendor.NVIDIA`. AMD HIP is additive.
