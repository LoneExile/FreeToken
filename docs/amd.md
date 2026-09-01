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
| `--gpu 0,1` infers TP=2; HIP uses **RCCL** (`torch.distributed` backend `rccl` or the ROCm `nccl` alias). PyNCCL / NCCL 2.27 is NVIDIA-only and fails loudly on HIP | RCCL P2P / dual-card generate on this SKU (vLLM RCCL was reported fragile; not claimed here) |
| Shared file-backed expert banks (`FREETOKEN_BANK_SHARE_DIR`, default `$XDG_CACHE_HOME/freetoken/banks-{port}` / `~/.cache/...` — **not** `/tmp`) | Rank-0-only bank fill; FTW `HostBank()` still anonymous-mmaps **2×137 GiB** on TP=2 |
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
| `FREETOKEN_BANK_SHARE_DIR` | `$HOME/.cache/freetoken/banks-1919` | File-backed `MAP_SHARED` expert banks. Default is `$XDG_CACHE_HOME/freetoken/banks-{port}` (or `~/.cache/...`), **not** `/tmp` (tmpfs). Must be a disk with ≥137.1 GiB free for DSV4. |
| `FREETOKEN_BANK_SHARE` | `0` | Disable shared banks (each rank anonymous-mmaps the full pool — likely OOM on 192 GB) |

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

Expert banks: TP>1 maps one file-backed pool under
`$XDG_CACHE_HOME/freetoken/banks-{port}` (or `~/.cache/freetoken/banks-{port}`),
**not** `/tmp` (systemd `/tmp` is often tmpfs at ~50% RAM). Launch fails if that
filesystem is tmpfs or has less than the pool free. Export
`FREETOKEN_BANK_SHARE_DIR` to a real disk. Both ranks still *read* the
checkpoint (rank-0-only fill is leftover). FTW `HostBank()` still
anonymous-mmaps **2×137 GiB** on TP=2.

DSV4 / MLA in this slice: **replicated** resident `Linear` + full Triton
`n_heads`. KV is a **per-rank** `DSV4PagedKVCache` — dual-card does **not**
shard KV across 64 GB. 128k context is **not** enabled by `--gpu 0,1`. Each
32 GB card must still hold its own KV. Models that already use `LinearOProj` /
`LinearRowParallel` (Llama, Qwen, …) all-reduce through `TorchDistributedImpl`
on HIP.

```bash
export FREETOKEN_GFX_ARCH=gfx1201
# Disk, not tmpfs /tmp. {port} matches --port (default 1919).
export FREETOKEN_BANK_SHARE_DIR="$HOME/.cache/freetoken/banks-1919"
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

### Dual-card leftover (run on the box — not faked here)

1. `ft gpu`: two `gfx1201` R9700s; iGPU hidden; `tp_discrete: 2`.
2. Command above reaches READY (RCCL init, no PyNCCL/NCCL 2.27, no flashinfer).
3. One short `chat/completions` (`max_tokens=16`) — leftover until you run it.
4. Watch host RAM: one ~137 GiB pool, not two. Free RAM ≳ 160 GB before load.
5. Do **not** record tok/s. **R9700 tok/s is not published.**
6. If RCCL P2P hangs, try `NCCL_P2P_DISABLE=1` and report; that is leftover.

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
6. Dual-card checklist above (`--gpu 0,1`, disk `FREETOKEN_BANK_SHARE_DIR`, shared banks). KV is per-card; dual TP does not unlock 128k.

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
