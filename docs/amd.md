# AMD ROCm / HIP (Linux)

Native HIP backend for this fork. **ZLUDA is not supported**
([FlashML-org/FreeToken#60](https://github.com/FlashML-org/FreeToken/issues/60)).

First-class target: **Linux x86_64, 2× AMD Radeon AI PRO R9700, RDNA 4, `gfx1201`**.

This cloud / CI environment does **not** have an R9700. The checks below that
need a GPU must be run on the workstation. Nothing in this document invents
tok/s or unpublished numbers.

## What this PR does vs leftover

| Works in-tree (additive; NVIDIA CUDA path unchanged) | Leftover / on-box only |
|---|---|
| Detect AMD vs NVIDIA (`ft gpu`); hide Granite Ridge iGPU | Dual-card TP / RCCL not validated |
| HIP JIT of tvm-ffi kernels (`index`, `store`, `fast_index_copy`, …) for `gfx1201` | Vulkan (community: the hard problem) |
| HIP `host_register` / `device_ptr` (#122 TDR fix) | Full e2e on R9700 (must run locally) |
| Attention auto → `triton`; NVIDIA cubin backends fail loudly | Qwen3.8-27B FP8 / NVFP4 fused donors; DSV4/MLA/BSA (need NVIDIA cubins) |
| GGUF HIP compile (`-DUSE_ROCM`, `--offload-arch`) + rocThrust shim | gfx1201 HIP **graph replay** (TDR on Windows #82) |
| MoE `offload` / `fused` / `cpu` with Triton experts | Windows ROCm (see community forks) |

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

```bash
ft gpu
# vendor: amd
# gcn_arch: gfx1201
#   gpu0: AMD Radeon AI PRO R9700 ...
```

A Ryzen 9 9950X iGPU (`AMD Radeon Graphics` / `gfx1036`) is hidden so it does
not become device 0. Strix Halo is **not** hidden.

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

1. `ft gpu` lists both R9700s as `gfx1201`; iGPU absent or marked hidden.
2. `ft serve` on a small dense model reaches READY and returns a completion.
3. Same with `--moe-backend offload` on a small/supported MoE or GGUF (no TDR).
4. Optional: `FREETOKEN_HIP_GRAPH_REPLAY=1` — report whether graph replay is
   stable on **Linux** gfx1201 (Windows #82 said no for MoE).
5. Optional second card: `--gpu 0,1` / TP — leftover, say what happened.

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

Unchanged. `uv pip install -e ".[accel]"` still resolves cu130 torch. AMD is
additive.
