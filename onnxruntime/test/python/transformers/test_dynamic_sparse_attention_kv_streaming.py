# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Bit-exactness and lifecycle tests for DynamicSparseAttention QSA KV cache streaming
(`ep.cuda.dsa_kv_hot_cache_tokens`): pinned device-mapped host K/V + on-GPU CLOCK hot-block cache
and selective single-layer prefill VRAM staging, including CUDA graph capture/replay, speculative
draft rollback, unaligned prefill chunks, and new-turn self-reset (`dest_start == 0`).
"""

import ctypes
import ctypes.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from cuda_plugin_ep_helper import CUDA_PLUGIN_EP_NAME, ensure_cuda_plugin_ep_registered
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort

NUM_HEADS = 24
KV_HEADS = 2
HEAD_SIZE = 256
SOFTMAX_SCALE = 0.0625

_DLL_HANDLES = []


def _ensure_dll_directories():
    if not hasattr(os, "add_dll_directory") or _DLL_HANDLES:
        return
    torch_lib = str(Path(torch.__file__).resolve().parent / "lib")
    cuda_path = os.environ.get("CUDA_PATH", "")
    cudnn_home = os.environ.get("CUDNN_HOME", os.environ.get("CUDNN_PATH", ""))
    plugin_path = os.environ.get("ORT_CUDA_PLUGIN_PATH", "")
    candidates = [
        os.path.dirname(plugin_path) if plugin_path else "",
        torch_lib,
        os.path.join(cuda_path, "bin", "x64") if cuda_path else "",
        os.path.join(cuda_path, "bin") if cuda_path else "",
        os.path.join(cudnn_home, "bin", "x64") if cudnn_home else "",
        os.path.join(cudnn_home, "bin") if cudnn_home else "",
    ]
    seen = set()
    for d in candidates:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            _DLL_HANDLES.append(os.add_dll_directory(d))


def _load_cudart():
    _ensure_dll_directories()
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    cuda_root = Path(os.environ["CUDA_PATH"]) if os.environ.get("CUDA_PATH") else None
    candidates: list[str] = []

    if sys.platform == "win32":
        search_dirs = [torch_lib]
        if cuda_root is not None:
            search_dirs.extend([cuda_root / "bin" / "x64", cuda_root / "bin"])
        for d in search_dirs:
            if d.is_dir():
                candidates.extend(str(p) for p in sorted(d.glob("cudart64_*.dll"), reverse=True))
        found = ctypes.util.find_library("cudart")
        if found:
            candidates.append(found)
        candidates.extend(["cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll"])
    elif sys.platform == "darwin":
        found = ctypes.util.find_library("cudart")
        if found:
            candidates.append(found)
        candidates.append("libcudart.dylib")
    else:
        search_dirs = [torch_lib]
        if cuda_root is not None:
            search_dirs.extend([cuda_root / "lib64", cuda_root / "lib"])
        for d in search_dirs:
            if d.is_dir():
                candidates.extend(str(p) for p in sorted(d.glob("libcudart.so*"), reverse=True))
        found = ctypes.util.find_library("cudart")
        if found:
            candidates.append(found)
        candidates.extend(["libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0"])

    lib = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            lib = None
    if lib is None:
        raise RuntimeError(f"Could not load CUDA runtime library (tried: {candidates})")
    lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    lib.cudaHostAlloc.restype = ctypes.c_int
    lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
    lib.cudaFreeHost.restype = ctypes.c_int
    lib.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    lib.cudaHostGetDevicePointer.restype = ctypes.c_int
    return lib


class MappedHostBuffer:
    def __init__(self, cudart, nbytes: int):
        self._cudart = cudart
        self.nbytes = max(16, int(nbytes))
        self.host_ptr = ctypes.c_void_p()
        self.dev_ptr = ctypes.c_void_p()
        # cudaHostAllocPortable (1) | cudaHostAllocMapped (2) = 3
        err = self._cudart.cudaHostAlloc(ctypes.byref(self.host_ptr), self.nbytes, 3)
        if err != 0:
            raise RuntimeError(f"cudaHostAlloc failed with error {err}")
        err = self._cudart.cudaHostGetDevicePointer(ctypes.byref(self.dev_ptr), self.host_ptr, 0)
        if err != 0:
            self._cudart.cudaFreeHost(self.host_ptr)
            self.host_ptr = ctypes.c_void_p()
            raise RuntimeError(f"cudaHostGetDevicePointer failed with error {err}")
        ctypes.memset(self.host_ptr, 0, self.nbytes)

    def numpy_view(self, dtype, shape):
        arr_t = ctypes.c_byte * int(np.prod(shape) * np.dtype(dtype).itemsize)
        buf = arr_t.from_address(self.host_ptr.value)
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    def close(self):
        if self.host_ptr.value:
            self._cudart.cudaFreeHost(self.host_ptr)
            self.host_ptr = ctypes.c_void_p()
            self.dev_ptr = ctypes.c_void_p()

    def __del__(self):
        self.close()


def _build_dsa_model(cache_capacity: int, max_selected: int, cache_elem_type: int, key_scale=None, value_scale=None):
    def vi(name, elem_type, shape):
        return helper.make_tensor_value_info(name, elem_type, shape)

    inputs = [
        "query",
        "key",
        "value",
        "past_key",
        "past_value",
        "",
        "",
        "selected_indices",
        "selected_counts",
        "seqlens_k",
        "total_sequence_length",
    ]
    initializers = [numpy_helper.from_array(np.array(cache_capacity, dtype=np.int32), name="total_sequence_length")]
    if key_scale is not None and value_scale is not None:
        inputs.extend([""] * 7 + ["key_scale", "value_scale"])
        initializers.extend(
            [
                numpy_helper.from_array(key_scale, name="key_scale"),
                numpy_helper.from_array(value_scale, name="value_scale"),
            ]
        )

    cache_shape = [1, KV_HEADS, cache_capacity, HEAD_SIZE]
    graph = helper.make_graph(
        [
            helper.make_node(
                "DynamicSparseAttention",
                inputs,
                ["output", "present_key", "present_value"],
                name="dsa_node_0",
                domain="com.microsoft",
                num_heads=NUM_HEADS,
                kv_num_heads=KV_HEADS,
                scale=SOFTMAX_SCALE,
            )
        ],
        "dsa_kv_stream_test",
        [
            vi("query", TensorProto.BFLOAT16, [1, "rows", NUM_HEADS * HEAD_SIZE]),
            vi("key", TensorProto.BFLOAT16, [1, "rows", KV_HEADS * HEAD_SIZE]),
            vi("value", TensorProto.BFLOAT16, [1, "rows", KV_HEADS * HEAD_SIZE]),
            vi("past_key", cache_elem_type, cache_shape),
            vi("past_value", cache_elem_type, cache_shape),
            vi("selected_indices", TensorProto.INT32, ["rows", max_selected]),
            vi("selected_counts", TensorProto.INT32, ["rows"]),
            vi("seqlens_k", TensorProto.INT32, [1]),
        ],
        [
            vi("output", TensorProto.BFLOAT16, [1, "rows", NUM_HEADS * HEAD_SIZE]),
            vi("present_key", cache_elem_type, cache_shape),
            vi("present_value", cache_elem_type, cache_shape),
        ],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model.SerializeToString()


def _create_session(model_bytes: bytes, session_entries: dict[str, str], enable_cuda_graph: bool = False):
    _ensure_dll_directories()
    so = ort.SessionOptions()
    for k, v in session_entries.items():
        so.add_session_config_entry(k, v)
    ep_opts = {"enable_cuda_graph": "1"} if enable_cuda_graph else {}
    if ensure_cuda_plugin_ep_registered():
        ep_devs = [d for d in ort.get_ep_devices() if d.ep_name == CUDA_PLUGIN_EP_NAME]
        so.add_provider_for_devices(ep_devs[:1], ep_opts)
        return ort.InferenceSession(model_bytes, sess_options=so)
    return ort.InferenceSession(model_bytes, sess_options=so, providers=[("CUDAExecutionProvider", ep_opts)])


def _pick_sparse_blocks(rng: np.random.Generator, rows: int, dest_start: int, max_selected: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.full((rows, max_selected), -1, dtype=np.int32)
    counts = np.zeros((rows,), dtype=np.int32)
    for r in range(rows):
        qpos = dest_start + r
        avail_blocks = (qpos // 4) + 1
        max_blocks = max_selected // 4
        # Always include sink block 0 and tail block, plus random historical blocks to stress CLOCK eviction.
        chosen_blocks = {0, qpos // 4}
        if avail_blocks > 2 and len(chosen_blocks) < max_blocks:
            pool = np.array([b for b in range(avail_blocks) if b not in chosen_blocks], dtype=np.int32)
            take = min(len(pool), max_blocks - len(chosen_blocks))
            if take > 0:
                chosen_blocks.update(int(x) for x in rng.choice(pool, size=take, replace=False))
        flat = []
        for b in sorted(chosen_blocks):
            for sub in range(4):
                tok = b * 4 + sub
                if tok <= qpos and len(flat) < max_selected:
                    flat.append(tok)
        counts[r] = len(flat)
        indices[r, : len(flat)] = np.array(flat, dtype=np.int32)
    return indices, counts


class TestDynamicSparseAttentionKvStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
            raise unittest.SkipTest("Requires SM80+ CUDA GPU")
        cls.cudart = _load_cudart()

    def _run_parity_scenario(self, cache_mode: str, decode_tc: bool, enable_cuda_graph: bool, max_selected: int = 512):
        rng = np.random.default_rng(20261115)
        torch.manual_seed(20261115)
        if max_selected > 256:
            # Exercises DynamicSparseAttentionInt8TcKernel / DynamicSparseAttentionFp16TcKernel (max_selected > 256)
            cache_capacity = 8192
            hot_tokens = 4160  # 1,040 slots = 8 * (512/4 + 2) < 2,048 total blocks -> active CLOCK evictions!
            prefill_chunks = [(1024, 0), (37, 1024), (4096, 1061)]  # ends at 5197 > 4160 hot_tokens!
        else:
            # Exercises FusedDynamicSparseAttentionKernel (max_selected <= 256)
            cache_capacity = 1024
            hot_tokens = 640
            prefill_chunks = [(512, 0), (37, 512), (200, 549)]

        if cache_mode == "int8":
            elem_type = TensorProto.INT8
            np_dtype = np.int8
            torch_cache_dtype = torch.int8
            key_scale = rng.uniform(0.005, 0.05, KV_HEADS * HEAD_SIZE).astype(np.float32)
            value_scale = rng.uniform(0.005, 0.03, KV_HEADS * HEAD_SIZE).astype(np.float32)
        else:
            elem_type = TensorProto.FLOAT16
            np_dtype = np.float16
            torch_cache_dtype = torch.float16
            key_scale = None
            value_scale = None

        model_bytes = _build_dsa_model(cache_capacity, max_selected, elem_type, key_scale, value_scale)
        base_entries = {
            "ep.cuda.dsa_decode_int8_tc": "1" if decode_tc else "0",
            "ep.cuda.dsa_kv_hot_cache_tokens": "0",
        }
        stream_entries = {
            "ep.cuda.dsa_decode_int8_tc": "1" if decode_tc else "0",
            "ep.cuda.dsa_kv_hot_cache_tokens": str(hot_tokens),
        }
        ref_sess = _create_session(model_bytes, base_entries, enable_cuda_graph=enable_cuda_graph)
        dut_sess = _create_session(model_bytes, stream_entries, enable_cuda_graph=enable_cuda_graph)

        cache_shape = (1, KV_HEADS, cache_capacity, HEAD_SIZE)
        cache_nbytes = int(np.prod(cache_shape) * np.dtype(np_dtype).itemsize)
        ref_k = torch.zeros(cache_shape, dtype=torch_cache_dtype, device="cuda")
        ref_v = torch.zeros(cache_shape, dtype=torch_cache_dtype, device="cuda")
        dut_k_map = MappedHostBuffer(self.cudart, cache_nbytes)
        dut_v_map = MappedHostBuffer(self.cudart, cache_nbytes)

        total_seq_cpu = np.array([cache_capacity], dtype=np.int32)

        def run_step(rows: int, dest_start: int, fixed_buffers=None, graph_run_opts=None):
            total_len = dest_start + rows
            if fixed_buffers is None:
                q = torch.randn((1, rows, NUM_HEADS * HEAD_SIZE), dtype=torch.bfloat16, device="cuda")
                k = torch.randn((1, rows, KV_HEADS * HEAD_SIZE), dtype=torch.bfloat16, device="cuda")
                v = torch.randn((1, rows, KV_HEADS * HEAD_SIZE), dtype=torch.bfloat16, device="cuda")
                sel_idx_np, sel_cnt_np = _pick_sparse_blocks(rng, rows, dest_start, max_selected)
                sel_idx = torch.from_numpy(sel_idx_np).to("cuda")
                sel_cnt = torch.from_numpy(sel_cnt_np).to("cuda")
                seqlens = torch.tensor([total_len - 1], dtype=torch.int32, device="cuda")
                ref_out = torch.empty_like(q)
                dut_out = torch.empty_like(q)
            else:
                q, k, v, sel_idx, sel_cnt, seqlens, ref_out, dut_out = fixed_buffers
                q.normal_()
                k.normal_()
                v.normal_()
                sel_idx_np, sel_cnt_np = _pick_sparse_blocks(rng, rows, dest_start, max_selected)
                sel_idx.copy_(torch.from_numpy(sel_idx_np))
                sel_cnt.copy_(torch.from_numpy(sel_cnt_np))
                seqlens.fill_(total_len - 1)

            b_ref = ref_sess.io_binding()
            b_dut = dut_sess.io_binding()
            for b, pk_ptr, pv_ptr, out_t in (
                (b_ref, ref_k.data_ptr(), ref_v.data_ptr(), ref_out),
                (b_dut, dut_k_map.dev_ptr.value, dut_v_map.dev_ptr.value, dut_out),
            ):
                b.bind_input("query", "cuda", 0, TensorProto.BFLOAT16, tuple(q.shape), q.data_ptr())
                b.bind_input("key", "cuda", 0, TensorProto.BFLOAT16, tuple(k.shape), k.data_ptr())
                b.bind_input("value", "cuda", 0, TensorProto.BFLOAT16, tuple(v.shape), v.data_ptr())
                b.bind_input("past_key", "cuda", 0, elem_type, cache_shape, pk_ptr)
                b.bind_input("past_value", "cuda", 0, elem_type, cache_shape, pv_ptr)
                b.bind_input("selected_indices", "cuda", 0, TensorProto.INT32, tuple(sel_idx.shape), sel_idx.data_ptr())
                b.bind_input("selected_counts", "cuda", 0, TensorProto.INT32, tuple(sel_cnt.shape), sel_cnt.data_ptr())
                b.bind_input("seqlens_k", "cuda", 0, TensorProto.INT32, (1,), seqlens.data_ptr())
                b.bind_output("output", "cuda", 0, TensorProto.BFLOAT16, tuple(out_t.shape), out_t.data_ptr())
                b.bind_output("present_key", "cuda", 0, elem_type, cache_shape, pk_ptr)
                b.bind_output("present_value", "cuda", 0, elem_type, cache_shape, pv_ptr)

            ref_sess.run_with_iobinding(b_ref, graph_run_opts)
            dut_sess.run_with_iobinding(b_dut, graph_run_opts)
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(ref_out.view(torch.uint16), dut_out.view(torch.uint16)),
                f"Output mismatch at rows={rows}, dest_start={dest_start}, mode={cache_mode}, tc={decode_tc}",
            )

        eager_ro = ort.RunOptions()
        eager_ro.add_run_config_entry("gpu_graph_id", "-1")

        # 1. Chunked prefill including an unaligned chunk
        for chunk_rows, chunk_start in prefill_chunks:
            run_step(chunk_rows, chunk_start, graph_run_opts=eager_ro)
        committed_after_prefill = prefill_chunks[-1][0] + prefill_chunks[-1][1]

        # Verify authoritative host KV matches VRAM KV across all committed positions
        np.testing.assert_array_equal(
            ref_k[:, :, :committed_after_prefill, :].cpu().numpy(),
            dut_k_map.numpy_view(np_dtype, cache_shape)[:, :, :committed_after_prefill, :],
        )
        np.testing.assert_array_equal(
            ref_v[:, :, :committed_after_prefill, :].cpu().numpy(),
            dut_v_map.numpy_view(np_dtype, cache_shape)[:, :, :committed_after_prefill, :],
        )

        # 2. Multi-step S=8 speculative verification with partial acceptance (rollback!) & CLOCK evictions
        dec_rows = 8
        fixed = (
            torch.empty((1, dec_rows, NUM_HEADS * HEAD_SIZE), dtype=torch.bfloat16, device="cuda"),
            torch.empty((1, dec_rows, KV_HEADS * HEAD_SIZE), dtype=torch.bfloat16, device="cuda"),
            torch.empty((1, dec_rows, KV_HEADS * HEAD_SIZE), dtype=torch.bfloat16, device="cuda"),
            torch.empty((dec_rows, max_selected), dtype=torch.int32, device="cuda"),
            torch.empty((dec_rows,), dtype=torch.int32, device="cuda"),
            torch.empty((1,), dtype=torch.int32, device="cuda"),
            torch.empty((1, dec_rows, NUM_HEADS * HEAD_SIZE), dtype=torch.bfloat16, device="cuda"),
            torch.empty((1, dec_rows, NUM_HEADS * HEAD_SIZE), dtype=torch.bfloat16, device="cuda"),
        )
        pos = committed_after_prefill
        for step in range(25):
            run_step(dec_rows, pos, fixed_buffers=fixed, graph_run_opts=None if enable_cuda_graph else eager_ro)
            # Accept 1..8 tokens (forcing frequent speculative draft rejections and partial-block overwrites)
            accepted = 1 + (step % 6)
            pos += accepted
            if pos + dec_rows > cache_capacity:
                break

        np.testing.assert_array_equal(
            ref_k[:, :, :pos, :].cpu().numpy(),
            dut_k_map.numpy_view(np_dtype, cache_shape)[:, :, :pos, :],
        )

        # 3. Turn reset (`dest_start == 0`) directly through the captured S=8 decode graph!
        ref_k.zero_()
        ref_v.zero_()
        ctypes.memset(dut_k_map.host_ptr, 0, cache_nbytes)
        ctypes.memset(dut_v_map.host_ptr, 0, cache_nbytes)
        pos = 0
        for step in range(8):
            run_step(dec_rows, pos, fixed_buffers=fixed, graph_run_opts=None if enable_cuda_graph else eager_ro)
            pos += 3 + (step % 4)

        dut_k_map.close()
        dut_v_map.close()

    def test_int8_tc_kv_streaming_with_cuda_graph(self):
        self._run_parity_scenario(cache_mode="int8", decode_tc=True, enable_cuda_graph=True)

    def test_int8_cuda_core_kv_streaming(self):
        self._run_parity_scenario(cache_mode="int8", decode_tc=False, enable_cuda_graph=False)

    def test_fp16_tc_kv_streaming_with_cuda_graph(self):
        self._run_parity_scenario(cache_mode="fp16", decode_tc=True, enable_cuda_graph=True)

    def test_int8_fused_small_budget_kv_streaming(self):
        self._run_parity_scenario(cache_mode="int8", decode_tc=True, enable_cuda_graph=True, max_selected=64)

    def test_device_backed_kv_bypasses_streaming_when_hot_cache_option_set(self):
        rng = np.random.default_rng(42)
        torch.manual_seed(42)
        cache_capacity = 2048
        max_selected = 512
        key_scale = rng.uniform(0.005, 0.05, KV_HEADS * HEAD_SIZE).astype(np.float32)
        value_scale = rng.uniform(0.005, 0.03, KV_HEADS * HEAD_SIZE).astype(np.float32)
        model_bytes = _build_dsa_model(cache_capacity, max_selected, TensorProto.INT8, key_scale, value_scale)
        ref_sess = _create_session(model_bytes, {"ep.cuda.dsa_kv_hot_cache_tokens": "0"}, enable_cuda_graph=True)
        dut_sess = _create_session(model_bytes, {"ep.cuda.dsa_kv_hot_cache_tokens": "512"}, enable_cuda_graph=True)

        cache_shape = (1, KV_HEADS, cache_capacity, HEAD_SIZE)
        ref_k = torch.zeros(cache_shape, dtype=torch.int8, device="cuda")
        ref_v = torch.zeros(cache_shape, dtype=torch.int8, device="cuda")
        dut_k = torch.zeros(cache_shape, dtype=torch.int8, device="cuda")
        dut_v = torch.zeros(cache_shape, dtype=torch.int8, device="cuda")
        q = torch.randn((1, 8, NUM_HEADS * HEAD_SIZE), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((1, 8, KV_HEADS * HEAD_SIZE), dtype=torch.bfloat16, device="cuda")
        v = torch.randn((1, 8, KV_HEADS * HEAD_SIZE), dtype=torch.bfloat16, device="cuda")
        sel_idx_np, sel_cnt_np = _pick_sparse_blocks(rng, 8, 0, max_selected)
        sel_idx = torch.from_numpy(sel_idx_np).to("cuda")
        sel_cnt = torch.from_numpy(sel_cnt_np).to("cuda")
        seqlens = torch.tensor([7], dtype=torch.int32, device="cuda")
        ref_out = torch.empty_like(q)
        dut_out = torch.empty_like(q)
        for sess, pk, pv, out in ((ref_sess, ref_k, ref_v, ref_out), (dut_sess, dut_k, dut_v, dut_out)):
            b = sess.io_binding()
            b.bind_input("query", "cuda", 0, TensorProto.BFLOAT16, tuple(q.shape), q.data_ptr())
            b.bind_input("key", "cuda", 0, TensorProto.BFLOAT16, tuple(k.shape), k.data_ptr())
            b.bind_input("value", "cuda", 0, TensorProto.BFLOAT16, tuple(v.shape), v.data_ptr())
            b.bind_input("past_key", "cuda", 0, TensorProto.INT8, cache_shape, pk.data_ptr())
            b.bind_input("past_value", "cuda", 0, TensorProto.INT8, cache_shape, pv.data_ptr())
            b.bind_input("selected_indices", "cuda", 0, TensorProto.INT32, tuple(sel_idx.shape), sel_idx.data_ptr())
            b.bind_input("selected_counts", "cuda", 0, TensorProto.INT32, tuple(sel_cnt.shape), sel_cnt.data_ptr())
            b.bind_input("seqlens_k", "cuda", 0, TensorProto.INT32, (1,), seqlens.data_ptr())
            b.bind_output("output", "cuda", 0, TensorProto.BFLOAT16, tuple(out.shape), out.data_ptr())
            b.bind_output("present_key", "cuda", 0, TensorProto.INT8, cache_shape, pk.data_ptr())
            b.bind_output("present_value", "cuda", 0, TensorProto.INT8, cache_shape, pv.data_ptr())
            sess.run_with_iobinding(b)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(ref_out.view(torch.uint16), dut_out.view(torch.uint16)))


if __name__ == "__main__":
    unittest.main()
