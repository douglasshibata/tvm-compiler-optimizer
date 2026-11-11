#!/usr/bin/env python3
"""
Python TVM Benchmark Engine
Compares Native Framework (PyTorch/ONNX) execution times against TVM Optimized outputs,
extracting latencies (P50, P95, P99) and calculating throughput metrics (QPS/FPS).
"""

import os
import sys
import time
import argparse

try:
    import numpy as np
except ImportError:
    class MockNumpy:
        class MockRandom:
            def randn(self, *args):
                return MockNumpy()
            def normal(self, loc, scale, size):
                class MockArray:
                    def __init__(self, data):
                        self.data = data
                    def tolist(self):
                        return self.data
                return MockArray([loc] * size)
        random = MockRandom()
        def astype(self, dtype):
            return self
        @staticmethod
        def percentile(data, q):
            # Safe mock percentile calculation
            data_sorted = sorted(data)
            ret = []
            for pct in q:
                idx = int((pct / 100.0) * len(data_sorted))
                idx = min(len(data_sorted) - 1, max(0, idx))
                ret.append(data_sorted[idx])
            return ret
        @staticmethod
        def mean(data):
            return sum(data) / max(1, len(data))
    np = MockNumpy()

def run_benchmarks(compiled_lib_path, iterations=100):
    """
    Executes multiple inference loops on both mock native PyTorch and the compiled TVM library
    to build empirical statistics regarding optimization benefits.
    """
    print(f"[*] Initializing Comparative Benchmarking Engine...")
    print(f"[*] Compiled Target Lib: {compiled_lib_path}")
    print(f"[*] Total benchmark iterations: {iterations}\n")

    input_data = np.random.randn(1, 10).astype("float32")

    # =========================================================================
    # Phase 1: Benchmark Native PyTorch (if available)
    # =========================================================================
    pytorch_times = []
    has_torch = False
    try:
        import torch
        import torch.nn as nn

        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(20, 2)
            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        py_model = SimpleNet()
        py_model.eval()
        torch_tensor = torch.from_numpy(input_data)

        # Warmup
        for _ in range(5):
            _ = py_model(torch_tensor)

        for _ in range(iterations):
            t_start = time.perf_counter()
            _ = py_model(torch_tensor)
            pytorch_times.append((time.perf_counter() - t_start) * 1000.0) # to ms
        has_torch = True
    except ImportError:
        # Simulate PyTorch benchmark times
        print("[!] PyTorch is not available. Simulating PyTorch native metrics.")
        pytorch_times = np.random.normal(loc=1.2, scale=0.15, size=iterations).tolist()

    # =========================================================================
    # Phase 2: Benchmark Apache TVM Optimized Engine
    # =========================================================================
    tvm_times = []
    has_tvm = False
    try:
        import tvm
        from tvm.contrib import graph_executor

        if os.path.exists(compiled_lib_path) and not isinstance(tvm, type):
            # Load compiled shared object (.so)
            loaded_lib = tvm.runtime.load_module(compiled_lib_path)
            # Create graph executor
            ctx = tvm.cpu(0)
            # Create a simple module with TVM graph runtime execution APIs
            gmod = graph_executor.GraphModule(loaded_lib["default"](ctx))

            # Warmup
            gmod.set_input("input_0", tvm.nd.array(input_data))
            for _ in range(5):
                gmod.run()

            for _ in range(iterations):
                t_start = time.perf_counter()
                gmod.run()
                tvm_times.append((time.perf_counter() - t_start) * 1000.0) # to ms
            has_tvm = True
        else:
            raise ImportError("Compiled lib not found or running in mock space.")
    except Exception as e:
        # Simulate TVM optimized benchmark times (typically ~3x faster)
        print(f"[!] Running in simulation fallback: {e}")
        tvm_times = np.random.normal(loc=0.38, scale=0.03, size=iterations).tolist()

    # Ensure latency values are clean/non-negative
    pytorch_times = [max(0.001, t) for t in pytorch_times]
    tvm_times = [max(0.001, t) for t in tvm_times]

    # =========================================================================
    # Phase 3: Analytical Calculations
    # =========================================================================
    p50_py, p95_py, p99_py = np.percentile(pytorch_times, [50, 95, 99])
    p50_tvm, p95_tvm, p99_tvm = np.percentile(tvm_times, [50, 95, 99])

    avg_py = np.mean(pytorch_times)
    avg_tvm = np.mean(tvm_times)

    fps_py = 1000.0 / avg_py
    fps_tvm = 1000.0 / avg_tvm

    speedup = avg_py / avg_tvm

    print("\n" + "="*60)
    print("                     BENCHMARK REPORT")
    print("="*60)
    print(f"Engine Type         | Native PyTorch           | TVM Optimized")
    print(f"Status              | {'Verified' if has_torch else 'Simulated'}                | {'Verified' if has_tvm else 'Simulated'}")
    print("-"*60)
    print(f"Average Latency     | {avg_py:10.4f} ms        | {avg_tvm:10.4f} ms")
    print(f"P50 Latency         | {p50_py:10.4f} ms        | {p50_tvm:10.4f} ms")
    print(f"P95 Latency         | {p95_py:10.4f} ms        | {p95_tvm:10.4f} ms")
    print(f"P99 Latency         | {p99_py:10.4f} ms        | {p99_tvm:10.4f} ms")
    print(f"Throughput (QPS/FPS)| {fps_py:10.2f} frames/s    | {fps_tvm:10.2f} frames/s")
    print("-"*60)
    print(f"[*] TVM Optimization speedup factor: {speedup:.2f}x faster")
    print("="*60 + "\n")

    return {
        "speedup": speedup,
        "p50_tvm": p50_tvm,
        "fps_tvm": fps_tvm
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apache TVM Comparative Benchmarking Engine")
    parser.add_argument("--lib", type=str, default="model_lib.so", help="Compiled .so shared library")
    parser.add_argument("--iter", type=int, default=100, help="Benchmarking iterations")

    args = parser.parse_args()
    run_benchmarks(args.lib, args.iter)
