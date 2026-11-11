#!/usr/bin/env python3
"""
Unit and Integration Tests for Apache TVM Compilation Pipeline
"""

import os
import unittest
import tempfile
from tvm_compiler import compile_model
from tuning import setup_and_run_tuning
from benchmark import run_benchmarks

class TestTVMCompilationPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_so = os.path.join(self.test_dir.name, "test_model.so")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_compilation_pytorch_flow(self):
        """Verifies the complete compilation pipeline flow and final binary generation."""
        res = compile_model(
            model_type="pytorch",
            model_path="dummy_path.pt",
            target_device="llvm",
            output_path=self.output_so
        )
        self.assertTrue(res)
        self.assertTrue(os.path.exists(self.output_so))

    def test_compilation_onnx_flow(self):
        """Verifies ONNX import pipeline handling constraints."""
        res = compile_model(
            model_type="onnx",
            model_path="dummy_path.onnx",
            target_device="llvm",
            output_path=self.output_so
        )
        self.assertTrue(res)
        self.assertTrue(os.path.exists(self.output_so))

    def test_tuning_meta_schedule(self):
        """Checks the tuning engine database initialization and output generation."""
        tuning_log_dir = os.path.join(self.test_dir.name, "tuning_logs")
        db = setup_and_run_tuning(
            model_path=self.output_so,
            target_device="llvm",
            work_dir=tuning_log_dir,
            num_trials=5
        )
        self.assertIsNotNone(db)
        self.assertTrue(os.path.exists(tuning_log_dir))

    def test_benchmarking_report(self):
        """Tests execution and accuracy stats collection within benchmarking."""
        # Create a mock library file for benchmark parsing
        with open(self.output_so, "w") as f:
            f.write("/* Simulated Binary */")

        stats = run_benchmarks(self.output_so, iterations=10)
        self.assertIn("speedup", stats)
        self.assertIn("p50_tvm", stats)
        self.assertIn("fps_tvm", stats)
        self.assertGreater(stats["speedup"], 0)

if __name__ == "__main__":
    unittest.main()
