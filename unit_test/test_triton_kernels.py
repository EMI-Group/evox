import unittest
from unittest import mock

import torch

import evox.triton_kernels.backend as triton_backend
from evox.triton_kernels import (
    fused_add,
    has_triton,
    register_triton_device_type,
    register_triton_op,
    triton_device_types,
    triton_supports_device,
)


class TestTritonKernels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_has_triton(self):
        # has_triton() should always return a bool regardless of whether
        # Triton is actually installed on this machine.
        self.assertIsInstance(has_triton(), bool)

    def test_triton_supports_device_cpu(self):
        # Triton kernels are CUDA-only, so CPU must never be supported.
        self.assertFalse(triton_supports_device("cpu"))

    def test_triton_supports_device_invalid_string(self):
        # Any device string should be accepted without raising; the result
        # is a bool (True only when Triton + CUDA + cuda.is_available()).
        self.assertIsInstance(triton_supports_device("cuda"), bool)

    def test_fused_add_cpu(self):
        x = torch.randn(100)
        y = torch.randn(100)
        result = fused_add(x, y)
        self.assertTrue(torch.allclose(result, x + y))

    def test_fused_add_shapes(self):
        # 1D
        x1 = torch.randn(64)
        y1 = torch.randn(64)
        r1 = fused_add(x1, y1)
        self.assertEqual(r1.shape, x1.shape)
        self.assertTrue(torch.allclose(r1, x1 + y1))

        # 2D
        x2 = torch.randn(32, 16)
        y2 = torch.randn(32, 16)
        r2 = fused_add(x2, y2)
        self.assertEqual(r2.shape, x2.shape)
        self.assertTrue(torch.allclose(r2, x2 + y2))

    def test_fused_add_dtype(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(50, dtype=dtype)
            y = torch.randn(50, dtype=dtype)
            result = fused_add(x, y)
            self.assertEqual(result.dtype, dtype)
            self.assertTrue(torch.allclose(result, x + y))

    def test_fused_add_vmap_cpu(self):
        # Batched execution via vmap. The custom op falls back to the
        # BatchedFallback on CPU, which may emit a warning — that is fine.
        import warnings

        x_batch = torch.randn(4, 100)
        y_batch = torch.randn(4, 100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = torch.vmap(fused_add)(x_batch, y_batch)
        self.assertTrue(torch.allclose(result, x_batch + y_batch))

    def test_fused_add_fake_meta(self):
        # The fake function registered for the op enables torch.compile
        # tracing. Under the meta device it must produce a tensor with the
        # correct shape without performing real computation.
        with torch.device("meta"):
            x = torch.randn(100)
            y = torch.randn(100)
            result = fused_add(x, y)
        self.assertEqual(result.shape, x.shape)

    def test_triton_device_types_default(self):
        # By default, only the "cuda" backend is registered for Triton kernels.
        types = triton_device_types()
        self.assertIsInstance(types, frozenset)
        self.assertIn("cuda", types)

    def test_register_triton_device_type(self):
        # Registering an extra device type (e.g. Ascend NPU) makes it
        # supported when Triton is importable. Restore the original set
        # afterwards so this test does not pollute other tests.
        original = set(triton_backend._triton_device_types)
        try:
            register_triton_device_type("npu")
            self.assertIn("npu", triton_device_types())
            with mock.patch.object(triton_backend, "has_triton", return_value=True):
                self.assertTrue(triton_supports_device("npu"))
                self.assertFalse(triton_supports_device("cpu"))
        finally:
            triton_backend._triton_device_types = original

    def test_register_triton_op_device_types(self):
        # register_triton_op must accept triton_device_types as a string and
        # as a list without raising. Uses throwaway ops with unique names.
        # Note: torch.library.custom_op requires type annotations for schema
        # inference, so the throwaway functions must be fully annotated.
        def _fake(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        def _fallback(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        def _triton(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        op1 = register_triton_op(
            name="evox::_test_str_device",
            fake_fn=_fake,
            triton_fn=_triton,
            device_types=["cpu"],
            triton_device_types="npu",
        )(_fallback)
        self.assertIsNotNone(op1)

        op2 = register_triton_op(
            name="evox::_test_list_device",
            fake_fn=_fake,
            triton_fn=_triton,
            device_types=["cpu"],
            triton_device_types=["cuda", "npu"],
        )(_fallback)
        self.assertIsNotNone(op2)


if __name__ == "__main__":
    unittest.main()
