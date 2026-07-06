import unittest

import torch

from evox.triton_kernels import fused_add, has_triton, triton_supports_device


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


if __name__ == "__main__":
    unittest.main()
