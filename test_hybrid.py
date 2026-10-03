import unittest

import numpy as np

from hybrid import (
    convolve_2d,
    cross_correlation_2d,
    gaussian_blur_kernel_2d,
    high_pass,
    low_pass,
)


class HybridFilteringTests(unittest.TestCase):
    def test_cross_correlation_identity_grayscale(self):
        image = np.arange(12, dtype=float).reshape(3, 4)
        result = cross_correlation_2d(image, np.array([[1.0]]))
        np.testing.assert_allclose(result, image)

    def test_cross_correlation_identity_color(self):
        image = np.arange(36, dtype=float).reshape(3, 4, 3)
        result = cross_correlation_2d(image, np.array([[1.0]]))
        np.testing.assert_allclose(result, image)

    def test_convolution_flips_kernel(self):
        image = np.arange(1, 10, dtype=float).reshape(3, 3)
        kernel = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = cross_correlation_2d(image, np.flip(kernel, axis=(0, 1)))
        np.testing.assert_allclose(convolve_2d(image, kernel), expected)

    def test_gaussian_kernel_is_normalized_and_symmetric(self):
        kernel = gaussian_blur_kernel_2d(1.7, 5, 7)
        self.assertAlmostEqual(float(np.sum(kernel)), 1.0)
        np.testing.assert_allclose(kernel, np.flip(kernel, axis=(0, 1)))

    def test_low_plus_high_reconstructs_image(self):
        rng = np.random.default_rng(42)
        image = rng.random((8, 9, 3))
        low = low_pass(image, sigma=1.2, size=5)
        high = high_pass(image, sigma=1.2, size=5)
        np.testing.assert_allclose(low + high, image, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
