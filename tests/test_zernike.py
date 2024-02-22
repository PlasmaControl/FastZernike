from zernipy.zernike import *
import numpy as np


def test_zernike_radial(self):  # noqa: C901
    """Test zernike_radial function, comparing to analytic formulas."""
    # https://en.wikipedia.org/wiki/Zernike_polynomials#Radial_polynomials

    def Z3_1(x, dx=0):
        if dx == 0:
            return 3 * x**3 - 2 * x
        if dx == 1:
            return 9 * x**2 - 2
        if dx == 2:
            return 18 * x
        if dx == 3:
            return np.full_like(x, 18)
        if dx >= 4:
            return np.zeros_like(x)

    def Z4_2(x, dx=0):
        if dx == 0:
            return 4 * x**4 - 3 * x**2
        if dx == 1:
            return 16 * x**3 - 6 * x
        if dx == 2:
            return 48 * x**2 - 6
        if dx == 3:
            return 96 * x
        if dx == 4:
            return np.full_like(x, 96)
        if dx >= 5:
            return np.zeros_like(x)

    def Z6_2(x, dx=0):
        if dx == 0:
            return 15 * x**6 - 20 * x**4 + 6 * x**2
        if dx == 1:
            return 90 * x**5 - 80 * x**3 + 12 * x
        if dx == 2:
            return 450 * x**4 - 240 * x**2 + 12
        if dx == 3:
            return 1800 * x**3 - 480 * x
        if dx == 4:
            return 5400 * x**2 - 480
        if dx == 5:
            return 10800 * x
        if dx == 6:
            return np.full_like(x, 10800)
        if dx >= 7:
            return np.zeros_like(x)

    l = np.array([3, 4, 6, 4])
    m = np.array([1, 2, 2, 2])
    r = np.linspace(0, 1, 11)  # rho coordinates
    max_dr = 4
    desired = {
        dr: np.array([Z3_1(r, dr), Z4_2(r, dr), Z6_2(r, dr), Z4_2(r, dr)]).T
        for dr in range(max_dr + 1)
    }
    radial = {dr: zernike_radial(r, l, m, dr) for dr in range(max_dr + 1)}
    for dr in range(max_dr + 1):
        np.testing.assert_allclose(radial[dr], desired[dr], err_msg=dr)


def test_jacobi_poly_single(self):
    """Test Jacobi Polynomial evaluation for special cases."""
    # https://en.wikipedia.org/wiki/Jacobi_polynomials#Special_cases

    def exact(r, n, alpha, beta):
        if n == 0:
            return np.ones_like(r)
        elif n == 1:
            return (alpha + 1) + (alpha + beta + 2) * ((r - 1) / 2)
        elif n == 2:
            a0 = (alpha + 1) * (alpha + 2) / 2
            a1 = (alpha + 2) * (alpha + beta + 3)
            a2 = (alpha + beta + 3) * (alpha + beta + 4) / 2
            z = (r - 1) / 2
            return a0 + a1 * z + a2 * z**2
        elif n < 0:
            return np.zeros_like(r)

    r = np.linspace(0, 1, 11)
    # alpha and beta pairs for test
    pairs = np.array([[2, 3], [3, 0], [1, 1], [10, 4]])
    n_values = np.array([-1, -2, 0, 1, 2])

    for pair in pairs:
        alpha = pair[0]
        beta = pair[1]
        P0 = jacobi_poly_single(r, 0, alpha, beta)
        P1 = jacobi_poly_single(r, 1, alpha, beta)
        desired = {n: exact(r, n, alpha, beta) for n in n_values}
        values = {n: jacobi_poly_single(r, n, alpha, beta, P1, P0) for n in n_values}

        for n in n_values:
            np.testing.assert_allclose(values[n], desired[n], err_msg=n)


def test_fourier(self):
    """Test Fourier series evaluation."""
    m = np.array([-1, 0, 1])
    t = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # theta coordinates

    correct_vals = np.array([np.sin(t), np.ones_like(t), np.cos(t)]).T
    correct_ders = np.array([np.cos(t), np.zeros_like(t), -np.sin(t)]).T

    values = fourier(t[:, np.newaxis], m, dt=0)
    derivs = fourier(t[:, np.newaxis], m, dt=1)

    np.testing.assert_allclose(values, correct_vals, atol=1e-8)
    np.testing.assert_allclose(derivs, correct_ders, atol=1e-8)
