"""Functions for evaluating Zernike polynomials and their derivatives."""

import functools

from zernipy.backend import cond, custom_jvp, fori_loop, gammaln, jit, jnp, select, switch


def jacobi_poly_single(x, n, alpha, beta=0, P_n1=0, P_n2=0):
    """Evaluate Jacobi for single alpha and n pair."""
    c = 2 * n + alpha + beta
    a1 = 2 * n * (c - n) * (c - 2)
    a2 = (c - 1) * (c * (c - 2) * x + (alpha - beta) * (alpha + beta))
    a3 = 2 * (n + alpha - 1) * (n + beta - 1) * c

    a1 = jnp.where(a1 == 0, 1e-6, a1)
    P_n = (a2 * P_n1 - a3 * P_n2) / a1
    # Checks for special cases
    P_n = jnp.where(n < 0, 0, P_n)
    P_n = jnp.where(n == 0, 1, P_n)
    P_n = jnp.where(n == 1, (alpha + 1) + (alpha + beta + 2) * (x - 1) / 2, P_n)
    return P_n


@functools.partial(jit, static_argnums=3)
def zernike_radial_no_check(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes. This version
    is almost the same as zernike_radial_old function but way faster and more
    accurate. First version of this function is zernike_radial_separate which has
    many function for each derivative definition. User can refer that for clarity.


    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    if dr > 4:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]
        P_n1 = P_past[1]
        P_n = jnp.zeros((dr + 1, r.size))

        def find_inter_jacobi(dx, args):
            N, alpha, P_n1, P_n2, P_n = args
            P_n = P_n.at[dx, :].set(
                jacobi_poly_single(r_jacobi, N - dx, alpha + dx, dx, P_n1[dx], P_n2[dx])
            )
            return (N, alpha, P_n1, P_n2, P_n)

        # Calculate Jacobi polynomial and derivatives for (m,n)
        _, _, _, _, P_n = fori_loop(
            0, dr + 1, find_inter_jacobi, (N, alpha, P_n1, P_n2, P_n)
        )

        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )
        # TODO: A version without if statements are possible?
        if dr == 0:
            result = (-1) ** N * r**alpha * P_n[0]
        elif dr == 1:
            result = (-1) ** N * (
                alpha * r ** jnp.maximum(alpha - 1, 0) * P_n[0]
                - coef[1] * 4 * r ** (alpha + 1) * P_n[1]
            )
        elif dr == 2:
            result = (-1) ** N * (
                (alpha - 1) * alpha * r ** jnp.maximum(alpha - 2, 0) * P_n[0]
                - coef[1] * 4 * (2 * alpha + 1) * r**alpha * P_n[1]
                + coef[2] * 16 * r ** (alpha + 2) * P_n[2]
            )
        elif dr == 3:
            result = (-1) ** N * (
                (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 3, 0)
                * P_n[0]
                - coef[1] * 12 * alpha**2 * r ** jnp.maximum(alpha - 1, 0) * P_n[1]
                + coef[2] * 48 * (alpha + 1) * r ** (alpha + 1) * P_n[2]
                - coef[3] * 64 * r ** (alpha + 3) * P_n[3]
            )
        elif dr == 4:
            result = (-1) ** N * (
                (alpha - 3)
                * (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 4, 0)
                * P_n[0]
                - coef[1]
                * 8
                * alpha
                * (2 * alpha**2 - 3 * alpha + 1)
                * r ** jnp.maximum(alpha - 2, 0)
                * P_n[1]
                + coef[2] * 48 * (2 * alpha**2 + 2 * alpha + 1) * r**alpha * P_n[2]
                - coef[3] * 128 * (2 * alpha + 3) * r ** (alpha + 2) * P_n[3]
                + coef[4] * 256 * r ** (alpha + 4) * P_n[4]
            )
        index = jnp.where(
            jnp.logical_and(m == alpha, n == N),
            jnp.arange(m.size),
            0,
        )
        idx = jnp.sum(index)
        out = out.at[:, idx].set(result)

        # Shift past values if needed
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask[:, None], P_n1, P_n2)
        P_n1 = jnp.where(mask[:, None], P_n, P_n1)
        P_past = P_past.at[0, :, :].set(P_n2)
        P_past = P_past.at[1, :, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        def find_init_jacobi(dx, args):
            alpha, P_past = args
            P_past = P_past.at[0, dx, :].set(
                jacobi_poly_single(r_jacobi, 0, alpha + dx, beta=dx)
            )
            P_past = P_past.at[1, dx, :].set(
                jacobi_poly_single(r_jacobi, 1, alpha + dx, beta=dx)
            )
            return (alpha, P_past)

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, dr + 1, r.size))
        _, P_past = fori_loop(0, dr + 1, find_init_jacobi, (alpha, P_past))

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    r = jnp.atleast_1d(r)
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = int(dr)

    out = jnp.zeros((r.size, m.size))
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)
    dxs = jnp.arange(0, dr + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jit, static_argnums=3)
def zernike_radial_no_duplicate(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes. This version
    is almost the same as zernike_radial_old function but way faster and more
    accurate. First version of this function is zernike_radial_separate which has
    many function for each derivative definition. User can refer that for clarity.


    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    if dr > 4:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]
        P_n1 = P_past[1]
        P_n = jnp.zeros((dr + 1, r.size))

        def find_inter_jacobi(dx, args):
            N, alpha, P_n1, P_n2, P_n = args
            P_n = P_n.at[dx, :].set(
                jacobi_poly_single(r_jacobi, N - dx, alpha + dx, dx, P_n1[dx], P_n2[dx])
            )
            return (N, alpha, P_n1, P_n2, P_n)

        # Calculate Jacobi polynomial and derivatives for (m,n)
        _, _, _, _, P_n = fori_loop(
            0, dr + 1, find_inter_jacobi, (N, alpha, P_n1, P_n2, P_n)
        )

        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )
        # TODO: A version without if statements are possible?
        if dr == 0:
            result = (-1) ** N * r**alpha * P_n[0]
        elif dr == 1:
            result = (-1) ** N * (
                alpha * r ** jnp.maximum(alpha - 1, 0) * P_n[0]
                - coef[1] * 4 * r ** (alpha + 1) * P_n[1]
            )
        elif dr == 2:
            result = (-1) ** N * (
                (alpha - 1) * alpha * r ** jnp.maximum(alpha - 2, 0) * P_n[0]
                - coef[1] * 4 * (2 * alpha + 1) * r**alpha * P_n[1]
                + coef[2] * 16 * r ** (alpha + 2) * P_n[2]
            )
        elif dr == 3:
            result = (-1) ** N * (
                (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 3, 0)
                * P_n[0]
                - coef[1] * 12 * alpha**2 * r ** jnp.maximum(alpha - 1, 0) * P_n[1]
                + coef[2] * 48 * (alpha + 1) * r ** (alpha + 1) * P_n[2]
                - coef[3] * 64 * r ** (alpha + 3) * P_n[3]
            )
        elif dr == 4:
            result = (-1) ** N * (
                (alpha - 3)
                * (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 4, 0)
                * P_n[0]
                - coef[1]
                * 8
                * alpha
                * (2 * alpha**2 - 3 * alpha + 1)
                * r ** jnp.maximum(alpha - 2, 0)
                * P_n[1]
                + coef[2] * 48 * (2 * alpha**2 + 2 * alpha + 1) * r**alpha * P_n[2]
                - coef[3] * 128 * (2 * alpha + 3) * r ** (alpha + 2) * P_n[3]
                + coef[4] * 256 * r ** (alpha + 4) * P_n[4]
            )
        index = jnp.where(
            jnp.logical_and(m == alpha, n == N),
            jnp.arange(1, m.size + 1),
            0,
        )
        idx = jnp.sum(index)
        # needed for proper index
        idx -= 1
        out = out.at[:, idx].set(jnp.where(idx >= 0, result, out.at[:, idx].get()))

        # Shift past values if needed
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask[:, None], P_n1, P_n2)
        P_n1 = jnp.where(mask[:, None], P_n, P_n1)
        P_past = P_past.at[0, :, :].set(P_n2)
        P_past = P_past.at[1, :, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        def find_init_jacobi(dx, args):
            alpha, P_past = args
            P_past = P_past.at[0, dx, :].set(
                jacobi_poly_single(r_jacobi, 0, alpha + dx, beta=dx)
            )
            P_past = P_past.at[1, dx, :].set(
                jacobi_poly_single(r_jacobi, 1, alpha + dx, beta=dx)
            )
            return (alpha, P_past)

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, dr + 1, r.size))
        _, P_past = fori_loop(0, dr + 1, find_init_jacobi, (alpha, P_past))

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    r = jnp.atleast_1d(r)
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = int(dr)

    out = jnp.zeros((r.size, m.size))
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)
    dxs = jnp.arange(0, dr + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jit, static_argnums=3)
def zernike_radial_separate(r, l, m, dr=0):  # noqa: C901
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes. This version
    is almost the same as zernike_radial_old function but way faster and more
    accurate.

    There was even faster version of this code but that doesn't have checks
    for duplicate modes

    # Find the index corresponding to the original array
    # I changed arange function to get rid of 0 as index confusion
    # so if index is full of 0s, there is no such mode
    # (FAST BUT NEED A CHECK FOR DUPLICATE MODES)
    index = jnp.where(
        jnp.logical_and(m == alpha, n == N),
        jnp.arange(1, m.size + 1),
        0,
    )
    idx = jnp.sum(index)
    # needed for proper index
    idx -= 1
    result = (-1) ** N * r**alpha * P_n
    out = out.at[:, idx].set(jnp.where(idx >= 0, result, out.at[:, idx].get()))

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    if dr > 4:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )

    def falseFun(args):
        _, _, out = args
        return out

    def trueFun(args):
        idx, result, out = args
        out = out.at[:, idx].set(result)
        return out

    def update(x, args):
        alpha, N, result, out = args
        idx = jnp.where(jnp.logical_and(m[x] == alpha, n[x] == N), x, -1)
        out = cond(idx >= 0, trueFun, falseFun, (idx, result, out))
        return (alpha, N, result, out)

    # Zeroth Derivative
    def body_inner(N, args):
        alpha, out, P_past = args
        P_n1, P_n2 = P_past
        # Calculate Jacobi polynomial for m,n pair
        P_n = jacobi_poly_single(r_jacobi, N, alpha, 0, P_n1, P_n2)
        result = (-1) ** N * r**alpha * P_n

        # All the checks necessary (FAST BUT NOT THE FASTEST)
        _, _, _, out = fori_loop(0, m.size, update, (alpha, N, result, out))

        P_n2 = jnp.where(N >= 2, P_n1, P_n2)
        P_n1 = jnp.where(N >= 2, P_n, P_n1)
        return (alpha, out, (P_n1, P_n2))

    # First Derivative
    def body_inner_d1(N, args):
        alpha, out, P_past = args
        P_n1, P_n2, dP_n1, dP_n2 = P_past
        # Calculate Jacobi polynomial for m,n pair
        P_n = jacobi_poly_single(r_jacobi, N, alpha, 0, P_n1, P_n2)
        dP_n = jacobi_poly_single(r_jacobi, N - dr, alpha + dr, dr, dP_n1, dP_n2)

        coef = gammaln(alpha + N + 1 + dr) - dr * jnp.log(2) - gammaln(alpha + N + 1)
        coef = jnp.exp(coef)

        result = (-1) ** N * (
            alpha * r ** jnp.maximum(alpha - 1, 0) * P_n
            - coef * 4 * r ** (alpha + 1) * dP_n
        )
        _, _, _, out = fori_loop(0, m.size, update, (alpha, N, result, out))

        P_n2 = jnp.where(N >= 2, P_n1, P_n2)
        P_n1 = jnp.where(N >= 2, P_n, P_n1)
        dP_n2 = jnp.where(N >= 3, dP_n1, dP_n2)
        dP_n1 = jnp.where(N >= 3, dP_n, dP_n1)

        return (alpha, out, (P_n1, P_n2, dP_n1, dP_n2))

    # Second Derivative
    def body_inner_d2(N, args):
        alpha, out, P_past = args
        P_n1, P_n2, dP_n1, dP_n2, ddP_n1, ddP_n2 = P_past
        # Calculate Jacobi polynomial for m,n pair
        P_n = jacobi_poly_single(r_jacobi, N, alpha, 0, P_n1, P_n2)
        dP_n = jacobi_poly_single(r_jacobi, N - 1, alpha + 1, 1, dP_n1, dP_n2)
        ddP_n = jacobi_poly_single(r_jacobi, N - 2, alpha + 2, 2, ddP_n1, ddP_n2)

        coef_1 = gammaln(alpha + N + 2) - jnp.log(2) - gammaln(alpha + N + 1)
        coef_2 = gammaln(alpha + N + 3) - 2 * jnp.log(2) - gammaln(alpha + N + 1)
        coef_1 = jnp.exp(coef_1)
        coef_2 = jnp.exp(coef_2)

        result = (-1) ** N * (
            (alpha - 1) * alpha * r ** jnp.maximum(alpha - 2, 0) * P_n
            - coef_1 * 4 * (2 * alpha + 1) * r**alpha * dP_n
            + coef_2 * 16 * r ** (alpha + 2) * ddP_n
        )
        _, _, _, out = fori_loop(0, m.size, update, (alpha, N, result, out))

        P_n2 = jnp.where(N >= 2, P_n1, P_n2)
        P_n1 = jnp.where(N >= 2, P_n, P_n1)
        dP_n2 = jnp.where(N >= 3, dP_n1, dP_n2)
        dP_n1 = jnp.where(N >= 3, dP_n, dP_n1)
        ddP_n2 = jnp.where(N >= 4, ddP_n1, ddP_n2)
        ddP_n1 = jnp.where(N >= 4, ddP_n, ddP_n1)

        return (
            alpha,
            out,
            (P_n1, P_n2, dP_n1, dP_n2, ddP_n1, ddP_n2),
        )

    # Third Derivative
    def body_inner_d3(N, args):
        alpha, out, P_past = args
        P_n1, P_n2, dP_n1, dP_n2, ddP_n1, ddP_n2, dddP_n1, dddP_n2 = P_past
        # Calculate Jacobi polynomial for m,n pair
        P_n = jacobi_poly_single(r_jacobi, N, alpha, 0, P_n1, P_n2)
        dP_n = jacobi_poly_single(r_jacobi, N - 1, alpha + 1, 1, dP_n1, dP_n2)
        ddP_n = jacobi_poly_single(r_jacobi, N - 2, alpha + 2, 2, ddP_n1, ddP_n2)
        dddP_n = jacobi_poly_single(r_jacobi, N - 3, alpha + 3, 3, dddP_n1, dddP_n2)

        coef_1 = gammaln(alpha + N + 2) - jnp.log(2) - gammaln(alpha + N + 1)
        coef_2 = gammaln(alpha + N + 3) - 2 * jnp.log(2) - gammaln(alpha + N + 1)
        coef_3 = gammaln(alpha + N + 4) - 3 * jnp.log(2) - gammaln(alpha + N + 1)
        coef_1 = jnp.exp(coef_1)
        coef_2 = jnp.exp(coef_2)
        coef_3 = jnp.exp(coef_3)

        result = (-1) ** N * (
            (alpha - 2) * (alpha - 1) * alpha * r ** jnp.maximum(alpha - 3, 0) * P_n
            - coef_1 * 12 * alpha**2 * r ** jnp.maximum(alpha - 1, 0) * dP_n
            + coef_2 * 48 * (alpha + 1) * r ** (alpha + 1) * ddP_n
            - coef_3 * 64 * r ** (alpha + 3) * dddP_n
        )
        _, _, _, out = fori_loop(0, m.size, update, (alpha, N, result, out))

        P_n2 = jnp.where(N >= 2, P_n1, P_n2)
        P_n1 = jnp.where(N >= 2, P_n, P_n1)
        dP_n2 = jnp.where(N >= 3, dP_n1, dP_n2)
        dP_n1 = jnp.where(N >= 3, dP_n, dP_n1)
        ddP_n2 = jnp.where(N >= 4, ddP_n1, ddP_n2)
        ddP_n1 = jnp.where(N >= 4, ddP_n, ddP_n1)
        dddP_n2 = jnp.where(N >= 5, dddP_n1, dddP_n2)
        dddP_n1 = jnp.where(N >= 5, dddP_n, dddP_n1)

        return (
            alpha,
            out,
            (P_n1, P_n2, dP_n1, dP_n2, ddP_n1, ddP_n2, dddP_n1, dddP_n2),
        )

    # Fourth Derivative
    def body_inner_d4(N, args):
        alpha, out, P_past = args
        (
            P_n1,
            P_n2,
            dP_n1,
            dP_n2,
            ddP_n1,
            ddP_n2,
            dddP_n1,
            dddP_n2,
            ddddP_n1,
            ddddP_n2,
        ) = P_past
        # Calculate Jacobi polynomial for m,n pair
        P_n = jacobi_poly_single(r_jacobi, N, alpha, 0, P_n1, P_n2)
        dP_n = jacobi_poly_single(r_jacobi, N - 1, alpha + 1, 1, dP_n1, dP_n2)
        ddP_n = jacobi_poly_single(r_jacobi, N - 2, alpha + 2, 2, ddP_n1, ddP_n2)
        dddP_n = jacobi_poly_single(r_jacobi, N - 3, alpha + 3, 3, dddP_n1, dddP_n2)
        ddddP_n = jacobi_poly_single(r_jacobi, N - 4, alpha + 4, 4, ddddP_n1, ddddP_n2)

        coef_1 = gammaln(alpha + N + 2) - jnp.log(2) - gammaln(alpha + N + 1)
        coef_2 = gammaln(alpha + N + 3) - 2 * jnp.log(2) - gammaln(alpha + N + 1)
        coef_3 = gammaln(alpha + N + 4) - 3 * jnp.log(2) - gammaln(alpha + N + 1)
        coef_4 = gammaln(alpha + N + 5) - 4 * jnp.log(2) - gammaln(alpha + N + 1)
        coef_1 = jnp.exp(coef_1)
        coef_2 = jnp.exp(coef_2)
        coef_3 = jnp.exp(coef_3)
        coef_4 = jnp.exp(coef_4)

        result = (-1) ** N * (
            (alpha - 3)
            * (alpha - 2)
            * (alpha - 1)
            * alpha
            * r ** jnp.maximum(alpha - 4, 0)
            * P_n
            - coef_1
            * 8
            * alpha
            * (2 * alpha**2 - 3 * alpha + 1)
            * r ** jnp.maximum(alpha - 2, 0)
            * dP_n
            + coef_2 * 48 * (2 * alpha**2 + 2 * alpha + 1) * r**alpha * ddP_n
            - coef_3 * 128 * (2 * alpha + 3) * r ** (alpha + 2) * dddP_n
            + coef_4 * 256 * r ** (alpha + 4) * ddddP_n
        )
        _, _, _, out = fori_loop(0, m.size, update, (alpha, N, result, out))

        P_n2 = jnp.where(N >= 2, P_n1, P_n2)
        P_n1 = jnp.where(N >= 2, P_n, P_n1)
        dP_n2 = jnp.where(N >= 3, dP_n1, dP_n2)
        dP_n1 = jnp.where(N >= 3, dP_n, dP_n1)
        ddP_n2 = jnp.where(N >= 4, ddP_n1, ddP_n2)
        ddP_n1 = jnp.where(N >= 4, ddP_n, ddP_n1)
        dddP_n2 = jnp.where(N >= 5, dddP_n1, dddP_n2)
        dddP_n1 = jnp.where(N >= 5, dddP_n, dddP_n1)
        ddddP_n2 = jnp.where(N >= 6, ddddP_n1, ddddP_n2)
        ddddP_n1 = jnp.where(N >= 6, ddddP_n, ddddP_n1)

        return (
            alpha,
            out,
            (
                P_n1,
                P_n2,
                dP_n1,
                dP_n2,
                ddP_n1,
                ddP_n2,
                dddP_n1,
                dddP_n2,
                ddddP_n1,
                ddddP_n2,
            ),
        )

    def body(alpha, args):
        out = args
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # First 2 Jacobi Polynomials (they don't need recursion)
        P_n2 = jacobi_poly_single(r_jacobi, 0, alpha, beta=0)
        P_n1 = jacobi_poly_single(r_jacobi, 1, alpha, beta=0)
        P_past = (
            P_n1,
            P_n2,
        )
        if dr == 0:
            # Loop over every n value
            _, out, _ = fori_loop(
                0,
                (N_max + 1).astype(int),
                body_inner,
                (alpha, out, P_past),
            )
        if dr >= 1:
            dP_n2 = jacobi_poly_single(r_jacobi, 0, alpha + 1, beta=1)
            dP_n1 = jacobi_poly_single(r_jacobi, 1, alpha + 1, beta=1)
            P_past += (
                dP_n1,
                dP_n2,
            )
            if dr == 1:
                _, out, _ = fori_loop(
                    0,
                    (N_max + 1).astype(int),
                    body_inner_d1,
                    (alpha, out, P_past),
                )
        if dr >= 2:
            ddP_n2 = jacobi_poly_single(r_jacobi, 0, alpha + 2, beta=2)
            ddP_n1 = jacobi_poly_single(r_jacobi, 1, alpha + 2, beta=2)
            P_past += (
                ddP_n1,
                ddP_n2,
            )
            if dr == 2:
                _, out, _ = fori_loop(
                    0,
                    (N_max + 1).astype(int),
                    body_inner_d2,
                    (alpha, out, P_past),
                )
        if dr >= 3:
            dddP_n2 = jacobi_poly_single(r_jacobi, 0, alpha + 3, beta=3)
            dddP_n1 = jacobi_poly_single(r_jacobi, 1, alpha + 3, beta=3)
            P_past += (
                dddP_n1,
                dddP_n2,
            )
            if dr == 3:
                _, out, _ = fori_loop(
                    0,
                    (N_max + 1).astype(int),
                    body_inner_d3,
                    (alpha, out, P_past),
                )
        if dr == 4:
            ddddP_n2 = jacobi_poly_single(r_jacobi, 0, alpha + 4, beta=4)
            ddddP_n1 = jacobi_poly_single(r_jacobi, 1, alpha + 4, beta=4)
            P_past += (
                ddddP_n1,
                ddddP_n2,
            )
            _, out, _ = fori_loop(
                0,
                (N_max + 1).astype(int),
                body_inner_d4,
                (alpha, out, P_past),
            )

        return out

    r = jnp.atleast_1d(r)
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)

    out = jnp.zeros((r.size, m.size))
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))

    return out


@custom_jvp
@jit
def zernike_radial_rory(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes. This version
    is almost the same as zernike_radial_old function but way faster and more
    accurate. First version of this function is zernike_radial_separate which has
    many function for each derivative definition. User can refer that for clarity.

    There was even faster version of this code but that doesn't have checks
    for duplicate modes

    # Find the index corresponding to the original array
    # I changed arange function to get rid of 0 as index confusion
    # so if index is full of 0s, there is no such mode
    # (FAST BUT NEED A CHECK FOR DUPLICATE MODES)
    index = jnp.where(
        jnp.logical_and(m == alpha, n == N),
        jnp.arange(1, m.size + 1),
        0,
    )
    idx = jnp.sum(index)
    # needed for proper index
    idx -= 1
    result = (-1) ** N * r**alpha * P_n
    out = out.at[:, idx].set(jnp.where(idx >= 0, result, out.at[:, idx].get()))

    Above part replaces the matrix update conducted by following code,

    _, _, _, out = fori_loop(0, m.size, update, (alpha, N, result, out))

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    return _zernike_radial_vectorized_rory(r, l, m, dr)


@functools.partial(jit, static_argnums=3)
def zernike_radial(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes. This version
    is almost the same as zernike_radial_old function but way faster and more
    accurate. First version of this function is zernike_radial_separate which has
    many function for each derivative definition. User can refer that for clarity.

    There was even faster version of this code but that doesn't have checks
    for duplicate modes

    # Find the index corresponding to the original array
    # I changed arange function to get rid of 0 as index confusion
    # so if index is full of 0s, there is no such mode
    # (FAST BUT NEED A CHECK FOR DUPLICATE MODES)
    index = jnp.where(
        jnp.logical_and(m == alpha, n == N),
        jnp.arange(1, m.size + 1),
        0,
    )
    idx = jnp.sum(index)
    # needed for proper index
    idx -= 1
    result = (-1) ** N * r**alpha * P_n
    out = out.at[:, idx].set(jnp.where(idx >= 0, result, out.at[:, idx].get()))

    Above part replaces the matrix update conducted by following code,

    _, _, _, out = fori_loop(0, m.size, update, (alpha, N, result, out))

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    if dr > 4:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )

    def update(x, args):
        alpha, N, result, out = args
        idx = jnp.where(jnp.logical_and(m[x] == alpha, n[x] == N), x, -1)

        def falseFun(args):
            _, _, out = args
            return out

        def trueFun(args):
            idx, result, out = args
            out = out.at[:, idx].set(result)
            return out

        out = cond(idx >= 0, trueFun, falseFun, (idx, result, out))
        return (alpha, N, result, out)

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]
        P_n1 = P_past[1]
        P_n = jnp.zeros((dr + 1, r.size))

        def find_inter_jacobi(dx, args):
            N, alpha, P_n1, P_n2, P_n = args
            P_n = P_n.at[dx, :].set(
                jacobi_poly_single(r_jacobi, N - dx, alpha + dx, dx, P_n1[dx], P_n2[dx])
            )
            return (N, alpha, P_n1, P_n2, P_n)

        # Calculate Jacobi polynomial and derivatives for (m,n)
        _, _, _, _, P_n = fori_loop(
            0, dr + 1, find_inter_jacobi, (N, alpha, P_n1, P_n2, P_n)
        )

        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )
        # TODO: A version without if statements are possible?
        if dr == 0:
            result = (-1) ** N * r**alpha * P_n[0]
        elif dr == 1:
            result = (-1) ** N * (
                alpha * r ** jnp.maximum(alpha - 1, 0) * P_n[0]
                - coef[1] * 4 * r ** (alpha + 1) * P_n[1]
            )
        elif dr == 2:
            result = (-1) ** N * (
                (alpha - 1) * alpha * r ** jnp.maximum(alpha - 2, 0) * P_n[0]
                - coef[1] * 4 * (2 * alpha + 1) * r**alpha * P_n[1]
                + coef[2] * 16 * r ** (alpha + 2) * P_n[2]
            )
        elif dr == 3:
            result = (-1) ** N * (
                (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 3, 0)
                * P_n[0]
                - coef[1] * 12 * alpha**2 * r ** jnp.maximum(alpha - 1, 0) * P_n[1]
                + coef[2] * 48 * (alpha + 1) * r ** (alpha + 1) * P_n[2]
                - coef[3] * 64 * r ** (alpha + 3) * P_n[3]
            )
        elif dr == 4:
            result = (-1) ** N * (
                (alpha - 3)
                * (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 4, 0)
                * P_n[0]
                - coef[1]
                * 8
                * alpha
                * (2 * alpha**2 - 3 * alpha + 1)
                * r ** jnp.maximum(alpha - 2, 0)
                * P_n[1]
                + coef[2] * 48 * (2 * alpha**2 + 2 * alpha + 1) * r**alpha * P_n[2]
                - coef[3] * 128 * (2 * alpha + 3) * r ** (alpha + 2) * P_n[3]
                + coef[4] * 256 * r ** (alpha + 4) * P_n[4]
            )
        _, _, _, out = fori_loop(0, m.size, update, (alpha, N, result, out))

        # Shift past values if needed
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask[:, None], P_n1, P_n2)
        P_n1 = jnp.where(mask[:, None], P_n, P_n1)
        P_past = P_past.at[0, :, :].set(P_n2)
        P_past = P_past.at[1, :, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        def find_init_jacobi(dx, args):
            alpha, P_past = args
            P_past = P_past.at[0, dx, :].set(
                jacobi_poly_single(r_jacobi, 0, alpha + dx, beta=dx)
            )
            P_past = P_past.at[1, dx, :].set(
                jacobi_poly_single(r_jacobi, 1, alpha + dx, beta=dx)
            )
            return (alpha, P_past)

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, dr + 1, r.size))
        _, P_past = fori_loop(0, dr + 1, find_init_jacobi, (alpha, P_past))

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    r = jnp.atleast_1d(r)
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = int(dr)

    out = jnp.zeros((r.size, m.size))
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)
    dxs = jnp.arange(0, dr + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jit, static_argnums=3)
def zernike_radial_old_desc(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Evaluates basis functions using JAX and a stable
    evaluation scheme based on jacobi polynomials and
    binomial coefficients. Generally faster for L>24
    and differentiable, but slower for low resolution.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    m = jnp.abs(m)
    alpha = m
    beta = 0
    n = (l - m) // 2
    s = (-1) ** n
    jacobi_arg = 1 - 2 * r**2
    if dr == 0:
        out = r**m * _jacobi(n, alpha, beta, jacobi_arg, 0)
    elif dr == 1:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        out = m * r ** jnp.maximum(m - 1, 0) * f - 4 * r ** (m + 1) * df
    elif dr == 2:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        out = (
            (m - 1) * m * r ** jnp.maximum(m - 2, 0) * f
            - 4 * (2 * m + 1) * r**m * df
            + 16 * r ** (m + 2) * d2f
        )
    elif dr == 3:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        d3f = _jacobi(n, alpha, beta, jacobi_arg, 3)
        out = (
            (m - 2) * (m - 1) * m * r ** jnp.maximum(m - 3, 0) * f
            - 12 * m**2 * r ** jnp.maximum(m - 1, 0) * df
            + 48 * (m + 1) * r ** (m + 1) * d2f
            - 64 * r ** (m + 3) * d3f
        )
    elif dr == 4:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        d3f = _jacobi(n, alpha, beta, jacobi_arg, 3)
        d4f = _jacobi(n, alpha, beta, jacobi_arg, 4)
        out = (
            (m - 3) * (m - 2) * (m - 1) * m * r ** jnp.maximum(m - 4, 0) * f
            - 8 * m * (2 * m**2 - 3 * m + 1) * r ** jnp.maximum(m - 2, 0) * df
            + 48 * (2 * m**2 + 2 * m + 1) * r**m * d2f
            - 128 * (2 * m + 3) * r ** (m + 2) * d3f
            + 256 * r ** (m + 4) * d4f
        )
    else:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )
    return s * jnp.where((l - m) % 2 == 0, out, 0)


@custom_jvp
@jit
def zernike_radial_if_switch(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes.
    https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations

    For the derivatives, the following formula is used with above recursion relation,
    https://en.wikipedia.org/wiki/Jacobi_polynomials#Derivatives

    Used formulas are also in the zerike_eval.ipynb notebook in docs.

    This function can be made faster. However, JAX reverse mode AD causes problems.
    In future, we may use vmap() instead of jnp.vectorize() to be able to set dr as
    static argument, and not calculate every derivative even thoguh not asked.

    Parameters
    ----------
    r : ndarray, shape(N,) or scalar
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,) or integer
        radial mode number(s)
    m : ndarray of int, shape(K,) or integer
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    dr = jnp.asarray(dr).astype(int)

    def ZeroOne(args):
        def Zero(args):
            (r, l, m, dr) = args
            return _zernike_radial_vectorized(r, l, m, dr)

        def One(args):
            (r, l, m, dr) = args
            return _zernike_radial_vectorized_d1(r, l, m, dr)

        return cond(dr == 0, Zero, One, (r, l, m, dr))

    def Rest(args):
        def Two(args):
            (r, l, m, dr) = args
            return _zernike_radial_vectorized_d2(r, l, m, dr)

        def ThreeFour(args):
            def Three(args):
                (r, l, m, dr) = args
                return _zernike_radial_vectorized_d3(r, l, m, dr)

            def Four(args):
                (r, l, m, dr) = args
                return _zernike_radial_vectorized_d4(r, l, m, dr)

            return cond(dr == 3, Three, Four, (r, l, m, dr))

        return cond(dr == 2, Two, ThreeFour, (r, l, m, dr))

    # Switch doesn't work. Under JIT, only viable option seems this conditional
    return cond(dr < 2, ZeroOne, Rest, (r, l, m, dr))


@custom_jvp
@jit
def zernike_radial_switch(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes.
    https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations

    For the derivatives, the following formula is used with above recursion relation,
    https://en.wikipedia.org/wiki/Jacobi_polynomials#Derivatives

    Used formulas are also in the zerike_eval.ipynb notebook in docs.

    This function can be made faster. However, JAX reverse mode AD causes problems.
    In future, we may use vmap() instead of jnp.vectorize() to be able to set dr as
    static argument, and not calculate every derivative even thoguh not asked.

    Parameters
    ----------
    r : ndarray, shape(N,) or scalar
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,) or integer
        radial mode number(s)
    m : ndarray of int, shape(K,) or integer
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    dr = jnp.asarray(dr).astype(int)

    branches = [
        _zernike_radial_vectorized,
        _zernike_radial_vectorized_d1,
        _zernike_radial_vectorized_d2,
        _zernike_radial_vectorized_d3,
        _zernike_radial_vectorized_d4,
    ]
    return switch(dr, branches, r, l, m, dr)


@custom_jvp
@functools.partial(jit, static_argnums=3)
def zernike_radial_jvp(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes. This version
    is almost the same as zernike_radial_old function but way faster and more
    accurate. First version of this function is zernike_radial_separate which has
    many function for each derivative definition. User can refer that for clarity.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    if dr > 4:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )

    def update(x, args):
        alpha, N, result, out = args
        idx = jnp.where(jnp.logical_and(m[x] == alpha, n[x] == N), x, -1)

        def falseFun(args):
            _, _, out = args
            return out

        def trueFun(args):
            idx, result, out = args
            out = out.at[:, idx].set(result)
            return out

        out = cond(idx >= 0, trueFun, falseFun, (idx, result, out))
        return (alpha, N, result, out)

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]
        P_n1 = P_past[1]
        P_n = jnp.zeros((dr + 1, r.size))

        def find_inter_jacobi(dx, args):
            N, alpha, P_n1, P_n2, P_n = args
            P_n = P_n.at[dx, :].set(
                jacobi_poly_single(r_jacobi, N - dx, alpha + dx, dx, P_n1[dx], P_n2[dx])
            )
            return (N, alpha, P_n1, P_n2, P_n)

        # Calculate Jacobi polynomial and derivatives for (m,n)
        _, _, _, _, P_n = fori_loop(
            0, dr + 1, find_inter_jacobi, (N, alpha, P_n1, P_n2, P_n)
        )

        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )
        # TODO: A version without if statements are possible?
        if dr == 0:
            result = (-1) ** N * r**alpha * P_n[0]
        elif dr == 1:
            result = (-1) ** N * (
                alpha * r ** jnp.maximum(alpha - 1, 0) * P_n[0]
                - coef[1] * 4 * r ** (alpha + 1) * P_n[1]
            )
        elif dr == 2:
            result = (-1) ** N * (
                (alpha - 1) * alpha * r ** jnp.maximum(alpha - 2, 0) * P_n[0]
                - coef[1] * 4 * (2 * alpha + 1) * r**alpha * P_n[1]
                + coef[2] * 16 * r ** (alpha + 2) * P_n[2]
            )
        elif dr == 3:
            result = (-1) ** N * (
                (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 3, 0)
                * P_n[0]
                - coef[1] * 12 * alpha**2 * r ** jnp.maximum(alpha - 1, 0) * P_n[1]
                + coef[2] * 48 * (alpha + 1) * r ** (alpha + 1) * P_n[2]
                - coef[3] * 64 * r ** (alpha + 3) * P_n[3]
            )
        elif dr == 4:
            result = (-1) ** N * (
                (alpha - 3)
                * (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 4, 0)
                * P_n[0]
                - coef[1]
                * 8
                * alpha
                * (2 * alpha**2 - 3 * alpha + 1)
                * r ** jnp.maximum(alpha - 2, 0)
                * P_n[1]
                + coef[2] * 48 * (2 * alpha**2 + 2 * alpha + 1) * r**alpha * P_n[2]
                - coef[3] * 128 * (2 * alpha + 3) * r ** (alpha + 2) * P_n[3]
                + coef[4] * 256 * r ** (alpha + 4) * P_n[4]
            )
        _, _, _, out = fori_loop(0, m.size, update, (alpha, N, result, out))

        # Shift past values if needed
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask[:, None], P_n1, P_n2)
        P_n1 = jnp.where(mask[:, None], P_n, P_n1)
        P_past = P_past.at[0, :, :].set(P_n2)
        P_past = P_past.at[1, :, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        def find_init_jacobi(dx, args):
            alpha, P_past = args
            P_past = P_past.at[0, dx, :].set(
                jacobi_poly_single(r_jacobi, 0, alpha + dx, beta=dx)
            )
            P_past = P_past.at[1, dx, :].set(
                jacobi_poly_single(r_jacobi, 1, alpha + dx, beta=dx)
            )
            return (alpha, P_past)

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, dr + 1, r.size))
        _, P_past = fori_loop(0, dr + 1, find_init_jacobi, (alpha, P_past))

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    r = jnp.atleast_1d(r)
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = int(dr)

    out = jnp.zeros((r.size, m.size))
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)
    dxs = jnp.arange(0, dr + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@custom_jvp
@functools.partial(jit, static_argnums=[3,4])
def zernike_radial_jvp_gpu(r, l, m, dr=0, repeat=1):
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes. This version
    is almost the same as zernike_radial_old function but way faster and more
    accurate. First version of this function is zernike_radial_separate which has
    many function for each derivative definition. User can refer that for clarity.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    if dr > 4:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )
    def update(x, args):
        index, result, out = args
        idx = index[x]
        out = out.at[:, idx].set(result[:, None])
        return (index, result, out)
    
    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]
        P_n1 = P_past[1]
        P_n = jnp.zeros((dr + 1, r.size))

        def find_inter_jacobi(dx, args):
            N, alpha, P_n1, P_n2, P_n = args
            P_n = P_n.at[dx, :].set(
                jacobi_poly_single(r_jacobi, N - dx, alpha + dx, dx, P_n1[dx], P_n2[dx])
            )
            return (N, alpha, P_n1, P_n2, P_n)

        # Calculate Jacobi polynomial and derivatives for (m,n)
        _, _, _, _, P_n = fori_loop(
            0, dr + 1, find_inter_jacobi, (N, alpha, P_n1, P_n2, P_n)
        )

        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )
        # TODO: A version without if statements are possible?
        if dr == 0:
            result = (-1) ** N * r**alpha * P_n[0]
        elif dr == 1:
            result = (-1) ** N * (
                alpha * r ** jnp.maximum(alpha - 1, 0) * P_n[0]
                - coef[1] * 4 * r ** (alpha + 1) * P_n[1]
            )
        elif dr == 2:
            result = (-1) ** N * (
                (alpha - 1) * alpha * r ** jnp.maximum(alpha - 2, 0) * P_n[0]
                - coef[1] * 4 * (2 * alpha + 1) * r**alpha * P_n[1]
                + coef[2] * 16 * r ** (alpha + 2) * P_n[2]
            )
        elif dr == 3:
            result = (-1) ** N * (
                (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 3, 0)
                * P_n[0]
                - coef[1] * 12 * alpha**2 * r ** jnp.maximum(alpha - 1, 0) * P_n[1]
                + coef[2] * 48 * (alpha + 1) * r ** (alpha + 1) * P_n[2]
                - coef[3] * 64 * r ** (alpha + 3) * P_n[3]
            )
        elif dr == 4:
            result = (-1) ** N * (
                (alpha - 3)
                * (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 4, 0)
                * P_n[0]
                - coef[1]
                * 8
                * alpha
                * (2 * alpha**2 - 3 * alpha + 1)
                * r ** jnp.maximum(alpha - 2, 0)
                * P_n[1]
                + coef[2] * 48 * (2 * alpha**2 + 2 * alpha + 1) * r**alpha * P_n[2]
                - coef[3] * 128 * (2 * alpha + 3) * r ** (alpha + 2) * P_n[3]
                + coef[4] * 256 * r ** (alpha + 4) * P_n[4]
            )
        index = jnp.argwhere(jnp.logical_and(m == alpha, n == N),  size=repeat)
        _, _, out = fori_loop(0, repeat, update, (index, result, out))
        
        # Shift past values if needed
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask[:, None], P_n1, P_n2)
        P_n1 = jnp.where(mask[:, None], P_n, P_n1)
        P_past = P_past.at[0, :, :].set(P_n2)
        P_past = P_past.at[1, :, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        def find_init_jacobi(dx, args):
            alpha, P_past = args
            P_past = P_past.at[0, dx, :].set(
                jacobi_poly_single(r_jacobi, 0, alpha + dx, beta=dx)
            )
            P_past = P_past.at[1, dx, :].set(
                jacobi_poly_single(r_jacobi, 1, alpha + dx, beta=dx)
            )
            return (alpha, P_past)

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, dr + 1, r.size))
        _, P_past = fori_loop(0, dr + 1, find_init_jacobi, (alpha, P_past))

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    r = jnp.atleast_1d(r)
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = int(dr)

    out = jnp.zeros((r.size, m.size))
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)
    dxs = jnp.arange(0, dr + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@custom_jvp
@jit
def zernike_radial_switch_gpu(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes.
    https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations

    For the derivatives, the following formula is used with above recursion relation,
    https://en.wikipedia.org/wiki/Jacobi_polynomials#Derivatives

    Used formulas are also in the zerike_eval.ipynb notebook in docs.

    This function can be made faster. However, JAX reverse mode AD causes problems.
    In future, we may use vmap() instead of jnp.vectorize() to be able to set dr as
    static argument, and not calculate every derivative even thoguh not asked.

    Parameters
    ----------
    r : ndarray, shape(N,) or scalar
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,) or integer
        radial mode number(s)
    m : ndarray of int, shape(K,) or integer
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    dr = jnp.asarray(dr).astype(int)

    branches = [
        _zernike_radial_vectorized_gpu,
        _zernike_radial_vectorized_d1_gpu,
        _zernike_radial_vectorized_d2_gpu,
        _zernike_radial_vectorized_d3_gpu,
        _zernike_radial_vectorized_d4_gpu,
    ]
    return switch(dr, branches, r, l, m, dr)


def find_intermadiate_jacobi(dx, args):
    """Finds Jacobi function and its derivatives for nth loop."""
    r_jacobi, N, alpha, P_n1, P_n2, P_n = args
    P_n = P_n.at[dx].set(
        jacobi_poly_single(r_jacobi, N - dx, alpha + dx, dx, P_n1[dx], P_n2[dx])
    )
    return (r_jacobi, N, alpha, P_n1, P_n2, P_n)


def update_zernike_output(i, args):
    """Updates Zernike radial output, if the mode is in the inputs."""
    m, n, alpha, N, result, out = args
    idx = jnp.where(jnp.logical_and(m[i] == alpha, n[i] == N), i, -1)

    def falseFun(args):
        _, _, out = args
        return out

    def trueFun(args):
        idx, result, out = args
        out = out.at[idx].set(result)
        return out

    out = cond(idx >= 0, trueFun, falseFun, (idx, result, out))
    return (m, n, alpha, N, result, out)


def find_initial_jacobi(dx, args):
    """Finds initial values of Jacobi Polynomial and derivatives."""
    r_jacobi, alpha, P_past = args
    # Jacobi for n=0
    P_past = P_past.at[0, dx].set(jacobi_poly_single(r_jacobi, 0, alpha + dx, beta=dx))
    # Jacobi for n=1
    P_past = P_past.at[1, dx].set(jacobi_poly_single(r_jacobi, 1, alpha + dx, beta=dx))
    return (r_jacobi, alpha, P_past)


@functools.partial(jnp.vectorize, excluded=(1, 2, 3), signature="()->(k)")
def _zernike_radial_vectorized(r, l, m, dr):
    """Calculation of Radial part of Zernike polynomials."""

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]  # Jacobi at N-2
        P_n1 = P_past[1]  # Jacobi at N-1
        P_n = jacobi_poly_single(r_jacobi, N, alpha, 0, P_n1, P_n2)

        # Only calculate the function at dr th index with input r
        result = (-1) ** N * r**alpha * P_n
        # Check if the calculated values is in the given modes
        _, _, _, _, _, out = fori_loop(
            0, m.size, update_zernike_output, (m, n, alpha, N, result, out)
        )

        # Shift past values if needed
        # For derivative order dx, if N is smaller than 2+dx, then only the initial
        # value calculated by find_init_jacobi function will be used. So, if you update
        # P_n's, preceeding values will be wrong.
        mask = N >= 2
        P_n2 = jnp.where(mask, P_n1, P_n2)
        P_n1 = jnp.where(mask, P_n, P_n1)
        # Form updated P_past matrix
        P_past = P_past.at[0].set(P_n2)
        P_past = P_past.at[1].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros(2)
        P_past = P_past.at[0].set(jacobi_poly_single(r_jacobi, 0, alpha, beta=0))
        # Jacobi for n=1
        P_past = P_past.at[1].set(jacobi_poly_single(r_jacobi, 1, alpha, beta=0))

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    # Make inputs 1D arrays in case they aren't
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)

    # From the vectorization, the overall output will be (r.size, m.size)
    out = jnp.zeros(m.size)
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jnp.vectorize, excluded=(1, 2, 3), signature="()->(k)")
def _zernike_radial_vectorized_d1(r, l, m, dr):
    """First derivative calculation of Radial part of Zernike polynomials."""

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]  # Jacobi at N-2
        P_n1 = P_past[1]  # Jacobi at N-1
        P_n = jnp.zeros(MAXDR + 1)  # Jacobi at N

        # Calculate Jacobi polynomial and derivatives for (alpha,N)
        _, _, _, _, _, P_n = fori_loop(
            0,
            MAXDR + 1,
            find_intermadiate_jacobi,
            (r_jacobi, N, alpha, P_n1, P_n2, P_n),
        )
        # Calculate coefficients for derivatives. coef[0] will never be used. Jax
        # doesn't have Gamma function directly, that's why we calculate Logarithm of
        # Gamma function and then exponentiate it.
        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )
        # 1th Derivative of Zernike Radial
        result = (-1) ** N * (
            alpha * r ** jnp.maximum(alpha - 1, 0) * P_n[0]
            - coef[1] * 4 * r ** (alpha + 1) * P_n[1]
        )
        # Check if the calculated values is in the given modes
        _, _, _, _, _, out = fori_loop(
            0, m.size, update_zernike_output, (m, n, alpha, N, result, out)
        )

        # Shift past values if needed
        # For derivative order dx, if N is smaller than 2+dx, then only the initial
        # value calculated by find_init_jacobi function will be used. So, if you update
        # P_n's, preceeding values will be wrong.
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask, P_n1, P_n2)
        P_n1 = jnp.where(mask, P_n, P_n1)
        # Form updated P_past matrix
        P_past = P_past.at[0, :].set(P_n2)
        P_past = P_past.at[1, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, MAXDR + 1))
        _, _, P_past = fori_loop(
            0, MAXDR + 1, find_initial_jacobi, (r_jacobi, alpha, P_past)
        )

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    # Make inputs 1D arrays in case they aren't
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = jnp.asarray(dr).astype(int)

    # From the vectorization, the overall output will be (r.size, m.size)
    out = jnp.zeros(m.size)
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    # This part can be better implemented. Try to make dr as static argument
    # jnp.vectorize doesn't allow it to be static
    MAXDR = 1
    dxs = jnp.arange(0, MAXDR + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jnp.vectorize, excluded=(1, 2, 3), signature="()->(k)")
def _zernike_radial_vectorized_d2(r, l, m, dr):
    """Second derivative calculation of Radial part of Zernike polynomials."""

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]  # Jacobi at N-2
        P_n1 = P_past[1]  # Jacobi at N-1
        P_n = jnp.zeros(MAXDR + 1)  # Jacobi at N

        # Calculate Jacobi polynomial and derivatives for (alpha,N)
        _, _, _, _, _, P_n = fori_loop(
            0,
            MAXDR + 1,
            find_intermadiate_jacobi,
            (r_jacobi, N, alpha, P_n1, P_n2, P_n),
        )

        # Calculate coefficients for derivatives. coef[0] will never be used. Jax
        # doesn't have Gamma function directly, that's why we calculate Logarithm of
        # Gamma function and then exponentiate it.
        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )

        result = (-1) ** N * (
            (alpha - 1) * alpha * r ** jnp.maximum(alpha - 2, 0) * P_n[0]
            - coef[1] * 4 * (2 * alpha + 1) * r**alpha * P_n[1]
            + coef[2] * 16 * r ** (alpha + 2) * P_n[2]
        )
        # Check if the calculated values is in the given modes
        _, _, _, _, _, out = fori_loop(
            0, m.size, update_zernike_output, (m, n, alpha, N, result, out)
        )

        # Shift past values if needed
        # For derivative order dx, if N is smaller than 2+dx, then only the initial
        # value calculated by find_init_jacobi function will be used. So, if you update
        # P_n's, preceeding values will be wrong.
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask, P_n1, P_n2)
        P_n1 = jnp.where(mask, P_n, P_n1)
        # Form updated P_past matrix
        P_past = P_past.at[0, :].set(P_n2)
        P_past = P_past.at[1, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, MAXDR + 1))
        _, _, P_past = fori_loop(
            0, MAXDR + 1, find_initial_jacobi, (r_jacobi, alpha, P_past)
        )

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    # Make inputs 1D arrays in case they aren't
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = jnp.asarray(dr).astype(int)

    # From the vectorization, the overall output will be (r.size, m.size)
    out = jnp.zeros(m.size)
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    # This part can be better implemented. Try to make dr as static argument
    # jnp.vectorize doesn't allow it to be static
    MAXDR = 2
    dxs = jnp.arange(0, MAXDR + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jnp.vectorize, excluded=(1, 2, 3), signature="()->(k)")
def _zernike_radial_vectorized_d3(r, l, m, dr):
    """Third derivative calculation of Radial part of Zernike polynomials."""

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]  # Jacobi at N-2
        P_n1 = P_past[1]  # Jacobi at N-1
        P_n = jnp.zeros(MAXDR + 1)  # Jacobi at N

        # Calculate Jacobi polynomial and derivatives for (alpha,N)
        _, _, _, _, _, P_n = fori_loop(
            0,
            MAXDR + 1,
            find_intermadiate_jacobi,
            (r_jacobi, N, alpha, P_n1, P_n2, P_n),
        )

        # Calculate coefficients for derivatives. coef[0] will never be used. Jax
        # doesn't have Gamma function directly, that's why we calculate Logarithm of
        # Gamma function and then exponentiate it.
        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )

        # 3rd Derivative of Zernike Radial
        result = (-1) ** N * (
            (alpha - 2) * (alpha - 1) * alpha * r ** jnp.maximum(alpha - 3, 0) * P_n[0]
            - coef[1] * 12 * alpha**2 * r ** jnp.maximum(alpha - 1, 0) * P_n[1]
            + coef[2] * 48 * (alpha + 1) * r ** (alpha + 1) * P_n[2]
            - coef[3] * 64 * r ** (alpha + 3) * P_n[3]
        )
        # Check if the calculated values is in the given modes
        _, _, _, _, _, out = fori_loop(
            0, m.size, update_zernike_output, (m, n, alpha, N, result, out)
        )

        # Shift past values if needed
        # For derivative order dx, if N is smaller than 2+dx, then only the initial
        # value calculated by find_init_jacobi function will be used. So, if you update
        # P_n's, preceeding values will be wrong.
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask, P_n1, P_n2)
        P_n1 = jnp.where(mask, P_n, P_n1)
        # Form updated P_past matrix
        P_past = P_past.at[0, :].set(P_n2)
        P_past = P_past.at[1, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, MAXDR + 1))
        _, _, P_past = fori_loop(
            0, MAXDR + 1, find_initial_jacobi, (r_jacobi, alpha, P_past)
        )

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    # Make inputs 1D arrays in case they aren't
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = jnp.asarray(dr).astype(int)

    # From the vectorization, the overall output will be (r.size, m.size)
    out = jnp.zeros(m.size)
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    # This part can be better implemented. Try to make dr as static argument
    # jnp.vectorize doesn't allow it to be static
    MAXDR = 3
    dxs = jnp.arange(0, MAXDR + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jnp.vectorize, excluded=(1, 2, 3), signature="()->(k)")
def _zernike_radial_vectorized_d4(r, l, m, dr):
    """Fourth derivative calculation of Radial part of Zernike polynomials."""

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]  # Jacobi at N-2
        P_n1 = P_past[1]  # Jacobi at N-1
        P_n = jnp.zeros(MAXDR + 1)  # Jacobi at N

        # Calculate Jacobi polynomial and derivatives for (alpha,N)
        _, _, _, _, _, P_n = fori_loop(
            0,
            MAXDR + 1,
            find_intermadiate_jacobi,
            (r_jacobi, N, alpha, P_n1, P_n2, P_n),
        )

        # Calculate coefficients for derivatives. coef[0] will never be used. Jax
        # doesn't have Gamma function directly, that's why we calculate Logarithm of
        # Gamma function and then exponentiate it.
        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )

        # 4th Derivative of Zernike Radial
        result = (-1) ** N * (
            (alpha - 3)
            * (alpha - 2)
            * (alpha - 1)
            * alpha
            * r ** jnp.maximum(alpha - 4, 0)
            * P_n[0]
            - coef[1]
            * 8
            * alpha
            * (2 * alpha**2 - 3 * alpha + 1)
            * r ** jnp.maximum(alpha - 2, 0)
            * P_n[1]
            + coef[2] * 48 * (2 * alpha**2 + 2 * alpha + 1) * r**alpha * P_n[2]
            - coef[3] * 128 * (2 * alpha + 3) * r ** (alpha + 2) * P_n[3]
            + coef[4] * 256 * r ** (alpha + 4) * P_n[4]
        )
        # Check if the calculated values is in the given modes
        _, _, _, _, _, out = fori_loop(
            0, m.size, update_zernike_output, (m, n, alpha, N, result, out)
        )

        # Shift past values if needed
        # For derivative order dx, if N is smaller than 2+dx, then only the initial
        # value calculated by find_init_jacobi function will be used. So, if you update
        # P_n's, preceeding values will be wrong.
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask, P_n1, P_n2)
        P_n1 = jnp.where(mask, P_n, P_n1)
        # Form updated P_past matrix
        P_past = P_past.at[0, :].set(P_n2)
        P_past = P_past.at[1, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, MAXDR + 1))
        _, _, P_past = fori_loop(
            0, MAXDR + 1, find_initial_jacobi, (r_jacobi, alpha, P_past)
        )

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    # Make inputs 1D arrays in case they aren't
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = jnp.asarray(dr).astype(int)

    # From the vectorization, the overall output will be (r.size, m.size)
    out = jnp.zeros(m.size)
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    # This part can be better implemented. Try to make dr as static argument
    # jnp.vectorize doesn't allow it to be static
    MAXDR = 4
    dxs = jnp.arange(0, MAXDR + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jnp.vectorize, excluded=(1, 2, 3), signature="()->(k)")
def _zernike_radial_vectorized_rory(r, l, m, dr):
    """Calculation of Radial part of Zernike polynomials."""

    def update(i, args):
        alpha, N, result, out = args
        idx = jnp.where(jnp.logical_and(m[i] == alpha, n[i] == N), i, -1)

        def falseFun(args):
            _, _, out = args
            return out

        def trueFun(args):
            idx, result, out = args
            out = out.at[idx].set(result)
            return out

        out = cond(idx >= 0, trueFun, falseFun, (idx, result, out))
        return (alpha, N, result, out)

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]  # Jacobi at N-2
        P_n1 = P_past[1]  # Jacobi at N-1
        P_n = jnp.zeros(MAXDR + 1)  # Jacobi at N

        def find_inter_jacobi(dx, args):
            N, alpha, P_n1, P_n2, P_n = args
            P_n = P_n.at[dx].set(
                jacobi_poly_single(r_jacobi, N - dx, alpha + dx, dx, P_n1[dx], P_n2[dx])
            )
            return (N, alpha, P_n1, P_n2, P_n)

        # Calculate Jacobi polynomial and derivatives for (alpha,N)
        _, _, _, _, P_n = fori_loop(
            0, MAXDR + 1, find_inter_jacobi, (N, alpha, P_n1, P_n2, P_n)
        )

        # Calculate coefficients for derivatives. coef[0] will never be used. Jax
        # doesn't have Gamma function directly, that's why we calculate Logarithm of
        # Gamma function and then exponentiate it.
        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )

        # Since we cannot make dr static, we cannot use if statement. Instead define
        # functions and execute only the function at dr th index

        # 0th Derivative of Zernike Radial
        branch0 = lambda x: (-1) ** N * x**alpha * P_n[0]
        # 1th Derivative of Zernike Radial
        branch1 = lambda x: (-1) ** N * (
            alpha * x ** jnp.maximum(alpha - 1, 0) * P_n[0]
            - coef[1] * 4 * x ** (alpha + 1) * P_n[1]
        )
        # 2nd Derivative of Zernike Radial
        branch2 = lambda x: (-1) ** N * (
            (alpha - 1) * alpha * x ** jnp.maximum(alpha - 2, 0) * P_n[0]
            - coef[1] * 4 * (2 * alpha + 1) * x**alpha * P_n[1]
            + coef[2] * 16 * x ** (alpha + 2) * P_n[2]
        )
        # 3rd Derivative of Zernike Radial
        branch3 = lambda x: (-1) ** N * (
            (alpha - 2) * (alpha - 1) * alpha * x ** jnp.maximum(alpha - 3, 0) * P_n[0]
            - coef[1] * 12 * alpha**2 * x ** jnp.maximum(alpha - 1, 0) * P_n[1]
            + coef[2] * 48 * (alpha + 1) * x ** (alpha + 1) * P_n[2]
            - coef[3] * 64 * x ** (alpha + 3) * P_n[3]
        )
        # 4th Derivative of Zernike Radial
        branch4 = lambda x: (-1) ** N * (
            (alpha - 3)
            * (alpha - 2)
            * (alpha - 1)
            * alpha
            * x ** jnp.maximum(alpha - 4, 0)
            * P_n[0]
            - coef[1]
            * 8
            * alpha
            * (2 * alpha**2 - 3 * alpha + 1)
            * x ** jnp.maximum(alpha - 2, 0)
            * P_n[1]
            + coef[2] * 48 * (2 * alpha**2 + 2 * alpha + 1) * x**alpha * P_n[2]
            - coef[3] * 128 * (2 * alpha + 3) * x ** (alpha + 2) * P_n[3]
            + coef[4] * 256 * x ** (alpha + 4) * P_n[4]
        )
        # if dr is greater than 4, this will be executed
        branch5 = lambda x: jnp.nan
        branches = [branch0, branch1, branch2, branch3, branch4, branch5]
        # Only calculate the function at dr th index with input r
        result = switch(dr, branches, r)
        # Check if the calculated values is in the given modes
        _, _, _, out = fori_loop(0, m.size, update, (alpha, N, result, out))

        # Shift past values if needed
        # For derivative order dx, if N is smaller than 2+dx, then only the initial
        # value calculated by find_init_jacobi function will be used. So, if you update
        # P_n's, preceeding values will be wrong.
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask, P_n1, P_n2)
        P_n1 = jnp.where(mask, P_n, P_n1)
        # Form updated P_past matrix
        P_past = P_past.at[0, :].set(P_n2)
        P_past = P_past.at[1, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # Find initial values of Jacobi Polynomial and derivatives
        def find_init_jacobi(dx, args):
            alpha, P_past = args
            # Jacobi for n=0
            P_past = P_past.at[0, dx].set(
                jacobi_poly_single(r_jacobi, 0, alpha + dx, beta=dx)
            )
            # Jacobi for n=1
            P_past = P_past.at[1, dx].set(
                jacobi_poly_single(r_jacobi, 1, alpha + dx, beta=dx)
            )
            return (alpha, P_past)

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, MAXDR + 1))
        _, P_past = fori_loop(0, MAXDR + 1, find_init_jacobi, (alpha, P_past))

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    # Make inputs 1D arrays in case they aren't
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = jnp.asarray(dr).astype(int)

    # From the vectorization, the overall output will be (r.size, m.size)
    out = jnp.zeros(m.size)
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    # This part can be better implemented. Try to make dr as static argument
    # jnp.vectorize doesn't allow it to be static
    MAXDR = 4
    dxs = jnp.arange(0, MAXDR + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jnp.vectorize, excluded=(1, 2, 3), signature="()->(k)")
def _zernike_radial_vectorized_gpu(r, l, m, dr):
    """Calculation of Radial part of Zernike polynomials."""

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]  # Jacobi at N-2
        P_n1 = P_past[1]  # Jacobi at N-1
        P_n = jacobi_poly_single(r_jacobi, N, alpha, 0, P_n1, P_n2)

        # Check if the calculated values is in the given modes
        mask = jnp.logical_and(m == alpha, n == N)
        result = (-1) ** N * r**alpha * P_n
        out = jnp.where(mask, result, out)

        # Shift past values if needed
        # For derivative order dx, if N is smaller than 2+dx, then only the initial
        # value calculated by find_init_jacobi function will be used. So, if you update
        # P_n's, preceeding values will be wrong.
        mask = N >= 2
        P_n2 = jnp.where(mask, P_n1, P_n2)
        P_n1 = jnp.where(mask, P_n, P_n1)
        # Form updated P_past matrix
        P_past = P_past.at[0].set(P_n2)
        P_past = P_past.at[1].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros(2)
        P_past = P_past.at[0].set(jacobi_poly_single(r_jacobi, 0, alpha, beta=0))
        # Jacobi for n=1
        P_past = P_past.at[1].set(jacobi_poly_single(r_jacobi, 1, alpha, beta=0))

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    # Make inputs 1D arrays in case they aren't
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)

    # From the vectorization, the overall output will be (r.size, m.size)
    out = jnp.zeros(m.size)
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jnp.vectorize, excluded=(1, 2, 3), signature="()->(k)")
def _zernike_radial_vectorized_d1_gpu(r, l, m, dr):
    """First derivative calculation of Radial part of Zernike polynomials."""

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]  # Jacobi at N-2
        P_n1 = P_past[1]  # Jacobi at N-1
        P_n = jnp.zeros(MAXDR + 1)  # Jacobi at N

        # Calculate Jacobi polynomial and derivatives for (alpha,N)
        _, _, _, _, _, P_n = fori_loop(
            0,
            MAXDR + 1,
            find_intermadiate_jacobi,
            (r_jacobi, N, alpha, P_n1, P_n2, P_n),
        )
        # Calculate coefficients for derivatives. coef[0] will never be used. Jax
        # doesn't have Gamma function directly, that's why we calculate Logarithm of
        # Gamma function and then exponentiate it.
        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )
        # 1th Derivative of Zernike Radial
        result = (-1) ** N * (
            alpha * r ** jnp.maximum(alpha - 1, 0) * P_n[0]
            - coef[1] * 4 * r ** (alpha + 1) * P_n[1]
        )
        # Check if the calculated values is in the given modes
        mask = jnp.logical_and(m == alpha, n == N)
        out = jnp.where(mask, result, out)

        # Shift past values if needed
        # For derivative order dx, if N is smaller than 2+dx, then only the initial
        # value calculated by find_init_jacobi function will be used. So, if you update
        # P_n's, preceeding values will be wrong.
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask, P_n1, P_n2)
        P_n1 = jnp.where(mask, P_n, P_n1)
        # Form updated P_past matrix
        P_past = P_past.at[0, :].set(P_n2)
        P_past = P_past.at[1, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, MAXDR + 1))
        _, _, P_past = fori_loop(
            0, MAXDR + 1, find_initial_jacobi, (r_jacobi, alpha, P_past)
        )

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    # Make inputs 1D arrays in case they aren't
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = jnp.asarray(dr).astype(int)

    # From the vectorization, the overall output will be (r.size, m.size)
    out = jnp.zeros(m.size)
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    # This part can be better implemented. Try to make dr as static argument
    # jnp.vectorize doesn't allow it to be static
    MAXDR = 1
    dxs = jnp.arange(0, MAXDR + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jnp.vectorize, excluded=(1, 2, 3), signature="()->(k)")
def _zernike_radial_vectorized_d2_gpu(r, l, m, dr):
    """Second derivative calculation of Radial part of Zernike polynomials."""

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]  # Jacobi at N-2
        P_n1 = P_past[1]  # Jacobi at N-1
        P_n = jnp.zeros(MAXDR + 1)  # Jacobi at N

        # Calculate Jacobi polynomial and derivatives for (alpha,N)
        _, _, _, _, _, P_n = fori_loop(
            0,
            MAXDR + 1,
            find_intermadiate_jacobi,
            (r_jacobi, N, alpha, P_n1, P_n2, P_n),
        )

        # Calculate coefficients for derivatives. coef[0] will never be used. Jax
        # doesn't have Gamma function directly, that's why we calculate Logarithm of
        # Gamma function and then exponentiate it.
        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )

        result = (-1) ** N * (
            (alpha - 1) * alpha * r ** jnp.maximum(alpha - 2, 0) * P_n[0]
            - coef[1] * 4 * (2 * alpha + 1) * r**alpha * P_n[1]
            + coef[2] * 16 * r ** (alpha + 2) * P_n[2]
        )
        # Check if the calculated values is in the given modes
        mask = jnp.logical_and(m == alpha, n == N)
        out = jnp.where(mask, result, out)

        # Shift past values if needed
        # For derivative order dx, if N is smaller than 2+dx, then only the initial
        # value calculated by find_init_jacobi function will be used. So, if you update
        # P_n's, preceeding values will be wrong.
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask, P_n1, P_n2)
        P_n1 = jnp.where(mask, P_n, P_n1)
        # Form updated P_past matrix
        P_past = P_past.at[0, :].set(P_n2)
        P_past = P_past.at[1, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, MAXDR + 1))
        _, _, P_past = fori_loop(
            0, MAXDR + 1, find_initial_jacobi, (r_jacobi, alpha, P_past)
        )

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    # Make inputs 1D arrays in case they aren't
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = jnp.asarray(dr).astype(int)

    # From the vectorization, the overall output will be (r.size, m.size)
    out = jnp.zeros(m.size)
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    # This part can be better implemented. Try to make dr as static argument
    # jnp.vectorize doesn't allow it to be static
    MAXDR = 2
    dxs = jnp.arange(0, MAXDR + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jnp.vectorize, excluded=(1, 2, 3), signature="()->(k)")
def _zernike_radial_vectorized_d3_gpu(r, l, m, dr):
    """Third derivative calculation of Radial part of Zernike polynomials."""

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]  # Jacobi at N-2
        P_n1 = P_past[1]  # Jacobi at N-1
        P_n = jnp.zeros(MAXDR + 1)  # Jacobi at N

        # Calculate Jacobi polynomial and derivatives for (alpha,N)
        _, _, _, _, _, P_n = fori_loop(
            0,
            MAXDR + 1,
            find_intermadiate_jacobi,
            (r_jacobi, N, alpha, P_n1, P_n2, P_n),
        )

        # Calculate coefficients for derivatives. coef[0] will never be used. Jax
        # doesn't have Gamma function directly, that's why we calculate Logarithm of
        # Gamma function and then exponentiate it.
        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )

        # 3rd Derivative of Zernike Radial
        result = (-1) ** N * (
            (alpha - 2) * (alpha - 1) * alpha * r ** jnp.maximum(alpha - 3, 0) * P_n[0]
            - coef[1] * 12 * alpha**2 * r ** jnp.maximum(alpha - 1, 0) * P_n[1]
            + coef[2] * 48 * (alpha + 1) * r ** (alpha + 1) * P_n[2]
            - coef[3] * 64 * r ** (alpha + 3) * P_n[3]
        )
        # Check if the calculated values is in the given modes
        mask = jnp.logical_and(m == alpha, n == N)
        out = jnp.where(mask, result, out)

        # Shift past values if needed
        # For derivative order dx, if N is smaller than 2+dx, then only the initial
        # value calculated by find_init_jacobi function will be used. So, if you update
        # P_n's, preceeding values will be wrong.
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask, P_n1, P_n2)
        P_n1 = jnp.where(mask, P_n, P_n1)
        # Form updated P_past matrix
        P_past = P_past.at[0, :].set(P_n2)
        P_past = P_past.at[1, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, MAXDR + 1))
        _, _, P_past = fori_loop(
            0, MAXDR + 1, find_initial_jacobi, (r_jacobi, alpha, P_past)
        )

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    # Make inputs 1D arrays in case they aren't
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = jnp.asarray(dr).astype(int)

    # From the vectorization, the overall output will be (r.size, m.size)
    out = jnp.zeros(m.size)
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    # This part can be better implemented. Try to make dr as static argument
    # jnp.vectorize doesn't allow it to be static
    MAXDR = 3
    dxs = jnp.arange(0, MAXDR + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jnp.vectorize, excluded=(1, 2, 3), signature="()->(k)")
def _zernike_radial_vectorized_d4_gpu(r, l, m, dr):
    """Fourth derivative calculation of Radial part of Zernike polynomials."""

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]  # Jacobi at N-2
        P_n1 = P_past[1]  # Jacobi at N-1
        P_n = jnp.zeros(MAXDR + 1)  # Jacobi at N

        # Calculate Jacobi polynomial and derivatives for (alpha,N)
        _, _, _, _, _, P_n = fori_loop(
            0,
            MAXDR + 1,
            find_intermadiate_jacobi,
            (r_jacobi, N, alpha, P_n1, P_n2, P_n),
        )

        # Calculate coefficients for derivatives. coef[0] will never be used. Jax
        # doesn't have Gamma function directly, that's why we calculate Logarithm of
        # Gamma function and then exponentiate it.
        coef = jnp.exp(
            gammaln(alpha + N + 1 + dxs) - dxs * jnp.log(2) - gammaln(alpha + N + 1)
        )

        # 4th Derivative of Zernike Radial
        result = (-1) ** N * (
            (alpha - 3)
            * (alpha - 2)
            * (alpha - 1)
            * alpha
            * r ** jnp.maximum(alpha - 4, 0)
            * P_n[0]
            - coef[1]
            * 8
            * alpha
            * (2 * alpha**2 - 3 * alpha + 1)
            * r ** jnp.maximum(alpha - 2, 0)
            * P_n[1]
            + coef[2] * 48 * (2 * alpha**2 + 2 * alpha + 1) * r**alpha * P_n[2]
            - coef[3] * 128 * (2 * alpha + 3) * r ** (alpha + 2) * P_n[3]
            + coef[4] * 256 * r ** (alpha + 4) * P_n[4]
        )
        # Check if the calculated values is in the given modes
        mask = jnp.logical_and(m == alpha, n == N)
        out = jnp.where(mask, result, out)

        # Shift past values if needed
        # For derivative order dx, if N is smaller than 2+dx, then only the initial
        # value calculated by find_init_jacobi function will be used. So, if you update
        # P_n's, preceeding values will be wrong.
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask, P_n1, P_n2)
        P_n1 = jnp.where(mask, P_n, P_n1)
        # Form updated P_past matrix
        P_past = P_past.at[0, :].set(P_n2)
        P_past = P_past.at[1, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, MAXDR + 1))
        _, _, P_past = fori_loop(
            0, MAXDR + 1, find_initial_jacobi, (r_jacobi, alpha, P_past)
        )

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    # Make inputs 1D arrays in case they aren't
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = jnp.asarray(dr).astype(int)

    # From the vectorization, the overall output will be (r.size, m.size)
    out = jnp.zeros(m.size)
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)

    # This part can be better implemented. Try to make dr as static argument
    # jnp.vectorize doesn't allow it to be static
    MAXDR = 4
    dxs = jnp.arange(0, MAXDR + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@jit
def fourier(theta, m, NFP=1, dt=0):
    """Fourier series.

    Parameters
    ----------
    theta : ndarray, shape(N,)
        poloidal/toroidal coordinates to evaluate basis
    m : ndarray of int, shape(K,)
        poloidal/toroidal mode number(s)
    NFP : int
        number of field periods (Default = 1)
    dt : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    theta, m, NFP, dt = map(jnp.asarray, (theta, m, NFP, dt))
    m_pos = (m >= 0).astype(int)
    m_abs = jnp.abs(m) * NFP
    shift = m_pos * jnp.pi / 2 + dt * jnp.pi / 2
    return m_abs**dt * jnp.sin(m_abs * theta + shift)


@custom_jvp
@jit
@jnp.vectorize
def _jacobi(n, alpha, beta, x, dx=0):
    """Jacobi polynomial evaluation.

    Implementation is only correct for non-negative integer coefficients,
    returns 0 otherwise.

    Parameters
    ----------
    n : int, array_like
        Degree of the polynomial.
    alpha : int, array_like
        Parameter
    beta : int, array_like
        Parameter
    x : float, array_like
        Points at which to evaluate the polynomial

    Returns
    -------
    P : ndarray
        Values of the Jacobi polynomial
    """
    # adapted from scipy:
    # https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/
    # scipy/special/orthogonal_eval.pxd#L144

    def _jacobi_body_fun(kk, d_p_a_b_x):
        d, p, alpha, beta, x = d_p_a_b_x
        k = kk + 1.0
        t = 2 * k + alpha + beta
        d = (
            (t * (t + 1) * (t + 2)) * (x - 1) * p + 2 * k * (k + beta) * (t + 2) * d
        ) / (2 * (k + alpha + 1) * (k + alpha + beta + 1) * t)
        p = d + p
        return (d, p, alpha, beta, x)

    n, alpha, beta, x = map(jnp.asarray, (n, alpha, beta, x))

    # coefficient for derivative
    c = (
        gammaln(alpha + beta + n + 1 + dx)
        - dx * jnp.log(2)
        - gammaln(alpha + beta + n + 1)
    )
    c = jnp.exp(c)
    # taking derivative is same as coeff*jacobi but for shifted n,a,b
    n -= dx
    alpha += dx
    beta += dx

    d = (alpha + beta + 2) * (x - 1) / (2 * (alpha + 1))
    p = d + 1
    d, p, alpha, beta, x = fori_loop(
        0, jnp.maximum(n - 1, 0).astype(int), _jacobi_body_fun, (d, p, alpha, beta, x)
    )
    out = _binom(n + alpha, n) * p
    # should be complex for n<0, but it gets replaced elsewhere so just return 0 here
    out = jnp.where(n < 0, 0, out)
    # other edge cases
    out = jnp.where(n == 0, 1.0, out)
    out = jnp.where(n == 1, 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (x - 1)), out)
    return c * out


@jit
@jnp.vectorize
def _binom(n, k):
    """Binomial coefficient.

    Implementation is only correct for positive integer n,k and n>=k

    Parameters
    ----------
    n : int, array-like
        number of things to choose from
    k : int, array-like
        number of things chosen

    Returns
    -------
    val : int, float, array-like
        number of possible combinations
    """
    # adapted from scipy:
    # https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/
    # scipy/special/orthogonal_eval.pxd#L68

    n, k = map(jnp.asarray, (n, k))

    def _binom_body_fun(i, b_n):
        b, n = b_n
        num = n + 1 - i
        den = i
        return (b * num / den, n)

    kx = k.astype(int)
    b, n = fori_loop(1, 1 + kx, _binom_body_fun, (1.0, n))
    return b


@zernike_radial_rory.defjvp
def _zernike_radial_rory_jvp(x, xdot):
    (r, l, m, dr) = x
    (rdot, ldot, mdot, drdot) = xdot
    f = zernike_radial_rory(r, l, m, dr)
    df = zernike_radial_rory(r, l, m, dr + 1)
    # in theory l, m, dr aren't differentiable (they're integers)
    # but marking them as non-diff argnums seems to cause escaped tracer values.
    # probably a more elegant fix, but just setting those derivatives to zero seems
    # to work fine.
    return f, (df.T * rdot).T + 0 * ldot + 0 * mdot + 0 * drdot


@zernike_radial_jvp.defjvp
def _zernike_radial_jvp(x, xdot):
    (r, l, m, dr) = x
    (rdot, ldot, mdot, drdot) = xdot
    f = zernike_radial_jvp(r, l, m, dr)
    df = zernike_radial_jvp(r, l, m, dr + 1)
    # in theory l, m, dr aren't differentiable (they're integers)
    # but marking them as non-diff argnums seems to cause escaped tracer values.
    # probably a more elegant fix, but just setting those derivatives to zero seems
    # to work fine.
    return f, (df.T * rdot).T + 0 * ldot + 0 * mdot + 0 * drdot


@zernike_radial_jvp_gpu.defjvp
def _zernike_radial_jvp_gpu_jvp(x, xdot):
    (r, l, m, dr) = x
    (rdot, ldot, mdot, drdot) = xdot
    f = zernike_radial_jvp_gpu(r, l, m, dr)
    df = zernike_radial_jvp_gpu(r, l, m, dr + 1)
    # in theory l, m, dr aren't differentiable (they're integers)
    # but marking them as non-diff argnums seems to cause escaped tracer values.
    # probably a more elegant fix, but just setting those derivatives to zero seems
    # to work fine.
    return f, (df.T * rdot).T + 0 * ldot + 0 * mdot + 0 * drdot


@zernike_radial_switch_gpu.defjvp
def _zernike_radial_switch_gpu_jvp(x, xdot):
    (r, l, m, dr) = x
    (rdot, ldot, mdot, drdot) = xdot
    f = zernike_radial_switch_gpu(r, l, m, dr)
    df = zernike_radial_switch_gpu(r, l, m, dr + 1)
    # in theory l, m, dr aren't differentiable (they're integers)
    # but marking them as non-diff argnums seems to cause escaped tracer values.
    # probably a more elegant fix, but just setting those derivatives to zero seems
    # to work fine.
    return f, (df.T * rdot).T + 0 * ldot + 0 * mdot + 0 * drdot


@zernike_radial_switch.defjvp
def _zernike_radial_switch_jvp(x, xdot):
    (r, l, m, dr) = x
    (rdot, ldot, mdot, drdot) = xdot
    f = zernike_radial_switch(r, l, m, dr)
    df = zernike_radial_switch(r, l, m, dr + 1)
    # in theory l, m, dr aren't differentiable (they're integers)
    # but marking them as non-diff argnums seems to cause escaped tracer values.
    # probably a more elegant fix, but just setting those derivatives to zero seems
    # to work fine.
    return f, (df.T * rdot).T + 0 * ldot + 0 * mdot + 0 * drdot


@zernike_radial_if_switch.defjvp
def _zernike_radial_if_switch_jvp(x, xdot):
    (r, l, m, dr) = x
    (rdot, ldot, mdot, drdot) = xdot
    f = zernike_radial_if_switch(r, l, m, dr)
    df = zernike_radial_if_switch(r, l, m, dr + 1)
    # in theory l, m, dr aren't differentiable (they're integers)
    # but marking them as non-diff argnums seems to cause escaped tracer values.
    # probably a more elegant fix, but just setting those derivatives to zero seems
    # to work fine.
    return f, (df.T * rdot).T + 0 * ldot + 0 * mdot + 0 * drdot


@_jacobi.defjvp
def _jacobi_jvp(x, xdot):
    (n, alpha, beta, x, dx) = x
    (ndot, alphadot, betadot, xdot, dxdot) = xdot
    f = _jacobi(n, alpha, beta, x, dx)
    df = _jacobi(n, alpha, beta, x, dx + 1)
    # in theory n, alpha, beta, dx aren't differentiable (they're integers)
    # but marking them as non-diff argnums seems to cause escaped tracer values.
    # probably a more elegant fix, but just setting those derivatives to zero seems
    # to work fine.
    return f, df * xdot + 0 * ndot + 0 * alphadot + 0 * betadot + 0 * dxdot
