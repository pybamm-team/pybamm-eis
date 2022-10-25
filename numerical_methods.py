#
# Linear algebra methods
#

import numpy as np
import scipy.sparse


def empty():
    # An empty callback function. Callbacks can be written as desired.
    pass


def conjugate_gradient(A, b, start_point=None, callback=empty, tol=1e-3):
    """
    Uses the conjugate gradient method to solve Ax = b. Should not be used
    unless A is Hermitian. If A is not hermitian, use BicgSTAB instead.
    For best performance A should be a scipy csr sparse matrix.

    Parameters
    ----------
    A : scipy sparse csr matrix
        A square matrix.
    b : numpy 1xn array
    start_point : numpy 1xn array, optional
        Where the iteration starts.  If not provided the initial guess will be zero.
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
        A tolerance at which to stop the iteration. The default is 1e-3.

    Returns
    -------
    xk : numpy 1xn array
        The solution of Ax = b.

    """
    if start_point is None:
        start_point = np.zeros_like(b)

    xk = np.array(start_point)  # isn't start point already an array?
    rk = b - A @ xk
    pk = rk

    max_num_iter = len(b)
    rk1rk1 = np.dot(np.conj(rk), rk)

    for k in range(max_num_iter):
        Apk = A @ pk
        rkrk = rk1rk1
        pkApk = np.dot(np.conj(pk), Apk)

        alpha_k = rkrk / pkApk

        xk = xk + alpha_k * pk

        callback(xk)

        # Stop if the change in the last entry is under tolerance
        if alpha_k * pk[-1] < tol:
            break
        else:
            rk = rk - alpha_k * Apk

            rk1rk1 = np.dot(np.conj(rk), rk)

            beta_k = rk1rk1 / rkrk

            pk = rk + beta_k * pk

    return xk


def bicgstab(A, b, start_point=None, callback=empty, tol=10**-3):
    """
    Uses the BicgSTAB method to solve Ax = b

    Parameters
    ----------
    A : scipy sparse csr matrix
        A square matrix.
    b : numpy 1xn array
    start_point : numpy 1xn array, optional
        Where the iteration starts. If not provided the initial guess will be zero.
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
        A tolerance at which to stop the iteration. The default is 10**-3.

    Returns
    -------
    xk : numpy 1xn array
        The solution of Ax = b.

    """
    if start_point is None:
        start_point = np.zeros_like(b)

    xk = np.array(start_point)
    rk = b - A @ xk
    r0 = np.conj(rk)
    rhok = 1
    alpha_k = 1
    wk = 1

    pk = np.zeros(np.shape(b))
    vk = pk
    max_num_iter = 2 * np.shape(b)[0]

    for k in range(1, max_num_iter + 1):
        rhok1 = rhok
        rhok = np.dot(r0.T, rk)
        beta_k = (rhok / rhok1) * (alpha_k / wk)

        pk = rk + beta_k * (pk - wk * vk)
        vk = A @ pk

        alpha_k = rhok / np.dot(r0.T, vk)

        h = xk + alpha_k * pk

        s = rk - alpha_k * vk
        t = A @ s

        wk = np.dot(np.conj(t.T), s) / np.dot(np.conj(t.T), t)
        xk = h + wk * s
        callback(xk)
        if np.linalg.norm(rk, 1) < tol:
            break
        else:
            rk = s - wk * t
    return xk


def prebicgstab(A, b, L, U=None, start_point=None, callback=empty, tol=1e-3):
    """
    Uses the preconditioned BicgSTAB method to solve Ax = b. The preconditioner
    is of the form LU, or just of the form L.

    Parameters
    ----------
    A : scipy sparse csr matrix
        A square matrix.
    b : numpy 1xn array
    L : scipy sparse csr matrix
        A lower (block) triangular matrix (or superLU object)
    U : scipy sparse crs matrix, optional
        An Upper (block) triangular matrix. None used if not given.
    start_point : numpy 1xn array, optional
        Where the iteration starts. If not provided the initial guess will be zero.
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
        A tolerance at which to stop the iteration. The default is 1e-3.

    Returns
    -------
    xk : numpy 1xn array
        The solution of Ax = b.

    """
    if start_point is None:
        start_point = np.zeros_like(b)

    xk = np.array(start_point)
    rk = b - A @ xk

    r0 = np.conj(rk)

    rhok = 1
    alpha_k = 1
    wk = 1

    pk = np.zeros_like(b)
    vk = pk

    max_num_iter = 2 * np.shape(b)[0]

    if type(L) == scipy.sparse.linalg.SuperLU:
        superLU = True
    else:
        superLU = False

    for k in range(1, max_num_iter + 1):
        rhok1 = rhok
        rhok = np.dot(r0.T, rk)
        beta_k = (rhok / rhok1) * (alpha_k / wk)
        pk = rk + beta_k * (pk - wk * vk)

        if superLU:
            y = np.array(L.solve(pk))
        else:
            y = scipy.sparse.linalg.spsolve(L, pk)
            if U:
                y = np.array(scipy.sparse.linalg.spsolve(U, y))

        y = np.reshape(y, np.shape(b))

        vk = A @ y
        alpha_k = rhok / np.dot(r0.T, vk)

        h = xk + alpha_k * y

        s = rk - alpha_k * vk

        s = rk - alpha_k * vk

        if superLU:
            z = np.array(L.solve(s))
        else:
            z = scipy.sparse.linalg.spsolve(L, s)
            if U:
                z = np.array(scipy.sparse.linalg.spsolve(U, z))

        z = np.reshape(z, np.shape(b))

        t = A @ z

        wk = np.dot(np.conj(t.T), s) / np.dot(np.conj(t.T), t)
        xk = h + wk * z
        callback(xk)
        if np.linalg.norm(rk, 1) < tol:
            break
        else:
            rk = s - wk * t
    return xk


def matrix_rescale(M, J, b):
    n = np.shape(M)[0]
    multiplier = np.ones(n, dtype="complex")

    multiplier[-2] = 3
    multiplier[-1] = 2

    multiplier = scipy.sparse.diags(multiplier)
    M = multiplier @ M
    J = multiplier @ J
    b = multiplier @ b

    return M, J, b
