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
    b : numpy nx1 array
    start_point : numpy nx1 array, optional
        Where the iteration starts.  If not provided the initial guess will be zero.
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
        A tolerance at which to stop the iteration. The default is 1e-3.

    Returns
    -------
    xk : numpy nx1 array
        The solution of Ax = b.

    """

    # The default start point is b unless specified otherwise
    if start_point is None:
        start_point = np.zeros_like(b)

    xk = np.array(start_point)
    # Find the residual and set the search direction to the residual
    rk = b - A @ xk
    pk = rk

    max_num_iter = np.shape(b)[0]
    rk1rk1 = np.dot(np.conj(rk), rk)

    # start the iterative step
    for k in range(max_num_iter):
        # Find alpha_k, the distance to move in the search direction
        Apk = A @ pk
        rkrk = rk1rk1
        pkApk = np.dot(np.conj(pk), Apk)

        alpha_k = rkrk / pkApk

        xk = xk + alpha_k * pk

        # run the callback
        callback(xk)

        # Stop if the change in the last entry is under tolerance
        if alpha_k * pk[-1] < tol:
            break
        else:
            # Find the new residual
            rk = rk - alpha_k * Apk

            rk1rk1 = np.dot(np.conj(rk), rk)

            beta_k = rk1rk1 / rkrk

            # Update the search direction
            pk = rk + beta_k * pk

    return xk


def bicgstab(A, b, start_point=None, callback=empty, tol=10**-3):
    """
    Uses the BicgSTAB method to solve Ax = b

    Parameters
    ----------
    A : scipy sparse csr matrix
        A square matrix.
    b : numpy nx1 array
    start_point : numpy nx1 array, optional
        Where the iteration starts. If not provided the initial guess will be zero.
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
        A tolerance at which to stop the iteration. The default is 10**-3.

    Returns
    -------
    xk : numpy nx1 array
        The solution of Ax = b.

    """
    # The default start point is b unless specified otherwise
    if start_point is None:
        start_point = np.zeros_like(b)

    xk = np.array(start_point)
    # Find the residual
    rk = b - A @ xk
    r0 = np.conj(rk)
    rhok = 1
    alpha_k = 1
    wk = 1

    # set the search direction to the residual
    pk = np.zeros(np.shape(b))
    vk = pk

    # Since bicgstab uses cg on a matrix of size 2n, set the max number of
    # iterations as follows
    max_num_iter = 2 * np.shape(b)[0]

    for k in range(1, max_num_iter + 1):
        # Calculate the next search direction pk
        rhok1 = rhok
        rhok = np.dot(r0.T, rk)
        beta_k = (rhok / rhok1) * (alpha_k / wk)

        pk = rk + beta_k * (pk - wk * vk)
        vk = A @ pk

        # Calculate the distance to move in the pk direction
        alpha_k = rhok / np.dot(r0.T, vk)

        # Move alpha_k in the pk direction
        h = xk + alpha_k * pk

        s = rk - alpha_k * vk
        t = A @ s

        wk = np.dot(np.conj(t.T), s) / np.dot(np.conj(t.T), t)
        # Update xk
        xk = h + wk * s
        # Run the callback
        callback(xk)

        # Check whether the 1-norm of the residual is less than the tolerance
        if np.linalg.norm(rk, 1) < tol:
            break
        else:
            # Update the residual
            rk = s - wk * t
    return xk


def prebicgstab(A, b, LU, start_point=None, callback=empty, tol=1e-3):
    """
    Uses the preconditioned BicgSTAB method to solve Ax = b. The preconditioner
    is of the form LU, or just of the form L.

    Parameters
    ----------
    A : scipy sparse csr matrix
        A square matrix.
    b : numpy nx1 array
    LU : scipy sparse csr matrix
        The LU decomposition (typically a superLU object)
    start_point : numpy nx1 array, optional
        Where the iteration starts. If not provided the initial guess will be zero.
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
        A tolerance at which to stop the iteration. The default is 1e-3.

    Returns
    -------
    xk : numpy nx1 array
        The solution of Ax = b.

    """
    # The default start point is b unless specified otherwise
    if start_point is None:
        start_point = np.zeros_like(b)

    xk = np.array(start_point)
    # Find the residual
    rk = b - A @ xk

    r0 = np.conj(rk)

    rhok = 1
    alpha_k = 1
    wk = 1

    pk = np.zeros_like(b)
    vk = pk

    # Since bicgstab uses cg on a matrix of size 2n, set the max number of
    # iterations as follows
    max_num_iter = 2 * np.shape(b)[0]

    # Check the format of LU (super LU or scipy sparse matrix)
    if type(LU) == scipy.sparse.linalg.SuperLU:
        superLU = True
    else:
        superLU = False

    # Start the iterative step
    for k in range(1, max_num_iter + 1):
        # Calculate the search direction pk
        rhok1 = rhok
        rhok = np.dot(r0.T, rk)
        beta_k = (rhok / rhok1) * (alpha_k / wk)
        pk = rk + beta_k * (pk - wk * vk)

        # Use the preconditioning to solve LUy = pk. Do this depending
        # on the format of L.
        if superLU:
            y = np.array(LU.solve(pk))
        else:
            y = scipy.sparse.linalg.spsolve(LU, pk)

        # Reshape y to a nx1 matrix, so the rest of the calculations can be done.
        y = np.reshape(y, np.shape(b))

        vk = A @ y
        alpha_k = rhok / np.dot(r0.T, vk)

        h = xk + alpha_k * y

        s = rk - alpha_k * vk

        s = rk - alpha_k * vk

        # Perform the preconditioning to solve LUz = s. Do this depending
        # on the format of L.
        if superLU:
            z = np.array(LU.solve(s))
        else:
            z = scipy.sparse.linalg.spsolve(LU, s)

        # Reshape z to a nx1 matrix, so the rest of the calculations can be done.
        z = np.reshape(z, np.shape(b))

        t = A @ z

        wk = np.dot(np.conj(t.T), s) / np.dot(np.conj(t.T), t)

        # Update xk
        xk = h + wk * z

        # Run the callback
        callback(xk)

        # Check whether the 1-norm of the residual is less than the tolerance
        if np.linalg.norm(rk, 1) < tol:
            break
        else:
            # Update the residual
            rk = s - wk * t
    return xk


def matrix_rescale(M, J, b):
    """
    Rescale the matrices to increase the convergence of the last 2 entries
    by increasing their weight.

    Parameters
    ----------
    M : scipy sparse csr matrix
        Mass matrix
    J : scipy sparse csr matrix
        Jacobian
    b : numpy nx1
        the RHS

    Returns
    -------
    M : scipy sparse csr matrix
        Mass matrix
    J : scipy sparse csr matrix
        Jacobian
    b : numpy nx1
        the RHS

    """

    n = np.shape(M)[0]
    # set a multiplier to multiply the rows. All are multiplied by 1 except the
    # last 2. The last row is multiplied by 2 and second last by 3.
    multiplier = np.ones(n, dtype="complex")

    multiplier[-2] = 3
    multiplier[-1] = 2

    # Make the multiplier a nxn matrix with elements on the diagonal to scale
    # the rows.
    multiplier = scipy.sparse.diags(multiplier)
    M = multiplier @ M
    J = multiplier @ J
    b = multiplier @ b

    return M, J, b
