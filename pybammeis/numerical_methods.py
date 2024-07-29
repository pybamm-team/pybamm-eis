import numpy as np
import scipy.sparse


def empty():
    pass


def bicgstab(A, b, start_point=None, callback=None, tol=1e-3):
    """
    Solves the linear equation Ax = b using the BiCGSTAB method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        A square matrix represented as a SciPy sparse CSR matrix.
    b : numpy.ndarray
        Right-hand side vector in the equation Ax = b, expected to be an n x 1 dimensional array.
    start_point : numpy.ndarray, optional
        The starting guess for the iteration. Defaults to a zero vector if not provided.
    callback : callable, optional
        A function `callback(xk)` that is called after each iteration, where `xk` is the current solution vector.
        If no function is provided, no action is taken at each iteration.
    tol : float, optional
        The tolerance level for convergence. The iteration will stop when the residual is below this tolerance.
        Default is 1e-3.

    Returns
    -------
    xk : numpy.ndarray
        The solution vector to the equation Ax = b, as an n x 1 dimensional array.
    """
    callback = callback or empty

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

    for _ in range(1, max_num_iter + 1):
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


def prebicgstab(A, b, LU, start_point=None, callback=None, tol=1e-3):
    """
    Solves the linear equation Ax = b using the preconditioned BiCGSTAB method.
    The preconditioner is specified as LU, which could be the form of an LU decomposition.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        A square matrix.
    b : numpy.ndarray
        Right-hand side vector in the equation Ax = b, expected to be an n x 1 dimensional array.
    LU : scipy.sparse.csr_matrix
        The LU decomposition of A, typically a scipy.sparse.linalg.SuperLU object.
    start_point : numpy.ndarray, optional
        The starting guess for the iteration. Defaults to a zero vector if not provided.
    callback : callable, optional
        A function callback(xk) that is called after each iteration, where `xk` is the current solution vector.
        The default behavior is to perform no action on callback.
    tol : float, optional
        The tolerance level for convergence. The iteration will stop when the residual is below this tolerance.
        Default is 1e-3.

    Returns
    -------
    xk : numpy.ndarray
        The solution vector to the equation Ax = b, as an n x 1 dimensional array.
    """
    callback = callback or empty

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
    if isinstance(LU, scipy.sparse.linalg.SuperLU):
        superLU = True
    else:
        superLU = False

    # Start the iterative step
    for _ in range(1, max_num_iter + 1):
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
