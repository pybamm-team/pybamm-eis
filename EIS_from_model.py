# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:00:27 2022

@author: rish3
"""

import numerical_methods as nm
import numpy as np
import time
import scipy.sparse
import matplotlib.pyplot as plt

global start_point, st, LUt
LUt = 1
st = 0


def EIS(M, J, b, start_freq, end_freq, num_points, method="auto"):
    """
    Calculates impedence for a range of frequencies

    Solves iwMc = Jc + b

    Voltage must be the 2nd to last entry in the matrix and Current should be last

    Parameters
    ----------
    M : sparse csr matrix
        The mass matrix
    J : sparse csr matrix
        The Jacobian
    b : numpy 1xn array
        The RHS
    start_freq : float
        The initial frequency in a frequency range to be evaluated at.
    end_freq : float
        The final frequency in a frequency range to be evaluated at.
    num_points : int
        the minimum number of frequencies impedence is calculated at over
        the range.
    method : string, optional
        the numerical algorithm to use. Options are:
        cg - conjugate gradient - only use for Hermitian matrices
        bicgstab - use bicg with no preconditioning
        prebicgstab - use bicg with preconditioning
        direct - use Gaussian elimination
        semi-direct - use Gaussian elimination and approximate other points -
        can be a little unreliable.
        auto - chooses a good method automatically

        The default is 'auto'.

    Raises
    ------
    Exception
        If invalid data is entered.

    Returns
    -------
    Zs : list
        Complex values of impedence at each frequecy.
    ws : list
        Frequencies evaluated at.
    time_taken : float
        How long the calculation took.

    """

    # A = iwM - J, Ac = b

    start_timer = time.time()

    if start_freq <= 0 or end_freq < start_freq:
        raise Exception("Invalid range of frequencies")

    if method == "bicgstab" or method == "prebicgstab" or method == "cg":
        Zs, ws = iterative_method(M, J, b, start_freq, end_freq, num_points, method)
    elif method == "direct":
        Zs, ws = direct_method(M, J, b, start_freq, end_freq, num_points)
    elif method == "semi-direct":
        Zs, ws = semi_direct_method(M, J, b, start_freq, end_freq, num_points)
    elif method == "auto":
        Zs, ws = iterative_method(
            M, J, b, start_freq, end_freq, num_points, "prebicgstab", preconditioner=ELU
        )
    else:
        raise Exception("Not a valid method")

    end_timer = time.time()
    time_taken = end_timer - start_timer

    return Zs, ws, time_taken


def ILU(A, M, J, L, U, b=None):
    """
    An ILU preconditioner method to be used with prebicgstab

    Parameters
    ----------

    A : Sparse csr matrix
        The matrix to solve for
    M : Sparse csr matrix
        The mass matrix
    J : Sparse csr matrix
        The Jacobian
    L : Sparse csr matrix
        The previous L in an LU preconditioner
    U : Sparse csr matrix
        The previous U in an LU preconditioner
    b : n x 1 numpy array (optional)
        The RHS of Ax = b. This is rarely needed for preconditioners.

    Returns
    -------

    L : Sparse csr matrix
        The L in an approximate LU factorisation
    U : Sparse csr matrix
        The U in an approximate LU factorisation
    """
    if type(L) == str:
        k = get_k(M)
        M_diags, A_diags = get_block_diagonals(M, A, k)

        L, U = nm.ILUpreconditioner(A_diags[0], A_diags[1], A_diags[2])
    return L, U


def G_S(A, M, J, L, U, b=None):
    """
    A Gauss-Siedel preconditioner method to be used with prebicgstab

    Parameters
    ----------

    A : Sparse csr matrix
        The matrix to solve for
    M : Sparse csr matrix
        The mass matrix
    J : Sparse csr matrix
        The Jacobian
    L : Sparse csr matrix
        The previous L in an LU preconditioner
    U : Sparse csr matrix
        The previous U in an LU preconditioner
    b : n x 1 numpy array (optional)
        The RHS of Ax = b. This is rarely needed for preconditioners.

    Returns
    -------

    L : Sparse csr matrix
        The L in an approximate LU factorisation
    U : String
        A return just to fill the space for U
    """
    L = scipy.sparse.tril(A, format="csr")
    U = None
    return L, U


def G_S_V(A, M, J, L, U, b=None):
    """
    A variant of the Gauss-Siedel preconditioner to be used with prebicgstab

    Parameters
    ----------

    A : Sparse csr matrix
        The matrix to solve for
    M : Sparse csr matrix
        The mass matrix
    J : Sparse csr matrix
        The Jacobian
    L : Sparse csr matrix
        The previous L in an LU preconditioner
    U : Sparse csr matrix
        The previous U in an LU preconditioner
    b : n x 1 numpy array (optional)
        The RHS of Ax = b. This is rarely needed for preconditioners.

    Returns
    -------

    L : Sparse csr matrix
        The L in an approximate LU factorisation
    U : Sparse csr matrix
        The U in an approximate LU factorisation

    Note L and U are reversed here because this is actually a UL factorisation.
    This doesn't affect bicgstab but is useful for consistency with returns
    from other preconditioners.
    """

    U = scipy.sparse.tril(A, format="csr")
    L = scipy.sparse.triu(A, k=1, format="csr")
    Id = scipy.sparse.eye(A.shape[0], dtype="complex", format="csr")
    L = L + Id
    return L, U


def ELU(A, M, J, L, U, b):
    """
    A  preconditioner method using exact factorisations to be used with
    prebicgstab. Also decides when to do factorisations based on time taken.

    Parameters
    ----------

    A : Sparse csr matrix
        The matrix to solve for
    M : Sparse csr matrix
        The mass matrix
    J : Sparse csr matrix
        The Jacobian
    L : Sparse csr matrix
        The previous L in an LU preconditioner
    U : Sparse csr matrix
        The previous U in an LU preconditioner
    b : n x 1 numpy array (optional)
        The RHS of Ax = b. This is rarely needed for preconditioners.

    Returns
    -------

    L : SuperLU
        SuperLU data type containing L and U
    U : String
        A return just to fill the space for U
    """
    global start_point, st, LUt

    et = time.time()

    # What does this block do?
    try:
        t = et - st
    except:
        pass

    if type(L) == str or LUt <= t:
        LUstart_time = time.time()
        L = scipy.sparse.linalg.splu(A.tocsc())
        start_point = L.solve(b)
        U = None
        LUend_time = time.time()
        LUt = LUend_time - LUstart_time

    st = time.time()

    return L, U


def iterative_method(
    M, J, b, start_freq, end_freq, num_points, method="prebicgstab", preconditioner=ELU
):
    """
    Calculates impedence for a range of frequencies using an iterative method

    solves iwMc = Jc + b

    Parameters
    ----------
    M : sparse csr matrix
    J : sparse csr matrix
    b : numpy 1xn array
    start_freq : float
        The initial frequency in a frequency range to be evaluated at.
    end_freq : float
        The final frequency in a frequency range to be evaluated at.
    num_points : int
        the minimum number of frequencies impedence is calculated at over
        the range.
    method : string, optional
        the numerical algorithm to use. Options are:
        cg - conjugate gradient - only use for Hermitian matrices
        bicgstab - use bicgstab with no preconditioner
        prebicgstab - use bicgstab with a preconditioner
    preconditioner: function, optional
        A function that calculates a preconditioner from A, M, J and the previous
        preconditioner. Returns L, U, triangular. Only relevent when using prebicgstab.
        Default is ELU. Return 'none' as a string for U if only L is
        being used.

    Returns
    -------
    Zs : list
        Complex values of impedence at each frequecy.
    ws : list
        Frequencies evaluated at.

    """

    # rescale the matrices to get faster convergence in the last 2 entries
    A = 1.0j * start_freq * M - J
    M, J, b = nm.matrix_rescale(A, M, J, b)

    Zs = []
    ws = []
    w = start_freq

    L = None
    U = None

    start_point = b

    w_log_increment_max = (np.log(end_freq) - np.log(start_freq)) / num_points
    iters = []
    while w <= end_freq:
        A = 1.0j * w * M - J
        num_iters = 0
        stored_vals = []
        ns = []

        if method == "prebicgstab":
            L, U = preconditioner(A, M, J, L, U, b)

        def callback(xk):
            nonlocal num_iters
            num_iters += 1
            stored_vals.append(xk[-1])
            ns.append(num_iters)

        if method == "bicgstab":
            c = nm.bicgstab(A, b, start_point=start_point, callback=callback)
        elif method == "prebicgstab":
            c = nm.prebicgstab(A, b, L, U, start_point=start_point, callback=callback)
        else:
            c = nm.conjugate_gradient(A, b, start_point=start_point, callback=callback)

        V = c[-2]
        I = c[-1]
        Z = V / I
        Zs.append(Z)

        es = np.abs(np.array(stored_vals) - V)

        ns = num_iters + 1 - np.array(ns)

        old_c = np.array(c)
        if len(Zs) == 1:
            w_log_increment = float(w_log_increment_max)
            start_point = c
        else:
            old_increment = float(w_log_increment)
            kappa = np.abs(V - start_point[-2]) / w_log_increment**2
            ys = []
            for j, e in enumerate(es):
                y = (
                    2
                    * ns[j]
                    / (
                        -w_log_increment
                        + np.sqrt((w_log_increment) ** 2 + 4 * (e + 0.01) / kappa)
                    )
                )
                ys.append(y)
            y_min = min(ys)
            if ys[-1] == y_min:
                n_val = ns[-1] + 1
                w_log_increment = min(n_val / y_min[0], w_log_increment_max)
            else:
                w_log_increment = min(
                    ns[ys.index(y_min)] / y_min[0], w_log_increment_max
                )

            start_point = c + w_log_increment / old_increment * (c - old_c)

        multiplier = np.exp(w_log_increment)

        ws.append(w)
        iters.append(num_iters)

        w = w * multiplier

    plt.plot(ws, iters)
    plt.show()
    return Zs, ws


def direct_method(M, J, b, start_freq, end_freq, num_points):
    """
    Calculates impedence for a range of frequencies using scipy

    solves iwMc = Jc + b

    Parameters
    ----------
    M : sparse csr matrix
        Must be block diagonal
    J : sparse csr matrix
        Must be block tridiagonal
    b : numpy 1xn array
    start_freq : float
        The initial frequency in a frequency range to be evaluated at.
    end_freq : float
        The final frequency in a frequency range to be evaluated at.
    num_points : int
        the minimum number of frequencies impedence is calculated at over
        the range.

    Returns
    -------
    Zs : list
        Complex values of impedence at each frequecy.
    ws : list
        Frequencies evaluated at.

    """
    Zs = []
    ws = np.exp(np.linspace(np.log(start_freq), np.log(end_freq), num_points))
    M = scipy.sparse.csc_matrix(M)
    J = scipy.sparse.csc_matrix(J)
    for w in ws:
        A = 1.0j * w * M - J
        lu = scipy.sparse.linalg.splu(A)
        ans = lu.solve(np.array(b))
        V = ans[-2]
        I = ans[-1]
        Z = V / I
        Zs.append(Z)
    return Zs, ws


def semi_direct_method(M, J, b, start_freq, end_freq, num_points):
    """
    Calculates impedence for a range of frequencies using scipy

    solves iwMc = Jc + b

    Parameters
    ----------
    M : sparse csr matrix
        Must be block diagonal
    J : sparse csr matrix
        Must be block tridiagonal
    b : numpy 1xn array
    start_freq : float
        The initial frequency in a frequency range to be evaluated at.
    end_freq : float
        The final frequency in a frequency range to be evaluated at.
    num_points : int
        the minimum number of frequencies impedence is calculated at over
        the range.

    Returns
    -------
    Zs : list
        Complex values of impedence at each frequecy.
    ws : list
        Frequencies evaluated at.

    """
    Zs = []
    ws = np.exp(np.linspace(np.log(start_freq), np.log(end_freq), num_points))
    M = scipy.sparse.csc_matrix(M)
    J = scipy.sparse.csc_matrix(J)
    ratio = 0
    for i, w in enumerate(ws):

        if i % (ratio + 1) == 0 or i % (ratio + 1) == 1:
            A = 1.0j * w * M - J
            lu = scipy.sparse.linalg.splu(A)

            ans = lu.solve(np.array(b))

            V = ans[-2]
            I = ans[-1]
            Z = V / I
            Zs.append(Z)

            if i != 0:
                if np.abs(Z - Zs[-2]) < 0.01:
                    ratio += 1
                    print(ratio)

        else:
            Z = 2 * Zs[-1] - Zs[-2]
            Zs.append(Z)

    return Zs, ws


def get_k(M):
    """
    Gets the block size from a block diagonal matrix

    Parameters
    ----------

    M : scipy sparse csr matrix
        Must be block diagonal. (Or near if using for approximation)

    Returns
    -------

    k : int
        the block size
    """
    for i in range(np.shape(M)[0]):
        if M[i, 0] == 0:
            if all(M[i, j] == 0 for j in range(i)):
                k = int(i)
                break
    return k


def get_block_diagonals(M, J, k):
    """
    Converts a scipy sparse csr matrix to a block diagonal storage form

    Parameters
    ----------

    M : scipy sparse csr matrix
        Must be block diagonal
    J : scipy sparse csr matrix
        Must be block tridiagonal
    k : int
        is the block size

    Returns
    -------

    M_diag : list
        list of all the block matrices on the diagonal of M

    (diag1, diag2, diag3) : tuple
        a tuple of lists of block matrices on the 3
        diagonals of J. 1 is below the main diagonal, 2 is the main diagonal, 3
        is above.
    """
    n = np.shape(M)[0]
    m = int(n / k)
    diag1 = []
    diag2 = []
    diag3 = []
    M_diag = []
    for i in range(m - 1):
        diag1.append(
            scipy.sparse.csr_matrix.todense(
                J[range(i * k, (i + 1) * k), :][:, range((i + 1) * k, (i + 2) * k)]
            ).astype("complex")
        )
        diag2.append(
            scipy.sparse.csr_matrix.todense(
                J[range(i * k, (i + 1) * k), :][:, range((i) * k, (i + 1) * k)]
            ).astype("complex")
        )
        diag3.append(
            scipy.sparse.csr_matrix.todense(
                J[range((i + 1) * k, (i + 2) * k), :][:, range((i) * k, (i + 1) * k)]
            ).astype("complex")
        )
        M_diag.append(
            scipy.sparse.csr_matrix.todense(
                M[range(i * k, (i + 1) * k), :][:, range((i) * k, (i + 1) * k)]
            ).astype("complex")
        )
    diag2.append(
        scipy.sparse.csr_matrix.todense(
            J[range((m - 1) * k, m * k), :][:, range((m - 1) * k, (m) * k)]
        ).astype("complex")
    )
    M_diag.append(
        scipy.sparse.csr_matrix.todense(
            M[range((m - 1) * k, m * k), :][:, range((m - 1) * k, (m) * k)]
        ).astype("complex")
    )
    return M_diag, (diag1, diag2, diag3)
