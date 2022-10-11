#
# Preconditioners
#
import numpy as np
import numerical_methods as nm
import time
from scipy.sparse import tril, triu, eye, csr_matrix
from scipy.sparse.linalg import splu


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
            csr_matrix.todense(
                J[range(i * k, (i + 1) * k), :][:, range((i + 1) * k, (i + 2) * k)]
            ).astype("complex")
        )
        diag2.append(
            csr_matrix.todense(
                J[range(i * k, (i + 1) * k), :][:, range((i) * k, (i + 1) * k)]
            ).astype("complex")
        )
        diag3.append(
            csr_matrix.todense(
                J[range((i + 1) * k, (i + 2) * k), :][:, range((i) * k, (i + 1) * k)]
            ).astype("complex")
        )
        M_diag.append(
            csr_matrix.todense(
                M[range(i * k, (i + 1) * k), :][:, range((i) * k, (i + 1) * k)]
            ).astype("complex")
        )
    diag2.append(
        csr_matrix.todense(
            J[range((m - 1) * k, m * k), :][:, range((m - 1) * k, (m) * k)]
        ).astype("complex")
    )
    M_diag.append(
        csr_matrix.todense(
            M[range((m - 1) * k, m * k), :][:, range((m - 1) * k, (m) * k)]
        ).astype("complex")
    )
    return M_diag, (diag1, diag2, diag3)


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
    L = tril(A, format="csr")
    U = None
    return L, U


def GSV(A, M, J, L, U, b=None):
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

    U = tril(A, format="csr")
    L = triu(A, k=1, format="csr")
    Id = eye(A.shape[0], dtype="complex", format="csr")
    L = L + Id
    return L, U


def ELU(A, M, J, L, U, b):
    """
    A preconditioner method using exact factorisations to be used with
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

    try:
        t = et - st
    except:
        pass

    if type(L) == str or LUt <= t:
        LUstart_time = time.time()
        L = splu(A.tocsc())
        start_point = L.solve(b)
        U = None
        LUend_time = time.time()
        LUt = LUend_time - LUstart_time

    st = time.time()

    return L, U
