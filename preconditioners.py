#
# Preconditioners
#
import numpy as np
import time
from scipy.sparse import tril, triu, eye, bmat
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
            np.array(
                J[range(i * k, (i + 1) * k), :][:, range((i + 1) * k, (i + 2) * k)],
                dtype="complex",
            )
        )
        diag2.append(
            np.array(
                J[range(i * k, (i + 1) * k), :][:, range((i) * k, (i + 1) * k)],
                dtype="complex",
            )
        )
        diag3.append(
            np.array(
                J[range((i + 1) * k, (i + 2) * k), :][:, range((i) * k, (i + 1) * k)],
                dtype="complex",
            )
        )
        M_diag.append(
            np.array(
                M[range(i * k, (i + 1) * k), :][:, range((i) * k, (i + 1) * k)],
                dtype="complex",
            )
        )
    diag2.append(
        np.array(
            J[range((m - 1) * k, m * k), :][:, range((m - 1) * k, (m) * k)],
            dtype="complex",
        )
    )
    M_diag.append(
        np.array(
            M[range((m - 1) * k, m * k), :][:, range((m - 1) * k, (m) * k)],
            dtype="complex",
        )
    )
    return M_diag, (diag1, diag2, diag3)


def ILUpreconditioner(diag1, diag2, diag3):
    """
    Creates an ILU preconditioner for A. A is block tridiagonal.
    We have A equals approximately LU.

    Parameters
    ----------
    diag1 : list of matrices
        The lower diagonal of A.
    diag2 : list of matrices
        The main diagonal of A
    diag3 : list of matrices
        The upper diagonal of A

    Returns
    -------
    L : scipy sparse csr matrix
        A lower triangular matrix
    U : scipy sparse csr matrix
        An upper triangular matrix

    """
    m = len(diag2)

    k = np.shape(diag1[0])[0]
    Si = diag2[0]
    Ss = [Si]
    Ti = np.linalg.solve(Si, diag3[0])
    Ts = [Ti]

    prev = []

    for i in range(2, m + 1):
        current = [diag2[i - 1], diag1[i - 2], diag2[i - 2], diag3[i - 3], diag3[i - 2]]
        if current == prev:
            Ss.append(Si)
            Ts.append(Ti)
        else:
            M = np.linalg.solve(current[2], current[3])
            H = current[1] @ M
            G = np.linalg.solve(current[0], H)
            F = np.eye(k) + 2 * G
            Si = current[0] @ np.linalg.inv(F)
            Ss.append(Si)

            if i != m:
                Ti = np.linalg.solve(Si, current[4])
                Ts.append(Ti)
                prev = current

    L = bmat(
        [
            [Ss[i] if i == j else diag1[j] if i - j == 1 else None for j in range(m)]
            for i in range(m)
        ],
        format="csr",
    )
    U = bmat(
        [
            [np.eye(k) if i == j else Ts[i] if i - j == -1 else None for j in range(m)]
            for i in range(m)
        ],
        format="csr",
    )

    return L, U


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

        L, U = ILUpreconditioner(A_diags[0], A_diags[1], A_diags[2])
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

    L = splu(A)
    U = None

    return L, U
