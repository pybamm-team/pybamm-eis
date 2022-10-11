#
# Linear algebra methods
#

import numpy as np
import scipy.sparse
<<<<<<< HEAD


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
    rk = np.array(b) - A @ xk
    pk = rk

    max_num_iter = len(b)
    rk1rk1 = np.dot(np.conj(rk), rk)

=======

def empty(xk):
    #An empty callback function. Callbacks can be written as desired.
    pass

def conjugate_gradient(A, b, start_point = 'zero', callback = empty, tol = 10**-3):
    '''
    conjugate_gradient(A, b, start_point = 'zero', callback = empty, tol = 10**-3)
    
    Uses the conjugate gradient method to solve Ax = b
    
    Parameters
    ----------
    A : scipy sparse csr matrix
        A square matrix.
    b : numpy 1xn array
    start_point : numpy 1xn array, optional
        Where the iteration starts. The default is 'zero'.
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
        A tolerance at which to stop the iteration. The default is 10**-5.

    Returns
    -------
    xk : numpy 1xn array
        The solution to Ax = b.

    '''
    #Uses conjugate gradient algorithm
    #Should not be used unless A Hermitian. If A not hermitian, use BicgSTAB instead.
    #For best performance A should be a scipy csr sparse matrix
    if str(start_point) == 'zero':
        start_point = np.zeros(np.shape(b))
    
    xk = np.array(start_point)
    rk = np.array(b) - A@xk
    pk = rk
    
    max_num_iter = len(b)
    rk1rk1 = np.dot(np.conj(rk), rk)
    
    
>>>>>>> main
    for k in range(max_num_iter):
        Apk = A @ pk
        rkrk = rk1rk1
        pkApk = np.dot(np.conj(pk), Apk)

        alpha_k = rkrk / pkApk

        xk = xk + alpha_k * pk
<<<<<<< HEAD

        callback(xk)

        # Stop if the change in the last entry is under tolerance
        if alpha_k * pk[-1] < tol:
=======
        
        callback(xk)
        
        #Stop if the change in the last entry is under tolerance
        if alpha_k*pk[-1] < tol:
>>>>>>> main
            break
        else:
            rk = rk - alpha_k * Apk

            rk1rk1 = np.dot(np.conj(rk), rk)

            beta_k = rk1rk1 / rkrk

            pk = rk + beta_k * pk

    return xk


<<<<<<< HEAD
def bicgstab(A, b, start_point=None, callback=empty, tol=10**-3):
    """
    Uses the BicgSTAB method to solve Ax = b

=======
def bicgstab(A, b, start_point = 'zero', callback = empty, tol = 10**-3):    
    '''
    bicgstab(A, b, start_point = 'zero', callback = empty, tol = 10**-3)
    
    Uses the BicgSTAB method to solve Ax = b
    
>>>>>>> main
    Parameters
    ----------
    A : scipy sparse csr matrix
        A square matrix.
    b : numpy 1xn array
    start_point : numpy 1xn array, optional
<<<<<<< HEAD
        Where the iteration starts. If not provided the initial guess will be zero.
=======
        Where the iteration starts. The default is 'zero'.
>>>>>>> main
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
        A tolerance at which to stop the iteration. The default is 10**-3.
<<<<<<< HEAD

    Returns
    -------
    xk : numpy 1xn array
        The solution of Ax = b.

    """
    if start_point is None:
        start_point = np.zeros_like(b)
=======
>>>>>>> main

    Returns
    -------
    xk : numpy 1xn array
        The solution to Ax = b.

    '''
    
    if str(start_point) == 'zero':
        start_point = np.zeros(np.shape(b))
    
    
    xk = np.array(start_point)
    rk = np.array(b) - A @ xk
    r0 = np.conj(rk)
    rhok = 1
    alpha_k = 1
    wk = 1

    pk = np.zeros(np.shape(b))
    vk = pk
<<<<<<< HEAD
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
=======
    max_num_iter = 2*np.shape(b)[0]
    
    for k in range(1, max_num_iter+1):
        rhok1 = rhok
        rhok = np.dot(r0.T, rk)
        beta_k = (rhok/rhok1)*(alpha_k/wk)
        
        pk = rk + beta_k*(pk - wk*vk)
        vk = A@pk
        
        alpha_k = rhok/np.dot(r0.T, vk)
        
        h = xk + alpha_k*pk
        
        s = rk - alpha_k*vk
        t = A@s
        
        wk = np.dot(np.conj(t.T), s)/np.dot(np.conj(t.T), t)
        xk = h + wk*s
>>>>>>> main
        callback(xk)
        if np.linalg.norm(rk, 1) < tol:
            break
        else:
            rk = s - wk * t
    return xk

<<<<<<< HEAD

def prebicgstab(A, b, L, U=None, start_point=None, callback=empty, tol=1e-3):
    """
    Uses the preconditioned BicgSTAB method to solve Ax = b. The preconditioner
    is of the form LU, or just of the form L.

=======
def prebicgstab(A, b, L, U = 'none', start_point = 'zero', callback = empty, tol = 10**-3):
    '''
    prebicgstab(A, b, L, U, start_point = 'zero', callback = empty, tol = 10**-3)
    
    Uses the preconditioned BicgSTAB method to solve Ax = b. The preconditioner
    is of the form LU, or just of the form L.
    
>>>>>>> main
    Parameters
    ----------
    A : scipy sparse csr matrix
        A square matrix.
    b : numpy 1xn array
    L : scipy sparse csr matrix
        A lower (block) triangular matrix (or superLU object)
    U : scipy sparse crs matrix, optional
        An Upper (block) triangular matrix. None used if not given.
<<<<<<< HEAD
    start_point : numpy 1xn array, optional
        Where the iteration starts. If not provided the initial guess will be zero.
=======
    triangular: Specifies if the 2 above matrices are triangular. Note longer
    time will be taken if non-triangular.
    start_point : numpy 1xn array, optional
        Where the iteration starts. The default is 'zero'.
>>>>>>> main
    callback : function, optional
        a function callback(xk) that can be written to happen each iteration.
        The default is empty.
    tol : float, optional
<<<<<<< HEAD
        A tolerance at which to stop the iteration. The default is 1e-3.
=======
        A tolerance at which to stop the iteration. The default is 10**-3.
>>>>>>> main

    Returns
    -------
    xk : numpy 1xn array
<<<<<<< HEAD
        The solution of Ax = b.

    """
    if start_point is None:
        start_point = np.zeros_like(b)
=======
        The solution to Ax = b.
>>>>>>> main

    '''
    size = np.shape(b)
    
    if str(start_point) == 'zero':
        start_point = np.zeros(size)
        
    xk = np.array(start_point)
<<<<<<< HEAD
    rk = np.array(b) - A @ xk

    r0 = np.conj(rk)

=======
    rk = np.array(b) - A@xk
    
    r0 = np.conj(rk)
    
>>>>>>> main
    rhok = 1
    alpha_k = 1
    wk = 1

    pk = np.zeros_like(b)
    vk = pk
<<<<<<< HEAD

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
=======
    
    max_num_iter = 2*np.shape(b)[0]
    
    superLU = False
    if type(L) == scipy.sparse.linalg.SuperLU:
        superLU = True
    
    if type(U) != str:
        U_exists = True
    else:
        U_exists = False
        
    for k in range(1, max_num_iter+1):
        rhok1 = rhok
        rhok = np.dot(r0.T, rk)
        beta_k = (rhok/rhok1)*(alpha_k/wk)
        pk = rk + beta_k*(pk - wk*vk)
>>>>>>> main

        if superLU:
            y = np.array(L.solve(pk))
        else:
            y = scipy.sparse.linalg.spsolve(L, pk)
<<<<<<< HEAD
            if U:
                y = np.array(scipy.sparse.linalg.spsolve(U, y))

        y = np.reshape(y, np.shape(b))

        vk = A @ y
        alpha_k = rhok / np.dot(r0.T, vk)
=======
            if U_exists:
                y = np.array(scipy.sparse.linalg.spsolve(U, y))
        
        y = np.reshape(y, size)
        
        vk = A@y
        alpha_k = rhok/np.dot(r0.T, vk)
>>>>>>> main

        h = xk + alpha_k * y
        
        s = rk - alpha_k*vk    

<<<<<<< HEAD
        s = rk - alpha_k * vk

=======
>>>>>>> main
        if superLU:
            z = np.array(L.solve(s))
        else:
            z = scipy.sparse.linalg.spsolve(L, s)
<<<<<<< HEAD
            if U:
                z = np.array(scipy.sparse.linalg.spsolve(U, z))

        z = np.reshape(z, np.shape(b))

        t = A @ z

        wk = np.dot(np.conj(t.T), s) / np.dot(np.conj(t.T), t)
        xk = h + wk * z
=======
            if U_exists:
                z = np.array(scipy.sparse.linalg.spsolve(U, z))
                
        z = np.reshape(z, size)
        
        t = A@z
        
        wk = np.dot(np.conj(t.T), s)/np.dot(np.conj(t.T), t)
        xk = h + wk*z
>>>>>>> main
        callback(xk)
        if np.linalg.norm(rk, 1) < tol:
            break
        else:
            rk = s - wk * t
    return xk


def ILUpreconditioner(diag1, diag2, diag3):
<<<<<<< HEAD
    """
    Creates an ILU preconditioner for A. A is block tridiagonal.
    We have A equals approximately LU.

=======
    '''
    ILUpreconditioner(diag1, diag2, diag3)
    
    Creates an ILU preconditioner for A. A is block tridiagonal.
    
>>>>>>> main
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
<<<<<<< HEAD
        A lower triangular matrix
    U : scipy sparse csr matrix
        An upper triangular matrix

    """
    m = len(diag2)

=======
        A lower triangular matrix 
    U : scipy sparse csr matrix
        An upper triangular matrix 
        
        We have A equals approximately LU

    '''
    m = len(diag2)
    
>>>>>>> main
    k = np.shape(diag1[0])[0]
    Si = diag2[0]
    Ss = [Si]
    Ti = np.linalg.solve(Si, diag3[0])
    Ts = [Ti]
<<<<<<< HEAD

    prev = []

    for i in range(2, m + 1):
        current = [diag2[i - 1], diag1[i - 2], diag2[i - 2], diag3[i - 3], diag3[i - 2]]
=======
    
    prev = []
    
    for i in range(2, m+1):
        current = [diag2[i-1], diag1[i-2], diag2[i-2], diag3[i-3], diag3[i-2]]
>>>>>>> main
        if current == prev:
            Ss.append(Si)
            Ts.append(Ti)
        else:
            M = np.linalg.solve(current[2], current[3])
<<<<<<< HEAD
            H = current[1] @ M
            G = np.linalg.solve(current[0], H)
            F = np.eye(k) + 2 * G
            Si = current[0] @ np.linalg.inv(F)
            Ss.append(Si)

=======
            H = current[1]@M
            G = np.linalg.solve(current[0], H)
            F = np.eye(k) + 2*G
            Si = current[0]@np.linalg.inv(F)
            Ss.append(Si)
            
>>>>>>> main
            if i != m:
                Ti = np.linalg.solve(Si, current[4])
                Ts.append(Ti)
                prev = current

<<<<<<< HEAD
    L = scipy.sparse.bmat(
        [
            [Ss[i] if i == j else diag1[j] if i - j == 1 else None for j in range(m)]
            for i in range(m)
        ],
        format="csr",
    )
    U = scipy.sparse.bmat(
        [
            [np.eye(k) if i == j else Ts[i] if i - j == -1 else None for j in range(m)]
            for i in range(m)
        ],
        format="csr",
    )

    return L, U


def matrix_rescale(A, M, J, b):
    n = np.shape(M)[0]
    multiplier = np.ones(n, dtype="complex")

=======
    L = scipy.sparse.bmat([[Ss[i] if i == j else diag1[j] if i-j==1
                else None for j in range(m)]
                for i in range(m)], format='csr')
    U = scipy.sparse.bmat([[np.eye(k) if i == j else Ts[i] if i-j==-1
            else None for j in range(m)]
            for i in range(m)], format='csr')
    
    return L, U

def matrix_rescale(A, M, J, b):
    n = np.shape(M)[0]
    multiplier = np.ones(n, dtype = 'complex')
    
>>>>>>> main
    multiplier[-2] = 3
    multiplier[-1] = 2

    multiplier = scipy.sparse.diags(multiplier)
<<<<<<< HEAD
    M = multiplier @ M
    J = multiplier @ J
    b = multiplier @ b

    return M, J, b
=======
    M = multiplier@M
    J = multiplier@J
    b = multiplier@b

    return M, J, b
    
>>>>>>> main
