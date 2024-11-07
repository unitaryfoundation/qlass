import numpy as np
from numba import njit

from typing import List, Tuple, Dict, Union

def pauli_string_to_matrix(pauli_string: str) -> np.ndarray:
    res = 1.0

    # Define the Pauli matrices.
    I = np.eye(2)
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0, -1.0j],[1.0j, 0]])
    Z = np.array([[1,0],[0,-1]])

    for c in pauli_string:
        if c=='X':
            res = np.kron(res, X)
        elif c=='Y':
            res = np.kron(res, Y)
        elif c=='Z':
            res = np.kron(res, Z)
        else:
            res = np.kron(res, I)
        
    return res

def hamiltonian_matrix(H: Dict[str, float]) -> np.ndarray:

    coeffs = list(H.values())
    matrices = [coeff*pauli_string_to_matrix(pauli_string) for pauli_string, coeff in zip(H.keys(), coeffs)]

    return sum(matrices)


def brute_force_minimize(H: Dict[str, float]) -> np.ndarray:

    H_matrix = hamiltonian_matrix(H)
    l0 = np.linalg.eigvals(H_matrix)
    l0.sort()

    return l0[0]


@njit
def Lanczos( A, v, m=100):
    n = len(v)
    if m>n: m = n;
    # from here https://en.wikipedia.org/wiki/Lanczos_algorithm
    V = np.zeros( (m,n) , dtype = np.complex128)
    T = np.zeros( (m,m) , dtype = np.complex128)
    V[0, :] = v

    # step 2.1 - 2.3 in https://en.wikipedia.org/wiki/Lanczos_algorithm
    w = np.dot(A, v)
    alfa = np.dot(np.conj(w),v)
    w = w - alfa*v
    T[0,0] = alfa

    # needs to start the iterations from indices 1
    for j in range(1, m-1 ):
        # beta = np.sqrt( np.abs( np.dot( np.conj(w), w ) ) )
        beta = np.linalg.norm(w)

        V[j,:] = w/beta

        # This performs some rediagonalization to make sure all the vectors are orthogonal to eachother
        for i in range(j-1):
            V[j, :] = V[j,:] - np.dot(V[j,:], np.conj(V[i, :]))*V[i,:]
        V[j, :] = V[j, :]/np.linalg.norm(V[j, :])


        w = np.dot(A, V[j, :])
        alfa = np.dot(np.conj(w), V[j, :])
        w = w - alfa * V[j, :] - beta*V[j-1, :]

        T[j,j  ] = alfa
        T[j-1 ,j] = beta
        T[j,j-1] = beta


    return T, V

def eig_decomp_lanczos(R, n=1, m=100):        
    v0   = np.array(np.random.rand( np.shape(R)[0]) , dtype=np.complex128); v0 /= np.sqrt( np.abs(np.dot( v0, np.conjugate(v0) ) ) )

    T, V = Lanczos(R, v0, m=m )
    esT, vsT = np.linalg.eigh( T )
    esT_sort_idx = np.argsort(esT)[::-1]
    lm_eig = np.matrix(V.T @ (vsT[:, esT_sort_idx[:n].squeeze() ]))

    return esT
