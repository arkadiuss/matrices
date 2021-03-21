import numpy as np


def det(A):
    n = A.shape[0]
    if n == 1:
        return A.copy().flat[0]
    S = 0
    column_mask = np.ones(n, dtype=bool)
    for i in range(n):
        fact = 1 if i % 2 == 0 else -1
        column_mask[i] = False
        S = S + fact * A[0, i] * det(A[1:, column_mask])
        column_mask[i] = True
    return S


def cof(A):
    n = A.shape[0]
    if n == 1:
        return A[0, 0]
    row_mask = np.ones(n, dtype=bool)
    column_mask = np.ones(n, dtype=bool)
    C = np.zeros(A.shape)
    for i in range(n):
        for j in range(n):
            fact = 1 if (i + j) % 2 == 0 else -1
            row_mask[i] = False
            column_mask[j] = False
            C[i, j] = fact * det(A[row_mask, :][:, column_mask])
            column_mask[j] = True
            row_mask[i] = True
    return C


def adj(A):
    return cof(A).T


def inv(A):
    return 1.0 / det(A) * adj(A)


def matrix_p_norm(A, p=1):
    return np.max(np.sum(np.abs(A) ** p, axis=0) ** (1/p))


def matrix_norm_inf(A):
    return np.max(np.sum(np.abs(A), axis=1))


def cond(A, p=1):
    return matrix_p_norm(A, p) * matrix_p_norm(inv(A), p)


def cond_mag(A, k=10000000, p=1):
    vs = np.random.randn(A.shape[0], k)  # k random vectors
    norm = np.sum(np.abs(vs) ** p, axis=0) ** (1 / p)  # norm of each of random vectors
    norm_vs = vs / norm  # normalized random vectors
    Avs = A @ norm_vs  # apply A to each vector
    normAvs = np.sum(np.abs(Avs) ** p, axis=0) ** (1 / p)  # norm of each vector after applying A
    maxmag = np.max(normAvs)
    minmag = np.min(normAvs)
    return maxmag / minmag


if __name__ == '__main__':
    A = np.array([[1, 2], [3, 4]])
    # A = np.array([[4, 9, 2], [3, 5, 6], [8, 1, 6]])
    # A = np.array([[2, 2, 3], [4, 5, 6], [7, 8, 9]])
    # A = np.array([[999, 998], [1000, 999]])
    print("Correct value:")
    print(cond(A))
    print("Estimated value:")
    print(cond_mag(A))
