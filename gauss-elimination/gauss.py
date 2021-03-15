import numpy as np


def prepare(A, b):
    M = np.zeros((A.shape[0], A.shape[1] + 1))
    M[:, :-1] = A
    M[:, -1] = b
    return M


def backward(M):
    X = np.zeros((M.shape[0],))
    for i in range(M.shape[0] - 1, -1, -1):
        s = 0
        for j in range(i + 1, M.shape[0]):
            s = s + X[j] * M[i][j]
        X[i] = (M[i, -1] - s) / M[i, i]
    return X


def gauss_with1(A, b):
    M = prepare(A, b)
    for i in range(M.shape[0]):
        M[i] = M[i] / M[i][i]
        for j in range(i + 1, M.shape[0]):
            M[j] = M[j] - M[i] * M[j][i]
    print(M)
    return backward(M)


def gauss_without1(A, b):
    M = prepare(A, b)
    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[0]):
            M[j] = M[j] - M[i] * M[j][i] / M[i][i]
    print(M)
    return backward(M)


def pivoting(M, index):
    max_value = M[index, index]
    max_index = index
    for i in range(index+1, M.shape[0]):
        if M[i, index] > max_value:
            max_value = M[i, index]
            max_index = i

    if max_index != index:
        # M[max_index, :], M[index, :] = M[index, :], M[max_index, :]
        tmp = M[max_index, :].copy()
        M[max_index, :] = M[index, :]
        M[index, :] = tmp
    return M


def gauss_with_pivoting(A, b):
    M = prepare(A, b)
    for i in range(M.shape[0]):
        pivoting(M, i)
        for j in range(i+1, M.shape[0]):
            M[j] = M[j] - M[i] * M[j, i] / M[i, i]
    print(M)
    return backward(M)


def lu_without_pivoting(A, b):
    M = prepare(A, b)
    L = np.zeros(A.shape)
    for i in range(A.shape[0]):
        L[i, i] = 1

    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[0]):
            L[j, i] = M[j, i] / M[i, i]
            M[j] = M[j] - M[i] * M[j][i] / M[i][i]
    print(M)
    return backward(M), L


if __name__ == "__main__":
    A = np.random.rand(3, 3)
    b = np.array([1, 2, 3])
    print(A, b)
    print("First")
    x = gauss_with1(A, b)
    print(x, A @ x, b)
    print("Second")
    x = gauss_without1(A, b)
    print(x, A @ x, b)
    print("Gauss with pivoting")
    x = gauss_with_pivoting(A, b)
    print(x, A @ x, b)
    print("LU without pivoting")
    x, L = lu_without_pivoting(A, b)
    print(x, A @ x, b)
    print("L matrix")
    print(L)
