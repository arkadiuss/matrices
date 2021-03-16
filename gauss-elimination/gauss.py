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


def lu_without_pivoting(A):
    L = np.zeros(A.shape)
    M = np.zeros(A.shape)
    for i in range(A.shape[0]):
        L[i, i] = 1
    M[:,:] = A[:,:]
    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[0]):
            L[j, i] = M[j, i] / M[i, i]
            M[j] = M[j] - M[i] * M[j][i] / M[i][i]
    print(M)
    U = M
    return L, U


def backward_L(M):
    X = np.zeros((M.shape[0],))
    for i in range(0,M.shape[0]):
        s = 0
        for j in range(0, i):
            s = s + X[j] * M[i][j]
        X[i] = (M[i, -1] - s) / M[i, i]
    return X


def solve_LU(L, U, b):
    M = prepare(L, b)
    c = backward_L(M)
    M = prepare(U, c)
    return backward(M)


if __name__ == "__main__":
    A = np.random.rand(3, 3) # np.array([[4,5,6], [3,6,5], [3,3,3]])#
    b = np.array([1, 2, 3])
    print(A, b)
    print("With ones")
    x = gauss_with1(A, b)
    print(x, A @ x, b)
    print("\n\nWithout ones")
    x = gauss_without1(A, b)
    print(x, A @ x, b)
    print("\n\nGauss with pivoting")
    x = gauss_with_pivoting(A, b)
    print(x, A @ x, b)
    print("\n\nLU without pivoting")
    L, U = lu_without_pivoting(A)
    print("L matrix")
    print(L)
    print("U matrix")
    print(U)
    print("A")
    print(A)
    print("L@U")
    print(L @ U)
    print("LU solution")
    x = solve_LU(L, U, b)
    print(x, A @ x, b)
