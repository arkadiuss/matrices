import numpy as np
from numpy.linalg import norm


def power_method(A, eps):
    vec = np.random.rand(A.shape[0])
    vec = vec / norm(vec)
    matrix = A @ A.T
    while True:
        prev_vec = vec
        vec = matrix @ prev_vec
        vec = vec / norm(vec)
        if norm(vec - prev_vec) < eps:
            return vec


def custom_svd(A, eps=0.01):
    values = []
    matrix = A
    for i in range(A.shape[0]):
        
        if i != 0:
            u,s,v = values[-1]
            tmp = np.outer(u, v)
            matrix -= s * tmp

        u = power_method(matrix, eps)
        v_unormalized = A.T @ u
        s = norm(v_unormalized)
        v = v_unormalized / s
        values.append((u, s, v))
    u, s, v = [np.array(x) for x in zip(*values)]
    return u.T, s, v


if __name__ == '__main__':
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype='float32')
    # A = np.array([[1, 2, 0], [2, 0, 2]], dtype='float32')

    u, s, v = custom_svd(A, 0.01)
    np.set_printoptions(suppress=True, precision=3)

    print(u @ np.diag(s) @ v)

    print("\nLeft Singular Vectors:")
    print(u)

    print("\nSingular Values:")
    print(np.diag(s))

    print("\nRight Singular Vectors:")
    print(v)
