import numpy as np

def prepare(A,b):
    M = np.zeros((A.shape[0],A.shape[1]+1))
    M[:,:-1] = A
    M[:,-1] = b
    return M

def backward(M):
    X = np.zeros((M.shape[0],))
    for i in range(M.shape[0]-1, -1, -1):
        s = 0
        for j in range(i+1, M.shape[0]):
            s = s + X[j]*M[i][j] 
        X[i] = (M[i,-1] - s)/M[i,i]
    return X

def gauss_with1(A, b):
    M = prepare(A,b)
    for i in range(M.shape[0]):
        M[i] = M[i] / M[i][i]
        for j in range(i+1, M.shape[0]):
            M[j] = M[j] - M[i]*M[j][i]
    print(M)
    return backward(M)

def gauss_without1(A, b):
    M = prepare(A,b)
    for i in range(M.shape[0]):
        for j in range(i+1, M.shape[0]):
            M[j] = M[j] - M[i]*M[j][i]/ M[i][i]
    print(M)
    return backward(M)

if __name__ == "__main__":
    A = np.random.rand(3,3)
    b = np.array([1,2,3])
    print(A,b)
    print("First")
    x = gauss_with1(A,b)
    print(x, A@x, b)
    print("Second")
    x = gauss_without1(A,b)
    print(x, A@x, b)