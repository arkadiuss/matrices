import numpy as np

# standard block mul

def std_mul(A,B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i,j] += A[i,k] * B[k,j]
    return C


def block_mul(A, B, m, fmul):
    if(A.shape[1] != B.shape[0]):
        raise Exception("Can't do that")
    
    n = int(A.shape[0]/m)
    l = int(A.shape[1]/m)
    k = int(B.shape[1]/m)

    Ablk = np.zeros((n,l,m,m)) 
    Bblk = np.zeros((l,k,m,m))
    for i in range(n):
        for j in range(l):
            Ablk[i,j,:,:] = A[i*m:(i+1)*m,j*m:(j+1)*m]

    for i in range(l):
        for j in range(k):
            Bblk[i,j,:,:] = B[i*m:(i+1)*m,j*m:(j+1)*m]

    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(n):
        for j in range(k):
            for t in range(l):
                C[i*m:(i+1)*m,j*m:(j+1)*m] += std_mul(Ablk[i,t], Bblk[t,j])
                
    return C

# svd block mul
class SvdBlock():
    def __init__(self, U,V):
        self.U = U
        self.V = V

    def __mul__(self, B):
        tmp = self.V@B.U
        return SvdBlock(self.U@tmp, B.V)
    
    def __add__(self, B):
        u = np.hstack((self.U, B.U))
        v = np.vstack((self.V, B.V))
        return SvdBlock(u,v)

def svd_blockize(Ablk, eps=0.1):
    C = []
    for i in range(Ablk.shape[0]):
        rC = []
        for j in range(Ablk.shape[1]):
            U,S,V = np.linalg.svd(Ablk[i,j])
            # print(S) #uncomment to see if epsilon applies
            k=0
            while k<S.shape[0] and S[k]>=eps:
                k += 1
            u = U[:, :k]@np.diag(S[:k])
            v = V[:k,:]
            rC.append(SvdBlock(u,v))
        C.append(rC)
    return C


def block_svd_mul(A, B, m, eps):
    if(A.shape[1] != B.shape[0]):
        raise Exception("Can't do that")
    
    n = int(A.shape[0]/m)
    l = int(A.shape[1]/m)
    k = int(B.shape[1]/m)

    Ablk = np.zeros((n,l,m,m)) 
    Bblk = np.zeros((l,k,m,m))
    for i in range(n):
        for j in range(l):
            Ablk[i,j,:,:] = A[i*m:(i+1)*m,j*m:(j+1)*m]

    for i in range(l):
        for j in range(k):
            Bblk[i,j,:,:] = B[i*m:(i+1)*m,j*m:(j+1)*m]

    Asvd = svd_blockize(Ablk,eps)
    Bsvd = svd_blockize(Bblk,eps)   

    Cb = []
    for i in range(n):
        rCb = []
        for j in range(k):
            sblk = Asvd[i][0]*Bsvd[0][j]
            for t in range(1, l):
                sblk += Asvd[i][t]*Bsvd[t][j]    
            rCb.append(sblk)   
        Cb.append(rCb)
    
    C = np.zeros((n*m,k*m))

    for i in range(n):
        for j in range(k):
            C[i*m:(i+1)*m, j*m:(j+1)*m] = Cb[i][j].U@Cb[i][j].V
    return C

if __name__ == '__main__':
    # A = np.array([[0,0,0,1], [0,1,0,0], [0,0,1,0], [1,0,0,0]])
    # B = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
    A = np.array([[3,5,2,1], [2,1,4,5], [2,1,4,5], [9,4,2,1]])
    B = np.array([[1,2,3,4], [5,6,7,8], [12,11,10,9], [13,14,15,16]])
    m = 2 # block size
    print(A@B)

    print(block_mul(A,B,m, std_mul))

    print(block_svd_mul(A,B,m, 0.01))