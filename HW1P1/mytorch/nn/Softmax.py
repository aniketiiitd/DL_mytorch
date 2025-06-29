from Activation import activation
import numpy as np

class softmax(activation):
    def forward(self,Z:np.ndarray)->np.ndarray:
        expZ=np.exp(Z)  #exp(Z)
        sum_expZ = np.sum(expZ,axis=1,keepdims=True)
        self.A = expZ/(sum_expZ)
        return self.A

    def backward(self, dLdA:np.ndarray)->np.ndarray:
        C=self.A.shape[1]
        J=np.zeros(shape=(C,C))
        list=[]
        for r in range(self.A.shape[0]):
            #print(r)
            A_rth_row=self.A[r,:]
            J.fill(0)
            for i in range(C):
                for j in range(C):
                    if(i==j):
                        J[i,j]=A_rth_row[i]*(1-A_rth_row[i])
                    else:
                        J[i,j]= - (A_rth_row[i] * A_rth_row[j])
            
            #print(J)
            dLdZ_rth_row=dLdA[r,:] @ J
            #print(dLdZ_rth_row.shape)
            list.append(dLdZ_rth_row)
        
        dLdZ=np.vstack(list)
        #print(dLdZ.shape)
        #print(dLdZ)
        return dLdZ


#sft=softmax()
#sft.forward(Z=[[1,3],[0,-1],[4,-2]])
#sft.backward(np.array([[1,2],[0.5,-3],[0,6]]))

Z = np.array([[1.0, 2.0, 3.0],
            [0.5, 1.5, -1.0]])

s = softmax()
A = s.forward(Z)
print("Softmax output:", A)

dLdA = np.zeros_like(A)
dLdA[:, 0] = 1.0  # derivative of sum(A[:,0]) wrt A

grad_custom = s.backward(dLdA)
#dLdZ = s.backward(dLdA)
print("Gradient w.r.t. input Z:", grad_custom)