import numpy as np
class flatten():
    def forward(self,A:np.ndarray)->np.ndarray:
        #print(A)
        self.A_shape=A.shape
        return np.reshape(A,(A.shape[0],A.shape[1]*A.shape[2]))
    def backward(self,dLdZ:np.ndarray)->np.ndarray:
        return np.reshape(dLdZ,(self.A_shape))
    
#obj=flatten()
#Z=obj.forward(np.random.rand(2,3,4))
#print(Z)
#print(obj.backward(Z))