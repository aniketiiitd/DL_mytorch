from Activation import activation
import numpy as np

class relu(activation):

    def forward(self,Z:np.ndarray)->np.ndarray:
        self.A=np.maximum(Z,np.zeros(shape=Z.shape))
        return self.A

    def backward(self, dLdA:np.ndarray)->np.ndarray:
        dAdZ=np.where(self.A>0,1,0)
        dLdZ=dLdA * dAdZ
        return dLdZ