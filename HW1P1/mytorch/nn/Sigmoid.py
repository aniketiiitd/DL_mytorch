from .Activation import activation
import numpy as np

class sigmoid(activation):

    def forward(self,Z:np.ndarray)->np.ndarray:
        self.A=1/(1+np.exp(np.negative(Z)))
        return self.A
    
    def backward(self,dLdA:np.ndarray)->np.ndarray:
        dAdZ=self.A-(self.A * self.A)   #sig'(x) =sig'(x)[1-sig'(x)], * for element-wise product
        dLdZ=dLdA * dAdZ
        return dLdZ
