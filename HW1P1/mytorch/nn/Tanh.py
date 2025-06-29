from Activation import activation
import numpy as np

class tanh(activation):

    def forward(self,Z:np.ndarray)->np.ndarray:
        expZ=np.exp(Z)  #e^Z
        exp_Z=np.exp(np.negative(Z)) #e^(-Z)
        self.A=(expZ-exp_Z)/(expZ+exp_Z)
        return self.A
    
    def backward(self,dLdA:np.ndarray)->np.ndarray:
        dAdZ=1-(self.A * self.A)    #tanh'(x)=1-tanh^2(x) , * for element-wise product
        dLdZ=dLdA * dAdZ
        return dLdZ