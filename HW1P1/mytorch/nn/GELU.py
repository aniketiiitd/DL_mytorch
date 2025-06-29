from Activation import activation
import numpy as np
from scipy.special import erf
from math import sqrt,pi

class gelu(activation):

    def forward(self,Z:np.ndarray)->np.ndarray:
        self.Z=Z
        self.var=(1/2)*(1+erf(Z/(sqrt(2))))
        self.A=(Z*self.var)     #A=Z*phi(Z)
        return self.A
    
    def backward(self, dLdA:np.ndarray)->np.ndarray:
        dAdZ=self.var + (self.Z/sqrt(2*pi)) * np.exp(np.negative(np.square(self.Z)/2))
        dLdZ=dLdA * dAdZ
        return dLdZ

#print(np.arange(5)/sqrt(2))