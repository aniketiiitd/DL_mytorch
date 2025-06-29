from ..nn.Linear import linear
from ..nn.Sigmoid import sigmoid
import numpy as np

class mlp0:
    def __init__(self):
        self.layer0=linear(2,3)
        self.f0=sigmoid()

    def forward(self,A0:np.ndarray)->np.ndarray:
        Z0=self.layer0.forward(A0)
        A1=self.f0.forward(Z0)
        return A1
    
    def backward(self,dLdA1:np.ndarray)->np.ndarray:
        dLdZ0=self.f0.backward(dLdA1)
        dLdA0=self.layer0.backward(dLdZ0)
        return dLdA0
    



        
