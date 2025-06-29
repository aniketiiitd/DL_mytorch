from ..nn.Linear import linear
from ..nn.Sigmoid import sigmoid
import numpy as np

class mlp1:
    def __init__(self):
        self.layers=[]
        layer0=linear(2,3)
        f0,f1=sigmoid(),sigmoid()
        layer1=linear(3,1)

        self.layers.extend([[layer0,f0],[layer1,f1]])
        #print(self.layers)

    
    def forward(self,A:np.ndarray)->np.ndarray:
        layers=self.layers
        for i in range(len(self.layers)):
            Z=layers[i][0].forward(A)
            A=layers[i][1].forward(Z)

        return A
    
    def backward(self,dLdA:np.ndarray)->np.ndarray:
        layers=self.layers
        for i in range(len(self.layers)-1,-1,-1):
            dLdZ=layers[i][1].backward(dLdA)
            dLdA=layers[i][0].backward(dLdZ)

        return dLdA

