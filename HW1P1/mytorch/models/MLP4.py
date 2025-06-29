from ..nn.Linear import linear
from ..nn.Sigmoid import sigmoid
import numpy as np

class mlp4:
    def __init__(self):
        self.layers=[]
        dims=[2,4,8,8,4,2]

        for i in range(1,len(dims)):
            self.layers.append([linear(dims[i-1],dims[i]),sigmoid()])
        
        print(self.layers)

    def forward(self,A):
        layers=self.layers
        for i in range(len(self.layers)):
            Z=layers[i][0].forward(A)
            A=layers[i][1].forward(Z)

        return A
    
    def backward(self,dLdA):
        layers=self.layers
        for i in range(len(self.layers)-1,-1,-1):
            dLdZ=layers[i][1].backward(dLdA)
            dLdA=layers[i][0].backward(dLdZ)

        return dLdA