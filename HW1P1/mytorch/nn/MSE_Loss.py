from .Loss import loss
import numpy as np

class mse_loss(loss):
    def forward(self,A:np.ndarray,Y:np.ndarray)->np.float64:
        self.A=A
        self.Y=Y
        se=np.square(A-Y)
        sse=np.sum(se)
        mse=sse/(A.shape[0]*A.shape[1])
        self.loss=mse
        return mse
    
    def backward(self):
        dLdA=2*(self.A-self.Y)/(self.A.shape[0]*self.A.shape[1])
        return dLdA