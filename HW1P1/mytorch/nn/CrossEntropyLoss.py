from .Loss import loss
from .Softmax import softmax
import numpy as np

class cross_entropy_loss(loss):
    def forward(self, A, Y):
        self.Y=Y
        sftmax=softmax()
        sigA=sftmax.forward(A)
        self.sigA=sigA
        crossentropy= np.sum(- (Y*np.log(sigA)),axis=1)
        sum_crossentropy=np.sum(crossentropy,axis=0)
        mean_crossentropy=sum_crossentropy/A.shape[0]
        return mean_crossentropy
    
    def backward(self):
        return (self.sigA-self.Y)/self.Y.shape[0]