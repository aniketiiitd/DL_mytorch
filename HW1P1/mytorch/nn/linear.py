import numpy as np

class Linear:
    def __init__(self, num_in_features:int,num_out_features:int):

        ''' W : weight matrix of shape -> num_out_features x num_in_features
            B : bias matrix of shape -> num_out_features'''
        
        self.W=np.random.Generator.uniform(-0.5,0.5,(num_out_features,num_in_features))
        self.B=np.random.Generator.uniform(-0.5,0.5,(num_out_features,1))

    def forward(self,A:np.ndarray)->np.ndarray:

        '''A : input data matrix/previous layer's activation of shape -> batch_size x num_in_features
            Z : output data matrix from the layer of shape -> batch_size x num_out_features (num of neurons)'''
        
        self.A=A
        self.N=A.shape[0]
        Z= (A @ self.W.T) + self.B.T
        return Z
    
    def backward(self,dLdZ):

        '''dLdA : grad of Loss wrt to the layer's neurons' activations for all samples, shape -> N x num_out_features (num of neurons)
            dLdW : grad of Loss wrt to the layer's weights' for all samples, shape -> same as W
            dLdB : grad of Loss wrt to the layer's biases' for all samples, shape -> same as B
            
            returns the current layer's dLdA which acts as dLdZ for previous layer'''
        
        dLdA=dLdZ @ self.W
        dLdW=(dLdZ.T) @ self.A
        dLdB=(dLdZ.T) @ np.ones(shape=(dLdZ.shape[0],1))

        self.dLdW=dLdW
        self.dLdB=dLdB

        return dLdA
