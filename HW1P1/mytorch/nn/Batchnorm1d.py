import numpy as np
class batchnorm1d:
    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))
        self.num_feat=num_features

    def forward(self,Z:np.ndarray,eval:bool=False)->np.ndarray:
        """
        Args:
            eval (bool,default=False):  indicates whether we are in the evaluating/inference phase of the problem or not (training phase).
            Z (np.ndarray,default=None): The input to the layer
        """

        assert self.num_feat==Z.shape[1], "num_features of batchnorm layer is different from num_features given in input"
        if eval==False:
            # training mode
            self.Z = Z
            self.N = Z.shape[0] #batch size
            self.M = np.sum(Z,axis=0,keepdims=True)/self.N
            self.V = np.sum(np.square(Z-self.M),axis=0,keepdims=True)/self.N
            #print(self.M,self.V)
            self.NZ = (Z-self.M)/(self.V+self.eps)
            #print(self.NZ)
            self.BZ = (self.NZ*self.BW)+self.Bb
            #print(self.BZ)
            self.running_M = (self.alpha*self.running_M)+((1-self.alpha)*self.M)
            self.running_V = (self.alpha*self.running_V)+((1-self.alpha)*self.V)
            #print(self.running_M.shape,self.running_V.shape)
        else:
            # inference mode
            self.NZ = (Z-self.running_M)/(self.running_V+self.eps)
            #print(self.NZ)
            self.BZ = (self.NZ*self.BW)+self.Bb
            #print(self.BZ)

        return self.BZ
    
bn1=batchnorm1d(num_features=5)
arr=np.random.rand(2,5)
#print(arr)
bn1.forward(arr,True)