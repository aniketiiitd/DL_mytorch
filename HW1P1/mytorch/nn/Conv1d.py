import numpy as np
from Resampling import downsampling
import sys

def weight_init_fn(out_chn:int,in_chn:int,kernel_size:tuple[int])->np.ndarray:

    """initializes random weights (kernels) of dim: num_out_channels x num_in_channels x filter_size"""
    #np.random.seed(42) #Seed for testing purpose
    return(np.random.random(size=(out_chn,in_chn,kernel_size)))

def bias_init_fn(out_chn:int)->np.ndarray:
    """initalizes random bias of dim: num_out_channels"""
    #np.random.seed(42) #Seed for testing purpose
    return(np.random.random(size=(out_chn)))

class conv1d_stride1:

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, wt_init_fn=weight_init_fn, bs_init_fn=bias_init_fn):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.W = wt_init_fn(out_channels, in_channels, kernel_size)
        self.B = bs_init_fn(out_channels)
        self.dLdW = np.zeros(self.W.shape)
        self.dLdB = np.zeros(self.B.shape)
        self.flipped_W=np.flip(self.W,axis=2)
        #self.out_channel_size=in_channels-kernel_size+1

    def convolve(self,Y:np.ndarray, fil:np.ndarray)->np.ndarray:
        """finds convolution (∗) of an output_channel with the corresponding filter """
        A=[]
        size=len(fil)
        ndim=Y.ndim
        for i in range(len(Y)-size+1):
            A.append(np.tensordot(Y[i:i+size],fil,ndim))

        return np.array(A)


    def forward(self, A:np.ndarray)->np.ndarray:

        """ A is an array of all input channels for N samples, 
            the method returns an array of all the output channels, 
            shape of A: N x C_in x W_in """
        
        assert A.ndim==3, "A should have exactly 3 dimensions"

        Z=[]
        self.A=A
        for n in range(len(A)):
            Zn=[]
            for i in range(self.out_channels):
                Zi=np.zeros(shape=(len(A[0,0])-self.kernel_size+1))
                #print(Zi)
                #sys.exit()
                ith_bias=self.B[i]

                for j in range(self.in_channels):
                    jth_filter=self.W[i,j]
                    Zi+=(self.convolve(self.A[n,j],jth_filter))
                    #print(Zi)
                    #sys.exit()
                Zi+=ith_bias
                Zn.append(Zi)
        
            Z.append(Zn)
        
        return np.array(Z)

    def backward(self,dLdZ:np.ndarray)->np.ndarray:
        """shape of dLdZ: N x C_out x W_out"""
        assert dLdZ.ndim==3, "dLdZ should have exactly 3 dimensions"

        dLdA=np.zeros(self.A.shape)
        padded_dLdZ=np.copy(dLdZ)

        for _ in range(self.kernel_size-1):
            padded_dLdZ=np.insert(padded_dLdZ,values=0,obj=(0,len(padded_dLdZ[0,0])),axis=2)
        #print(padded_dLdZ)
        #print(self.W)
        for n in range(len(dLdZ)):
            for j in range(len(dLdZ[n])):
                self.dLdB[j]+=np.sum(dLdZ[n,j])
                for k in range(len(self.A[n])):
                    
                    self.dLdW[j,k]+=self.convolve(self.A[n,k],dLdZ[n,j])
                    dLdA[n,k]+=self.convolve(padded_dLdZ[n,j],self.flipped_W[j,k])
                    #print(self.A[n,k])
                    #sys.exit()
                
        self.dLdB=self.dLdB/len(dLdZ)
        self.dLdW=self.dLdW/len(dLdZ)

        #print("dLdB:",self.dLdB,'\n',"dLdW:",self.dLdW)
        #print(self.W.shape==self.dLdW.shape, self.B.shape==self.dLdB.shape, self.A.shape==dLdA.shape)
        #print(dLdA)
        
        return dLdA
#print(convolve(np.array([1,2,0,-1,-1]),np.array([1,2])))


#print("forward pass result:",Z)

#obj.backward(Z)

class conv1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.padding = padding
        self.conv1d_stride1 = conv1d_stride1(in_channels,out_channels,kernel_size)
        self.downsample1d = downsampling(downsampling_factor=stride)

    def forward(self,A:np.ndarray)->np.ndarray:

        #print(A.shape)
        #Padding with zeros
        for i in range(self.padding):
            A=np.insert(A,values=0,obj=(0,len(A[0,0])),axis=2)
            #A=np.insert(A,values=0,obj=(0,len(A[0])),axis=1)
        
        #print(A.shape)
        #conv1d stride1 forward
        Z=self.conv1d_stride1.forward(A)
        #print(Z.shape)
        #downsampling to mimic stride>1
        downsampled_Z=[]
        for sample_ind in range(len(A)):        #! For loop is needed because all methods in resampling file assume only one sample (in the form of a 2d array) as arguments to their parameters (A or dLdZ)
            downsampled_Z.append(self.downsample1d.forward(Z[sample_ind]))
        
        #print(Z)
        #print(np.array(downsampled_Z))
        
        Z=np.array(downsampled_Z)
        #print(Z.shape)
        return Z
    
    def backward(self,dLdZ:np.ndarray)->np.ndarray:
        
        #reverse downsampling (i.e. upsampling)
        downsampled_dLdZ_backward=[]
        for sample_ind in range(len(dLdZ)):
            downsampled_dLdZ_backward.append(self.downsample1d.backward(dLdZ[sample_ind]))
        dLdZ=np.array(downsampled_dLdZ_backward)
        
        #print(dLdZ.shape)
        #Conv1d stride_1 backward
        dLdA=self.conv1d_stride1.backward(dLdZ)
        
        #print(dLdA.shape)
        
        #Unpadding
        if self.padding>0:
            dLdA=dLdA[:,:,self.padding:-self.padding]
        #print(dLdA.shape)

        return dLdA

#obj=conv1d(4,out_channels=3,kernel_size=2,stride=1,padding=0)
#print("Initialized weights:",obj.W)
#Z=(obj.forward(np.array([[[1,0,0,-1,2],[1,-1,0,-1,1],[1,0,1,-1,2],[1,2,3,4,5]],[[1,0,0,-1,2],[1,-1,0,-1,1],[1,0,1,-1,2],[1,2,3,4,5]]])))
#print(Z.shape)
#print(obj.backward(Z).shape)