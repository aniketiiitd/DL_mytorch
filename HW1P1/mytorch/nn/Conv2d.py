import numpy as np
from Resampling import downsampling

def weight_init_fn(out_chn:int,in_chn:int,kernel_size:tuple[int])->np.ndarray:

    """initializes random weights (kernels) of dim: num_out_channels x num_in_channels x (kernel_size[0] x kernel_size[1]) assuming a square filter"""
    #np.random.seed(42) #Seed for testing purpose
    return(np.random.random(size=(out_chn,in_chn,kernel_size[0],kernel_size[1])))     #4d array is generated

def bias_init_fn(out_chn:int)->np.ndarray:
    """initalizes random bias of dim: num_out_channels"""
    #np.random.seed(42) #Seed for testing purpose
    return(np.random.random(size=(out_chn)))


class conv2d_stride1:

    def __init__(self, in_channels:int, out_channels:int, kernel_size:tuple[int], weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.W = weight_init_fn(out_channels, in_channels, kernel_size)
        self.B = bias_init_fn(out_channels)
        self.dLdW = np.zeros(self.W.shape)
        self.dLdB = np.zeros(self.B.shape)

    def convolve(self,Y:np.ndarray, fil:np.ndarray)->np.ndarray:
        """finds convolution (∗) of an output_channel with the corresponding filter """

        A=[]
        size=len(fil)   #assuming filter is a square, else need to find the x_size and y_size of the fil
        ndim_Y=Y.ndim

        for H in range(len(Y)-len(fil)+1):
            A_H=[]
            for B in range(len(Y[0])-len(fil[0])+1):
                A_H.append(np.tensordot(Y[H:H+size,B:B+size],fil,ndim_Y))
            A.append(A_H)

        return np.array(A)

    def forward(self, A:np.ndarray)->np.ndarray:
        """ A is an array of all input channels for N samples, 
            the method returns an array of all the output channels, 
            shape of A: N x C_in x W_in  (W_in is itself of 2 dim)"""
        
        assert A.ndim==4, "A should have exactly 4 dimensions"
        assert A.shape[1]==self.in_channels, "The number of channels in every sample of the input should be equal to the in_channels of the conv layer "
        self.A = A
        Z_list=[]   #array for out channels of n samples
        num_samples=len(A)

        for n in range(num_samples):
            Z_n=[]  #array for out_channels of nth sample

            for o in range(self.out_channels):
                Z_n_o=np.zeros(shape=(len(A[n,0])-self.kernel_size[0]+1, len(A[n,0,0])-self.kernel_size[1]+1))  #array for oth out_channel of nth sample
                
                for i in range(self.in_channels):
                    filter_o_i=self.W[o,i]  #slicing the filter corresponding to oth output channel and i_th input channel
                    Z_n_o+=self.convolve(A[n,i],filter_o_i) #Adding all the in_channel convolutions to get the o_th out channel for nth sample
                
                Z_n_o+=self.B[o]    #Adding bias to the computed out_channel
                Z_n.append(Z_n_o)   #appending the oth output channel of nth sample to the Z_n list
            
            Z_list.append(Z_n)  #appending Z_n to the Z
    
        return np.array(Z_list)
    

    def backward(self, dLdZ:np.ndarray)->np.ndarray:
        """ dLdZ is an array of the derivatives (wrt L) of all output channels for N samples, 
            the method returns an array of derivatives (wrt L) all the input channels, 
            shape of dLdZ: N x C_out x W_out  (W_out is itself of 2 dim)"""
        
        assert dLdZ.ndim==4, "dLdZ should have exactly 4 dimensions"
        num_samples=len(dLdZ)

        #Creating padded dLdZ
        padded_dLdZ=np.copy(dLdZ)
        padding=self.kernel_size[0]-1   #Again, assuming kernel is a square
        
        #Adding zeros for padding padded_dLdZ
        for i in range(padding):
            padded_dLdZ=np.insert(padded_dLdZ,values=0,obj=(0,len(padded_dLdZ[0,0,0])),axis=3)
            padded_dLdZ=np.insert(padded_dLdZ,values=0,obj=(0,len(padded_dLdZ[0,0])),axis=2)

        #Flipping the weights (filters)
        flipped_W=np.flip(self.W,axis=3)
        flipped_W=np.flip(flipped_W,axis=2)

        dLdA=np.zeros(shape=self.A.shape)   #Initailizing dLdA

        for n in range(num_samples):
            for o in range(self.out_channels):
                self.dLdB[o]+=np.sum(dLdZ[n,o])     #Calculates dLdB

                for i in range(self.in_channels):
                    self.dLdW[o,i]+=self.convolve(self.A[n,i],dLdZ[n,o])    #calculates dLdW
                    dLdA[n,i]+=self.convolve(padded_dLdZ[n,o],flipped_W[o,i])   #calculated dLdA
        
        #Finding avg derivative for W and B
        self.dLdB=self.dLdB/num_samples
        self.dLdW=self.dLdW/num_samples
        
        #print(self.dLdW.shape==self.W.shape, self.dLdB.shape==self.B.shape,self.A.shape==dLdA.shape)

        return dLdA

#obj=Conv2d_stride1(in_channels=2,out_channels=2,kernel_size=(2,2))
#np.random.seed(43)
#arr=np.random.rand(1,2,3,3)     # 1 sample, 2 in_channels, each channel is 3x3
#print("input with 1 sample of 2 channels each of size 3x3:\n",arr)

#print("filters:\n",obj.W)
#print("bias:\n",obj.B)

#Z=obj.forward(arr)
#obj.backward(Z)

class conv2d():
    def __init__(self, in_channels:int, out_channels:int, kernel_size:tuple[int], stride:int, padding=0,weight_init_fn=None, bias_init_fn=None):
        self.stride = stride
        self.padding = padding
        self.conv2d_stride1 = conv2d_stride1(in_channels,out_channels,kernel_size)
        self.downsample2d = downsampling(stride,is_2d=True)

    def forward(self,A:np.ndarray)->np.ndarray:
        """ Adds padding to the input(A).
            Then performs a conv with stride 1. Then downsamples the output by a value=stride"""
        
        #Padding
        for i in range(self.padding):
            A=np.insert(A,values=0,obj=(0,len(A[0,0,0])),axis=3)
            A=np.insert(A,values=0,obj=(0,len(A[0,0])),axis=2)
        
        #Perform a Conv1d with stride=1
        Z=self.conv2d_stride1.forward(A)

        #Perform downsampling by k=stride
        if self.stride>1:
            downsampled_Z=[]
            for n in range(len(Z)):  #looping through n samples #! looping is neccesarry because downsampling method take a 2d array as input
                Z_n=[]
                for o in range(len(Z[n])):  #looping through each channel in nth sample
                    Z_n.append(self.downsample2d.forward(Z[n,o]))   #appending the returned array to A_n
                downsampled_Z.append(Z_n)

        return np.array(downsampled_Z)
    
    def backward(self, dLdZ:np.ndarray)->np.ndarray:
        # Call downsample2d backward
        if self.stride>1:
            upsampled_dLdZ=[]
            for n in range(len(dLdZ)):  #looping through n samples #! looping is neccesarry because downsampling method take a 2d array as input
                dLdZ_n=[]
                for o in range(len(dLdZ[n])):  #looping through each channel in nth sample
                    dLdZ_n.append(self.downsample2d.backward(dLdZ[n,o]))   #appending the returned array to A_n
                upsampled_dLdZ.append(dLdZ_n)
            dLdZ=np.array(upsampled_dLdZ)
        
        # Call conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Unpad the gradient
        if self.padding>0:
            dLdA=dLdA[:,:,self.padding:-self.padding,self.padding:-self.padding]

        return dLdA
    
#obj=Conv2d(in_channels=2,out_channels=2,kernel_size=(2,2),stride=2,padding=1)
#arr=np.random.rand(3,4,3,3) 
#print(arr.shape)
#Z=obj.forward(arr)
#print(Z.shape)
#obj.backward(Z)