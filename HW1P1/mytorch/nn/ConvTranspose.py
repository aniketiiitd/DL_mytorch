import numpy as np
import Conv1d,Conv2d
from Resampling import upsampling

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,weight_init_fn=None, bias_init_fn=None):
        self.upsampling_factor = upsampling_factor
        self.upsample1d = upsampling(self.upsampling_factor)
        self.conv1d_stride1 = Conv1d.conv1d_stride1(in_channels,out_channels,kernel_size)
    
    def forward(self, A:np.ndarray)->np.ndarray:
        #print(A.shape)
        # Step 1: Upsampling1d forward
        upsampA=[]
        for sample_id in range(len(A)):
            upsampA.append(self.upsample1d.forward(A[sample_id]))
        upsampA=np.array(upsampA)
        #print(upsampA.shape)
        
        # Step 2: Conv1d_stride1 forward
        Z=self.conv1d_stride1.forward(upsampA)
        return Z
    
    def backward(self, dLdZ:np.ndarray)->np.ndarray:
        
        #Step 1: Conv1d_stride1 backward
        dLdA =self.conv1d_stride1.backward(dLdZ)
        #print(dLdA.shape)
        
        # Step 2: Upsampling1d backward
        downsamp_dLdA=[]
        for sample_id in range(len(dLdA)):
            downsamp_dLdA.append(self.upsample1d.backward(dLdA[sample_id]))
        dLdA=np.array(downsamp_dLdA)
        #print(dLdA.shape)
        return dLdA
    
#obj=ConvTranspose1d(4,out_channels=3,kernel_size=2,upsampling_factor=2)
#Z=obj.forward(np.array([[[1,0,0,-1,2],[1,-1,0,-1,1],[1,0,1,-1,2],[1,2,3,4,5]],[[1,0,0,-1,2],[1,-1,0,-1,1],[1,0,1,-1,2],[1,2,3,4,5]]]))
##print(Z.shape)
#obj.backward(Z)

class ConvTranspose2d():
    def __init__(self, in_channels:int, out_channels:int, kernel_size:tuple[int], upsampling_factor:int,weight_init_fn=None, bias_init_fn=None):
        self.upsampling_factor = upsampling_factor
        self.upsample2d = upsampling(upsampling_factor,is_2d=True)
        self.conv2d_stride1 = Conv2d.conv2d_stride1(in_channels,out_channels,kernel_size)

    def forward(self, A:np.ndarray)->np.ndarray:
        
        # Step 1: Upsampling1d forward
        upsampA=[]
        for n in range(len(A)):
            A_n=[]
            for o in range(len(A[n])):
                A_n.append(self.upsample2d.forward(A[n,o]))
            upsampA.append(A_n)

        upsampA=np.array(upsampA)

        # Line 2: conv2d_stride1 forward
        Z=self.conv2d_stride1.forward(upsampA)
        
        return Z
    
    def backward(self, dLdZ:np.ndarray)->np.ndarray:
        
        #Step 1: conv1d_stride2 backward
        dLdA =self.conv2d_stride1.backward(dLdZ)

        # Line 2: Upsampling2d backward
        downsamp_dLdA=[]
        for n in range(len(dLdA)):
            dLdA_n=[]
            for i in range(len(dLdA[n])):
                dLdA_n.append(self.upsample2d.backward(dLdA[n,i]))
            downsamp_dLdA.append(dLdA_n)
        
        dLdA = np.array(downsamp_dLdA)
        return dLdA
    
#obj=ConvTranspose2d(2,out_channels=3,kernel_size=(3,3),upsampling_factor=3)
#A=np.random.rand(2,2,3,3)
#Z=obj.forward(A)
#print(obj.backward(Z).shape)