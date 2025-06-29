import numpy as np
class upsampling:
    def __init__(self,upsampling_factor,*,is_2d:bool=False):
        self.upsampling_factor=upsampling_factor
        self.is_2d=is_2d
    
    def forward(self,A:np.ndarray)->np.ndarray:
        
        """ Upsamples 1D data by inserting k-1 (k=upsampling_factor) zeros between every 2 pixels of the input map.
            For 2d upsampling is done by inserting k-1 columns of zeros between every 2 columns and k-1 rows
            of zeros between every 2 rows"""    
        
        assert A.ndim==2, "A should have exactly 2 dimensions"  #! Added this later because the logic below assumed a 2d inpy
        
        gap=self.upsampling_factor  #Declared this for simplicity
        
        
        if self.is_2d:
            shape=(A.shape[0]*self.upsampling_factor-(gap-1), A.shape[1]*self.upsampling_factor-(gap-1))    #! Only for 2d
        else:
            shape=(A.shape[0],A.shape[1]*self.upsampling_factor-(gap-1))    #! Only for 1d
        
        Z=np.zeros(shape=shape)
        
        if self.is_2d:
            for i in range(0,len(A)):
                for j in range(0,len(A[i,])):
        
                    Z[gap*i,gap*j]=A[i,j]   #! Only for 2d
        else:
            for i in range(0,len(A)):
                for j in range(0,len(A[i,])):
                    Z[i,gap*j]=A[i,j]       #! Only for 1d
        
        return Z
    
    def backward(self,dLdZ:np.ndarray)->np.ndarray:

        """The exact opposite of forward takes place in backward where intermediate k-1 elements are dropped, 
            because their derivative wrt Z is 0 and for the remaining elements, derivative wrt Z is 1"""
        
        assert dLdZ.ndim==2, "A should have exactly 2 dimensions"
        gap=self.upsampling_factor
        
        if self.is_2d:
            shape=(int((dLdZ.shape[0]+(gap-1))/self.upsampling_factor),int((dLdZ.shape[1]+(gap-1))/self.upsampling_factor))     #! Only for 2d
        else:
            shape=(dLdZ.shape[0],int((dLdZ.shape[1]+(gap-1))/self.upsampling_factor))   #! Only for 1d
        
        dLdA=np.zeros(shape=shape)
        
        if self.is_2d:
            for i in range(0,shape[0]):
                for j in range(0,shape[1]):
                    dLdA[i,j]=dLdZ[gap*i,gap*j]     #dLdA = dLdZ * dZdA = dLdZ since dZdA=1    #! Only for 2d
                
        else:
            for i in range(0,shape[0]):
                for j in range(0,shape[1]):    
                    dLdA[i,j]=dLdZ[i,gap*j]     #! Only for 1d

        return dLdA
    
class downsampling:
    def __init__(self,downsampling_factor,*,is_2d:bool=False):
        self.upsamp=upsampling(downsampling_factor,is_2d)
    
    def forward(self,A:np.ndarray)->np.ndarray:
        """ forward for downsampling is same as backward of upsampling, i.e. remove the intermediate k-1 elements"""

        self.W_in=A.shape   #! Storing the original shape of A to be used in backward(), see below
        return self.upsamp.backward(A)
    
    
    def backward(self,dLdZ:np.ndarray)->np.ndarray:
        """backward is same as forward of upsampling, because the elements which were removed (in downsampling forward),
            will have derivative wrt Z=0 and for the remaining elements, derivative wrt Z is 1"""
        
        dLdA=self.upsamp.forward(dLdZ)
        
        """#! The following step is necessary because while downsampling forward(), suppose if A is 1x3 and downsampling_factor=2
            #! we get a 1x2 array, also if A is 1x4 then also we get 1x2 array. So while doing backwrad() we need to restore to the original shape of A 
        """
        for _ in range(self.W_in[1]-dLdA.shape[1]):
            dLdA=np.insert(dLdA,values=0,obj=len(dLdA[0]),axis=1)

        for _ in range(self.W_in[0]-dLdA.shape[0]):
            dLdA=np.insert(dLdA,values=0,obj=len(dLdA),axis=0)
        #print(dLdA.shape)
        return dLdA
        
#arr=np.random.rand(1,4)
#print(arr)
#upsamp=downsampling(2,True)
#Z=(upsamp.forward(arr))
#print(Z)
#print(upsamp.backward(Z))