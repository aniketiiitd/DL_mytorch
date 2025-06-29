import numpy as np
class sgd:
    def __init__(self,model,lr=0.1,momentum=0):
        self.l=[layer[0] for layer in model.layers]
        self.L=len(model.layers)
        self.lr=lr 
        self.mu = momentum
        if(self.mu!=0):
            self.v_W = [np.zeros(self.l[i].W.shape) for i in range(self.L)]
            self.v_B = [np.zeros(self.l[i].B.shape) for i in range(self.L)]

    def step(self):
        if(self.mu==0):
            for i in range(self.L):
                self.l[i].W = self.l[i].W - self.lr*(self.l[i].dLdW)
                self.l[i].B = self.l[i].B - self.lr*(self.l[i].dLdB)
        else:
            for i in range(self.L):
                self.v_W[i] = (self.mu*self.v_W[i]) + self.l[i].dLdW
                self.v_B[i] = (self.mu*self.v_B[i]) + self.l[i].dLdB

                self.l[i].W = self.l[i].W - self.lr*(self.v_W[i])
                self.l[i].B = self.l[i].B - self.lr*(self.v_B[i])
