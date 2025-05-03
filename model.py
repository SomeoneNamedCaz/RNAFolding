import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

class Model(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim,device):
        super().__init__()
        expandDim = 128
        self.embedding = nn.Embedding(inputDim, hiddenDim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hiddenDim, nhead=8,batch_first=True)
        self.tBlock1 = nn.TransformerEncoder(encoder_layer, 5)
        self.expandChannels = nn.Linear(1,expandDim)
        
        self.triangleMul = TriangleMultiplication(expandDim)
        self.lastLayer = nn.Linear(hiddenDim,outputDim)
        self.outputDim = outputDim
        self.hiddenDim = hiddenDim
        
        print("made model with", sum(p.numel() for p in self.parameters()), "parameters of which", sum(p.numel() for p in self.parameters() if p.requires_grad), "are trainable")
        self = self.to(device)
        self.device = device
    def forward(self, input:torch.tensor):
        input = input.to(self.device)
        output = self.embedding(input)
        output = self.tBlock1(output) # has shape batch, seqLen, Channels
        output = torch.matmul(output, output.transpose(1,2)).unsqueeze(-1) # change to distogram shape: batchSize, seqLen, seqLen, 1
        output = self.expandChannels(output)
        output = self.triangleMul(output)

        # outputShape = output.shape[:2] + (self.outputDim,)
        # output = output.view(-1,self.hiddenDim)
 
        
        output = self.lastLayer(output)

        return output
    

class TriangleMultiplication(nn.Module):
    def __init__(self,input_channels, internalChannels=128):
        super().__init__()
        self.layerNorm = nn.LayerNorm(input_channels)
        self.firstAttenLin = nn.Linear(input_channels, internalChannels)
        self.secondAttenLin = nn.Linear(input_channels, internalChannels)
        self.thirdAttenLin = nn.Linear(input_channels, internalChannels)
        self.outNorm = nn.LayerNorm(internalChannels)
        self.outLin = nn.Linear(internalChannels, input_channels)

    def forward(self, input:torch.tensor):
        normalized = self.layerNorm(input)
        a = F.sigmoid(self.firstAttenLin(normalized)) * self.secondAttenLin(normalized)
        b = F.sigmoid(self.firstAttenLin(normalized)) * self.secondAttenLin(normalized)
        g = F.sigmoid(self.thirdAttenLin(normalized))
        out = torch.einsum('bikc,bjkc->bijc', a, b)        

        output = g * self.outLin(self.outNorm(out))
        

        return output
    


def testValues(a,b,i,j):
    tot = torch.zeros(a.shape)
    for batchIdx in range(a.shape[0]):
        for k in range(a.shape[2]):
            tot[batchIdx,i,j] += a[batchIdx,i,k] * b[batchIdx,j,k]


    return tot


if __name__ == "__main__":
    # test code 
    #               batch, seqlen,seqlen, channel
    a = torch.rand(10,40,40,12)
    b = torch.rand(10,40,40,12)
    out = torch.einsum('bikc,bjkc->bijc', a, b)  
    for i in range(out.shape[1]):
        for j in range(out.shape[2]):
            
            correctSlowOut = testValues(a,b,i,j)
            # testing fast batched dot product from alphafold paper
            assert torch.mean(correctSlowOut[:,i,j] - out[:,i,j]) < 0.00005
    
    test_input = torch.randint(0,4,(10,42))
    print(test_input.shape)
    model = Model(4,128,3, "cpu")
    print("out shape",model(test_input).shape)