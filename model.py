import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim,device):
        super().__init__()
        
        self.embedding = nn.Embedding(inputDim, hiddenDim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hiddenDim, nhead=8,batch_first=True)
        self.tBlock1 = nn.TransformerEncoder(encoder_layer, 5)
        self.lastLayer = nn.Linear(hiddenDim,outputDim)
        self.outputDim = outputDim
        self.hiddenDim = hiddenDim
        print("made model with", sum(p.numel() for p in self.parameters()), "parameters of which", sum(p.numel() for p in self.parameters() if p.requires_grad), "are trainable")
        self = self.to(device)
        self.device = device
    def forward(self, input:torch.tensor):
        input = input.to(self.device)
        output = self.embedding(input)
        output = self.tBlock1(output)
        # output = output.transpose(1,2)
        outputShape = output.shape[:2] + (self.outputDim,)
        output = output.view(-1,self.hiddenDim)
        # print(output.shape)

        
        output = self.lastLayer(output)

        return output.view(outputShape)