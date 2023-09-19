import torch 
from torch import nn, optim 
from torch.nn import functional as F

class TemperatureScaler(nn.Module):
    def __init__(self, lr=0.001, max_iter=100):
        super().__init__()  
        self.lr = lr
        self.max_iter = max_iter
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels):
        optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=100)
        loss_fn = nn.CrossEntropyLoss()
        def eval():
            scaled_logits = logits/self.temperature
            loss = loss_fn(scaled_logits, labels)
            loss.backward()
            return loss 
        optimizer.step(eval)
        
        
class MLPScalar(nn.Module):
    def __init__(self, logits_dims=10,
                 hidden_neurons=100, lr=0.01, max_iter=50): 
        super().__init__()
        self.epochs = max_iter
        self.lr =  lr
        self.fc1 = nn.Linear(logits_dims, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, logits_dims)
    
    def forward(self, logits):
        x = F.relu(self.fc1(logits))
        x = self.fc2(x)
        return x
    
    def fit(self, logits, labels):
        optimizer = optim.SGD(self.parameters(), 
                              lr=self.lr, 
                              momentum=0.9,
                              weight_decay=0.0001
                              )
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            def closure():
                optimizer.zero_grad()
                outputs = self.forward(logits)
                loss = loss_fn(outputs, labels)
                loss.backward()
                return loss
            optimizer.step(closure)
            
    def to_canonical_representation(self):
        with torch.no_grad():
            l2_norm = torch.norm(self.fc1.weight.data, dim=1)
            sorted_indices = torch.argsort(l2_norm, descending=True)
            self.fc1.weight.data = self.fc1.weight.data[sorted_indices]
            self.fc2.bias = self.fc2.bias[sorted_indices]
            self.fc2.weight.data = self.fc2.weight.data[:, sorted_indices]
            
        
       
    
    


    