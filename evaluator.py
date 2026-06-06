import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device 
        
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0 
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc= "Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs,labels)
                
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
        return total_loss / len(test_loader), 100. * correct/total