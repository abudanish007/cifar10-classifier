import torch
import torch.nn as nn 
import os

from config import config 
from dataset import get_dataloaders
from model import CNN
from trainer import Trainer 
from evaluator import Evaluator
from utils.plots import plot_metrics

def main():
    device = torch.device(config["device"] if torch.backends.mps.is_available() else "cpu")
    print(f"Using device:{device}")
    
    os.makedirs("./checkpoints", exist_ok = True)
    
    
    train_loader, test_loader = get_dataloaders(config["data_dir"], config["batch_size"])
    
    model = CNN(num_classes=config("num_classes")).to(device)
    
    
    criterion = nn.CrossEnropyLoss()
    optimize = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)
    
    trainer = Trainer(mode, optimizer, criterion, device)
    evaluator = Evaluator(model, criterion, device)
    
    train_losses, test_losses = [],[]
    train_acces, test_accs = [],[]
    best_acc = 0 
    
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train.loss, train_acc = trainer.train_rpoch(train_loader)
        test_loss, test_acc = evaluator.evaluate(test_loader)
        scheduler.step()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), config["save_path"])
            print(f"Model saved with accuracy: {best_acc:.2f}%")

    # Plot results
    plot_metrics(train_losses, test_losses, train_accs, test_accs)
    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()