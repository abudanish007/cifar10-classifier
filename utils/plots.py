import matplotlib.pyplot as plt
import os

def plot_metrics(train_losses, test_losses, train_accs, test_accs):
    os.makedirs("./results", exist_ok = True)
    
    epochs = range(1, len(train_losses)+1)
    
    plt.figure(figsize = (12, 4))
    
    plt.figure(figsize=(12,4))
    
    # Loss plot
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label = "Train Loss")
    plt.plot(epochs, test_losses, label = "Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend() 
    
    
    #Accuracy plot
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label = "Train Accuracy")
    plt.plot(epochs, test_accs, label = "Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over epochs")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("./results/training_metrics.png")
    plt.show()
    print("Plots saved to ./results/training_metrics.png")    
       