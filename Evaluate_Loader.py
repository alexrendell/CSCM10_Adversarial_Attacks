import torch


def evaluate_natural(model, test_loader, device):
    
    # Set the model to evaluation mode
    model.eval()
    # Initialize variables to track accuracy
    correct_total = 0
    total_samples = 0
    
    # For per-class accuracy - 0 = Benign, 1 = Malignant
    correct_per_class = [0, 0] 
    total_per_class = [0, 0]
    
    
    
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for images, labels in tqdm(test_loader, desc="Progress", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # Forward pass
            _, predicted = torch.max(outputs, 1) # Get predicted class
            
            # Update overall counts
            correct_total += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Update per-class counts
            for i in [0, 1]:
                index = (labels == i)
                correct_per_class[i] += (predicted[index] == labels[index]).sum().item()
                total_per_class[i] += index.sum().item()
                
    # Compute accuracies
    overall_accuracy = 100 * correct_total / total_samples
    per_class_accuracy = [100 * correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0 
                          for i in [0, 1]]
    
    # Print results
    print("Overall Natural Accuracy: ", overall_accuracy, "%")
    print("Benign Class Accuracy: ", per_class_accuracy[0], "%")
    print("Malignant Class Accuracy: ", per_class_accuracy[1], "%")

    return overall_accuracy, per_class_accuracy
