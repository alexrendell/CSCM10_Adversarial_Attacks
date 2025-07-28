import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

import PGD

class ResNetModel(nn.Module):
    """
    A PyTorch model that wraps a pre-trained ResNet50 network.
    But, changes its parameters to what we need
    """
    def __init__(self, num_classes=3, pretrained=True):
        """
        Initialize the ResNetModel with our custom layers
        
        Args:
            num_classes (int): Number of classes to predict (default=3)
            pretrained (bool): Whether to load pretrained weights (default=True)
        """
        super(ResNetModel, self).__init__()
        
        #weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        #self.resnet = models.resnet18(weights=weights)
        
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        
        # Load the pretrained ResNet50 model
        # Used for Deep CNNs - image classification - feature extraction
        #self.resnet = models.resnet50(pretrained=pretrained)
        
        #Get thes the number of input layers to the final fully connected layer
        in_features = self.resnet.fc.in_features #Usually 2048 for ResNetModels
        
        #Replace the orignal final fully connected layer with our custom one
        #Lets us fine tune the model
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512), # Reduce dimensionality from 2048 to 512
            nn.ReLU(),                   # Apply ReLU acivation function for non-linearity
            nn.Dropout(0.3),             # Apply dropout with 0.3 probability
            nn.Linear(512, num_classes)  # Final layer that outputs data for each class
        )
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        
        #The entire ResNet model is applied to the input
        return self.resnet(x)
    
    def get_features(self, x):
        """
        Extract features from the input using all layers except the final fully connected layer
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Features extracted from the input tensor
        """
        
        
        # Get all the layers except the final fully connected layer
        # children() returns all children modules
        # [:-1] excludes the final layer
        modules = list(self.resnet.children())[:-1]
        
        # Creates a new sequential model with these layers
        feature_extractor = nn.Sequential(*modules)
            
        # Extract features -> doesnt compute gradients 
        with torch.no_grad():
            features = feature_extractor(x) # shape: (batch_size, in_features, 1, 1)
            features = torch.flatten(features, 1) # Flatten to shape (batch_size, in_features)
                
        return features
    
def create_Resnet_Model(num_classes=3, device='cuda'):
    """
    Creates and initializes the ResNetModel

    Args:
        num_classes (int): Number of classes to predict (default=3)
        device (str): Device to use for computation ('cuda' or 'cpu') (default='cuda')
            
    Returns:
        ResNetModel: Initialized ResNetModel
    """
        
    # Create a new ResNetModel with x number of classes
    model = ResNetModel(num_classes=num_classes, pretrained=True)
        
    # Move the model to specified device (if using cloud gpu or something)
    model = model.to(device)
        
    return model

# Training parameters
#num_epochs = 8         # number of epochs to train the model
#batch_size = 16         # Number of images to process in each batch
#learning_rate = 0.01    # Nlearning rate for the optimizer

def train_model(model, train_loader, val_loader,
                num_epochs, batch_size,
                learning_rate, device='cuda'
):
    """
    Trains the ResNetModel using Stochastic Gradient Descent (SGD) and with momentum
    and cross-entropy loss
    
    Args:
        model (torch.nn.Module): The ResNet model to be trained.
        training_dataset (torch.utils.data.Dataset): The training dataset.
        val_dataset (torch.utils.data.Dataset): The validation dataset.
        num_epochs (int): Number of epochs to train the model (default=10).
        batch_size (int): Number of samples in each batch (default=32).
        learning_rate (float): The learning rate for the optimizer (default=0.01).
        device (str): The device to train the model on ('cuda' for GPU or 'cpu' for CPU).
    
    Returns:
         model (torch.nn.Module): The trained ResNet model.
    """
    print('Starting training...')
    
    # Move the model to specified device
    model = model.to(device)

    # Create data loaders for training and validation datasets
    # The dataLoader stores the images and their labels
    #train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the loss function (cros-entropy loss)
    criterion = nn.CrossEntropyLoss()
    
    # Define the optimizer (Stochastic Gradient Descent with momentum)
    # Updates the model's parameters
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training loop: iterate throguh the dataset epoch many times
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} beginning")
        
        model.train() # Set the model to training mode (allowing it to be updated)
        
        running_loss = 0.0  # accumalate the loss over each batch
        correct = 0         # Tracks the number of correct predictions
        total = 0           # Track the total number of correct predictions
        
        # Iterate through the training data in batches
        for images, labels, in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            
            # Move images and labels to the same device as the model (They must be the same)
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients (avoid scores from previous iterations)
            optimizer.zero_grad()
            
            # Forward pass: to get predictions
            outputs = model(images)
            
            # Compute the loss: How far the real labels are from the predicted ones
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradients 
            loss.backward()
            
            # Update model weights: take a step in the direction of the gradients
            optimizer.step() # update model weights
            
            # The total loss for this batch
            running_loss += loss.item()
            
            # The class with highest probability is the predicted one
            _, predicted = torch.max(outputs, 1)
            
            # Update the count of the correct predictions
            correct += (predicted == labels).sum().item()
            
            # Update the total number of samples processed
            total += labels.size(0)
        
        # Training accuracy 
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validation phase: Evaluate the models performance
        model.eval() # Model to evaluation mode
        
        val_loss = 0.0  # accumalate the loss over each batch
        val_correct = 0 # Tracks the number of correct predictions
        val_total = 0   # Track the total number of correct predictions
        
        # Iterate through the validation data in batches
        with torch.no_grad():
            for images, labels, in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass: to get predictions
                outputs = model(images)
            
                # Compute the loss: How far the real labels are from the predicted ones
                loss = criterion(outputs, labels)
                # Backward pass: compute gradients
                val_loss += loss.item()
                # The class with highest probability is the predicted one
                _, predicted = torch.max(outputs, 1)
                # Update the count of the correct predictions
                val_correct += (predicted == labels).sum().item()
                # Update the total number of samples processed
                val_total += labels.size(0)
        
        # Validation accuracy
        val_acc = 100 * val_correct / val_total
        
        # Print the training and validation metrics for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss/len(train_loader):.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
            f"Val Acc: {val_acc:.2f}%")

    print("training Complete!")
    return model


def train_model_pgd(model, train_loader, val_loader, 
                    num_epochs, batch_size, 
                    learning_rate, epsilon, alpha, 
                    iters, adversarial_percent, 
                    device='cuda', random_start=True,
                    resume_training=False, checkpoint_path=None):
    
    '''
    trains the ResNetModel using PGD adversarial training.
    
    Args:
        model (torch.nn.Module): The ResNet model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Number of samples in each batch.
        learning_rate (float): Learning rate for the optimizer.
        epsilon (float): Maximum perturbation for the adversarial attack.
        alpha (float): Step size for the adversarial attack.
        iters (int): Number of iterations for the adversarial attack.
        adversarial_percent (float): Percentage of training data to use for adversarial training.
        device (str): Device to use for computation ('cuda' for GPU or 'cpu').
        random_start (bool): Whether to use random initialization for adversarial perturbations.
        
    Returns:
        model (torch.nn.Module): The trained ResNet model with adversarial training.
    '''
    
    # Print starting message
    print('Starting training with PGD adversarial training...')
    
    #model.eval()  # Set the model to evaluation mode initially
    
    # Move model to the specified device (GPU or CPU)
    model = model.to(device)
    
    # Define the loss function (cross-entropy loss for classification)
    criterion = nn.CrossEntropyLoss()
    
    # Define the optimizer (Stochastic Gradient Descent with momentum)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Initialize starting epoch for training
    start_epoch = 0  
    
    # Resume from checkpoint if necessary
    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch + 1}...")

    # Loop over the total number of epochs with a progress bar for all epochs
    for epoch in range(start_epoch, num_epochs):
    
        # Slowly adding adversarial percent - TEST
        if epoch < 3:
            adv_percent = 10
        elif epoch < 5:
            adv_percent = 25
        elif epoch < 7:
            adv_percent = 50
        elif epoch < 10:
            adv_percent = 75
        else:
            adv_percent = adversarial_percent

    
        # Print epoch start message
        print(f"\nEpoch {epoch+1} beginning")

        # Set model to training mode (enables dropout, batchnorm updates etc.)
        model.train()
        
        # Initialize running loss, correct prediction count and total sample count for this epoch
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Loop over batches in the training data loader with progress bar per epoch
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            
            # Move images and labels to the device
            images, labels = images.to(device), labels.to(device)
            
            # Get batch size
            full_batch_size = images.size(0)
            
            # Calculate how many images in this batch to apply adversarial attack to
            num_adv = int((adv_percent / 100) * full_batch_size)

            # If adversarial images are required, generate them using PGD attack
            if num_adv > 0:
                adv_images = PGD.pgd_attack_group(
                    model, images[:num_adv], labels[:num_adv],
                    epsilon=epsilon, alpha=alpha, iters=iters,
                    device=device, random_start=random_start
                )
                # Combine adversarial images with the rest of the clean images in the batch
                combined_images = torch.cat([adv_images, images[num_adv:]], dim=0)
                # Combine corresponding labels accordingly
                combined_labels = torch.cat([labels[:num_adv], labels[num_adv:]], dim=0)
                
                # SHuffle the combined images and labels to mix adversarial and clean samples
                perm = torch.randperm(combined_images.size(0))
                combined_images = combined_images[perm]
                combined_labels = combined_labels[perm]
                
            else:
                # If no adversarial images, use original images and labels as is
                combined_images = images
                combined_labels = labels

            # Zero out gradients from previous step before backpropagation
            optimizer.zero_grad()
            
            # Forward pass: compute predictions for combined images
            outputs = model(combined_images)
            
            # Compute the loss between predictions and true labels
            loss = criterion(outputs, combined_labels)
            
            # Backward pass: compute gradients of loss w.r.t model parameters
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update model parameters using optimizer
            optimizer.step()

            # Accumulate loss for reporting
            running_loss += loss.item()
            
            # Get predicted classes by selecting the class with highest score
            _, predicted = torch.max(outputs, 1)
            
            # Count number of correct predictions in this batch
            correct += (predicted == combined_labels).sum().item()
            
            # Update total number of samples processed so far
            total += combined_labels.size(0)

            # Calculate and print batch loss and accuracy for monitoring
            #batch_loss = loss.item()
            #batch_acc = 100 * (predicted == combined_labels).sum().item() / combined_labels.size(0)
            #print(f"  Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%")

        # Training accuracy for the epoch   
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

        # Compute average training loss for the epoch
        #train_loss_epoch = running_loss / len(train_loader)
        
        # Compute training accuracy for the epoch
        #train_acc_epoch = 100 * correct / total
        
        #print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.2f}%")

        # Set model to evaluation mode for validation (disables dropout etc.)
        model.eval()
        
        # Initialize validation loss, correct predictions, and total samples
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Disable gradient calculations for validation to save memory/computation
        #with torch.no_grad():
        # Iterate over validation data loader batches
        for images, labels in val_loader:
            # Move validation images and labels to device
            images, labels = images.to(device), labels.to(device)
            
            # Get batch size
            full_batch_size = images.size(0)
            
            # Calculate how many images in this batch to apply adversarial attack to
            num_adv = int((adv_percent / 100) * full_batch_size)
            
            # If adversarial images are required, generate them using PGD attack
            if num_adv > 0:
                adv_images = PGD.pgd_attack_group(
                    model, images[:num_adv], labels[:num_adv],
                    epsilon=epsilon, alpha=alpha, iters=iters,
                    device=device, random_start=random_start
                )
                # Combine adversarial images with the rest of the clean images in the batch
                combined_images = torch.cat([adv_images, images[num_adv:]], dim=0)
                # Combine corresponding labels accordingly
                combined_labels = torch.cat([labels[:num_adv], labels[num_adv:]], dim=0)
            
                # SHuffle the combined images and labels to mix adversarial and clean samples
                perm = torch.randperm(combined_images.size(0))
                combined_images = combined_images[perm]
                combined_labels = combined_labels[perm]
            
            else:
                # If no adversarial images, use original images and labels as is
                combined_images = images
                combined_labels = labels
            
            # Forward pass through the model
            outputs = model(combined_images)
            
            # Calculate loss on validation batch
            loss = criterion(outputs, combined_labels)
            
            # Accumulate validation loss
            val_loss += loss.item()
            
            # Get predicted classes for validation batch
            _, predicted = torch.max(outputs, 1)
            
            # Count correct predictions for validation batch
            val_correct += (predicted == combined_labels).sum().item()
            
            # Accumulate total samples in validation
            val_total += combined_labels.size(0)

        # Validation accuracy for the epoch
        val_acc = 100 * val_correct / val_total
        
        # Print the training and validation metrics for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {running_loss/len(train_loader):.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, "
            f"Val Acc: {val_acc:.2f}%")
        
        # Calculate average validation loss for the epoch
        #val_loss_epoch = val_loss / len(val_loader)
        
        # Calculate validation accuracy for the epoch
        #val_acc_epoch = 100 * val_correct / val_total

        # Print epoch summary showing training and validation loss and accuracy
        #print(f"Epoch [{epoch+1}/{num_epochs}] Summary - "
              #f"Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.2f}%, "
              #f"Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.2f}%")

        # Save checkpoint
        if checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

    # Print message when training completes
    print("Training complete!")
    
    # Return the trained model
    return model
        