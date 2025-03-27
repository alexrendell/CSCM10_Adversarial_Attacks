import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader

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
        for images, labels, in train_loader:
            print('Batch')
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


    