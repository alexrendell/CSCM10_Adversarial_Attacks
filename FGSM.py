import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FGSM:
    def __init__(self, model, epsilon=0.1, device='cuda'):
        """
        intialise the FGSM adversarial attack model.
    
        Args:
            model (torch.nn.Module): The model to be attacked.
            epsilon (float): The maximum magnitude of the adversarial perturbation.
            device (str): The device to perform the computations on.
        """
    
        self.model = model
        self.epsilon = epsilon
        self.device = device
    
    def generate_adversarial_images(self, test_loader):
        """
        Generate the adversarial images using FGSM attack.
    
        Args:
            images (torch.Tensor): The input images to generate adversarial images
            labels (torch.Tensor): the true labels fo the input images
        """
    
        # Put the model in evaluation mode (disables some stuff)
        self.model.eval()
        
        correct_normal, correct_adv, total = 0, 0, 0
        
        for images, labels in test_loader:
            
            # Move the images and labels to the same device
            images, labels = images.to(self.device), labels.to(self.device)
            
            # require the gradients for the images
            images = images.clone().detach().requires_grad_(True)    
    
            # Forward pass: compute the outputs of the model
            outputs = self.model(images)
    
            # Compute the loss using CrossEntropyLoss
            loss = nn.CrossEntropyLoss()(outputs, labels)
    
            # Set all gradients to zero
            self.model.zero_grad()
    
            # Backward pass: compute the gradients of loss with respect to the input images
            loss.backward()
    
            # Get the sign of the gradients (direction for the perturbations)
            perturbation = self.epsilon * images.grad.sign()
    
            # Create the adversarial images by adding the perturbations to the original images
            adversarial_images = images + perturbation
    
            # Clip the adversarial images to ensure they are within the valid range (0-1)
            adversarial_images = torch.clamp(adversarial_images, 0, 1)
    
            # Get the predictions on normal and adversarial images
            normal_preds = outputs.argmax(dim=1)
            adv_preds = self.model(adversarial_images).argmax(dim=1)
            
            correct_normal += (normal_preds == labels).sum().item()
            correct_adv += (adv_preds == labels).sum().item()
            total += labels.size(0)
            

        # Print accuracy for both normal and adversarial images
        print(f"Accuracy on normal test images: {100 * correct_normal / total:.2f}%")
        print(f"Accuracy on adversarial images (FGSM, ε={self.epsilon}): {100 * correct_adv / total:.2f}%")
    
    # Function to plot the original and adversarial image side by side
    
    
def plot_adversarial_vs_original(model, test_loader, img_num, epsilon=0.1, device='cuda'):
    '''
    Visualizes the original image and the original image permutated
        
    Args:
        model (torch.nn.Module): The model to generate adversarial example
        test_loader (torch.utils.data.DataLoader): Dataloader containing test images
        epsilon (float): Magnitude of adversarial perturbation
        device (str): Device to run computations on
    '''        
        
    # Set model to evaluation mode
    model.eval()
        
    # get the first batch from the loader
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
        
    # Select the image number from the batch
    original_image = images[img_num]
    #original_image = images[img_num].detach().cpu
    original_label = labels[img_num].item()
        
    # Clone image and prepare for gradient computation
    #perturbed_image = original_image[img_num]
    perturbed_image = original_image.clone().to(device).unsqueeze(0).requires_grad_(True)
        
    # Compute the model output
    output = model(perturbed_image)
    loss = torch.nn.CrossEntropyLoss()(output, labels[0:1])
        
    # Zero gradients and compute gradients
    model.zero_grad()
    loss.backward()
        
    # Compute adversarial perturbation
    # peturbed_image.grad - ``vontains the gradients from backpropagation
    # .sign() - Converts the gradient into its direction -1 or +1
    # epsilon - scales the magnitude of the perturbation
    # torch.clamp - Makes sure the image stays in the pixel range
    # p + p, 0, 1 - Adds the perturbation to the original image
    perturbation = epsilon * perturbed_image.grad.sign()
    adversarial_image = torch.clamp(perturbed_image + perturbation, 0, 1)
        
    # Get the predictions
    # model(perturbed_image) - Does a forward pass through the CNN and returns its probabilities
    # .argmax(dim=1) - Finds the index with the highest probability
    # .item() - Converts the tensor value to a python integer
    original_pred = model(perturbed_image).argmax(dim=1).item()
    adversarial_pred = model(adversarial_image).argmax(dim=1).item()
        
    # Prepare for visualization
    original_image = original_image.squeeze()
    adversarial_image = adversarial_image.detach().cpu().squeeze()
        
    # Visualization
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.title(f'Original Image\nTrue Label: {original_label}\nPrediction: {original_pred}')
    plt.imshow(original_image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
    plt.axis('off')
    
    # Adversarial Image
    plt.subplot(1, 2, 2)
    plt.title(f'Adversarial Image (ε={epsilon})\nTrue Label: {original_label}\nPrediction: {adversarial_pred}')
    plt.imshow(adversarial_image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
    plt.axis('off')
    
    # Difference Visualization (Optional)
    difference = torch.abs(original_image - adversarial_image)
    plt.figure(figsize=(5, 5))
    plt.title('Perturbation Difference')
    plt.imshow(difference.permute(1, 2, 0))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
        
    '''
    def evaluate_adversarial_attack(self, model, test_loader, epsilon):
        """
        
        """
        
        model.eval()
        correct_normal, correct_adv = 0, 0
        
        fgsm_attack = FGSM(model, epsilon)
        
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            
            # Get predictions on normal image
            with torch.np_grad():
                normal_preds = model(images).argmax(dim=1)
                
            # Generate adversarial examples and get predictions
            adv_images = self.generate_adversarial_images(images, labels)
            
            # get adversarial predictions
            with torch.no_grad():
                adv_preds = model(adv_images).argmax(dim=1)
            
            correct_normal += (normal_preds == labels).sum().item()
            correct_adv += (adv_preds == labels).sum().item()
            total += labels.size(0)
            
        print(f"Accuracy on normal test images: {100 * correct_normal / total:.2f}%")
        print(f"Accuracy on adversarial images (FGSM, ε={self.epsilon}): {100 * correct_adv / total:.2f}%")
    '''  
