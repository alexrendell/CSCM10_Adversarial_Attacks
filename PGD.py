import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

class PGD:
    def __init__ (self, model, epsilon, alpha, iters, random_start=True, device='cuda'):
        '''
        Initialise the PGD attack model
        
        Args:
            model (torch.nn.Module): The model to be attacked.
            epsilon (float): The maximum perturbation.
            alpha (float): The step size for each iteration.
            iters (int): The number of iterations to perform.
            random_start (bool): Whether to start with a random perturbation.
            device (str): The device to perform the computations on.    
        '''
        
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.random_start = random_start
        self.device = device
        
    def generate_adversarial_images(self, test_loader):
        '''
        Generate the adversarial images using PGD attack.
        
        Args:
            test_loader (torch.utils.data.DataLoader): The DataLoader containing the test images and labels.
        '''
        
        # Put the model in evaluation mode (disables some stuff)
        self.model.eval()
        
        # Used to keep track of the number of correct predictions
        correct_normal, correct_adv, total = 0, 0, 0
        
        # Iterate through the test datasetx
        for images, labels in tqdm(test_loader, desc='PGD Attack Progress'):
            
            # Move images and labels to the same device
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Make a copy of the original images
            adv_images = images.clone().detach()
            
            # If random start is enabled, add a small random perturbation to the images
            if self.random_start:
                # Generate a random perturbation to each pixel in the epsilon range
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
                # Clamp the images to ensure they dont excede the range (0 to 1)
                adv_images = torch.clamp(adv_images, 0, 1)
                
            # Do PGD for iteration times
            for _ in range(self.iters):
                
                # Treated like a constant, so we detach it from the computation graph
                adv_images = adv_images.detach()

                # Track the gradients of the adversarial images
                adv_images.requires_grad = True
                
                # Forward pass: compute the outputs of the model after each iteration
                outputs = self.model(adv_images)
                
                # Compute the loss using CrossEntropyLoss
                loss = nn.CrossEntropyLoss()(outputs, labels)

                # Compute the gradient of the loss (direction of loss)
                grad = torch.autograd.grad(loss, adv_images)[0]
                
                # Update the images in the direction that maximuses loss
                # grad.sign() gives the direction of the steepest ascent
                # alpha is the step size for each iteration
                adv_images = adv_images + self.alpha * grad.sign()

                # Computes the perturbation and ensures its within the epsilon range
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                # Reconstruct the images after confirming perturbation
                adv_images = torch.clamp(images + delta, 0, 1)
                
            with torch.no_grad():
                normal_preds = self.model(images).argmax(dim=1)
                adv_preds = self.model(adv_images).argmax(dim=1)
                
            # Count the number of correct predictions for normal and adversarial images
            correct_normal += (normal_preds == labels).sum().item()
            correct_adv += (adv_preds == labels).sum().item()
            total+= labels.size(0)
            
        print(f"Accuracy on normal test images: {100 * correct_normal / total:.2f}%")
        print(f"Accuracy on adversarial images (PGD, ε={self.epsilon}, α={self.alpha}, iters={self.iters}): {100 * correct_adv / total:.2f}%")

        
def plot_pgd_adversarial_vs_original(model, test_loader, img_num, epsilon=0.1, alpha=0.01, iters=10, random_start=True, device='cuda'):
    """
    Function to plot the original and adversarial image side by side.
    
    Args:
        model (torch.nn.Module): The model to be attacked.
        test_loader (torch.utils.data.DataLoader): The DataLoader containing the test images and labels.
        img_num (int): The index of the image to plot.
        epsilon (float): The maximum perturbation.
        alpha (float): The step size for each iteration.
        iters (int): The number of iterations to perform.
        random_start (bool): Whether to start with a random perturbation.
        device (str): The device to perform the computations on.
    """
    
    # Put the model in evaluation mode (disables some stuff)
    model.eval()
    
    # Get a batch from the test loader
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Get the selected image and its label
    original_image = images[img_num].unsqueeze(0).clone().detach().to(device)
    original_label = labels[img_num].unsqueeze(0).to(device)
    
    # Clone original image for adversarial generation
    adv_image = original_image.clone().detach()
    
    # Perform adversarial generation using PGD to the seleted image
    
    # Generate a random perturbation to each pixel in the epsilon range
    adv_image = adv_image + torch.empty_like(adv_image).uniform_(-epsilon, epsilon)
    # Clamp the images to ensure they dont excede the range (0 to 1)
    adv_image = torch.clamp(adv_image, 0, 1)
    
    for _ in range(iters):
        
        # Treated like a constant, so we detach it from the computation graph
        adv_image = adv_image.detach()
        
        # Track the gradients of the adversarial images
        adv_image.requires_grad = True
        
        # Forward pass: compute the outputs of the model after each iteration
        outputs = model(adv_image)
        
        # Compute the loss using CrossEntropyLoss
        # Ensure the label is in the correct shape for the loss function
        loss = torch.nn.CrossEntropyLoss()(outputs, original_label)
        
        # Compute the gradient of the loss (direction of loss)
        grad = torch.autograd.grad(loss, adv_image)[0]
        
        # Update the images in the direction that maximuses loss
        # grad.sign() gives the direction of the steepest ascent
        # alpha is the step size for each iteration
        adv_image = adv_image + alpha * grad.sign()
        
        # Computes the perturbation and ensures its within the epsilon range
        delta = torch.clamp(adv_image - original_image, min=-epsilon, max=epsilon)
        # Reconstruct the images after confirming perturbation
        adv_image = torch.clamp(original_image + delta, 0, 1)

    # Get predictions for original and adversarial images
    original_pred = model(original_image).argmax(dim=1).item()
    adversarial_pred = model(adv_image).argmax(dim=1).item()
    true_label = original_label.item()
        
    # Preparearing variables for for visualization
    original_image = original_image.squeeze().detach().cpu()
    adv_image = adv_image.squeeze().detach().cpu()
    difference = (adv_image - original_image).abs()
        
    # Plot original and adversarial
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title(f'Original Image\nTrue Label: {true_label}\nPrediction: {original_pred}')
    plt.imshow(original_image.permute(1, 2, 0))
    plt.axis('off')

    # Adversarial Image
    plt.subplot(1, 3, 2)
    plt.title(f'PGD Image (ε={epsilon})\nTrue Label: {true_label}\nPrediction: {adversarial_pred}')
    plt.imshow(adv_image.permute(1, 2, 0))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f'Perturbation Difference')
    difference = torch.abs(adv_image - original_image)
    plt.imshow(difference.permute(1, 2, 0))  # amplify difference
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def pgd_attack_group(model, images, labels, epsilon, alpha, iters, device, random_start=True):
    """
    Perform PGD attack on a group of images.
    
    Args:
        model (torch.nn.Module): The model to be attacked.
        images (torch.Tensor): Batch of images to attack.
        labels (torch.Tensor): Corresponding labels for the images.
        epsilon (float): The maximum perturbation.
        alpha (float): The step size for each iteration.
        iters (int): The number of iterations to perform.
        random_start (bool): Whether to start with a random perturbation.
        device (str): The device to perform the computations on.
        
    Returns:
        torch.Tensor: Adversarial images after the PGD attack.
    """
    
    # Put the model in evaluation mode (disables some stuff)
    model.eval()
    
    # Move images and labels to the same device
    images = images.to(device).detach().clone()
    labels = labels.to(device)
    
    # Clone the original images to create adversarial images
    adv_images = images.clone().detach()
    
    criterion = nn.CrossEntropyLoss()
    
    # If random start is enabled, add a small random perturbation to the images
    if random_start:
        # Generate a random perturbation to each pixel in the epsilon range
        adv_images += torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        # Clamp the images to ensure they dont excede the range (0 to 1)
        adv_images = torch.clamp(adv_images, 0, 1)
    
    # Do PGD for iteration times
    for _ in range(iters):
        
        # Treated like a constant, so we detach it from the computation graph
        adv_images = adv_images.detach()
        
        # Track the gradients of the adversarial images
        adv_images.requires_grad = True
        
        # Forward pass: compute the outputs of the model after each iteration
        outputs = model(adv_images)
        
        # Compute the loss using CrossEntropyLoss
        # Ensure the label is in the correct shape for the loss function
        loss = criterion(outputs, labels)
        
        # Compute the gradient of the loss (direction of loss)
        grad = torch.autograd.grad(loss, adv_images)[0]
        
        with torch.no_grad():
            # Update the images in the direction that maximuses loss
            # grad.sign() gives the direction of the steepest ascent
            # alpha is the step size for each iteration
            adv_images = adv_images + alpha * grad.sign()
        
            # Computes the perturbation and ensures its within the epsilon range
            delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
            # Reconstruct the images after confirming perturbation
            adv_images = torch.clamp(images + delta, 0, 1)

    # Return the adversarial images
    return adv_images
    
    