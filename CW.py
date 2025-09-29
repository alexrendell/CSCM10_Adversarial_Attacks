import torch
from torchattacks import CW as AttackCW
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm

class CW :
    def __init__(self, model, c, kappa, steps, lr, device='cuda'):
        """
        Initialise the CW attack model
        
        Args:
            model (torch.nn.Module): The model to be attacked.
            c (float): Constant for the loss function. - The larger C is, the more noise - Controls balance between noise and misclassification.
            kappa (float): Confidence parameter for the attack. - CW tries to make the wrong prediction by atleast kappa amount
            steps (int): Number of optimization steps. - Iterations
            lr (float): Learning rate for the optimization. - The size of the step taken in each iteration
            device (str): The device to perform the computations on.
        """
        self.model = model
        self.c = c
        self.kappa = kappa 
        self.steps = steps
        self.lr = lr
        
        self.cw_attack = AttackCW(self.model, c=self.c, kappa=self.kappa, steps=self.steps, lr=self.lr)
        
        self.device = device

        
    def generate_adversarial_images(self, test_loader):
        """
        Generate adversarial images using CW attack and print progress per batch.
        """
        self.model.eval()
        
        correct_adv, total = 0, 0
        
        # Per-class counters - [Benign (0), Malignant (1)]
        correct_adv_class = [0, 0]
        fooled_class = [0, 0]
        total_per_class = [0, 0]

        for i, (images, labels) in enumerate(tqdm(test_loader, desc='C&W Attack Progress')):
            images, labels = images.to(self.device), labels.to(self.device)
            total += labels.size(0)

            # Generate adversarial images
            adv_images = self.cw_attack(images, labels)

            # Predictions on adversarial images
            adv_preds = self.model(adv_images).argmax(dim=1)

            # Count correct predicitions
            correct_adv += (adv_preds == labels).sum().item()
            
             # Update per-class accuracy and fooled co
            for i in [0, 1]:
                index = (labels == i)
                correct_adv_class[i] += (adv_preds[index] == labels[index]).sum().item()
                fooled_class[i] += (adv_preds[index] != labels[index]).sum().item()
                total_per_class[i] += index.sum().item()
                
        # Compute all metrics
        overall_adv_accuracy = 100 * correct_adv / total
        per_class_accuracy = [100 * correct_adv_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0 
                              for i in [0, 1]]
        overall_fooling_rate = 100 * sum(fooled_class) / total
        per_class_fooling_rate = [100 * fooled_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0 
                                  for i in [0, 1]]

        # Print results
        print("Overall Adversarial Accuracy: ", overall_adv_accuracy, "%  Fooling Rate: ", overall_fooling_rate, "%")
        print("Benign Class Accuracy: ", per_class_accuracy[0], "%  Fooling Rate: ", per_class_fooling_rate[0], "%")
        print("Malignant Class Accuracy: ", per_class_accuracy[1], "%  Fooling Rate: ", per_class_fooling_rate[1], "%")




def plot_cw_adversarial_vs_original(model, test_loader, img_num, c=0.01, kappa=0, steps=100, lr=0.001, device='cuda'):
    """
    Plot original and CW adversarial images side by side.

    Args:
        model (torch.nn.Module): The model to be attacked.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        img_num (int): Index of image in the batch to attack and plot.
        c (float): Constant for the CW loss function.
        kappa (float): Confidence margin for misclassification.
        steps (int): Number of optimization steps.
        lr (float): Learning rate for optimization.
        device (str): Device to run the attack on.
    """

    # Ensure model is in eval mode and on the correct device
    model = model.to(device)
    model.eval()

    # Get one batch from test_loader
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Select the image and label by img_num
    original_image = images[img_num].unsqueeze(0).clone().detach().to(device)
    original_label = labels[img_num].unsqueeze(0).to(device)

    # Create a CW attack instance
    cw_attack = AttackCW(model=model, c=c, kappa=kappa, steps=steps, lr=lr)

    # Generate CW adversarial image
    adv_image = cw_attack(original_image, original_label)
    adv_image = torch.clamp(adv_image, 0, 1)

    # Get predictions
    original_pred = model(original_image).argmax(dim=1).item()
    adversarial_pred = model(adv_image).argmax(dim=1).item()
    true_label = original_label.item()

    # Prepare images for plotting
    original_image_cpu = original_image.squeeze().detach().cpu()
    adv_image_cpu = adv_image.squeeze().detach().cpu()
    difference = (adv_image_cpu - original_image_cpu).abs()

    # Plot side-by-side
    plt.figure(figsize=(12, 4))

    # Original
    plt.subplot(1, 3, 1)
    plt.title(f'Original Image\nTrue Label: {true_label}\nPrediction: {original_pred}')
    plt.imshow(original_image_cpu.permute(1, 2, 0))
    plt.axis('off')

    # Adversarial
    plt.subplot(1, 3, 2)
    plt.title(f'C&W Image (c={c}, kappa={kappa})\nTrue Label: {true_label}\nPrediction: {adversarial_pred}')
    plt.imshow(adv_image_cpu.permute(1, 2, 0))
    plt.axis('off')

    # Perturbation
    plt.subplot(1, 3, 3)
    plt.title('Perturbation Difference')
    plt.imshow(difference.permute(1, 2, 0))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

