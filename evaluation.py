# here i test the trained model on my test set
import torch
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device):
    # set model to evaluation mode
    model.eval()

    # to calculate accuracy we need to collect scores
    n = 0
    correct = 0

    # loop over test set without gradient updates
    with torch.no_grad():
        for images, labels, _, _ in test_loader:
            # move data and infer data in model
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # compare prediction against true label
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            
            num_images_displayed = 0
            images_to_display = 10

            # i also want to print a couple of test images
            if num_images_displayed < images_to_display:

                # loop until 10 images are printed
                for i in range(min(images_to_display - num_images_displayed, len(images))):
                    image = images[i].cpu().numpy().transpose((1, 2, 0))

                    # compare true vs predicted label
                    true_label = labels[i].item()
                    predicted_label = predicted[i].item()

                    # visualize with matplot
                    plt.imshow(image.squeeze(), cmap='gray')
                    plt.title(f'True: {true_label}, Predicted: {predicted_label}')
                    plt.show()

                    # image counter adjusted
                    num_images_displayed += 1
                    if num_images_displayed >= images_to_display:
                        break

            # update total counter
            n += labels.size(0)

    # finally calculate the overall accuracy
    accuracy = correct / n
    print(f"Accuracy: {accuracy * 100:.2f}%")
