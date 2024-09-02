from data import split_data
from architecture import MyCNN
from training import training_loop
import torch.nn as nn
import torch.optim as optim
import torch
from evaluation import evaluate_model

# CONFIGURAtIONS FOR MY MODEL WILL BE UPPER LETTER
DATA_DIR = '/Users/peter/VS_Studio/Python II/CNN/training_data'
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
LEARNING_RATE = 0.001
EPOCHS = 2
LOSS_FUNCTION = nn.CrossEntropyLoss()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} for training.")


# I simply hard coded our labels
labels = ('book', 'bottle', 'car', 'cat', 'chair', 'computermouse', 'cup', 'dog', 'flower', 'fork',
           'glass', 'glasses', 'headphones', 'knife', 'laptop', 'pen', 'plate', 'shoes', 'spoon', 'tree')


def main():
    # Load data
    train_loader, test_loader = split_data(DATA_DIR, BATCH_SIZE, TRAIN_SPLIT)

    # Initialize model
    model = MyCNN().to(DEVICE)

    # Train model
    OPTIMIZER = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training_loop(model, train_loader, test_loader, EPOCHS, OPTIMIZER, LOSS_FUNCTION, DEVICE)
    print('Training loop finished')

    # Save model
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved to model.pth')

    # Evaluate model
    evaluate_model(model, test_loader, DEVICE)


if __name__ == '__main__':
    main()