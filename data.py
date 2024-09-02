from dataset import ImagesDataset
from torch.utils.data import DataLoader, random_split

# first step is always to make the data split
def split_data(data_dir, batch_size=32, train_split=0.8):
    # create instance of our dataset
    data = ImagesDataset(data_dir)

    # calculate split sizes
    n = len(data)
    training_size = int(n * train_split)
    test_size = n - training_size

    # using random_split to create splits
    train_dataset, test_dataset = random_split(data, [training_size, test_size])

    # create DataLoader instances for use in training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader