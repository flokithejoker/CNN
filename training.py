import torch 
from tqdm import tqdm

def training_loop(model, train_loader, test_loader, num_epochs, optimizer, loss_function, device):
    # for flexibility i put all configurational parts as input such that i can modify them from the main.py file
    # we can initialize our loss storage and start the training loop directly

    # here I will store my losses
    train_losses = []
    test_losses = []
    
    ############################## LOOPING OVER EPOCHS ##############################
    # Perform num_epoch full iterations over train_data. For every minibatch of each epoch
    for epoch in range(num_epochs):
        # set network to training mode
        model.train()

        # here we will store our total loss per epoch must be set to 0 every new epoch starts
        epoch_loss = 0

        # progress bars to follow during training
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")  
        
        ############################## LOOPING OVER MINIBATCHES ##############################
        # we loop over our batches as given by our train_loader
        for inputs, targets, _, _ in train_loader:
            # send everything to gpu
            inputs = inputs.to(device)
            targets = targets.to(device)

            # our standard procedure
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute the loss of the given batch
            loss = loss_function(outputs, targets)
            loss.backward()

            # Update the network’s weights according to this minibatch loss.
            optimizer.step()
            epoch_loss += loss.item()

        # Collect the minibatch loss to compute the average loss of the epoch (averaged over all minibatch losses).
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        ############################## EPOCH EVALUATION ##############################
        # After every epoch of training, a full iteration over test_data is performed. 
        model.eval()
        test_loss = 0.0

        # .no_grad such that we dont update the weights
        for inputs, targets, _, _ in test_loader:
            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                test_loss += loss.item()

        # Again, the loss needs to be computed and stored, but no weights should be updated.
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # After training and evaluation, the function returns a 2-tuple, where the 
        # first entry is the list of (averaged) training losses and the second entry is the list of evaluation losses.
        tqdm.write(f"Epoch {epoch+1} --- Train loss: {avg_epoch_loss:.4f} --- Test loss: {avg_test_loss:.4f}")
    
    return train_losses, test_losses

