import torch
import numpy as np

def train(model, inputs, outputs, batch_size=100, lr=1e-2, num_epochs=50):
    """
    Train the model.
    Args:
        model: The model to be trained.
        inputs: The input data.
        outputs: The output data.
        batch_size: The size of the mini-batch.
        lr: The learning rate.
        num_epochs: The number of epochs.
    """
    # Set the number of training samples
    num_samples = len(outputs)
    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Train the model
    model.train()
    for e in range(num_epochs):
        # Randomly shuffle the training data
        indices = np.random.permutation(num_samples)
        inputs = [inputs[i] for i in indices]
        outputs = [outputs[i] for i in indices]
        # Split the training data into mini-batches
        for i in range(0, num_samples, batch_size):
            # Get the current mini-batch
            input_batch = inputs[i:i + batch_size]
            output_batch = outputs[i:i + batch_size]
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass through the model
            error = model.evaluate(input_batch, output_batch)
            # Compute the loss
            loss = torch.mean(error)
            # Backward pass through the model
            loss.backward()
            # Update the gradients
            optimizer.step()
