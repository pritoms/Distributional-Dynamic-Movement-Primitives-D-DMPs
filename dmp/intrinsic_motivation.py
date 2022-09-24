import torch
import numpy as np

def train(model, inputs, outputs, batch_size=100, lr=1e-2, num_epochs=50):
    """
    Train the model.
    :param model: The model to be trained.
    :param inputs: The training inputs.
    :param outputs: The training outputs.
    :param batch_size: The batch size.
    :param lr: The learning rate.
    :param num_epochs: The number of epochs.
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

def transfer_learning(source, target, num_epochs=50):
    """
    Transfer the knowledge from the source model to the target model.
    :param source: The source model.
    :param target: The target model.
    :param num_epochs: The number of epochs.
    """
    # Set the model to be untrained
    target.intention = False
    # Initialize the weights
    initialize_weights(target)
    # Initialize the projection matrix weights - we keep one for each output dimension
    target.projection_matrix_weights = torch.nn.Parameter(torch.normal(torch.zeros((target.num_basis, target.output_dim)), torch.ones((target.num_basis, target.output_dim))).type(target.dtype))
    # Get the semi-definite representation from the model
    psi = target.model(torch.zeros((1, target.input_dim), dtype=target.dtype)).squeeze()
    # Create GPs for each column of the semi-definite representation
    target.gps = nn.ModuleList([GaussianProcess(1, 1, num_basis=source.num_basis, basis_stddev=source.stddev) for _ in range(psi.shape[0])])
    # Initialize the spring rate weights - we keep one for each output dimension
    target.spring_rate_weights = torch.nn.Parameter(torch.normal(torch.zeros(target.output_dim), torch.ones(target.output_dim)).type(target.dtype))
    # Learn the parameters of the transfer model
    train(target, source.train_inputs, source.train_outputs, num_epochs=num_epochs)
    # Get the normalized projection matrix weights
    alpha = torch.sum(target.projection_matrix_weights, dim=1)
    target.projection_matrix_weights /= alpha
    # Set the model to be intentional
    target.intention = True
