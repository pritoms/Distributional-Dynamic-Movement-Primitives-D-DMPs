import matplotlib.pyplot as plt
import numpy as np

def plot_demonstration(demonstration, title='Demonstration'):
    """
    Plot the demonstration as a solid blue line
    :param demonstration: the demonstration to plot
    :param title: the title of the plot
    :return: None
    """
    # Get the ax used for plotting
    ax = plt.gca()
    # Plot the demonstration as a solid blue line
    ax.plot(np.arange(0, len(demonstration)), demonstration, color='b', linewidth=1.5)
    # Set the title
    plt.title(title)
    # Turn on the legend
    ax.legend(['Demonstration'])


def plot_error(model, inputs, outputs, title='Error', show_plot=True):
    """
    Plot the error of the model
    :param model: the model to evaluate
    :param inputs: the inputs to the model
    :param outputs: the outputs to the model
    :param title: the title of the plot
    :param show_plot: whether or not to show the plot
    :return: the error of the model
    """
    # Compute the error
    error = model.evaluate(inputs, outputs, plot=show_plot)
    # Check if we want to show the plot
    if show_plot:
        # Get the ax used for plotting
        ax = plt.gca()
        # Set the title
        plt.title(title)
        # Turn on the legend
        ax.legend(['Error'])
    return error
