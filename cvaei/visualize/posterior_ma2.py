import numpy as np
import matplotlib.pyplot as plt
import torch

def scatter_plot_ma2(data):
    """
    Creates a scatter plot with a triangular prior indicated by dotted lines and scatter points for estimates.

    :param data: A 2D NumPy array with each row being a data point (Theta 1, Theta 2).
    """
    # Define the corners of the triangular prior
    triangle_corners = np.array([[-2, 1], [2, 1], [0, -1]])

    # Create the scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(data[:, 0], data[:, 1], alpha=1,  s=100.5, label='Estimated Posterior')

    # Draw the triangle with dashed lines
    plt.plot([triangle_corners[0][0], triangle_corners[1][0]], [triangle_corners[0][1], triangle_corners[1][1]], 'k--', label='Prior')
    plt.plot([triangle_corners[1][0], triangle_corners[2][0]], [triangle_corners[1][1], triangle_corners[2][1]], 'k--')
    plt.plot([triangle_corners[2][0], triangle_corners[0][0]], [triangle_corners[2][1], triangle_corners[0][1]], 'k--')

    plt.scatter([0.6], [0.2], color='red', s=10, label='True Value', zorder=5)

    # Set the axes limits
    plt.xlim(-2, 2)
    plt.ylim(-1, 1)

    # Add labels and title
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 2')
    plt.title('VAE: Posterior MA2')
    
    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def scatter_plot(tensor):
    """
    Plots a scatter plot from a 1D or 2D PyTorch tensor.
    
    Parameters:
    - tensor: A 1D or 2D PyTorch tensor. If 2D, columns represent different dimensions.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
        
    if tensor.ndim > 2:
        raise ValueError("Function supports up to 2D tensors only.")
    
    tensor = tensor.cpu().detach()  # Ensure tensor is on CPU and detached from the computation graph
    
    plt.figure(figsize=(8, 6))
    
    if tensor.ndim == 1:
        # For 1D tensors, plot values against their index
        plt.scatter(torch.arange(len(tensor)), tensor, alpha=0.6, edgecolors='w', label='Data Points')
    elif tensor.ndim == 2:
        # For 2D tensors, plot column 0 vs column 1
        if tensor.size(1) < 2:
            raise ValueError("2D tensors must have at least 2 columns.")
        plt.scatter(tensor[:, 0], tensor[:, 1], alpha=0.6, edgecolors='w', label='Data Points')
    
    plt.title('Scatter Plot of Tensor Data')
    plt.xlabel('Dimension 1' if tensor.ndim == 2 else 'Index')
    plt.ylabel('Dimension 2' if tensor.ndim == 2 else 'Value')
    plt.legend()
    plt.grid(True)
    plt.show()
