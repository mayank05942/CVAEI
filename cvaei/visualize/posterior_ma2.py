import numpy as np
import matplotlib.pyplot as plt

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
