import torch
import matplotlib.pyplot as plt



def plot_variable_gt(variable):

    variable_np = variable.cpu().numpy()

    plt.figure(figsize=(10, 4))
    for i, axis in enumerate(['w', 'x', 'y', 'z']):
        plt.plot(variable_np[:, i], label=f'var_{axis} (GT)')

    plt.title('Ground Truth')
    plt.xlabel('Índice GPS')
    plt.ylabel('')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()