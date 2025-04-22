import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

#the neural network class
class ReLUApproxNet(nn.Module):
    def __init__(self, p_layers=4, d_neurons=10):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, d_neurons))
        for _ in range(p_layers - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(d_neurons, d_neurons))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(d_neurons, 1))  # output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



#function to compute L∞ error
def linf_error(model, x_eval, f_true):
    with torch.no_grad():
        y_pred = model(x_eval).squeeze()
        y_true = f_true(x_eval).squeeze()
        error = torch.abs(y_pred - y_true).max().item()
    return error

def f(x):
    return x * (1 - x)

#gereating data
def generate_training_data(n_samples=500):
    x = torch.linspace(0, 1, n_samples).view(-1, 1)
    y = x * (1 - x)
    return x, y
#train model for given p and d
def train_model(p, d, epochs=500, batch_size=64):
    model = ReLUApproxNet(p_layers=p, d_neurons=d)
    x,y = generate_training_data()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for xb, yb in dataloader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model



#main
if __name__ == "__main__":
    x_evaluation, y_true = generate_training_data()

    configs = [(4, 3), (4, 50), (4, 100)]
    for p, d in configs:
        model = train_model(p, d)
        y_predicted = model(x_evaluation).detach().numpy()
        plt.plot(x_evaluation.numpy(), y_predicted, label=f'p={p}, d={d}')

    plt.plot(x_evaluation.numpy(), y_true, 'k--', label='True function')
    plt.title('Function Approximations')
    plt.legend()
    plt.grid()
    plt.show()
    
    #calculate L error for different depths and widths
    errors = {}

    for d in [3, 10, 50, 100]:
        errors[d] = []
        for p in [2, 3, 4, 5, 6]:
            model = train_model(p, d)
            err = linf_error(model, x_evaluation, f)
            errors[d].append(err)
            print(f'p={p}, d={d}, L infinite error={err:.5f}')
    
    
    for d, error_list in errors.items():
        plt.plot(range(2, 7), np.log(error_list), label=f'd={d}')
    plt.xlabel('Depth (p)')
    plt.ylabel('log(L infinite error)')
    plt.title('Error Convergence for Different Widths')
    plt.legend()
    plt.grid()
    plt.show()

