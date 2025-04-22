import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#the function f(x)
def f(x):
    return x*(1 - x)

#the neural network clsas
class YarotskyNetwork(nn.Module):
    def __init__(self, p):
        super(YarotskyNetwork, self).__init__()
        self.p = p

        #1st layer weights and bias
        W0 = torch.tensor([[1.0], [1.0], [0.0]])
        b0 = torch.tensor([0.0, -0.5, 0.0])
        self.W0 = nn.Parameter(W0, requires_grad=False)
        self.b0 = nn.Parameter(b0, requires_grad=False)

        #hidden layers
        self.Wr = []
        self.br = []
        for r in range(1, p):
            Wr = torch.tensor([
                [2.0, -4.0, 0.0],
                [2.0, -4.0, 0.0],
                [2.0 / (4 ** r), -4.0 / (4 ** r), 1.0]])
            br = torch.tensor([0.0, -0.5, 0.0])
            self.Wr.append(nn.Parameter(Wr, requires_grad=False))
            self.br.append(nn.Parameter(br, requires_grad=False))

        #final layer
        Wp = torch.tensor([[2 * (4 ** (-p)), -4 * (4 ** (-p)), 1.0]])
        bp = torch.tensor([0.0])
        self.Wp = nn.Parameter(Wp, requires_grad=False)
        self.bp = nn.Parameter(bp, requires_grad=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.matmul(self.W0, x) + self.b0
        x = self.relu(x)
        for r in range(self.p - 1):
            x = torch.matmul(self.Wr[r], x) + self.br[r]
            x = self.relu(x)
        x = torch.matmul(self.Wp, x) + self.bp
        return x.squeeze()

#calculate approximation error for varying p
x_values = torch.linspace(0, 1, 101).unsqueeze(1)
true_values = f(x_values)

errors = []
p_values = list(range(2, 11))

for p in p_values:
    network = YarotskyNetwork(p)
    predictions = torch.tensor([network(x) for x in x_values])
    error = torch.max(torch.abs(predictions - true_values.squeeze()))
    errors.append(error.item())
    print(f"p={p}, Error={error.item():.8f}")
    
    if p  in [2,4,8]:
      plt.plot(x_values.numpy(), predictions.numpy(), label=f'p={p}')

plt.plot(x_values.numpy(), true_values.numpy(), label='True f(x)', linestyle='--', color='black')
plt.title("Function Approximations")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()      

#plot log of approximation error
plt.plot(p_values, np.log(errors), marker='o')
plt.title("Log of Approximation Error vs. p")
plt.xlabel("Number of hidden layers (p)")
plt.ylabel("log(Error)")
plt.grid(True)
plt.show()
