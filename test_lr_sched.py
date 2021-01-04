import torch
import matplotlib.pyplot as plt
import numpy as np

epochs = 50
initial_lr = 0.01
final_lr = 0.00001
lr_decay_gamma = (final_lr / initial_lr) ** (1 / epochs)

model = torch.nn.Sequential(
    torch.nn.Linear(10, 10)
)

criterion = torch.nn.MSELoss(reduction='sum').cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-5)
sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay_gamma, last_epoch=-1, verbose=True)

start_epoch = 25
for epoch in range(start_epoch):
    sched.step()
print("#"*80)

model.train()
rates = []
for epoch in range(start_epoch, epochs):
    rates.append(optimizer.param_groups[0]['lr'])
    out = model(torch.ones((10,)))
    loss = criterion(out, torch.ones((10,)))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sched.step()

rates = np.array(rates)
gt_rates = np.logspace(-2, -5, num=epochs)

print(rates.dtype)

plt.plot(np.abs(gt_rates - rates))
plt.plot(np.abs(gt_rates - initial_lr * np.power(lr_decay_gamma, np.arange(epochs))))
plt.plot(np.abs(rates - initial_lr * np.power(lr_decay_gamma, np.arange(epochs))))
plt.show()
