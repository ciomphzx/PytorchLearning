import torch

x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
print(y.size())
y.backward()
print(x.grad)

x.grad.zero_()

y = x.sum()
print(y)
# 反向传播计算梯度，梯度存储在grad属性中
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.backward(torch.ones_like(z))
print(u)
print(x.grad)

x.grad.zero_()
y.backward(torch.ones_like(y))
print(x.grad)