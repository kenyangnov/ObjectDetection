# y.backward(w) 求的不是 y 对 x 的导数，而是 l = torch.sum(y*w) 对 x 的导数
import torch
from torch.autograd import Variable

x = Variable(torch.randn(3), requires_grad=True)
y = Variable(torch.randn(3), requires_grad=True)
z = Variable(torch.randn(3), requires_grad=True)

t = x + y
l = t.dot(z)

l.backward(retain_graph=True)
print(x.grad) # x.grad = y.grad = z
print(y.grad)
print(z)

print(z.grad) # z.grad = t = x + y
print(t)

# 把 z(即dl/dt) 作为 t.backward() 的参数
# variables 是 t，grad_variables 是 variables 的导数 dl/dt=z
x.grad.data.zero_()
y.grad.data.zero_()
z.grad.data.zero_()

t.backward(z)
print(x.grad)
print(y.grad)
# 上式说明，不一定需要从计算图最后的节点 l 往前反向传播，
#从中间某个节点 t 开始传也可以，
# 只要你能把损失函数 l 关于这个节点的导数 dl/dt 记录下来，
# t.backward(dl/dt),即t.backward(z) 照样能往前回传，
# 正确地计算出损失函数 l 对于节点 t 之前的节点的导数。


# via:
#	https://zhuanlan.zhihu.com/p/29923090
# 	https://zhuanlan.zhihu.com/p/27808095