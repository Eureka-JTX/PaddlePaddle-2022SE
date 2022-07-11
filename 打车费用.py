import paddle
# 里程
data_x = [3, 8, 9, 12, 15]
# 费用
data_y = [10, 25, 29, 38, 48]
# 神经网络
net = paddle.nn.Linear(in_features=1, out_features=1)
# 损失函数
loss_func = paddle.nn.MSELoss()
# 优化器
opt = paddle.optimizer.SGD(parameters=net.parameters())
# 训练
for epoch in range(10):
    for x, y in zip(data_x, data_y):
        x = paddle.to_tensor([x], dtype="float32")
        y = paddle.to_tensor([y], dtype="float32")
        in_y = net(x)
        loss = loss_func(in_y, y)
        loss.backward()
        opt.step()
        opt.clear_gradients()
        print(f"Epoch: {epoch} \t loss: {loss.numpy()}.")
# 测试
x = paddle.to_tensor([10], dtype="float32")
y = net(x)
print(f"费用为{y.numpy()}.")
