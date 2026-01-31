import torch
from d2l import torch as d2l
import torch.nn as nn

#设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#读取数据并创建迭代器
batchSize = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batchSize)
train_iter.num_workers = 0
test_iter.num_workers = 0

#模型
# net = nn.Sequential(nn.Conv2d(1,6,kernel_size=(5,5),padding=2),nn.Sigmoid(),
#                     nn.MaxPool2d(kernel_size=(2,2),stride=2),
#                     nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
#                     nn.MaxPool2d(kernel_size=2,stride=2),
#                     nn.Flatten(),
#                     nn.Linear(16*5*5,120),
#                     nn.Linear(120,84),
#                     nn.Linear(84,10))

net = nn.Sequential(nn.Conv2d(1,8,kernel_size=(3,3)),nn.ReLU(),#26*26输出
                    nn.MaxPool2d(kernel_size=(2,2),stride=2),#1*8*13*13输出
                    nn.Conv2d(8,16,kernel_size=4),nn.ReLU(),#1*16*10*10
                    nn.MaxPool2d(kernel_size=2,stride=2),#1*16*5*5
                    nn.Flatten(),
                    nn.Linear(400,256),nn.ReLU(),
                    nn.Linear(256,64),nn.ReLU(),
                    nn.Linear(64,10))



#每一层的输出形状
# x = torch.rand(size=(1,1,28,28))
# for m in net:
#     x = m(x)
#     print(m.__class__.__name__,x.shape)

#训练
numEpochs, lr = 10, 0.5
d2l.train_ch6(net,train_iter,test_iter,numEpochs,lr,device)
torch.save(net.state_dict(),'LeNetV1.params')
for X,y in test_iter:
    break

X,y = X.to(device),y.to(device)
labels = d2l.get_fashion_mnist_labels(y)
# for idx,label in enumerate(labels):
#     if(label == 't-shirt'):
#         x = X[idx]
#         for layer in net:
#             if(isinstance(layer,nn.ReLU)):
#                 x = layer(x)
#                 activeValue.append(x)
#             else: 
#                 x= layer(x)
d2l.show_images(X[0:9].reshape(9,28,28).cpu().detach(),1,9,titles=labels[0:9])
x_first_reluOut = net[0:2](X)[0:9,1,:,:]
d2l.show_images(x_first_reluOut.reshape(9,26,26).cpu().detach(),1,9)
x_second_reluOut = net[0:5](X)[0:9,1,:,:]
d2l.show_images(x_second_reluOut.reshape(9,10,10).cpu().detach(),1,9)

d2l.plt.show()
