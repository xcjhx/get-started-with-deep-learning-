#引入文件路径相关文件
import os
#引入系统相关文件
import sys
#引入加载丈量数据的pytorch文件
from torch.utils.data import DataLoader
#引入进度条
from tqdm import tqdm

#引入pytorch库，它封装好了很多神经网络模块以及相关优化算法
import torch
import torch.nn as nn
import torch.optim as optim

#从data_loader文件引入iris...函数
from data_loader import iris_dataloader

#初始化神经网络模型
#定义NN模型类，以nn.module为父类继承
class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim1,hidden_dim2,out_dim):
        #这是父类nn.moudle的构造函数,用于做模型的初始化
        super().__init__()
        #定义神经网络模型的三个层的权重，输入层到第隐藏层1，隐藏层1到隐藏层2，隐藏层2到输出层
        self.layer1 = nn.Linear(in_dim,hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, out_dim)

    #定义神经网络向前传输的计算图
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
    
#告诉神经网络应该在gpu还是cpu进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#训练集 与 验证集 和 测试集  的划分
#利用iris_dataloader类从文件导入数据集
custom_dataset = iris_dataloader("Iris_data.txt")
#定义各数据集的大小
train_size = int(len(custom_dataset)*0.7)
val_size = int(len(custom_dataset)*0.2)
test_size = len(custom_dataset) - train_size - val_size
#以custom_dataset为原始数据，将其打乱拆分为“train_size,val_size,test_size”三种大小的数据集，并将其分别放置于train_dataset,val_dataset与test_dataset中
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset,[train_size,val_size,test_size])

#定义从train_dataset数据集中一轮抽取的数据量为batch_size-16，并且每次抽取后打乱剩余数据shuffle = True
train_loader = DataLoader(train_dataset,batch_size=16, shuffle=True)
#定义从验证集中每轮抽取1个数据，并且不打乱剩余数据
val_loader = DataLoader(val_dataset,batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=1, shuffle=False)
#打印各数据集的大小，计算方法 - 轮数*每轮抽取个数 其实就是len(train_size)
print("训练集的大小",len(train_loader)*16,"验证集的大小",len(val_loader),"测试集的大小",len(test_loader))

#定义一个推理函数，用于计算并返回准确率

def infer(model, dataset, device):
    #将模型的模式调整为评估模式
    model.eval()
    #将测试正确的数量赋0
    acc_num = 0
    #与model.eval()配合使用，表示以下代码段禁用梯度算法，即不改变各层各枝的权重，但是退出该代码段后恢复
    with torch.no_grad():
        for data in dataset:
            datas,label = data
            outputs = model(datas.to(device))
            predict_y = torch.max(outputs, dim = 1)[1]
            acc_num += torch.eq(predict_y,label.to(device)).sum().item()
        
    acc = acc_num / len(dataset)
    return acc

def main(lr = 0.005,epochs = 20):
    model = NN(4, 12 , 6, 3).to(device)
    loss_f = nn.CrossEntropyLoss()

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=lr)

    #权重文件存储路径
    save_path = os.path.join(os.getcwd(),"results/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    #开始训练
    for epoch in range(epochs):
        model.train()
        acc_num = torch.zeros(1).to(device)
        sample_num = 0

        train_bar = tqdm(train_loader, file = sys.stdout, ncols=100)
        for datas in train_bar:
            data,label = datas
            label = label.squeeze(-1)
            sample_num += data.shape[0]

            optimizer.zero_grad()
            outputs = model(data.to(device))
            pred_class = torch.max(outputs, dim=1)[1]
            acc_num = torch.eq(pred_class, label.to(device)).sum()

            loss = loss_f(outputs, label.to(device))
            loss.backward()
            optimizer.step()

            train_acc = acc_num / sample_num
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)

        val_acc = infer(model,val_loader,device)
        print("train epoch[{}/{}] loss:{:.3f} train_acc{:.3f} val_acc{:.3f}".format(epoch+1,epochs,loss,train_acc,val_acc))
        torch.save(model.state_dict(),os.path.join(save_path, "nn.pth"))

        #每次数据集迭代之后，要对初始化的指标要清零
        train_acc = 0.
        val_acc = 0.

    print("Fished Training")
    test_acc = infer(model, test_loader,device)
    print("test_acc:",test_acc)

if __name__ == "__main__":
    main()


