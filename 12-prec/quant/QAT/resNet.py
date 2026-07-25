import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.quantization import QuantStub, DeQuantStub, fuse_modules, prepare_qat, convert

class QuantizableResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.quant = QuantStub()  # 量化入口
        self.dequant = DeQuantStub()  # 反量化出口
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)  # 修改输出层

    def forward(self, x):
        x = self.quant(x)  # 输入量化
        x = self.model(x)
        x = self.dequant(x)  # 输出反量化
        return x

    def fuse_model(self):
        # 融合Conv+BN+ReLU层
        fuse_modules(self.model, 
                    [['conv1', 'bn1', 'relu'],
                     ['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu'],
                     ['layer1.1.conv1', 'layer1.1.bn1']], 
                    inplace=True)

def train_qat(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')  # x86 CPU配置
    model = prepare_qat(model)  # 插入伪量化节点

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        
def inference(model, test_loader, device):
    model.eval()
    model = convert(model)  # 转换为真正的量化模型
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / len(test_loader.dataset):.2f}%')

# 数据加载
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuantizableResNet18(num_classes=10).to(device)
model.fuse_model()  # 必须融合层

# 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 执行QAT训练
train_qat(model, train_loader, criterion, optimizer, device)

# 推理测试
test_set = datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)
inference(model, test_loader, device)