import torch
import torch.nn.functional as F
from torch.utils.data import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from data_processing import dataPreprocess, spiltDatast
from Model_PMA import Model

# 定义训练函数和测试函数
def train(model, device, train_loader, optimizer, epoch):  # 训练模型
    model.train()
    best_acc = 0.0
    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        start_time = time.time()
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

        outputs, pooled, y_pred = model([x1, x2, x3])  # 得到预测结果
        model.zero_grad()  # 梯度清零

        loss = F.cross_entropy(y_pred, y.squeeze())  # 得到loss
        loss.backward()

        optimizer.step()
        if (batch_idx + 1) % 100 == 0:  # 打印loss
            print('Train Epoch: {} [{}/{} ({:.2f}%)]/t Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))  # 记得为loss.item()


def validation(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            outputs, pooled, y_ = model([x1, x2, x3])

        test_loss += F.cross_entropy(y_, y.squeeze()).item()

        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'],
                yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # 保存混淆矩阵图
    plt.savefig('confusion_matrix.png')

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

def main():
    input_ids = []  # input char ids
    input_types = []  # segment ids
    input_masks = []  # attention mask
    label = []  # 标签
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataPreprocess("/hy-tmp/urls/dataset/f_benign_urls.txt", input_ids, input_types, input_masks, label, 0)
    dataPreprocess("/hy-tmp/urls/dataset/f_malware_urls.txt", input_ids, input_types, input_masks, label, 1)

    input_ids_train, input_types_train, input_masks_train, y_train, input_ids_test, input_types_test, input_masks_test, y_test = spiltDatast(
        input_ids, input_types, input_masks, label)

    # 加载到高效的DataLoader
    BATCH_SIZE = 64
    train_data = TensorDataset(torch.tensor(input_ids_train).to(DEVICE),
                               torch.tensor(input_types_train).to(DEVICE),
                               torch.tensor(input_masks_train).to(DEVICE),
                               torch.tensor(y_train).to(DEVICE))
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    test_data = TensorDataset(torch.tensor(input_ids_test).to(DEVICE),
                              torch.tensor(input_types_test).to(DEVICE),
                              torch.tensor(input_masks_test).to(DEVICE),
                              torch.tensor(y_test).to(DEVICE))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    model = Model().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    best_acc = 0.0
    NUM_EPOCHS = 3
    PATH = '/hy-tmp/roberta_modelxs.pth'  # 定义模型保存路径
    for epoch in range(1, NUM_EPOCHS + 1):  # 3个epoch
        train(model, DEVICE, train_loader, optimizer, epoch)
        acc, precision, recall, f1 = validation(model, DEVICE, test_loader)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), PATH)  # 保存最优模型
        print("acc is: {:.4f}, best acc is {:.4f}n".format(acc, best_acc))

if __name__ == '__main__':
    main()