import torch
import torch.nn.functional as F
from torch.utils.data import *
from data_processing import  dataPreprocessFromCSV
from Model_PMA import Model, CharBertModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

def test_mutilple(model, device, test_loader, num_classes=4):
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []  # 保存所有类别的预测概率

    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            outputs, pooled, y_ = model([x1, x2, x3])

        # 使用多分类交叉熵损失函数
        loss = F.cross_entropy(y_, y.squeeze())  # y是类别索引
        test_loss += loss.item()

        pred = y_.max(-1, keepdim=True)[1]  # 获取预测的类别索引

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

        # 保存所有类别的预测概率
        y_probs.extend(torch.softmax(y_, dim=1).cpu().numpy())

    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')  # 修改为多分类的宏平均精度
    recall = recall_score(y_true, y_pred, average='macro')  # 修改为多分类的宏平均召回率
    f1 = f1_score(y_true, y_pred, average='macro')  # 修改为多分类的宏平均F1分数

    cm = confusion_matrix(y_true, y_pred)

    # 保存混淆矩阵图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes),
                yticklabels=range(num_classes))  # 根据类别数调整标签
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    y_probs = np.array(y_probs)
    # Calculate ROC curves for each class using One-vs-Rest (OvR) method
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_probs[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves for each class
        plt.figure()
        plt.plot(fpr[i], tpr[i], color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc[i]:0.2f}) for class {i}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Class {i}')
        plt.legend(loc='lower right')
        plt.savefig(f'roc_curve_class_{i}.png')

    # 保存预测结果、原结果、预测概率到txt文件
    results_array = np.column_stack((y_true, y_pred) + tuple(y_probs.T))  # 保存所有类别的预测概率
    header_text = "True label, Predicted label" + ''.join([f", Probability Class {i}" for i in range(num_classes)])
    np.savetxt('results.txt', results_array, fmt='%1.6f', delimiter='\t', header=header_text)

    print('Test set: Average loss: {:.4f}'.format(test_loss))
    return accuracy, precision, recall, f1

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = []  # input char ids
    input_types = []  # segment ids
    input_masks = []  # attention mask
    label = []  # 标签

    dataPreprocessFromCSV("/hy-tmp/dataset/multi_test.csv", input_ids, input_types, input_masks, label)
    # 加载到高效的DataLoader
    BATCH_SIZE = 64
    test_data = TensorDataset(torch.tensor(input_ids).to(DEVICE),
                              torch.tensor(input_types).to(DEVICE),
                              torch.tensor(input_masks).to(DEVICE),
                              torch.tensor(label).to(DEVICE))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    # 加载训练好的模型
    model = Model().to(DEVICE)  # 替换为你的模型定义
    model.load_state_dict(torch.load("/hy-tmp/multiple.pth"))  # 加载模型参数

    # 测试模型
    accuracy, precision, recall, f1 = test_mutilple(model, DEVICE, test_loader)

if __name__ == '__main__':
    main()