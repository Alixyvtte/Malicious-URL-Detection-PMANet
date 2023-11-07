import torch
import torch.nn.functional as F
from torch.utils.data import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from data_processing import dataPreprocess, spiltDatast
from Model_PMA import Model, CharBertModel


def train(model, device, train_loader, optimizer, epoch): 
    """
     Train the model.

    :param model: The model to be trained.
    :param device: The device to run training on (e.g., CPU or GPU).
    :param train_loader: The data loader for training data.
    :param optimizer: The optimization algorithm.
    :param epoch: The current epoch number.
    :return: None
    """
    model.train()
    best_acc = 0.0
    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        start_time = time.time()
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

        outputs, pooled, y_pred = model([x1, x2, x3])  # Get the prediction results
        model.zero_grad()  # Reset gradients

        loss = F.cross_entropy(y_pred, y.squeeze())  # Calculate the loss
        loss.backward()

        optimizer.step()
        if (batch_idx + 1) % 100 == 0:  # Print the loss
            print('Train Epoch: {} [{}/{} ({:.2f}%)]/t Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))  # Remember to use loss.item()



def validation(model, device, test_loader):
    """
    Perform model validation on the test data.

    :param model: The model to be validated.
    :param device: The device to run validation on (e.g., CPU or GPU).
    :param test_loader: The data loader for test data.
    :return: A tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            outputs, pooled, y_ = model([x1, x2, x3])

        test_loss += F.cross_entropy(y_, y.squeeze()).item()

        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2 outputs, representing the maximum value and its index

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'],
                yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the confusion matrix plot
    plt.savefig('confusion_matrix.png')

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1


def main():
    input_ids = []  # input char ids
    input_types = []  # segment ids
    input_masks = []  # attention mask
    label = []  # Label
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataPreprocess("benign_urls.txt", input_ids, input_types, input_masks, label, 0)
    dataPreprocess("malware_urls.txt", input_ids, input_types, input_masks, label, 1)

    input_ids_train, input_types_train, input_masks_train, y_train, input_ids_val, input_types_val, input_masks_val, y_val = spiltDatast(
        input_ids, input_types, input_masks, label)

    # Load data into efficient DataLoaders
    BATCH_SIZE = 64
    train_data = TensorDataset(torch.tensor(input_ids_train).to(DEVICE),
                               torch.tensor(input_types_train).to(DEVICE),
                               torch.tensor(input_masks_train).to(DEVICE),
                               torch.tensor(y_train).to(DEVICE))
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    val_data = TensorDataset(torch.tensor(input_ids_val).to(DEVICE),
                              torch.tensor(input_types_val).to(DEVICE),
                              torch.tensor(input_masks_val).to(DEVICE),
                              torch.tensor(y_val).to(DEVICE))
    val_sampler = SequentialSampler(val_data)
    val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    model = Model().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    best_acc = 0.0
    NUM_EPOCHS = 3
    PATH = '/model.pth'  # Define the model saving path
    for epoch in range(1, NUM_EPOCHS + 1):  # 3 epochs
        train(model, DEVICE, train_loader, optimizer, epoch)
        acc, precision, recall, f1 = validation(model, DEVICE, val_loader)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), PATH)  # Save the best model
        print("acc is: {:.4f}, best acc is {:.4f}n".format(acc, best_acc))
if __name__ == '__main__':
    main()
