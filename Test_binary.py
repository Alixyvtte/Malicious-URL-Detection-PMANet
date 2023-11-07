import torch
import torch.nn.functional as F
from torch.utils.data import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from data_processing import  dataPreprocessFromCSV
from Model_PMA import Model, CharBertModel
import numpy as np

def test_binary(model, device, test_loader):
    """
    Perform binary classification testing using the given model.

    :param model: The model for binary classification.
    :param device: The device to run testing on (e.g., CPU or GPU).
    :param test_loader: The data loader for test data.
    :return: A tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []  # Save predicted probabilities

    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            outputs, pooled, y_ = model([x1, x2, x3])

        test_loss += F.cross_entropy(y_, y.squeeze()).item()

        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2 outputs, representing the maximum value and its index

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_probs.extend(torch.softmax(y_, dim=1).cpu().numpy()[:, 1])  # Save predicted probabilities

    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    # Save the confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malware'],
                yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # Save the ROC curve plot
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')

    # Save predicted results, original results, and predicted probabilities to a txt file
    results_array = np.column_stack((y_true, y_pred, y_probs))
    header_text = "True label, Predicted label, Predicted Probability"
    np.savetxt('results.txt', results_array, fmt='%1.6f', delimiter='\t', header=header_text)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
        test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100))

    return accuracy, precision, recall, f1

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = []  # input char ids
    input_types = []  # segment ids
    input_masks = []  # attention mask
    label = []  # Labels

    dataPreprocessFromCSV("/dataset/multi_test.csv", input_ids, input_types, input_masks, label)
    # Load data into efficient DataLoaders
    BATCH_SIZE = 64
    test_data = TensorDataset(torch.tensor(input_ids).to(DEVICE),
                              torch.tensor(input_types).to(DEVICE),
                              torch.tensor(input_masks).to(DEVICE),
                              torch.tensor(label).to(DEVICE))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    # Load the pre-trained model
    model = Model().to(DEVICE)  # Replace with your model definition
    model.load_state_dict(torch.load("/Experiment/model_Pro.pth"))  # Load model parameters

    # Test the model
    accuracy, precision, recall, f1 = test_binary(model, DEVICE, test_loader)

if __name__ == '__main__':
    main()
