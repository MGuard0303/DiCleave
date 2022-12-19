"""
The sklearn package uses the Keras style, which the True value in the first place and the Predicted value in the
second place. However, These "get" methods use the PyTorch style, where the Predicted value is in the
first place, followed by the True value.
"""


import torch
import sklearn.metrics


# Return a list of [tn, fp, fn, tp] for binary labels
# Return a confusion matrix in tensor format for multi-class labels. Rows indicate label values and columns indicate
# predict values
def get_confusion(y_pred, y_true, task):

    if len(y_pred) != len(y_true):
        raise ValueError("The length of prediction tensor and label tensor should be identical.")

    # Get flatten confusion matrix of binary classification
    if task == "binary":
        pred = y_pred.detach().round()
        lbl = y_true.detach()

        mtx = sklearn.metrics.confusion_matrix(lbl, pred)
        mtx_list = list(mtx.ravel())

        return mtx_list

    # Get the confusion matrix of multiple classifier
    if task == "multi":
        pred = y_pred.detach()
        lbl = y_true.detach()

        temp = []
        for i in pred:
            temp.append(i.argmax().item())

        pred = torch.tensor(temp).unsqueeze(1)

        mtx = sklearn.metrics.confusion_matrix(lbl, pred)
        mtx = torch.tensor(mtx, dtype=torch.long)

        return mtx


# Return a list of [accuracy, precision, recall, specificity, sensitivity, f_one]
# Input confusion list is [tn, fp, fn, tp]
def bi_metrics(confusion_list):
    epsilon = 1e-8
    accuracy = (confusion_list[3] + confusion_list[0]) / (confusion_list[3] + confusion_list[0] + confusion_list[1] +
                                                          confusion_list[2])

    precision = confusion_list[3] / (confusion_list[3] + confusion_list[1] + epsilon)
    recall = confusion_list[3] / (confusion_list[3] + confusion_list[2] + epsilon)

    specificity = confusion_list[0] / (confusion_list[0] + confusion_list[1] + epsilon)
    sensitivity = confusion_list[3] / (confusion_list[3] + confusion_list[2] + epsilon)

    f_one = 2 * ((precision * recall) / (precision + recall + epsilon))

    return [accuracy, precision, recall, specificity, sensitivity, f_one]


# Calculate acc, precision, recall, specificity, sensitivity and f_one from confusion matrix
# Use the macro-average approach
def tri_metrics(mtx):
    # Calculate accuracy
    accuracy = (mtx[0, 0] + mtx[1, 1] + mtx[2, 2]) / torch.sum(mtx).item()

    # Calculate precision
    pre0 = (mtx[0, 0]) / (mtx[0, 0] + mtx[1, 0] + mtx[2, 0])
    pre1 = (mtx[1, 1]) / (mtx[1, 1] + mtx[0, 1] + mtx[2, 1])
    pre2 = (mtx[2, 2]) / (mtx[2, 2] + mtx[0, 2] + mtx[1, 2])

    precision = (pre0 + pre1 + pre2) / 3

    # Calculate recall
    re0 = (mtx[0, 0]) / (mtx[0, 0] + mtx[0, 1] + mtx[0, 2])
    re1 = (mtx[1, 1]) / (mtx[1, 1] + mtx[1, 0] + mtx[1, 2])
    re2 = (mtx[2, 2]) / (mtx[2, 2] + mtx[2, 0] + mtx[2, 1])

    recall = (re0 + re1 + re2) / 3

    # Calculate specificity
    spe0 = (mtx[1, 1] + mtx[1, 2] + mtx[2, 1] + mtx[2, 2]) / (mtx[1, 1] + mtx[1, 2] + mtx[2, 1] + mtx[2, 2] +
                                                              mtx[1, 0] + mtx[2, 0])
    spe1 = (mtx[0, 0] + mtx[0, 2] + mtx[2, 0] + mtx[2, 2]) / (mtx[0, 0] + mtx[0, 2] + mtx[2, 0] + mtx[2, 2] +
                                                              mtx[0, 1] + mtx[2, 1])
    spe2 = (mtx[0, 0] + mtx[0, 1] + mtx[1, 0] + mtx[1, 1]) / (mtx[0, 0] + mtx[0, 1] + mtx[1, 0] + mtx[1, 1] +
                                                              mtx[0, 2] + mtx[1, 2])

    specificity = (spe0 + spe1 + spe2) / 3

    # Calculate sensitivity
    sn0 = (mtx[0, 0]) / (mtx[0, 0] + mtx[0, 1] + mtx[0, 2])
    sn1 = (mtx[1, 1]) / (mtx[1, 1] + mtx[1, 0] + mtx[1, 2])
    sn2 = (mtx[2, 2]) / (mtx[2, 2] + mtx[2, 0] + mtx[2, 1])

    sensitivity = (sn0 + sn1 + sn2) / 3

    # Calculate F1 score
    f_one_0 = 2 * ((pre0 * re0) / (pre0 + re0))
    f_one_1 = 2 * ((pre1 * re1) / (pre1 + re1))
    f_one_2 = 2 * ((pre2 * re2) / (pre2 + re2))

    f_one = (f_one_0 + f_one_1 + f_one_2) / 3

    return [accuracy, precision, recall, specificity, sensitivity, f_one]
