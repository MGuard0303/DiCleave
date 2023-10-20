import copy
import datetime
import os
from collections import deque

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
from sklearn import preprocessing


class ModelQ:
    def __init__(self, k):
        self.queue = deque()
        self.k = k

    def __repr__(self):
        return f"{self.queue}"

    def size(self):
        size = len(self.queue)
        return size

    def stack(self, model):
        if len(self.queue) < self.k:
            self.queue.append(model)
        else:
            self.queue.popleft()
            self.queue.append(model)


def one_hot_encoding(dataframe, column, token):
    sigma = token
    temp = []

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(sigma)

    for item in dataframe[column]:
        integer_seq = label_encoder.transform(list(item))
        temp.append(integer_seq)

    temp = np.array(temp)  # Convert python list to ndarray to speed up
    integer_tensor = torch.tensor(temp, dtype=torch.long)
    one_hot_tensor = F.one_hot(integer_tensor)
    one_hot_tensor = torch.transpose(one_hot_tensor, 1, 2)
    one_hot_tensor = one_hot_tensor.type(torch.float)    # Sigmoid output layer requires Float type

    return one_hot_tensor


def confusion_mtx(y_pred, y_true, task):

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


def multi_metrics(mtx):
    if mtx.size() == torch.Size([1, 1]):
        [accuracy, precision, recall, specificity, sensitivity, f_one] = [1, 1, 1, 1, 1, 1]

    if mtx.size() == torch.Size([2, 2]):
        # A tensor confusion matrix should be [0, 0] -> TN, [0, 1] -> FP, [1, 0] -> FN, [1, 1] -> TP
        epsilon = 1e-8
        accuracy = (mtx[1, 1] + mtx[0, 0]) / (mtx[1, 1] + mtx[0, 0] + mtx[0, 1] + mtx[1, 1])

        precision = mtx[1, 1] / (mtx[1, 1] + mtx[0, 1] + epsilon)
        recall = mtx[1, 1] / (mtx[1, 1] + mtx[1, 0] + epsilon)

        specificity = mtx[0, 0] / (mtx[0, 0] + mtx[0, 1] + epsilon)
        sensitivity = mtx[1, 1] / (mtx[1, 1] + mtx[1, 0] + epsilon)

        f_one = 2 * ((precision * recall) / (precision + recall + epsilon))

    if mtx.size() == torch.Size([3, 3]):
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

def save_paras(model, path, filename):
    if not os.path.exists(path):
        os.mkdir(path)

    if path.endswith("/"):
        pth = path + filename
    else:
        pth = path + f"/{filename}"

    torch.save(model.state_dict(), pth)


def print_bar():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=========="*8 + f" {current_time}")


def del_files(path, chars):
    del_file = []
    files = os.listdir(path)

    for file in files:
        if file.find(chars) != -1:
            del_file.append(file)

    for f in del_file:
        del_path = os.path.join(path, f)

        # Check deleting a file
        if os.path.isfile(del_path):
            os.remove(del_path)


def train_step(mdl, inputs, embeds, labels):
    # Set to the training mode, dropout and batch normalization will work under this mode
    mdl.train()
    mdl.optimizer.zero_grad()

    # Forward propagation
    predictions = mdl(inputs, embeds)
    loss = mdl.loss_func(predictions, labels)

    # Backward propagation
    loss.backward()
    mdl.optimizer.step()

    return loss.item(), predictions


@torch.no_grad()
def valid_step(mdl, inputs, embeds, labels):
    # Set to the evaluation mode, dropout and batch normalization will not work
    mdl.eval()

    predictions = mdl(inputs, embeds)
    loss = mdl.loss_func(predictions, labels)

    return loss.item(), predictions


"""Deprecated. Will be deleted in the future."""
def bi_train(mdl, train_loader, valid_loader, epochs, num_mdl=3, tolerance=3, verbose=False):
    print()
    print_bar()
    print(f"{mdl.name}: Start training...")

    best_val_acc = 0
    model_queue = ModelQ(num_mdl)

    TOLERANCE = tolerance
    tol = 0

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0

        # Training step
        if verbose:
            print()
            print("| Training...")
            print(f"| Epoch: {epoch:02}")

        for step_train, (inputs, embeds, labels) in enumerate(train_loader, 1):

            batch_loss, batch_pred = train_step(mdl, inputs, embeds, labels)

            # Batch level report shows batch_loss and batch_acc
            batch_confusion = confusion_mtx(batch_pred, labels, mdl.task)
            batch_acc, batch_pre, batch_rec, batch_spe, batch_sen, batch_f_one = bi_metrics(batch_confusion)

            epoch_loss += batch_loss

            # Print batch level report
            if verbose:
                if step_train % 10 == 0:
                    print_bar()
                    print(f"| Step {step_train:03}")
                    print(f"| Batch Loss: {batch_loss:.4f} | Batch Accuracy = {batch_acc:.4f}")

        # Validate model in certain epoch
        if epoch % 1 == 0:
            # Validation step
            print()
            print("Validating...")

            valid_loss_total = 0.0
            valid_confusion_total = [0, 0, 0, 0]
            pred_list = []
            label_list = []

            for step_valid, (inputs, embeds, labels) in enumerate(valid_loader, 1):

                batch_loss_valid, batch_pred_valid = valid_step(mdl, inputs, embeds, labels)

                # Calculation for validation epoch level reports
                valid_confusion = confusion_mtx(batch_pred_valid, labels, mdl.task)
                valid_confusion_total = np.sum([valid_confusion_total, valid_confusion], axis=0).tolist()
                valid_loss_total += batch_loss_valid

                # Create prediction and label list for one batch
                labels = labels.squeeze(1).tolist()
                predictions = batch_pred_valid.squeeze(1).round().tolist()

                for lbl in labels:
                    label_list.append(lbl)

                for pred in predictions:
                    pred_list.append(pred)

            valid_acc, valid_pre, valid_rec, valid_spe, valid_sen, valid_f_one = bi_metrics(valid_confusion_total)
            valid_mcc = sklearn.metrics.matthews_corrcoef(label_list, pred_list)

            # Preparation for epoch level reports
            avg_training_loss = epoch_loss / step_train
            avg_valid_loss = valid_loss_total / step_valid

            # Select model based on valid accuracy
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_model = copy.deepcopy(mdl)
                model_queue.stack(best_model)

            # Epoch level reports
            print_bar()
            print(f"| Epoch {epoch:02}")
            print(f"| Average Training Loss: {avg_training_loss:.4f}")
            print(f"| Average Valid Loss: {avg_valid_loss:.4f} | Valid Accuracy: {valid_acc:.4f} | Valid Precision: "
                  f"{valid_pre:.4f} | Valid Recall: {valid_rec:.4f} | Valid Specificity: {valid_spe:.4f} | "
                  f"Valid sensitivity: {valid_sen:.4f} | Valid F1-score: {valid_f_one:.4f} | Valid MCC: "
                  f"{valid_mcc:.4f} |")

            # Early-stopping mechanism
            if avg_valid_loss >= 2 * avg_training_loss:
                if tol <= TOLERANCE:
                    tol += 1
                elif tol > TOLERANCE:
                    print()
                    print_bar()
                    print("Stopped by early-stopping")
                    break

    print()
    print_bar()
    print(f"{mdl.name}: Training complete.")

    return model_queue, mdl


"""Deprecated. Will be deleted in the future."""
def multi_train(mdl, train_loader, valid_loader, epochs, num_mdl=3, tolerance=3, verbose=False):
    print()
    print_bar()
    print(f"{mdl.name}: Start training...")

    best_val_acc = 0
    model_queue = ModelQ(num_mdl)

    TOLERANCE = tolerance
    tol = 0

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0

        # Training step
        if verbose:
            print()
            print("Training...")
            print(f"| Epoch: {epoch:02}")

        for step_train, (inputs, embeds, labels) in enumerate(train_loader, 1):
            labels = labels.squeeze(1)  # The shape of NLLLoss label is (N), it's different from BCELoss
            labels = labels.type(torch.long)
            batch_loss, batch_pred = train_step(mdl, inputs, embeds, labels)

            # Batch level reports about batch_acc and batch_loss
            batch_confusion = confusion_mtx(batch_pred, labels, mdl.task)
            batch_acc, batch_precision, batch_recall, batch_specificity, batch_sensitivity, batch_f_one = multi_metrics(batch_confusion)

            epoch_loss += batch_loss

            if verbose:
                if step_train % 10 == 0:
                    print_bar()
                    print(f"| Step {step_train:03}")
                    print(f"| Batch Loss: {batch_loss:.4f} | Batch Accuracy = {batch_acc:.4f}")

        # Validate model at certain epoch
        if epoch % 1 == 0:
            print()
            print("Validating...")

            valid_loss_total = 0.0
            valid_confusion_total = torch.zeros(3, 3, dtype=torch.long)
            pred_list = []
            label_list = []

            for step_valid, (inputs, embeds, labels) in enumerate(valid_loader, 1):
                labels = labels.squeeze(1)
                labels = labels.type(torch.long)

                batch_loss_valid, batch_pred_valid = valid_step(mdl, inputs, embeds, labels)

                valid_loss_total += batch_loss_valid

                # Get the confusion matrix on the whole validation set
                valid_confusion = confusion_mtx(batch_pred_valid, labels, mdl.task)
                valid_confusion_total += valid_confusion

                # Create prediction list and label list for this batch
                labels = labels.tolist()

                pred_temp = []

                for i in batch_pred_valid:
                    pred_temp.append(i.argmax().item())

                for pred in pred_temp:
                    pred_list.append(pred)

                for lbl in labels:
                    label_list.append(lbl)

            # Get acc, precision, recall and f_one from confusion matrix
            valid_acc, valid_precision, valid_recall, valid_specificity, valid_sensitivity, valid_f_one = multi_metrics(valid_confusion_total)

            valid_mcc = sklearn.metrics.matthews_corrcoef(label_list, pred_list)

            avg_training_loss = epoch_loss / step_train
            avg_valid_loss = valid_loss_total / step_valid

            # Select model based on valid accuracy
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_model = copy.deepcopy(mdl)
                model_queue.stack(best_model)

            print_bar()
            print(f"| Epoch {epoch:02}")
            print(f"| Average Training Loss: {avg_training_loss:.4f}")
            print(f"| Average Valid Loss: {avg_valid_loss:.4f} | Valid Accuracy: {valid_acc:.4f} | Valid Precision: "
                  f"{valid_precision:.4f} | Valid Recall: {valid_recall:.4f} | Valid Specificity: "
                  f"{valid_specificity:.4f} | Valid Sensitivity: {valid_sensitivity:.4f} | Valid F1-score: "
                  f"{valid_f_one:.4f} | Valid MCC: {valid_mcc:.4f} |")

            # Early-stopping mechanism
            if avg_valid_loss >= 2 * avg_training_loss:
                if tol <= TOLERANCE:
                    tol += 1
                elif tol > TOLERANCE:
                    print()
                    print_bar()
                    print("Stopped by early-stopping")
                    break

    print()
    print_bar()
    print(f"{mdl.name}: Training complete.")

    return model_queue, mdl


def evaluate(mdl, test_loader, returns=False):
    mdl.eval()

    # Evaluate binary classifier
    if mdl.task == "binary":
        loss_total = 0.0
        confusion_total = [0, 0, 0, 0]
        pred_list = []
        label_list = []
        raw_pred = []

        for test_step, (inputs, embeds, labels) in enumerate(test_loader, 1):
            predictions = mdl(inputs, embeds)
            loss = mdl.loss_func(predictions, labels)
            loss = loss.item()
            loss_total += loss

            # Compute confusion matrix for each batch
            test_confusion = confusion_mtx(predictions, labels, mdl.task)
            confusion_total = np.sum([confusion_total, test_confusion], axis=0).tolist()

            # Create label list and prediction list for one batch
            labels = labels.squeeze(1).tolist()
            labeled_predictions = predictions.squeeze(1).round().tolist()
            raw_predictions = predictions.squeeze(1).tolist()

            # Get list of all predicts and labels of test set
            for lbl in labels:
                label_list.append(lbl)

            for pred in labeled_predictions:
                pred_list.append(pred)

            for rp in raw_predictions:
                raw_pred.append(rp)

        acc, precision, recall, specificity, sensitivity, f_one = bi_metrics(confusion_total)
        mcc = sklearn.metrics.matthews_corrcoef(label_list, pred_list)
        avg_loss = loss_total / test_step

        # Print evaluation result
        print_bar()
        print(f"| Evaluate {mdl.name} on test set")
        print(f"| Average Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: "
              f"{recall:.4f} | Specificity: {specificity:.4f} | Sensitivity: {sensitivity:.4f} | F1-score: {f_one:.4f} "
              f"| MCC: {mcc:.4f} |")

        if returns:
            return raw_pred, pred_list, label_list

    # Test multi-classifier
    if mdl.task == "multi":
        loss_total = 0.0
        pred_list = []
        label_list = []
        raw_pred = []
        confusion_total = torch.zeros(3, 3).type(torch.LongTensor)

        for test_step, (inputs, embeds, labels) in enumerate(test_loader, 1):
            labels = labels.squeeze(1)
            labels = labels.type(torch.long)

            predictions = mdl(inputs, embeds)
            loss = mdl.loss_func(predictions, labels)
            loss = loss.item()
            loss_total += loss

            # Compute confusion matrix for each crop
            test_confusion = confusion_mtx(predictions, labels, mdl.task)
            test_confusion = torch.LongTensor(test_confusion)
            confusion_total += test_confusion

            # Create predict list and label list for this batch
            labels = labels.tolist()
            batch_pred = []

            for i in predictions:
                batch_pred.append(i.argmax().item())

            for lbl in labels:
                label_list.append(lbl)

            for pred in batch_pred:
                pred_list.append(pred)

            for rp in predictions:
                rp = torch.exp(rp).tolist()
                raw_pred.append(rp)

            raw_pred = np.array(raw_pred)

        acc, precision, recall, specificity, sensitivity, f_one = multi_metrics(confusion_total)
        avg_loss = loss_total / test_step

        mcc = sklearn.metrics.matthews_corrcoef(label_list, pred_list)

        # Print evaluation result
        print_bar()
        print(f"| Evaluate {mdl.name} on test set")
        print(f"| Average Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: "
              f"{recall:.4f} | Specificity: {specificity:.4f} | Sensitivity: {sensitivity:.4f} | F1-score: {f_one:.4f} "
              f"| MCC: {mcc:.4f} |")

        if returns:
            return raw_pred, pred_list, label_list


def predict(mode, model, data_loader):
    model.eval()

    # Binary prediction
    if mode == "3" or mode == "5":
        prediction_list = []
        raw_prediction_list = []

        for _, (inputs, embeds) in enumerate(data_loader):
            raw_prediction = model(inputs, embeds)

            labeled_prediction = raw_prediction.squeeze(1).round().tolist()
            raw_prediction = raw_prediction.squeeze(1).tolist()

            # Get list of predictions
            for lp in labeled_prediction:
                lp = int(lp)
                prediction_list.append(lp)

            for rp in raw_prediction:
                raw_prediction_list.append(rp)

        return raw_prediction_list, prediction_list
    

    # Multiple prediction
    if mode == "multi":
        prediction_list = []
        raw_prediction_list = []

        for _, (inputs, embeds) in enumerate(data_loader):
            raw_prediction = model(inputs, embeds)

            # Get list of predictions
            batch_prediction = []

            for i in raw_prediction:
                batch_prediction.append(i.argmax().item())

            for p in batch_prediction:
                p = int(p)
                prediction_list.append(p)

            for rp in raw_prediction:
                rp = torch.exp(rp).tolist()
                raw_prediction_list.append(rp)

        return raw_prediction_list, prediction_list


def train(model, loader_t, loader_v, epoch, k, t):
    print()
    print_bar()
    print(f"{model.name}: Start training...")

    best_val_acc = 0
    model_queue = ModelQ(k)

    TOLERANCE = t
    tol = 0

    for epoch in range(1, epoch+1):
        epoch_loss = 0.0

        # Training step
        for step_t, (inputs, embeds, labels) in enumerate(loader_t, 1):
            if model.task == "multi":
                labels = labels.squeeze(1)
                labels = labels.type(torch.long)
            
            batch_loss, batch_pred = train_step(model, inputs, embeds, labels)
            batch_confusion = confusion_mtx(batch_pred, labels, model.task)
            
            if model.task == "binary":
                batch_acc, _, _, _, _, _ = bi_metrics(batch_confusion)
            elif model.task == "multi":
                batch_acc, _, _, _, _, _ = multi_metrics(batch_confusion)

            epoch_loss += batch_loss

            # Validate model in certain epoch
        if epoch % 1 == 0:
            print()
            print("Validating...")

            valid_loss_total = 0.0
                
            if model.task == "binary":
                valid_confusion_total = [0, 0, 0, 0]
            elif model.task == "multi":
                valid_confusion_total = torch.zeros(3, 3, dtype=torch.long)
                
            pred_list = []
            label_list = []

            for step_v, (inputs, embeds, labels) in enumerate(loader_v, 1):
                if model.task == "multi":
                    labels = labels.squeeze(1)
                    labels = labels.type(torch.long)

                batch_loss_v, batch_pred_v = valid_step(model, inputs, embeds, labels)
                valid_confusion = confusion_mtx(batch_pred_v, labels, model.task)

                if model.task == "binary":
                    valid_confusion_total = np.sum([valid_confusion_total, valid_confusion], axis=0).tolist()
                elif model.task == "multi":
                    valid_confusion_total += valid_confusion

                valid_loss_total += batch_loss_v

                # Create prediction and label list for this batch
                if model.task == "binary":
                    labels = labels.squeeze(1).tolist()
                    predictions = batch_pred_v.squeeze(1).round().tolist()

                    for lbl in labels:
                        label_list.append(lbl)

                    for pred in predictions:
                        pred_list.append(pred)

                if model.task == "multi":
                    labels = labels.tolist()
                        
                    pred_temp = []

                    for i in batch_pred_v:
                        pred_temp.append(i.argmax().item())

                    for lbl in labels:
                        label_list.append(lbl)

                    for pred in pred_temp:
                        pred_list.append(pred)

            if model.task == "binary":
                acc_v, pre_v, rec_v, spe_v, sen_v, f_one_v = bi_metrics(valid_confusion_total)
                mcc_v = sklearn.metrics.matthews_corrcoef(label_list, pred_list)

            elif model.task == "multi":
                acc_v, pre_v, rec_v, spe_v, sen_v, f_one_v = multi_metrics(valid_confusion_total)
                mcc_v = sklearn.metrics.matthews_corrcoef(label_list, pred_list)

            avg_loss_t = epoch_loss / step_t
            avg_loss_v = valid_loss_total / step_v

            # Select model model on valid accuracy
            if acc_v > best_val_acc:
                best_val_acc = acc_v
                best_model = copy.deepcopy(model)
                model_queue.stack(best_model)

            # Epoch level reports
            print_bar()
            print(f"| Epoch {epoch:02}")
            print(f"| Average Training Loss: {avg_loss_t:.4f}")
            print(f"| Average Valid Loss: {avg_loss_v:.4f} | Valid Precision: {pre_v:.4f} | Valid Recall: {rec_v:.4f} | Valid Specificity: {spe_v:.4f} | Valid Sensitivity: {sen_v:.4f} | Valid F1-score: {f_one_v:.4f} | Valid MCC: {mcc_v:.4f} |")

            # Early-stopping
            if avg_loss_v >= 1.5 * avg_loss_t:
                if tol < TOLERANCE:
                    tol += 1
                elif tol >= TOLERANCE:
                    print()
                    print_bar()
                    print("Stopped by early-stopping")
                    break

    print()
    print_bar()
    print()
    print(f"{model.name}: Training complete.")

    return model_queue
