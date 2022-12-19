import copy
import sklearn.metrics
import torch
import numpy as np
import metrics
import utility
from utility import print_bar


def train_step(mdl, inputs, embeds, labels):
    # Set to the training mode, dropout and batch normalization will work under this mode
    mdl.train()
    mdl.optimizer.zero_grad()  # Clear the gradient everytime

    # Forward propagation
    predictions = mdl(inputs, embeds)
    loss = mdl.loss_func(predictions, labels)

    # Backward propagation
    loss.backward()
    mdl.optimizer.step()  # Update the parameters of the model

    return loss.item(), predictions  # loss is a one-element tensor, so it can use .item() method


@torch.no_grad()  # This decorator makes following function not calculate gradient
def valid_step(mdl, inputs, embeds, labels):
    # Set to the evaluation mode, dropout and batch normalization will not work
    mdl.eval()

    predictions = mdl(inputs, embeds)
    loss = mdl.loss_func(predictions, labels)

    return loss.item(), predictions


# At the end of training, return the best-k models and the model which pass through all epochs
def bi_train(mdl, train_loader, valid_loader, epochs, verbose=False):
    print()
    print_bar()
    print(f"{mdl.name}: Start training...")

    best_val_acc = 0
    model_queue = utility.ModelQ(3)

    TOLERANCE = 3
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
            batch_confusion = metrics.get_confusion(batch_pred, labels, mdl.task)
            batch_acc, batch_pre, batch_rec, batch_spe, batch_sen, batch_f_one = metrics.bi_metrics(batch_confusion)

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
                valid_confusion = metrics.get_confusion(batch_pred_valid, labels, mdl.task)
                valid_confusion_total = np.sum([valid_confusion_total, valid_confusion], axis=0).tolist()
                valid_loss_total += batch_loss_valid

                # Create prediction and label list for one batch
                labels = labels.squeeze(1).tolist()
                predictions = batch_pred_valid.squeeze(1).round().tolist()

                for lbl in labels:
                    label_list.append(lbl)

                for pred in predictions:
                    pred_list.append(pred)

            valid_acc, valid_pre, valid_rec, valid_spe, valid_sen, valid_f_one = \
                metrics.bi_metrics(valid_confusion_total)
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


# At the end of training, return the best-k models and the model which pass through all epochs
def tri_train(mdl, train_loader, valid_loader, epochs, verbose=False):
    print()
    print_bar()
    print(f"{mdl.name}: Start training...")

    best_val_acc = 0
    model_queue = utility.ModelQ(3)

    TOLERANCE = 3
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
            batch_confusion = metrics.get_confusion(batch_pred, labels, mdl.task)
            batch_acc, batch_precision, batch_recall, batch_specificity, batch_sensitivity, batch_f_one = \
                metrics.tri_metrics(batch_confusion)

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
                labels = labels.squeeze(1)  # The shape of NLLLoss label is (N), it's different from BCELoss
                labels = labels.type(torch.long)

                batch_loss_valid, batch_pred_valid = valid_step(mdl, inputs, embeds, labels)

                valid_loss_total += batch_loss_valid

                # Get the confusion matrix on the whole validation set
                valid_confusion = metrics.get_confusion(batch_pred_valid, labels, mdl.task)
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
            valid_acc, valid_precision, valid_recall, valid_specificity, valid_sensitivity, valid_f_one = \
                metrics.tri_metrics(valid_confusion_total)

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


@torch.no_grad()
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
            test_confusion = metrics.get_confusion(predictions, labels, mdl.task)
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

        acc, precision, recall, specificity, sensitivity, f_one = metrics.bi_metrics(confusion_total)
        # mcc = metrics.get_mcc(confusion_total)
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
            test_confusion = metrics.get_confusion(predictions, labels, mdl.task)
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

        acc, precision, recall, specificity, sensitivity, f_one = metrics.tri_metrics(confusion_total)
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
