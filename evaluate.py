from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from torch.utils.data import TensorDataset, DataLoader

import model
import dc


@torch.no_grad()
def verify(mdl, test_loader, mode):
    mdl.eval()

    if mode == "binary":
        loss_total = 0.0
        confusion_total = [0, 0, 0, 0]
        pred_list = []
        label_list = []

    
        for step, (inputs, embeds, labels) in enumerate(test_loader, 1):
            predictions = mdl(inputs, embeds)
            loss = mdl.loss_func(predictions, labels)
            loss = loss.item()
            loss_total += loss

            # Compute confusion matrix for each batch
            confusion = dc.confusion_mtx(predictions, labels, mdl.task)
            confusion_total = np.sum([confusion_total, confusion], axis=0).tolist()

            # Create label list and prediction list for one batch
            labels = labels.squeeze(1).tolist()
            labeled_predictions = predictions.squeeze(1).round().tolist()

            for lbl in labels:
                label_list.append(lbl)

            for pred in labeled_predictions:
                pred_list.append(pred)


        acc, pre, rec, spe, sen, f1 = dc.bi_metrics(confusion_total)
        mcc = sklearn.metrics.matthews_corrcoef(label_list, pred_list)
        avg_loss = loss_total / step


        # Print results
        dc.print_bar()
        print(f"Average Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | Precision: {pre:.4f} | Recall: {rec:.4f} | Specificity: {spe:.4f} | Sensitivity: {sen:.4f} | F1-score: {f1:.4f} | MCC: {mcc:.4f} |")


    if mode == "multi":
        loss_total = 0.0
        pred_list = []
        label_list = []
        confusion_total = torch.zeros(3, 3, dtype=torch.long)


        for step, (inputs, embeds, labels) in enumerate(test_loader, 1):
            labels = labels.squeeze(1)
            labels = labels.type(torch.long)

            predictions = mdl(inputs, embeds)
            loss = mdl.loss_func(predictions, labels)
            loss = loss.item()
            loss_total += loss


            # Compute confusion matrix for each batch
            confusion = dc.confusion_mtx(predictions, labels, mdl.task)
            confusion = torch.LongTensor(confusion)
            confusion_total += confusion

            # Create predict list and label list for this batch
            labels = labels.tolist()
            batch_pred = []

            for i in predictions:
                batch_pred.append(i.argmax().item())

            for pred in batch_pred:
                pred_list.append(pred)

            for lbl in labels:
                label_list.append(lbl)


        acc, pre, rec, spe, sen, f1 = dc.multi_metrics(confusion_total)
        avg_loss = loss_total / step
        mcc = sklearn.metrics.matthews_corrcoef(label_list, pred_list)


        # Print results
        dc.print_bar
        print(f"Average Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | Precision: {pre:.4f} | Recall: {rec:.4f} | Specificity: {spe:.4f} | Sensitivity: {sen:.4f} | F1-score: {f1:.4f} | MCC: {mcc:.4f} |")



if __name__ == "__main__":
    mode = sys.argv[1]

    ae_para = Path("./paras/autoencoder.pt")
    ae = model.AutoEncoder()
    ae.load_state_dict(torch.load(ae_para))

    if mode == "binary":
        df_path_5p = Path("./dataset/test/test_5p.csv")
        df_path_3p = Path("./dataset/test/test_3p.csv")
        mdl_para_5p = Path("./paras/model_5p.pt")
        mdl_para_3p = Path("./paras/model_3p.pt")

        # Initialize models
        model_5p = model.CNNModel(task=mode, name="model_5p")
        model_5p.loss_func = torch.nn.BCELoss()
        model_5p.load_state_dict(torch.load(mdl_para_5p))

        model_3p = model.CNNModel(task=mode, name="model_3p")
        model_3p.loss_func = torch.nn.BCELoss()
        model_3p.load_state_dict(torch.load(mdl_para_3p))


        # Prepare data
        df_5p = pd.read_csv(df_path_5p, index_col=0).sample(frac=1.0).reset_index(drop=True)
        df_5p = df_5p[["dot_bracket", "cleavage_window", "window_dot_bracket", "cleavage_window_comp", "label"]]
        sec_struc_5p = df_5p[["dot_bracket"]].copy()
        
        df_3p = pd.read_csv(df_path_3p, index_col=0).sample(frac=1.0).reset_index(drop=True)
        df_3p = df_3p[["dot_bracket", "cleavage_window", "window_dot_bracket", "cleavage_window_comp", "label"]]
        sec_struc_3p = df_3p[["dot_bracket"]].copy()

        for i in sec_struc_5p.index:
            sec_struc_5p.dot_bracket[i] = sec_struc_5p.dot_bracket[i].ljust(200, "N")
        
        for j in sec_struc_3p.index:
            sec_struc_3p.dot_bracket[j] = sec_struc_3p.dot_bracket[j].ljust(200, "N")

        sec_struc_5p = dc.one_hot_encoding(sec_struc_5p, "dot_bracket", [".", "(", ")", "N"])
        sec_struc_3p = dc.one_hot_encoding(sec_struc_3p, "dot_bracket", [".", "(", ")", "N"])

        ae.eval()
        
        with torch.no_grad():
            embedding_5p, _, _ = ae(sec_struc_5p)
            embedding_3p, _, _ = ae(sec_struc_3p)

        seq_5p = dc.one_hot_encoding(df_5p, "cleavage_window", ["A", "C", "G", "U", "O"])
        comp_5p = dc.one_hot_encoding(df_5p, "cleavage_window_comp", ["A", "C", "G", "U", "O"])
        dot_5p = dc.one_hot_encoding(df_5p, "window_dot_bracket", [".", "(", ")"])
        label_5p = torch.tensor(df_5p.label, dtype=torch.float).unsqueeze(1)
        input_5p = torch.cat([seq_5p, comp_5p, dot_5p], dim=1)
        ds_5p = TensorDataset(input_5p, embedding_5p, label_5p)
        dl_5p = DataLoader(ds_5p, batch_size=len(ds_5p), shuffle=False)

        seq_3p = dc.one_hot_encoding(df_3p, "cleavage_window", ["A", "C", "G", "U", "O"])
        comp_3p = dc.one_hot_encoding(df_3p, "cleavage_window_comp", ["A", "C", "G", "U", "O"])
        dot_3p = dc.one_hot_encoding(df_3p, "window_dot_bracket", [".", "(", ")"])
        label_3p = torch.tensor(df_3p.label, dtype=torch.float).unsqueeze(1)
        input_3p = torch.cat([seq_3p, comp_3p, dot_3p], dim=1)
        ds_3p = TensorDataset(input_3p, embedding_3p, label_3p)
        dl_3p = DataLoader(ds_3p, batch_size=len(ds_3p), shuffle=False)

        # verify result
        print("Verify 5' model on test set")
        verify(model_5p, dl_5p, mode)
        print()
        print("Verify 3' model on test set")
        verify(model_3p, dl_3p, mode)


    if mode == "multi":
        df1_path = Path("./dataset/test/test_multi_positive.csv")
        df2_path = Path("./dataset/test/test_multi_negative_5p.csv")
        df3_path = Path("./dataset/test/test_multi_negative_3p.csv")
        mdl_para = Path("./paras/model_multi.pt")


        model_multi = model.CNNModel(task=mode, name="model_multi")
        w = torch.tensor([0.5, 1, 1], dtype=torch.float)
        model_multi.loss_func = torch.nn.NLLLoss(weight=w)
        model_multi.load_state_dict(torch.load(mdl_para))

        
        df1 = pd.read_csv(df1_path, index_col=0)
        df2 = pd.read_csv(df2_path, index_col=0)
        df3 = pd.read_csv(df3_path, index_col=0)
        df = pd.concat([df1, df2, df3], ignore_index=True).sample(frac=1.0).reset_index(drop=True)
        sec_struc = df[["dot_bracket"]].copy()

        for i in sec_struc.index:
            sec_struc.dot_bracket[i] = sec_struc.dot_bracket[i].ljust(200, "N")

        sec_struc = dc.one_hot_encoding(sec_struc, "dot_bracket", [".", "(", ")", "N"])

        ae.eval()

        with torch.no_grad():
            embedding, _, _ = ae(sec_struc)

        seq = dc.one_hot_encoding(df, "cleavage_window", ["A", "C", "G", "U", "O"])
        comp = dc.one_hot_encoding(df, "cleavage_window_comp", ["A", "C", "G", "U", "O"])
        dot = dc.one_hot_encoding(df, "window_dot_bracket", [".", "(", ")"])
        label = torch.tensor(df.label, dtype=torch.float).unsqueeze(1)
        input_multi = torch.cat([seq, comp, dot], dim=1)
        ds = TensorDataset(input_multi, embedding, label)
        dl = DataLoader(ds, batch_size=len(ds), shuffle=False)


        # Verify results
        print("Verify multiple model on test set")
        verify(model_multi, dl, mode)
