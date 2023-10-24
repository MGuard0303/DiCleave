import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import dc
import model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, help="DiCleave mode, should be '3', '5' or 'multi'.")
    parser.add_argument("--input_file", "-i", type=str, help="Path of input dataset.")
    parser.add_argument("--data_index", "-di", type=tuple, help="Column index of input dataset. The index order should be 'dot-bracket of full-length sequence', 'cleavage pattern', 'complementary sequence', 'secondary structure of pattern', and 'label'.")
    parser.add_argument("--output_file", "-o", type=str, help="Path of output file.")
    parser.add_argument("--valid_ratio", "-vr", type=float, default=0.1, help="The ratio of valid set in dataset, default is 0.1.")
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="Batch size during training, default is 20.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.005, help="Learning rate of optimizer, default is 0.005.")
    parser.add_argument("--weight_decay", "-wd", type=float, default=0.001, help="Weight decay parameter of optimizer, default is 0.001")
    parser.add_argument("--nll_weight", "-nw", type=float, nargs="+", default=[1, 1, 1], help="Weight of each class in NLLLoss function. Should be a list with three elements, the first element represents negative label (i.e. label=0).")
    parser.add_argument("--max_epoch", "-me", type=int, default=75, help="Max epoch of training process, default is 75.")
    parser.add_argument("-k", type=int, default=3, help="Top-k models is outputed after training, default is 3, meaning the training process will output 3 best models on validation set.")
    parser.add_argument("--tolerance", "-t", type=int, default=3, help="Tolerance for overfitting, default is 3. The higher the value, it is more likely to overfitting.")

    args = parser.parse_args()

    EPOCH = args.max_epoch
    K = args.k
    TOL = args.tolerance


    ae_para = Path("./paras/autoencoder.pt")
    ae = model.AutoEncoder()
    ae.load_state_dict(torch.load(ae_para))

    df_path = Path(args.input_file)
    df = pd.read_csv(df_path)
    df = df.iloc[:, [int(args.data_index[i]) for i in range(5)]]
    """
    Columns order must be:
    0. Dot-bracket secondary structure
    1. Cleavage pattern sequence
    2. Complementary sequence
    3. Dot-bracket cleavage pattern
    4. Label
    """

    df_v = df.sample(frac=0.1)
    df_t = df.loc[~df.index.isin(df_v.index)]

    column_name = df.columns
    
    sec_struc_t = df_t[f"{column_name[0]}"].copy()
    sec_struc_v = df_v[f"{column_name[0]}"].copy()
    sec_struc_t = pd.DataFrame({"sec_struc": sec_struc_t})
    sec_struc_v = pd.DataFrame({"sec_struc": sec_struc_v})

    for i in sec_struc_t.index:
        sec_struc_t.sec_struc[i] = sec_struc_t.sec_struc[i].ljust(200, "N")

    for j in sec_struc_v.index:
        sec_struc_v.sec_struc[j] = sec_struc_v.sec_struc[j].ljust(200, "N")

    sec_struc_t = dc.one_hot_encoding(sec_struc_t, "sec_struc", [".", "(", ")", "N"])
    sec_struc_v = dc.one_hot_encoding(sec_struc_v, "sec_struc", [".", "(", ")", "N"])

    ae.eval()
    with torch.no_grad():
        embedding_t, _, _ = ae(sec_struc_t)
        embedding_v, _, _ = ae(sec_struc_v)

    pattern_t = dc.one_hot_encoding(df_t, f"{column_name[1]}", ["A", "C", "G", "U", "O"])
    complementary_t = dc.one_hot_encoding(df_t, f"{column_name[2]}", ["A", "C", "G", "U", "O"])
    pat_sec_t = dc.one_hot_encoding(df_t, f"{column_name[3]}", [".", "(", ")"])
    label_t = torch.tensor(df_t[f"{column_name[4]}"].to_numpy(), dtype=torch.float).unsqueeze(1)
    input_t = torch.cat([pattern_t, complementary_t, pat_sec_t], dim=1)

    pattern_v = dc.one_hot_encoding(df_v, f"{column_name[1]}", ["A", "C", "G", "U", "O"])
    complementary_v = dc.one_hot_encoding(df_v, f"{column_name[2]}", ["A", "C", "G", "U", "O"])
    pat_sec_v = dc.one_hot_encoding(df_v, f"{column_name[3]}", [".", "(", ")"])
    label_v = torch.tensor(df_v[f"{column_name[4]}"].to_numpy(), dtype=torch.float).unsqueeze(1)
    input_v = torch.cat([pattern_v, complementary_v, pat_sec_v], dim=1)

    ds_t = TensorDataset(input_t, embedding_t, label_t)
    ds_v = TensorDataset(input_v, embedding_v, label_v)
    dl_t = DataLoader(ds_t, batch_size=args.batch_size, shuffle=True)
    dl_v = DataLoader(ds_v, batch_size=len(ds_v), shuffle=False)

    # Initial models
    if args.mode == "3":
        mdl = model.CNNModel(task="binary", name="3 prime binary model")
        mdl.loss_func = torch.nn.BCELoss()
        mdl.optimizer = torch.optim.Adam(mdl.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.mode == "5":
        mdl = model.CNNModel(task="binary", name="5 prime binary model")
        mdl.loss_func = torch.nn.BCELoss()
        mdl.optimizer = torch.optim.Adam(mdl.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.mode == "multi":
        mdl = model.CNNModel(task="multi", name="multiple classification model")
        w = torch.tensor(args.nll_weight, dtype=torch.float)
        mdl.loss_func = torch.nn.NLLLoss(weight=w)
        mdl.optimizer = torch.optim.Adam(mdl.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_models = dc.train(mdl, dl_t, dl_v, EPOCH, K, TOL)
        
    for index, model in enumerate(best_models.queue, 1):
        dc.save_paras(mdl, args.output_file, f"model_{index}.pt")
