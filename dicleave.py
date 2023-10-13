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
    parser.add_argument("--input_path", "-i", type=str, help="Path of input dataset.")
    parser.add_argument("--data_index", "-di",type=tuple, help="Column index of input dataset. The index order should be 'dot-bracket of full-length sequence', 'cleavage pattern', , 'complementory sequence', 'secondary structure of pattern'.")
    parser.add_argument("--output_path", "-o", type=str, help="Path of output file")

    args = parser.parse_args()
    print(args)

    ae_para = Path("./paras/autoencoder.pt")
    ae = model.AutoEncoder()
    ae.load_state_dict(torch.load(ae_para))

    df_path = Path(args.input_path)
    df = pd.read_csv(df_path, index_col=0)
    df = df.iloc[:, [int(args.data_index[0]), int(args.data_index[1]), int(args.data_index[2]), int(args.data_index[3])]]
    """
    Columns oredr must be:
    0. Dot-bracket secondary structure
    1. Cleavage pattern
    2. complemetary sequence
    3. Dot-bracket cleavage pattern
    """
    column_name = df.columns
    
    sec_struc = df[f"{column_name[0]}"].copy()
    sec_struc = pd.DataFrame({"sec_struc": sec_struc})

    for i in sec_struc.index:
        sec_struc.sec_struc[i] = sec_struc.sec_struc[i].ljust(200, "N")
    
    sec_struc = dc.one_hot_encoding(sec_struc, "sec_struc", [".", "(", ")", "N"])

    ae.eval()
    with torch.no_grad():
        embedding, _, _ = ae(sec_struc)

    pattern = dc.one_hot_encoding(df, f"{column_name[1]}", ["A", "C", "G", "U", "O"])
    complementory = dc.one_hot_encoding(df, f"{column_name[2]}", ["A", "C", "G", "U", "O"])
    pat_sec = dc.one_hot_encoding(df, f"{column_name[3]}", [".", "(", ")", "N"])
    input_tensor = torch.cat([pattern, complementory, pat_sec], dim=1)
    ds = TensorDataset(input_tensor, embedding)
    dl = DataLoader(ds, batch_size=len(ds), shuffle=False)

    # Initial models
    if args.mode == "3":
        model_para = Path("./paras/model_3p.pt")
        mdl = model.CNNModel(task="binary", name="3 prime binary model")
        mdl.load_state_dict(torch.load(model_para))
    elif args.mode == "5":
        model_para = Path("./paras/model_5p.pt")
        mdl = model.CNNModel(task="binary", name="5 prime binary model")
        mdl.load_state_dict(torch.load(model_para))
    elif args.mode == "multi":
        model_para = Path("./paras/model_multi.pt")
        mdl = model.CNNModel(task="multi", name="multiple classification model")
        w = torch.tensor([0.5, 1, 1])
        mdl.loss_func = torch.nn.NLLLoss(weight=w)
        mdl.load_state_dict(torch.load(model_para))


    raw_pred, pred = dc.predict(args.mode, mdl, dl)

    with open(os.path.join(f"{args.output_path}", "output.txt"), "w") as f:
        if args.mode == "3" or args.mode == "5":
            f.write(f"DiCleave prediction of {args.mode}' arm pattern.\n")
            f.write("0 indicates negative pattern, 1 indicates positive pattern.\n")
        elif args.mode == "multi":
            f.write("DiCleave prediction.\n")
            f.write("0 indicates negative pattern, 1 indicates positive pattern from 5' arm, 2 indicates positive pattern from 3' arm.\n")
        
        for i in pred:
            f.write(f"{str(i)}\n")

