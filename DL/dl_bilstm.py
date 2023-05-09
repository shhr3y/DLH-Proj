import csv
import sys
import os
import pandas as pd
import argparse


module_path = os.path.join(os.chdir('/Users/shreygupta/Documents/Classes/CS598DLH/'))
sys.path.append(module_path)
from feature_generation import FeatureGeneration
os.chdir(module_path)
import collections

def training(layer_1, layer_2, n_split, epochs, X, Y):
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import f1_score
   
    class BiLSTM(nn.Module):
        def __init__(self, input_size, layer_1, layer_2, no_layers, output_size):
            super(BiLSTM, self).__init__()
            self.layer_1 = layer_1
            self.layer_2 = layer_2
            self.no_layers = no_layers
            self.bilstm1 = nn.LSTM(input_size, layer_1, no_layers, batch_first=True, bidirectional=True)
            self.bilstm2 = nn.LSTM(layer_1*2, layer_2, no_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(2*layer_2, output_size)

        def forward(self, x):
            h0_1 = torch.zeros(self.no_layers*2, x.size(0), self.layer_1, dtype=torch.float32).to(x.device)
            c0_1 = torch.zeros(self.no_layers*2, x.size(0), self.layer_1, dtype=torch.float32).to(x.device)
            out, _ = self.bilstm1(x, (h0_1, c0_1))
            # print(out.shape)
            h0_2 = torch.zeros(self.no_layers*2, x.size(0), self.layer_2, dtype=torch.float32).to(x.device)
            c0_2 = torch.zeros(self.no_layers*2, x.size(0), self.layer_2, dtype=torch.float32).to(x.device)
            out, _ = self.bilstm2(out, (h0_2, c0_2))
            # print(out.shape)
            out = self.fc(out[:, -1, :])
            return out
    
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    # print(X.shape)

    #Model parameters
    input_size = X.size(2)
    layer_1 = layer_1 #Following the paper
    layer_2 = layer_2 #Following the paper
    no_layers = 1
    output_size = 1

    f1_macro_list = []
    f1_micro_list = []

    n_split = n_split #Mentioned in the paper
    # Train model    
    skf = KFold(split=n_split, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, Y):
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        bilstm = BiLSTM(input_size, layer_1, layer_2, no_layers, output_size)


        if torch.cuda.is_available():
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
            X_val = X_val.cuda()
            Y_val = Y_val.cuda()
            bilstm = bilstm.cuda()
            bilstm = torch.nn.DataParallel(bilstm)

        # print(np.unique(Y_train))
        class_counter = collections.Counter(np.array(Y_train).tolist())
        weight  = class_counter[0.0]/class_counter[1.0]

        if (0.0 not in class_counter.keys()) or (1.0 not in class_counter.keys()):
            f1_micro = 1
            f1_macro = 1
        else:
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
            optimizer = torch.optim.Adam(bilstm.parameters(), lr=0.01)
            bilstm.train()
            for epoch in range(epochs):
                # Forward pass
                outputs = bilstm(X_train)
                # print(outputs.shape, Y_train.unsqueeze(1).shape)
                loss = criterion(outputs, Y_train.unsqueeze(1))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch} loss is {loss}")

            bilstm.eval()
            with torch.no_grad():
                y_hat = bilstm(X_val)
            y_hat = y_hat.view(y_hat.shape[0])

            # print(Y_val.shape, y_hat.shape)

            y_pred = []
            for val in y_hat.data:
                if val <= 0.6:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
                    
            y_pred = torch.tensor(y_pred)

            f1_macro = f1_score(Y_val.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
            f1_micro = f1_score(Y_val.cpu().numpy(), y_pred.cpu().numpy(), average='micro')
        # print(f"The f1 macro score is {f1_macro} and f1_micro score is {f1_micro}")

        f1_macro_list.append(f1_macro)
        f1_micro_list.append(f1_micro)

    f1_macro = np.mean(f1_macro_list)
    f1_micro = np.mean(f1_micro_list)
    print(f"The f1 macro score is {f1_macro} and f1_micro score is {f1_micro}")

    return f1_macro, f1_micro


def main(layer_1, layer_2, split, epochs):
    all_f1_macro_scores = []
    all_f1_micro_scores = []
    morbidities = ['Diabetes', 'Gallstones', 'GERD', 'Gout', 'Hypercholesterolemia', 'Hypertension', 'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous_Insufficiency']

    column_headings = ["Morbidity Class", "DL_Macro F1", "DL_Micro F1"]

    with open("./results/word-embedding/performance_DL.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_headings)
    
    for morbidity in morbidities[:1]:
        train_preprocessed_df = pd.read_csv('./dataset/train/train_intuitive_preprocessed.csv')
        train_preprocessed_df = train_preprocessed_df[train_preprocessed_df[morbidity].isin([1.0, 0.0])]

        X, Y, words = FeatureGeneration(train_preprocessed_df, morbidity).word2vec()
    
        f1_macro, f1_micro = training(layer_1, layer_2, split, epochs, X, Y)
        data = [f1_macro, f1_micro]
        all_f1_macro_scores.append(f1_macro)
        all_f1_micro_scores.append(f1_micro)

        row_heading = morbidity
        with open("./results/word-embedding/performance_DL.csv", "a", newline="") as file:
            writer = csv.writer(file)
            row = [row_heading]
            row.extend(data)
            writer.writerow(row)

    with open("./results/word-embedding/performance_DL.csv", "a", newline="") as file:
        writer = csv.writer(file)
        row = ["Overall-Average"]
        row.extend([sum(all_f1_macro_scores)/len(all_f1_macro_scores),  sum(all_f1_micro_scores)/len(all_f1_micro_scores) ])
        writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Validate DL model')

    parser.add_argument('--epochs', '-e', type=str, help='Number of epochs')
    parser.add_argument('--split', '-ns', type=str, help='Number of splits for K fold validation')
    parser.add_argument('--layer_1', '-hs1', type=str, help='Hidden size1 of Biltsm layer 1')
    parser.add_argument('--layer_2', '-hs2', type=str, help='Hidden size2 of Biltsm layer 2')

    # Parse the arguments
    args = parser.parse_args()
    layer_1 = 20
    layer_2 = 10
    split = 10
    epochs = 10

    # Access the arguments
    if args.epochs:
        epochs = int(args.epochs)
    if args.layer_1:
        layer_1 = int(args.layer_1)
    if args.layer_2:
        layer_2 = int(args.layer_2)
    if args.split:
        split = int(args.split)

    main(layer_1, layer_2, split, epochs)