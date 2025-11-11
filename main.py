import util  # Import the module
import os
import sys
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch import amp
import time
import copy
from model import FeedForwardNN

########################################################
# Initialization of constant stuff..

global_start_time = time.time()  # Total script start
pd.set_option("display.precision", 5)
pd.set_option("display.width", 200)
torch.backends.cudnn.benchmark = True
scaler = amp.GradScaler(device='cuda')

rm_dates = {'20241004','20241011','20241018','20241025','20241031','20241108','20241114'}
directory = "/workspace/keshav/projects/hftalpha/data/split-000000-093500-0.run1/"
sampling_frac = 0.5
os.chdir(directory)

alpha_headers = ["greek_", "seq_", "pairalpha_", "obalpha_", "revalpha_"]
non_alpha_headers = ["mid","spread","Sspread","Satr","volume_factor"]

########################################################
# Defining hyperparameters..

print(f"Setting parameters..");

config = {
    "hidden_layers": [64, 32],
    "epochs": 30,
    "batch_size": 2048,
    "learning_rate": 0.001,
    "validation_split": 0.2,  # Last x% as validation set
    "normalization": "standard",  # Options: "standard" (Z-score), "minmax" (0-1), "RobustScaling : Uses median and IQR (Interquartile Range)"
    "regularization": {"L1": 0, "L2": 1e-4, "dropout": 0.2},
    "activation": "relu",  # Options: "relu", "tanh", "sigmoid", "leaky_relu", "elu", try with relu or leaky_relu, "tanh", leaky_relu helps when output collapses to a constant
    "save_model": True,  # Save trained models in csv and .pt file
    "early_stop_threshold": 4.0 , # Stop if avg corr drops > this from best
    "min_epochs": 5, # early stopping only allowed after x epochs
    "use_non_predictive_features": True,
    "trimvalue" : 0.9995,
    "constraints": {"monotonic_weight": 1e+2, "shrink_weight": 1e+2, "sum_zero": 1e+2, "bias_shrink": 1e+1}
}

output_size = 3 # ret10, ret30, ret60

########################################################
# constraints on the input features - features to be removed, features wanting negatives weights, pair of features wanting to sum to zero

rm_headers = ["greek_optpair_ret","greek_optpair_Iret","greek_optpairb_ret","greek_optpairb_Iret"]
negative_headers = ["revalpha_self","obalpha_explevel_ssz","obalpha_explevel_ssqrtsz","obalpha_explevel_snorder","obalpha_explevel_snordersqrtsz","obalpha_pxgap_ssz","obalpha_pxgap_ssqrtsz","obalpha_pxgap_snorder","obalpha_pxgap_snordersqrtsz","greek_optpair_ret","greek_siv1_ret","greek_siv2_ret","greek_optpairb_ret","greek_siv1b_ret","greek_siv2b_ret"]
sum_constraint_headers = [["greek_optpair_ret","greek_optpair_Iret"],["greek_siv1_ret","greek_siv1_nIret","greek_siv1_Iretcombo"],["greek_siv2_ret","greek_siv2_nIret","greek_siv2_Iretcombo"],["greek_optpairb_ret","greek_optpairb_Iret"],["greek_siv1b_ret","greek_siv1b_nIret","greek_siv1b_Iretcombo"],["greek_siv2b_ret","greek_siv2b_nIret","greek_siv2b_Iretcombo"]]

########################################################

rets = []
data = util.combine_csv(".", r".*alphasampler.*", cores=16, sampling_fraction=sampling_frac)
unique_dates = sorted(set(data["date"]) - rm_dates)
df = data[data["date"].isin(unique_dates)]

forward_mid_columns = [col for col in df.columns if re.match(r"^mid_\d+\.000s$", col)]

for col in forward_mid_columns:
    x = re.search(r"mid_(\d+)\.000s", col).group(1)  # Extract numeric part
    ret_col_name = f"ret{x}"  # Construct return column name
    df[ret_col_name] = df[col] / df["mid"] - 1   # Compute return
    rets.append(ret_col_name)  # Store return column names

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df_rets = df[rets]*1e4 # key step to scale in 1e4, so that gradient descent doesn't die.

selected_columns = [col for col in data.columns if col.startswith(tuple(alpha_headers))]
if config["use_non_predictive_features"]:
    selected_columns = selected_columns + [col for col in non_alpha_headers if col in data.columns]

df_features = df[selected_columns]
df_features = util.trimtails(df_features, config["trimvalue"])
df_rets = util.trimtails(df_rets, config["trimvalue"])
alpha_columns_ = [col for col in data.columns if col.startswith(tuple(alpha_headers))]
feature_index_map = {name: idx for idx, name in enumerate(df_features.columns)}

X = df_features.values  # Input features
y_all = df_rets.values
y_all = df_rets.iloc[:, -output_size:].values  # Taking last 3 columns (ret10, ret30, ret60)

positive_headers = set(alpha_columns_) - set(negative_headers)

########################################################

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
mean_std_df = pd.DataFrame({ "feature": df_features.columns, "mean": scaler_X.mean_, "std": scaler_X.scale_})
mean_std_df.to_csv("/workspace/keshav/projects/hftalpha/results/normalization_stats.csv", index=False)
print("📄 Normalization stats (mean/std) saved to normalization_stats.csv")

########################################################

# Chronological Split (80% Train, 20% Validation)
split_idx = int(len(X) * (1 - config["validation_split"]))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y_all[:split_idx], y_all[split_idx:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


########################################################

model = FeedForwardNN(input_size=X.shape[1],  hidden_layers=config["hidden_layers"],  output_size=output_size,  reg_config=config["regularization"],  activation=config["activation"])
model = nn.DataParallel(model)
#if torch.cuda.device_count() > 1:
#    print(f"Using {torch.cuda.device_count()} GPUs!", flush=True)
#    model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["regularization"]["L2"])  # L2 Regularization

batch_size = config["batch_size"]  # Use batch_size from config
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)  

########################################################

# Training Loop
epoch_start_time = time.time()  # Training loop start
best_ret10_corr = -np.inf
weights_best_model = copy.deepcopy(model.state_dict())

for epoch in range(config["epochs"]):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        with amp.autocast(device_type='cuda'):
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            # Apply L1 Regularization
            if config["regularization"]["L1"] > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += config["regularization"]["L1"] * l1_norm

            # L2 regularization on biases (excluding input layer bias which is frozen)
            if config["constraints"]["bias_shrink"] > 0:
                bias_l2 = 0.0
                for name, param in model.named_parameters():
                    if "bias" in name and param.requires_grad:  # excludes input_layer.bias
                        bias_l2 += torch.sum(param ** 2)
                loss += config["constraints"]["bias_shrink"] * bias_l2

            # ====== Custom Soft Constraints ======

            # Get first layer weights [output_dim, input_dim]
            input_weights = model.module.input_layer.weight

            # 1. Positive weight constraint
            for feat in positive_headers:
                idx = feature_index_map.get(feat, None)
                if idx is not None: 
                    w = input_weights[:, idx]
                    loss += config["constraints"]["monotonic_weight"] * torch.mean(F.relu(-w))  # penalize negative weights

            # 2. Negative weight constraint
            for feat in negative_headers:
                idx = feature_index_map.get(feat, None)
                if idx is not None: 
                    w = input_weights[:, idx]
                    loss += config["constraints"]["monotonic_weight"] * torch.mean(F.relu(w))  # penalize positive weights

            # 3. Zero-weight constraint (shrink to zero)
            for feat in rm_headers:
                idx = feature_index_map.get(feat, None)
                if idx is not None: 
                    w = input_weights[:, idx]
                    loss += config["constraints"]["shrink_weight"] * torch.mean(w ** 2)

            # 4. Sum-to-zero constraint
            for group in sum_constraint_headers:
                idxs = [feature_index_map[feat] for feat in group if feat in feature_index_map]
                if not idxs:
                    continue  # or handle differently depending on your logic
                w_group = input_weights[:, idxs]  # shape: [output_dim, len(group)]
                w_sum = w_group.sum(dim=1)  # sum across the group
                loss += config["constraints"]["sum_zero"] * torch.mean(w_sum ** 2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Validation
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train_tensor.to(device)).cpu().numpy()
        train_loss = criterion(torch.tensor(train_predictions), y_train_tensor.cpu()).item()
        val_predictions = model(X_val_tensor.to(device)).cpu().numpy()
        val_loss = criterion(torch.tensor(val_predictions), y_val_tensor.cpu()).item()
        mse_train = mean_squared_error(y_train, train_predictions)
        mse_val = mean_squared_error(y_val, val_predictions)
        mae_train = mean_absolute_error(y_train, train_predictions)
        mae_val = mean_absolute_error(y_val, val_predictions)
        r2_train = r2_score(y_train, train_predictions, multioutput='uniform_average')
        r2_val = r2_score(y_val, val_predictions, multioutput='uniform_average')
        corr_train = [np.corrcoef(y_train[:, i], train_predictions[:, i])[0, 1] * 100 for i in range(3)]
        corr_val = [np.corrcoef(y_val[:, i], val_predictions[:, i])[0, 1] * 100 for i in range(3)]

    # Print progress every n epochs
    ret10_corr_val = corr_val[0]
    print(f"\n\n----- Epoch {epoch} -----")
    print("\nSample validation outputs (pred -> true):")
    for i in range(min(10, len(val_predictions))):
        pred = val_predictions[i]
        true = y_val[i]
        print(f"  {i}: {pred} -> {true}")
    print(f"  Train Loss: {train_loss:.8f}, MSE: {mse_train:.8f}, MAE: {mae_train:.8f}, R²: {r2_train:.4f}, Corr: {corr_train}")
    print(f"  Val Loss  : {val_loss:.8f}, MSE: {mse_val:.8f}, MAE: {mae_val:.8f}, R²: {r2_val:.4f}, Corr: {corr_val}")
    print(f"  Corr Val: {ret10_corr_val:.2f}, Best So Far: {best_ret10_corr:.2f}")

    # Only check for early stop after min_epochs
    if epoch >= config["min_epochs"]:
        if ret10_corr_val > best_ret10_corr:
            best_ret10_corr = ret10_corr_val
            weights_best_model = copy.deepcopy(model.state_dict())
        elif (best_ret10_corr - ret10_corr_val) > config["early_stop_threshold"]:
            print(f"🛑 Early stopping at epoch {epoch}: corr dropped by > {config['early_stop_threshold']:.2f}")
            break
    else:
        if ret10_corr_val > best_ret10_corr:
            best_ret10_corr = ret10_corr_val
            weights_best_model = copy.deepcopy(model.state_dict())

print(f"Training completed")

########################################################

with open("/workspace/keshav/projects/hftalpha/results/model_weights_nn.csv", "w") as f:
    input_feature_names = df_features.columns
    f.write(",".join(input_feature_names) + "\n")
    for layer_name, weight_tensor in weights_best_model.items():
        print(weight_tensor.cpu().numpy().shape)
        np.savetxt(f, weight_tensor.cpu().numpy().reshape(-1, 1).T, delimiter=",", header=layer_name, comments="")
print("Weights saved in a single CSV file!")

########################################################

# At the end of training, instead of saving weights as CSV: save in *.pt file which ensures easy loading
if config["save_model"]:
    torch.save({ 'model_state_dict': weights_best_model, 'scaler_mean': scaler_X.mean_, 'scaler_std': scaler_X.scale_, 'input_features': list(df_features.columns), 'config': config }, "/workspace/keshav/projects/hftalpha/results/trained_model.pt")
    print("📦 Entire model and normalization stats saved as .pt file")

########################################################

epoch_total_time = time.time() - epoch_start_time
global_total_time = time.time() - global_start_time
non_training_time = global_total_time - epoch_total_time

print("\n🧠 Training Summary:")
print(f"⏱️  Total time spent: {global_total_time:.2f} seconds")
print(f"🧪 Time spent in training epochs: {epoch_total_time:.2f} seconds")
print(f"⏳ Time spent outside training loop (data prep, setup, eval): {non_training_time:.2f} seconds")

########################################################