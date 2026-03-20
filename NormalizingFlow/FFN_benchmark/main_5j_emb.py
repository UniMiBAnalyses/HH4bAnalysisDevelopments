import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import from lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib import data_loader as dl
import model as md
import test_model as tm

# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()

from lib.features import features

semi_cat_features = [
    "add_jet1pt_pt", "add_jet1pt_eta", "add_jet1pt_phi", "add_jet1pt_mass",
    "add_jet1pt_Higgs1_deta", "add_jet1pt_Higgs1_dphi", #"add_jet1pt_Higgs1_m",
    "add_jet1pt_Higgs2_deta", "add_jet1pt_Higgs2_dphi" #"add_jet1pt_Higgs2_m"
]

features += semi_cat_features

dir_path = '/eos/user/a/amorandi/HH4b/FFN_5j_embedding/'
os.makedirs(dir_path, exist_ok=True)

# Load raw data first (before converting to DataLoader)
X, y = dl.data_preparation(
    bootstrap_coef_2b=1,
    bootstrap_coef_4b=None,
    features=features, 
    apply_scaling=False,
    apply_clipping=False
)
    
# =========================================================================
# SEMICATEGORICAL FEATURES (The Jet)
# =========================================================================
semicategorical_feature_names = [ 
    "add_jet1pt_pt", "add_jet1pt_eta", "add_jet1pt_phi", "add_jet1pt_mass",
    "add_jet1pt_Higgs1_deta", "add_jet1pt_Higgs1_dphi", 
    "add_jet1pt_Higgs2_deta", "add_jet1pt_Higgs2_dphi"
]

# Calculate global indices (for splitting from X) and local indices (for the mask)
semi_cat_global_indices = [features.index(feat) for feat in semicategorical_feature_names]
local_pt_index = semicategorical_feature_names.index("add_jet1pt_pt")
local_jet_feature_indices = list(range(len(semicategorical_feature_names)))

# Plotting
semi_cat_dir_path = dir_path + 'semi_categorical_features/'
os.makedirs(semi_cat_dir_path, exist_ok=True)
for cat_feat in tqdm(semicategorical_feature_names, desc="Plotting semi-categorical"):
    plt.figure(figsize=(6, 4))
    plt.hist(X[:, features.index(cat_feat)], bins=200, alpha=0.7, color='blue')
    plt.title(f"Distribution of {cat_feat}")
    plt.xlabel(cat_feat)
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.xlim(-5.5, None)  
    plt.grid()
    plt.xticks(np.arange(-5, 6, 1))
    plt.savefig(semi_cat_dir_path + f"distribution_{cat_feat}.png")
    plt.close()


X_train, y_train, X_val, y_val = dl.data_split(
    X, y, 
    test_size=0, 
    validation_size=0.25, 
    seed=42
)

train_data = dl.torch_data(X_train, y_train, batch_size=512, shuffle=True)
val_data = dl.torch_data(X_val, y_val, batch_size=512, shuffle=False) 

# =========================================================================
# PLOT ALL FEATURES
# =========================================================================
dir_all = dir_path + 'all_features/'
os.makedirs(dir_all, exist_ok=True)
for i, feat in tqdm(enumerate(features), desc="Plotting all features", total=len(features)):
    plt.figure(figsize=(6, 4))
    plt.hist(X[:, i], bins=200, alpha=0.7, color='green')
    plt.title(f"Distribution of {feat}")
    plt.xlabel(feat)
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.grid()
    plt.savefig(dir_all + f"distribution_{feat}.png")
    plt.close()

# =========================================================================
# MODEL
# =========================================================================
model = md.FFN(
    input_dim=len(features), 
    dir_path=dir_path, 
    semicategorical_features=semi_cat_global_indices, 
    pt_index=local_pt_index,                        
    jet_feature_indices=local_jet_feature_indices,  
    embedding_dim=8,               
    embedding_noise_std=0.1,       
    EarlyStopper_patience=30
)
model.print_model_summary()

if os.path.exists(dir_path + 'weights/flow_model.pth'):
    model.load_model(dir_path + 'weights/flow_model.pth')
else:
    print("Starting training...")
    model.train(train_data, val_data, epochs=999, batch_size=512)
    model.plot_training_loss()
    model.load_model(dir_path + 'weights/flow_model.pth')

# =========================================================================
# MODEL EVALUATION
# =========================================================================
X_test, y_test = dl.data_preparation(
    bootstrap_coef_2b=1, 
    bootstrap_coef_4b=1, 
    features=features,
    apply_scaling=False,
    apply_clipping=False
)

if torch.is_tensor(X_test):
    X_test = X_test.cpu().numpy()

test_data_all = dl.torch_data(X_test, y_test, batch_size=512, shuffle=False)

test_model = tm.FFNModelTester(
    model=model, 
    test_data_loader=test_data_all, 
    features=features, 
    dir_path=dir_path
)

# 2b to 4b classification pattern analysis
test_model.transform_2b_to_4b(test_data_all, h5_filename='transform_2b_to_4b_scenario', use_light_analysis=False)

# Binary classification analysis (2b vs 4b)
test_model.classification_analysis(h5_filename='classification_analysis')

# Per-feature importance analysis
test_model.per_feature_likelihood(h5_filename='per_feature_likelihood_scenario', top_n=len(features))

# Reweighting pattern analysis
test_model.reweighting_patterns(h5_filename='reweighting_patterns')