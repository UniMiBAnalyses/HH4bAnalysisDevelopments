import os
import sys
import torch

# Add parent directory to path to import from lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib import data_loader as dl
import model as md
import test_model as tm


# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()


features = [
    "era",
    "HT",
    # "sigma_higgs1", 
    "sigma_over_higgs1_reco_mass",
    "higgs1_reco_pt", "higgs1_reco_eta", 
    # "higgs1_reco_phi", 
    "higgs1_reco_mass",
    # "sigma_higgs2", 
    "sigma_over_higgs2_reco_mass",
    "higgs2_reco_pt", "higgs2_reco_eta", 
    #"higgs2_reco_phi", 
    "higgs2_reco_mass",
    "hh_vec_mass", "hh_vec_pt", "hh_vec_eta", 
    #"hh_vec_phi", "hh_vec_DeltaR", 
    "hh_vec_DeltaPhi", "hh_vec_DeltaEta", 
    # "hh_vec_ptOmass",
    "njet",
    "higgs1_reco_jet1_pt", "higgs1_reco_jet1_eta", "higgs1_reco_jet1_phi", "higgs1_reco_jet1_mass",
    "higgs1_reco_jet2_pt", "higgs1_reco_jet2_eta", "higgs1_reco_jet2_phi", "higgs1_reco_jet2_mass",
    # "higgs1_DeltaPhijj", "higgs1_DeltaEtajj", 
    "higgs1_DeltaRjj",
    "higgs2_reco_jet1_pt", "higgs2_reco_jet1_eta", "higgs2_reco_jet1_phi", "higgs2_reco_jet1_mass",
    "higgs2_reco_jet2_pt", "higgs2_reco_jet2_eta", "higgs2_reco_jet2_phi", "higgs2_reco_jet2_mass",
    # "higgs2_DeltaPhijj", "higgs2_DeltaEtajj", 
    "higgs2_DeltaRjj",
    "minDeltaR_Higgjj", "maxDeltaR_Higgjj",
    # "higgs1_helicityCosTheta", "higgs2_helicityCosTheta",
    # "hh_CosThetaStar_CS"#,
    # "higgs_ST",
    # "jet1pt_pt", "jet2pt_pt", "jet3pt_pt", "jet4pt_pt",
    # "add_jet1pt_pt", "add_jet1pt_eta", "add_jet1pt_phi", "add_jet1pt_mass",
    # "add_jet1pt_Higgs1_deta", "add_jet1pt_Higgs1_dphi", "add_jet1pt_Higgs1_m",
    # "add_jet1pt_Higgs2_deta", "add_jet1pt_Higgs2_dphi", "add_jet1pt_Higgs2_m"
]


dir_path = '/eos/user/a/amorandi/HH4b/FFN_cat_embedding/'
os.makedirs(dir_path, exist_ok=True)

# Load raw data first (before converting to DataLoader)
X, y = dl.data_preparation(
    bootstrap_coef_2b=1,
    bootstrap_coef_4b=None,
    features=features, 
    reshaping_features=[], 
    min_scale=None, 
    max_scale=None
)

# Define categorical features and their properties
categorical_features = ["era", "njet"]
categorical_indices = [features.index(feat) for feat in categorical_features]

# Convert categorical features to integers and remap to start from 0
era_idx = features.index("era")
njet_idx = features.index("njet")

# era is already 0-1, just convert to int
X[:, era_idx] = X[:, era_idx].astype(int)
print(f"era: min={int(X[:, era_idx].min())}, max={int(X[:, era_idx].max())}")

# njet needs to be remapped from [4-15] to [0-11]
njet_min = 4
X[:, njet_idx] = (X[:, njet_idx] - njet_min).astype(int)
print(f"njet (remapped): min={int(X[:, njet_idx].min())}, max={int(X[:, njet_idx].max())}")

# Now split the data
X_train, y_train, X_val, y_val = dl.data_split(
    X, y, 
    test_size=0, 
    validation_size=0.25, 
    seed=42
)

# Convert to DataLoaders
train_data = dl.torch_data(X_train, y_train, batch_size=512, shuffle=True)
val_data = dl.torch_data(X_val, y_val, batch_size=512, shuffle=False) 

model = md.FFN(
    input_dim=len(features), 
    dir_path=dir_path, 
    categorical_features=categorical_indices,  
    categorical_dims=[2, 13],      # era: 2 values (0,1), njet: 13 values (4-16 remapped to 0-12)
    embedding_dim=5,               # embedding dimension for categorical features
    embedding_noise_std=0.1,       # Gaussian noise std for embedding regularization
    EarlyStopper_patience=30
)
model.print_model_summary()

if os.path.exists(dir_path + 'weights/flow_model.pt'):
    model.load_model(dir_path + 'weights/flow_model.pt')
else:
    print("Starting training...")
    model.train(train_data, val_data, epochs=9999, batch_size=512)
    model.plot_training_loss()

    model.load_model(dir_path + 'weights/flow_model.pt')

# =========================================================================
# FFN Model Evaluation
# =========================================================================
# Load test data with categorical features properly formatted and remapped
X_test, y_test = dl.data_preparation(
    bootstrap_coef_2b=1, 
    bootstrap_coef_4b=1, 
    features=features, 
    reshaping_features=[], 
    min_scale=None, 
    max_scale=None 
)

# Convert categorical features to integers and remap
X_test[:, era_idx] = X_test[:, era_idx].astype(int)
X_test[:, njet_idx] = (X_test[:, njet_idx] - njet_min).astype(int)  # Remap njet from [4-15] to [0-11]

# Convert to DataLoader
test_data_all = dl.torch_data(X_test, y_test, batch_size=512, shuffle=False)

test_model = tm.FFNModelTester(
    model=model, 
    test_data_loader=test_data_all, 
    features=features, 
    dir_path=dir_path
)

# 2b to 4b classification pattern analysis
test_model.transform_2b_to_4b(test_data_all, h5_filename='transform_2b_to_4b_scenario')

# Binary classification analysis (2b vs 4b)
test_model.classification_analysis(h5_filename='classification_analysis')

# Per-feature importance analysis
test_model.per_feature_likelihood(h5_filename='per_feature_likelihood_scenario', top_n=len(features))


