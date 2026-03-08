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
    # "era",
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
    # "njet",
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


dir_path = '/eos/user/a/amorandi/HH4b/NF_HQ/'
os.makedirs(dir_path, exist_ok=True)


train_data, val_data = dl.full_data_loader(
    test_size=0, 
    validation_size=0.25, 
    features=features, 
    seed=42
)

model = md.FlowModel(
    input_dim=len(features), 
    context_dim=1, 
    bins=20, 
    transforms=4, 
    hidden_features=[256, 256], 
    dir_path=dir_path, 
    EarlyStopper_patience=30
)
model.print_model_summary()

if os.path.exists(dir_path + 'weights/flow_model.pth'):
    model.load_model(dir_path + 'weights/flow_model.pth')
else:
    print("Starting training...")
    model.train(train_data, val_data, epochs=9999)
    model.plot_training_loss()

    model.load_model(dir_path + 'weights/flow_model.pth')

# =========================================================================
# Normalizing Flow Model Evaluation
# =========================================================================
test_data_all = dl.full_data_loader(
    bootstrap_coef_2b=1, 
    bootstrap_coef_4b=1, 
    test_size=0, 
    validation_size=0, 
    features=features, 
    seed=42
)

test_model = tm.FFNModelTester(
    model=model, 
    test_data_loader=test_data_all, 
    features=features, 
    dir_path=dir_path
)

dir_eval = dir_path + 'eval_data/'

# 2b to 4b transformation analysis
test_model.transform_2b_to_4b(test_data_all, h5_filename=dir_eval + 'transform_2b_to_4b_scenario')

# Binary classification analysis (2b vs 4b)
test_model.classification_analysis(h5_filename=dir_eval + 'classification_analysis')

# Per-feature likelihood analysis
test_model.per_feature_likelihood(h5_filename=dir_eval + 'per_feature_likelihood_scenario', top_n=len(features))





