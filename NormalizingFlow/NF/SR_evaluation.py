import os
import sys
import torch

# Add parent directory to path to import from lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import model as nf
import model_phi as nfp
from lib import data_loader as dl
import test_model as tm


# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()


from lib.features import features


dir_path_evaluation = '/eos/user/a/amorandi/HH4b/SR_evaluation/'
os.makedirs(dir_path_evaluation, exist_ok=True)

batch_size = 2048
_, _, scaler = dl.data_preparation(
    region="CR",
    bootstrap_coef_2b=1, 
    bootstrap_coef_4b=1, 
    test_size=0, 
    validation_size=0, 
    features=features
)

test_data_all, _ = dl.full_data_loader(
    region="SR",
    bootstrap_coef_2b=1, 
    bootstrap_coef_4b=1, 
    test_size=0, 
    validation_size=0, 
    features=features,
    scaler=scaler,
    batch_size=batch_size
)

# # =========================================================================
# # NF Model
# # =========================================================================
# dir_path_NF = '/eos/user/a/amorandi/HH4b/NF_models/NF/'
# os.makedirs(dir_path_NF, exist_ok=True)
# dir_path_evaluation_NF = dir_path_evaluation + 'NF/'
# os.makedirs(dir_path_evaluation_NF, exist_ok=True)

# NF = nf.FlowModel(
#     input_dim=len(features), 
#     context_dim=1, 
#     bins=20, 
#     transforms=4, 
#     hidden_features=[256, 256], 
#     dir_path=dir_path_NF, 
# )

# NF.print_model_summary()
# NF.load_model(dir_path_NF + 'weights/flow_model.pth')

# test_model_NF = tm.ModelTester(
#     model=NF, 
#     test_data_loader=test_data_all, 
#     features=features, 
#     dir_path=dir_path_evaluation_NF
# )

# dir_eval = dir_path_evaluation_NF + 'eval_data/'

# test_model_NF.transform_2b_to_4b(test_data_all, h5_filename=dir_eval + 'transform_2b_to_4b_scenario', use_light_analysis=False)
# test_model_NF.classification_analysis(h5_filename=dir_eval + 'classification_analysis')
# test_model_NF.per_feature_likelihood(h5_filename=dir_eval + 'per_feature_likelihood_scenario', top_n=len(features))
# test_model_NF.reweighting_patterns(h5_filename=dir_eval + 'reweighting_patterns')
# test_model_NF.feature_shift_analysis(test_data_all, h5_filename=dir_eval + 'feature_shift_analysis')

# =========================================================================
# NF|\phi Model
# =========================================================================
dir_path_NF_phi = '/eos/user/a/amorandi/HH4b/NF_models/NF_phi_conditioning/'
os.makedirs(dir_path_NF_phi, exist_ok=True)
dir_path_evaluation_NF_phi = dir_path_evaluation + 'NF_phi_conditioning/'
os.makedirs(dir_path_evaluation_NF_phi, exist_ok=True)

# Define phi_indices for angular features (if any)
phi_variables = [
    "higgs1_reco_jet1_phi", "higgs1_reco_jet2_phi",
    "higgs2_reco_jet1_phi", "higgs2_reco_jet2_phi"
]
phi_indices = [features.index(var) for var in phi_variables if var in features]

NF_phi = nfp.FlowModel(
    input_dim=len(features),
    context_dim=1,
    bins=20,
    transforms=4,
    hidden_features=[256, 256],
    dir_path=dir_path_NF_phi,
    phi_indices=phi_indices,
    phi_bins=20,
    phi_transforms=4,
    phi_hidden_features=[64, 64],
    EarlyStopper_patience=30
)

NF_phi.print_model_summary()
NF_phi.load_model(dir_path_NF_phi + 'weights/flow_model.pth')

test_model_NF_phi = tm.ModelTester(
    model=NF_phi, 
    test_data_loader=test_data_all, 
    features=features, 
    dir_path=dir_path_evaluation_NF_phi
)

dir_eval = dir_path_evaluation_NF_phi + 'eval_data/'

# test_model_NF_phi.transform_2b_to_4b(test_data_all, h5_filename=dir_eval + 'transform_2b_to_4b_scenario', use_light_analysis=False)
test_model_NF_phi.classification_analysis(h5_filename=dir_eval + 'classification_analysis')
test_model_NF_phi.per_feature_likelihood(h5_filename=dir_eval + 'per_feature_likelihood_scenario', top_n=len(features))
test_model_NF_phi.reweighting_patterns(h5_filename=dir_eval + 'reweighting_patterns')
test_model_NF_phi.feature_shift_analysis(test_data_all, h5_filename=dir_eval + 'feature_shift_analysis')


