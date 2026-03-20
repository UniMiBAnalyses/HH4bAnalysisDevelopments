import os
import sys
import torch

# Add parent directory to path to import from lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import model as ffn
from lib import data_loader as dl
import test_model as tm


# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()


from lib.features import features


dir_path_evaluation = '/eos/user/a/amorandi/HH4b/SR_evaluation/'
os.makedirs(dir_path_evaluation, exist_ok=True)

batch_size = 2048
test_data_all, scaler = dl.full_data_loader(
    region="SR",
    bootstrap_coef_2b=1, 
    bootstrap_coef_4b=1, 
    test_size=0, 
    validation_size=0, 
    features=features,
    batch_size=batch_size,
    apply_scaling=False,
    apply_clipping=False
)

# # =========================================================================
# # FFN Model
# # =========================================================================
# dir_path_FFN = '/eos/user/a/amorandi/HH4b/FFN_models/FFN/'
# os.makedirs(dir_path_FFN, exist_ok=True)
# dir_path_evaluation_FFN = dir_path_evaluation + 'FFN/'
# os.makedirs(dir_path_evaluation_FFN, exist_ok=True)

# FFN = ffn.FFN(
#     input_dim=len(features), 
#     dir_path=dir_path_FFN
# )

# FFN.print_model_summary()
# FFN.load_model(dir_path_FFN + 'weights/flow_model.pth')

# test_model = tm.FFNModelTester(
#     model=FFN, 
#     test_data_loader=test_data_all, 
#     features=features, 
#     dir_path=dir_path_evaluation_FFN
# )

# dir_eval = dir_path_evaluation_FFN + 'eval_data/'

# test_model.transform_2b_to_4b(test_data_all, h5_filename=dir_eval + 'transform_2b_to_4b_scenario', use_light_analysis=False)
# test_model.classification_analysis(h5_filename=dir_eval + 'classification_analysis')
# test_model.per_feature_likelihood(h5_filename=dir_eval + 'per_feature_likelihood_scenario', top_n=len(features))
# test_model.reweighting_patterns(h5_filename=dir_eval + 'reweighting_patterns')

# =========================================================================
# FFN continuous Model
# =========================================================================
dir_path_FFN_continuous = '/eos/user/a/amorandi/HH4b/FFN_models/FFN_5j_continuos/'
os.makedirs(dir_path_FFN_continuous, exist_ok=True)
dir_path_evaluation_FFN_continuous = dir_path_evaluation + 'FFN_continuos/'
os.makedirs(dir_path_evaluation_FFN_continuous, exist_ok=True)

semi_cat_features = [
    "add_jet1pt_pt", "add_jet1pt_eta", "add_jet1pt_phi", "add_jet1pt_mass",
    "add_jet1pt_Higgs1_deta", "add_jet1pt_Higgs1_dphi", #"add_jet1pt_Higgs1_m",
    "add_jet1pt_Higgs2_deta", "add_jet1pt_Higgs2_dphi" #"add_jet1pt_Higgs2_m"
]

features += semi_cat_features

semi_cat_global_indices = [features.index(feat) for feat in semi_cat_features]
local_pt_index = 0  # Index 0 within semi_cat_features ("add_jet1pt_pt")
local_jet_feature_indices = list(range(len(semi_cat_features)))

# test_data_continuous, scaler_continuous = dl.full_data_loader(
#     region="SR",
#     bootstrap_coef_2b=1, 
#     bootstrap_coef_4b=1, 
#     test_size=0, 
#     validation_size=0, 
#     features=features,
#     batch_size=batch_size,
#     apply_scaling=False,
#     apply_clipping=False
# )

# FFN_continuous = ffn.FFN(
#     input_dim=len(features), 
#     dir_path=dir_path_FFN_continuous
# )

# FFN_continuous.print_model_summary()
# FFN_continuous.load_model(dir_path_FFN_continuous + 'weights/flow_model.pth')

# test_model_cont = tm.FFNModelTester(
#     model=FFN_continuous, 
#     test_data_loader=test_data_continuous, 
#     features=features, 
#     dir_path=dir_path_evaluation_FFN_continuous
# )

# dir_eval = dir_path_evaluation_FFN_continuous + 'eval_data/'

# test_model_cont.transform_2b_to_4b(test_data_continuous, h5_filename=dir_eval + 'transform_2b_to_4b_scenario', use_light_analysis=False)
# test_model_cont.classification_analysis(h5_filename=dir_eval + 'classification_analysis')
# test_model_cont.per_feature_likelihood(h5_filename=dir_eval + 'per_feature_likelihood_scenario', top_n=len(features))
# test_model_cont.reweighting_patterns(h5_filename=dir_eval + 'reweighting_patterns')

# =========================================================================
# FFN embedding Model
# =========================================================================
dir_path_FFN_embedding = '/eos/user/a/amorandi/HH4b/FFN_models/FFN_5j_embedding/'
os.makedirs(dir_path_FFN_embedding, exist_ok=True)
dir_path_evaluation_FFN_embedding = dir_path_evaluation + 'FFN_embedding/'
os.makedirs(dir_path_evaluation_FFN_embedding, exist_ok=True)

test_data_embedding, scaler_continuous = dl.full_data_loader(
    region="SR",
    bootstrap_coef_2b=1, 
    bootstrap_coef_4b=1, 
    test_size=0, 
    validation_size=0, 
    features=features,
    batch_size=batch_size,
    apply_scaling=False,
    apply_clipping=False
)

FFN_embedding = ffn.FFN(
    input_dim=len(features), 
    dir_path=dir_path_FFN_embedding, 
    semicategorical_features=semi_cat_global_indices, 
    pt_index=local_pt_index,                        
    jet_feature_indices=local_jet_feature_indices,  
    embedding_dim=8,               
    embedding_noise_std=0.1,       
)

FFN_embedding.print_model_summary()
FFN_embedding.load_model(dir_path_FFN_embedding + 'weights/flow_model.pth')

test_model_emb = tm.FFNModelTester(
    model=FFN_embedding, 
    test_data_loader=test_data_embedding, 
    features=features, 
    dir_path=dir_path_evaluation_FFN_embedding
)

dir_eval = dir_path_evaluation_FFN_embedding + 'eval_data/'

test_model_emb.transform_2b_to_4b(test_data_embedding, h5_filename=dir_eval + 'transform_2b_to_4b_scenario', use_light_analysis=False)
test_model_emb.classification_analysis(h5_filename=dir_eval + 'classification_analysis')
test_model_emb.per_feature_likelihood(h5_filename=dir_eval + 'per_feature_likelihood_scenario', top_n=len(features))
test_model_emb.reweighting_patterns(h5_filename=dir_eval + 'reweighting_patterns')

