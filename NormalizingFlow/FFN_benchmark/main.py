import os
import sys
import torch

# Add parent directory to path to import from lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib import data_loader as dl
import model as fmd
import test_model as tm


# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()


from lib.features import features


dir_path = '/eos/user/a/amorandi/HH4b/FFN/'
os.makedirs(dir_path, exist_ok=True)

train_data, val_data = dl.full_data_loader(
    test_size=0, 
    validation_size=0.25, 
    features=features, 
    apply_scaling=False, 
    apply_clipping=False,
    seed=42
)

model = fmd.FFN(
    input_dim=len(features), 
    dir_path=dir_path, 
    EarlyStopper_patience=30
)
model.print_model_summary()

if os.path.exists(dir_path + 'weights/flow_model.pth'):
    model.load_model(dir_path + 'weights/flow_model.pth')
else:
    print("Starting training...")
    model.train(train_data, val_data, epochs=9999, batch_size=512)
    model.plot_training_loss()

    model.load_model(dir_path + 'weights/flow_model.pth')

# =========================================================================
# FFN Model Evaluation
# =========================================================================
test_data_all = dl.full_data_loader(
    bootstrap_coef_2b=1, 
    bootstrap_coef_4b=1, 
    test_size=0, 
    validation_size=0, 
    features=features, 
    apply_scaling=False, 
    apply_clipping=False,
    seed=42
)

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
