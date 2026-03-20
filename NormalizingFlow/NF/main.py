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


from lib.features import features


dir_path = '/eos/user/a/amorandi/HH4b/NF/'
os.makedirs(dir_path, exist_ok=True)

# Increase batch size for faster training with powerful GPUs
# Larger batches = fewer iterations = faster training
# A100/H100 can handle much larger batches than V100
batch_size = 2048  # Increased from default 512

train_data, val_data = dl.full_data_loader(
    test_size=0, 
    validation_size=0.25, 
    features=features,
    batch_size=batch_size,
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

# Print GPU info
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {batch_size}")

model.print_model_summary()

if os.path.exists(dir_path + 'weights/flow_model.pth'):
    model.load_model(dir_path + 'weights/flow_model.pth')
else:
    print("Starting training...")
    model.train(train_data, val_data, epochs=999)
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
    batch_size=batch_size,
    seed=42
)

test_model = tm.ModelTester(
    model=model, 
    test_data_loader=test_data_all, 
    features=features, 
    dir_path=dir_path
)

dir_eval = dir_path + 'eval_data/'

# 2b to 4b transformation analysis
test_model.transform_2b_to_4b(test_data_all, h5_filename=dir_eval + 'transform_2b_to_4b_scenario', use_light_analysis=False)

# Binary classification analysis (2b vs 4b)
test_model.classification_analysis(h5_filename=dir_eval + 'classification_analysis')

# Per-feature likelihood analysis
test_model.per_feature_likelihood(h5_filename=dir_eval + 'per_feature_likelihood_scenario', top_n=len(features))

# Reweighting pattern analysis
test_model.reweighting_patterns(h5_filename=dir_eval + 'reweighting_patterns')

# Feature shift analysis
test_model.feature_shift_analysis(test_data_all, h5_filename=dir_eval + 'feature_shift_analysis')


