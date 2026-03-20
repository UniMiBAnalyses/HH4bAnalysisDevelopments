import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to import from lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib import data_loader as dl
from lib.features import features
import model as md

# Make sure we use the same semi-categorical features as training
semi_cat_features = [
    "add_jet1pt_pt", "add_jet1pt_eta", "add_jet1pt_phi", "add_jet1pt_mass",
    "add_jet1pt_Higgs1_deta", "add_jet1pt_Higgs1_dphi", #"add_jet1pt_Higgs1_m",
    "add_jet1pt_Higgs2_deta", "add_jet1pt_Higgs2_dphi" #"add_jet1pt_Higgs2_m"
]
features_with_semicat = features + semi_cat_features


def calculate_njet_fractions(njet_2b, njet_4b):
    """
    Task 1: Calculate yields and fractions for n_jet=4 vs n_jet>=5.
    """
    print("\n" + "="*60)
    print("TASK 1: JET MULTIPLICITY ANALYSIS")
    print("="*60)
    
    # Filter masks
    mask_4j_2b = (njet_2b == 4)
    mask_5j_2b = (njet_2b >= 5)
    
    mask_4j_4b = (njet_4b == 4)
    mask_5j_4b = (njet_4b >= 5)
    
    # 2. Calculate Yields
    N_2b_4j = np.sum(mask_4j_2b)
    N_2b_5j = np.sum(mask_5j_2b)
    N_2b_total = len(njet_2b)
    
    N_4b_4j = np.sum(mask_4j_4b)
    N_4b_5j = np.sum(mask_5j_4b)
    N_4b_total = len(njet_4b)
    
    # 3. Calculate Fractions
    frac_2b_4j = N_2b_4j / N_2b_total if N_2b_total > 0 else 0
    frac_2b_5j = N_2b_5j / N_2b_total if N_2b_total > 0 else 0
    
    frac_4b_4j = N_4b_4j / N_4b_total if N_4b_total > 0 else 0
    frac_4b_5j = N_4b_5j / N_4b_total if N_4b_total > 0 else 0
    
    # 4. Calculate Scale Factors
    sf_4j = N_4b_4j / N_2b_4j if N_2b_4j > 0 else 0
    sf_5j = N_4b_5j / N_2b_5j if N_2b_5j > 0 else 0
    
    # 5. Print Statistics
    print(f"{'Metric':<25} | {'CR2b':<15} | {'CR4b':<15} | {'Scale Factor (4b/2b)':<20}")
    print("-" * 80)
    print(f"{'Total Events':<25} | {N_2b_total:<15} | {N_4b_total:<15} | {'-':<20}")
    print(f"{'Events (n_jet=4)':<25} | {N_2b_4j:<15} | {N_4b_4j:<15} | {sf_4j:.4f}")
    print(f"{'Events (n_jet>=5)':<25} | {N_2b_5j:<15} | {N_4b_5j:<15} | {sf_5j:.4f}")
    print("-" * 80)
    print(f"{'Fraction (n_jet=4)':<25} | {frac_2b_4j:.4f}          | {frac_4b_4j:.4f}          | -")
    print(f"{'Fraction (n_jet>=5)':<25} | {frac_2b_5j:.4f}          | {frac_4b_5j:.4f}          | -")
    print("="*60 + "\n")


def plot_split_kinematics(features_to_plot, feature_names, X_2b, X_4b, njet_2b, njet_4b, weights_no_embed, weights_embed, output_dir):
    """
    Task 2: Comparison histograms split by jet multiplicity.
    """
    print("TASK 2: SPLIT KINEMATICS PLOTTING")
    
    # Masks
    mask_4j_2b = (njet_2b == 4)
    mask_5j_2b = (njet_2b >= 5)
    
    mask_4j_4b = (njet_4b == 4)
    mask_5j_4b = (njet_4b >= 5)

    for i, feature_name in enumerate(tqdm(features_to_plot, desc="Plotting features")):
        # Get feature index from name
        try:
            feat_idx = feature_names.index(feature_name)
        except ValueError:
            print(f"Warning: Feature {feature_name} not found in feature list. Skipping.")
            continue
            
        data_2b = X_2b[:, feat_idx]
        data_4b = X_4b[:, feat_idx]
        
        # Setup Figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1], 'wspace': 0.2, 'hspace': 0.1})
        
        # --- LEFT: n_jet == 4 ---
        ax_main_4j = axes[0, 0]
        ax_ratio_4j = axes[1, 0]
        
        # Data filtering
        d2b_4j = data_2b[mask_4j_2b]
        d4b_4j = data_4b[mask_4j_4b]
        w_no_embed_4j = weights_no_embed[mask_4j_2b]
        w_embed_4j = weights_embed[mask_4j_2b]
        
        # Ranges
        if len(d2b_4j) == 0 or len(d4b_4j) == 0:
            print(f"Skipping {feature_name} (n_jet==4) due to empty data.")
            continue

        low = min(np.min(d2b_4j), np.min(d4b_4j))
        high = max(np.max(d2b_4j), np.max(d4b_4j))
        bins = 40
        
        # Histograms
        # Use simple np.histogram
        h_target, edges = np.histogram(d4b_4j, bins=bins, range=(low, high), density=True)
        h_no_embed, _ = np.histogram(d2b_4j, bins=bins, range=(low, high), weights=w_no_embed_4j, density=True)
        h_embed, _ = np.histogram(d2b_4j, bins=bins, range=(low, high), weights=w_embed_4j, density=True)
        
        centers = (edges[:-1] + edges[1:]) / 2
        
        # Plot Main
        ax_main_4j.step(centers, h_target, where='mid', label='Target (CR4b)', color='black', linewidth=2)
        ax_main_4j.step(centers, h_no_embed, where='mid', label='Reco-4b (No Embed)', color='blue', linestyle='--')
        ax_main_4j.step(centers, h_embed, where='mid', label='Reco-4b (With Embed)', color='red', linestyle='--')
        
        ax_main_4j.set_title(f"{feature_name} (n_jet == 4)", fontsize=14)
        ax_main_4j.set_ylabel("Normalized Density", fontsize=12)
        ax_main_4j.legend()
        ax_main_4j.grid(True, alpha=0.3)
        ax_main_4j.set_xticklabels([]) # Hide x labels for main plot
        
        # Plot Ratio
        # Avoid division by zero
        ratio_no_embed = np.divide(h_no_embed, h_target, out=np.zeros_like(h_no_embed), where=h_target!=0)
        ratio_embed = np.divide(h_embed, h_target, out=np.zeros_like(h_embed), where=h_target!=0)
        
        ax_ratio_4j.step(centers, ratio_no_embed, where='mid', color='blue', linestyle='--')
        ax_ratio_4j.step(centers, ratio_embed, where='mid', color='red', linestyle='--')
        ax_ratio_4j.axhline(1, color='black', linewidth=1)
        ax_ratio_4j.set_ylabel("Ratio to Target", fontsize=10)
        ax_ratio_4j.set_xlabel(feature_name, fontsize=12)
        ax_ratio_4j.set_ylim(0.5, 1.5)
        ax_ratio_4j.grid(True, alpha=0.3)
        
        
        # --- RIGHT: n_jet >= 5 ---
        ax_main_5j = axes[0, 1]
        ax_ratio_5j = axes[1, 1]
        
        # Data filtering
        d2b_5j = data_2b[mask_5j_2b]
        d4b_5j = data_4b[mask_5j_4b]
        w_no_embed_5j = weights_no_embed[mask_5j_2b]
        w_embed_5j = weights_embed[mask_5j_2b]
        
        if len(d2b_5j) == 0 or len(d4b_5j) == 0:
            print(f"Skipping {feature_name} (n_jet>=5) due to empty data.")
            # Clear axes to not show garbage
            ax_main_5j.axis('off')
            ax_ratio_5j.axis('off')
        else:
            # Histograms (Use same range/bins as 4j for consistency? Or recalculate? Recalculate usually safer for tails)
            h_target_5, edges_5 = np.histogram(d4b_5j, bins=bins, range=(low, high), density=True)
            h_no_embed_5, _ = np.histogram(d2b_5j, bins=bins, range=(low, high), weights=w_no_embed_5j, density=True)
            h_embed_5, _ = np.histogram(d2b_5j, bins=bins, range=(low, high), weights=w_embed_5j, density=True)
            
            centers_5 = (edges_5[:-1] + edges_5[1:]) / 2
            
            # Plot Main
            ax_main_5j.step(centers_5, h_target_5, where='mid', label='Target (CR4b)', color='black', linewidth=2)
            ax_main_5j.step(centers_5, h_no_embed_5, where='mid', label='Reco-4b (No Embed)', color='blue', linestyle='--')
            ax_main_5j.step(centers_5, h_embed_5, where='mid', label='Reco-4b (With Embed)', color='red', linestyle='--')
            
            ax_main_5j.set_title(f"{feature_name} (n_jet >= 5)", fontsize=14)
            ax_main_5j.set_yticks([]) # Hide y ticks for right plot to save space
            # ax_main_5j.set_ylabel("Normalized Density", fontsize=12) 
            ax_main_5j.legend()
            ax_main_5j.grid(True, alpha=0.3)
            ax_main_5j.set_xticklabels([])
            
            # Plot Ratio
            ratio_no_embed_5 = np.divide(h_no_embed_5, h_target_5, out=np.zeros_like(h_no_embed_5), where=h_target_5!=0)
            ratio_embed_5 = np.divide(h_embed_5, h_target_5, out=np.zeros_like(h_embed_5), where=h_target_5!=0)
            
            ax_ratio_5j.step(centers_5, ratio_no_embed_5, where='mid', color='blue', linestyle='--')
            ax_ratio_5j.step(centers_5, ratio_embed_5, where='mid', color='red', linestyle='--')
            ax_ratio_5j.axhline(1, color='black', linewidth=1)
            # ax_ratio_5j.set_ylabel("Ratio to Target", fontsize=10)
            ax_ratio_5j.set_xlabel(feature_name, fontsize=12)
            ax_ratio_5j.set_ylim(0.5, 1.5)
            ax_ratio_5j.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{feature_name}_split_comparison.png"))
        plt.close()


def predict_with_progress(model, X, batch_size=8192, desc="Predicting"):
    """
    Helper to run prediction with a progress bar.
    """
    n_samples = len(X)
    probs_list = []
    
    with tqdm(total=n_samples, desc=desc) as pbar:
        for i in range(0, n_samples, batch_size):
            batch = X[i : i + batch_size]
            # Assumes model.predict_proba handles numpy arrays and returns numpy arrays
            p = model.predict_proba(batch)
            probs_list.append(p)
            pbar.update(len(batch))
            
    return np.vstack(probs_list)


if __name__ == "__main__":
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 0. Setup Paths
    dir_path_no_embed = '/eos/user/a/amorandi/HH4b/FFN_models/FFN_5j_continuos/'
    dir_path_embed = '/eos/user/a/amorandi/HH4b/FFN_models/FFN_5j_embedding/'
    output_dir = '/eos/user/a/amorandi/HH4b/FFN_models/4j_vs_5j_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    # We need "njet" in addition to model features.
    # Assuming "njet" is the key in the root file.
    feature_list_loading = features_with_semicat + ["njet"]
    
    print("Loading Data...")
    try:
        # Load raw data with njet
        # We need test_size=0 and validation_size=0 to get the full dataset.
        # data_preparation returns (X, y, scaler) when split sizes are 0.
        X_all, y_all, _ = dl.data_preparation(
            region="CR",
            bootstrap_coef_2b=1,
            bootstrap_coef_4b=None, # Balanced
            test_size=0,
            validation_size=0,
            features=feature_list_loading,
            apply_scaling=False,
            apply_clipping=False
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        # Raise error to stop execution if data loading fails
        raise e

    # Extract njet and split X used for model
    if "njet" in feature_list_loading:
        njet_idx = feature_list_loading.index("njet")
        njet = X_all[:, njet_idx].astype(int)
        
        # Get columns for model
        features_indices = [i for i, f in enumerate(feature_list_loading) if f != "njet"]
        X_model = X_all[:, features_indices]
    else:
        print("CRITICAL: 'njet' not found in features. Cannot perform split analysis.")
        sys.exit(1)
        
    
    # 2. Load Models
    print("Loading Models...")
    input_dim = X_model.shape[1]
    
    # Model 1: No Embed
    md_no_embed = md.FFN(
        input_dim=input_dim, 
        dir_path=dir_path_no_embed, 
        EarlyStopper_patience=30
    )
    md_no_embed.load_model(dir_path_no_embed + 'weights/flow_model.pth')
    
    # Model 2: With Embed
    semi_cat_global_indices = [features_with_semicat.index(feat) for feat in semi_cat_features]
    local_pt_index = semi_cat_features.index("add_jet1pt_pt")
    local_jet_feature_indices = list(range(len(semi_cat_features)))

    md_embed = md.FFN(
        input_dim=input_dim, 
        dir_path=dir_path_embed, 
        semicategorical_features=semi_cat_global_indices, 
        pt_index=local_pt_index,                        
        jet_feature_indices=local_jet_feature_indices,  
        embedding_dim=8,               
        embedding_noise_std=0.1,       
        EarlyStopper_patience=30
    )
    md_embed.load_model(dir_path_embed + 'weights/flow_model.pth')
    

    # 3. Predict Weights (CR2b -> CR4b)
    print("Predicting Weights...")
    # Separate CR2b and CR4b
    mask_2b = (y_all == 0)
    mask_4b = (y_all == 1)
    
    X_2b_np = X_model[mask_2b]
    njet_2b = njet[mask_2b]
    X_4b_numpy = X_model[mask_4b] # Keep as numpy for plotting
    njet_4b = njet[mask_4b]
    
    # Predict Probabilities
    # We pass numpy array, predict_proba handles conversion
    print("  Predicting No Embed (Continuous)...")
    probs_no_embed = predict_with_progress(md_no_embed, X_2b_np, desc="Predicting (No Embed)")
    
    print("  Predicting With Embed...")
    probs_embed = predict_with_progress(md_embed, X_2b_np, desc="Predicting (Embed)")
    
    # Calculate Weights: P(4b)/P(2b)
    eps = 1e-8
    weights_no_embed = probs_no_embed[:, 1] / (probs_no_embed[:, 0] + eps)
    weights_embed = probs_embed[:, 1] / (probs_embed[:, 0] + eps)
    
    # 4. Run Tasks
    # Task 1: Statistics
    calculate_njet_fractions(njet_2b, njet_4b)
    
    # Task 2: Plotting
    plot_split_kinematics(features, features_with_semicat, X_model[mask_2b], X_4b_numpy, njet_2b, njet_4b, weights_no_embed, weights_embed, output_dir)
    
    print("\nAnalysis Complete. Results in:", output_dir)


# ============================================================
# TASK 1: JET MULTIPLICITY ANALYSIS
# ============================================================
# Metric                    | CR2b            | CR4b            | Scale Factor (4b/2b)
# --------------------------------------------------------------------------------
# Total Events              | 5187675         | 5187675         | -
# Events (n_jet=4)          | 1643649         | 627055          | 0.3815
# Events (n_jet>=5)         | 3544026         | 4560620         | 1.2868
# --------------------------------------------------------------------------------
# Fraction (n_jet=4)        | 0.3168          | 0.1209          | -
# Fraction (n_jet>=5)       | 0.6832          | 0.8791          | -
# ============================================================