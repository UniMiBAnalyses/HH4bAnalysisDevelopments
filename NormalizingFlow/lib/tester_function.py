import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report, confusion_matrix
from scipy.stats import norm, ks_2samp
from tqdm import tqdm
import os


# =========================================================================
# Basic utility functions
# =========================================================================

def number_of_bins(samples, NN_type='NF'):
    """
    Freedman-Diaconis rule to determine optimal number of bins for histogram.

    param samples: array-like, samples from which to determine the number of bins
    param NN_type: str, type of neural network used ('NF' or 'FFN'), default='NF'
    return: int, optimal number of bins
    """
    samples = np.asarray(samples)
    q25, q75 = np.percentile(samples, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(samples) ** (-1/3)
    if bin_width == 0:
        return int(np.sqrt(len(samples)/2))
    else:
        if NN_type == "NF":
            return int((samples.max() - samples.min()) / bin_width)
        if NN_type == "FFN":
            return int((samples.max() - samples.min()) /(2 * bin_width))


# =========================================================================
# Classification analysis functions
# =========================================================================

def ROC_preparation(test, y_np):
    """
    Prepare data for ROC curve calculation.
    
    param test: array-like, test scores or probabilities
    param y_np: array-like, true labels
    return: fpr, tpr, roc_auc, optimal_threshold
    """     
    roc_auc = roc_auc_score(y_np, test)
    fpr, tpr, thresholds = roc_curve(y_np, test)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    return fpr, tpr, roc_auc, optimal_threshold


def ROC_curve_plot(scores, y_true, filename):
    """
    Plot ROC curve for test data only.

    param scores: array-like, test scores or probabilities
    param y_true: array-like, true labels
    param filename: str, path to save the plot
    return: roc_auc, optimal_threshold
    """
    fpr, tpr, roc_auc, optimal_threshold = ROC_preparation(scores, y_true)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='blue', linewidth=2)
    
    optimal_idx = np.argmax(tpr - fpr)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], s=100, c='red', zorder=5, 
               label=f'Optimal Threshold = {optimal_threshold:.4f}')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    return roc_auc, optimal_threshold


def ConfusionMatrix(y_true, y_rec, filename):
    """
    Plot confusion matrix comparing true labels and predicted labels.

    param y_true: array-like, true labels   
    param y_rec: array-like, predicted labels
    param filename: str, path to save the plot
    return: 0
    """
    N = len(y_true)
    y_pred_binary = (y_rec > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    cm_normalized = cm.astype('float') / N
    cm_4comparison = confusion_matrix(y_true, y_true)
    cm_4comparison_normalized = cm_4comparison.astype('float') / N

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", ax=axes[0])
    axes[0].set_title('True vs Predict')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_4comparison_normalized, annot=True, fmt=".2f", cmap="Oranges", ax=axes[1])
    axes[1].set_title('True vs True')
    axes[1].set_xlabel('True')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return 0


def perform_binary_classification_analysis(scores_test, y_test, 
                                          method_name, dir_path, xlabel, NN_type,
                                          use_threshold_for_prediction=True):
    """
    Perform comprehensive binary classification analysis for a given scoring method.
    
    param scores_test: np.ndarray, classification scores for test set
    param y_test: np.ndarray, true labels for test set
    param method_name: str, name of the classification method (e.g., "Likelihood Ratio")
    param dir_path: str, directory path for saving plots
    param xlabel: str, x-axis label for distribution plot
    param NN_type: str, type of neural network ('NF' or 'FFN')
    param use_threshold_for_prediction: bool, if True use optimal threshold, else use 0.5
    
    return: dict with keys 'roc_auc_test', 'accuracy_test', 'optimal_threshold_test'
    """
    print("\n" + "-"*60)
    print(f"{method_name} Classification")
    print("-"*60)
    
    os.makedirs(dir_path, exist_ok=True)
    
    # ROC curve analysis
    roc_auc_test, optimal_threshold_test = ROC_curve_plot(
        scores_test, y_test, 
        dir_path + f'roc_curve_{method_name.lower().replace(" ", "_")}.png'
    )
    
    # Predictions using optimal threshold
    if use_threshold_for_prediction:
        y_pred_test = (scores_test > optimal_threshold_test).astype(int)
    else:
        y_pred_test = (scores_test > 0.5).astype(int)
    
    # Accuracy
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"\nAccuracy (optimal threshold): {accuracy_test:.4f}")
    
    # Classification report
    print(f"\nClassification Report ({method_name}):")
    print(classification_report(y_test, y_pred_test, target_names=['2b events', '4b events']))
    
    # Plot score distribution
    plt.figure(figsize=(8, 6))
    plt.hist(scores_test[y_test == 0], bins=number_of_bins(scores_test[y_test == 0], NN_type),
             histtype='step', label='2b events', color='blue')
    plt.hist(scores_test[y_test == 1], bins=number_of_bins(scores_test[y_test == 1], NN_type),
             histtype='step', label='4b events', color='red')
    plt.axvline(optimal_threshold_test, color='black', linestyle='--', linewidth=2, 
                label='Optimal threshold')
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(f'{method_name} Distribution')
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(dir_path + f'{method_name.lower().replace(" ", "_")}_distribution.png', dpi=150)
    plt.close()
    
    # Confusion Matrix
    ConfusionMatrix(y_test, y_pred_test, 
                   dir_path + f'confusion_matrix_{method_name.lower().replace(" ", "_")}.png')
    
    return {
        'roc_auc_test': roc_auc_test,
        'accuracy_test': accuracy_test,
        'optimal_threshold_test': optimal_threshold_test
    }


# =========================================================================
# Data preparation functions
# =========================================================================

def data_preparing_for_comparison(data):
    """
    Prepare test data for comparison of distributions.

    param data: DataLoader for test data
    return: data_2b, data_4b (torch.Tensors for 2b and 4b samples)
    """
    X_test = []
    y_test = []
    for batch_X, batch_y in data:
        X_test.append(batch_X)
        y_test.append(batch_y)
    
    X_test_np = torch.cat(X_test, dim=0)
    y_test_np = torch.cat(y_test, dim=0).squeeze()
    
    data_2b = X_test_np[y_test_np == 0]
    data_4b = X_test_np[y_test_np == 1]
    
    return data_2b, data_4b


# =========================================================================
# Visualization functions for distribution comparison
# =========================================================================

def plot_4bvs2b_and_4bvs4breco(counts_2b, counts_reco_4b, counts_4b, error_4b, 
                                feature_name, dir_path, bin_edges, bin_centers, bin_width):
    """
    Create a 2x2 grid of plots comparing the distributions of a feature for 4b vs 2b and 4b vs reco 4b.

    param counts_2b: array-like, histogram counts for the 2b class
    param counts_reco_4b: array-like, histogram counts for the reconstructed 4b samples
    param counts_4b: array-like, histogram counts for the 4b class
    param error_4b: array-like, reconstruction error for 4b samples
    param feature_name: str, name of the feature being plotted (used for titles and filenames)
    param dir_path: str, directory path for saving the plot
    param bin_edges: array-like, histogram bin edges
    param bin_centers: array-like, histogram bin centers
    param bin_width: array-like, histogram bin widths
    return: 0
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        residuals_4b_vs_2b = np.where(error_4b > 0, (counts_4b - counts_2b) / error_4b, 0)
        residuals_4b_vs_reco = np.where(error_4b > 0, (counts_4b - counts_reco_4b) / error_4b, 0)
    
    # Calculate max residual for y-axis limits
    max_residual = 5 # max(np.abs(residuals_4b_vs_2b).max(), np.abs(residuals_4b_vs_reco).max())

    # Create 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top-left: 4b vs 2b distribution
    axes[0, 0].step(bin_edges, np.append(counts_2b, counts_2b[-1]), where='post', color='green', 
                   label='2b', linewidth=1.5)
    axes[0, 0].fill_between(bin_edges, np.append(counts_2b, counts_2b[-1]), step='post', 
                            color='green', alpha=0.3)
    axes[0, 0].errorbar(bin_centers, counts_4b, yerr=error_4b, fmt='o', 
                       color='red', label='4b', markersize=4, capsize=3, elinewidth=1.5)
    axes[0, 0].set_title(f'4b vs 2b: {feature_name}', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('Density', fontsize=11)
    axes[0, 0].legend(loc='best', frameon=True, shadow=True, fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[0, 0].tick_params(labelbottom=False)
    
    # Bottom-left: 4b vs 2b residue
    axes[1, 0].bar(bin_centers, residuals_4b_vs_2b, width=bin_width, color='gray', alpha=0.8)
    axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=1.5)
    axes[1, 0].set_xlabel(feature_name, fontsize=11)
    axes[1, 0].set_ylabel(r'$(4b - 2b) / \sigma_{4b}$', fontsize=10)
    axes[1, 0].set_ylim(-max_residual, max_residual)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Top-right: 4b vs reco 4b distribution
    axes[0, 1].step(bin_edges, np.append(counts_reco_4b, counts_reco_4b[-1]), where='post', color='blue', 
                   label='Reco 4b', linewidth=1.5)
    axes[0, 1].fill_between(bin_edges, np.append(counts_reco_4b, counts_reco_4b[-1]), step='post', 
                           color='blue', alpha=0.3)
    axes[0, 1].errorbar(bin_centers, counts_4b, yerr=error_4b, fmt='o', 
                       color='red', label='4b', markersize=4, capsize=3, elinewidth=1.5)
    axes[0, 1].set_title(f'4b vs Reco 4b: {feature_name}', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel('Density', fontsize=11)
    axes[0, 1].legend(loc='best', frameon=True, shadow=True, fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[0, 1].tick_params(labelbottom=False)
    
    # Bottom-right: 4b vs reco 4b residue
    axes[1, 1].bar(bin_centers, residuals_4b_vs_reco, width=bin_width, color='gray', alpha=0.8)
    axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1.5)
    axes[1, 1].set_xlabel(feature_name, fontsize=11)
    axes[1, 1].set_ylabel(r'$(4b - Reco 4b) / \sigma_{4b}$', fontsize=10)
    axes[1, 1].set_ylim(-max_residual, max_residual)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig(dir_path + f'{feature_name}_latent_space_comparison.png', dpi=150)
    plt.close()

    return 0


def plot_single_distribution_comparison(counts_comparison, counts_4b, error_4b, 
                                        feature_name, dir_path, bin_edges, bin_centers, bin_width,
                                        comparison_label, comparison_color, max_residual=5):
    """
    Create a single comparison plot (distribution + residue) for one pair of distributions.
    
    param counts_comparison: array-like, histogram counts for comparison distribution (2b or target)
    param counts_4b: array-like, histogram counts for the 4b class (reference)
    param error_4b: array-like, error bars for 4b samples
    param feature_name: str, name of the feature being plotted
    param dir_path: str, directory path for saving the plot
    param bin_edges: array-like, histogram bin edges
    param bin_centers: array-like, histogram bin centers
    param bin_width: array-like, histogram bin widths
    param comparison_label: str, label for the comparison distribution (e.g., '2b', 'Reco 4b')
    param comparison_color: str, color for the comparison distribution
    param max_residual: float or None, maximum residual value for y-axis limits (auto-calculated if None)
    return: 0
    """
    # Calculate residuals
    with np.errstate(divide='ignore', invalid='ignore'):
        residuals = np.where(error_4b > 0, (counts_4b - counts_comparison) / error_4b, 0)
    
    # Create 2x1 grid of plots
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top: distribution comparison
    axes[0].step(bin_edges, np.append(counts_comparison, counts_comparison[-1]), where='post', color=comparison_color, 
                 label=comparison_label, linewidth=1.5)
    axes[0].fill_between(bin_edges, np.append(counts_comparison, counts_comparison[-1]), step='post', 
                        color=comparison_color, alpha=0.3)
    axes[0].errorbar(bin_centers, counts_4b, yerr=error_4b, fmt='o', 
                     color='red', label='4b', markersize=4, capsize=3, elinewidth=1.5)
    axes[0].set_title(f'4b vs {comparison_label}: {feature_name}', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].legend(loc='best', frameon=True, shadow=True, fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[0].tick_params(labelbottom=False)
    
    # Bottom: residue
    axes[1].bar(bin_centers, residuals, width=bin_width, color='gray', alpha=0.8)
    axes[1].axhline(0, color='black', linestyle='-', linewidth=1.5)
    axes[1].set_xlabel(feature_name, fontsize=11)
    axes[1].set_ylabel(f'$(4b - {comparison_label}) / \\sigma_{{4b}}$', fontsize=10)
    axes[1].set_ylim(-max_residual, max_residual)
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    plt.savefig(dir_path + f'{feature_name}_comparison.png', dpi=150)
    plt.close()
    
    return 0


def plot_single_scatter(x_data, y_data, feature_name, dir_path, 
                       x_label, y_label, title, color='blue', add_identity_line=True):
    """
    Create a single scatter plot.
    
    param x_data: array-like, x-axis data
    param y_data: array-like, y-axis data
    param feature_name: str, name of the feature being plotted
    param dir_path: str, directory path for saving the plot
    param x_label: str, x-axis label
    param y_label: str, y-axis label
    param title: str, plot title
    param color: str, scatter point color
    param add_identity_line: bool, whether to add y=x line
    return: 0
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(x_data, y_data, alpha=0.5, color=color)
    
    if add_identity_line and len(x_data) > 0:
        min_val = min(x_data.min(), y_data.min())
        max_val = max(x_data.max(), y_data.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        plt.legend()
    
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel(y_label, fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(dir_path + f'{feature_name}_scatter.png', dpi=150)
    plt.close()
    
    return 0


def create_summary_bar_plot(data_dict, features_list, title, ylabel, filename, 
                           color='purple', ylim=None, figsize=(12, 6), sort_descending=True,
                           add_hline=None, hline_color='black'):
    """
    Create a summary bar plot for feature-wise metrics.
    
    param data_dict: dict, dictionary mapping feature names to values, OR list of values (will use features_list for keys)
    param features_list: list of str, feature names (used if data_dict is a list)
    param title: str, plot title
    param ylabel: str, y-axis label
    param filename: str, full path for saving the plot
    param color: str, bar color (default: 'purple')
    param ylim: tuple or None, y-axis limits (e.g., (0, 10))
    param figsize: tuple, figure size (default: (12, 6))
    param sort_descending: bool, whether to sort bars in descending order (default: True)
    param add_hline: float or None, y-value for horizontal reference line (default: None)
    param hline_color: str, color of the horizontal line (default: 'black')
    """
    # Convert list to dict if needed
    if isinstance(data_dict, list):
        data_dict = dict(zip(features_list, data_dict))
    
    # Sort the data
    if sort_descending:
        sorted_data = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True))
    else:
        sorted_data = dict(sorted(data_dict.items(), key=lambda item: item[1]))
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.bar(range(len(sorted_data)), list(sorted_data.values()), color=color, alpha=0.7)
    
    # Add horizontal reference line if specified
    if add_hline is not None:
        plt.axhline(add_hline, color=hline_color, linestyle='-', linewidth=1)
    
    plt.xticks(range(len(sorted_data)), list(sorted_data.keys()), rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3, axis='y')
    
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def distribution_comparison_analysis_light(X_2b, X_4b, X_target, features, dir_path_base, NN_type,
                                           target_name="Target", compute_chi2_2b=False):
    """
    Light version of distribution comparison analysis.
    Computes only statistical metrics and generates summary plots, without individual distribution 
    and scatter plots for each feature.
    
    param X_2b: torch.Tensor or np.ndarray, background distribution (2b events), can be None if not needed
    param X_4b: torch.Tensor or np.ndarray, signal/target distribution (real 4b events)
    param X_target: torch.Tensor or np.ndarray, generated/reconstructed distribution to compare against X_4b
    param features: list of str, feature names for labeling plots
    param dir_path_base: str, base directory for saving summary plots
    param NN_type: str, type of neural network ('NF' or 'FFN')
    param target_name: str, name for the target distribution (e.g., "Reco 4b", "Sampled 4b")
    param compute_chi2_2b: bool, whether to compute chi2 between 2b and target (requires X_2b)
    
    return: dict with keys 'chi2', 'chi2_2b', 'kl_divergence', 'mmd', 'ks_statistic'
    """
    print(f"\nPerforming Light Distribution Comparison Analysis...")
    
    # Create base directory for summary plots
    os.makedirs(dir_path_base, exist_ok=True)
    
    # Convert to numpy if needed
    if torch.is_tensor(X_4b):
        X_4b = X_4b.cpu().numpy()
    if torch.is_tensor(X_target):
        X_target = X_target.cpu().numpy()
    if X_2b is not None and torch.is_tensor(X_2b):
        X_2b = X_2b.cpu().numpy()
    
    # Initialize result containers
    chi2 = {}
    chi2_2b = {} if (X_2b is not None and compute_chi2_2b) else None
    kl_divergence = {}
    mmd = {}
    ks_statistic = {}
    
    # Check for NaN values and warn user
    if np.isnan(X_target).any():
        nan_count = np.isnan(X_target).sum()
        total_elements = X_target.size
        print(f"Warning: X_target contains {nan_count}/{total_elements} NaN values ({100*nan_count/total_elements:.2f}%)")
        print("NaN values will be filtered out for analysis.")
    
    number_of_samples_4b = X_4b.shape[0]
    number_of_samples_2b = X_2b.shape[0] if X_2b is not None else 0
    
    # =======================================================================
    # Per-feature statistical analysis (no plotting)
    # =======================================================================
    
    for i, feature in enumerate(tqdm(features, desc="Computing metrics")):
        
        # Filter out NaN values for this feature
        X_target_feature = X_target[:, i]
        X_4b_feature = X_4b[:, i]
        
        # Create mask for valid (non-NaN) values
        valid_mask = ~np.isnan(X_target_feature)
        X_target_feature_clean = X_target_feature[valid_mask]
        
        # Calculate histogram bins based on real 4b data
        bins = number_of_bins(X_4b_feature)
        
        # Calculate histograms for 4b and target
        X_4b_raw, bin_edges = np.histogram(X_4b_feature, bins=bins)
        bin_width = np.diff(bin_edges)
        
        # Normalize to density
        counts_4b = X_4b_raw / (bin_width * number_of_samples_4b)
        error_4b = np.sqrt(X_4b_raw) / (bin_width * number_of_samples_4b)
        
        # Calculate histogram for target
        X_target_raw, _ = np.histogram(X_target_feature_clean, bins=bin_edges)
        number_of_samples_target = len(X_target_feature_clean)
        counts_target = X_target_raw / (bin_width * number_of_samples_target)
        
        # Chi2 calculation for 4b vs target
        valid_bins_4b = error_4b > 0
        if np.sum(valid_bins_4b) > 0:
            chi2_value = np.sum((counts_4b[valid_bins_4b] - counts_target[valid_bins_4b]) ** 2 / (error_4b[valid_bins_4b] ** 2))
            chi2[feature] = chi2_value / np.sum(valid_bins_4b)
        else:
            chi2[feature] = np.nan
        
        # Kullback-Leibler divergence calculation (4b vs target)
        epsilon = 1e-10
        P = counts_4b * bin_width
        Q = counts_target * bin_width
        P = P / np.sum(P)
        Q = Q / np.sum(Q)
        valid_kl = (P > epsilon) & (Q > epsilon)
        if np.sum(valid_kl) > 0:
            kl_value = np.sum(P[valid_kl] * np.log(P[valid_kl] / Q[valid_kl]))
            kl_divergence[feature] = kl_value
        else:
            kl_divergence[feature] = np.nan
        
        # Maximum Mean Discrepancy (MMD) calculation
        def gaussian_kernel(x, y, sigma):
            return np.exp(-0.5 * ((x - y) ** 2) / (sigma ** 2))
        
        # Subsample for computational efficiency
        n_subsample = min(10000, len(X_4b_feature), len(X_target_feature_clean))
        if len(X_4b_feature) > n_subsample:
            idx_4b = np.random.choice(len(X_4b_feature), n_subsample, replace=False)
            X_4b_sub = X_4b_feature[idx_4b]
        else:
            X_4b_sub = X_4b_feature
        
        if len(X_target_feature_clean) > n_subsample:
            idx_target = np.random.choice(len(X_target_feature_clean), n_subsample, replace=False)
            X_target_sub = X_target_feature_clean[idx_target]
        else:
            X_target_sub = X_target_feature_clean
        
        # Compute pairwise distances for bandwidth selection
        all_samples = np.concatenate([X_4b_sub, X_target_sub])
        pairwise_dists = np.abs(all_samples[:, None] - all_samples[None, :])
        sigma_mmd = np.median(pairwise_dists[pairwise_dists > 0])
        
        if sigma_mmd > 0:
            K_xx = gaussian_kernel(X_4b_sub[:, None], X_4b_sub[None, :], sigma_mmd)
            K_yy = gaussian_kernel(X_target_sub[:, None], X_target_sub[None, :], sigma_mmd)
            K_xy = gaussian_kernel(X_4b_sub[:, None], X_target_sub[None, :], sigma_mmd)
            
            mmd_squared = K_xx.sum() / (len(X_4b_sub) ** 2) + K_yy.sum() / (len(X_target_sub) ** 2) - 2 * K_xy.sum() / (len(X_4b_sub) * len(X_target_sub))
            mmd[feature] = np.sqrt(max(0, mmd_squared))
        else:
            mmd[feature] = np.nan
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(X_4b_feature, X_target_feature_clean)
        ks_statistic[feature] = ks_stat
        
        # Chi2 for 2b vs target (optional)
        if compute_chi2_2b and X_2b is not None:
            X_2b_feature = X_2b[:, i]
            X_2b_raw, _ = np.histogram(X_2b_feature, bins=bin_edges)
            counts_2b = X_2b_raw / (bin_width * number_of_samples_2b)
            error_2b = np.sqrt(X_2b_raw) / (bin_width * number_of_samples_2b)
            
            valid_bins_2b = error_2b > 0
            if np.sum(valid_bins_2b) > 0:
                chi2_value_2b = np.sum((counts_2b[valid_bins_2b] - counts_target[valid_bins_2b]) ** 2 / (error_2b[valid_bins_2b] ** 2))
                chi2_2b[feature] = chi2_value_2b / np.sum(valid_bins_2b)
            else:
                chi2_2b[feature] = np.nan
    
    # =======================================================================
    # Summary plots only
    # =======================================================================
    
    print("\nGenerating summary plots...")
    
    # Plot chi-squared values for 4b vs target
    create_summary_bar_plot(
        data_dict=chi2,
        features_list=features,
        title=f'Chi-squared Comparison of 4b vs {target_name} Distributions',
        ylabel='Chi-squared Value',
        filename=dir_path_base + 'chi2_comparison.png',
        color='purple',
        ylim=(0, 10),
        add_hline=2,
        hline_color='red'
    )
    
    # Plot chi-squared for 2b vs target if computed
    if chi2_2b is not None:
        create_summary_bar_plot(
            data_dict=chi2_2b,
            features_list=features,
            title=f'Chi-squared Comparison of 2b vs {target_name} Distributions',
            ylabel='Chi-squared Value',
            filename=dir_path_base + 'chi2_2b_comparison.png',
            color='orange',
            ylim=(0, 10),
            add_hline=2,
            hline_color='red'
        )
    
    # Plot Kullback-Leibler divergence
    create_summary_bar_plot(
        data_dict=kl_divergence,
        features_list=features,
        title=f'Kullback-Leibler Divergence: KL(4b || {target_name})',
        ylabel='KL Divergence',
        filename=dir_path_base + 'kl_divergence_comparison.png',
        color='darkgreen',
        sort_descending=True,
        ylim=(0, 0.005)
    )
    
    # Plot Maximum Mean Discrepancy
    create_summary_bar_plot(
        data_dict=mmd,
        features_list=features,
        title=f'Maximum Mean Discrepancy: 4b vs {target_name}',
        ylabel='MMD',
        filename=dir_path_base + 'mmd_comparison.png',
        color='darkred',
        sort_descending=True,
        ylim=(0, 0.04)
    )
    
    # Plot Kolmogorov-Smirnov test statistic
    create_summary_bar_plot(
        data_dict=ks_statistic,
        features_list=features,
        title=f'Kolmogorov-Smirnov Test Statistic: 4b vs {target_name}',
        ylabel='KS Statistic',
        filename=dir_path_base + 'ks_statistic_comparison.png',
        color='steelblue',
        sort_descending=True,
        ylim=(0, 0.03)
    )
    
    print(f"  Summary plots saved to {dir_path_base}")
    
    return {
        'chi2': chi2,
        'chi2_2b': chi2_2b,
        'kl_divergence': kl_divergence,
        'mmd': mmd,
        'ks_statistic': ks_statistic
    }


def distribution_comparison_analysis(X_2b, X_4b, X_target, features, dir_path_base, NN_type,
                                     include_residue_analysis=True, include_scatter_plots=True,
                                     target_name="Target", paired_with_2b=False):
    """
    Comprehensive distribution comparison and statistical analysis between real and generated/reconstructed data.
    This function can be used for both:
    - Comparing 2b, 4b, and reconstructed 4b distributions (transform_2b_to_4b)
    - Comparing 2b, 4b, and sampled 4b distributions (sample_generation)
    
    Note: For a faster analysis with only summary plots (no individual distributions or scatter plots),
    use distribution_comparison_analysis_light() instead.
    
    param X_2b: torch.Tensor or np.ndarray, background distribution (2b events), can be None if not needed
    param X_4b: torch.Tensor or np.ndarray, signal/target distribution (real 4b events)
    param X_target: torch.Tensor or np.ndarray, generated/reconstructed distribution to compare against X_4b
    param features: list of str, feature names for labeling plots
    param dir_path_base: str, base directory for saving plots
    param NN_type: str, type of neural network ('NF' or 'FFN')
    param include_residue_analysis: bool, whether to include residue difference analysis (requires X_2b)
    param include_scatter_plots: bool, whether to include scatter and QQ plots
    param target_name: str, name for the target distribution (e.g., "Reco 4b", "Sampled 4b")
    param paired_with_2b: bool, True if X_target was generated from X_2b (for scatter plots)
    
    return: dict with keys 'chi2', 'chi2_2b', 'correlations', 'qq_distances', 'mean_residue_differences', 'kl_divergence', 'mmd', 'ks_statistic'
    """
    print(f"\nPerforming Distribution Comparison Analysis...")
    
    # Create directories
    dir_path_transformation = dir_path_base + 'distribution/'
    os.makedirs(dir_path_transformation, exist_ok=True)
    
    # Subdirectories for individual distribution plots
    dir_path_4b_vs_target = dir_path_transformation + '4b_vs_target/'
    os.makedirs(dir_path_4b_vs_target, exist_ok=True)
    
    if X_2b is not None:
        dir_path_4b_vs_2b = dir_path_transformation + '4b_vs_2b/'
        os.makedirs(dir_path_4b_vs_2b, exist_ok=True)
    
    if include_scatter_plots:
        dir_path_scatter = dir_path_base + 'scatter/'
        os.makedirs(dir_path_scatter, exist_ok=True)
        
        # Subdirectories for individual scatter plots
        dir_path_qq = dir_path_scatter + 'qq_4b_vs_target/'
        os.makedirs(dir_path_qq, exist_ok=True)
        
        if X_2b is not None and paired_with_2b:
            dir_path_2b_vs_target = dir_path_scatter + '2b_vs_target/'
            os.makedirs(dir_path_2b_vs_target, exist_ok=True)
    
    if include_residue_analysis and X_2b is not None:
        dir_path_residue = dir_path_base + 'residue_analysis/'
        os.makedirs(dir_path_residue, exist_ok=True)
    
    # Convert to numpy if needed
    if torch.is_tensor(X_4b):
        X_4b = X_4b.cpu().numpy()
    if torch.is_tensor(X_target):
        X_target = X_target.cpu().numpy()
    if X_2b is not None and torch.is_tensor(X_2b):
        X_2b = X_2b.cpu().numpy()
    
    # Initialize result containers
    chi2 = {}
    chi2_2b = {} if X_2b is not None else None
    correlations = []
    qq_distances = []
    mean_residue_differences = [] if (include_residue_analysis and X_2b is not None) else None
    kl_divergence = {}
    mmd = {}
    ks_statistic = {}
    
    # Check for NaN values and warn user
    if np.isnan(X_target).any():
        nan_count = np.isnan(X_target).sum()
        total_elements = X_target.size
        print(f"Warning: X_target contains {nan_count}/{total_elements} NaN values ({100*nan_count/total_elements:.2f}%)")
        print("NaN values will be filtered out for plotting.")
    
    number_of_samples_4b = X_4b.shape[0]
    number_of_samples_2b = X_2b.shape[0] if X_2b is not None else 0
    
    # =======================================================================
    # Per-feature analysis
    # =======================================================================
    
    for i, feature in enumerate(tqdm(features, desc="Analyzing features")):
        
        # Filter out NaN values for this feature
        X_target_feature = X_target[:, i]
        X_4b_feature = X_4b[:, i]
        
        # Create mask for valid (non-NaN) values
        valid_mask = ~np.isnan(X_target_feature)
        X_target_feature_clean = X_target_feature[valid_mask]
        
        # Calculate histogram bins based on real 4b data
        bins = number_of_bins(X_4b_feature, NN_type)
        
        # Calculate histograms for 4b and target
        X_4b_raw, bin_edges = np.histogram(X_4b_feature, bins=bins)
        bin_width = np.diff(bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize to density
        counts_4b = X_4b_raw / (bin_width * number_of_samples_4b)
        error_4b = np.sqrt(X_4b_raw) / (bin_width * number_of_samples_4b)
        
        # Calculate histogram for target
        X_target_raw, _ = np.histogram(X_target_feature_clean, bins=bin_edges)
        number_of_samples_target = len(X_target_feature_clean)
        counts_target = X_target_raw / (bin_width * number_of_samples_target)
        
        # Chi2 calculation for 4b vs target
        valid_bins_4b = error_4b > 0
        if np.sum(valid_bins_4b) > 0:
            chi2_value = np.sum((counts_4b[valid_bins_4b] - counts_target[valid_bins_4b]) ** 2 / (error_4b[valid_bins_4b] ** 2))
            chi2[feature] = chi2_value / np.sum(valid_bins_4b)
        else:
            chi2[feature] = np.nan
        
        # Kullback-Leibler divergence calculation (4b vs target)
        # KL(P||Q) = sum(P * log(P/Q)), where P is reference (4b) and Q is target
        epsilon = 1e-10  # Small value to avoid log(0)
        P = counts_4b * bin_width  # Convert density back to probability
        Q = counts_target * bin_width
        P = P / np.sum(P)  # Normalize to sum to 1
        Q = Q / np.sum(Q)
        # Only compute KL where both distributions have support
        valid_kl = (P > epsilon) & (Q > epsilon)
        if np.sum(valid_kl) > 0:
            kl_value = np.sum(P[valid_kl] * np.log(P[valid_kl] / Q[valid_kl]))
            kl_divergence[feature] = kl_value
        else:
            kl_divergence[feature] = np.nan
        
        # Maximum Mean Discrepancy (MMD) calculation using Gaussian kernel
        # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        # Use median heuristic for bandwidth selection
        def gaussian_kernel(x, y, sigma):
            """Gaussian/RBF kernel"""
            return np.exp(-0.5 * ((x - y) ** 2) / (sigma ** 2))
        
        # Subsample for computational efficiency (max 10000 samples each)
        n_subsample = min(10000, len(X_4b_feature), len(X_target_feature_clean))
        if len(X_4b_feature) > n_subsample:
            idx_4b = np.random.choice(len(X_4b_feature), n_subsample, replace=False)
            X_4b_sub = X_4b_feature[idx_4b]
        else:
            X_4b_sub = X_4b_feature
        
        if len(X_target_feature_clean) > n_subsample:
            idx_target = np.random.choice(len(X_target_feature_clean), n_subsample, replace=False)
            X_target_sub = X_target_feature_clean[idx_target]
        else:
            X_target_sub = X_target_feature_clean
        
        # Compute pairwise distances for bandwidth selection
        all_samples = np.concatenate([X_4b_sub, X_target_sub])
        pairwise_dists = np.abs(all_samples[:, None] - all_samples[None, :]) # Produce a symmetric matrix of pairwise distances
        sigma_mmd = np.median(pairwise_dists[pairwise_dists > 0]) # Median of non-zero distances
        
        if sigma_mmd > 0:
            # Compute kernel matrices
            K_xx = gaussian_kernel(X_4b_sub[:, None], X_4b_sub[None, :], sigma_mmd)
            K_yy = gaussian_kernel(X_target_sub[:, None], X_target_sub[None, :], sigma_mmd)
            K_xy = gaussian_kernel(X_4b_sub[:, None], X_target_sub[None, :], sigma_mmd)
            
            # MMD^2 estimate
            mmd_squared = K_xx.sum() / (len(X_4b_sub) ** 2) + K_yy.sum() / (len(X_target_sub) ** 2) - 2 * K_xy.sum() / (len(X_4b_sub) * len(X_target_sub))
            mmd[feature] = np.sqrt(max(0, mmd_squared))  # Take sqrt and ensure non-negative
        else:
            mmd[feature] = np.nan
        
        # Kolmogorov-Smirnov test
        # Two-sample KS test comparing cumulative distributions
        ks_stat, ks_pvalue = ks_2samp(X_4b_feature, X_target_feature_clean)
        ks_statistic[feature] = ks_stat
        
        # Calculate 2b histogram and chi2 if provided
        counts_2b = None
        error_2b = None
        residuals_4b_vs_2b = None
        
        if X_2b is not None:
            X_2b_feature = X_2b[:, i]
            X_2b_raw, _ = np.histogram(X_2b_feature, bins=bin_edges)
            counts_2b = X_2b_raw / (bin_width * number_of_samples_2b)
            error_2b = np.sqrt(X_2b_raw) / (bin_width * number_of_samples_2b)
            
            # Chi2 for 4b vs 2b
            # Use bins where 4b has data (reference distribution) instead of where 2b has data
            # This ensures we're comparing distributions in the same regions where 4b signal exists
            valid_bins_2b = error_2b > 0
            if np.sum(valid_bins_2b) > 0:
                chi2_value_2b = np.sum((counts_2b[valid_bins_2b] - counts_target[valid_bins_2b]) ** 2 / (error_2b[valid_bins_2b] ** 2))
                chi2_2b[feature] = chi2_value_2b / np.sum(valid_bins_2b)
            else:
                chi2_2b[feature] = np.nan
        
        # Calculate residuals
        with np.errstate(divide='ignore', invalid='ignore'):
            residuals_4b_vs_target = np.where(error_4b > 0, (counts_4b - counts_target) / error_4b, 0)
            if X_2b is not None:
                residuals_4b_vs_2b = np.where(error_4b > 0, (counts_4b - counts_2b) / error_4b, 0)
        
        # Plot distributions - combined 2x2 grid
        plot_4bvs2b_and_4bvs4breco(counts_2b, counts_target, counts_4b, error_4b, 
                                   feature, dir_path_transformation, bin_edges, bin_centers, bin_width)
        
        # Calculate max residual for consistent y-axis limits
        with np.errstate(divide='ignore', invalid='ignore'):
            residuals_4b_vs_target_tmp = np.where(error_4b > 0, (counts_4b - counts_target) / error_4b, 0)
            if X_2b is not None:
                residuals_4b_vs_2b_tmp = np.where(error_4b > 0, (counts_4b - counts_2b) / error_4b, 0)
                max_residual = max(np.abs(residuals_4b_vs_2b_tmp).max(), np.abs(residuals_4b_vs_target_tmp).max())
            else:
                max_residual = np.abs(residuals_4b_vs_target_tmp).max()
        
        # Save individual distribution plots in subdirectories
        # 4b vs target
        plot_single_distribution_comparison(
            counts_comparison=counts_target,
            counts_4b=counts_4b,
            error_4b=error_4b,
            feature_name=feature,
            dir_path=dir_path_4b_vs_target,
            bin_edges=bin_edges,
            bin_centers=bin_centers,
            bin_width=bin_width,
            comparison_label=target_name,
            comparison_color='blue'
        )
        
        # 4b vs 2b (if available)
        if X_2b is not None:
            plot_single_distribution_comparison(
                counts_comparison=counts_2b,
                counts_4b=counts_4b,
                error_4b=error_4b,
                feature_name=feature,
                dir_path=dir_path_4b_vs_2b,
                bin_edges=bin_edges,
                bin_centers=bin_centers,
                bin_width=bin_width,
                comparison_label='2b',
                comparison_color='green'
            )
        
        # Residue Difference Analysis (only if X_2b is provided)
        if include_residue_analysis and X_2b is not None and residuals_4b_vs_2b is not None:
            residue_difference = residuals_4b_vs_2b - residuals_4b_vs_target
            mean_residue_diff = np.mean(np.abs(residue_difference))
            mean_residue_differences.append(mean_residue_diff)
            
            # Plot residue difference for this feature
            plt.figure(figsize=(10, 6))
            plt.bar(bin_centers, residue_difference, width=bin_width, color='purple', alpha=0.7)
            plt.axhline(0, color='black', linestyle='-', linewidth=1.5)
            plt.xlabel(feature, fontsize=11)
            plt.ylabel(r'$(4b-2b)/\sigma_{4b} - (4b-' + target_name + r')/\sigma_{4b}$', fontsize=10)
            plt.title(f'Residue Difference: {feature}', fontsize=13, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
            plt.tight_layout()
            plt.savefig(dir_path_residue + f'{feature}_residue_difference.png', dpi=150)
            plt.close()
        
        # Correlation calculation (only if paired with 2b)
        if paired_with_2b and X_2b is not None:
            X_2b_feature = X_2b[:, i]
            X_2b_aligned = X_2b_feature[valid_mask]
            
            if len(X_2b_aligned) > 1 and len(X_target_feature_clean) > 1:
                corr_coef = np.corrcoef(X_2b_aligned, X_target_feature_clean)[0, 1]
            else:
                corr_coef = np.nan
            correlations.append(corr_coef)
        
        # Scatter and QQ plots
        if include_scatter_plots:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            
            # Left plot: scatter if paired with 2b, otherwise skip
            if paired_with_2b and X_2b is not None:
                X_2b_feature = X_2b[:, i]
                X_2b_aligned = X_2b_feature[valid_mask]
                
                ax[0].scatter(X_2b_aligned, X_target_feature_clean, alpha=0.5, label=f'2b vs {target_name}', color='green')
                ax[0].set_xlabel(f'2b {feature}')
                ax[0].set_ylabel(f'{target_name} {feature}')
                ax[0].set_title(f'2b vs {target_name}: {feature}')
                if len(X_2b_aligned) > 0:
                    ax[0].plot([X_2b_aligned.min(), X_2b_aligned.max()], 
                              [X_2b_aligned.min(), X_2b_aligned.max()], 
                              'r--', label='y=x')
                ax[0].legend()
            else:
                ax[0].text(0.5, 0.5, 'Scatter plot not applicable', 
                          ha='center', va='center', transform=ax[0].transAxes)
                ax[0].set_title(f'2b vs {target_name}: {feature}')
            
            # Right plot: QQ plot of 4b vs target
            # Bootstrap 4b to match sample size for fair comparison
            bootstrap_indices = np.random.choice(len(X_4b_feature), size=len(X_target_feature_clean), replace=True)
            X_4b_bootstrap = X_4b_feature[bootstrap_indices]
            sorted_4b = np.sort(X_4b_bootstrap)
            sorted_target = np.sort(X_target_feature_clean)
            
            ax[1].scatter(sorted_4b, sorted_target, alpha=0.5, color='blue')
            ax[1].plot([sorted_4b.min(), sorted_4b.max()], 
                       [sorted_4b.min(), sorted_4b.max()], 
                       'r--')
            ax[1].set_xlabel(f'Quantiles of 4b {feature}')
            ax[1].set_ylabel(f'Quantiles of {target_name} {feature}')
            ax[1].set_title(f'QQ Plot: 4b vs {target_name} {feature}')
            
            # Calculate distance from y=x line for QQ plot
            qq_dist = np.abs(sorted_target - sorted_4b) / np.sqrt(2)
            mean_qq_dist = np.mean(qq_dist)
            qq_distances.append(mean_qq_dist)
            
            plt.tight_layout()
            plt.savefig(dir_path_scatter + f'{feature}_scatter.png', dpi=150)
            plt.close()
            
            # Save individual scatter plots in subdirectories
            # 2b vs target scatter (if applicable)
            if paired_with_2b and X_2b is not None:
                X_2b_feature = X_2b[:, i]
                X_2b_aligned = X_2b_feature[valid_mask]
                
                plot_single_scatter(
                    x_data=X_2b_aligned,
                    y_data=X_target_feature_clean,
                    feature_name=feature,
                    dir_path=dir_path_2b_vs_target,
                    x_label=f'2b {feature}',
                    y_label=f'{target_name} {feature}',
                    title=f'2b vs {target_name}: {feature}',
                    color='green',
                    add_identity_line=True
                )
            
            # QQ plot (4b vs target)
            plot_single_scatter(
                x_data=sorted_4b,
                y_data=sorted_target,
                feature_name=feature,
                dir_path=dir_path_qq,
                x_label=f'Quantiles of 4b {feature}',
                y_label=f'Quantiles of {target_name} {feature}',
                title=f'QQ Plot: 4b vs {target_name} {feature}',
                color='blue',
                add_identity_line=True
            )
    
    # =======================================================================
    # Summary plots
    # =======================================================================
    
    print("\nGenerating summary plots...")
    
    # Plot chi-squared values for 4b vs target
    create_summary_bar_plot(
        data_dict=chi2,
        features_list=features,
        title=f'Chi-squared Comparison of 4b vs {target_name} Distributions',
        ylabel='Chi-squared Value',
        filename=dir_path_base + 'chi2_comparison.png',
        color='purple',
        ylim=(0, 10),
        add_hline=2,
        hline_color='red'
    )
    
    # Plot chi-squared for 2b vs 4b if available
    if chi2_2b is not None:
        create_summary_bar_plot(
            data_dict=chi2_2b,
            features_list=features,
            title='Chi-squared Comparison of 2b vs 4b Distributions',
            ylabel='Chi-squared Value',
            filename=dir_path_base + 'chi2_2b_comparison.png',
            color='orange',
            ylim=(0, 10),
            add_hline=2,
            hline_color='red'
        )
    
    # Plot correlations if computed
    if correlations and paired_with_2b:
        create_summary_bar_plot(
            data_dict=correlations,
            features_list=features,
            title=f'Correlation between 2b and {target_name} for Each Feature',
            ylabel='Pearson Correlation Coefficient',
            filename=dir_path_base + 'correlation_comparison.png',
            color='orange'
        )
    
    # Plot QQ distances if computed
    if qq_distances and include_scatter_plots:
        create_summary_bar_plot(
            data_dict=qq_distances,
            features_list=features,
            title=f'QQ Plot: Mean Distance from Perfect Agreement (4b vs {target_name})',
            ylabel='Mean Distance from y=x line',
            filename=dir_path_base + 'qq_distance_comparison.png',
            color='teal',
            ylim=(0, 0.1)
        )
    
    # Plot mean residue differences if computed
    if mean_residue_differences:
        create_summary_bar_plot(
            data_dict=mean_residue_differences,
            features_list=features,
            title=f'Mean Residue Difference: |(4b-2b)/σ - (4b-{target_name})/σ|',
            ylabel='Mean Absolute Residue Difference',
            filename=dir_path_base + 'mean_residue_difference_summary.png',
            color='purple',
            ylim=(0, 10)
        )
    
    # Plot Kullback-Leibler divergence
    create_summary_bar_plot(
        data_dict=kl_divergence,
        features_list=features,
        title=f'Kullback-Leibler Divergence: KL(4b || {target_name})',
        ylabel='KL Divergence',
        filename=dir_path_base + 'kl_divergence_comparison.png',
        color='darkgreen',
        sort_descending=True,
        ylim=(0, 0.005)
    )
    
    # Plot Maximum Mean Discrepancy
    create_summary_bar_plot(
        data_dict=mmd,
        features_list=features,
        title=f'Maximum Mean Discrepancy: 4b vs {target_name}',
        ylabel='MMD',
        filename=dir_path_base + 'mmd_comparison.png',
        color='darkred',
        sort_descending=True,
        ylim=(0, 0.04)
    )
    
    # Plot Kolmogorov-Smirnov test statistic
    create_summary_bar_plot(
        data_dict=ks_statistic,
        features_list=features,
        title=f'Kolmogorov-Smirnov Test Statistic: 4b vs {target_name}',
        ylabel='KS Statistic',
        filename=dir_path_base + 'ks_statistic_comparison.png',
        color='steelblue',
        sort_descending=True,
        ylim=(0, 0.03)
    )
    
    print(f"  Summary plots saved to {dir_path_base}")
    
    return {
        'chi2': chi2,
        'chi2_2b': chi2_2b,
        'correlations': correlations,
        'qq_distances': qq_distances,
        'mean_residue_differences': mean_residue_differences,
        'kl_divergence': kl_divergence,
        'mmd': mmd,
        'ks_statistic': ks_statistic
    }


def latent_space_normality_analysis(z_latent, features, dir_path_base):
    """
    Analyze normality of latent space representations.
    Tests whether latent dimensions follow N(0,1) distribution.
    
    param z_latent: np.ndarray, latent space representations (n_samples, n_features)
    param features: list of str, feature names for labeling plots
    param dir_path_base: str, base directory for saving plots
    
    return: dict with fit statistics {'mu': array, 'sigma': array}
    """
    print("\n" + "="*60)
    print("Latent Space Normality Analysis")
    print("="*60)
    print("\nAnalyzing latent space normality (should be N(0,1))...")
    
    # Convert to numpy if needed
    if torch.is_tensor(z_latent):
        z_latent = z_latent.cpu().numpy()
    
    # Create directories
    dir_path_normal = dir_path_base + 'latent_normality/'
    os.makedirs(dir_path_normal, exist_ok=True)
    
    # Lists to store fit statistics
    fit_stats = {
        'mu': [],
        'sigma': []
    }
    
    print("\nPer-dimension Fit Results:")
    for i in tqdm(range(len(features)), desc="Analyzing latent dimensions"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        # Histogram of latent space
        ax.hist(z_latent[:, i], bins=50, density=True, alpha=0.3, color='blue', label='Latent Histogram')
        
        # Fit Gaussian to latent distribution
        mu, std = norm.fit(z_latent[:, i])
        
        # Save stats
        fit_stats['mu'].append(mu)
        fit_stats['sigma'].append(std)
        
        # Print stats
        print(f"  z_{features[i]}: μ={mu:.3f}, σ={std:.3f}")
        
        # Generate points for plotting
        x_range = np.linspace(-5, 5, 200)
        
        # Plot Fitted Gaussian
        ax.plot(x_range, norm.pdf(x_range, mu, std), 'b-', linewidth=2,
               label=f'Fitted N({mu:.2f}, {std:.2f})')
        
        # Plot Target N(0,1)
        ax.plot(x_range, norm.pdf(x_range, 0, 1), 'k--', linewidth=2, alpha=0.7,
               label='Target N(0,1)')
        
        ax.set_xlabel(f'z_{features[i]}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Latent Space Normality Test: z_{features[i]}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(dir_path_normal + f'latent_z_{features[i]}_normality.png', dpi=150)
        plt.close()
    
    # Convert to arrays
    fit_stats['mu'] = np.array(fit_stats['mu'])
    fit_stats['sigma'] = np.array(fit_stats['sigma'])
    
    # Summary Plot: Mean Residuals (μ - 0)
    create_summary_bar_plot(
        data_dict=fit_stats['mu'].tolist(),
        features_list=features,
        title='Latent Space: Mean Deviation from N(0,1) Target',
        ylabel='Deviation from Target Mean 0 (μ - 0)',
        filename=dir_path_base + 'latent_mean_residuals.png',
        color='blue',
        sort_descending=False,
        add_hline=0
    )
    
    # Summary Plot: Sigma Residuals (σ - 1)
    sigma_deviations = (fit_stats['sigma'] - 1).tolist()
    create_summary_bar_plot(
        data_dict=sigma_deviations,
        features_list=features,
        title='Latent Space: Sigma Deviation from N(0,1) Target',
        ylabel='Deviation from Target Sigma 1 (σ - 1)',
        filename=dir_path_base + 'latent_sigma_residuals.png',
        color='red',
        sort_descending=False,
        add_hline=0
    )
    
    # Print summary statistics
    print(f"\n--- Latent Space Normality Summary ---")
    print(f"  Mean μ: {np.mean(fit_stats['mu']):.4f} ± {np.std(fit_stats['mu']):.4f}")
    print(f"  Mean σ: {np.mean(fit_stats['sigma']):.4f} ± {np.std(fit_stats['sigma']):.4f}")
    print(f"  Dimensions with |μ| > 0.1: {np.sum(np.abs(fit_stats['mu']) > 0.1)}/{len(features)}")
    print(f"  Dimensions with |σ-1| > 0.1: {np.sum(np.abs(fit_stats['sigma'] - 1) > 0.1)}/{len(features)}")
    
    return fit_stats


def plot_reweighting_patterns(events_2b, weights, features_to_analyze, all_features, dir_path, window_size=1000):
    """
    Create reweighting pattern heatmap plots showing how weights vary with feature values.
    
    param events_2b: np.ndarray, 2b events data (n_samples, n_features)
    param weights: np.ndarray, reweighting weights for each event (n_samples,)
    param features_to_analyze: list of str, feature names to analyze and plot
    param all_features: list of str, all feature names (for indexing into events_2b)
    param dir_path: str, base directory path for saving plots
    param window_size: int, window size for moving average trend line
    
    return: 0
    """
    print("\n" + "="*60)
    print("Reweighting Pattern Analysis - VISUALIZATION PHASE")
    print("="*60)
    
    # Setup directories
    dir_path_weights = dir_path + 'weights_analysis/'
    os.makedirs(dir_path_weights, exist_ok=True)
    
    # Print weight statistics
    print(f"\nWeight statistics: min={weights.min():.4f}, max={weights.max():.4f}, "
          f"mean={weights.mean():.4f}, median={np.median(weights):.4f}")
    
    print(f"\nGenerating heatmap plots for {len(features_to_analyze)} features...")
    
    for i, feature in enumerate(tqdm(features_to_analyze, desc="Heatmap plots")):
        feature_idx = all_features.index(feature)
        feature_values = events_2b[:, feature_idx]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x_bins = 250
        y_bins = 250
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(feature_values, weights, bins=[x_bins, y_bins])
        hist = hist.T  # Transpose so rows=y, columns=x
        
        # --- NEW: Convert raw counts to fraction of total events ---
        total_events = hist.sum()
        if total_events > 0:
            hist = hist / total_events
        
        # Plot heatmap showing the fraction of events
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(hist, origin='lower', aspect='auto', cmap='viridis',
                      extent=extent, interpolation='gaussian', alpha=0.7)
        cbar = plt.colorbar(im, ax=ax, label='Fraction of events')
        
        # Blue line at y = 1 (reference weight)
        ax.axhline(y=1, color='blue', linestyle='--', linewidth=2, 
                  label='Reference weight (w = 1)', zorder=5)
        
        # Moving average to show trend
        sorted_idx = np.argsort(feature_values)
        if len(feature_values) > window_size:
            # Use convolution for moving average
            weights_sorted = np.clip(weights[sorted_idx], 0, 100)
            moving_avg = np.convolve(weights_sorted, np.ones(window_size)/window_size, mode='valid')
            x_smooth = feature_values[sorted_idx][window_size-1:]
            ax.plot(x_smooth, moving_avg, color='red', linewidth=3, 
                   label=f'Moving average (window={window_size})', zorder=10)
        
        ax.set_xlabel(f'{feature}', fontsize=11)
        ax.set_ylabel('Weight (log scale)', fontsize=11)
        ax.set_title(f'Reweighting Pattern: {feature}', fontsize=13, fontweight='bold')
        ax.set_ylim(0.01, 100)  # Set y-axis limits
        ax.set_yscale('log')  # Log scale
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        plt.savefig(dir_path_weights + f'weight_pattern_{feature}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n  Plots saved in: {dir_path_weights}")
    print("\nReweighting pattern visualization completed")
    return 0