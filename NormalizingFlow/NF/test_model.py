import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys
import h5py

# Add parent directory to path to import from lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model and data loader
from lib import data_loader as dl
import model as md

# Import all testing utility functions
from lib.tester_function import (
    number_of_bins,
    ROC_preparation,
    ROC_curve_plot,
    ConfusionMatrix,
    perform_binary_classification_analysis,
    data_preparing_for_comparison,
    plot_4bvs2b_and_4bvs4breco,
    create_summary_bar_plot,
    distribution_comparison_analysis,
    distribution_comparison_analysis_light,
    latent_space_normality_analysis,
    plot_reweighting_patterns
)


# =========================================================================
# ModelTester Class
# =========================================================================

class ModelTester:
    """
    Comprehensive testing suite for the normalizing flow model.
    """
    
    def __init__(self, 
                model, 
                test_data_loader, 
                features, 
                dir_path
                ):
        """
        Initialize the tester.
        
        param model: FlowModel instance
        param test_data_loader: DataLoader for test data
        param features: list of feature names
        param dir_path: str, directory path for saving plots and results
        """
        self.model = model
        self.test_data_loader = test_data_loader
        self.features = features
        self.device = model.device
        self.dir_path = dir_path + 'plots/'
        os.makedirs(self.dir_path, exist_ok=True)
        
        # Extract all test data
        self.X_test = []
        self.y_test = []
        for batch_X, batch_y in test_data_loader:
            self.X_test.append(batch_X)
            self.y_test.append(batch_y)
        
        self.X_test = torch.cat(self.X_test, dim=0)
        self.y_test = torch.cat(self.y_test, dim=0)
        print(f"Test data shape: X={self.X_test.shape}, y={self.y_test.shape}")
        
        # Setup evaluation data directory
        self.eval_data_dir = dir_path + 'eval_data/'
        os.makedirs(self.eval_data_dir, exist_ok=True)
        print(f"Evaluation data will be saved to: {self.eval_data_dir}")


    def _save_to_h5(self, filename, data_dict):
        """
        Save data dictionary to HDF5 file.
        
        param filename: str, name of HDF5 file (without extension)
        param data_dict: dict, dictionary with string keys and numpy array or scalar values
        """
        filepath = os.path.join(self.eval_data_dir, filename + '.h5')
        with h5py.File(filepath, 'w') as f:
            for key, value in data_dict.items():
                if isinstance(value, (list, tuple)):
                    value = np.array(value)
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                f.create_dataset(key, data=value)
        print(f"  Data saved to: {filename}.h5")


    def _load_from_h5(self, filename):
        """
        Load data from HDF5 file.
        
        param filename: str, name of HDF5 file (without extension)
        return: dict with loaded data
        """
        filepath = os.path.join(self.eval_data_dir, filename + '.h5')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Evaluation data not found: {filepath}\n"
                                  f"Please run the corresponding evaluate_* method first.")
        
        data_dict = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data_dict[key] = f[key][()]
        print(f"  Data loaded from: {filename}.h5")
        return data_dict
        

    def _check_eval_data_exists(self, filename):
        """Check if evaluation data file exists."""
        filepath = os.path.join(self.eval_data_dir, filename + '.h5')
        return os.path.exists(filepath)
        

    def data_preprocessing(self, X, y, batch_size=1024):
        """
        Preprocess test data and compute log probabilities.

        param X: Features
        param y: Labels
        param batch_size: Size of each batch for processing
        return: Numpy arrays of X, y, log_prob_given_2b, log_prob_given_4b

        """
        log_prob_given_2b_list = []
        log_prob_given_4b_list = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(X), batch_size), desc="Computing log probabilities"):
                X_batch = X[i:i+batch_size].to(self.device)
                y_batch = y[i:i+batch_size].to(self.device)

                # Compute log probabilities for both classes                
                log_prob_2b = self.model.evaluate_log_prob(X_batch, torch.zeros_like(y_batch))
                log_prob_4b = self.model.evaluate_log_prob(X_batch, torch.ones_like(y_batch))
                
                log_prob_given_2b_list.append(log_prob_2b.cpu())
                log_prob_given_4b_list.append(log_prob_4b.cpu())
        
        # Concatenate all batches
        log_prob_given_2b = torch.cat(log_prob_given_2b_list).numpy()
        log_prob_given_4b = torch.cat(log_prob_given_4b_list).numpy()
        # Convert to numpy arrays
        y_np = y.cpu().numpy().flatten()
        X_np = X.cpu().numpy()

        return X_np, y_np, log_prob_given_2b, log_prob_given_4b


    # =========================================================================
    # CLASSIFICATION ANALYSIS 
    # =========================================================================

    def evaluate_classification(self, h5_filename='classification', batch_size=1024):
        """
        EVALUATION: Compute log probabilities and classification scores.
        Saves all computed data to HDF5 for later visualization.
        
        param h5_filename: Name of the h5 file to save (without extension)
        param batch_size: Size of batches for processing
        """
        print("\n" + "="*60)
        print("Classification Analysis - EVALUATION PHASE")
        print("="*60)
        
        # =======================================================================
        # Compute log probabilities for test set only
        # =======================================================================
        print(f"\nComputing log probabilities in batches of {batch_size}...")
        X_test_np, y_test_np, log_prob_given_2b_test, log_prob_given_4b_test = self.data_preprocessing(
            self.X_test, self.y_test, batch_size)
        
        # =======================================================================
        # LIKELIHOOD RATIO METHOD
        # =======================================================================
        
        # Log likelihood ratio: log[P(X|y=1) / P(X|y=0)]
        log_likelihood_ratio_test = log_prob_given_4b_test - log_prob_given_2b_test
        
        # =======================================================================
        # Save all computed data
        # =======================================================================
        
        classification_data = {
            'y_test': y_test_np,
            'log_likelihood_ratio_test': log_likelihood_ratio_test,
        }
        
        self._save_to_h5(h5_filename, classification_data)
        print("\nClassification evaluation completed")
        return 0


    def visualize_classification(self, h5_filename='classification'):
        """
        VISUALIZATION: Create all classification plots using pre-computed data.
        Loads data from HDF5 and generates all visualizations.
        
        param h5_filename: Name of the h5 file to load (without extension)
        """
        print("\n" + "="*60)
        print("Classification Analysis - VISUALIZATION PHASE")
        print("="*60)
        
        # =======================================================================
        # Load pre-computed data
        # =======================================================================
        
        data = self._load_from_h5(h5_filename)
        y_test_np = data['y_test']
        log_likelihood_ratio_test = data['log_likelihood_ratio_test']
        
        # Setup directories
        dir_path_classification = self.dir_path + 'classification/'
        os.makedirs(dir_path_classification, exist_ok=True)
        
        # =======================================================================
        # LIKELIHOOD RATIO METHOD - Visualization
        # =======================================================================
        
        print("\nGenerating likelihood ratio plots...")
        metrics_lr = perform_binary_classification_analysis(
            scores_test=log_likelihood_ratio_test,
            y_test=y_test_np,
            method_name="Likelihood Ratio",
            dir_path=dir_path_classification,
            xlabel='Log Likelihood Ratio log[P(X|4b) / P(X|2b)]',
            NN_type='NF'
        )
        
        print("\nClassification visualization completed")
        return 0


    def classification_analysis(self, h5_filename='classification', batch_size=1024, force_recompute=False):
        """
        COMBINED: Full classification analysis (backward compatible).
        Runs evaluation if needed, then visualization.
        
        param h5_filename: Name of the h5 file to produce/load (without extension)
        param batch_size: Size of batches for processing
        param force_recompute: If True, recompute even if data exists
        """
        if force_recompute or not self._check_eval_data_exists(h5_filename):
            self.evaluate_classification(h5_filename, batch_size)
        else:
            print("\n[INFO] Classification data found. Skipping evaluation. Use force_recompute=True to recompute.")
        
        self.visualize_classification(h5_filename)
        return 0


    # =========================================================================
    # SAMPLE GENERATION
    # =========================================================================

    def evaluate_sample_generation(self, data_test_prebootstrap, h5_filename='sample_generation', num_samples=int(1e5), batch_size=2000):
        """
        EVALUATION: Generate samples from the model.
        Saves generated samples and real data to HDF5.

        param data_test_prebootstrap: DataLoader for test data without bootstrap
        param h5_filename: Name of the h5 file to save (without extension)
        param num_samples: number of samples to generate for each class
        param batch_size: number of samples to generate in each batch
        """
        print("\n" + "="*60)
        print("Sample Generation - EVALUATION PHASE")
        print("="*60)
        
        # =======================================================================
        # Generate samples from the model
        # =======================================================================
        
        print(f"\nGenerating {num_samples} samples for 4b class in batches of {batch_size}...")
        
        samples_4b_list = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Sampling batches"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, num_samples)
            current_batch_size = batch_end - batch_start
            
            # Sample from 4b class
            try:
                context_4b = torch.ones((current_batch_size, 1)).to(self.device)
                samples_4b_batch = self.model.sample_generator(current_batch_size, context_4b).cpu().numpy()
                samples_4b_list.append(samples_4b_batch)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n  CUDA out of memory at batch {i+1}. Clearing cache and retrying...")
                    torch.cuda.empty_cache()
                    context_4b = torch.ones((current_batch_size, 1)).to(self.device)
                    samples_4b_batch = self.model.sample_generator(current_batch_size, context_4b).cpu().numpy()
                    samples_4b_list.append(samples_4b_batch)
                else:
                    raise
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"  Generated {batch_end}/{num_samples} samples...")
        
        samples_4b = np.vstack(samples_4b_list)
        print(f"Generated {samples_4b.shape[0]} total samples")
        
        # =======================================================================
        # Prepare real data for comparison
        # =======================================================================
        
        real_2b, real_4b = data_preparing_for_comparison(data_test_prebootstrap)
        
        # Convert to numpy
        if torch.is_tensor(real_2b):
            real_2b = real_2b.cpu().numpy()
        if torch.is_tensor(real_4b):
            real_4b = real_4b.cpu().numpy()
        
        # =======================================================================
        # Save all data
        # =======================================================================
        
        sampling_data = {
            'samples_4b': samples_4b,
            'real_2b': real_2b,
            'real_4b': real_4b
        }
        
        self._save_to_h5(h5_filename, sampling_data)
        print("\nSample generation evaluation completed")
        return 0


    def visualize_sample_generation(self, h5_filename='sample_generation'):
        """
        VISUALIZATION: Analyze and plot sample generation quality.
        Loads pre-generated samples from HDF5 and creates all plots.
        
        param h5_filename: Name of the h5 file to load (without extension)
        """
        print("\n" + "="*60)
        print("Sample Generation - VISUALIZATION PHASE")
        print("="*60)
        
        # =======================================================================
        # Load pre-computed data
        # =======================================================================
        
        data = self._load_from_h5(h5_filename)
        samples_4b = data['samples_4b']
        real_2b = data['real_2b']
        real_4b = data['real_4b']
        
        dir_path_sampling = self.dir_path + 'sampling/'
        os.makedirs(dir_path_sampling, exist_ok=True)
        
        # =======================================================================
        # Distribution comparison analysis
        # =======================================================================
        
        distribution_comparison_analysis(
            X_2b=real_2b,
            X_4b=real_4b,
            X_target=samples_4b,
            features=self.features,
            dir_path_base=dir_path_sampling,
            NN_type='NF',
            include_residue_analysis=True,
            include_scatter_plots=True,
            target_name="Sampled 4b",
            paired_with_2b=False  # Samples are not paired with 2b events
        )
        
        print("\nSample generation visualization completed")
        return 0


    def sample_generation(self, data_test_prebootstrap, h5_filename='sample_generation', num_samples=int(1e5), batch_size=2000, force_recompute=False):
        """
        COMBINED: Full sample generation analysis (backward compatible).
        Runs evaluation if needed, then visualization.
        
        param data_test_prebootstrap: DataLoader for test data without bootstrap
        param h5_filename: Name of the h5 file to produce/load (without extension)
        param num_samples: number of samples to generate for each class
        param batch_size: number of samples to generate in each batch
        param force_recompute: If True, recompute even if data exists
        """
        if force_recompute or not self._check_eval_data_exists(h5_filename):
            self.evaluate_sample_generation(data_test_prebootstrap, h5_filename, num_samples, batch_size)
        else:
            print("\n[INFO] Sample generation data found. Skipping evaluation. Use force_recompute=True to recompute.")
        
        self.visualize_sample_generation(h5_filename)
        return 0


    # =========================================================================
    # PER-FEATURE LIKELIHOOD
    # =========================================================================

    def evaluate_per_feature_likelihood(self, h5_filename='per_feature_likelihood', batch_size=1024):
        """
        EVALUATION: Compute per-feature likelihood contributions.
        Saves correlation data to HDF5.

        param h5_filename: Name of the h5 file to save (without extension)
        param batch_size: Size of batches for processing
        """
        print("\n" + "="*60)
        print("Per-Feature Likelihood Analysis - EVALUATION PHASE")
        print("="*60)
        
        print(f"Computing log probabilities in batches of {batch_size}...")
        
        X_test_np, y_test_np, log_prob_given_2b, log_prob_given_4b = self.data_preprocessing(self.X_test, self.y_test, batch_size)
        
        # Compute correlation between features and likelihood ratio
        log_likelihood_ratio = log_prob_given_4b - log_prob_given_2b
        
        correlations = []
        for i in range(len(self.features)):
            corr = np.corrcoef(X_test_np[:, i], log_likelihood_ratio)[0, 1]
            correlations.append(abs(corr))
        
        # =======================================================================
        # Save data
        # =======================================================================
        
        feature_likelihood_data = {
            'correlations': np.array(correlations)
        }
        
        self._save_to_h5(h5_filename, feature_likelihood_data)
        print("\nPer-feature likelihood evaluation completed")
        return 0


    def visualize_per_feature_likelihood(self, h5_filename='per_feature_likelihood', top_n=20):
        """
        VISUALIZATION: Create per-feature likelihood plots.
        Loads pre-computed correlations from HDF5.

        param h5_filename: Name of the h5 file to load (without extension)
        param top_n: int, number of top features to display
        """
        print("\n" + "="*60)
        print("Per-Feature Likelihood Analysis - VISUALIZATION PHASE")
        print("="*60)
        
        # =======================================================================
        # Load pre-computed data
        # =======================================================================
        
        data = self._load_from_h5(h5_filename)
        correlations = data['correlations'].tolist()
        
        # Sort by importance
        feature_importance = list(zip(self.features, correlations))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        # Plot feature importance
        top_features = [f[0] for f in feature_importance[:top_n]]
        top_corrs = [f[1] for f in feature_importance[:top_n]]
        
        plt.figure(figsize=(20, 20))
        plt.barh(range(top_n), top_corrs, color='steelblue')
        plt.yticks(range(top_n), [f for f in top_features])
        plt.xlabel('Absolute Correlation with Log Likelihood Ratio')
        plt.title(f'Top {top_n} Most Discriminative Features')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.dir_path + 'feature_importance.png', dpi=150)
        print(f"\n  Plot saved: feature_importance.png")
        plt.close()
        
        print("\nPer-feature likelihood visualization completed")
        return 0


    def per_feature_likelihood(self, h5_filename='per_feature_likelihood', top_n=20, force_recompute=False):
        """
        COMBINED: Full per-feature likelihood analysis (backward compatible).
        Runs evaluation if needed, then visualization.

        param h5_filename: Name of the h5 file to produce/load (without extension)
        param top_n: int, number of top features to display
        param force_recompute: If True, recompute even if data exists
        """
        if force_recompute or not self._check_eval_data_exists(h5_filename):
            self.evaluate_per_feature_likelihood(h5_filename)
        else:
            print("\n[INFO] Per-feature likelihood data found. Skipping evaluation. Use force_recompute=True to recompute.")
        
        self.visualize_per_feature_likelihood(h5_filename, top_n)
        return 0


    # =========================================================================
    # TRANSFORM 2b TO 4b
    # =========================================================================

    def evaluate_transform_2b_to_4b(self, data_test_prebootstrap, h5_filename='transform_2b_to_4b', batch_size=1024):
        """
        EVALUATION: Transform 2b events to 4b through latent space.
        Saves transformed data and latent representations to HDF5.

        param data_test_prebootstrap: DataLoader for test data
        param h5_filename: Name of the h5 file to save (without extension)
        param batch_size: Size of batches for processing
        """
        print("\n" + "="*60)
        print("Latent Space Transform 2b→4b - EVALUATION PHASE")
        print("="*60)

        # Prepare test data for comparison
        X_2b, X_4b = data_preparing_for_comparison(data_test_prebootstrap)

        # =======================================================================
        # Transform 2b events to 4b through latent space
        # =======================================================================
        
        print("\nTransforming 2b events to 4b through latent space...")
        print("The flow acts as an encoder-decoder:")
        print("  ENCODER: 2b data -> latent space (conditioned on 2b)")
        print("  DECODER: latent space -> 4b data (conditioned on 4b)")
        
        # Process in batches to avoid CUDA out of memory error
        n_batches = (X_2b.shape[0] + batch_size - 1) // batch_size
        X_reco_4b_batches = []
        z_latent_2b_batches = []  # Save latent representations for normality test

        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, X_2b.shape[0])
            X_2b_batch = X_2b[start_idx:end_idx]
            
            # ENCODER: Transform test data to latent space
            # Condition the flow to focus on 2b events (vector of zeros)
            condition_2b = torch.zeros(X_2b_batch.shape[0], 1, dtype=torch.float32).to(self.device)
            z = self.model.transform_feature_to_latent(X_2b_batch.to(self.device), condition_2b)
            
            # Save latent space for normality analysis
            z_latent_2b_batches.append(z.cpu())
            
            # DECODER: Transform latent space back to original space
            # Condition the flow to focus on 4b events (vector of ones)
            condition_4b = torch.ones(X_2b_batch.shape[0], 1, dtype=torch.float32).to(self.device)
            X_reco_4b_batch = self.model.transform_latent_to_feature(z, condition_4b)
            
            X_reco_4b_batches.append(X_reco_4b_batch.cpu())
        
        X_reco_4b = torch.cat(X_reco_4b_batches, dim=0)
        z_latent_2b = torch.cat(z_latent_2b_batches, dim=0)
        
        print(f"Generated {X_reco_4b.shape[0]} reconstructed 4b events")
        
        # Convert to numpy
        if torch.is_tensor(X_2b):
            X_2b = X_2b.cpu().numpy()
        if torch.is_tensor(X_4b):
            X_4b = X_4b.cpu().numpy()
        if torch.is_tensor(X_reco_4b):
            X_reco_4b = X_reco_4b.cpu().numpy()
        if torch.is_tensor(z_latent_2b):
            z_latent_2b = z_latent_2b.cpu().numpy()
        
        # =======================================================================
        # Save all data
        # =======================================================================
        
        transform_data = {
            'X_2b': X_2b,
            'X_4b': X_4b,
            'X_reco_4b': X_reco_4b,
            'z_latent_2b': z_latent_2b
        }
        
        self._save_to_h5(h5_filename, transform_data)
        print("\nTransform 2b→4b evaluation completed")
        return 0


    def visualize_transform_2b_to_4b(self, h5_filename='transform_2b_to_4b', use_light_analysis=False):
        """
        VISUALIZATION: Analyze and plot 2b to 4b transformation quality.
        Loads pre-computed transformation data from HDF5.
        
        param h5_filename: Name of the h5 file to load (without extension)
        param use_light_analysis: bool, if True use light analysis (summary plots only), else use complete analysis
        """
        print("\n" + "="*60)
        print("Latent Space Transform 2b→4b - VISUALIZATION PHASE")
        print("="*60)

        # =======================================================================
        # Load pre-computed data
        # =======================================================================
        
        data = self._load_from_h5(h5_filename)
        X_2b = data['X_2b']
        X_4b = data['X_4b']
        X_reco_4b = data['X_reco_4b']
        z_latent_2b = data['z_latent_2b']
        
        dir_path_latent = self.dir_path + 'latent_space_analysis/'
        os.makedirs(dir_path_latent, exist_ok=True)
        
        # =======================================================================
        # Distribution comparison analysis
        # =======================================================================
        
        if use_light_analysis:
            print("\nUsing light analysis (summary plots only)...")
            distribution_comparison_analysis_light(
                X_2b=X_2b,
                X_4b=X_4b,
                X_target=X_reco_4b,
                features=self.features,
                dir_path_base=dir_path_latent,
                NN_type='NF',
                target_name="Reco 4b",
                compute_chi2_2b=True
            )
        else:
            print("\nUsing complete analysis (all plots)...")
            distribution_comparison_analysis(
                X_2b=X_2b,
                X_4b=X_4b,
                X_target=X_reco_4b,
                features=self.features,
                dir_path_base=dir_path_latent,
                NN_type='NF',
                include_residue_analysis=True,
                include_scatter_plots=True,
                target_name="Reco 4b",
                paired_with_2b=True
            )
        
        # =======================================================================
        # Latent space normality analysis
        # =======================================================================
        
        latent_space_normality_analysis(
            z_latent=z_latent_2b,
            features=self.features,
            dir_path_base=dir_path_latent
        )
        
        print("\nTransform 2b→4b visualization completed")
        return 0


    def transform_2b_to_4b(self, data_test_prebootstrap, h5_filename='transform_2b_to_4b', batch_size=1024, 
                          use_light_analysis=False, force_recompute=False):
        """
        COMBINED: Full 2b to 4b transformation analysis (backward compatible).
        Runs evaluation if needed, then visualization.

        param data_test_prebootstrap: DataLoader for test data
        param h5_filename: Name of the h5 file to produce/load (without extension)
        param batch_size: Size of batches for processing
        param use_light_analysis: bool, if True use light analysis (summary plots only), else use complete analysis
        param force_recompute: If True, recompute even if data exists
        """
        if force_recompute or not self._check_eval_data_exists(h5_filename):
            self.evaluate_transform_2b_to_4b(data_test_prebootstrap, h5_filename, batch_size)
        else:
            print("\n[INFO] Transform 2b→4b data found. Skipping evaluation. Use force_recompute=True to recompute.")
        
        self.visualize_transform_2b_to_4b(h5_filename, use_light_analysis)

        return 0


    # =========================================================================
    # REWEIGHTING PATTERN ANALYSIS
    # =========================================================================

    def evaluate_reweighting_patterns(self, features=[], h5_filename='reweighting_patterns', batch_size=1024):
        """
        EVALUATION: Compute reweighting weights and feature values.
        Saves data to HDF5 for later visualization.

        param features: list of feature names to analyze (if empty, analyze all except 'era')
        param h5_filename: Name of the h5 file to save (without extension)
        param batch_size: number of samples to process in each batch (to avoid memory issues)
        """
        print("\n" + "="*60)
        print("Reweighting Pattern Analysis - EVALUATION PHASE")
        print("="*60)
        
        # Compute log probabilities
        print(f"\nComputing log probabilities in batches of {batch_size}...")
        X_test_np, y_test_np, log_prob_given_2b_test, log_prob_given_4b_test = self.data_preprocessing(
            self.X_test, self.y_test, batch_size)
        
        # Extract 2b events
        mask_2b = (y_test_np == 0)
        events_2b = X_test_np[mask_2b]
        log_prob_2b = log_prob_given_2b_test[mask_2b]
        log_prob_4b = log_prob_given_4b_test[mask_2b]
        
        # Compute weights, avoiding inf/-inf
        valid_mask = (log_prob_2b != -np.inf) & (log_prob_2b != np.inf)
        events_2b = events_2b[valid_mask]
        weights = np.exp(log_prob_4b[valid_mask] - log_prob_2b[valid_mask])
        
        if len(features) == 0:
            features = [f for f in self.features if f != 'era']
        
        print(f"\nAnalyzing {len(features)} features...")
        print(f"Number of 2b events: {len(events_2b)}")
        print(f"Weight statistics (post-clip): min={weights.min():.4f}, max={weights.max():.4f}, "
              f"mean={weights.mean():.4f}, median={np.median(weights):.4f}")
        
        # =======================================================================
        # Save data
        # =======================================================================
        
        # Convert feature names to ASCII for HDF5 compatibility
        features_array = np.array(features, dtype='S')
        
        reweighting_data = {
            'events_2b': events_2b,
            'weights': weights,
            'features_analyzed': features_array
        }
        
        self._save_to_h5(h5_filename, reweighting_data)
        print("\nReweighting pattern evaluation completed")
        return 0


    def visualize_reweighting_patterns(self, h5_filename='reweighting_patterns', window_size=1000):
        """
        VISUALIZATION: Create reweighting pattern scatter plots.
        Loads pre-computed weights and feature values from HDF5.

        param h5_filename: Name of the h5 file to load (without extension)
        param window_size: window size for moving average (to show trend in scatter plot)
        """
        # =======================================================================
        # Load pre-computed data
        # =======================================================================
        
        data = self._load_from_h5(h5_filename)
        events_2b = data['events_2b']
        weights = data['weights']
        
        features = [f for f in self.features if f != 'era']
        
        # Use common visualization function from tester_function
        plot_reweighting_patterns(
            events_2b=events_2b,
            weights=weights,
            features_to_analyze=features,
            all_features=self.features,
            dir_path=self.dir_path,
            window_size=window_size
        )
        
        return 0


    def reweighting_patterns(self, features=[], h5_filename='reweighting_patterns', batch_size=1024, 
                            window_size=1000, force_recompute=False):
        """
        COMBINED: Full reweighting pattern analysis (backward compatible).
        Runs evaluation if needed, then visualization.

        param features: list of feature names to analyze (if empty, analyze all except 'era')
        param h5_filename: Name of the h5 file to produce/load (without extension)
        param batch_size: number of samples to process in each batch (to avoid memory issues)
        param window_size: window size for moving average (to show trend in scatter plot)
        param force_recompute: If True, recompute even if data exists
        """
        if force_recompute or not self._check_eval_data_exists(h5_filename):
            self.evaluate_reweighting_patterns(features, h5_filename, batch_size)
        else:
            print("\n[INFO] Reweighting patterns data found. Skipping evaluation. Use force_recompute=True to recompute.")
        
        self.visualize_reweighting_patterns(h5_filename, window_size)
        return 0


# =========================================================================
    # EVENT-BY-EVENT FEATURE SHIFT ANALYSIS (RESIDUALS)
    # =========================================================================

    def evaluate_feature_shifts(self, data_test_prebootstrap, h5_filename='feature_shifts', batch_size=1024):
        """
        EVALUATION: Calculate event-by-event shifts (residuals) applied by the flow.
        Saves original 2b, transformed 4b, and their residuals to HDF5.
        """
        print("\n" + "="*60)
        print("Event-by-Event Feature Shifts - EVALUATION PHASE")
        print("="*60)

        # 1. Get real 2b data
        X_2b, _ = data_preparing_for_comparison(data_test_prebootstrap)
        
        print(f"\nTransforming {X_2b.shape[0]} 2b events to 4b to calculate residuals...")
        
        # 2. Process in batches to get X_reco_4b
        n_batches = (X_2b.shape[0] + batch_size - 1) // batch_size
        X_reco_4b_batches = []

        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, X_2b.shape[0])
            X_2b_batch = X_2b[start_idx:end_idx].to(self.device)
            
            with torch.no_grad():
                # Encode 2b
                condition_2b = torch.zeros(X_2b_batch.shape[0], 1, dtype=torch.float32).to(self.device)
                z = self.model.transform_feature_to_latent(X_2b_batch, condition_2b)
                
                # Decode to 4b
                condition_4b = torch.ones(X_2b_batch.shape[0], 1, dtype=torch.float32).to(self.device)
                X_reco_4b_batch = self.model.transform_latent_to_feature(z, condition_4b)
            
            X_reco_4b_batches.append(X_reco_4b_batch.cpu())
            
        X_reco_4b = torch.cat(X_reco_4b_batches, dim=0)
        
        # Convert to numpy
        if torch.is_tensor(X_2b):
            X_2b = X_2b.cpu().numpy()
        X_reco_4b = X_reco_4b.numpy()

        # 3. Calculate Residuals (Transformed - Original)
        residuals = X_reco_4b - X_2b

        # 4. Save to HDF5
        shift_data = {
            'X_2b': X_2b,
            'X_reco_4b': X_reco_4b,
            'residuals': residuals
        }
        
        self._save_to_h5(h5_filename, shift_data)
        print("\nFeature shift evaluation completed")
        return 0


    def visualize_feature_shifts(self, h5_filename='feature_shifts'):
        """
        VISUALIZATION: Plot 1D residual distributions and 2D correlation maps.
        Loads data from HDF5.
        """
        print("\n" + "="*60)
        print("Event-by-Event Feature Shifts - VISUALIZATION PHASE")
        print("="*60)

        # Load data
        data = self._load_from_h5(h5_filename)
        X_2b = data['X_2b']
        X_reco_4b = data['X_reco_4b']
        residuals = data['residuals']

        dir_path_shifts = self.dir_path + 'feature_shifts/'
        os.makedirs(dir_path_shifts, exist_ok=True)
        dir_path_residuals = dir_path_shifts + 'residuals/'
        os.makedirs(dir_path_residuals, exist_ok=True)

        means = {}
        stds = {}

        print("\nGenerating residual and 2D mapping plots for each feature...")
        
        for i, feature_name in enumerate(tqdm(self.features)):
            # Skip categorical/era if you don't want to plot shifts for them
            if feature_name in ['era', 'njet']:
                continue

            orig_vals = X_2b[:, i]
            trans_vals = X_reco_4b[:, i]
            res_vals = residuals[:, i]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # --- Plot 1: 1D Histogram of Residuals ---
            ax1.hist(res_vals, bins=number_of_bins(res_vals), color='purple', alpha=0.7)
            ax1.set_title(f'Residuals for {feature_name}\n($\Delta x = \hat{{x}}_{{4b}} - x_{{2b}}$)')
            ax1.set_xlabel('Shift applied by flow')
            ax1.set_ylabel('Events')
            ax1.set_xlim(-5, 5)
            ax1.set_yscale('log')
            ax1.grid(alpha=0.3)

            mean = np.mean(res_vals)
            std = np.std(res_vals)
            means[feature_name] = mean
            stds[feature_name] = std

            # Add text box with stats
            stats_text = f"Mean: {mean:.3f}\nStd: {std:.3f}"
            ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # --- Plot 2: 2D Hist (Original vs Transformed) ---
            # Using hist2d instead of scatter because of high event count
            h = ax2.hist2d(orig_vals, trans_vals, bins=60, cmap='viridis', cmin=1)
            fig.colorbar(h[3], ax=ax2, label='Counts')
            
            # Draw a diagonal line y=x for reference (no shift line)
            min_val = min(np.min(orig_vals), np.min(trans_vals))
            max_val = max(np.max(orig_vals), np.max(trans_vals))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x (No Shift)')
            
            ax2.set_title(f'Event Mapping: {feature_name}')
            ax2.set_xlabel('Original 2b Value')
            ax2.set_ylabel('Transformed 4b Value')
            ax2.set_xlim(-5, 5)
            ax2.set_ylim(-5, 5)
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(dir_path_residuals, f'shift_{feature_name}.png'), dpi=150)
            plt.close()
        
        create_summary_bar_plot(
            data_dict=means,
            features_list=self.features,
            title=f'Mean Shifts Applied by Flow',
            ylabel='Mean Shift',
            filename=dir_path_shifts + 'mean_shifts.png',
            color='blue'
        )

        create_summary_bar_plot(
            data_dict=stds,
            features_list=self.features,
            title=f'Standard Deviation of Shifts Applied by Flow',
            ylabel='Std of Shift',
            filename=dir_path_shifts + 'std_shifts.png',
            color='red'
        )

        print("\nFeature shift visualization completed")
        return 0


    def feature_shift_analysis(self, data_test_prebootstrap, h5_filename='feature_shifts', batch_size=1024, force_recompute=False):
        """
        COMBINED: Full event-by-event feature shift analysis.
        """
        if force_recompute or not self._check_eval_data_exists(h5_filename):
            self.evaluate_feature_shifts(data_test_prebootstrap, h5_filename, batch_size)
        else:
            print("\n[INFO] Feature shift data found. Skipping evaluation. Use force_recompute=True to recompute.")
        
        self.visualize_feature_shifts(h5_filename)
        return 0