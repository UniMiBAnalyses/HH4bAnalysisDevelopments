import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import h5py
import shutil

# Add parent directory to path to import from lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import data loader
from lib import data_loader as dl

# Import all testing utility functions from existing tester
from lib.tester_function import (
    number_of_bins,
    ROC_preparation,
    ROC_curve_plot,
    ConfusionMatrix,
    perform_binary_classification_analysis,
    data_preparing_for_comparison,
    plot_4bvs2b_and_4bvs4breco,
    plot_single_distribution_comparison,
    plot_single_scatter,
    create_summary_bar_plot,
    distribution_comparison_analysis,
    distribution_comparison_analysis_light,
    plot_reweighting_patterns
)


# =========================================================================
# FFNModelTester Class
# =========================================================================

class FFNModelTester:
    """
    Comprehensive testing suite for the FFN classifier model.
    """
    
    def __init__(self, 
                model, 
                test_data_loader, 
                features, 
                dir_path
                ):
        """
        Initialize the tester.
        
        param model: FFN instance
        param test_data_loader: DataLoader for test data
        param features: list of feature names
        param dir_path: str, directory path for saving plots and results
        """
        self.model = model
        self.test_data_loader = test_data_loader
        self.features = features
        self.dir_path = dir_path + 'plots/'
        os.makedirs(self.dir_path, exist_ok=True)
        
        # Extract all test data
        self.X_test = []
        self.y_test = []
        for batch_X, batch_y in test_data_loader:
            self.X_test.append(batch_X)
            self.y_test.append(batch_y)
        
        self.X_test = np.vstack(self.X_test)
        self.y_test = np.concatenate(self.y_test).flatten()
        
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
        

    def _check_eval_data_exists(self, filename, required_keys=None):
        """Check if evaluation data file exists and contains all required keys."""
        filepath = os.path.join(self.eval_data_dir, filename + '.h5')
        if not os.path.exists(filepath):
            return False
        if required_keys is not None:
            try:
                with h5py.File(filepath, 'r') as f:
                    if not all(k in f for k in required_keys):
                        print(f"  [WARN] Stale cache '{filename}.h5' (missing keys). Will recompute.")
                        return False
            except Exception:
                return False
        return True


    # =========================================================================
    # CLASSIFICATION ANALYSIS 
    # =========================================================================

    def evaluate_classification(self, h5_filename='classification_analysis', batch_size=1024):
        """
        EVALUATION: Compute predicted probabilities and classification scores.
        Saves all computed data to HDF5 for later visualization.
        
        param h5_filename: Name of the h5 file to save (without extension)
        param batch_size: Size of batches for processing
        """
        print("\n" + "="*60)
        print("Classification Analysis - EVALUATION PHASE")
        print("="*60)
        
        # =======================================================================
        # Compute predicted probabilities for test set
        # =======================================================================
        print(f"\nComputing predictions in batches of {batch_size}...")
        
        proba_list = []
        
        for i in tqdm(range(0, len(self.X_test), batch_size), desc="Computing predictions"):
            batch_X = self.X_test[i:i+batch_size]
            proba_batch = self.model.predict_proba(batch_X)
            proba_list.append(proba_batch)
        
        # Concatenate all batches
        proba_all = np.vstack(proba_list)
        y_test_np = self.y_test
        
        # Extract probabilities for each class
        prob_4b = proba_all[:, 1]  # Probability of class 1 (4b)
        prob_2b = proba_all[:, 0]  # Probability of class 0 (2b) 
        epsilon = 1e-10

        log_likelihood_ratio = np.log((prob_4b) / (prob_2b + epsilon))  # Add small epsilon for numerical stability
        
        print(f"\nPredictions computed for {len(y_test_np)} samples")
        
        # =======================================================================
        # Save all computed data
        # =======================================================================
        
        classification_data = {
            'y_test': y_test_np,
            'log_likelihood_ratio': log_likelihood_ratio
        }
        
        self._save_to_h5(h5_filename, classification_data)
        print("\nClassification evaluation completed")
        return 0


    def visualize_classification(self, h5_filename='classification_analysis'):
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
        log_likelihood_ratio = data['log_likelihood_ratio']
        
        # Setup directories
        dir_path_classification = self.dir_path + 'classification/'
        os.makedirs(dir_path_classification, exist_ok=True)
        
        # =======================================================================
        # LIKELIHOOD RATIO METHOD - Visualization
        # =======================================================================
        
        print("\nGenerating probability-based classification plots...")
        metrics = perform_binary_classification_analysis(
            scores_test=log_likelihood_ratio,
            y_test=y_test_np,
            method_name="FFN Classifier",
            dir_path=dir_path_classification,
            xlabel='Log-Likelihood Ratio log(P(4b|X)/P(2b|X))',
            NN_type='FFN'
        )
        
        print("\nClassification visualization completed")
        return 0


    def classification_analysis(self, h5_filename='classification_analysis', batch_size=1024, force_recompute=False):
        """
        COMBINED: Full classification analysis (backward compatible).
        Runs evaluation if needed, then visualization.
        
        param h5_filename: Name of the h5 file to produce/load (without extension)
        param batch_size: Size of batches for processing
        param force_recompute: If True, recompute even if data exists
        """
        if force_recompute or not self._check_eval_data_exists(h5_filename, required_keys=['y_test', 'log_likelihood_ratio']):
            self.evaluate_classification(h5_filename, batch_size)
        else:
            print("\n[INFO] Classification data found. Skipping evaluation. Use force_recompute=True to recompute.")
        
        self.visualize_classification(h5_filename)
        return 0


    # =========================================================================
    # REWEIGHTING ANALYSIS (adapted from transform_2b_to_4b)
    # =========================================================================

    def evaluate_transform_2b_to_4b(self, data_test_prebootstrap, h5_filename='transform_2b_to_4b_scenario', batch_size=1024):
        """
        EVALUATION: Compute reweighting weights to transform 2b distribution to 4b.
        For FFN, we use the classifier to predict density ratio weights = P(4b|X) / P(2b|X).
        
        param data_test_prebootstrap: DataLoader for test data
        param h5_filename: Name of the h5 file to save (without extension)
        param batch_size: Size of batches for processing
        """
        print("\n" + "="*60)
        print("2b→4b Reweighting Analysis - EVALUATION PHASE")
        print("="*60)

        # Prepare test data for comparison
        X_2b, X_4b = data_preparing_for_comparison(data_test_prebootstrap)
        
        print(f"\n2b events: {X_2b.shape[0]}")
        print(f"4b events: {X_4b.shape[0]}")
        
        # =======================================================================
        # Compute reweighting weights for 2b events
        # =======================================================================
        
        print("\nComputing reweighting weights for 2b events...")
        print("Weight = P(4b|X) / P(2b|X)")
        
        # Process in batches
        n_batches = (X_2b.shape[0] + batch_size - 1) // batch_size
        proba_2b_list = []
        
        for batch_idx in tqdm(range(n_batches), desc="Processing 2b batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, X_2b.shape[0])
            X_2b_batch = X_2b[start_idx:end_idx]
            
            proba_batch = self.model.predict_proba(X_2b_batch)
            proba_2b_list.append(proba_batch)
    
        proba_2b_events = np.vstack(proba_2b_list)
        prob_4b = proba_2b_events[:, 1]
        prob_2b = proba_2b_events[:, 0]
        
        # Compute weights with numerical stability
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        weights = prob_4b / (prob_2b + epsilon)
        normalized_weights = weights / np.sum(weights) 
        
        reweighted_indices = np.random.choice(
            len(X_2b),
            size=len(X_2b),
            replace=True,
            p=normalized_weights
        )
        X_4b_reco = X_2b[reweighted_indices]
        
        print(f"Created reweighted 2b distribution with {len(X_4b_reco)} samples.")
        
        # =======================================================================
        # Save all data
        # =======================================================================
        
        transform_data = {
            'X_2b': X_2b,
            'X_4b': X_4b,
            'X_4b_reco': X_4b_reco,
            'weights': weights,
        }
        
        self._save_to_h5(h5_filename, transform_data)
        print("\n2b→4b reweighting analysis evaluation completed")
        return 0


    def visualize_transform_2b_to_4b(self, h5_filename='transform_2b_to_4b_scenario', use_light_analysis=False):
        """
        VISUALIZATION: Analyze and plot reweighted distributions.
        Shows how 2b events are reweighted to match 4b distribution.
        
        param h5_filename: Name of the h5 file to load (without extension)
        param use_light_analysis: bool, if True use light analysis (summary plots only), else use complete analysis
        """
        print("\n" + "="*60)
        print("2b→4b Reweighting Analysis - VISUALIZATION PHASE")
        print("="*60)

        # =======================================================================
        # Load pre-computed data
        # =======================================================================
        
        data = self._load_from_h5(h5_filename)
        X_2b = data['X_2b']
        X_4b = data['X_4b']
        X_4b_reco = data['X_4b_reco']
        weights = data['weights']
        
        dir_path_reweight = self.dir_path + 'reweighting_analysis/'
        os.makedirs(dir_path_reweight, exist_ok=True)
        
        # =======================================================================
        # Distribution comparison: 2b vs 4b vs Reweighted 2b
        # =======================================================================
        
        if use_light_analysis:
            print("\nUsing light analysis (summary plots only)...")
            distribution_comparison_analysis_light(
                X_2b=X_2b,
                X_4b=X_4b,
                X_target=X_4b_reco,
                features=self.features,
                dir_path_base=dir_path_reweight,
                NN_type='FFN',
                target_name="Reco 4b",
                compute_chi2_2b=True
            )
        else:
            print("\nUsing complete analysis (all plots)...")
            distribution_comparison_analysis(
                X_2b=X_2b,
                X_4b=X_4b,
                X_target=X_4b_reco,
                features=self.features,
                dir_path_base=dir_path_reweight,
                NN_type='FFN',
                include_residue_analysis=True,
                include_scatter_plots=True,
                target_name="Reco 4b",
                paired_with_2b=True            
            )

        print("\nTransform 2b→4b visualization completed")
        return 0


    def transform_2b_to_4b(self, data_test_prebootstrap, h5_filename='transform_2b_to_4b_scenario', 
                          batch_size=1024, use_light_analysis=False, force_recompute=False):
        """
        COMBINED: Full 2b reweighting analysis (backward compatible).
        Runs evaluation if needed, then visualization.
        
        Uses the FFN classifier to predict density ratio weights = P(4b|X) / P(2b|X),
        then applies these weights to transform the 2b distribution to match 4b.

        param data_test_prebootstrap: DataLoader for test data
        param h5_filename: Name of the h5 file to produce/load (without extension)
        param batch_size: Size of batches for processing
        param use_light_analysis: bool, if True use light analysis (summary plots only), else use complete analysis
        param force_recompute: If True, recompute even if data exists
        """
        if force_recompute or not self._check_eval_data_exists(h5_filename, required_keys=['X_2b', 'X_4b', 'X_4b_reco', 'weights']):
            self.evaluate_transform_2b_to_4b(data_test_prebootstrap, h5_filename, batch_size)
        else:
            print("\n[INFO] Transform 2b→4b data found. Skipping evaluation. Use force_recompute=True to recompute.")
        
        self.visualize_transform_2b_to_4b(h5_filename, use_light_analysis)
        return 0


    # =========================================================================
    # PER-FEATURE IMPORTANCE ANALYSIS
    # =========================================================================

    def evaluate_per_feature_likelihood(self, h5_filename='per_feature_likelihood_scenario', batch_size=1024):
        """
        EVALUATION: Compute per-feature importance for classification.
        Saves correlation data to HDF5.

        param h5_filename: Name of the h5 file to save (without extension)
        param batch_size: Size of batches for processing
        """
        print("\n" + "="*60)
        print("Per-Feature Importance Analysis - EVALUATION PHASE")
        print("="*60)
        
        print(f"Computing predictions in batches of {batch_size}...")
        
        proba_list = []
        
        for i in tqdm(range(0, len(self.X_test), batch_size), desc="Computing predictions"):
            batch_X = self.X_test[i:i+batch_size]
            proba_batch = self.model.predict_proba(batch_X)
            proba_list.append(proba_batch)
        
        # Concatenate all batches
        proba_all = np.vstack(proba_list)
        prob_4b = proba_all[:, 1]  # Probability of class 1 (4b)
        
        # Test data is already numpy
        X_test_np = self.X_test
        
        # Compute correlation between features and predicted probability
        print("\nComputing feature-probability correlations...")
        correlations = []
        for i in range(len(self.features)):
            corr = np.corrcoef(X_test_np[:, i], prob_4b)[0, 1]
            correlations.append(abs(corr))
        
        # =======================================================================
        # Save data
        # =======================================================================
        
        feature_importance_data = {
            'correlations': np.array(correlations)
        }
        
        self._save_to_h5(h5_filename, feature_importance_data)
        print("\nPer-feature importance evaluation completed")
        return 0


    def visualize_per_feature_likelihood(self, h5_filename='per_feature_likelihood_scenario', top_n=20):
        """
        VISUALIZATION: Create per-feature importance plots.
        Loads pre-computed correlations from HDF5.

        param h5_filename: Name of the h5 file to load (without extension)
        param top_n: int, number of top features to display
        """
        print("\n" + "="*60)
        print("Per-Feature Importance Analysis - VISUALIZATION PHASE")
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
        plt.xlabel('Absolute Correlation with Predicted Probability P(4b|X)')
        plt.title(f'Top {top_n} Most Important Features for Classification')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.dir_path + 'feature_importance.png', dpi=150)
        print(f"\n  Plot saved: feature_importance.png")
        plt.close()
        
        print("\nPer-feature importance visualization completed")
        return 0


    def per_feature_likelihood(self, h5_filename='per_feature_likelihood_scenario', top_n=20, 
                              batch_size=1024, force_recompute=False):
        """
        COMBINED: Full per-feature importance analysis (backward compatible).
        Runs evaluation if needed, then visualization.

        param h5_filename: Name of the h5 file to produce/load (without extension)
        param top_n: int, number of top features to display
        param batch_size: Size of batches for processing
        param force_recompute: If True, recompute even if data exists
        """
        if force_recompute or not self._check_eval_data_exists(h5_filename, required_keys=['correlations']):
            self.evaluate_per_feature_likelihood(h5_filename, batch_size)
        else:
            print("\n[INFO] Per-feature importance data found. Skipping evaluation. Use force_recompute=True to recompute.")
        
        self.visualize_per_feature_likelihood(h5_filename, top_n)
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
        
        # Compute predicted probabilities
        print(f"\nComputing predictions in batches of {batch_size}...")
        proba_list = []
        
        for i in tqdm(range(0, len(self.X_test), batch_size), desc="Computing predictions"):
            batch_X = self.X_test[i:i+batch_size]
            proba_batch = self.model.predict_proba(batch_X)
            proba_list.append(proba_batch)
        
        proba_all = np.vstack(proba_list)
        y_test_np = self.y_test
        
        # Extract 2b events
        mask_2b = (y_test_np == 0)
        events_2b = self.X_test[mask_2b]
        prob_2b = proba_all[mask_2b, 0]  # P(class=0|X) for 2b events
        prob_4b = proba_all[mask_2b, 1]  # P(class=1|X) for 2b events
        
        # Compute weights: ratio of probabilities
        epsilon = 1e-10
        weights = prob_4b / (prob_2b + epsilon)
        
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
        VISUALIZATION: Create reweighting pattern heatmap plots.
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
        param window_size: window size for moving average (to show trend in heatmap plot)
        param force_recompute: If True, recompute even if data exists
        """
        if force_recompute or not self._check_eval_data_exists(h5_filename, required_keys=['weights', 'events_2b']):
            self.evaluate_reweighting_patterns(features, h5_filename, batch_size)
        else:
            print("\n[INFO] Reweighting patterns data found. Skipping evaluation. Use force_recompute=True to recompute.")
        
        self.visualize_reweighting_patterns(h5_filename, window_size)
        return 0
