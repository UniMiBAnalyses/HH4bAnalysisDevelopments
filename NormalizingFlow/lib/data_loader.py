import uproot
import os
import numpy as np
from random import choice as randchoice
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from . import tester_function as tf
except ImportError:
    import tester_function as tf


path_CR2b_2022EE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/2b_control_region_only/data_2022EE.root"
path_CR4b_2022EE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/4b_control_region_only/data_2022EE.root"
path_CR2b_2022preEE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/2b_control_region_only/data_2022preEE.root"
path_CR4b_2022preEE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/4b_control_region_only/data_2022preEE.root"

path_SR2b_2022EE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/2b_signal_region_only/data_2022EE.root"
path_SR4b_2022EE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/4b_signal_region_only/data_2022EE.root"
path_SR2b_2022preEE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/2b_signal_region_only/data_2022preEE.root"
path_SR4b_2022preEE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/4b_signal_region_only/data_2022preEE.root"


def bootstrap(
    X, 
    number_of_samples
    ):
    """
    Bootstrap the data to create a new dataset of specified size.

    param X: np.ndarray, feature matrix
    param number_of_samples: int, number of samples in the bootstrapped dataset
    return: np.ndarray, bootstrapped feature matrix
    """
    indices = np.random.choice(len(X), size=number_of_samples, replace=True)
    X_bootstrap = X[indices]

    return X_bootstrap


def load_data(
    path
    ):
    """
    Load data from a ROOT file using uproot.
    param path: str, path to the ROOT file
    return: dict, features extracted from the ROOT file 
    """
    with uproot.open(path) as file:
        tree = file["tree"]
        features = tree.arrays(library="np")
    return features


def era_loader(
    region="CR"
    ):
    """
    Load data for both 2b and 4b control regions for the year 2022, combining pre-EE and EE datasets.

    return: tuple of dicts (CR2b, CR4b) containing features for 2b and 4b control regions
    """
    if region == "CR":
        CR2b_2022EE = load_data(path_CR2b_2022EE)
        CR4b_2022EE = load_data(path_CR4b_2022EE)
        CR2b_2022preEE = load_data(path_CR2b_2022preEE)
        CR4b_2022preEE = load_data(path_CR4b_2022preEE)

        CR2b = {key: np.concatenate([CR2b_2022EE[key], CR2b_2022preEE[key]]) for key in CR2b_2022EE.keys()}
        CR4b = {key: np.concatenate([CR4b_2022EE[key], CR4b_2022preEE[key]]) for key in CR4b_2022EE.keys()}

        return CR2b, CR4b
    
    elif region == "SR":
        SR2b_2022EE = load_data(path_SR2b_2022EE)
        SR4b_2022EE = load_data(path_SR4b_2022EE)
        SR2b_2022preEE = load_data(path_SR2b_2022preEE)
        SR4b_2022preEE = load_data(path_SR4b_2022preEE)

        SR2b = {key: np.concatenate([SR2b_2022EE[key], SR2b_2022preEE[key]]) for key in SR2b_2022EE.keys()}
        SR4b = {key: np.concatenate([SR4b_2022EE[key], SR4b_2022preEE[key]]) for key in SR4b_2022EE.keys()}

        return SR2b, SR4b


reshaping_features = [
    "HT",
    "sigma_over_higgs1_reco_mass", "higgs1_reco_pt", 
    "sigma_over_higgs2_reco_mass", "higgs2_reco_pt",
    "hh_vec_mass", "hh_vec_pt",
    "higgs1_reco_jet1_pt", "higgs1_reco_jet1_mass", "higgs1_reco_jet2_pt", "higgs1_reco_jet2_mass",
    "higgs1_DeltaRjj",
    "higgs2_reco_jet1_pt", "higgs2_reco_jet1_mass", "higgs2_reco_jet2_pt", "higgs2_reco_jet2_mass",
    "minDeltaR_Higgjj"
]


def data_reshaping(
    data_2b, 
    data_4b, 
    features=reshaping_features, 
    plot_path=None
    ):
    """
    Reshape the data using a logarithmic transformation for specified features.

    param data_2b: dict, input data for 2b category containing features to be reshaped
    param data_4b: dict, input data for 4b category containing features to be reshaped
    param features: list of str, features to be reshaped
    param plot_path: str, path to save reshaping plots (if any)
    return: dict, reshaped data
    """
    def reshape_function(data):
        return np.log(1 + data)
    
    for feat in features:
        data_2b[feat] = reshape_function(data_2b[feat])
        data_4b[feat] = reshape_function(data_4b[feat])

        if plot_path:
            os.makedirs(plot_path, exist_ok=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Left subplot: pre-reshape
            ax1.hist(data_2b[feat], bins=tf.number_of_bins(data_2b[feat]), alpha=0.5, color='blue', density=True, label="2b")
            ax1.hist(data_4b[feat], bins=tf.number_of_bins(data_4b[feat]), alpha=0.5, color='red', density=True, label="4b")
            ax1.set_xlabel(feat)
            ax1.set_ylabel("Events")
            ax1.set_title("Pre-reshape")
            ax1.legend()
            
            # Right subplot: post-reshape
            ax2.hist(data_2b[feat], bins=tf.number_of_bins(data_2b[feat]), alpha=0.5, color='blue', density=True, label="2b")
            ax2.hist(data_4b[feat], bins=tf.number_of_bins(data_4b[feat]), alpha=0.5, color='red', density=True, label="4b")
            ax2.set_xlabel(feat)
            ax2.set_ylabel("Events")
            ax2.set_title("Post-reshape")
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, f"{feat}_reshaped.png"))
            plt.close()
    
    return data_2b, data_4b


def data_split(
    X, 
    y, 
    test_size=0.2, 
    validation_size=0.1, 
    seed=42
    ):
    """
    Split the data into training and testing sets.

    param X: np.ndarray, feature matrix
    param y: np.ndarray, labels
    param test_size: float, proportion of the dataset to include in the test split
    param validation_size: float, proportion of the training set to include in the validation split
    param seed: int, random seed for reproducibility
    return: tuple (X_train, X_test, y_train, y_test) or (X_train, X_test, y_train, y_test, X_val, y_val)
    """
    if test_size > 0 and validation_size == 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        return X_train, X_test, y_train, y_test
    elif test_size == 0 and validation_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=seed)
        return X_train, y_train, X_val, y_val
    elif test_size > 0 and validation_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=seed)
        return X_train, X_test, y_train, y_test, X_val, y_val
    else:
        X_train, y_train = X, y
        return X_train, y_train


def data_scaling(
    data, 
    features, 
    scaler=None, 
    plot_path=None
    ):
    """
    Scale the data using RobustScaler for specified features.

    param data: np.ndarray, input data matrix (N_samples, N_features)
    param features: list of str, feature names corresponding to columns
    param scaler: RobustScaler, pre-fitted scaler (if any)
    param plot_path: str, path to save scaler plots (if any)
    return: tuple (scaled_data, scaler)
    """
    if scaler is None:
        scaler = RobustScaler()
        scaler.fit(data)

    data_scaled = scaler.transform(data)

    if plot_path:
        os.makedirs(plot_path, exist_ok=True)
        
        for i, feat in enumerate(features):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Left subplot: pre-scale
            ax1.hist(data[:, i], bins=tf.number_of_bins(data[:, i]), alpha=0.5, color='blue', density=True, label="Original")
            ax1.set_xlabel(feat)
            ax1.set_ylabel("Events")
            ax1.set_title("Pre-scale")
            ax1.legend()
            
            # Right subplot: post-scale
            ax2.hist(data_scaled[:, i], bins=tf.number_of_bins(data_scaled[:, i]), alpha=0.5, color='blue', density=True, label="Scaled")
            ax2.set_xlabel(feat)
            ax2.set_ylabel("Events")
            ax2.set_title("Post-scale")
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, f"{feat}_scaled.png"))
            plt.close()
    
    return data_scaled, scaler    


def data_preparation(
    region="CR", 
    bootstrap_coef_2b=1, 
    bootstrap_coef_4b=None, 
    test_size=0.2, 
    validation_size=0.1, 
    seed=42, 
    features=None, 
    reshaping_features=reshaping_features, 
    reshaping_plot_path=None, 
    apply_scaling=True,
    scaler=None,
    scaler_plot_path=None,
    apply_clipping=True
    ):
    """
    Prepare data for classification task.

    param region: str, region to load data for ("CR" or "SR")
    param bootstrap_coef_2b: float, fraction of 2b data to bootstrap
    param bootstrap_coef_4b: float, fraction of 4b data to bootstrap (if None, it will be set to balance the dataset with 2b)
    param features: list of str, features to be used for classification
    param reshaping_features: list of str, features to be reshaped (if any)
    param reshaping_plot_path: str, path to save reshaping plots (if any)
    param apply_scaling: bool, whether to apply RobustScaler to the features
    param apply_clipping: bool, whether to apply clipping to the features
    param scaler_plot_path: str, path to save scaler plots (if any, only used if apply_scaling=True)
    return: tuple (X, y)
        X: np.ndarray, feature matrix
        y: np.ndarray, labels
    """
    data_2b, data_4b = era_loader(region=region)

    if features is None:
        features = list(data_2b.keys())

    data_2b, data_4b = data_reshaping(data_2b, data_4b, features=reshaping_features, plot_path=reshaping_plot_path)
    X_2b_pre_bootstrap = np.column_stack([data_2b[feat] for feat in features])
    X_4b_pre_bootstrap = np.column_stack([data_4b[feat] for feat in features])    
    y_2b_pre_bootstrap = np.zeros(X_2b_pre_bootstrap.shape[0])
    y_4b_pre_bootstrap = np.ones(X_4b_pre_bootstrap.shape[0])

    if bootstrap_coef_4b is None:
        bootstrap_coef_4b = (bootstrap_coef_2b * len(X_2b_pre_bootstrap)) / len(X_4b_pre_bootstrap)

    X_2b = bootstrap(X_2b_pre_bootstrap, int(bootstrap_coef_2b * len(X_2b_pre_bootstrap)))
    X_4b = bootstrap(X_4b_pre_bootstrap, int(bootstrap_coef_4b * len(X_4b_pre_bootstrap)))
    y_2b = bootstrap(y_2b_pre_bootstrap, int(bootstrap_coef_2b * len(y_2b_pre_bootstrap)))
    y_4b = bootstrap(y_4b_pre_bootstrap, int(bootstrap_coef_4b * len(y_4b_pre_bootstrap)))

    X = np.vstack((X_2b, X_4b))
    y = np.concatenate((y_2b, y_4b))
    split_data = data_split(X, y, test_size, validation_size, seed)

    if test_size > 0 and validation_size > 0:
        X_train, X_test, y_train, y_test, X_val, y_val = split_data
        if apply_scaling:
            X_train, scaler_train = data_scaling(X_train, features, scaler, scaler_plot_path)
            X_train = np.clip(X_train, -5, 5) if apply_clipping else X_train
            X_test, _ = data_scaling(X_test, features, scaler_train, scaler_plot_path)
            X_test = np.clip(X_test, -5, 5) if apply_clipping else X_test
            X_val, _ = data_scaling(X_val, features, scaler_train, scaler_plot_path)
            X_val = np.clip(X_val, -5, 5) if apply_clipping else X_val

        return X_train, X_test, y_train, y_test, X_val, y_val, scaler

    elif test_size == 0 and validation_size > 0:
        X_train, y_train, X_val, y_val = split_data
        if apply_scaling:
            X_train, scaler_train = data_scaling(X_train, features, scaler, scaler_plot_path)
            X_train = np.clip(X_train, -5, 5) if apply_clipping else X_train
            X_val, _ = data_scaling(X_val, features, scaler_train, scaler_plot_path)
            X_val = np.clip(X_val, -5, 5) if apply_clipping else X_val
        
        return X_train, y_train, X_val, y_val, scaler        

    elif test_size > 0 and validation_size == 0:
        X_train, X_test, y_train, y_test = split_data
        if apply_scaling:
            X_train, scaler_train = data_scaling(X_train, features, scaler, scaler_plot_path)
            X_train = np.clip(X_train, -5, 5) if apply_clipping else X_train
            X_test, _ = data_scaling(X_test, features, scaler_train, scaler_plot_path)
            X_test = np.clip(X_test, -5, 5) if apply_clipping else X_test

        return X_train, X_test, y_train, y_test, scaler

    else:
        X_train, y_train = split_data
        if apply_scaling:
            X_train, scaler_train = data_scaling(X_train, features, scaler, scaler_plot_path)
            X_train = np.clip(X_train, -5, 5) if apply_clipping else X_train

        return X_train, y_train, scaler


def torch_data(
    X, 
    y, 
    batch_size=512, 
    shuffle=True
    ):
    """
    Convert data to PyTorch DataLoader.

    param X: np.ndarray, feature matrix
    param y: np.ndarray, labels
    param batch_size: int, size of each batch
    param shuffle: bool, whether to shuffle the data
    return: DataLoader object
    """
    tensor_X = torch.tensor(X, dtype=torch.float32) # Convert features to float32 tensor
    tensor_y = torch.tensor(y, dtype=torch.float32) 

    dataset = TensorDataset(tensor_X, tensor_y) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 
    # A DataLoader wraps an iterable around the Dataset to enable easy access to the data in batches

    return dataloader


def full_data_loader(
    region="CR", 
    bootstrap_coef_2b=1, 
    bootstrap_coef_4b=None, 
    test_size=0.2, 
    validation_size=0.1, 
    seed=42, 
    features=None, 
    reshaping_features=reshaping_features, 
    reshaping_plot_path=None, 
    apply_scaling=True,
    scaler=None,
    scaler_plot_path=None,
    apply_clipping=True,
    batch_size=512
    ):
    """
    Full data preparation pipeline including loading, reshaping, scaling and conversion to PyTorch DataLoader.

    param region: str, region to load data for ("CR" or "SR")
    param bootstrap_coef_2b: float, fraction of 2b data to bootstrap
    param bootstrap_coef_4b: float, fraction of 4b data to bootstrap (if None, it will be set to balance the dataset with 2b)
    param features: list of str, features to be used for classification
    param reshaping_features: list of str, features to be reshaped (if any)
    param reshaping_plot_path: str, path to save reshaping plots (if any)
    param apply_scaling: bool, whether to apply RobustScaler to the features
    param scaler: RobustScaler, pre-fitted scaler (if any)
    param scaler_plot_path: str, path to save scaler plots (if any, only used if apply_scaling=True)
    param apply_clipping: bool, whether to apply clipping to the features
    param batch_size: int, size of each batch for DataLoader
    return: tuple (train_loader, test_loader) or (train_loader, test_loader, val_loader) depending on the split configuration
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        val_loader: DataLoader for validation data (if validation_size > 0)
        scaler: RobustScaler object used for scaling the features (if apply_scaling=True)
    """
    split_data = data_preparation(
        region=region, 
        bootstrap_coef_2b=bootstrap_coef_2b, 
        bootstrap_coef_4b=bootstrap_coef_4b, 
        test_size=test_size, 
        validation_size=validation_size, 
        seed=seed, 
        features=features, 
        reshaping_features=reshaping_features, 
        reshaping_plot_path=reshaping_plot_path, 
        apply_scaling=apply_scaling, 
        scaler=scaler, 
        scaler_plot_path=scaler_plot_path,
        apply_clipping=apply_clipping
    )

    if test_size > 0 and validation_size > 0:
        X_train, X_test, y_train, y_test, X_val, y_val, scaler = split_data
        train_loader = torch_data(X_train, y_train, batch_size)
        test_loader = torch_data(X_test, y_test, batch_size)
        val_loader = torch_data(X_val, y_val, batch_size)

        return train_loader, test_loader, val_loader, scaler   

    elif test_size == 0 and validation_size > 0:
        X_train, y_train, X_val, y_val, scaler = split_data
        train_loader = torch_data(X_train, y_train, batch_size)
        val_loader = torch_data(X_val, y_val, batch_size)

        return train_loader, val_loader, scaler

    elif test_size > 0 and validation_size == 0:    
        X_train, X_test, y_train, y_test, scaler = split_data
        train_loader = torch_data(X_train, y_train, batch_size)
        test_loader = torch_data(X_test, y_test, batch_size)

        return train_loader, test_loader, scaler

    else:
        X_train, y_train, scaler = split_data
        train_loader = torch_data(X_train, y_train, batch_size)

        return train_loader, scaler


try:
    from .features import features
except ImportError:
    from features import features


if __name__ == "__main__":
    path_plot = '/eos/user/a/amorandi/HH4b/data_preprocessing/'
    X, y, scaler = data_preparation(
        region="CR",
        bootstrap_coef_2b=1,
        bootstrap_coef_4b=None,
        test_size=0.2,
        validation_size=0.1,
        seed=42,
        features=features,
        reshaping_features=reshaping_features,
        reshaping_plot_path=path_plot,
        apply_scaling=True,
        scaler=None,
        scaler_plot_path=path_plot,
        apply_clipping=True
    )   