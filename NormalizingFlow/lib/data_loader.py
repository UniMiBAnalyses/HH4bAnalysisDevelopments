import uproot
import os
import numpy as np
from random import choice as randchoice
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import tester_function as tf


path_CR2b_2022EE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/2b_control_region_only/data_2022EE.root"
path_CR4b_2022EE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/4b_control_region_only/data_2022EE.root"
path_CR2b_2022preEE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/2b_control_region_only/data_2022preEE.root"
path_CR4b_2022preEE = "/eos/cms/store/group/phys_smp/rgerosa/HH4b/analysis_2022_ParkingHH/4b_control_region_only/data_2022preEE.root"


def bootstrap_data(X, number_of_samples):
    """
    Bootstrap the data to create a new dataset of specified size.

    param X: np.ndarray, feature matrix
    param number_of_samples: int, number of samples in the bootstrapped dataset
    return: np.ndarray, bootstrapped feature matrix
    """
    indices = np.random.choice(len(X), size=number_of_samples, replace=True)
    X_bootstrap = X[indices]

    return X_bootstrap


def load_data(path):
    """
    Load data from a ROOT file using uproot.
    param path: str, path to the ROOT file
    return: dict, features extracted from the ROOT file 
    """
    with uproot.open(path) as file:
        tree = file["tree"]
        features = tree.arrays(library="np")
    return features


def era_loader():
    """
    Load data for both 2b and 4b control regions for the year 2022, combining pre-EE and EE datasets.

    return: tuple of dicts (CR2b, CR4b) containing features for 2b and 4b control regions
    """
    CR2b_2022EE = load_data(path_CR2b_2022EE)
    CR4b_2022EE = load_data(path_CR4b_2022EE)
    CR2b_2022preEE = load_data(path_CR2b_2022preEE)
    CR4b_2022preEE = load_data(path_CR4b_2022preEE)

    CR2b = {key: np.concatenate([CR2b_2022EE[key], CR2b_2022preEE[key]]) for key in CR2b_2022EE.keys()}
    CR4b = {key: np.concatenate([CR4b_2022EE[key], CR4b_2022preEE[key]]) for key in CR4b_2022EE.keys()}
    
    return CR2b, CR4b


def scaler(data, min=-5, max=5):
    """
    Scale the data using StandardScaler.

    param data: np.ndarray, input data to be scaled
    param min: float, minimum value for scaling, if None, no scaling is applied
    param max: float, maximum value for scaling if None, no scaling is applied
    return: data_scaled: np.ndarray, scaled data
    """
    if min == None and max == None: 
        return data

    min_data = np.min(data)
    max_data = np.max(data)
    
    if max_data == min_data:
        return np.full_like(data, (min + max) / 2)  # Return the midpoint if all values are the same
    
    else:
        R = (max - min) / (max_data - min_data)
        data_scaled = min + R * (data - min_data)

        return data_scaled


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


def data_preparation(bootstrap_coef_2b=1, bootstrap_coef_4b=None, features=None, reshaping_features=reshaping_features, reshaping_plot_path=None, min_scale=-5, max_scale=5, scaler_plot_path=None):
    """
    Prepare data for classification task.

    param bootstrap_coef_2b: float, fraction of 2b data to bootstrap
    param bootstrap_coef_4b: float, fraction of 4b data to bootstrap (if None, it will be set to balance the dataset with 2b)
    param features: list of str, features to be used for classification
    param reshaping_features: list of str, features to be reshaped (if any)
    param reshaping_plot_path: str, path to save reshaping plots (if any)
    param min_scale: float, minimum value for scaling if scaler_plot_path is not None, otherwise no scaling is applied
    param max_scale: float, maximum value for scaling if scaler_plot_path is not None, otherwise no scaling is applied
    param scaler_plot_path: str, path to save scaler plots (if any)
    return: tuple (X, y)
        X: np.ndarray, feature matrix
        y: np.ndarray, labels
    """
    data_CR2b, data_CR4b = era_loader()

    if features is None:
        features = list(data_CR2b.keys())

    def reshape_function(data):
        """
        Reshape the data using a logarithmic transformation.

        param data: np.ndarray, input data to be reshaped
        return: np.ndarray, reshaped data
        """
        return np.log(1 + data)
    
    for feat in tqdm(features, desc="Processing features"):
        # =================================================
        # Reshaping of the feature to be more Gaussian-like, if necessary
        # =================================================
        if feat in reshaping_features:
            if reshaping_plot_path:
                data_pre_reshape_2b = data_CR2b[feat]
                data_pre_reshape_4b = data_CR4b[feat]
                data_CR2b[feat] = reshape_function(data_CR2b[feat])
                data_CR4b[feat] = reshape_function(data_CR4b[feat])

                os.makedirs(reshaping_plot_path, exist_ok=True)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Left subplot: pre-reshape
                ax1.hist(data_pre_reshape_2b, bins=tf.number_of_bins(data_pre_reshape_2b), alpha=0.5, color='blue', density=True, label="2b")
                ax1.hist(data_pre_reshape_4b, bins=tf.number_of_bins(data_pre_reshape_4b), alpha=0.5, color='red', density=True, label="4b")
                ax1.set_xlabel(feat)
                ax1.set_ylabel("Events")
                ax1.set_title("Pre-reshape")
                ax1.legend()
                
                # Right subplot: post-reshape
                ax2.hist(data_CR2b[feat], bins=tf.number_of_bins(data_CR2b[feat]), alpha=0.5, color='blue', density=True, label="2b")
                ax2.hist(data_CR4b[feat], bins=tf.number_of_bins(data_CR4b[feat]), alpha=0.5, color='red', density=True, label="4b")
                ax2.set_xlabel(feat)
                ax2.set_ylabel("Events")
                ax2.set_title("Post-reshape")
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(reshaping_plot_path, f"{feat}_reshaped.png"))
                plt.close()

            else:
                data_CR2b[feat] = reshape_function(data_CR2b[feat])
                data_CR4b[feat] = reshape_function(data_CR4b[feat])
        
        # =================================================
        # Scaling of the feature to a specified range, if necessary
        # =================================================
        if scaler_plot_path:
            data_pre_scale_2b = data_CR2b[feat]
            data_pre_scale_4b = data_CR4b[feat]
            data_CR2b[feat] = scaler(data_CR2b[feat], min_scale, max_scale)
            data_CR4b[feat] = scaler(data_CR4b[feat], min_scale, max_scale)

            os.makedirs(scaler_plot_path, exist_ok=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Left subplot: pre-scale
            ax1.hist(data_pre_scale_2b, bins=tf.number_of_bins(data_pre_scale_2b), alpha=0.5, color='blue', density=True, label="2b")
            ax1.hist(data_pre_scale_4b, bins=tf.number_of_bins(data_pre_scale_4b), alpha=0.5, color='red', density=True, label="4b")
            ax1.set_xlabel(feat)
            ax1.set_ylabel("Events")
            ax1.set_title("Pre-scale")
            ax1.legend()
            
            # Right subplot: post-scale
            ax2.hist(data_CR2b[feat], bins=tf.number_of_bins(data_CR2b[feat]), alpha=0.5, color='blue', density=True, label="2b")
            ax2.hist(data_CR4b[feat], bins=tf.number_of_bins(data_CR4b[feat]), alpha=0.5, color='red', density=True, label="4b")
            ax2.set_xlabel(feat)
            ax2.set_ylabel("Events")
            ax2.set_title("Post-scale")
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(scaler_plot_path, f"{feat}_scaled.png"))
            plt.close()

        else:
            data_CR2b[feat] = scaler(data_CR2b[feat], min_scale, max_scale)
            data_CR4b[feat] = scaler(data_CR4b[feat], min_scale, max_scale)      

    #  convert the data into numpy arrays 
    X_2b_pb = np.column_stack([data_CR2b[v] for v in features])
    X_4b_pb = np.column_stack([data_CR4b[v] for v in features])
    print(f"Shape before bootstrapping: 2b={X_2b_pb.shape}, 4b={X_4b_pb.shape}")

    # =================================================
    # Bootstrapping
    # =================================================
    X_2b = bootstrap_data(X_2b_pb, number_of_samples=int(len(X_2b_pb) * bootstrap_coef_2b))

    if bootstrap_coef_4b == None:
        bootstrap_coef_4b = len(X_2b_pb) / len(X_4b_pb)
    X_4b = bootstrap_data(X_4b_pb, number_of_samples=int(len(X_4b_pb) * bootstrap_coef_4b))
    print(f"Shape after bootstrapping: 2b={X_2b.shape}, 4b={X_4b.shape}")

    # =================================================
    # Create targets (labels)
    # =================================================
    y_2b = np.zeros((X_2b.shape[0], 1))  # Label 0 for 2b
    y_4b = np.ones((X_4b.shape[0], 1))   # Label 1 for 4b

    # Combine 2b and 4b data
    X = np.vstack([X_2b, X_4b])
    y = np.vstack([y_2b, y_4b])
    print(f"Final data shape: X={X.shape}, y={y.shape}")

    return X, y # X.shape = (N, 92), y.shape = (N, 1)


def data_split(X, y, test_size=0.2, validation_size=0.1, seed=42):
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
    

def torch_data(X, y, batch_size=512, shuffle=True):
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


def full_data_loader(bootstrap_coef_2b=1, bootstrap_coef_4b=None, features=None, reshaping_features=reshaping_features, reshaping_plot_path=None, scaler_check=False, min_scale=-5, max_scale=5, scaler_plot_path=None, test_size=0.2, validation_size=0.1, batch_size=512, seed=42):
    """
    Full data loading pipeline.

    param bootstrap_coef_2b: float, fraction of 2b data to bootstrap
    param bootstrap_coef_4b: float, fraction of 4b data to bootstrap
    param features: list of str, features to be used for classification
    param reshaping_features: list of str, features to be reshaped (if any)
    param reshaping_plot_path: str, path to save reshaping plots (if any)
    param scaler_check: bool, whether to apply scaling
    param min_scale: float, minimum value for scaling   
    param max_scale: float, maximum value for scaling
    param scaler_plot_path: str, path to save scaler plots (if any)
    param test_size: float, proportion of the dataset to include in the test split
    param validation_size: float, proportion of the training set to include in the validation split
    param batch_size: int, size of each batch for DataLoader
    param seed: int, random seed for reproducibility
    return: tuple of DataLoader objects (train_loader, test_loader, val_loader) or
            (train_loader, val_loader) or (train_loader, test_loader) depending on the specified sizes      
    """
    X, y = data_preparation(bootstrap_coef_2b, bootstrap_coef_4b, features, reshaping_features, reshaping_plot_path, min_scale, max_scale, scaler_plot_path)
    print(f"Full data shape: X={X.shape}, y={y.shape}")
    
    if scaler_check:
        X_scaled, _ = scaler(X)
    else:
        X_scaled = X

    split_data = data_split(X_scaled, y, test_size, validation_size)

    if test_size > 0 and validation_size > 0:
        X_train, X_test, y_train, y_test, X_val, y_val = split_data

        train_loader = torch_data(X_train, y_train, batch_size)
        test_loader = torch_data(X_test, y_test, batch_size)
        val_loader = torch_data(X_val, y_val, batch_size)

        return train_loader, test_loader, val_loader

    elif test_size == 0 and validation_size > 0:
        X_train, y_train, X_val, y_val = split_data

        train_loader = torch_data(X_train, y_train, batch_size)
        val_loader = torch_data(X_val, y_val, batch_size)

        return train_loader, val_loader

    elif test_size > 0 and validation_size == 0:
        X_train, X_test, y_train, y_test = split_data

        train_loader = torch_data(X_train, y_train, batch_size)
        test_loader = torch_data(X_test, y_test, batch_size)

        return train_loader, test_loader

    else:
        X_train, y_train = split_data

        train_loader = torch_data(X_train, y_train, batch_size)

        return train_loader
    

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


if __name__ == "__main__":
    path_plot = '/eos/user/a/amorandi/HH4b/data_preprocessing/'
    data_preparation(bootstrap_coef_2b=1, bootstrap_coef_4b=None, features=features, reshaping_plot_path=path_plot + 'reshaping/', scaler_plot_path=path_plot + 'scaler/')
