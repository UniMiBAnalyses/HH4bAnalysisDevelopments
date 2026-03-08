import zuko
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


# =========================================================================
# Useful functions for testing and evaluation of the normalizing flow model
# ==========================================================================

def scientific_notation_formatter(x):
    """
    Formatter for scientific notation in matplotlib plots.

    param x: float, value to format
    param pos: int, position (not used)
    return: str, formatted string in scientific notation
    """
    if x == 0:
        return "0"
    else:
        exponent = int(np.floor(np.log10(abs(x))))
        coeff = x / 10**exponent
        return "{:.4f} * 10^({})".format(coeff, exponent)


def LinearWarmupCosineDecay(optimizer, warmup_steps, total_steps):
    """
    Learning rate scheduler with linear warmup followed by cosine decay.
    
    Phase 1 (Warmup): LR increases linearly from 0 to initial_lr over warmup_steps
    Phase 2 (Decay): LR decreases following a cosine curve from initial_lr to ~0
    
    param optimizer: torch.optim.Optimizer, optimizer to schedule
    param warmup_steps: int, number of steps for linear warmup phase
    param total_steps: int, total number of training steps
    return: LambdaLR scheduler
    """
    def lr_lambda(current_step):
        # Phase 1: Linear warmup from 0 -> 1.0 over warmup_steps
        if current_step <= warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # After total_steps, keep LR at minimum (prevent it from increasing again)
        if current_step >= total_steps:
            return 1e-8  # Very small but non-zero
        
        # Phase 2: Cosine decay from 1.0 -> 0 over (warmup_steps, total_steps)
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Cosine function: goes from 1.0 -> 0.0 as progress goes from 0 -> 1
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


class EarlyStopper:
    def __init__(self, patience=30, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# =========================================================================
# Normalizing Flow Model
# ==========================================================================

class FlowModel:
    def __init__(
        self, 
        input_dim, 
        context_dim,
        bins, 
        transforms,
        hidden_features,
        dir_path,
        EarlyStopper_patience=30,
        device='cuda' if torch.cuda.is_available() else 'cpu'
        ):
        """
        Initializes the normalizing flow model.

        param input_dim: int, dimensionality of input features
        param context_dim: int, dimensionality of context variable (Target label)   
        param bins: int, number of bins for the splines in the flow
        param transforms: int, number of transformations in the flow
        param hidden_features: list of int, hidden layer sizes for the conditioner networks
        param dir_path: str, directory path for saving models and plots
        param EarlyStopper_patience: int, patience for early stopping
        param device: str, 'cuda' or 'cpu' for model training
        """
        
        self.device = torch.device(device)
        self.dir_path = dir_path  # Store the directory path

        self.lr = 1e-3
        
        # Initialize Flow (Neural Spline Flow via Zuko)
        self.flow = zuko.flows.NSF(
            features=input_dim, # Dimensionality of input data
            context=context_dim, # Dimensionality of context variable (Target label)
            bins=bins, # 12 # Number of bins for the splines
            transforms=transforms, # Number of transformations in the flow
            hidden_features=hidden_features # [128, 128] # Hidden layer sizes for the conditioner networks
        ).to(self.device)
        # z -> x || p(x) = p(z) * |det(J)|
        # det(J) = det(J1) * det(J2) * ... * det(Jn) where n = number of transformations (transforms in this case)
        
        self.optimizer = optim.AdamW(self.flow.parameters(), lr=self.lr)
        
        # Scheduler will be initialized in train() with Linear Warmup + Cosine Decay
        self.scheduler = None
        
        self.early_stopper = EarlyStopper(patience=EarlyStopper_patience)
        
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}


    def print_model_summary(self): 
        """
        Prints a summary of the model architecture and number of parameters.
        """
        total_params = sum(p.numel() for p in self.flow.parameters() if p.requires_grad)
        print(self.flow)
        print(f"Total trainable parameters: {total_params}")


    def loss_function(self, batch_X, batch_y):
        """
        Computes the negative log likelihood loss.

        param batch_X: torch.Tensor, input features
        param batch_y: torch.Tensor, context labels
        return: torch.Tensor, computed loss
        """
        # Negative Log Likelihood
        # self.flow(batch_y) creates a distribution conditioned on batch_y
        #   The model define a latent space (Standard Multivariate Normal Distribution, N(0,1))
        #   from which it can sample, and to which it can map inputs.
        #   This is the reason why we can study both 2b and 4b events.
        # log_prob computes log probability of batch_X under that distribution
        loss = -self.flow(batch_y).log_prob(batch_X).mean()
        return loss
        

    def train(self, train_data_loader, val_data_loader, epochs=999):
        """
        Trains the flow to learn P(x | c) using Linear Warmup + Cosine Decay scheduler.

        param train_data_loader: DataLoader, training data loader
        param val_data_loader: DataLoader, validation data loader
        param epochs: int, number of training epochs
        """
        print(f"Starting training on {self.device}...")
        print("Learning rate scheduler: Linear Warmup + Cosine Decay")
        
        # Initialize Linear Warmup + Cosine Decay scheduler
        steps_per_epoch = len(train_data_loader)
        total_steps = steps_per_epoch * 300 # 300 epochs 
        warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
        
        print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        self.scheduler = LinearWarmupCosineDecay(
            self.optimizer, 
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        current_step = 0
        
        for epoch in (bar := tqdm(range(epochs))):
            # --- Training Step ---
            self.flow.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad() # Reset gradients
                # We want to maximize prob, so we minimize -log_prob
                loss = self.loss_function(batch_X, batch_y)
                loss.backward() # Backpropagation
                self.optimizer.step() # Update parameters through optimizer
                
                # Step scheduler per batch
                self.scheduler.step()
                current_step += 1
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_data_loader)
            
            # --- Validation Step ---
            self.flow.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_data_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    loss = self.loss_function(batch_X, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_data_loader)
            
            # Record history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            bar.set_description(f"Epoch {epoch+1}/{epochs}")
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scientific_notation_formatter(self.optimizer.param_groups[0]['lr'])}")
            
            if self.early_stopper.early_stop(avg_val_loss):
                print("Early stopping triggered!")
                break
                
        # Save best model
        dir_path_models = self.dir_path + 'weights/'
        os.makedirs(dir_path_models, exist_ok=True)
        save_path = dir_path_models + 'flow_model.pth'
        torch.save(self.flow.state_dict(), save_path)


    def plot_training_loss(self):
        """
        Plots the training and validation loss over epochs, plus learning rate evolution.
        """
        dir_path_plots = self.dir_path + 'plots/'
        os.makedirs(dir_path_plots, exist_ok=True)

        # Plot 1: Training and Validation Loss
        plt.figure(figsize=(8, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Negative Log Likelihood Loss')
        plt.title('Training and Validation Loss')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(dir_path_plots + 'training_validation_loss.png')
        print(f"  Plot saved: training_validation_loss.png")    
        plt.close()

        # Plot 2: Learning Rate Evolution
        plt.figure(figsize=(8, 6))
        plt.plot(self.history['lr'], label='Learning Rate', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Evolution (Linear Warmup + Cosine Decay)')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(dir_path_plots + 'lr_evolution.png')
        print(f"  Plot saved: lr_evolution.png")    
        plt.close()


    def load_model(self, model_path):
        """
        Loads a pre-trained model from the specified path.

        param model_path: str, path to the saved model file
        """
        self.flow.load_state_dict(torch.load(model_path, map_location=self.device)) # Load model weights
        self.flow.to(self.device) # Ensure model is on the correct device
        print(f"Model loaded from {model_path}")


    def evaluate_loss(self, test_data_loader):
        """
        Evaluates the flow on the test dataset.

        param test_data_loader: DataLoader, test data loader
        return: float, average test loss
        """
        self.flow.eval()
        test_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation -> Not needed without training
            for batch_X, batch_y in test_data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                loss = self.loss_function(batch_X, batch_y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_data_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")
        return avg_test_loss


    def evaluate_log_prob(self, X, condition):
        """
        Evaluate the probability density P(X | y) for given inputs.

        param X: torch.Tensor, input features
        param condition: torch.Tensor, context labels
        return: torch.Tensor, log probability density of X given y
        """
        self.flow.eval()
        X, condition = X.to(self.device), condition.to(self.device)
        with torch.no_grad():
            log_prob = self.flow(condition).log_prob(X)

        return log_prob.cpu()
   

    def sample_generator(self, num_samples, condition):
        """
        Generate samples from the learned distribution P(x | c).

        param num_samples: int, number of samples to generate (should match context batch size)
        param condition: torch.Tensor, context variable tensor of shape (num_samples, context_dim)
        """
        self.flow.eval()
        condition = condition.to(self.device) # Move the context vector to the appropriate CPU or GPU  
        with torch.no_grad(): 
            samples = self.flow(condition).sample() # Sample from the distribution conditioned on the context
        return samples.cpu()
        

    def transform_feature_to_latent(self, X, condition):
        """
        Transforms input data X to the latent space z using the flow.

        param X: torch.Tensor, input features
        param condition: torch.Tensor, context labels
        return: torch.Tensor, transformed latent variables z
        """
        self.flow.eval()
        X, condition = X.to(self.device), condition.to(self.device)
        with torch.no_grad():
            z = self.flow(condition).transform(X) # Transform input data to latent space conditioned on context
        return z.cpu()


    def transform_latent_to_feature(self, z, condition):
        """
        Transforms latent variables z back to the original space x using the inverse flow.

        param z: torch.Tensor, latent variables
        param condition: torch.Tensor, context labels
        return: torch.Tensor, transformed original features x
        """
        self.flow.eval()
        z, condition = z.to(self.device), condition.to(self.device)
        with torch.no_grad():
            x = self.flow(condition).transform.inv(z) # Transform latent variables back to original space conditioned on context
        return x.cpu()


