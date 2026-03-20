import zuko
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm

from model import scientific_notation_formatter, LinearWarmupCosineDecay, EarlyStopper


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
        phi_indices=None,
        phi_bins=None,
        phi_transforms=None,
        phi_hidden_features=None,
        EarlyStopper_patience=30,
        device='cuda' if torch.cuda.is_available() else 'cpu'
        ):
        """
        Initializes the normalizing flow model.
        
        If phi_indices is provided, uses a factorized architecture:
            P(features, phi | c) = P(features | phi, c) * P(phi | c)
        where phi features are modeled with a periodic NCSF flow and 
        the remaining features with a standard NSF conditioned on phi + c.

        param input_dim: int, dimensionality of input features (all features including phi)
        param context_dim: int, dimensionality of context variable (Target label)   
        param bins: int, number of bins for the splines in the main flow
        param transforms: int, number of transformations in the main flow
        param hidden_features: list of int, hidden layer sizes for the main flow conditioner networks
        param dir_path: str, directory path for saving models and plots
        param phi_indices: list of int or None, indices of phi features in the input. If None, uses a single NSF for all features.
        param phi_bins: int or None, number of bins for phi NCSF (defaults to bins if None)
        param phi_transforms: int or None, number of transforms for phi NCSF (defaults to transforms if None)
        param phi_hidden_features: list of int or None, hidden layers for phi NCSF (defaults to [64, 64] if None)
        param EarlyStopper_patience: int, patience for early stopping
        param device: str, 'cuda' or 'cpu' for model training
        """
        
        self.device = torch.device(device)
        self.dir_path = dir_path
        self.lr = 1e-3

        # Phi feature separation
        self.phi_indices = phi_indices if phi_indices is not None else []
        self.has_phi = len(self.phi_indices) > 0

        if self.has_phi:
            # Compute non-phi indices
            self.non_phi_indices = [i for i in range(input_dim) if i not in self.phi_indices]
            num_phi = len(self.phi_indices)
            num_non_phi = len(self.non_phi_indices)

            # P(phi | c): Periodic flow for phi features using Neural Circular Spline Flow
            self.phi_flow = zuko.flows.NCSF(
                features=num_phi,
                context=context_dim,
                bins=phi_bins if phi_bins is not None else bins,
                transforms=phi_transforms if phi_transforms is not None else transforms,
                hidden_features=phi_hidden_features if phi_hidden_features is not None else hidden_features
            ).to(self.device)

            # P(features | phi, c): Standard flow conditioned on both phi and context
            self.flow = zuko.flows.NSF(
                features=num_non_phi,
                context=context_dim + num_phi,  # context = c + phi
                bins=bins,
                transforms=transforms,
                hidden_features=hidden_features
            ).to(self.device)

            # Single optimizer for both flows
            all_params = list(self.flow.parameters()) + list(self.phi_flow.parameters())
            self.optimizer = optim.AdamW(all_params, lr=self.lr)

            print(f"  Factorized flow: P(features | phi, c) * P(phi | c)")
            print(f"  Phi features ({num_phi}): indices {self.phi_indices}")
            print(f"  Non-phi features ({num_non_phi}): indices {self.non_phi_indices}")
        else:
            # Standard single flow: P(x | c)
            self.flow = zuko.flows.NSF(
                features=input_dim,
                context=context_dim,
                bins=bins,
                transforms=transforms,
                hidden_features=hidden_features
            ).to(self.device)
            self.phi_flow = None
            self.optimizer = optim.AdamW(self.flow.parameters(), lr=self.lr)
        
        # Scheduler will be initialized in train() with Linear Warmup + Cosine Decay
        self.scheduler = None
        
        self.early_stopper = EarlyStopper(patience=EarlyStopper_patience)
        
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}


    def _split_phi(self, X):
        """
        Split input features into phi and non-phi components.

        param X: torch.Tensor, all input features (batch_size, input_dim)
        return: tuple (x_non_phi, x_phi)
        """
        x_phi = X[:, self.phi_indices]
        x_non_phi = X[:, self.non_phi_indices]
        return x_non_phi, x_phi

    def _reassemble(self, x_non_phi, x_phi):
        """
        Reassemble non-phi and phi features back into the original feature order.

        param x_non_phi: torch.Tensor, non-phi features
        param x_phi: torch.Tensor, phi features
        return: torch.Tensor, reassembled features in original order
        """
        batch_size = x_non_phi.shape[0]
        total_dim = x_non_phi.shape[1] + x_phi.shape[1]
        X = torch.zeros(batch_size, total_dim, device=x_non_phi.device)
        X[:, self.non_phi_indices] = x_non_phi
        X[:, self.phi_indices] = x_phi
        return X

    def print_model_summary(self): 
        """
        Prints a summary of the model architecture and number of parameters.
        """
        if self.has_phi:
            print("\n--- Phi Flow (NCSF) ---")
            print(self.phi_flow)
            phi_params = sum(p.numel() for p in self.phi_flow.parameters() if p.requires_grad)
            print(f"Phi flow trainable parameters: {phi_params}")
            
            print("\n--- Features Flow (NSF) ---")
            print(self.flow)
            flow_params = sum(p.numel() for p in self.flow.parameters() if p.requires_grad)
            print(f"Features flow trainable parameters: {flow_params}")
            
            print(f"\nTotal trainable parameters: {phi_params + flow_params}")
        else:
            total_params = sum(p.numel() for p in self.flow.parameters() if p.requires_grad)
            print(self.flow)
            print(f"Total trainable parameters: {total_params}")


    def loss_function(self, batch_X, batch_y):
        """
        Computes the negative log likelihood loss.
        
        If phi features are separated:
            loss = -[log P(features | phi, c) + log P(phi | c)].mean()
        Otherwise:
            loss = -log P(x | c).mean()

        param batch_X: torch.Tensor, input features (all features including phi)
        param batch_y: torch.Tensor, context labels
        return: torch.Tensor, computed loss
        """
        if batch_y.dim() == 1:
            batch_y = batch_y.unsqueeze(-1)

        if self.has_phi:
            x_non_phi, x_phi = self._split_phi(batch_X)
            
            # log P(phi | c)
            log_prob_phi = self.phi_flow(batch_y).log_prob(x_phi)
            
            # log P(features | phi, c) - context is [c, phi]
            context_for_features = torch.cat([batch_y, x_phi], dim=1) # dim=1 concatenate along feature dimension
            log_prob_features = self.flow(context_for_features).log_prob(x_non_phi)
            
            # Joint: log P(features, phi | c) = log P(features | phi, c) + log P(phi | c)
            loss = -(log_prob_features + log_prob_phi).mean()
        else:
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
            if self.has_phi:
                self.phi_flow.train()
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
            if self.has_phi:
                self.phi_flow.eval()
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
        torch.save(self.flow.state_dict(), dir_path_models + 'flow_model.pth')
        if self.has_phi:
            torch.save(self.phi_flow.state_dict(), dir_path_models + 'phi_flow_model.pth')


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

        param model_path: str, path to the saved main flow model file.
            If phi flow exists, expects phi_flow_model.pth in the same directory.
        """
        self.flow.load_state_dict(torch.load(model_path, map_location=self.device))
        self.flow.to(self.device)
        print(f"Features flow loaded from {model_path}")
        
        if self.has_phi:
            phi_path = os.path.join(os.path.dirname(model_path), 'phi_flow_model.pth')
            self.phi_flow.load_state_dict(torch.load(phi_path, map_location=self.device))
            self.phi_flow.to(self.device)
            print(f"Phi flow loaded from {phi_path}")


    def evaluate_loss(self, test_data_loader):
        """
        Evaluates the flow on the test dataset.

        param test_data_loader: DataLoader, test data loader
        return: float, average test loss
        """
        self.flow.eval()
        if self.has_phi:
            self.phi_flow.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                loss = self.loss_function(batch_X, batch_y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_data_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")
        return avg_test_loss


    def evaluate_log_prob(self, X, condition):
        """
        Evaluate the joint log probability density P(X | c) for given inputs.

        param X: torch.Tensor, input features (all features including phi)
        param condition: torch.Tensor, context labels
        return: torch.Tensor, log probability density of X given condition
        """
        self.flow.eval()
        X, condition = X.to(self.device), condition.to(self.device)
        
        if condition.dim() == 1:
            condition = condition.unsqueeze(-1)
        
        with torch.no_grad():
            if self.has_phi:
                self.phi_flow.eval()
                x_non_phi, x_phi = self._split_phi(X)
                
                # log P(phi | c)
                log_prob_phi = self.phi_flow(condition).log_prob(x_phi)
                
                # log P(features | phi, c)
                context_for_features = torch.cat([condition, x_phi], dim=1)
                log_prob_features = self.flow(context_for_features).log_prob(x_non_phi)
                
                log_prob = log_prob_features + log_prob_phi
            else:
                log_prob = self.flow(condition).log_prob(X)

        return log_prob.cpu()
   

    def sample_generator(self, num_samples, condition):
        """
        Generate samples from the learned distribution P(x | c).
        
        If factorized: first sample phi ~ P(phi | c), then sample features ~ P(features | phi, c),
        and reassemble into the original feature order.

        param num_samples: int, number of samples to generate (should match context batch size)
        param condition: torch.Tensor, context variable tensor of shape (num_samples, context_dim)
        return: torch.Tensor, generated samples in original feature order
        """
        self.flow.eval()
        condition = condition.to(self.device)
        
        if condition.dim() == 1:
            condition = condition.unsqueeze(-1)
        
        with torch.no_grad():
            if self.has_phi:
                self.phi_flow.eval()
                
                # Step 1: Sample phi ~ P(phi | c)
                x_phi = self.phi_flow(condition).sample()
                
                # Step 2: Sample features ~ P(features | phi, c)
                context_for_features = torch.cat([condition, x_phi], dim=1)
                x_non_phi = self.flow(context_for_features).sample()
                
                # Step 3: Reassemble into original order
                samples = self._reassemble(x_non_phi, x_phi)
            else:
                samples = self.flow(condition).sample()
        
        return samples.cpu()
        

    def transform_feature_to_latent(self, X, condition):
        """
        Transforms input data X to the latent space z using the flow.
        
        If factorized: transforms phi and non-phi separately, then reassembles.

        param X: torch.Tensor, input features (all features including phi)
        param condition: torch.Tensor, context labels
        return: torch.Tensor, transformed latent variables z (in original feature order)
        """
        self.flow.eval()
        X, condition = X.to(self.device), condition.to(self.device)
        
        if condition.dim() == 1:
            condition = condition.unsqueeze(-1)
        
        with torch.no_grad():
            if self.has_phi:
                self.phi_flow.eval()
                x_non_phi, x_phi = self._split_phi(X)
                
                # Transform phi to latent
                z_phi = self.phi_flow(condition).transform(x_phi)
                
                # Transform features to latent (conditioned on phi + c)
                context_for_features = torch.cat([condition, x_phi], dim=1)
                z_non_phi = self.flow(context_for_features).transform(x_non_phi)
                
                z = self._reassemble(z_non_phi, z_phi)
            else:
                z = self.flow(condition).transform(X)
        
        return z.cpu()


    def transform_latent_to_feature(self, z, condition):
        """
        Transforms latent variables z back to the original space x using the inverse flow.
        
        If factorized: first inverts phi, then uses phi as context to invert features.

        param z: torch.Tensor, latent variables (in original feature order)
        param condition: torch.Tensor, context labels
        return: torch.Tensor, transformed original features x (in original feature order)
        """
        self.flow.eval()
        z, condition = z.to(self.device), condition.to(self.device)
        
        if condition.dim() == 1:
            condition = condition.unsqueeze(-1)
        
        with torch.no_grad():
            if self.has_phi:
                self.phi_flow.eval()
                z_non_phi, z_phi = self._split_phi(z)
                
                # Invert phi latent -> phi features
                x_phi = self.phi_flow(condition).transform.inv(z_phi)
                
                # Invert features latent -> features (conditioned on recovered phi + c)
                context_for_features = torch.cat([condition, x_phi], dim=1)
                x_non_phi = self.flow(context_for_features).transform.inv(z_non_phi)
                
                x = self._reassemble(x_non_phi, x_phi)
            else:
                x = self.flow(condition).transform.inv(z)
        
        return x.cpu()


