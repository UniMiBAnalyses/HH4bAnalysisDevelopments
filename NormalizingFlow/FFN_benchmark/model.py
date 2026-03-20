import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class FFN_model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.20),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.20),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


class CategoricalEmbeddingNetwork(nn.Module):
    """
    Neural network with early-fusion input processing for semicategorical features.
    It structurally matches a standard FFN when semicategorical features are absent.
    """
    def __init__(
        self,
        num_continuous,
        num_categorical,
        categorical_dims,
        embedding_dim,
        pt_index=None,
        jet_feature_indices=None,
        embedding_noise_std=0.1
    ):
        super().__init__()
        
        self.num_continuous = num_continuous
        self.num_categorical = num_categorical
        self.pt_index = pt_index
        self.embedding_noise_std = embedding_noise_std
        
        # Determine if we are actively using the semicategorical branch
        self.use_semi_cat = pt_index is not None and jet_feature_indices is not None

        if self.use_semi_cat:
            self.register_buffer('jet_feature_indices', torch.tensor(jet_feature_indices, dtype=torch.long))
            self.missing_jet_embedding = nn.Embedding(2, embedding_dim)
            discrete_emb_dim = (num_categorical + 1) * embedding_dim # +1 for missing jet indicator
        else:
            discrete_emb_dim = num_categorical * embedding_dim if num_categorical > 0 else 0

        # Standard Categorical Embeddings
        if num_categorical > 0:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(num_classes, embedding_dim)
                for num_classes in categorical_dims
            ])
        else:
            self.categorical_embeddings = None

        # Batch Norm specifically for discrete features
        if discrete_emb_dim > 0:
            self.embedding_batch_norm = nn.BatchNorm1d(discrete_emb_dim)
        
        # Calculate total dimension for early fusion
        total_input_dim = num_continuous + discrete_emb_dim

        # Global Batch Norm for the concatenated inputs
        self.input_batch_norm = nn.BatchNorm1d(total_input_dim)

        # Main FFN directly takes the concatenated features (no continuous bottleneck)
        self.network = FFN_model(total_input_dim)


    def forward(self, x_cont, x_semi_cat, x_cat=None):
        cat_embeddings = []

        # Handle standard categorical features
        if self.num_categorical > 0 and x_cat is not None:
            cat_embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.categorical_embeddings)]
            
        # Handle semi-categorical features (The Mask & The Missing Jet Embedding)
        if self.use_semi_cat and x_semi_cat is not None:
            # Note: We create a boolean mask for torch.where, and a long mask for nn.Embedding
            is_missing_jet_bool = (x_semi_cat[:, self.pt_index] == -10.0) 
            is_missing_jet_idx = is_missing_jet_bool.long()
            
            # Get embedding for the "missing jet" state
            missing_jet_emb = self.missing_jet_embedding(is_missing_jet_idx)
            cat_embeddings.append(missing_jet_emb)

            # Clean semi-categorical features: replace -10.0 with 0.0
            x_semi_cat_cleaned = x_semi_cat.clone()
            jet_mask = torch.zeros(x_semi_cat.size(1), dtype=torch.bool, device=x_semi_cat.device)
            jet_mask[self.jet_feature_indices] = True
            
            mask_expanded = is_missing_jet_bool.unsqueeze(1).expand_as(x_semi_cat_cleaned[:, jet_mask])
            x_semi_cat_cleaned[:, jet_mask] = torch.where(
                mask_expanded,
                torch.zeros_like(x_semi_cat_cleaned[:, jet_mask]),
                x_semi_cat_cleaned[:, jet_mask]
            )
        else:
            x_semi_cat_cleaned = None

        # 3. Process Discrete Embeddings
        if len(cat_embeddings) > 0:
            discrete_output = torch.cat(cat_embeddings, dim=1)
            # Add Gaussian noise during training
            if self.training and self.embedding_noise_std > 0:
                noise = torch.randn_like(discrete_output) * self.embedding_noise_std
                discrete_output = discrete_output + noise
            discrete_output = self.embedding_batch_norm(discrete_output)
        else:
            discrete_output = None

        # 4. Gather Continuous Features
        continuous_tensors = []
        if x_cont is not None:
            continuous_tensors.append(x_cont)
        if x_semi_cat_cleaned is not None:
            continuous_tensors.append(x_semi_cat_cleaned)
            
        if len(continuous_tensors) > 0:
            continuous_output = torch.cat(continuous_tensors, dim=1)
        else:
            continuous_output = None

        # 5. Early Fusion & Pass to Network
        final_inputs = []
        if continuous_output is not None:
            final_inputs.append(continuous_output)
        if discrete_output is not None:
            final_inputs.append(discrete_output)

        x = torch.cat(final_inputs, dim=1)
        
        # Global input normalization
        x = self.input_batch_norm(x)
        
        return self.network(x)


class FFN:
    def __init__(
        self,
        input_dim,
        dir_path,
        categorical_features=[],
        categorical_dims=[],
        semicategorical_features=[],
        pt_index=None,
        jet_feature_indices=None,
        embedding_dim=None,
        embedding_noise_std=0.1,
        EarlyStopper_patience=15,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.input_dim = input_dim
        self.dir_path = dir_path
        self.categorical_features = categorical_features
        self.categorical_dims = categorical_dims
        self.semicategorical_features = semicategorical_features
        self.embedding_dim = embedding_dim
        self.embedding_noise_std = embedding_noise_std
        self.device = device
        
        self.pt_index = pt_index
        self.jet_feature_indices = jet_feature_indices

        self.num_categorical = len(categorical_features)
        self.num_semicategorical = len(semicategorical_features)
        self.num_continuous = input_dim - self.num_categorical - self.num_semicategorical

        # Validation checks
        if self.num_semicategorical > 0:
            if pt_index is None or jet_feature_indices is None:
                raise ValueError("pt_index and jet_feature_indices must be specified for semicategorical features")
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified when using semicategorical features")

        # Initialize Model
        self.model = CategoricalEmbeddingNetwork(
            num_continuous=self.num_continuous + self.num_semicategorical,
            num_categorical=self.num_categorical,
            categorical_dims=self.categorical_dims,
            embedding_dim=self.embedding_dim,
            pt_index=self.pt_index,
            jet_feature_indices=self.jet_feature_indices,
            embedding_noise_std=self.embedding_noise_std
        ).to(self.device)

        self.lr = 1e-3
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr) 
        self.criterion = nn.CrossEntropyLoss()
        self.EarlyStopper_patience = EarlyStopper_patience
        
        self.gamma = 0.95
        self.gamma_min = 0.5
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'lr': []}
        self.best_val_acc_for_gamma = 0.0
        self.gamma_patience = 5
        self.gamma_counter = 0

    def _split_features(self, X):
        """
        Cleanly split input features using set operations.
        Works efficiently for both NumPy arrays and PyTorch tensors.
        """
        all_indices = set(range(self.input_dim))
        cat_set = set(self.categorical_features)
        semi_set = set(self.semicategorical_features)
        
        # The remaining indices belong to pure continuous features
        cont_indices = sorted(list(all_indices - cat_set - semi_set))

        # Convert to tensor/array subsets
        if isinstance(X, torch.Tensor):
            x_cont = X[:, cont_indices].float() if len(cont_indices) > 0 else None
            x_semi_cat = X[:, self.semicategorical_features].float() if self.num_semicategorical > 0 else None
            x_cat = X[:, self.categorical_features].long() if self.num_categorical > 0 else None
        else:
            x_cont = torch.tensor(X[:, cont_indices], dtype=torch.float32) if len(cont_indices) > 0 else None
            x_semi_cat = torch.tensor(X[:, self.semicategorical_features], dtype=torch.float32) if self.num_semicategorical > 0 else None
            x_cat = torch.tensor(X[:, self.categorical_features], dtype=torch.long) if self.num_categorical > 0 else None

        # Move to correct device
        if x_cont is not None: x_cont = x_cont.to(self.device)
        if x_semi_cat is not None: x_semi_cat = x_semi_cat.to(self.device)
        if x_cat is not None: x_cat = x_cat.to(self.device)

        return x_cont, x_semi_cat, x_cat
    

    def print_model_summary(self):
        """
        Prints a summary of the model architecture and number of parameters.
        """
        print("\n" + "=" * 60)
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Continuous features: {self.num_continuous}")
        if self.num_semicategorical > 0:
            print(f"Semi-categorical features: {self.num_semicategorical}")
        if self.num_categorical > 0:
            print(f"Categorical features: {self.num_categorical}")
            print(f"Embedding dimension: {self.embedding_dim}")
            print(f"Categorical dimensions: {self.categorical_dims}")
        if self.num_categorical > 0 or self.num_semicategorical > 0:
            print(f"Embedding noise std: {self.embedding_noise_std}")
        print()

        return 0


    def train(self, train_data_loader, val_data_loader, epochs=999, batch_size=512):
        """
        Trains the FFN model.

        param train_data_loader: DataLoader, training data loader
        param val_data_loader: DataLoader, validation data loader
        param epochs: int, number of training epochs
        param batch_size: int, batch size (unused, batch size comes from the DataLoader)
        """
        print(f"Starting training...")

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # --- Training ---
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in tqdm(train_data_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).long().squeeze(-1)

                x_cont, x_semi_cat, x_cat = self._split_features(batch_X)
                if x_cont is not None: x_cont = x_cont.to(self.device)
                if x_semi_cat is not None: x_semi_cat = x_semi_cat.to(self.device)
                if x_cat is not None: x_cat = x_cat.to(self.device)

                self.optimizer.zero_grad()
                if self.num_semicategorical > 0:
                    logits = self.model(x_cont, x_semi_cat, x_cat)
                else:
                    logits = self.model(x_cont, x_cat)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * batch_X.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            # --- Validation ---
            self.model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_X, batch_y in tqdm(val_data_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device).long().squeeze(-1)

                    x_cont, x_semi_cat, x_cat = self._split_features(batch_X)
                    if x_cont is not None: x_cont = x_cont.to(self.device)
                    if x_semi_cat is not None: x_semi_cat = x_semi_cat.to(self.device)
                    if x_cat is not None: x_cat = x_cat.to(self.device)

                    if self.num_semicategorical > 0:
                        logits = self.model(x_cont, x_semi_cat, x_cat)
                    else:
                        logits = self.model(x_cont, x_cat)
                    loss = self.criterion(logits, batch_y)

                    val_running_loss += loss.item() * batch_X.size(0)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)

            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total

            # Step the LR scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            # Adaptive gamma: decrease gamma if no val_accuracy improvement
            if val_acc > self.best_val_acc_for_gamma:
                self.best_val_acc_for_gamma = val_acc
                self.gamma_counter = 0
            else:
                self.gamma_counter += 1
                if self.gamma_counter >= self.gamma_patience:
                    self.gamma = max(self.gamma - 0.05, self.gamma_min)
                    self.gamma_counter = 0
                    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)
                    print(f"  Gamma reduced to {self.gamma:.2f}")

            # Record history
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            self.history['lr'].append(current_lr)

            print(f"Epoch {epoch}/{epochs} - "
                  f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - "
                  f"lr: {current_lr:.2e}")

            # --- Early Stopping ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.EarlyStopper_patience:
                    print(f"Early stopping triggered at epoch {epoch} (patience {self.EarlyStopper_patience})")
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        # Save best model
        dir_path_models = self.dir_path + 'weights/'
        os.makedirs(dir_path_models, exist_ok=True)
        save_path = dir_path_models + 'flow_model.pth'
        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")

    
    def plot_training_loss(self):
        """
        Plots the training and validation loss and accuracy over epochs in two subplots,
        plus learning rate evolution.
        """
        if not self.dir_path or not self.history['loss']:
            print("No directory path specified or no training history available, skipping plot save.")
            return

        dir_path_plots = self.dir_path + 'plots/'
        os.makedirs(dir_path_plots, exist_ok=True)

        # Two subplots: loss on left, accuracy on right
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left subplot: Loss
        ax1.plot(self.history['loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # Right subplot: Accuracy
        ax2.plot(self.history['accuracy'], label='Train Accuracy')
        ax2.plot(self.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.grid(alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(dir_path_plots + 'training_validation_metrics.png')
        print(f"  Plot saved: training_validation_metrics.png")
        plt.close()

        # Plot learning rate
        if self.history['lr']:
            plt.figure(figsize=(8, 6))
            plt.plot(self.history['lr'], label='Learning Rate', color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Evolution (Adaptive ExponentialLR)')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(dir_path_plots + 'lr_evolution.png')
            print(f"  Plot saved: lr_evolution.png")
            plt.close()


    def load_model(self, model_path):
        """
        Loads a pre-trained model from the specified path.

        param model_path: str, path to the saved model file (.pt)
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        print(f"Model weights loaded from {model_path}")


    def predict_proba(self, X):
        """
        Predict class probabilities for input data.

        param X: np.ndarray or torch.Tensor, input features (batch_size, input_dim)
        return: np.ndarray, predicted class probabilities (batch_size, 2)
        """
        self.model.eval()

        x_cont, x_semi_cat, x_cat = self._split_features(X)

        with torch.no_grad():
            if self.num_semicategorical > 0:
                logits = self.model(x_cont, x_semi_cat, x_cat)
            else:
                logits = self.model(x_cont, x_cat)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()


    def predict(self, X):
        """
        Predict class labels for input data.

        param X: np.ndarray, input features (batch_size, input_dim)
        return: np.ndarray, predicted class labels (batch_size,)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)