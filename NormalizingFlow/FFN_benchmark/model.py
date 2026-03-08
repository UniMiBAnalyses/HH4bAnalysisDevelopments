import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class FFNEmbeddingNetwork(nn.Module):
    """
    Neural network with embedding layers for categorical features.

    param num_continuous: int, number of continuous features
    param num_categorical: int, number of categorical features
    param categorical_dims: list, number of unique values for each categorical feature
    param embedding_dim: int, embedding dimension for each categorical feature
    param total_input_dim: int, total input dimension after embedding
    """

    def __init__(
        self, 
        num_continuous, 
        num_categorical, 
        categorical_dims, 
        embedding_dim, 
        total_input_dim,
        embedding_noise_std=0.1
        ):
        super().__init__()

        self.num_continuous = num_continuous
        self.num_categorical = num_categorical
        self.embedding_noise_std = embedding_noise_std

        # Embedding layers for categorical features
        if num_categorical > 0:
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_classes, embedding_dim)
                for num_classes in categorical_dims
            ])
            self.embedding_batch_norm = nn.BatchNorm1d(num_categorical * embedding_dim)
        else:
            self.embeddings = None

        # Main FFN architecture
        self.input_batch_norm = nn.BatchNorm1d(total_input_dim)

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(total_input_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.20),
            # Layer 2
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.20),
            # Layer 3
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),
            # Layer 4
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.20),
            # Output layer
            nn.Linear(64, 2),
        )

        print(f"  Total input dimension after embedding: {total_input_dim} "
              f"(Continuous: {num_continuous}, Categorical: {num_categorical}"
              + (f" with embedding dim {embedding_dim})" if num_categorical > 0 else ")"))

    def forward(self, x_cont, x_cat=None):
        if self.num_categorical > 0 and x_cat is not None:
            embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat_emb = torch.cat(embedded, dim=1)
            
            # Add Gaussian noise during training for regularization
            if self.training and self.embedding_noise_std > 0:
                noise = torch.randn_like(x_cat_emb) * self.embedding_noise_std
                x_cat_emb = x_cat_emb + noise
            
            x_cat_emb = self.embedding_batch_norm(x_cat_emb)
            x = torch.cat([x_cont, x_cat_emb], dim=1)
        else:
            x = x_cont

        x = self.input_batch_norm(x)
        x = self.network(x)
        return x


class FFN:
    def __init__(
        self,
        input_dim,
        dir_path,
        categorical_features=[],
        categorical_dims=[],
        embedding_dim=None,
        embedding_noise_std=0.1,
        EarlyStopper_patience=15,
        device='cuda' if torch.cuda.is_available() else 'cpu'
        ):
        """
        FFN with categorical feature embeddings using PyTorch.

        param input_dim: int, total number of input features (continuous + categorical)
        param dir_path: str, directory path for saving models and plots
        param categorical_features: list, indices of categorical features
        param categorical_dims: list, number of unique values for each categorical feature
        param embedding_dim: int, embedding dimension for each categorical feature
        param embedding_noise_std: float, standard deviation of Gaussian noise added to embeddings during training
        param EarlyStopper_patience: int, patience for early stopping
        param device: str, device preference for training ('cuda' or 'cpu')
        """

        self.input_dim = input_dim
        self.dir_path = dir_path
        self.categorical_features = categorical_features
        self.categorical_dims = categorical_dims
        self.embedding_dim = embedding_dim
        self.embedding_noise_std = embedding_noise_std
        self.device = device

        self.num_categorical = len(categorical_features)
        self.num_continuous = input_dim - self.num_categorical

        # Calculate total input dimension
        if self.num_categorical > 0:
            if self.embedding_dim is None:
                raise ValueError("embedding_dim must be specified when using categorical features")
            self.total_input_dim = self.num_continuous + (self.num_categorical * self.embedding_dim)
        else:
            self.total_input_dim = self.num_continuous

        self.lr = 1e-3
        self.EarlyStopper_patience = EarlyStopper_patience

        # Create network with embeddings
        self.model = FFNEmbeddingNetwork(
            num_continuous=self.num_continuous,
            num_categorical=self.num_categorical,
            categorical_dims=self.categorical_dims,
            embedding_dim=self.embedding_dim,
            total_input_dim=self.total_input_dim,
            embedding_noise_std=self.embedding_noise_std
        ).to(self.device)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # Adaptive exponential LR scheduler
        # gamma decreases by 0.05 if no val_accuracy improvement for gamma_patience epochs
        self.gamma = 0.95
        self.gamma_min = 0.5
        self.gamma_patience = 10
        self.gamma_counter = 0
        self.best_val_acc_for_gamma = 0.0
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)

        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [], 'lr': []}


    def _split_features(self, X):
        """
        Split input features into continuous and categorical.

        param X: np.ndarray or torch.Tensor of all features (batch_size, input_dim)
        return: tuple (x_cont, x_cat)
        """
        if isinstance(X, torch.Tensor):
            all_indices = list(range(self.input_dim))
            cat_indices = self.categorical_features
            cont_indices = [i for i in all_indices if i not in cat_indices]

            x_cont = X[:, cont_indices].float()
            x_cat = X[:, cat_indices].long() if len(cat_indices) > 0 else None
        else:
            all_indices = np.arange(self.input_dim)
            cat_indices = np.array(self.categorical_features)
            cont_mask = ~np.isin(all_indices, cat_indices)
            cont_indices = all_indices[cont_mask]

            x_cont = X[:, cont_indices].astype(np.float32)
            x_cat = X[:, cat_indices].astype(np.int64) if len(cat_indices) > 0 else None

        return x_cont, x_cat


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
        print(f"Categorical features: {self.num_categorical}")
        if self.num_categorical > 0:
            print(f"Embedding dimension: {self.embedding_dim}")
            print(f"Categorical dimensions: {self.categorical_dims}")
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

                x_cont, x_cat = self._split_features(batch_X)
                x_cont = x_cont.to(self.device)
                if x_cat is not None:
                    x_cat = x_cat.to(self.device)

                self.optimizer.zero_grad()
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

                    x_cont, x_cat = self._split_features(batch_X)
                    x_cont = x_cont.to(self.device)
                    if x_cat is not None:
                        x_cat = x_cat.to(self.device)

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
        save_path = dir_path_models + 'flow_model.pt'
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

        param X: np.ndarray, input features (batch_size, input_dim)
        return: np.ndarray, predicted class probabilities (batch_size, 2)
        """
        self.model.eval()

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        x_cont, x_cat = self._split_features(X)
        x_cont = torch.tensor(x_cont, dtype=torch.float32).to(self.device)
        if x_cat is not None:
            x_cat = torch.tensor(x_cat, dtype=torch.long).to(self.device)

        with torch.no_grad():
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
