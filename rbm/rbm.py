import os
import json
import torch
import torch.nn as nn
from .util import build_dataloader

class RBMBackend(nn.Module):
    """
    Restricted Boltzmann Machine. 

    - Robust to missing data
    - Suitable for both continuous and binary data
    """
    
    def __init__(self, n_continuous: int, n_binary: int,
                 hidden_layer_size: int, initial_seed: int = 42):
        """
        Initializes an RBM suitable for learning both continuous and 
        binary data. Utilizes binary latent variables. 

        @args
        - n_continuous: int << number of continuous variables
        - n_binary: int << number of binary variables
        - hidden_layer_size: int << size of the latent space
        - initial_seed: int << initial random seed value
        """
        super().__init__()
        self.rng = torch.Generator()
        self.seed = initial_seed
        self.reset_seed(initial_seed)
        self.reset_hyperparameters(n_continuous, n_binary, hidden_layer_size)

    def reset_hyperparameters(self, n_continuous: int = None, 
                        n_binary: int = None, hidden_layer_size: int = None):
        """
        Resets hyperparameters, then randomly reinitializes the weights. 
        Model must be retrained after calling this function. 

        @args
        - n_continuous: int << number of continuous variables
        - n_binary: int << number of binary variables
        - hidden_layer_size: int << size of the latent space
        """
        if n_continuous is not None:
            self.n_vis_continuous = n_continuous
        if n_binary is not None:
            self.n_vis_binary = n_binary
        if hidden_layer_size is not None:
            self.n_hid = hidden_layer_size
        # Gaussian-Bernoulli RBM parameters
        if self.n_vis_continuous > 0:
            self.W_continuous = nn.Parameter(
                torch.Tensor(self.n_vis_continuous, self.n_hid))
            self.log_var = nn.Parameter(torch.Tensor(self.n_vis_continuous))
            self.mu = nn.Parameter(torch.Tensor(self.n_vis_continuous))
        # Bernoulli-Bernoulli RBM parameters
        if self.n_vis_binary > 0:
            self.W_binary = nn.Parameter(
                torch.Tensor(self.n_vis_binary, self.n_hid))
            self.a = nn.Parameter(torch.Tensor(self.n_vis_binary))
        # Shared parameters
        self.b = nn.Parameter(torch.Tensor(self.n_hid))
        self._reset_parameters()

    def reset_seed(self, seed: int):
        """
        Resets the interal random seed to ensure reproducibility.

        @args
        - seed: int
        """
        self.seed = seed
        self.rng.manual_seed(seed)

    def fit(self, X_continuous: torch.Tensor = None, 
            X_binary: torch.Tensor = None, batch_size: int = 1, 
            epochs: int = 10, lr: float = 0.01, n_gibbs: int = 1,
            rng_seed: int = 42, verbose_interval: int | None = 1):
        """
        Trains the RBM. Robust to missing data.

        X_continuous may be None if and only if n_continuous = 0.
        X_binary may be None if and only if n_binary = 0.

        @args
        - X_continuous: torch.Tensor ~ (n_examples, n_vis_continuous)
        - X_binary: torch.Tensor ~ (n_examples, n_vis_binary)
        - batch_size: int
        - epochs: int
        - lr: float
        - n_gibbs: int
        - rng_seed: int
        - verbose_interval: int | None << if None, nothing will be printed
        """
        self.reset_seed(rng_seed)
        self._reset_parameters()
        self.train()
        contains_missing = torch.any(torch.isnan(X_continuous)).item() and \
            torch.any(torch.isnan(X_binary)).item()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        dataloader = build_dataloader(X_continuous, X_binary, batch_size)
        for epoch in range(1, epochs + 1):
            epoch_averaged_loss = 0
            epoch_averaged_recon = 0
            for X_continuous_batch, X_binary_batch in dataloader:
                if contains_missing:
                    X_continuous_batch, X_binary_batch = self.impute_missing(
                        X_continuous=X_continuous_batch,
                        X_binary=X_binary_batch,
                        n_gibbs=n_gibbs)
                optimizer.zero_grad()
                loss = self._loss(X_continuous_batch, X_binary_batch, n_gibbs)
                loss.backward()
                optimizer.step()
                epoch_averaged_recon += self._recon(X_continuous_batch, 
                    X_binary_batch, n_gibbs).item()
                epoch_averaged_loss += loss.item()
            epoch_averaged_loss /= len(dataloader)
            epoch_averaged_recon /= len(dataloader)
            if verbose_interval is not None:
                if epoch % verbose_interval == 0:
                    msg = f"\rEpoch {str(epoch).zfill(len(str(epochs)))} "
                    msg += f"of {epochs} "
                    msg += f"| CD-loss: {round(epoch_averaged_loss, 3)} "
                    msg += f"| Reconstruction: {round(epoch_averaged_recon, 3)}"

    @torch.no_grad()
    def transform(self, X_continuous: torch.Tensor = None, 
                  X_binary: torch.Tensor = None, 
                  X_continuous_clamp: torch.Tensor = None, 
                  X_binary_clamp: torch.Tensor = None,
                  n_gibbs: int = 10, binary_proba: bool = False):
        """
        Returns fantasy particles sampled from initial X_continuous and 
        X_binary. If clamps exist (Boolean masks), then elements corresponding 
        to True in the mask will be held at their initial values. Robust to 
        missing values in inputs.

        If X_continuous is None, the first item returned will be None. 
        X_continuous may be None if and only if n_continuous = 0.
        If X_binary is None, the second item returned will be None. 
        X_binary may be None if and only if n_binary = 0.

        @args
        - X_continuous: torch.Tensor ~ (n_examples, n_vis_continuous)
        - X_binary: torch.Tensor ~ (n_examples, n_vis_binary)
        - X_continuous_clamp: torch.Tensor ~ (n_examples, n_vis_continuous)
        - X_binary_clamp: torch.Tensor ~ (n_examples, n_vis_binary)
        - n_gibbs: int
        - binary_proba: bool << if True, returns probabilities

        @returns
        - X_continuous
        - X_binary
        """
        X_continuous, X_binary = self.impute_missing(X_continuous, X_binary, 
            n_gibbs)
        latent, X_continuous, X_binary = self._block_gibbs_sample(
            v_continuous=X_continuous,
            v_binary=X_binary,
            v_continuous_clamp=X_continuous_clamp,
            v_binary_clamp=X_binary_clamp,
            n_gibbs=n_gibbs
        )
        if binary_proba and self.n_vis_binary > 0:
            X_binary = latent @ self.W_binary.t() + self.a
        return X_continuous, X_binary
    
    @torch.no_grad()
    def impute_missing(self, X_continuous: torch.Tensor = None, 
                       X_binary: torch.Tensor = None, n_gibbs=10):
        """
        Detects and fills in NaNs (i.e., missing values) by conditionally 
        sampling from the model distribution. 

        If X_continuous is None, the first item returned will be None. 
        X_continuous may be None if and only if n_continuous = 0.
        If X_binary is None, the second item returned will be None. 
        X_binary may be None if and only if n_binary = 0.

        @args
        - X_continuous: torch.Tensor ~ (n_examples, n_vis_continuous)
        - X_binary: torch.Tensor ~ (n_examples, n_vis_binary)
        - n_gibbs: int

        @returns
        - X_continuous
        - X_binary
        """
        if self.n_vis_continuous > 0 and X_continuous is not None:
            X_continuous_missing = torch.isnan(X_continuous)
            X_continuous_clamp = torch.logical_not(X_continuous_missing)
            X_continuous[X_continuous_missing] = 0
        if self.n_vis_binary > 0 and X_binary is not None:
            X_binary_missing = torch.isnan(X_binary)
            X_binary_clamp = torch.logical_not(X_binary_missing)
            X_binary[X_binary_missing] = 0
        _, X_continuous, X_binary = self._block_gibbs_sample(
            v_continuous=X_continuous,
            v_binary=X_binary,
            v_continuous_clamp=X_continuous_clamp,
            v_binary_clamp=X_binary_clamp,
            n_gibbs=n_gibbs)
        return X_continuous, X_binary
    
    def save_state(self, path: str, meta_path: str = None):
        """
        Saves the model state to a .pth file containing model weights and a 
        .json file containing model hyperparameter specifications. 

        @args
        - path: str << filepath, must be of the form XXXX.pth
        - meta_path: str << filepath, must be of the form YYYY.json; if None,
            an XXXX.json file will be created
        """
        if meta_path is None:
            meta_path = os.path.splitext(path)[0] + ".json"
        torch.save(self.state_dict(), path)
        with open(meta_path, "w") as json_file:
            json.dump(self.metadata(), json_file)

    def metadata(self):
        """
        Returns model metadata.
        """
        return {
            "n_vis_continuous": self.n_vis_continuous,
            "n_vis_binary": self.n_vis_binary,
            "n_hid": self.n_hid
        }

    def _reset_parameters(self):
        """
        Resets trainable parameters of the RBM.
        """
        torch.manual_seed(self.seed)
        if self.n_vis_continuous > 0:
            nn.init.xavier_normal_(self.W_continuous)
            nn.init.constant_(self.mu, 0)
            nn.init.constant_(self.log_var, 0)
        if self.n_vis_binary > 0:
            nn.init.xavier_normal_(self.W_binary)
            nn.init.constant_(self.a, 0)
        nn.init.constant_(self.b, 0)
    
    def _variance(self):
        """
        Returns the variance.

        @returns
        - var: torch.Tensor
        """
        if self.n_vis_continuous > 0 and self.log_var is not None :
            return torch.exp(self.log_var)
        return None

    def _energy(self, h: torch.Tensor, v_continuous: torch.Tensor | None, 
                v_binary: torch.Tensor | None):
        """
        Energy of the entire system for each example given. Not robust to
        missing data.

        @args
        - h: torch.Tensor ~ (batch_size, n_hid)
        - v_continuous: torch.Tensor ~ (batch_size, n_vis_continuous)
        - v_binary: torch.Tensor ~ (batch_size, n_vis_binary)

        @returns
        - total_energy: torch.Tensor ~ (batch_size)
        """
        var = self._variance()
        total_energy = 0
        if self.n_vis_continuous and v_continuous is not None: 
            total_energy = total_energy + (torch.sum(0.5 * (v_continuous - \
                self.mu) ** 2 / var, dim=1) - (torch.sum(((v_continuous / var) \
                @ (self.W_continuous)) * h, dim=1) + torch.sum(h * self.b, \
                dim=1)))
        if self.n_vis_binary > 0 and v_binary is not None:
            total_energy = total_energy - (torch.sum(h * self.b, dim=1) + \
                torch.sum(v_binary * self.a, dim=1) + \
                torch.sum((v_binary @ (self.W_binary)) * h, dim=1))
        return total_energy

    def _loss(self, v_continuous: torch.Tensor | None, 
              v_binary: torch.Tensor | None, n_gibbs: int = 1):
        """
        Generic contrastive divergence loss. Not robust to missing data.

        @args
        - v_continuous: torch.Tensor ~ (batch_size, n_vis_continuous)
        - v_binary: torch.Tensor ~ (batch_size, n_vis_binary)
        - n_gibbs: int 
        """
        h_data, _, _ = self._block_gibbs_sample(v_continuous, v_binary)
        h_model, v_continuous_model, v_binary_model = self._block_gibbs_sample(
            v_continuous, v_binary, n_gibbs=n_gibbs)
        L = self._energy(h_data, v_continuous, v_binary) - \
            self._energy(h_model, v_continuous_model, v_binary_model)
        return L.mean()
    
    @torch.no_grad()
    def _recon(self, v_continuous: torch.Tensor | None, 
              v_binary: torch.Tensor | None, n_gibbs: int = 1):
        """
        Reconstruction loss (mean squared error). Not robust to missing data. 

        @args
        - v_continuous: torch.Tensor ~ (batch_size, n_vis_continuous)
        - v_binary: torch.Tensor ~ (batch_size, n_vis_binary)
        - n_gibbs: int 

        @returns
        - mse: torch.Tensor
        """
        var = self._variance()
        mse = 0
        p_h_given_v = 0
        # compute expected h
        if self.n_vis_continuous > 0 and v_continuous is not None:
            p_h_given_v = p_h_given_v + (v_continuous / var) @ \
                self.W_continuous
        if self.n_vis_binary > 0 and v_binary is not None:
            p_h_given_v = p_h_given_v + self.b + v_binary @ self.W_binary
        # given expected h compute expected v
        if self.n_vis_continuous > 0 and v_continuous is not None:
            E_v_continuous_given_h = p_h_given_v @ self.W_continuous.t() + \
                self.mu
            mse = mse + torch.mean((E_v_continuous_given_h - v_continuous) ** 2)
        if self.n_vis_binary > 0 and v_binary is not None:
            E_v_binary_given_h = p_h_given_v @ self.W_binary.t() + self.a
            mse = mse + torch.mean((E_v_binary_given_h - v_binary) ** 2)
        return mse
    
    @torch.no_grad()
    def _block_gibbs_sample(self, v_continuous: torch.Tensor | None,
                            v_binary: torch.Tensor | None, 
                            v_continuous_clamp: torch.Tensor = None,
                            v_binary_clamp: torch.Tensor = None,
                            n_gibbs: int = 0):
        """
        Block Gibbs sampling of the visible units. If clamps exist 
        (Boolean masks), then elements corresponding to True in the mask will 
        be held at their initial values. Not robust
        to missing values in inputs. 

        @args
        - v_continuous: torch.Tensor ~ (batch_size, n_vis_continuous)
        - v_binary: torch.Tensor ~ (batch_size, n_vis_binary)
        - v_continuous_clamp: torch.Tensor ~ (batch_size, n_vis_continuous)
        - v_binary_clamp: torch.Tensor ~ (batch_size, n_vis_binary)
        - n_gibbs: int << same as burn in; if 0, only the hidden units 
            are changed

        @returns
        - h: torch.Tensor ~ (batch_size, n_hid)
        - v_continuous: torch.Tensor ~ (batch_size, n_vis_continuous)
        - v_binary: torch.Tensor ~ (batch_size, n_vis_binary)
        """
        var = self._variance()
        if var is None:
            std = None
        else:
            std = torch.sqrt(var)
        h_sample = self._sample_hid_given_vis(v_continuous, v_binary, std)
        if v_continuous_clamp is not None:
            v_continuous_orig = torch.clone(v_continuous)
        if v_binary_clamp is not None:
            v_binary_orig = torch.clone(v_binary)
        for i in range(n_gibbs):
            v_continuous_sample, v_binary_sample = self._sample_vis_given_hid(
                h_sample, std)
            if v_continuous_clamp is not None and \
                v_continuous_sample is not None:
                v_continuous_sample[v_continuous_clamp] = \
                    v_continuous_orig[v_continuous_clamp]
            if v_binary_clamp is not None and v_binary_sample is not None:
                v_binary_sample[v_binary_clamp] = v_binary_orig[v_binary_clamp]
            h_sample = self._sample_hid_given_vis(
                v_continuous_sample, v_binary_sample, std)
        return h_sample, v_continuous, v_binary
    
    @torch.no_grad()
    def _sample_hid_given_vis(self, v_continuous: torch.Tensor | None, 
                              v_binary: torch.Tensor | None,
                              std: torch.Tensor | None):
        """
        Samples hidden units given visible units

        @args
        - v_continuous: torch.Tensor ~ (batch_size, n_vis_continuous)
        - v_binary: torch.Tensor ~ (batch_size, n_vis_binary)
        - std: torch.Tensor ~ (batch_size, n_vis_continuous) << stdev

        @returns
        - h: torch.Tensor ~ (batch_size, n_hid)
        """
        probability = 0
        if self.n_vis_continuous > 0 and v_continuous is not None:
            probability = probability + (v_continuous / (std ** 2)) @ \
                self.W_continuous
        if self.n_vis_binary > 0 and v_binary is not None:
            probability = probability + self.b + v_binary @ self.W_binary
        return torch.bernoulli(p=torch.sigmoid(probability), generator=self.rng)
        
    @torch.no_grad()
    def _sample_vis_given_hid(self, h: torch.Tensor, std: torch.Tensor | None):
        """
        Samples visible units given hidden units

        @args
        - h: torch.Tensor ~ (batch_size, n_hid)
        - std: torch.Tensor ~ (batch_size, n_vis_continuous) << stdev

        @returns
        - v_continuous: torch.Tensor ~ (batch_size, n_vis_continuous)
        - v_binary: torch.Tensor ~ (batch_size, n_vis_binary)
        """
        batch_size = h.shape[0]
        v_continuous_sample = None
        v_binary_sample = None
        if self.n_vis_continuous > 0 and std is not None:
            v_continuous_sample = h @ self.W_continuous.t() + self.mu
            v_continuous_sample += torch.randn(
                size=(batch_size, self.n_vis_continuous), 
                generator=self.rng) * std
        if self.n_vis_binary > 0:
            v_binary_sample = torch.bernoulli(
                p=torch.sigmoid(h @ self.W_binary.t() + self.a),
                generator=self.rng)
        return v_continuous_sample, v_binary_sample

def build_from_checkpoint(path: str, meta_path: str = None) -> RBMBackend:
    """     
    Returns an RBM object 

    @args
    - path: str << filepath, must be of the form XXXX.pth
    - meta_path: str << filepath, must be of the form YYYY.json; if None,
        an XXXX.json file will be created
    """
    if meta_path is None:
        meta_path = ".".join(path.split(".")[:-1]) + ".json"
    with open(meta_path, "r") as json_file:
        metadata = json.load(json_file)
    model = RBMBackend(metadata["n_vis_continuous"], metadata["n_vis_binary"], 
        metadata["n_hid"])
    model.load_state_dict(torch.load(path))
    return model
