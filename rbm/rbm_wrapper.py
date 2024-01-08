import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .rbm import RBMBackend, build_from_checkpoint


class RestrictedBoltzmannMachine():
    """
    NumPy and pandas friendly RBM model. Robust to missing data. Handles
    both continuous and categorical data. 
    """

    def __init__(self, hidden_layer_size: int = None, 
                 folder_path: str = None):
        """
        Builds a RestrictedBoltzmannMachine object. If filepath to saved 
        model checkpoints is given, then will build from checkpoint.

        @args
        - hidden_layer_size: int << size of the RBM hidden layer
        - folder_path: str << path to directory containing saved checkpoint
            information
        """
        self.hidden_layer_size = hidden_layer_size
        self.rbm_backend = None
        self.continuous_vars = None
        self.categorical_vars = None
        self.binary_vars = None
        self.categorical_column_mapping = None
        if folder_path is not None:
            self.build_from_checkpoint(folder_path)
    
    def load_data(self, dataframe: pd.DataFrame, 
                  continuous_vars: list[str] = [],
                  categorical_vars: list[str] = []):
        """
        Reconfigures the RBM based on dataframe characteristics. 
        Automatically one-hot-encodes the categorical variables and scales the 
        continuous variables for training. 

        @args
        - dataframe: pd.DataFrame
        - continuous_vars: list[str]
        - categorical_vars: list[str]
        """
        if len(self.continuous_vars) + len(self.categorical_vars) == 0:
            raise ValueError("At least one variable must be identified.")
        self.df_orig = dataframe.copy()
        self.continuous_vars = continuous_vars
        self.categorical_vars = categorical_vars
        self.df = pd.get_dummies(self.df_orig, columns=categorical_vars, 
            drop_first=True)
        self.binary_vars = self.df.columns.to_list()
        self.categorical_column_mapping = {}
        for original_column in self.categorical_vars:
            one_hot_columns = [col for col in self.binary_vars if \
                "_".join(col.split("_")[:-1]) == original_column]
            self.categorical_column_mapping[original_column] = one_hot_columns
        self.standard_scaler = StandardScaler
        X_continuous_transformed = self.standard_scaler.fit_transform(
            self.df[self.continuous_vars].to_numpy())
        self.df[self.continuous_vars] = X_continuous_transformed
        if self.hidden_layer_size is None:
            self.hidden_layer_size = 2 * (len(self.continuous_vars) + \
                len(self.binary_vars))
        if len(self.continuous_vars) == 0:
            self.rbm_backend = RBMBackend(
                n_continuous=None,
                n_binary=len(self.binary_vars),
                hidden_layer_size=self.hidden_layer_size
            )
        elif len(self.binary_vars) == 0:
            self.rbm_backend = RBMBackend(
                n_continuous=len(self.continuous_vars),
                n_binary=None,
                hidden_layer_size=self.hidden_layer_size
            )
        else:
            self.rbm_backend = RBMBackend(
                n_continuous=len(self.continuous_vars),
                n_binary=len(self.binary_vars),
                hidden_layer_size=self.hidden_layer_size
            )

    def fit(self, burn_in: int = 1, epochs: int = 10, batch_size: int = 1, 
            lr: float = 1e-2, verbose_interval: int = 1):
        """
        Trains the RBM. 

        @args
        - burn_in: int << number of sampling steps; lower burn in 
            may lead to better mixing during training
        - epochs: int << number of training epochs
        - batch_size: int << size of each batch
        - lr: float << learning rate
        - verbose_interval: str << number of epochs before next train progress 
            update; if None, nothing printed
        """
        X_continuous = None
        X_binary = None
        if len(self.continuous_vars) > 0:
            X_continuous = torch.Tensor(self.df[self.continuous_vars].\
                to_numpy())
        if len(self.binary_vars) > 0:
            X_binary = torch.Tensor(self.df[self.binary_vars].to_numpy())
        self.rbm_backend.fit(
            X_continuous=X_continuous,
            X_binary=X_binary,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            n_gibbs=burn_in,
            verbose_interval=verbose_interval
        )

    def transform(self, n_samples: int = None, dataframe: pd.DataFrame = None, 
                  burn_in: int = 10):
        """
        Samples from the RBM. Outputs will be of the same format as the 
        DataFrame the RBM was trained on. 
        
        Either n_samples or dataframe is to be set to 
        None. If given n_samples, outputs a DataFrame with n_samples rows.
        If given a dataframe, the appropriate columns will be utilized and 
        sampling is done conditionally based on the data provided in the 
        dataframe. The dataframe must contain missing values to be filled in
        by the RBM.

        @args
        - n_samples: int
        - dataframe: pd.DataFrame
        - burn_in: int

        @returns
        - pd.DataFrame
        """
        if n_samples is None and dataframe is None:
            raise ValueError("Either n_samples or dataframe must be provided.")
        
        true_variable_names = self.continuous_vars + self.categorical_vars
        working_variable_names = self.continuous_vars + self.binary_vars
        if n_samples is not None:
            nan_data = np.full((n_samples, len(self.categorical_vars) + \
                len(self.continuous_vars)), np.nan)
            output_df = pd.DataFrame(nan_data, columns=true_variable_names)
        elif dataframe is not None:
            output_df = dataframe[true_variable_names].copy()
            n_samples = len(output_df)
        
        nan_data = np.full((n_samples, len(self.continuous_vars) + \
                len(self.binary_vars)), np.nan)
        working_df = pd.DataFrame(nan_data, columns=working_variable_names)
        working_df[self.continuous_vars] = \
            self.standard_scaler.transform(
                output_df[self.continuous_vars].to_numpy())
        for true_var in self.categorical_vars:
            categories = [var.split("_")[-1] for var in \
                self.categorical_column_mapping[true_var]]
            for category in categories:
                new_column = np.full((n_samples), 0)
                new_column[np.isnan(working_df[true_var].to_numpy())] = np.nan
                new_column[working_df[true_var].to_numpy() == category] = 1
                working_df[true_var + "_" + category] = new_column
        X_continuous = None
        X_binary = None
        if len(self.continuous_vars) > 0:
            X_continuous = torch.Tensor(working_df[self.continuous_vars].\
                to_numpy())
        if len(self.binary_vars) > 0:
            X_binary = torch.Tensor(working_df[self.binary_vars].to_numpy())
        raw_samples = self.rbm_backend.transform(
            X_continuous=X_continuous,
            X_binary=X_binary,
            n_gibbs=burn_in,
            binary_proba=True
        )
        working_output_df = pd.DataFrame(raw_samples, 
            columns=working_variable_names)
        nan_data = np.full((n_samples, len(self.continuous_vars) + \
                len(self.categorical_vars)), np.nan)
        output_df = pd.DataFrame(nan_data, columns=true_variable_names)
        output_df[self.continuous_vars] = \
            self.standard_scaler.inverse_transform(
                working_output_df[self.continuous_vars].to_numpy())
        for true_var in self.categorical_vars:
            category_vars = [var for var in \
                self.categorical_column_mapping[true_var]]
            output_df[true_var] = \
                working_output_df[category_vars].idxmax(axis=1)
            output_df[true_var] = output_df[true_var].str.split('_').str[-1]
        return output_df
    
    def save_state(self, folder_path: str):
        """
        Saves the trained model state, model metadata, and dataset metadata 
        into the folder. 

        @args
        - folder_path: path to the folder in which files will be saved
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.rbm_backend.save_state(f"{folder_path}/rbm_backend.pth")
        with open(f"{folder_path}/dataset_metadata.json", "w") as json_file:
            json.dump(self._dataset_metadata(), json_file, indent=2)
        print(f"State saved to {folder_path}.")

    def build_from_checkpoint(self, folder_path: str):
        """
        If an RBM has already been trained, it can be loaded directly. 

        @args
        - folder_path: path to the folder in which files will be saved
        """
        self.rbm_backend = build_from_checkpoint(
            f"{folder_path}/rbm_backend.pth")
        
        print(f"RBM successfully built from {folder_path}.")
        
    def reset(self, hidden_layer_size: int = None, random_seed: int = None):
        """
        The hidden layer size may be adjusted, as may the random seed. If 
        the hidden layer size is adjusted, then fit() must be called again. 
        """
        if hidden_layer_size is not None:
            self.hidden_layer_size = hidden_layer_size
            if self.rbm_backend is not None:
                self.rbm_backend.reset_hyperparameters(
                    hidden_layer_size=hidden_layer_size)
        if random_seed is not None and self.rbm_backend is not None:
            self.rbm_backend.reset_seed(random_seed)

    def _dataset_metadata(self):
        """
        Returns a dictionary of dataset metadata.

        @returns 
        - dict
        """
        return {
            "continuous_vars": self.continuous_vars,
            "categorical_vars": self.categorical_vars,
            "categorical_column_mapping": self.categorical_column_mapping,
            "binary_vars": self.binary_vars
        }


