import os
from typing import List, Tuple, Optional, Dict, Any, Union, Sequence

import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
#import scipy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

# Scikit-learn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
    r2_score
)

# import percentage_errors computation functions
from analysis_utils import compute_absolute_percentage_errors, compute_percentage_errors

# Functions used in LaneSpotRateForecaster Class methods
## 1. Data prep
### a.i Create Dataframe which allows to have Multivariates (Multi-features)
def prepare_dataframe_for_lstm(
    df: pd.DataFrame,
    date_col: str,
    feature_cols: List[str],
    lookback: int = 7,
    days_ahead: Optional[int] = None,
    future_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare DataFrame for LSTM by creating lagged features efficiently.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Name of date column
    feature_cols : List[str]
        List of feature columns to create lags for
    lookback : int, optional
        Number of lookback periods, by default 7
    days_ahead : Optional[int], optional
        Number of days ahead to predict. If provided, creates future features
    future_cols : Optional[List[str]], optional
        List of columns to create future features for. If None but days_ahead is provided,
        uses all feature_cols
        
    Returns
    -------
    pd.DataFrame
        DataFrame with lagged and future features
    """
    # Create a copy of the input DataFrame
    df = df.copy()
    
    # Sort by date to ensure correct shifting
    df = df.sort_values(date_col)
    
    # Initialize list to store all lagged features
    lagged_features = []
    
    # Create lagged features for each feature column
    for feature in feature_cols:
        # Create all lags for current feature at once
        lags = pd.concat(
            [df[feature].shift(i).rename(f'{feature}(t-{i})') for i in range(1, lookback + 1)],
            axis=1
        )
        lagged_features.append(lags)
    
    # Initialize list to store future features if requested
    future_features = []
    if days_ahead is not None:
        # If future_cols not specified, use all feature_cols
        future_cols = future_cols or feature_cols
        
        # Create future features for specified columns
        for feature in future_cols:
            future = df[feature].shift(-days_ahead).rename(f'{feature}(t+{days_ahead})')
            future_features.append(pd.DataFrame(future))
    
    # Combine all features with original DataFrame
    result = pd.concat(
        [df[[date_col]]] +  # Start with date column
        future_features +    # Then future features
        [df[feature_cols]] + # Then current features
        lagged_features,     # Finally lagged features
        axis=1
    )
    
    # Print shapes before and after dropping NaN
    print(f"\nShape before dropping NaN: {result.shape}")
    
    # Drop rows with NaN values only in the target column (if days_ahead is specified)
    if days_ahead is not None and future_cols:
        future_target = f"{future_cols[0]}(t+{days_ahead})"
        result = result.dropna(subset=[future_target])
        print(f"Shape after dropping NaN in target {future_target}: {result.shape}")
    
    # Drop rows with NaN in feature columns
    result = result.dropna()
    print(f"Shape after dropping all NaN: {result.shape}")
    
    return result    

### b. Create Lagged Dataset for Time Series
def create_lagged_dataset(
    shifted_df: pd.DataFrame,
    target_col: str,
    lagged_features: List[str],
    lookback: int,
    time_col: str = 'pu_date',
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create lagged dataset for LSTM model training by generating time-shifted features.
    Now using t to t-6 instead of t-1 to t-7 for lookback=7.

    Parameters
    ----------
    shifted_df : pd.DataFrame
        Input DataFrame containing features and target
    target_col : str
        Name of target column
    lagged_features : List[str]
        List of feature column names to create lags for
    lookback : int
        Number of time steps to look back (including current time t)
    time_col : str, optional
        Name of datetime column, by default 'pu_date'

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        X : np.ndarray
            3D array of shape (samples, lookback, features) containing lagged features
            ordered as [t-(lookback-1), ..., t-1, t]
        y : np.ndarray
            1D array of shape (samples,) containing target values
        dates : np.ndarray
            1D array of shape (samples,) containing datetime values
    """
    df = shifted_df.copy()
    
    if verbose:
        print("\nInput verification:")
        print(f"Target column: {target_col}")
        print("\nFeatures to create lags for:")
        for feat in lagged_features:
            print(f"{feat}: First value = {df[feat].iloc[0]:.4f}")
    
    print("\nCreating lagged features...")
    for feature in lagged_features:
        # Add current time column (t)
        df[f'{feature}(t-0)'] = df[feature]
        # Add lagged columns from t-1 to t-(lookback-1)
        for t in range(1, lookback):
            df[f'{feature}(t-{t})'] = df[feature].shift(t)
    
    # Get feature columns in correct order [t-(lookback-1) to t]
    feature_cols = []
    for feature in lagged_features:
        feature_cols.extend([f'{feature}(t-{t})' for t in range(lookback-1, -1, -1)])
    
    if verbose:
        print(f"\nFeature columns order: {feature_cols}")
        print(f"Shape before dropping NaN: {df.shape}")
    
    # Drop rows with NaN values
    df = df.dropna(subset=feature_cols + [target_col])
    print(f"\nShape after dropping NaN: {df.shape}")
    
    # Create feature arrays
    X = np.zeros((len(df), lookback, len(lagged_features)))
    
    # Fill the X array with values
    for i, feature in enumerate(lagged_features):
        lag_cols = [f'{feature}(t-{t})' for t in range(lookback-1, -1, -1)]
        X[:, :, i] = df[lag_cols].values
    
    # Get target and dates
    y = np.array(df[target_col].values)
    dates = np.array(df[time_col].values)
    
    if verbose:
        print("\nData quality check:")
        print(f"Any NaN in X: {np.isnan(X).any()}")
        print(f"Any NaN in y: {np.isnan(y).any()}")
        print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
        print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"\nFinal dataset shapes:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"dates shape: {dates.shape}")
        
        # Print first sample for verification
        print("\nFirst sample values:")
        for i, feature in enumerate(lagged_features):
            print(f"\n{feature}:")
            for t in range(lookback):
                print(f"t-{lookback-1-t}: {X[0, t, i]:.4f}")
    
    return X, y, dates

### b. Create Time Series for Predictino using he last datapoint
def create_prediction_lagged_dataset(
    shifted_df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int,
    time_col: str = 'pu_date',
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dataset for prediction using the last lookback days.

    Parameters
    ----------
    shifted_df : pd.DataFrame
        Input DataFrame containing features
    feature_cols : List[str]
        List of feature column names
    lookback : int
        Number of time steps to look back
    time_col : str, optional
        Name of datetime column, by default 'pu_date'
    verbose : bool, optional
        Whether to print debug information, by default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X : np.ndarray
            3D array of shape (1, lookback, features) containing features
        dates : np.ndarray
            1D array containing the prediction date
    """
    if verbose:
        print("\nPreparing prediction data...")
        print(f"Features being used: {feature_cols}")
        
    # Verify we have enough data
    if len(shifted_df) < lookback:
        raise ValueError(
            f"Not enough data for prediction. Need {lookback} days, got {len(shifted_df)}"
        )
    
    # Check for missing values
    missing_values = shifted_df[feature_cols].isna().any()
    if missing_values.any():
        if verbose:
            print("\nWarning: Missing values found in features:")
            print(missing_values[missing_values])
        
        # Handle missing values
        df = shifted_df.copy()
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
    else:
        df = shifted_df

    # Create feature array
    X = np.zeros((1, lookback, len(feature_cols)))
    
    # Fill the X array with the last lookback days of data
    for i, feature in enumerate(feature_cols):
        X[0, :, i] = df[feature].values[-lookback:]
    
    # Get prediction date
    prediction_date = df[time_col].iloc[-1]
    dates = np.array([prediction_date])
    
    if verbose:
        print(f"\nPrediction data shape: {X.shape}")
        print(f"Using data up to: {prediction_date}")
        print("\nFeature values for prediction:")
        for i, feature in enumerate(feature_cols):
            print(f"\n{feature}:")
            for t in range(lookback):
                # Changed to use t-(lookback-1-t) format to match create_lagged_dataset
                print(f"t-{lookback-1-t}: {X[0, t, i]:.4f}")
        
        # Verify data quality
        print("\nData quality check:")
        print(f"Any NaN in X: {np.isnan(X).any()}")
        print(f"X range: [{X.min():.4f}, {X.max():.4f}]")
    
    return X, dates


### c. Split and Scale Data
def split_and_scale_data(
    X: np.ndarray,
    y: np.ndarray,
    y_dates: np.ndarray,
    train_size: Union[float, int] = 0.7,
    val_size: Union[float, int] = 0.15,
    test_size: Union[float, int] = 0.15,
    gap_size: int = 0,
    shuffle: bool = False,
    feature_range: Tuple[float, float] = (-1, 1)
) -> Tuple[np.ndarray, ...]:
    """Split and scale data with optional validation set."""
    N = len(X)
    
    # Always take test_size from the end
    if isinstance(test_size, int):
        test_start = N - test_size
    else:
        test_start = N - int(N * test_size)
    
    # Handle the case when val_size = 0
    if val_size == 0:
        # If no validation, train data goes up to test_start minus gap
        train_start = 0
        train_end = test_start - gap_size if gap_size > 0 else test_start
        
        # Empty arrays for validation
        X_val = np.array([])
        y_val = np.array([])
        dates_val = np.array([])
        
    else:
        # Calculate validation split
        if isinstance(val_size, int):
            val_start = test_start - val_size - gap_size
        else:
            val_start = test_start - int(N * val_size) - gap_size
            
        # Calculate training split
        train_start = 0
        train_end = val_start - gap_size if gap_size > 0 else val_start
        
        # Validation data
        X_val = X[val_start:test_start-gap_size if gap_size > 0 else test_start]
        y_val = y[val_start:test_start-gap_size if gap_size > 0 else test_start]
        dates_val = y_dates[val_start:test_start-gap_size if gap_size > 0 else test_start]
    
    # Validate splits
    if train_end <= train_start:
        raise ValueError("Not enough samples for training data")
    if test_start >= N:
        raise ValueError("Not enough samples for test data")
    
    # Split the data
    X_train = X[train_start:train_end]
    X_test = X[test_start:]
    
    y_train = y[train_start:train_end]
    y_test = y[test_start:]
    
    dates_train = y_dates[train_start:train_end]
    dates_test = y_dates[test_start:]
    
    # Scale the data
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    scaler_X = MinMaxScaler(feature_range=feature_range)
    scaler_y = MinMaxScaler(feature_range=feature_range)
    
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
    X_test_scaled = scaler_X.transform(X_test_reshaped)
    
    X_train_scaled = X_train_scaled.reshape(X_train_shape)
    X_test_scaled = X_test_scaled.reshape(X_test_shape)
    
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Scale validation data if it exists
    if val_size > 0:
        X_val_shape = X_val.shape
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
        X_val_scaled = scaler_X.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(X_val_shape)
        y_val = y_val.reshape(-1, 1)
        y_val_scaled = scaler_y.transform(y_val)
    else:
        X_val_scaled = X_val
        y_val_scaled = y_val
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            dates_train, dates_val, dates_test,
            scaler_X, scaler_y)


### d. Tensorize the input, create Torch DataLoader    
def prepare_data_loaders(X_scaled: np.ndarray, y_scaled: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    Prepare PyTorch data loaders from scaled data.
    
    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled input features array (samples, time_steps, features).
    y_scaled : np.ndarray
        Scaled target variable array (samples, 1).
    batch_size : int
        The batch size for the data loader.
    shuffle : bool, optional
        Whether to shuffle the data at every epoch.
    
    Returns
    -------
    data_loader : DataLoader
        PyTorch DataLoader object to iterate over the data in batches.
    X_tensor : torch.Tensor
        Input features tensor for the scaled data.
    y_tensor : torch.Tensor
        Target variable tensor for the scaled data.
    """
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader, X_tensor, y_tensor    


## 2. Define the LSTM model with hyper parameters
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            dropout=dropout if num_layers > 1 else 0.0,  # Dropout applied only if num_layers > 1
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)  # Explicit dropout layer
        self.fc = nn.Linear(hidden_size, 1)  # Predicting one value

    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x.size(0))
        out, h = self.lstm(x, h)
        out = self.dropout(out)  # Apply dropout
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(next(self.parameters()).device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(next(self.parameters()).device)
        return (h0, c0)
    

# 3. Train and Validate the model with early stopping
def train_and_validate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    num_epochs: int = 50,
    patience: int = 15,
    learning_rate: float = 0.001,
    lr_factor: float = 0.5,
    criterion: Optional[nn.Module] = None,
    optimizer_class: Optional[torch.optim.Optimizer] = None,
    plot_losses: bool = True,
    save_fig: Optional[str] = None,
    save_mdl: Optional[str] = None,
    verbose: bool = True
) -> Tuple[nn.Module, List[float], List[float], int]:
    """
    Train and validate LSTM model with early stopping.
    """
    if criterion is None:
        criterion = nn.MSELoss()
    if optimizer_class is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    # Initialize for training
    use_validation = val_loader is not None
    if use_validation:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=5, verbose=verbose)
        best_val_loss = float('inf')
        patience_counter = 0
    
    train_losses = []
    val_losses = [] if use_validation else None
    epochs_used = num_epochs

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            # Reshape y_batch to match model output shape [batch_size, 1]
            y_batch = y_batch.view(-1, 1).to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase (if validation data exists)
        if use_validation:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch = X_val_batch.to(device)
                    y_val_batch = y_val_batch.view(-1, 1).to(device)
                    
                    val_outputs = model(X_val_batch)
                    val_loss += criterion(val_outputs, y_val_batch).item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling and early stopping
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                epochs_used = epoch + 1
                break

        # Print progress
        if verbose:
            status = f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}'
            if use_validation:
                status += f', Validation Loss: {avg_val_loss:.4f}'
            print(status)

    # Plot losses if requested
    if plot_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        if use_validation:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        
        if save_fig is not None:
            plt.savefig(save_fig, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Saving training plot to {save_fig}")
        
        plt.show()
        plt.close()

    # Save the final model if requested
    if save_mdl is not None:
        torch.save(model.state_dict(), save_mdl)
        if verbose:
            print(f"Final model saved to {save_mdl}")

    return model, train_losses, val_losses, epochs_used


# 4. Evaluate the model using Loss on the Test data
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    model_path: Optional[str] = None  # New parameter to specify the model file to load
) -> float:
    """
    Evaluate the model on the test set.

    Parameters
    ----------
    model : nn.Module
        The LSTM model instance.
    test_loader : DataLoader
        DataLoader for the test dataset.
    device : torch.device
        The device to run the evaluation on (e.g., 'cpu' or 'cuda').
    criterion : Optional[nn.Module], optional
        The loss function. Defaults to nn.MSELoss().
    model_path : Optional[str], optional
        Path to the saved model file. If provided, the model weights will be loaded from this file.

    Returns
    -------
    float
        The average test loss.
    """
    if criterion is None:
        criterion = nn.MSELoss()

    # Load the saved model weights if model_path is provided
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")

    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    test_loss = 0.0
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            test_outputs = model(X_test_batch)
            loss = criterion(test_outputs, y_test_batch)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Final Test Loss: {avg_test_loss:.4f}')
    return avg_test_loss    


## 5. Predict and Evaluate (Measure Model Performance)
def make_predictions_and_inverse_transform(
    model: nn.Module,
    X_train_tensor: torch.Tensor,
    X_test_tensor: torch.Tensor,
    y_train_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
    scaler_y: MinMaxScaler,
    device: torch.device,
    train_dates: np.ndarray,  # Add dates parameters
    test_dates: np.ndarray,
    plot_results: bool = True,
    model_path: Optional[str] = None,
    save_fig_train: Optional[str] = None,
    save_fig_test: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions and inverse transform the scaled data.
    
    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model
    X_train_tensor : torch.Tensor
        Training features tensor
    X_test_tensor : torch.Tensor
        Test features tensor
    y_train_scaled : np.ndarray
        Scaled training target values
    y_test_scaled : np.ndarray
        Scaled test target values
    scaler_y : MinMaxScaler
        Fitted scaler for target variable
    train_dates : np.ndarray
        Dates corresponding to training data
    test_dates : np.ndarray
        Dates corresponding to test data        
    device : torch.device
        Device to run predictions on
    plot_results : bool, optional
        Whether to plot results, by default True
    model_path : str, optional
        Path to saved model, by default None
    save_fig_train : str, optional
        Path to save training predictions plot, by default None
    save_fig_test : str, optional
        Path to save test predictions plot, by default None

        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Training predictions, training actuals, test predictions, test actuals
    """
    # Load model if path provided
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        train_pred = model(X_train_tensor.to(device)).cpu().numpy()
        test_pred = model(X_test_tensor.to(device)).cpu().numpy()
    
    # Inverse transform predictions and actual values
    train_pred_inversed = scaler_y.inverse_transform(train_pred)
    y_train_inversed = scaler_y.inverse_transform(y_train_scaled)
    test_pred_inversed = scaler_y.inverse_transform(test_pred)
    y_test_inversed = scaler_y.inverse_transform(y_test_scaled)
    
    if plot_results:
        # Plot training predictions
        plt.figure(figsize=(12, 6))
        plt.plot(train_dates, y_train_inversed, label='Actual', color='#4C72B0',
                marker='o', markersize=2, linestyle='-', alpha=0.7)
        plt.plot(train_dates, train_pred_inversed, label='Predicted', color='#DD8452',
                marker='s', markersize=2, linestyle='-', alpha=0.7)
        
        # Add metrics text
        train_mape = compute_absolute_percentage_errors(y_train_inversed, train_pred_inversed).mean()
        train_medape = np.median(compute_absolute_percentage_errors(y_train_inversed, train_pred_inversed))
        train_mpe = compute_percentage_errors(y_train_inversed, train_pred_inversed).mean()
        train_mdpe = np.median(compute_percentage_errors(y_train_inversed, train_pred_inversed))
        
        plt.text(0.02, 0.98, 
                f'MAPE: {train_mape:.2f}%\nMdAPE: {train_medape:.2f}%\n'
                f'MPE: {train_mpe:.2f}%\nMdPE: {train_mdpe:.2f}%',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.title('Training Data - Actual vs Predicted')
        plt.xlabel('Date')  # Updated x-label
        plt.ylabel('Target Variable')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)  # Rotate date labels
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        
        if save_fig_train:
            plt.savefig(save_fig_train, dpi=300, bbox_inches='tight')
            print(f"Saving training predictions plot to {save_fig_train}")
        plt.show()

        # Plot test predictions
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, y_test_inversed, label='Actual', color='#4C72B0',
                marker='o', markersize=4, linestyle='-', alpha=0.7)
        plt.plot(test_dates, test_pred_inversed, label='Predicted', color='#DD8452',
                marker='s', markersize=4, linestyle='-', alpha=0.7)
        
        # Add metrics text
        test_mape = compute_absolute_percentage_errors(y_test_inversed, test_pred_inversed).mean()
        test_medape = np.median(compute_absolute_percentage_errors(y_test_inversed, test_pred_inversed))
        test_mpe = compute_percentage_errors(y_test_inversed, test_pred_inversed).mean()
        test_mdpe = np.median(compute_percentage_errors(y_test_inversed, test_pred_inversed))
        
        plt.text(0.02, 0.98,
                f'MAPE: {test_mape:.2f}%\nMdAPE: {test_medape:.2f}%\n'
                f'MPE: {test_mpe:.2f}%\nMdPE: {test_mdpe:.2f}%',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Calculate and display RMSE
        rmse = np.sqrt(mean_squared_error(y_test_inversed, test_pred_inversed))
        plt.text(0.02, 0.02,
                f'Root Mean Squared Error (RMSE): {rmse:.3f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='bottom')
        
        plt.title('Testing Data - Actual vs Predicted')
        plt.xlabel('Date')  # Updated x-label
        plt.ylabel('Target Variable')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)  # Rotate date labels
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        
        if save_fig_test:
            plt.savefig(save_fig_test, dpi=300, bbox_inches='tight')
            print(f"Saving test predictions plot to {save_fig_test}")
        plt.show()

    return train_pred_inversed, y_train_inversed, test_pred_inversed, y_test_inversed
    

# II. Appendix
## 1. Data Prep
# Feature enrichment with momentum features (not used in v1)
## 1.a momentum
def calculate_momentum_features(
    df: pd.DataFrame,
    base_col: str = 'rate_per_mi_mean',
    periods: List[int] = [3, 7, 14],
    time_col: str = 'pu_date'
) -> pd.DataFrame:
    """
    Calculate various momentum indicators for rates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with rates and dates
    base_col : str
        Name of the rate column
    periods : List[int]
        List of periods to calculate momentum for
    time_col : str, optional
        Name of the time column, by default 'pu_date'
    """
    # Make a copy and sort by date
    df_out = df.copy()
    if time_col in df_out.columns:
        df_out = df_out.sort_values(time_col)
    
    for period in periods:
        # 1. Simple rate of change (percentage change)
        df_out[f'rate_momentum_{period}d'] = df[base_col].pct_change(periods=period)
        
        # 2. Rolling momentum (average daily change over period)
        df_out[f'rate_momentum_rolling_{period}d'] = (
            (df[base_col] - df[base_col].shift(period)) / period
        )
        
        # 3. Exponential momentum (gives more weight to recent changes)
        df_out[f'rate_momentum_exp_{period}d'] = (
            df[base_col].ewm(span=period).mean().pct_change()
        )
        
        # 4. Acceleration (change in momentum)
        df_out[f'rate_momentum_{period}d'].diff()
        
        # 5. Moving/Rolling average & Relative Strength
        ma = df[base_col].rolling(window=period).mean()
        df_out[f'rate_{period}d'] = ma
        
        # Safe division for rate strength calculation
        df_out[f'rate_strength_{period}d'] = (
            df_out[base_col] / ma.replace(0, np.nan)
        ).fillna(np.nan) - 1
        
    return df_out


## 1.b momentum ratio
def calculate_momentum_ratios(
    df: pd.DataFrame,
    base_col: str = 'rate_per_mi_mean',
    periods: List[int] = [3, 7, 14],
    time_col: str = 'pu_date',
    prefix: str = "momentum_ratio"
) -> pd.DataFrame:
    """
    Calculate momentum ratios between adjacent periods.
    """
    # Make a copy and sort by date
    df_out = df.copy()
    if time_col in df_out.columns:
        df_out = df_out.sort_values(time_col)
    
    # Sort periods to ensure correct pairing
    periods = sorted(periods)
    
    # Calculate ratios between adjacent periods
    for i in range(len(periods)-1):
        shorter_period = periods[i]
        longer_period = periods[i+1]
        
        col_name = f"{prefix}_{shorter_period}_{longer_period}"
        
        # Calculate moving averages for both periods
        ma_short = df[base_col].rolling(window=shorter_period, min_periods=1).mean()
        ma_long = df[base_col].rolling(window=longer_period, min_periods=1).mean()
        
        # Calculate ratio with safety checks:
        # 1. Replace 0s in denominator with NaN to avoid division by zero
        # 2. Handle cases where either numerator or denominator is NaN
        df_out[col_name] = (ma_short / ma_long.replace(0, np.nan)).fillna(np.nan)
    
    return df_out

## 1.c momentum trend reversal
def calculate_trend_reversal(
    df: pd.DataFrame,
    base_col: str = "rate_per_mi_mean",
    short_period: int = 7,
    long_period: int = 30,
    time_col: str = 'pu_date'
) -> pd.DataFrame:
    """
    Calculate trend reversal indicators and moving averages.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    base_col : str, optional
        Name of the base column for calculations, by default "rate_per_mi_mean"
    short_period : int, optional
        Window for short-term moving average, by default 7
    long_period : int, optional
        Window for long-term moving average, by default 30
    time_col : str, optional
        Name of time column, by default 'pu_date'
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - Original columns
        - ma_short: Short-term moving average
        - ma_long: Long-term moving average
        - trend_reversal: Crossover indicator (-1, 0, 1)
        - days_since_reversal: Days since last trend reversal
    """
    # Make a copy and sort
    df_out = df.copy()
    if time_col in df_out.columns:
        df_out = df_out.sort_values(time_col)
    
    # Calculate moving averages with safety for min_periods
    ma_short = df_out[base_col].rolling(window=short_period, min_periods=1).mean()
    ma_long = df_out[base_col].rolling(window=long_period, min_periods=1).mean()
    
    # Store moving averages in output
    df_out[f'ma_short_{short_period}d'] = ma_short
    df_out[f'ma_long_{long_period}d'] = ma_long
    
    # Calculate crossovers (trend reversals)
    # 1: Bullish crossover (short crosses above long)
    # -1: Bearish crossover (short crosses below long)
    # 0: No crossover
    crossover = np.where(
        (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1)), 1,
        np.where(
            (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1)), -1,
            0
        )
    )
    
    df_out['trend_reversal'] = crossover
    
    # Calculate days since last reversal
    reversal_idx = np.where(crossover != 0)[0]
    days_since = np.zeros(len(df_out))
    
    if len(reversal_idx) > 0:
        for i in range(len(df_out)):
            last_reversal = reversal_idx[reversal_idx <= i]
            if len(last_reversal) > 0:
                days_since[i] = i - last_reversal[-1]
            else:
                days_since[i] = i
    
    df_out['days_since_reversal'] = days_since
    
    return df_out

## 1.d compute all above in one function
def create_all_momentum_features(
    df: pd.DataFrame,
    base_col: str = "rate_per_mi_mean",
    periods: List[int] = [3, 7, 14, 30],
    time_col: str = 'pu_date'
) -> pd.DataFrame:
    """
    Create all momentum-related features.
    """
    # Create copy of input DataFrame and sort
    result = df.copy()
    if time_col in result.columns:
        result = result.sort_values(time_col)
    
    # Calculate and add each set of features
    momentum_features = calculate_momentum_features(result, base_col, periods, time_col)
    momentum_ratios = calculate_momentum_ratios(result, base_col, periods, time_col)
    trend_features = calculate_trend_reversal(result, base_col, time_col=time_col)
    
    # Add features to result DataFrame
    for col in momentum_features.columns:
        if col not in [time_col]:  # Skip time column to avoid duplication
            result[col] = momentum_features[col]
    for col in momentum_ratios.columns:
        if col not in [time_col]:
            result[col] = momentum_ratios[col]
    for col in trend_features.columns:
        if col not in [time_col]:
            result[col] = trend_features[col]
    
    return result
