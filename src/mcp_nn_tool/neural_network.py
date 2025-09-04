"""Neural network model definitions for the MCP NN Tool.

This module contains the neural network architecture and data loading utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MLPregression(nn.Module):
    """Multi-layer perceptron for regression tasks.
    
    A fully connected neural network with configurable layers for regression.
    Supports both single and multi-target predictions.
    
    Args:
        params (dict): Dictionary containing model hyperparameters:
            - layber_number (int): Number of hidden layers
            - unit (int): Number of units in hidden layers  
            - drop_out (float): Dropout probability
        feature_number (int): Number of input features
        target_number (int): Number of target outputs (default: 1)
    """
    
    def __init__(self, params: dict, feature_number: int, target_number: int = 1):
        """Initialize the MLP regression model.
        
        Args:
            params: Dictionary containing hyperparameters
            feature_number: Number of input features
            target_number: Number of target outputs
        """
        super(MLPregression, self).__init__()
        self.layber_number = params["layber_number"]
        self.unit = params["unit"]
        self.target_number = target_number
        
        # First hidden layer
        self.hidden1 = nn.Linear(in_features=feature_number, out_features=self.unit, bias=True)
        # Second hidden layer (for additional layers)
        self.hidden2 = nn.Linear(self.unit, self.unit)
        # Output layers
        self.hidden5 = nn.Linear(self.unit, 64)
        self.predict = nn.Linear(64, target_number)  # Support multiple targets
        self.relu = nn.functional.relu
        self.dropout = nn.Dropout(params["drop_out"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, feature_number)
            
        Returns:
            Output tensor of shape (batch_size, target_number) for multi-target
            or (batch_size,) for single target
        """
        x = self.relu(self.hidden1(x))
        for i in range(self.layber_number):
            x = self.dropout(self.relu(self.hidden2(x)))
        x = self.dropout(self.relu(self.hidden5(x)))
        output = self.predict(x)
        
        # For single target, return 1D tensor for backward compatibility
        if self.target_number == 1:
            return output[:, 0]
        else:
            return output


class MLPClassification(nn.Module):
    """Multi-layer perceptron for classification tasks.
    
    A fully connected neural network with configurable layers for classification.
    Supports both binary and multi-class classification.
    
    Args:
        params (dict): Dictionary containing model hyperparameters:
            - layber_number (int): Number of hidden layers
            - unit (int): Number of units in hidden layers  
            - drop_out (float): Dropout probability
        feature_number (int): Number of input features
        num_classes (int): Number of output classes
    """
    
    def __init__(self, params: dict, feature_number: int, num_classes: int):
        """Initialize the MLP classification model.
        
        Args:
            params: Dictionary containing hyperparameters
            feature_number: Number of input features
            num_classes: Number of output classes
        """
        super(MLPClassification, self).__init__()
        self.layber_number = params["layber_number"]
        self.unit = params["unit"]
        self.num_classes = num_classes
        
        # First hidden layer
        self.hidden1 = nn.Linear(in_features=feature_number, out_features=self.unit, bias=True)
        # Second hidden layer (for additional layers)
        self.hidden2 = nn.Linear(self.unit, self.unit)
        # Output layers
        self.hidden5 = nn.Linear(self.unit, 64)
        # For binary classification, output 1 logit; for multi-class, output num_classes logits
        output_size = 1 if num_classes == 2 else num_classes
        self.predict = nn.Linear(64, output_size)
        self.relu = nn.functional.relu
        self.dropout = nn.Dropout(params["drop_out"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, feature_number)
            
        Returns:
            Output tensor of shape (batch_size, num_classes) containing logits
        """
        x = self.relu(self.hidden1(x))
        for i in range(self.layber_number):
            x = self.dropout(self.relu(self.hidden2(x)))
        x = self.dropout(self.relu(self.hidden5(x)))
        logits = self.predict(x)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, feature_number)
            
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        if self.num_classes == 2:
            # Binary classification - use sigmoid, output single probability
            return torch.sigmoid(logits).squeeze()
        else:
            # Multi-class classification - use softmax
            return F.softmax(logits, dim=1)
    
    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices.
        
        Args:
            x: Input tensor of shape (batch_size, feature_number)
            
        Returns:
            Class prediction tensor of shape (batch_size,)
        """
        if self.num_classes == 2:
            # Binary classification
            probs = self.predict_proba(x)
            return (probs > 0.5).long()
        else:
            # Multi-class classification
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


class FastTensorDataLoader:
    """Fast DataLoader-like object for tensors.
    
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    
    def __init__(self, *tensors: torch.Tensor, batch_size: int = 32, shuffle: bool = False):
        """Initialize a FastTensorDataLoader.
        
        Args:
            *tensors: Tensors to store. Must have the same length at dim 0.
            batch_size: Batch size to load.
            shuffle: If True, shuffle the data in-place whenever an iterator 
                    is created out of this object.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors), \
            "All tensors must have the same length at dimension 0"
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate number of batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        
    def __iter__(self):
        """Create an iterator for the dataloader."""
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, ...]:
        """Get the next batch of data.
        
        Returns:
            Tuple of tensors representing a batch
            
        Raises:
            StopIteration: When all data has been processed
        """
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self) -> int:
        """Get the number of batches."""
        return self.n_batches 