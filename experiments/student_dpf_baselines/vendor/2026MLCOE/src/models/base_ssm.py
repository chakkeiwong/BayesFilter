"""
Base SSM Module
===============
Defines the Abstract Base Class for all State Space Models to ensure
consistency in the filtering pipeline.

Author: Joowon Lee
Date: 2026-02-26
"""

from abc import ABC, abstractmethod
import tensorflow as tf

class BaseSSM(ABC):
    """
    Abstract Base Class for State Space Models (SSMs).

    This class defines the interface for probabilistic sequence modeling.
    It separates the latent dynamics (transition) from the observable
    process (emission). All subclasses must implement these methods to
    be compatible with the particle filter and particle flow pipelines.
    """

    @abstractmethod
    def transition_map(self, z_prev, s_prev=None):
        """
        Computes the parameters of the transition distribution p(z_t | z_{t-1}).

        In classical models, this typically returns a mean and a covariance.
        In Neural SSMs, s_prev represents the hidden state of the RNN/LSTM
        controlling the non-Markovian dynamics.

        Args:
            z_prev: Latent state at time t-1.
            s_prev: (Optional) Deterministic internal state (e.g., LSTM cell state).

        Returns:
            A tuple containing (parameters_for_z, next_deterministic_state).
        """
        pass

    @abstractmethod
    def emission_map(self, z_curr):
        """
        Computes the parameters of the emission distribution p(x_t | z_t).

        Args:
            z_curr: The current latent state at time t.

        Returns:
            A tuple containing (mean, covariance/scale) of the observation.
        """
        pass

    @abstractmethod
    def get_initial_state(self, batch_size):
        """
        Initializes the state of the system at t=0.

        Args:
            batch_size: Number of parallel sequences (particles).

        Returns:
            Initial latent state z_0 and initial deterministic state s_0.
        """
        pass
