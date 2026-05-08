"""
Neural SSM Module
=================
Implements state space models with neural-network parameterized transitions.

Author: Joowon Lee
Date: 2026-02-26
"""

import tensorflow as tf
from .base_ssm import BaseSSM

class StateSpaceLSTM(BaseSSM):
    """
    State-Space LSTM (SSL) model for complex sequential dependencies.
    Bakes a neural network into the transition logic to handle
    non-Markovian dynamics as described in Zheng(17).
    """
    def __init__(self, lstm_cell, transition_net, emission_net):
        """
        Args:
            lstm_cell: A tf.keras.layers.LSTMCell for deterministic state tracking.
            transition_net: Neural network producing parameters for latent z_t.
            emission_net: Neural network producing parameters for observation x_t.
        """
        self.lstm_cell = lstm_cell
        self.transition_net = transition_net
        self.emission_net = emission_net

    def transition_map(self, z_prev, s_prev):
        """
        The transition is split into two phases:
        1. A deterministic LSTM update incorporating the previous latent sample.
        2. A stochastic projection from the LSTM state to the new latent distribution.
        """
        # Step 1: Update the internal memory (s_t) using the previous state (s_{t-1})
        # and the previous latent variable (z_{t-1}).
        # This allows long-range dependencies to influence the next transition.
        s_curr, _ = self.lstm_cell(z_prev, s_prev)

        # Step 2: Use the current LSTM hidden state to generate the
        # parameters (e.g., mean/cov) for the next latent state z_t.
        z_params = self.transition_net(s_curr)

        return z_params, s_curr

    def emission_map(self, z_curr):
        """
        Probabilistic emission h(z_curr) modeled by a neural network.
        Maps the latent dynamic state to the distribution of noisy indicators.
        """
        return self.emission_net(z_curr)

    def get_initial_state(self, batch_size):
        """
        Zero-initializes both the LSTM hidden/cell states and the latent state.
        """
        s_0 = self.lstm_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        # Assumes latent dimension matches the LSTM input requirement
        z_0 = tf.zeros([batch_size, self.lstm_cell.input_size])
        return z_0, s_0
