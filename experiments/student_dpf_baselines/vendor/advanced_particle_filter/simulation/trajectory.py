"""
Trajectory simulation and storage.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from numpy.random import Generator, default_rng

from ..models.base import StateSpaceModel


@dataclass
class Trajectory:
    """
    Container for simulated or recorded trajectory data.
    
    Attributes:
        states: [T+1, nx] State trajectory (x_0, x_1, ..., x_T)
        observations: [T, ny] Observations (y_1, y_2, ..., y_T)
        T: Number of time steps
        state_dim: State dimension
        obs_dim: Observation dimension
        metadata: Optional dictionary for additional info
    """
    states: np.ndarray
    observations: np.ndarray
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def T(self) -> int:
        """Number of time steps."""
        return self.observations.shape[0]
    
    @property
    def state_dim(self) -> int:
        """State dimension."""
        return self.states.shape[1]
    
    @property
    def obs_dim(self) -> int:
        """Observation dimension."""
        return self.observations.shape[1]
    
    def subset(self, start: int, end: int) -> "Trajectory":
        """
        Extract a subset of the trajectory.
        
        Args:
            start: Start time index (inclusive)
            end: End time index (exclusive)
            
        Returns:
            New Trajectory with subset of data
        """
        return Trajectory(
            states=self.states[start:end+1].copy(),
            observations=self.observations[start:end].copy(),
            metadata=self.metadata,
        )
    
    def save(self, path: str):
        """Save trajectory to .npz file."""
        np.savez(
            path,
            states=self.states,
            observations=self.observations,
            metadata=self.metadata,
        )
    
    @classmethod
    def load(cls, path: str) -> "Trajectory":
        """Load trajectory from .npz file."""
        data = np.load(path, allow_pickle=True)
        metadata = data['metadata'].item() if 'metadata' in data else None
        return cls(
            states=data['states'],
            observations=data['observations'],
            metadata=metadata,
        )


def simulate(
    model: StateSpaceModel,
    T: int,
    seed: Optional[int] = None,
    rng: Optional[Generator] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Trajectory:
    """
    Simulate a trajectory from a state space model.
    
    Args:
        model: StateSpaceModel instance
        T: Number of time steps
        seed: Random seed (ignored if rng is provided)
        rng: NumPy random generator (optional)
        metadata: Optional metadata to attach
        
    Returns:
        Trajectory object
    """
    if rng is None:
        rng = default_rng(seed)
    
    states, observations = model.simulate(T, rng)
    
    return Trajectory(
        states=states,
        observations=observations,
        metadata=metadata,
    )


def simulate_batch(
    model: StateSpaceModel,
    T: int,
    n_trajectories: int,
    seed: Optional[int] = None,
) -> list:
    """
    Simulate multiple independent trajectories.
    
    Args:
        model: StateSpaceModel instance
        T: Number of time steps
        n_trajectories: Number of trajectories to simulate
        seed: Random seed
        
    Returns:
        List of Trajectory objects
    """
    rng = default_rng(seed)
    
    trajectories = []
    for i in range(n_trajectories):
        traj = simulate(model, T, rng=rng, metadata={'trajectory_idx': i})
        trajectories.append(traj)
    
    return trajectories
