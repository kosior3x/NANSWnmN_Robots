"""
SWARM v5.5 - Simulation Module Package
=======================================

2D Physics Simulator with Spatial Memory and Exploration.

NEW IN v5.5:
    - Spatial Memory (occupancy grid + visited map)
    - Frontier-based Exploration
    - LIDAR 3m range (down from 5m)
    - Touch interface for Android
    - Random map generation

Modules:
    - swarm_simulator: Główny symulator v5.5
    - spatial_memory: Pamięć przestrzenna + eksploracja
"""

__version__ = "5.5.0"

from .swarm_simulator import SwarmSimulator, SimConfig
from .spatial_memory import SpatialMemory

__all__ = [
    'SwarmSimulator',
    'SimConfig',
    'SpatialMemory'
]
