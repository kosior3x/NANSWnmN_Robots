#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attractors.py – Zbiór atraktorów chaotycznych dla Swarm Core.
Zawiera klasy:
- LorenzAttractor
- RosslerAttractor
- DoubleScrollAttractor
"""

import numpy as np
from typing import Dict, Tuple

# Wspólna klasa bazowa (opcjonalna, ale ułatwia interfejs)
class BaseAttractor:
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self):
        """Resetuje stan atraktora do wartości początkowych."""
        raise NotImplementedError
    
    def step(self):
        """Wykonuje jeden krok ewolucji atraktora."""
        raise NotImplementedError
    
    def get_state(self) -> Dict[str, float]:
        """Zwraca pełny stan atraktora jako słownik."""
        raise NotImplementedError

# ----------------------------------------------------------------------
# 1. ATRAKTOR LORENZA
# ----------------------------------------------------------------------

class LorenzAttractor(BaseAttractor):
    """
    Atraktor Lorenza.
    Równania:
        dx/dt = sigma*(y - x)
        dy/dt = x*(rho - z) - y
        dz/dt = x*y - beta*z
    """
    def __init__(self, config):
        super().__init__(config)
        # Pobierz parametry z configu (zakładamy, że config ma te atrybuty)
        self.sigma = config.LORENZ_SIGMA
        self.rho = config.LORENZ_RHO
        self.beta = config.LORENZ_BETA
        self.dt = config.LORENZ_DT
        self.reset()
    
    def reset(self):
        self.x, self.y, self.z = 0.1, 0.0, 0.0
        self.last_x, self.last_y, self.last_z = self.x, self.y, self.z
        self.dx, self.dy, self.dz = 0.0, 0.0, 0.0
        self.x_norm, self.z_norm = 0.0, 0.0
    
    def step(self):
        # zachowaj poprzednie wartości
        self.last_x, self.last_y, self.last_z = self.x, self.y, self.z
        
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z
        
        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt
        
        # pochodne (różnice)
        self.dx = (self.x - self.last_x) / self.dt
        self.dy = (self.y - self.last_y) / self.dt
        self.dz = (self.z - self.last_z) / self.dt
        
        # znormalizowane wartości (dla wygody)
        self.x_norm = np.tanh(self.x / 15.0)
        self.z_norm = min(1.0, max(0.0, self.z / 40.0))
    
    def get_state(self) -> Dict[str, float]:
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'dx': self.dx, 'dy': self.dy, 'dz': self.dz,
            'x_norm': self.x_norm, 'z_norm': self.z_norm
        }

# ----------------------------------------------------------------------
# 2. ATRAKTOR ROSSLERA
# ----------------------------------------------------------------------

class RosslerAttractor(BaseAttractor):
    """
    Atraktor Rosslera.
    Równania:
        dx/dt = -y - z
        dy/dt = x + a*y
        dz/dt = b + z*(x - c)
    Typowe wartości: a=0.2, b=0.2, c=5.7
    """
    def __init__(self, config):
        super().__init__(config)
        # Parametry – można je dodać do configu lub ustawić domyślnie
        self.a = getattr(config, 'ROSSLER_A', 0.2)
        self.b = getattr(config, 'ROSSLER_B', 0.2)
        self.c = getattr(config, 'ROSSLER_C', 5.7)
        self.dt = getattr(config, 'ROSSLER_DT', 0.01)
        self.reset()
    
    def reset(self):
        self.x, self.y, self.z = 0.1, 0.0, 0.0
        self.last_x, self.last_y, self.last_z = self.x, self.y, self.z
        self.dx, self.dy, self.dz = 0.0, 0.0, 0.0
        # opcjonalnie normy, jeśli potrzebne
        self.x_norm = 0.0
        self.z_norm = 0.0
    
    def step(self):
        self.last_x, self.last_y, self.last_z = self.x, self.y, self.z
        
        dx = -self.y - self.z
        dy = self.x + self.a * self.y
        dz = self.b + self.z * (self.x - self.c)
        
        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt
        
        self.dx = (self.x - self.last_x) / self.dt
        self.dy = (self.y - self.last_y) / self.dt
        self.dz = (self.z - self.last_z) / self.dt
        
        # opcjonalne normalizacje – można dodać według potrzeb
        self.x_norm = np.tanh(self.x / 10.0)   # przykład
        self.z_norm = np.tanh(self.z / 10.0)
    
    def get_state(self) -> Dict[str, float]:
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'dx': self.dx, 'dy': self.dy, 'dz': self.dz,
            'x_norm': self.x_norm, 'z_norm': self.z_norm
        }

# ----------------------------------------------------------------------
# 3. ATRAKTOR DOUBLE-SCROLL (np. układ Chuy z parametrami dającymi double-scroll)
# ----------------------------------------------------------------------

class DoubleScrollAttractor(BaseAttractor):
    """
    Atraktor double-scroll (np. układ Chuy).
    Równania:
        dx/dt = α*(y - x - f(x))
        dy/dt = x - y + z
        dz/dt = -β*y
    gdzie f(x) jest nieliniowością, np. f(x) = m1*x + 0.5*(m0-m1)*(|x+1|-|x-1|)
    """
    def __init__(self, config):
        super().__init__(config)
        # Parametry
        self.alpha = getattr(config, 'DOUBLESCROLL_ALPHA', 9.0)
        self.beta = getattr(config, 'DOUBLESCROLL_BETA', 14.286)
        self.m0 = getattr(config, 'DOUBLESCROLL_M0', -1/7)
        self.m1 = getattr(config, 'DOUBLESCROLL_M1', 2/7)
        self.dt = getattr(config, 'DOUBLESCROLL_DT', 0.01)
        self.reset()
    
    def _f(self, x):
        """Nieliniowość dla atraktora Chuy."""
        return self.m1*x + 0.5*(self.m0 - self.m1)*(abs(x+1) - abs(x-1))
    
    def reset(self):
        self.x, self.y, self.z = 0.1, 0.0, 0.0
        self.last_x, self.last_y, self.last_z = self.x, self.y, self.z
        self.dx, self.dy, self.dz = 0.0, 0.0, 0.0
        self.x_norm, self.z_norm = 0.0, 0.0
    
    def step(self):
        self.last_x, self.last_y, self.last_z = self.x, self.y, self.z
        
        dx = self.alpha * (self.y - self.x - self._f(self.x))
        dy = self.x - self.y + self.z
        dz = -self.beta * self.y
        
        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt
        
        self.dx = (self.x - self.last_x) / self.dt
        self.dy = (self.y - self.last_y) / self.dt
        self.dz = (self.z - self.last_z) / self.dt
        
        # opcjonalne normalizacje
        self.x_norm = np.tanh(self.x / 5.0)
        self.z_norm = np.tanh(self.z / 5.0)
    
    def get_state(self) -> Dict[str, float]:
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'dx': self.dx, 'dy': self.dy, 'dz': self.dz,
            'x_norm': self.x_norm, 'z_norm': self.z_norm
        }

# ----------------------------------------------------------------------
# SŁOWNIK DOSTĘPNYCH ATRAKTORÓW (ułatwia tworzenie instancji)
# ----------------------------------------------------------------------

ATTRACTORS = {
    'lorenz': LorenzAttractor,
    'rossler': RosslerAttractor,
    'double_scroll': DoubleScrollAttractor,
}
