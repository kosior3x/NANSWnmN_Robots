"""
SWARM v5.5 - Core AI Module Package (PRODUCTION READY)
=======================================================

Naprawiony Bio-Hybrid Navigation System:
- Dystans docelowy: 10cm (było 15cm)
- Cofanie używa połowy wiązki LIDAR
- Auto-save concept_graph co 100 kroków
- Persistence między sesjami
- Mniej paniki, bardziej konserwatywny

Modules:
    - swarm_core_v5_5: Główny kontroler (NAPRAWIONY)
    - swarm_core_v5_4: Stara wersja (backup)
"""

__version__ = "5.5.0"
__author__ = "SWARM Team"

from .swarm_core_v5_5 import SwarmCoreV55, SwarmConfig, Action

# Backward compatibility
SwarmCoreV54 = SwarmCoreV55

__all__ = [
    'SwarmCoreV55',
    'SwarmCoreV54',  # alias dla kompatybilności
    'SwarmConfig',
    'Action'
]
