# ============================================================
# PLIK: frozen_maneuvers.py (NOWA WERSJA - ROZSZERZONA)
# ============================================================
"""
Zamrożone manewry dla systemu SWARM v5.5.
Manewry te są ładowane na starcie i stanowią bazę wiedzy instynktownej.
Mogą być łączone z wyuczonymi konceptami, ale nigdy nie są usuwane.
Flaga 'protected' oznacza, że manewr jest chroniony przed usunięciem,
ale może być używany do tworzenia nowych, połączonych konceptów.
"""

try:
    from swarm_core_v5_5 import Action
except ImportError:
    try:
        from swam.swarm_core_v5_5 import Action
    except ImportError:
        # Fallback: definicja zastępcza
        class Action:
            FORWARD = "FORWARD"
            TURN_LEFT = "TURN_LEFT"
            TURN_RIGHT = "TURN_RIGHT"
            SPIN_LEFT = "SPIN_LEFT"
            SPIN_RIGHT = "SPIN_RIGHT"
            REVERSE = "REVERSE"
            ESCAPE_MANEUVER = "ESCAPE_MANEUVER"
            STOP = "STOP"

FROZEN_MANEUVERS = [
    # ======================================================================
    # MANEWRY PODSTAWOWE (BAZOWE) - zawsze dostępne
    # ======================================================================
    {
        "name": "explore_straight",
        "sequence": [Action.FORWARD],
        "context": {"front_clr": (0.3, 1.0)},
        "protected": True,
        "usage_count_boost": 10000,
        "activation": 1.00
    },
    {
        "name": "core_turn_left_triple",
        "sequence": [Action.TURN_LEFT, Action.TURN_LEFT, Action.TURN_LEFT],
        "context": {"free_ang": (-90, -30)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.99
    },
    {
        "name": "core_turn_right_triple",
        "sequence": [Action.TURN_RIGHT, Action.TURN_RIGHT, Action.TURN_RIGHT],
        "context": {"free_ang": (30, 90)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.99
    },
    # ======================================================================
    # ORYGINALNE MANEWRY Z POPRZEDNIEJ WERSJI
    # ======================================================================
    {
        "name": "tight_corner_left",
        "sequence": [Action.TURN_LEFT] * 5,
        "context": {"front_clr": (0.0, 0.4)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "stagnation_breaker_left",
        "sequence": [Action.SPIN_LEFT] * 4 + [Action.FORWARD],
        "context": {"free_ang": (-30, 30), "front_clr": (0.5, 1.0)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "edge_follow_right",
        "sequence": [Action.TURN_RIGHT] * 4 + [Action.FORWARD],
        "context": {"us_right": (0.2, 0.8), "front_clr": (0.3, 0.8)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "deep_reverse_left",
        "sequence": [Action.REVERSE] * 4 + [Action.TURN_LEFT],
        "context": {"min_dist": (0.0, 0.3)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "escape_u_turn",
        "sequence": [Action.ESCAPE_MANEUVER] * 4 + [Action.TURN_RIGHT],
        "context": {"front_clr": (0.0, 0.2), "free_ang": (-10, 10)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "micro_adjust_left",
        "sequence": [Action.FORWARD] * 2 + [Action.TURN_LEFT] + [Action.FORWARD] * 2,
        "context": {"front_clr": (0.2, 0.6), "free_ang": (10, 30)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "precision_spin_left",
        "sequence": [Action.SPIN_LEFT] * 3 + [Action.FORWARD],
        "context": {"free_ang": (-20, 20), "front_clr": (0.6, 1.0)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    # ======================================================================
    # NOWE MANEWRY WYCIĄGNIĘTE Z PROCESU UCZENIA (Z LOGÓW)
    # ======================================================================
    {
        "name": "spin_then_turn_left",
        "sequence": [Action.SPIN_LEFT, Action.SPIN_LEFT, Action.TURN_LEFT],
        "context": {"free_ang": (-20, 20), "front_clr": (0.7, 1.0)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.97
    },
    {
        "name": "reverse_turn_right_double",
        "sequence": [Action.REVERSE, Action.TURN_RIGHT, Action.TURN_RIGHT],
        "context": {"front_clr": (0.0, 0.2), "min_dist": (0.0, 0.3)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.98
    },
    {
        "name": "spin_out_of_deadlock_right",
        "sequence": [Action.SPIN_RIGHT, Action.SPIN_RIGHT, Action.SPIN_RIGHT, Action.REVERSE, Action.REVERSE, Action.TURN_RIGHT],
        "context": {"min_dist": (0.0, 0.2), "free_ang": (-15, 15)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "complex_evasion_left",
        "sequence": [Action.TURN_LEFT, Action.TURN_LEFT, Action.REVERSE, Action.TURN_LEFT, Action.TURN_LEFT, Action.TURN_LEFT],
        "context": {"front_clr": (0.0, 0.3), "left_clr": (0.0, 0.4), "right_clr": (0.6, 1.0)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.96
    },
    {
        "name": "rhythmic_reverse_right",
        "sequence": [Action.REVERSE, Action.TURN_RIGHT, Action.REVERSE, Action.TURN_RIGHT, Action.TURN_RIGHT, Action.TURN_RIGHT],
        "context": {"front_clr": (0.0, 0.3), "right_clr": (0.2, 0.7)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "double_spin_left_adjust",
        "sequence": [Action.SPIN_LEFT, Action.SPIN_LEFT, Action.TURN_LEFT, Action.FORWARD],
        "context": {"free_ang": (-30, 10), "front_clr": (0.4, 0.8)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.96
    },
    {
        "name": "deep_reverse_sequence_right",
        "sequence": [Action.REVERSE, Action.TURN_RIGHT, Action.TURN_RIGHT, Action.TURN_RIGHT, Action.TURN_RIGHT, Action.TURN_RIGHT],
        "context": {"front_clr": (0.0, 0.25), "right_clr": (0.3, 0.9)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "reverse_spin_combo_right",
        "sequence": [Action.REVERSE, Action.REVERSE, Action.TURN_RIGHT, Action.SPIN_RIGHT, Action.SPIN_RIGHT, Action.TURN_RIGHT],
        "context": {"min_dist": (0.0, 0.3), "right_clr": (0.1, 0.5)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.94
    },
    {
        "name": "aggressive_left_breakout",
        "sequence": [Action.TURN_LEFT, Action.SPIN_LEFT, Action.TURN_LEFT, Action.TURN_LEFT, Action.TURN_LEFT, Action.TURN_LEFT],
        "context": {"left_clr": (0.0, 0.3), "front_clr": (0.2, 0.6)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "aggressive_right_breakout",
        "sequence": [Action.TURN_RIGHT, Action.SPIN_RIGHT, Action.SPIN_RIGHT, Action.TURN_RIGHT, Action.TURN_RIGHT, Action.TURN_RIGHT],
        "context": {"right_clr": (0.0, 0.3), "front_clr": (0.2, 0.6)},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.94
    },
    # ======================================================================
    # KONSOLIDOWANE ZŁOŻONE MANEWRY
    # ======================================================================
    {
        "name": "consolidated_evasion_pattern",
        "sequence": [Action.SPIN_LEFT, Action.TURN_LEFT, Action.REVERSE, Action.SPIN_RIGHT, Action.TURN_RIGHT, Action.FORWARD],
        "context": {"min_dist": (0.0, 0.4), "free_ang": (-45, 45)},
        "protected": True,
        "usage_count_boost": 2000,
        "activation": 0.98
    },
    {
        "name": "consolidated_left_priority",
        "sequence": [Action.SPIN_LEFT, Action.SPIN_LEFT, Action.TURN_LEFT, Action.REVERSE, Action.TURN_LEFT, Action.FORWARD],
        "context": {"left_clr": (0.1, 0.5), "front_clr": (0.0, 0.3)},
        "protected": True,
        "usage_count_boost": 1500,
        "activation": 0.96
    },
    {
        "name": "consolidated_right_priority",
        "sequence": [Action.SPIN_RIGHT, Action.SPIN_RIGHT, Action.TURN_RIGHT, Action.REVERSE, Action.TURN_RIGHT, Action.FORWARD],
        "context": {"right_clr": (0.1, 0.5), "front_clr": (0.0, 0.3)},
        "protected": True,
        "usage_count_boost": 1500,
        "activation": 0.96
    }
,    # =========================================================================
    # NOWE MANEWRY (dodane: 2026-02-27)
    # =========================================================================

    # --- ZŁOTE MANEWRY (wysoka prędkość, płynna nawigacja) ---
    {
        "name": "optimized_straight_bias_1",
        "sequence": ["FORWARD", "FORWARD", "FORWARD", "FORWARD", "FORWARD", "TURN_RIGHT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "optimized_straight_bias_2",
        "sequence": ["FORWARD", "FORWARD", "FORWARD", "FORWARD", "TURN_RIGHT", "TURN_RIGHT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "optimized_straight_bias_3",
        "sequence": ["FORWARD", "FORWARD", "FORWARD", "TURN_RIGHT", "TURN_RIGHT", "TURN_RIGHT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "optimized_straight_bias_4",
        "sequence": ["FORWARD", "FORWARD", "TURN_RIGHT", "TURN_RIGHT", "TURN_RIGHT", "TURN_RIGHT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "optimized_straight_bias_5",
        "sequence": ["FORWARD", "TURN_RIGHT", "TURN_RIGHT", "TURN_RIGHT", "TURN_RIGHT", "TURN_RIGHT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },

    # --- Precyzyjne skręty krawędziowe ---
    {
        "name": "precise_edge_turns_1",
        "sequence": ["TURN_RIGHT", "TURN_RIGHT", "TURN_RIGHT", "TURN_RIGHT", "TURN_RIGHT", "TURN_LEFT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "precise_edge_turns_2",
        "sequence": ["TURN_LEFT", "TURN_LEFT", "TURN_LEFT", "TURN_LEFT", "TURN_LEFT", "FORWARD"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "edge_corridor_stabilization",
        "sequence": ["FORWARD", "TURN_RIGHT", "TURN_RIGHT", "TURN_RIGHT", "TURN_RIGHT", "FORWARD"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },

    # --- Skuteczne wyjścia ze stagnacji ---
    {
        "name": "recovery_success_1",
        "sequence": ["SPIN_RIGHT", "SPIN_RIGHT", "SPIN_RIGHT", "SPIN_RIGHT", "TURN_LEFT", "TURN_LEFT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "recovery_success_2",
        "sequence": ["SPIN_RIGHT", "SPIN_RIGHT", "SPIN_RIGHT", "TURN_LEFT", "TURN_LEFT", "TURN_LEFT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "recovery_success_3",
        "sequence": ["SPIN_RIGHT", "SPIN_RIGHT", "TURN_LEFT", "TURN_LEFT", "TURN_LEFT", "TURN_LEFT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "recovery_success_4",
        "sequence": ["SPIN_RIGHT", "TURN_LEFT", "TURN_LEFT", "TURN_LEFT", "TURN_LEFT", "TURN_LEFT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },

    # --- Heurystyki pełnego pokrycia labiryntu ---
    {
        "name": "coverage_master_1",
        "sequence": ["FORWARD", "FORWARD", "TURN_RIGHT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "coverage_master_2",
        "sequence": ["STOP", "TURN_LEFT", "TURN_LEFT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "coverage_master_3",
        "sequence": ["ESCAPE_MANEUVER", "FORWARD", "FORWARD", "FORWARD", "FORWARD", "FORWARD", "FORWARD"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },

    # --- SKONSOLIDOWANE MANEWRY RATUNKOWE (z łączenia konceptów) ---
    {
        "name": "merged_complex_evasion_left",
        "sequence": ["TURN_LEFT", "TURN_LEFT", "FORWARD", "TURN_LEFT", "FORWARD"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "stagnation_precision_breaker",
        "sequence": ["SPIN_LEFT", "SPIN_LEFT", "TURN_LEFT", "FORWARD", "FORWARD"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "deep_rhythmic_reverse",
        "sequence": ["REVERSE", "REVERSE", "TURN_RIGHT", "REVERSE", "TURN_RIGHT"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    },
    {
        "name": "hard_reflex_recovery_flow",
        "sequence": ["ESCAPE_MANEUVER", "ESCAPE_MANEUVER", "REVERSE", "TURN_RIGHT", "FORWARD"],
        "context": {},
        "protected": True,
        "usage_count_boost": 1000,
        "activation": 0.95
    }

]
# ============================================================
# FUNKCJE POMOCNICZE DLA ŁADOWANIA (OPCJONALNIE)
# ============================================================
def get_all_maneuvers():
    return FROZEN_MANEUVERS

def get_maneuver_names():
    return [m["name"] for m in FROZEN_MANEUVERS]
