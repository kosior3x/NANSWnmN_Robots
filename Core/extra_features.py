#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
SWARM v5.5 ULTIMATE - EXTRA FEATURES MODULE
=============================================================================

NOWE FUNKCJE (bonus od mnie!):
1. Debug Panel - szczegółowe statystyki real-time
2. Performance Monitor - FPS, RAM, CPU usage
3. Statistics Logger - export wyników do CSV
4. Heat Map Visualization - mapa gorących punktów
5. Quick Help - interaktywna pomoc na ekranie

UŻYCIE:
    from simulation.extra_features import DebugPanel, PerformanceMonitor, StatsLogger

=============================================================================
"""

import time
import psutil
import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

logger = logging.getLogger('ExtraFeatures')


# =============================================================================
# 1. DEBUG PANEL - Szczegółowe statystyki
# =============================================================================

class DebugPanel:
    """
    Panel debugowania z szczegółowymi statystykami.
    
    Pokazuje:
    - FPS, step count, time elapsed
    - Robot state (position, velocity, orientation)
    - AI state (current action, Q-values, epsilon)
    - Sensor readings (LIDAR min/max, US)
    - Spatial memory stats
    - Performance metrics
    """
    
    def __init__(self, screen_width: int = 300, screen_height: int = 400):
        self.enabled = False
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        if PYGAME_AVAILABLE:
            self.font = pygame.font.Font(None, 18)
            self.title_font = pygame.font.Font(None, 22)
        
        # Stats storage
        self.stats = {
            'fps': 0.0,
            'steps': 0,
            'time_elapsed': 0.0,
            'robot_x': 0.0,
            'robot_y': 0.0,
            'robot_theta': 0.0,
            'vel_l': 0.0,
            'vel_r': 0.0,
            'lidar_min': 0.0,
            'lidar_max': 0.0,
            'us_front': 0.0,
            'current_action': 'NONE',
            'explored_pct': 0.0,
            'visited_pct': 0.0,
            'frontier_count': 0,
            'goal_distance': 0.0,
        }
    
    def update(self, **kwargs):
        """Aktualizuj statystyki"""
        self.stats.update(kwargs)
    
    def draw(self, screen, x: int = 10, y: int = 10):
        """Rysuje panel debugowania"""
        if not self.enabled or not PYGAME_AVAILABLE:
            return
        
        # Tło panelu (semi-transparent)
        panel_surface = pygame.Surface((self.screen_width, self.screen_height))
        panel_surface.set_alpha(220)
        panel_surface.fill((40, 40, 40))
        screen.blit(panel_surface, (x, y))
        
        # Border
        pygame.draw.rect(screen, (100, 200, 100), (x, y, self.screen_width, self.screen_height), 2)
        
        # Title
        title = self.title_font.render("DEBUG PANEL", True, (255, 255, 100))
        screen.blit(title, (x + 10, y + 5))
        
        # Statystyki
        y_offset = y + 35
        line_height = 20
        
        sections = [
            ("PERFORMANCE", [
                f"FPS: {self.stats.get('fps', 0):.1f}",
                f"Steps: {self.stats.get('steps', 0)}",
                f"Time: {self.stats.get('time_elapsed', 0):.1f}s",
            ]),
            ("ROBOT STATE", [
                f"Pos: ({self.stats.get('robot_x', 0):.2f}, {self.stats.get('robot_y', 0):.2f})",
                f"Theta: {self.stats.get('robot_theta', 0):.2f} rad",
                f"Vel L/R: {self.stats.get('vel_l', 0):.2f}/{self.stats.get('vel_r', 0):.2f}",
            ]),
            ("SENSORS", [
                f"LIDAR: {self.stats.get('lidar_min', 0):.2f}-{self.stats.get('lidar_max', 0):.2f}m",
                f"US Front: {self.stats.get('us_front', 0):.2f}m",
            ]),
            ("AI STATE", [
                f"Action: {self.stats.get('current_action', 'NONE')}",
            ]),
            ("EXPLORATION", [
                f"Explored: {self.stats.get('explored_pct', 0):.1f}%",
                f"Visited: {self.stats.get('visited_pct', 0):.1f}%",
                f"Frontiers: {self.stats.get('frontier_count', 0)}",
                f"Goal Dist: {self.stats.get('goal_distance', 0):.2f}m",
            ]),
        ]
        
        for section_title, lines in sections:
            # Section title
            title_surf = self.font.render(section_title, True, (100, 200, 255))
            screen.blit(title_surf, (x + 10, y_offset))
            y_offset += line_height
            
            # Lines
            for line in lines:
                text_surf = self.font.render(line, True, (220, 220, 220))
                screen.blit(text_surf, (x + 20, y_offset))
                y_offset += line_height
            
            y_offset += 5  # Spacing between sections
    
    def toggle(self):
        """Przełącz widoczność panelu"""
        self.enabled = not self.enabled
        logger.info(f"Debug Panel: {'ON' if self.enabled else 'OFF'}")


# =============================================================================
# 2. PERFORMANCE MONITOR - Monitorowanie wydajności
# =============================================================================

class PerformanceMonitor:
    """
    Monitoruje wydajność systemu.
    
    Pokazuje:
    - FPS (real-time)
    - RAM usage
    - CPU usage
    - Frame time
    """
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        
        # Historia
        self.fps_history = []
        self.frame_time_history = []
        
        # Timers
        self.last_update_time = time.time()
        self.frame_count = 0
        self.current_fps = 0.0
        
        # System stats
        self.process = psutil.Process()
        self.ram_usage_mb = 0.0
        self.cpu_percent = 0.0
        
        if PYGAME_AVAILABLE:
            self.font = pygame.font.Font(None, 18)
    
    def start_frame(self):
        """Rozpocznij pomiar klatki"""
        self.frame_start_time = time.time()
    
    def end_frame(self):
        """Zakończ pomiar klatki"""
        frame_time = time.time() - self.frame_start_time
        self.frame_time_history.append(frame_time)
        
        if len(self.frame_time_history) > self.history_length:
            self.frame_time_history.pop(0)
        
        self.frame_count += 1
        
        # Update FPS co sekundę
        current_time = time.time()
        if current_time - self.last_update_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_update_time)
            self.fps_history.append(self.current_fps)
            
            if len(self.fps_history) > self.history_length:
                self.fps_history.pop(0)
            
            # Update system stats
            self.ram_usage_mb = self.process.memory_info().rss / 1024 / 1024
            self.cpu_percent = self.process.cpu_percent()
            
            self.frame_count = 0
            self.last_update_time = current_time
    
    def get_stats(self) -> Dict[str, float]:
        """Pobierz statystyki"""
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0.0
        avg_frame_time = np.mean(self.frame_time_history) if self.frame_time_history else 0.0
        
        return {
            'fps': self.current_fps,
            'avg_fps': avg_fps,
            'frame_time_ms': avg_frame_time * 1000,
            'ram_mb': self.ram_usage_mb,
            'cpu_percent': self.cpu_percent,
        }
    
    def draw_overlay(self, screen, x: int = 10, y: int = 10):
        """Rysuje overlay z wydajnością (mały, w rogu)"""
        if not PYGAME_AVAILABLE:
            return
        
        stats = self.get_stats()
        
        # Background
        bg = pygame.Surface((150, 80))
        bg.set_alpha(180)
        bg.fill((30, 30, 30))
        screen.blit(bg, (x, y))
        
        # Stats
        lines = [
            f"FPS: {stats['fps']:.1f}",
            f"Frame: {stats['frame_time_ms']:.1f}ms",
            f"RAM: {stats['ram_mb']:.0f}MB",
            f"CPU: {stats['cpu_percent']:.1f}%",
        ]
        
        y_offset = y + 5
        for line in lines:
            # Color based on value
            color = (220, 220, 220)
            if 'FPS' in line and stats['fps'] < 20:
                color = (255, 100, 100)  # Red if low FPS
            elif 'RAM' in line and stats['ram_mb'] > 200:
                color = (255, 200, 100)  # Orange if high RAM
            
            text = self.font.render(line, True, color)
            screen.blit(text, (x + 5, y_offset))
            y_offset += 18


# =============================================================================
# 3. STATISTICS LOGGER - Zapis statystyk do CSV
# =============================================================================

class StatsLogger:
    """
    Zapisuje statystyki symulacji do pliku CSV.
    
    Przydatne do:
    - Analiza wydajności AI
    - Porównanie różnych konfiguracji
    - Badania naukowe
    """
    
    def __init__(self, log_dir: str = "logs", auto_save_interval: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.auto_save_interval = auto_save_interval
        self.step_count = 0
        
        # Buffer danych
        self.data_buffer = []
        
        # Nazwa pliku
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.log_dir / f"swarm_stats_{timestamp}.csv"
        
        # Header
        self.fieldnames = [
            'step', 'time_elapsed', 'robot_x', 'robot_y', 'robot_theta',
            'vel_l', 'vel_r', 'lidar_min', 'us_front',
            'current_action', 'explored_pct', 'visited_pct',
            'frontier_count', 'goal_distance', 'fps'
        ]
        
        # Utwórz plik z nagłówkiem
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        
        logger.info(f"Stats logger initialized: {self.csv_file}")
    
    def log(self, **kwargs):
        """Zapisz pojedynczy wiersz statystyk"""
        self.step_count += 1
        
        # Dodaj do buffera
        row = {field: kwargs.get(field, 0) for field in self.fieldnames}
        row['step'] = self.step_count
        self.data_buffer.append(row)
        
        # Auto-save co N kroków
        if self.step_count % self.auto_save_interval == 0:
            self.flush()
    
    def flush(self):
        """Zapisz buffer do pliku"""
        if not self.data_buffer:
            return
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(self.data_buffer)
        
        logger.debug(f"Flushed {len(self.data_buffer)} rows to CSV")
        self.data_buffer.clear()
    
    def finalize(self):
        """Zapisz pozostałe dane i zamknij"""
        self.flush()
        logger.info(f"Stats saved to: {self.csv_file}")


# =============================================================================
# 4. HEAT MAP VISUALIZATION - Wizualizacja gorących punktów
# =============================================================================

class HeatMapVisualizer:
    """
    Tworzy heat mapę ruchu robota.
    
    Pokazuje:
    - Gdzie robot spędził najwięcej czasu (gorące punkty)
    - Gradient kolorów (niebieski→zielony→żółty→czerwony)
    """
    
    def __init__(self, width_m: float, height_m: float, resolution: float = 0.2):
        self.width_m = width_m
        self.height_m = height_m
        self.resolution = resolution
        
        self.grid_width = int(width_m / resolution)
        self.grid_height = int(height_m / resolution)
        
        # Heat map grid (licznik czasu w każdej komórce)
        self.heat_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        self.enabled = False
    
    def update(self, robot_x: float, robot_y: float, dt: float = 0.033):
        """Aktualizuj heat map na podstawie pozycji robota"""
        gx = int(robot_x / self.resolution)
        gy = int(robot_y / self.resolution)
        
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            self.heat_grid[gy, gx] += dt  # Dodaj czas spędzony
    
    def draw(self, screen, pixels_per_meter: int):
        """Rysuje heat map"""
        if not self.enabled or not PYGAME_AVAILABLE:
            return
        
        # Normalizuj do 0-1
        max_heat = np.max(self.heat_grid)
        if max_heat == 0:
            return
        
        normalized = self.heat_grid / max_heat
        
        # Rysuj
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                heat = normalized[gy, gx]
                
                if heat < 0.01:
                    continue
                
                # Gradient: niebieski → zielony → żółty → czerwony
                if heat < 0.33:
                    # Niebieski → zielony
                    t = heat / 0.33
                    color = (0, int(255 * t), int(255 * (1 - t)))
                elif heat < 0.66:
                    # Zielony → żółty
                    t = (heat - 0.33) / 0.33
                    color = (int(255 * t), 255, 0)
                else:
                    # Żółty → czerwony
                    t = (heat - 0.66) / 0.34
                    color = (255, int(255 * (1 - t)), 0)
                
                # Pozycja
                wx = (gx + 0.5) * self.resolution
                wy = (gy + 0.5) * self.resolution
                px = int(wx * pixels_per_meter)
                py = int(wy * pixels_per_meter)
                cell_size = int(self.resolution * pixels_per_meter)
                
                # Rysuj z alpha
                s = pygame.Surface((cell_size, cell_size))
                alpha = int(150 * heat)
                s.set_alpha(alpha)
                s.fill(color)
                screen.blit(s, (px, py))
    
    def toggle(self):
        """Przełącz widoczność"""
        self.enabled = not self.enabled
        logger.info(f"Heat Map: {'ON' if self.enabled else 'OFF'}")


# =============================================================================
# 5. QUICK HELP - Interaktywna pomoc
# =============================================================================

class QuickHelp:
    """
    Wyświetla szybką pomoc na ekranie.
    
    Pokazuje:
    - Skróty klawiszowe / przyciski dotykowe
    - Co oznaczają kolory
    - Jak interpretować statystyki
    """
    
    def __init__(self):
        self.visible = False
        
        if PYGAME_AVAILABLE:
            self.font = pygame.font.Font(None, 20)
            self.title_font = pygame.font.Font(None, 28)
        
        self.help_text = [
            ("PRZYCISKI DOTYKOWE", [
                "[Pause] - pauza/wznowienie",
                "[Reset] - nowa losowa mapa",
                "[Map] - occupancy grid",
                "[Visit] - visited areas",
                "[Front] - frontiers (żółte kropki)",
                "[Goal] - cel eksploracji (różowy krzyżyk)",
                "[LIDAR] - promienie LIDAR",
                "[Path] - ścieżka ruchu",
                "[Debug] - panel debugowania",
                "[Heat] - heat map",
            ]),
            ("KOLORY", [
                "Zielone = wolna przestrzeń (LIDAR OK)",
                "Czerwone = przeszkody (LIDAR wykrył)",
                "Niebieski = odwiedzone obszary",
                "Żółte kropki = frontiers (granice)",
                "Różowy krzyżyk = cel eksploracji",
                "Heat map: niebieski→zielony→żółty→czerwony",
            ]),
            ("STATYSTYKI", [
                "Explored % = procent mapy zbadany przez LIDAR",
                "Visited % = procent obszaru gdzie był robot",
                "Expl.Bias = waga exploration (1.0=max)",
                "FPS = klatki na sekundę",
            ]),
        ]
    
    def draw(self, screen):
        """Rysuje pomoc"""
        if not self.visible or not PYGAME_AVAILABLE:
            return
        
        # Semi-transparent background
        overlay = pygame.Surface(screen.get_size())
        overlay.set_alpha(230)
        overlay.fill((30, 30, 30))
        screen.blit(overlay, (0, 0))
        
        # Title
        title = self.title_font.render("QUICK HELP - Naciśnij H aby zamknąć", True, (255, 255, 100))
        screen.blit(title, (20, 20))
        
        # Sections
        y_offset = 60
        for section_title, lines in self.help_text:
            # Section title
            section_surf = self.title_font.render(section_title, True, (100, 200, 255))
            screen.blit(section_surf, (40, y_offset))
            y_offset += 35
            
            # Lines
            for line in lines:
                text_surf = self.font.render(line, True, (220, 220, 220))
                screen.blit(text_surf, (60, y_offset))
                y_offset += 25
            
            y_offset += 15
    
    def toggle(self):
        """Przełącz widoczność"""
        self.visible = not self.visible
        logger.info(f"Quick Help: {'ON' if self.visible else 'OFF'}")


# =============================================================================
# HELPER: Inicjalizacja wszystkich ekstra funkcji
# =============================================================================

def init_extra_features(config):
    """
    Inicjalizuje wszystkie ekstra funkcje.
    
    Returns:
        Dict z wszystkimi obiektami
    """
    features = {
        'debug_panel': DebugPanel(),
        'perf_monitor': PerformanceMonitor(),
        'stats_logger': StatsLogger(),
        'heat_map': HeatMapVisualizer(config.world_width, config.world_height),
        'quick_help': QuickHelp(),
    }
    
    logger.info("✨ Extra features initialized!")
    logger.info("  • Debug Panel (toggle with D)")
    logger.info("  • Performance Monitor (always on)")
    logger.info("  • Statistics Logger (auto-save to CSV)")
    logger.info("  • Heat Map (toggle with H)")
    logger.info("  • Quick Help (toggle with ?)")
    
    return features
