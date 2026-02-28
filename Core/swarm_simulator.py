#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
SWARM SIMULATOR v5.5 - FULL INTEGRATION
=============================================================================

NOWE FUNKCJE v5.5:
1. ✅ Spatial Memory - buduje mapę z LIDAR 360°
2. ✅ Exploration Strategy - celuje w niezbadane obszary
3. ✅ LIDAR 3m - zmieniony z 5m
4. ✅ Interfejs dotykowy - przyciski zamiast klawiszy (Android!)
5. ✅ Losowa mapa - zmienia się co uruchomienie
6. ✅ Encodery - pełna integracja
7. ✅ Path planning - aktywne szukanie trasy

UŻYCIE:
    python android_launcher.py → opcja 1 (symulacja)

STEROWANIE (DOTYK/PRZYCISKI):
    Play/Pause - pauza
    Reset - reset symulacji
    Map/Visit/Front/Goal - toggle wizualizacji

=============================================================================
"""
import os
if os.name != 'nt':
    os.environ['SDL_VIDEODRIVER'] = 'android'
import numpy as np
import math
import time
import random
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Pygame (opcjonalne - dla wizualizacji)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("Pygame not available - running headless")

# SWARM core
try:
    from core.swarm_core_v5_5 import SwarmCoreV55
except ImportError:
    from core import SwarmCoreV55

# Spatial Memory
try:
    from simulation.spatial_memory import SpatialMemory
    SPATIAL_MEMORY_AVAILABLE = True
except ImportError:
    try:
        from spatial_memory import SpatialMemory
        SPATIAL_MEMORY_AVAILABLE = True
    except ImportError:
        SPATIAL_MEMORY_AVAILABLE = False
        logging.warning("SpatialMemory not available - exploration disabled")

# Extra Features (BONUS!)
try:
    from simulation.extra_features import init_extra_features
    EXTRA_FEATURES_AVAILABLE = True
except ImportError:
    try:
        from extra_features import init_extra_features
        EXTRA_FEATURES_AVAILABLE = True
    except ImportError:
        EXTRA_FEATURES_AVAILABLE = False
        logging.warning("Extra features not available")

logger = logging.getLogger('Simulator')

# ─── LOGGING SETUP ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-7s] [%(name)-12s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SimConfig:
    """Konfiguracja symulatora v5.5"""
    
    # WORLD
    world_width: float = 15.0  # metry (użytkownik: 11m)
    world_height: float = 12.0  # metry (użytkownik: 16m)
    pixels_per_meter: int = 80  # Rozdzielczość (zmniejszona dla wydajności)
    
    # ROBOT PHYSICS (wymiary fizyczne: 28x28cm body, 32cm wheelspan)
    robot_radius: float = 0.16  # metry (połowa rozstawu kół 32cm)
    robot_mass: float = 2.0  # kg
    wheel_base: float = 0.32  # metry (rozstaw kół)
    max_wheel_speed: float = 0.5  # m/s
    
    # FRICTION & DYNAMICS
    friction_coefficient: float = 0.8
    slip_threshold: float = 0.95
    inertia_coefficient: float = 0.3
    
    # SENSORS
    lidar_range: float = 3.0  # metry (ZMIENIONE Z 5m NA 3m!)
    lidar_rays: int = 36  # Liczba promieni (360° / 10° = 36)
    lidar_noise_std: float = 0.02  # Szum
    
    ultrasonic_range: float = 3.0  # metry
    ultrasonic_cone_angle: float = 15.0  # stopnie
    
    # SIMULATION
    dt: float = 1.0 / 30.0  # 30 Hz
    max_steps: int = 0  # 0 = bez limitu (symulacja trwa do ręcznego zakończenia)
    
    # VISUALIZATION
    visualize: bool = True
    fps: int = 30
    
    # MAP - LOSOWA CO URUCHOMIENIE!
    map_type: str = "random"  # "labyrinth", "corridor", "empty", "random"
    random_start: bool = True  # Losowa pozycja startowa
    seed: Optional[int] = None
    
    # SPATIAL MEMORY
    enable_spatial_memory: bool = True
    
    # Persistence (v5.5 NOWE)
    save_state_file: str = "simulator_state_v5_5.pkl"
    auto_save_interval: int = 500  # Zapis co 500 kroków
    occupancy_resolution: float = 0.1  # 10cm per cell
    visited_resolution: float = 0.2  # 20cm per cell


# =============================================================================
# OBSTACLE & MAP
# =============================================================================

class Obstacle:
    """Pojedyncza przeszkoda (prostokąt)"""
    
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def contains_point(self, px: float, py: float) -> bool:
        """Czy punkt jest wewnątrz przeszkody"""
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)
    
    def intersects_circle(self, cx: float, cy: float, radius: float) -> bool:
        """Czy okrąg (robot) koliduje z przeszkodą"""
        # Znajdź najbliższy punkt na prostokącie
        closest_x = max(self.x, min(cx, self.x + self.width))
        closest_y = max(self.y, min(cy, self.y + self.height))
        
        # Dystans od środka okręgu
        dx = cx - closest_x
        dy = cy - closest_y
        
        return (dx * dx + dy * dy) < (radius * radius)


class MapGenerator:
    """Generator map - LOSOWA CO URUCHOMIENIE!"""
    
    def __init__(self, width: float, height: float, map_type: str = "random"):
        self.width = width
        self.height = height
        self.map_type = map_type
        
        # Jeśli random, wybierz losowy typ
        if self.map_type == "random":
            self.map_type = random.choice(['labyrinth', 'corridor', 'empty', 'maze'])
            logger.info(f"Random map selected: {self.map_type}")
    
    def generate(self) -> List[Obstacle]:
        """Generuje mapę na podstawie typu"""
        if self.map_type == "labyrinth":
            return self._generate_labyrinth()
        elif self.map_type == "corridor":
            return self._generate_corridor()
        elif self.map_type == "maze":
            return self._generate_maze()
        elif self.map_type == "empty":
            return self._generate_empty()
        else:
            return self._generate_labyrinth()
    
    def _generate_empty(self) -> List[Obstacle]:
        """Pusta arena (tylko ściany)"""
        obstacles = []
        wall_thickness = 0.2
        
        # Ściany zewnętrzne
        obstacles.append(Obstacle(0, 0, self.width, wall_thickness))  # Dół
        obstacles.append(Obstacle(0, self.height - wall_thickness, self.width, wall_thickness))  # Góra
        obstacles.append(Obstacle(0, 0, wall_thickness, self.height))  # Lewo
        obstacles.append(Obstacle(self.width - wall_thickness, 0, wall_thickness, self.height))  # Prawo
        
        return obstacles
    
    def _generate_corridor(self) -> List[Obstacle]:
        """Korytarz z przeszkodami"""
        obstacles = self._generate_empty()  # Start ze ścianami
        
        # Dodaj losowe przeszkody w korytarzu
        num_obstacles = random.randint(5, 10)
        
        for _ in range(num_obstacles):
            w = random.uniform(0.3, 1.0)
            h = random.uniform(0.3, 1.0)
            x = random.uniform(1.0, self.width - w - 1.0)
            y = random.uniform(1.0, self.height - h - 1.0)
            
            obstacles.append(Obstacle(x, y, w, h))
        
        return obstacles
    
    def _generate_labyrinth(self) -> List[Obstacle]:
        """Labirynt"""
        obstacles = self._generate_empty()
        
        # Dodaj ściany wewnętrzne tworzące labirynt
        # Pionowe ściany
        for i in range(3):
            x = (i + 1) * self.width / 4
            y_start = random.uniform(1.0, 3.0)
            y_height = random.uniform(4.0, 8.0)
            obstacles.append(Obstacle(x, y_start, 0.2, y_height))
        
        # Poziome ściany
        for i in range(3):
            y = (i + 1) * self.height / 4
            x_start = random.uniform(1.0, 3.0)
            x_width = random.uniform(3.0, 6.0)
            obstacles.append(Obstacle(x_start, y, x_width, 0.2))
        
        return obstacles
    
    def _generate_maze(self) -> List[Obstacle]:
        """Gęsty labirynt"""
        obstacles = self._generate_empty()
        
        # Siatka przeszkód
        grid_size = 2.0
        for x in np.arange(2.0, self.width - 2.0, grid_size):
            for y in np.arange(2.0, self.height - 2.0, grid_size):
                if random.random() < 0.4:  # 40% szans na przeszkodę
                    w = random.uniform(0.5, 1.5)
                    h = random.uniform(0.5, 1.5)
                    obstacles.append(Obstacle(x, y, w, h))
        
        return obstacles


# =============================================================================
# ROBOT PHYSICS
# =============================================================================

class DifferentialDriveRobot:
    """Robot z napędem różnicowym + encodery"""
    
    def __init__(self, config: SimConfig, start_x: float, start_y: float, start_theta: float = 0.0):
        self.config = config
        
        # Pozycja i orientacja
        self.x = start_x
        self.y = start_y
        self.theta = start_theta
        
        # Prędkości kół (m/s)
        self.wheel_vel_l = 0.0
        self.wheel_vel_r = 0.0
        
        # Encodery (zliczają obroty)
        self.encoder_l = 0.0  # metry
        self.encoder_r = 0.0  # metry
        
        # Historia pozycji (dla path history)
        self.path_history = []
        self.history_interval = 5  # Co ile kroków zapisywać
        self.history_counter = 0
    
    def set_wheel_velocities(self, left: float, right: float):
        """Ustaw prędkości kół (przed limitowaniem)"""
        # Limit do max_wheel_speed
        self.wheel_vel_l = max(-self.config.max_wheel_speed, 
                              min(self.config.max_wheel_speed, left))
        self.wheel_vel_r = max(-self.config.max_wheel_speed,
                              min(self.config.max_wheel_speed, right))
    
    def update_physics(self, dt: float):
        """Aktualizuj fizykę (kinematyka differential drive)"""
        # Prędkość liniowa i kątowa
        v = (self.wheel_vel_l + self.wheel_vel_r) / 2.0
        omega = (self.wheel_vel_r - self.wheel_vel_l) / self.config.wheel_base
        
        # Aktualizuj pozycję (Euler integration)
        if abs(omega) < 0.001:  # Prosta linia
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt
        else:  # Łuk
            radius = v / omega
            self.x += radius * (math.sin(self.theta + omega * dt) - math.sin(self.theta))
            self.y += -radius * (math.cos(self.theta + omega * dt) - math.cos(self.theta))
            self.theta += omega * dt
        
        # Normalizuj theta do [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Aktualizuj encodery
        self.encoder_l += abs(self.wheel_vel_l) * dt
        self.encoder_r += abs(self.wheel_vel_r) * dt
        
        # Path history
        self.history_counter += 1
        if self.history_counter >= self.history_interval:
            self.path_history.append((self.x, self.y))
            self.history_counter = 0
            
            # Limit długości historii
            if len(self.path_history) > 1000:
                self.path_history.pop(0)
    
    def check_collision(self, obstacles: List[Obstacle]) -> bool:
        """Sprawdź kolizję z przeszkodami"""
        for obs in obstacles:
            if obs.intersects_circle(self.x, self.y, self.config.robot_radius):
                return True
        return False
    
    def get_lidar_scan(self, obstacles: List[Obstacle]) -> List[Tuple[float, float]]:
        """Symuluje skan LIDAR (ray-casting)"""
        scan_points = []
        
        angle_step = 360.0 / self.config.lidar_rays
        
        for i in range(self.config.lidar_rays):
            angle_deg = i * angle_step
            angle_rad = math.radians(angle_deg) + self.theta
            
            # Ray casting
            distance = self._cast_ray(angle_rad, obstacles)
            
            # Dodaj szum
            if self.config.lidar_noise_std > 0:
                distance += random.gauss(0, self.config.lidar_noise_std)
                distance = max(0, min(distance, self.config.lidar_range))
            
            scan_points.append((angle_deg, distance))
        
        return scan_points
    
    def _cast_ray(self, angle: float, obstacles: List[Obstacle]) -> float:
        """Rzuca promień i znajduje najbliższą przeszkodę"""
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Ray marching (small steps)
        step_size = 0.05
        distance = 0.0
        
        while distance < self.config.lidar_range:
            px = self.x + dx * distance
            py = self.y + dy * distance
            
            # Sprawdź kolizję
            for obs in obstacles:
                if obs.contains_point(px, py):
                    return distance
            
            distance += step_size
        
        return self.config.lidar_range  # Max range
    
    def get_ultrasonic_left(self, obstacles: List[Obstacle]) -> float:
        """Symuluje czujnik ultradźwiękowy LEWY (odchylony -15° od osi)"""
        cone_angles = [-self.config.ultrasonic_cone_angle - 7.5,
                       -self.config.ultrasonic_cone_angle,
                       -self.config.ultrasonic_cone_angle + 7.5]
        distances = []
        for offset_deg in cone_angles:
            angle = self.theta + math.radians(offset_deg)
            dist = self._cast_ray(angle, obstacles)
            distances.append(min(dist, self.config.ultrasonic_range))
        self.last_us_left = min(distances)
        return self.last_us_left
    
    def get_ultrasonic_right(self, obstacles: List[Obstacle]) -> float:
        """Symuluje czujnik ultradźwiękowy PRAWY (odchylony +15° od osi)"""
        cone_angles = [self.config.ultrasonic_cone_angle - 7.5,
                       self.config.ultrasonic_cone_angle,
                       self.config.ultrasonic_cone_angle + 7.5]
        distances = []
        for offset_deg in cone_angles:
            angle = self.theta + math.radians(offset_deg)
            dist = self._cast_ray(angle, obstacles)
            distances.append(min(dist, self.config.ultrasonic_range))
        self.last_us_right = min(distances)
        return self.last_us_right
    
    def get_rear_bumper(self, obstacles: List[Obstacle]) -> int:
        """
        Symuluje tylni bumper (0/1).
        Sprawdza czy przeszkoda jest bezpośrednio za robotem w promieniu bumpera.
        """
        check_dist = self.config.robot_radius + 0.03  # Radio + 3cm margines
        # Punkt z tyłu robota
        rear_x = self.x - math.cos(self.theta) * check_dist
        rear_y = self.y - math.sin(self.theta) * check_dist
        for obs in obstacles:
            if obs.contains_point(rear_x, rear_y):
                return 1  # Kolizja tylna!
        return 0


# =============================================================================
# PYGAME TOUCH INTERFACE (dla Androida)
# =============================================================================

class TouchButton:
    """Przycisk dotykowy dla pygame"""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str, color=(100, 100, 100)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = tuple(min(255, c + 50) for c in color)
        self.active = False
        self.font = pygame.font.Font(None, 20)
    
    def draw(self, screen):
        """Rysuje przycisk"""
        color = self.hover_color if self.active else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)  # Border
        
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
    
    def is_clicked(self, pos: Tuple[int, int]) -> bool:
        """Sprawdza czy kliknięto przycisk"""
        return self.rect.collidepoint(pos)


# =============================================================================
# MAIN SIMULATOR
# =============================================================================

class SwarmSimulator:
    """Główny symulator v5.5 - PEŁNA INTEGRACJA"""
    
    def __init__(self, config: SimConfig):
        self.config = config
        
        # Random seed dla powtarzalności
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        # Generuj mapę (LOSOWA!)
        self.map_generator = MapGenerator(config.world_width, config.world_height, config.map_type)
        self.obstacles = self.map_generator.generate()
        
        # Robot (losowa pozycja startowa)
        start_x, start_y, start_theta = self._get_random_start_position()
        self.robot = DifferentialDriveRobot(config, start_x, start_y, start_theta)
        
        # SWARM Core (AI brain)
        self.core = SwarmCoreV55()
        
        # Spatial Memory (NOWE!)
        self.spatial_memory = None
        if config.enable_spatial_memory and SPATIAL_MEMORY_AVAILABLE:
            self.spatial_memory = SpatialMemory(config.world_width, config.world_height)
            logger.info("✓ Spatial Memory enabled")
        else:
            logger.warning("⚠ Spatial Memory disabled")
        
        # v5.5 PERSISTENCE: Wczytaj spatial memory z poprzedniej sesji
        self._load_state()
        
        # Simulation state
        self.step_count = 0
        self.running = False
        self.paused = False
        self.clock = None
        
        # Visualization flags
        self.show_lidar = True
        self.show_path = True
        self.show_occupancy_map = False
        self.show_visited_map = False
        self.show_frontiers = False
        self.show_exploration_goal = True
        
        # Pygame setup
        if config.visualize and PYGAME_AVAILABLE:
            self._init_pygame()
    
    def _get_random_start_position(self) -> Tuple[float, float, float]:
        """Losowa pozycja startowa (bez kolizji)"""
        if not self.config.random_start:
            # Domyślna pozycja
            return 2.0, 2.0, 0.0
        
        # Próbuj znaleźć wolne miejsce
        for _ in range(100):
            x = random.uniform(1.5, self.config.world_width - 1.5)
            y = random.uniform(1.5, self.config.world_height - 1.5)
            theta = random.uniform(-math.pi, math.pi)
            
            # Sprawdź kolizję
            collision = False
            for obs in self.obstacles:
                if obs.intersects_circle(x, y, self.config.robot_radius):
                    collision = True
                    break
            
            if not collision:
                logger.info(f"Random start: ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°)")
                return x, y, theta
        
        # Fallback
        return 2.0, 2.0, 0.0
    
    def _init_pygame(self):
        """Inicjalizacja pygame + interfejsu dotykowego"""
        pygame.init()
        
        # Rozmiar okna
        screen_width = int(self.config.world_width * self.config.pixels_per_meter)
        screen_height = int(self.config.world_height * self.config.pixels_per_meter)
        
        # Dodaj miejsce na przyciski (100px na dole)
        self.button_area_height = 100
        total_height = screen_height + self.button_area_height
        
        self.screen = pygame.display.set_mode((screen_width, total_height))
        pygame.display.set_caption("SWARM v5.5 - Spatial Memory & Exploration")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Przyciski dotykowe (dla Androida!)
        button_width = 80
        button_height = 40
        button_y = screen_height + 10
        spacing = 10
        
        x_offset = 10
        self.buttons = {
            'pause': TouchButton(x_offset, button_y, button_width, button_height, 
                               'Pause', (50, 150, 50)),
            'reset': TouchButton(x_offset + (button_width + spacing), button_y, 
                               button_width, button_height, 'Reset', (150, 50, 50)),
            'map': TouchButton(x_offset + 2 * (button_width + spacing), button_y, 
                             button_width, button_height, 'Map', (100, 100, 150)),
            'visit': TouchButton(x_offset + 3 * (button_width + spacing), button_y, 
                               button_width, button_height, 'Visit', (100, 100, 150)),
            'front': TouchButton(x_offset + 4 * (button_width + spacing), button_y, 
                               button_width, button_height, 'Front', (100, 100, 150)),
            'goal': TouchButton(x_offset + 5 * (button_width + spacing), button_y, 
                              button_width, button_height, 'Goal', (150, 100, 100)),
            'lidar': TouchButton(x_offset, button_y + button_height + 5, 
                               button_width, button_height, 'LIDAR', (100, 150, 100)),
            'path': TouchButton(x_offset + (button_width + spacing), button_y + button_height + 5,
                              button_width, button_height, 'Path', (150, 150, 100))
        }
        
        logger.info("✓ Pygame initialized with touch interface")
    
    def handle_events(self) -> bool:
        """Obsługa eventów (touch + opcjonalnie klawisze)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Touch / Mouse click
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                
                # Sprawdź przyciski
                if self.buttons['pause'].is_clicked(pos):
                    self.paused = not self.paused
                    logger.info(f"{'Paused' if self.paused else 'Resumed'}")
                
                elif self.buttons['reset'].is_clicked(pos):
                    self.reset()
                    logger.info("Reset simulation")
                
                elif self.buttons['map'].is_clicked(pos):
                    self.show_occupancy_map = not self.show_occupancy_map
                
                elif self.buttons['visit'].is_clicked(pos):
                    self.show_visited_map = not self.show_visited_map
                
                elif self.buttons['front'].is_clicked(pos):
                    self.show_frontiers = not self.show_frontiers
                
                elif self.buttons['goal'].is_clicked(pos):
                    self.show_exploration_goal = not self.show_exploration_goal
                
                elif self.buttons['lidar'].is_clicked(pos):
                    self.show_lidar = not self.show_lidar
                
                elif self.buttons['path'].is_clicked(pos):
                    self.show_path = not self.show_path
            
            # Klawiatura (opcjonalnie, jeśli dostępna)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_m:
                    self.show_occupancy_map = not self.show_occupancy_map
                elif event.key == pygame.K_v:
                    self.show_visited_map = not self.show_visited_map
                elif event.key == pygame.K_f:
                    self.show_frontiers = not self.show_frontiers
                elif event.key == pygame.K_g:
                    self.show_exploration_goal = not self.show_exploration_goal
                elif event.key == pygame.K_l:
                    self.show_lidar = not self.show_lidar
                elif event.key == pygame.K_p:
                    self.show_path = not self.show_path
        
        return True
    
    
    def save_state(self):
        """
        v5.5: Zapisz pełny stan symulacji (spatial memory + core).
        """
        state = {
            'version': '5.5',
            'step_count': self.step_count,
            'spatial_memory': None,
            'core_saved': False
        }
        
        # Zapisz spatial memory
        if self.spatial_memory:
            try:
                # Zapisz tylko numpy arrays które faktycznie istnieją w SpatialMemory
                state['spatial_memory'] = {
                    'occupancy_grid': self.spatial_memory.occupancy.grid.copy(),
                    'occupancy_update_count': self.spatial_memory.occupancy.update_count.copy(),
                    'visited_count': self.spatial_memory.visited.visit_count.copy(),
                    'visited_last': self.spatial_memory.visited.last_visit_time.copy(),
                    'current_step': self.spatial_memory.visited.current_step,
                    'current_goal': self.spatial_memory.exploration.current_goal,
                    'step_count': self.step_count
                }
                logger.info("\u2713 Spatial memory saved")
            except Exception as e:
                logger.warning(f"Spatial memory save failed: {e}")
        
        # Core zapisuje się sam przez StateManager, ale zanotujmy
        try:
            self.core.save_state()
            state['core_saved'] = True
        except Exception as e:
            logger.warning(f"Core save failed: {e}")
        
        # Zapisz do pliku
        try:
            save_path = Path(self.config.save_state_file)
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"✓ Simulator state saved: {save_path}")
        except Exception as e:
            logger.error(f"State save failed: {e}")
    
    def _load_state(self):
        """
        v5.5: Wczytaj stan z poprzedniej sesji.
        """
        save_path = Path(self.config.save_state_file)
        if not save_path.exists():
            logger.info("No previous state found - starting fresh")
            return
        
        try:
            with open(save_path, 'rb') as f:
                state = pickle.load(f)
            
            if state.get('version') != '5.5':
                logger.warning(f"State version mismatch: {state.get('version')} != 5.5")
                return
            
            # Przywróć spatial memory
            if state.get('spatial_memory') and self.spatial_memory:
                try:
                    sm = state['spatial_memory']
                    self.spatial_memory.occupancy.grid = sm['occupancy_grid']
                    self.spatial_memory.occupancy.update_count = sm['occupancy_update_count']
                    self.spatial_memory.visited.visit_count = sm['visited_count']
                    self.spatial_memory.visited.last_visit_time = sm['visited_last']
                    self.spatial_memory.visited.current_step = sm.get('current_step', 0)
                    self.spatial_memory.exploration.current_goal = sm.get('current_goal')
                    logger.info(f"\u2713 Spatial memory loaded (from step {sm.get('step_count', 0)})")
                except Exception as e:
                    logger.warning(f"Spatial memory load failed: {e}")
            
            # Core wczytuje się sam przez StateManager w __init__
            if state.get('core_saved'):
                logger.info("✓ Core state was saved in previous session")
            
            logger.info("✓ Simulator state loaded successfully")
            
        except Exception as e:
            logger.error(f"State load failed: {e}")
    
    def reset(self):
        """Reset symulacji"""
        # Nowa losowa mapa
        self.map_generator = MapGenerator(self.config.world_width, 
                                         self.config.world_height,
                                         "random")  # Zawsze losowa!
        self.obstacles = self.map_generator.generate()
        
        # Nowa losowa pozycja startowa
        start_x, start_y, start_theta = self._get_random_start_position()
        self.robot = DifferentialDriveRobot(self.config, start_x, start_y, start_theta)
        
        # Reset spatial memory
        if self.spatial_memory:
            self.spatial_memory = SpatialMemory(self.config.world_width, 
                                               self.config.world_height)
        
        self.step_count = 0
        self.paused = False
        
        logger.info("✓ Simulation reset with new random map!")
    
    def step(self) -> bool:
        """Jeden krok symulacji"""
        if self.config.max_steps > 0 and self.step_count >= self.config.max_steps:
            logger.info("Max steps reached")
            return False
        
        # LIDAR scan
        lidar_scan = self.robot.get_lidar_scan(self.obstacles)
        
        # Ultrasonic L/R (±15° offset)
        us_left = self.robot.get_ultrasonic_left(self.obstacles)
        us_right = self.robot.get_ultrasonic_right(self.obstacles)
        
        # Rear bumper (0/1) — z cooldown żeby nie triggerować co klatkę!
        if not hasattr(self, '_rear_bumper_cooldown'):
            self._rear_bumper_cooldown = 0
        
        if self._rear_bumper_cooldown > 0:
            rear_bumper = 0       # cooldown — nie wysyłaj bumper sygnału
            self._rear_bumper_cooldown -= 1
        else:
            rear_bumper = self.robot.get_rear_bumper(self.obstacles)
            if rear_bumper == 1:
                self._rear_bumper_cooldown = 20  # 20 klatek cooldown (~0.66s @30Hz)
        
        # Encodery (prędkości kół)
        encoder_l = self.robot.wheel_vel_l
        encoder_r = self.robot.wheel_vel_r
        
        # Aktualizuj spatial memory
        if self.spatial_memory:
            self.spatial_memory.update(
                robot_x=self.robot.x,
                robot_y=self.robot.y,
                robot_theta=self.robot.theta,
                lidar_points=lidar_scan
            )
            
            # Pobierz exploration vector
            exploration_vec = self.spatial_memory.get_exploration_vector(
                self.robot.x, self.robot.y
            )
            
            # Logowanie pokrycia co 500 kroków
            if self.step_count % 500 == 0 and self.step_count > 0:
                stats = self.spatial_memory.get_stats()
                logger.info(
                    f"[COVERAGE] step={self.step_count} "
                    f"explored={stats['explored_ratio']*100:.1f}% "
                    f"({stats['explored_cells']} cells) "
                    f"visited={stats['visited_ratio']*100:.1f}% "
                    f"({stats['visited_cells']} cells) "
                    f"goal={stats['current_goal']}"
                )
        else:
            exploration_vec = None
        
        # CORE decision (AI brain) - dual US + rear bumper
        pwm_l, pwm_r = self.core.loop(
            lidar_points=lidar_scan,
            encoder_l=encoder_l,
            encoder_r=encoder_r,
            motor_current=0.5,  # Fake current
            us_left_dist=us_left,
            us_right_dist=us_right,
            rear_bumper=rear_bumper,
            dt=self.config.dt
        )
        
        # PWM → prędkości kół
        vel_l = (pwm_l / 100.0) * self.config.max_wheel_speed
        vel_r = (pwm_r / 100.0) * self.config.max_wheel_speed
        
        # Ustaw prędkości
        self.robot.set_wheel_velocities(vel_l, vel_r)
        
        # Aktualizuj fizykę
        self.robot.update_physics(self.config.dt)
        
        # Sprawdź kolizję — silne obsunięcie żeby naprawdę wyjść z geometrii
        if self.robot.check_collision(self.obstacles):
            if not hasattr(self, '_collision_count'):
                self._collision_count = 0
            self._collision_count += 1
            if self._collision_count % 10 == 1:  # loguj tylko co 10. kolizję
                logger.warning(f"Collision detected! (total={self._collision_count})")
            
            # Kierunek ucieczki: wzdłuż osi robota, w kierunku który COFNĄŁBY ruch
            # Oblicz średnią prędkość = kierunek aktualnego ruchu
            v_total = self.robot.wheel_vel_l + self.robot.wheel_vel_r
            push_dist = 0.20  # 20cm — silniejsze odr sunięcie (było 12cm)
            
            if v_total >= 0:
                # Robot jechał do przodu lub stał → wypchnij WSTECZ
                self.robot.x -= math.cos(self.robot.theta) * push_dist
                self.robot.y -= math.sin(self.robot.theta) * push_dist
            else:
                # Robot cofał → wypchnij DO PRZODU
                self.robot.x += math.cos(self.robot.theta) * push_dist
                self.robot.y += math.sin(self.robot.theta) * push_dist
            
            # Wyzeruj prędkości całkowicie (nie tylko ×0.1 — to nie wystarczało!)
            self.robot.wheel_vel_l = 0.0
            self.robot.wheel_vel_r = 0.0
            
            # Reset cooldown bumpera żeby nie triggować od razu po push-back
            if not hasattr(self, '_rear_bumper_cooldown'):
                self._rear_bumper_cooldown = 0
            self._rear_bumper_cooldown = max(self._rear_bumper_cooldown, 15)
        
        self.step_count += 1
        
        # v5.5 Auto-save co X kroków
        if self.step_count % self.config.auto_save_interval == 0:
            self.save_state()
        return True
    
    def draw(self, screen):
        """Rysuje symulację"""
        # Tło
        screen.fill((240, 240, 240))
        
        # Przeszkody
        for obs in self.obstacles:
            px = int(obs.x * self.config.pixels_per_meter)
            py = int(obs.y * self.config.pixels_per_meter)
            pw = int(obs.width * self.config.pixels_per_meter)
            ph = int(obs.height * self.config.pixels_per_meter)
            pygame.draw.rect(screen, (80, 80, 80), (px, py, pw, ph))
        
        # Occupancy Map
        if self.show_occupancy_map and self.spatial_memory:
            self._draw_occupancy_grid(screen)
        
        # Visited Map
        if self.show_visited_map and self.spatial_memory:
            self._draw_visited_map(screen)
        
        # Frontiers
        if self.show_frontiers and self.spatial_memory:
            self._draw_frontiers(screen)
        
        # Exploration Goal
        if self.show_exploration_goal and self.spatial_memory:
            self._draw_exploration_goal(screen)
        
        # Path history
        if self.show_path and len(self.robot.path_history) > 1:
            points = [(int(x * self.config.pixels_per_meter),
                      int(y * self.config.pixels_per_meter))
                     for x, y in self.robot.path_history]
            pygame.draw.lines(screen, (200, 100, 100), False, points, 2)
        
        # LIDAR rays
        if self.show_lidar:
            lidar_scan = self.robot.get_lidar_scan(self.obstacles)
            for angle_deg, dist in lidar_scan[::3]:  # Co trzeci promień
                angle_rad = math.radians(angle_deg) + self.robot.theta
                end_x = self.robot.x + dist * math.cos(angle_rad)
                end_y = self.robot.y + dist * math.sin(angle_rad)
                
                start_px = int(self.robot.x * self.config.pixels_per_meter)
                start_py = int(self.robot.y * self.config.pixels_per_meter)
                end_px = int(end_x * self.config.pixels_per_meter)
                end_py = int(end_y * self.config.pixels_per_meter)
                
                pygame.draw.line(screen, (100, 200, 100), (start_px, start_py), (end_px, end_py), 1)
                
            # Ultrasonic rays (±15°)
            us_l_dist = getattr(self.robot, 'last_us_left', self.config.ultrasonic_range)
            us_r_dist = getattr(self.robot, 'last_us_right', self.config.ultrasonic_range)
            
            # Left US (-15°)
            us_l_angle = self.robot.theta - math.radians(15)
            us_l_end_x = self.robot.x + us_l_dist * math.cos(us_l_angle)
            us_l_end_y = self.robot.y + us_l_dist * math.sin(us_l_angle)
            pygame.draw.line(screen, (255, 165, 0), (start_px, start_py), 
                             (int(us_l_end_x * self.config.pixels_per_meter), int(us_l_end_y * self.config.pixels_per_meter)), 3)
            
            # Right US (+15°)
            us_r_angle = self.robot.theta + math.radians(15)
            us_r_end_x = self.robot.x + us_r_dist * math.cos(us_r_angle)
            us_r_end_y = self.robot.y + us_r_dist * math.sin(us_r_angle)
            pygame.draw.line(screen, (255, 165, 0), (start_px, start_py), 
                             (int(us_r_end_x * self.config.pixels_per_meter), int(us_r_end_y * self.config.pixels_per_meter)), 3)
        
        # Robot
        robot_px = int(self.robot.x * self.config.pixels_per_meter)
        robot_py = int(self.robot.y * self.config.pixels_per_meter)
        robot_radius_px = int(self.config.robot_radius * self.config.pixels_per_meter)
        
        pygame.draw.circle(screen, (50, 50, 200), (robot_px, robot_py), robot_radius_px)
        
        # Kierunek robota
        dir_len = robot_radius_px * 1.5
        dir_end_x = robot_px + int(dir_len * math.cos(self.robot.theta))
        dir_end_y = robot_py + int(dir_len * math.sin(self.robot.theta))
        pygame.draw.line(screen, (255, 255, 0), (robot_px, robot_py), (dir_end_x, dir_end_y), 3)
        
        # Statystyki
        self._draw_stats(screen)
        
        # Przyciski (obszar na dole)
        button_area_y = int(self.config.world_height * self.config.pixels_per_meter)
        pygame.draw.rect(screen, (30, 30, 30), (0, button_area_y, screen.get_width(), self.button_area_height))
        
        for button in self.buttons.values():
            button.draw(screen)
    
    def _draw_occupancy_grid(self, screen):
        """Rysuje occupancy grid"""
        grid = self.spatial_memory.occupancy
        
        for gy in range(grid.grid_height):
            for gx in range(grid.grid_width):
                occupancy = grid.grid[gy, gx]
                
                if occupancy == 0.0:
                    continue  # Unknown
                
                # Kolor
                if occupancy < 0.7:
                    # Free - zielony
                    alpha = int(150 * (occupancy / 0.7))
                    color = (0, 255, 0)
                else:
                    # Occupied - czerwony
                    alpha = int(200 * ((occupancy - 0.7) / 0.3))
                    color = (255, 0, 0)
                
                # Pozycja
                wx, wy = grid.grid_to_world(gx, gy)
                px = int(wx * self.config.pixels_per_meter)
                py = int(wy * self.config.pixels_per_meter)
                cell_size = int(grid.resolution * self.config.pixels_per_meter)
                
                # Rysuj
                s = pygame.Surface((cell_size, cell_size))
                s.set_alpha(alpha)
                s.fill(color)
                screen.blit(s, (px, py))
    
    def _draw_visited_map(self, screen):
        """Rysuje visited map"""
        vmap = self.spatial_memory.visited
        
        max_visits = np.max(vmap.visit_count) if np.max(vmap.visit_count) > 0 else 1
        
        for gy in range(vmap.grid_height):
            for gx in range(vmap.grid_width):
                count = vmap.visit_count[gy, gx]
                
                if count == 0:
                    continue
                
                intensity = min(255, int(255 * count / max_visits))
                color = (0, 0, intensity)
                
                wx, wy = (gx + 0.5) * vmap.resolution, (gy + 0.5) * vmap.resolution
                px = int(wx * self.config.pixels_per_meter)
                py = int(wy * self.config.pixels_per_meter)
                cell_size = int(vmap.resolution * self.config.pixels_per_meter)
                
                s = pygame.Surface((cell_size, cell_size))
                s.set_alpha(120)
                s.fill(color)
                screen.blit(s, (px, py))
    
    def _draw_frontiers(self, screen):
        """Rysuje frontiers"""
        frontiers = self.spatial_memory.exploration.frontier_detector.find_frontiers(
            self.robot.x, self.robot.y, 
            self.spatial_memory.exploration.exploration_radius
        )
        
        for fx, fy in frontiers:
            px = int(fx * self.config.pixels_per_meter)
            py = int(fy * self.config.pixels_per_meter)
            pygame.draw.circle(screen, (255, 255, 0), (px, py), 3)
    
    def _draw_exploration_goal(self, screen):
        """Rysuje cel eksploracji"""
        goal = self.spatial_memory.exploration.current_goal
        
        if goal is None:
            return
        
        gx, gy = goal
        px = int(gx * self.config.pixels_per_meter)
        py = int(gy * self.config.pixels_per_meter)
        
        # Krzyżyk
        size = 15
        pygame.draw.line(screen, (255, 0, 255), (px - size, py), (px + size, py), 3)
        pygame.draw.line(screen, (255, 0, 255), (px, py - size), (px, py + size), 3)
        
        # Okrąg
        pygame.draw.circle(screen, (255, 0, 255), (px, py), 20, 2)
        
        # Linia od robota
        rx_px = int(self.robot.x * self.config.pixels_per_meter)
        ry_px = int(self.robot.y * self.config.pixels_per_meter)
        pygame.draw.line(screen, (255, 0, 255), (rx_px, ry_px), (px, py), 1)
    
    def _draw_stats(self, screen):
        """Rysuje statystyki"""
        
        # Pobieranie aktualnych wartości czujników / PWM jeśli istnieją
        us_l = getattr(self.robot, 'last_us_left', 3.0)
        us_r = getattr(self.robot, 'last_us_right', 3.0)
        pwm_l = getattr(self.core.motors, 'last_pwm_l', 0) if hasattr(self.core, 'motors') else 0
        pwm_r = getattr(self.core.motors, 'last_pwm_r', 0) if hasattr(self.core, 'motors') else 0
        min_lidar = getattr(self.core.lidar, 'min_dist', 3.0) if hasattr(self.core, 'lidar') else 3.0
        
        # Pobieranie stanu Chaos Engine
        anti_stag = getattr(self.core, 'anti_stagnation', None)
        lorenz = getattr(self.core, 'lorenz', None)
        chaos_mode = "NORMAL"
        lorenz_x = 0.0
        
        if anti_stag:
            if getattr(anti_stag, 'stagnation_force_remaining', 0) > 0:
                chaos_mode = "STAG_FORCE"
            elif getattr(anti_stag, 'is_stagnant', False):
                chaos_mode = "STAGNANT"
        if lorenz:
            lorenz_x = getattr(lorenz, 'x_norm', 0.0)

        stats_lines = [
            f"Step: {self.step_count}/{self.config.max_steps}",
            f"Pos: ({self.robot.x:.2f}, {self.robot.y:.2f})  Theta: {math.degrees(self.robot.theta):.1f}°",
            f"Encoders: L={self.robot.wheel_vel_l:.2f} R={self.robot.wheel_vel_r:.2f} | PWM: L={pwm_l:+.1f} R={pwm_r:+.1f}",
            f"Sensors: USL={us_l:.2f}  USR={us_r:.2f}  Lmin={min_lidar:.2f}",
            f"Chaos: {chaos_mode} | Lx(bias)={lorenz_x:+.2f}"
        ]
        
        if self.spatial_memory:
            stats = self.spatial_memory.get_stats()
            stats_lines.extend([
                f"Explored: {stats['explored_ratio']*100:.1f}%",
                f"Visited: {stats['visited_ratio']*100:.1f}%",
                f"Expl.Bias: {stats['exploration_bias']:.2f}"
            ])
        
        if self.paused:
            stats_lines.append("PAUSED")
        
        # Rysuj
        y_offset = 10
        for line in stats_lines:
            text_surf = self.small_font.render(line, True, (0, 0, 0))
            screen.blit(text_surf, (10, y_offset))
            y_offset += 20
    
    def run(self):
        """Główna pętla symulacji"""
        self.running = True
        logger.info("Simulation started")
        
        # Inicjalizacja Auto-Clean Managera (v5.5)
        # Przekazujemy core i spatial memory (o ile są dostępne w konfiguracji)
        if hasattr(self, 'core') and self.spatial_memory:
            from android_launcher import AutoCleanManager
            self.auto_clean = AutoCleanManager(self.core, self.spatial_memory)
            logger.info("AutoCleanManager initialized (24k step cycle)")
        else:
            self.auto_clean = None
        
        try:
            while self.running:
                # Handle events
                if self.config.visualize:
                    if not self.handle_events():
                        break
                
                # Step simulation
                if not self.paused:
                    if not self.step():
                        break
                    
                    # Sprawdź Auto-Clean podczas aktywnej, nie-zpauzowanej symulacji 
                    if self.auto_clean and hasattr(self.core, 'cycle_count'):
                        if self.auto_clean.check_and_clean(self.core.cycle_count):
                            # Delikatne zwiększenie eksploracji algorytmu po odcięciu biedy z macierzy
                            if hasattr(self.core.brain, 'epsilon'):
                                self.core.brain.epsilon = max(0.05, self.core.brain.epsilon * 1.2)
                
                # Draw
                if self.config.visualize:
                    self.draw(self.screen)
                    pygame.display.flip()
                    self.clock.tick(self.config.fps)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            if self.config.visualize:
                pygame.quit()
        
        # v5.5 Zapisz stan przy wyjściu
        logger.info("Saving final state before exit...")
        self.save_state()
        logger.info("Simulation ended")



# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test
    config = SimConfig()
    config.visualize = True
    config.enable_spatial_memory = True
    
    sim = SwarmSimulator(config)
    sim.run()
