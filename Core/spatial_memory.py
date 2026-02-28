#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
SWARM v5.4 - SPATIAL MEMORY & EXPLORATION MODULE
=============================================================================
"""

import numpy as np
import math
from typing import Tuple, List, Optional
from collections import deque
import logging

logger = logging.getLogger('SpatialMemory')


class OccupancyGrid:
    """
    Mapa zajętości przestrzeni (occupancy grid).
    
    Wartości w gridzie:
    - 0.0 = nieznane (nie było LIDAR)
    - 0.5 = wolne (LIDAR przeszedł, nic nie trafił)
    - 1.0 = zajęte (LIDAR wykrył przeszkodę)
    """
    
    def __init__(self, 
                 width_m: float = 15.0,
                 height_m: float = 12.0,
                 resolution: float = 0.1):  # 10cm per cell
        """
        Args:
            width_m: Szerokość mapy w metrach
            height_m: Wysokość mapy w metrach
            resolution: Rozmiar komórki w metrach
        """
        self.width_m = width_m
        self.height_m = height_m
        self.resolution = resolution
        
        # Rozmiar gridu w komórkach
        self.grid_width = int(width_m / resolution)
        self.grid_height = int(height_m / resolution)
        
        # Grid: 0=unknown, 0.5=free, 1.0=occupied
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        # Licznik aktualizacji dla każdej komórki
        self.update_count = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        
        logger.info(f"OccupancyGrid created: {self.grid_width}x{self.grid_height} cells")
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Konwersja współrzędnych świata → grid"""
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        
        # Clamp do granic
        gx = max(0, min(gx, self.grid_width - 1))
        gy = max(0, min(gy, self.grid_height - 1))
        
        return gx, gy
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Konwersja grid → współrzędne świata (środek komórki)"""
        x = (gx + 0.5) * self.resolution
        y = (gy + 0.5) * self.resolution
        return x, y
    
    def update_from_lidar(self, 
                         robot_x: float, 
                         robot_y: float,
                         robot_theta: float,
                         lidar_points: List[Tuple[float, float]]):
        """
        Aktualizuje grid na podstawie skanowania LIDAR.
        
        Args:
            robot_x, robot_y: Pozycja robota (m)
            robot_theta: Orientacja robota (rad)
            lidar_points: Lista (angle_deg, distance_m)
        """
        rx, ry = self.world_to_grid(robot_x, robot_y)
        
        for angle_deg, distance in lidar_points:
            if distance <= 0 or distance > 100:  # Invalid reading
                continue
            
            # Absolute angle w świecie
            angle_rad = math.radians(angle_deg) + robot_theta
            
            # Punkt końcowy promienia
            end_x = robot_x + distance * math.cos(angle_rad)
            end_y = robot_y + distance * math.sin(angle_rad)
            
            ex, ey = self.world_to_grid(end_x, end_y)
            
            # Bresenham line - wszystkie komórki na linii = FREE
            line_cells = self._bresenham_line(rx, ry, ex, ey)
            
            for lx, ly in line_cells[:-1]:  # Wszystkie oprócz ostatniej
                if 0 <= lx < self.grid_width and 0 <= ly < self.grid_height:
                    # Bayesian update (free space)
                    current = self.grid[ly, lx]
                    self.grid[ly, lx] = current * 0.8 + 0.5 * 0.2  # Powolna konwergencja do 0.5
                    self.update_count[ly, lx] += 1
            
            # Ostatnia komórka = OCCUPIED (jeśli nie max range)
            if distance < 4.9:  # Nie max range
                if 0 <= ex < self.grid_width and 0 <= ey < self.grid_height:
                    current = self.grid[ey, ex]
                    self.grid[ey, ex] = current * 0.6 + 1.0 * 0.4  # Szybka konwergencja do 1.0
                    self.update_count[ey, ex] += 1
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Algorytm Bresenhama - punkty na linii"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def get_occupancy(self, x: float, y: float) -> float:
        """Zwraca wartość occupancy w punkcie (0=unknown, 0.5=free, 1.0=occupied)"""
        gx, gy = self.world_to_grid(x, y)
        return self.grid[gy, gx]
    
    def is_free(self, x: float, y: float, threshold: float = 0.7) -> bool:
        """Sprawdza czy punkt jest wolny"""
        return self.get_occupancy(x, y) < threshold
    
    def is_explored(self, x: float, y: float) -> bool:
        """Sprawdza czy obszar był już zbadany (LIDAR go widział)"""
        gx, gy = self.world_to_grid(x, y)
        return self.update_count[gy, gx] > 0


class VisitedMap:
    """
    Mapa odwiedzonych obszarów.
    Pamięta gdzie robot już był (pozycje X, Y).
    """
    
    def __init__(self, 
                 width_m: float = 15.0,
                 height_m: float = 12.0,
                 resolution: float = 0.2):  # 20cm per cell (większe niż occupancy)
        self.width_m = width_m
        self.height_m = height_m
        self.resolution = resolution
        
        self.grid_width = int(width_m / resolution)
        self.grid_height = int(height_m / resolution)
        
        # Licznik wizyt w każdej komórce
        self.visit_count = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        
        # Timestamp ostatniej wizyty
        self.last_visit_time = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        
        self.current_step = 0
        
        logger.info(f"VisitedMap created: {self.grid_width}x{self.grid_height} cells")
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        gx = max(0, min(gx, self.grid_width - 1))
        gy = max(0, min(gy, self.grid_height - 1))
        return gx, gy
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = (gx + 0.5) * self.resolution
        y = (gy + 0.5) * self.resolution
        return x, y
    
    def mark_visited(self, x: float, y: float):
        """Oznacz pozycję jako odwiedzoną"""
        gx, gy = self.world_to_grid(x, y)
        self.visit_count[gy, gx] += 1
        self.last_visit_time[gy, gx] = self.current_step
        self.current_step += 1
    
    def get_visit_count(self, x: float, y: float) -> int:
        """Ile razy robot był w tym miejscu"""
        gx, gy = self.world_to_grid(x, y)
        return self.visit_count[gy, gx]
    
    def get_recency(self, x: float, y: float) -> int:
        """Ile kroków temu robot był tutaj (0 = teraz, duże = dawno)"""
        gx, gy = self.world_to_grid(x, y)
        if self.visit_count[gy, gx] == 0:
            return 999999  # Nigdy nie był
        return self.current_step - self.last_visit_time[gy, gx]
    
    # ===== NOWE METODY DLA ZGODNOŚCI Z CHAOS ENGINE =====
    
    def get_visited_ratio(self) -> float:
        """Zwraca procent odwiedzonych komórek (0-1)"""
        total_cells = self.grid_width * self.grid_height
        visited_cells = np.count_nonzero(self.visit_count > 0)
        return visited_cells / total_cells if total_cells > 0 else 0.0
    
    def get_visited_cells(self) -> int:
        """Zwraca liczbę odwiedzonych komórek"""
        return int(np.count_nonzero(self.visit_count > 0))


class FrontierDetector:
    """
    Wykrywa frontiers - granice między znanym i nieznanym obszarem.
    Frontiers to potencjalne cele eksploracji.
    """
    
    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
    
    def find_frontiers(self, 
                      robot_x: float,
                      robot_y: float,
                      max_distance: float = 3.0) -> List[Tuple[float, float]]:
        """
        Znajduje frontiers w promieniu max_distance od robota.
        
        Frontier = komórka FREE, która ma sąsiada UNKNOWN
        
        Returns:
            Lista punktów (x, y) w metrach
        """
        frontiers = []
        
        rx, ry = self.grid.world_to_grid(robot_x, robot_y)
        search_radius = int(max_distance / self.grid.resolution)
        
        # Przeszukaj obszar wokół robota
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                gx = rx + dx
                gy = ry + dy
                
                # Sprawdź granice
                if not (0 <= gx < self.grid.grid_width and 
                       0 <= gy < self.grid.grid_height):
                    continue
                
                # Czy to frontier?
                if self._is_frontier_cell(gx, gy):
                    wx, wy = self.grid.grid_to_world(gx, gy)
                    
                    # Dystans od robota
                    dist = math.sqrt((wx - robot_x)**2 + (wy - robot_y)**2)
                    if dist <= max_distance:
                        frontiers.append((wx, wy))
        
        return frontiers
    
    def _is_frontier_cell(self, gx: int, gy: int) -> bool:
        """
        Sprawdza czy komórka jest frontier.
        
        Warunki:
        1. Komórka jest FREE (occupancy < 0.7)
        2. Ma sąsiada UNKNOWN (update_count == 0)
        """
        # Sprawdź czy komórka jest free
        if self.grid.grid[gy, gx] >= 0.7:
            return False
        
        # Sprawdź czy była zbadana
        if self.grid.update_count[gy, gx] == 0:
            return False
        
        # Sprawdź sąsiadów (4-connected)
        neighbors = [
            (gx - 1, gy), (gx + 1, gy),
            (gx, gy - 1), (gx, gy + 1)
        ]
        
        for nx, ny in neighbors:
            if 0 <= nx < self.grid.grid_width and 0 <= ny < self.grid.grid_height:
                # Czy sąsiad jest unknown?
                if self.grid.update_count[ny, nx] == 0:
                    return True
        
        return False


class ExplorationStrategy:
    """
    Strategia eksploracji - celuje w niezbadane obszary.
    
    ALGORYTM:
    1. Znajdź frontiers (granice known/unknown)
    2. Oceń każdy frontier (odległość, novelty)
    3. Wybierz najlepszy cel
    4. Oblicz kierunek do celu → bias dla AI
    """
    
    def __init__(self, 
                 occupancy_grid: OccupancyGrid,
                 visited_map: VisitedMap):
        self.occupancy = occupancy_grid
        self.visited = visited_map
        self.frontier_detector = FrontierDetector(occupancy_grid)
        
        # Aktualny cel eksploracji
        self.current_goal: Optional[Tuple[float, float]] = None
        self.goal_reached_threshold = 0.3  # metry
        
        # Parametry
        self.exploration_radius = 3.0  # Jak daleko szukać frontiers
        self.novelty_weight = 2.0  # Waga dla obszarów nieodwiedzonych
    
    def update(self, robot_x: float, robot_y: float) -> Optional[Tuple[float, float]]:
        """
        Aktualizuje cel eksploracji.
        
        Returns:
            (angle_rad, magnitude) - kierunek do celu, None jeśli brak celu
        """
        # Sprawdź czy dotarł do obecnego celu
        if self.current_goal is not None:
            dist_to_goal = math.sqrt(
                (robot_x - self.current_goal[0])**2 + 
                (robot_y - self.current_goal[1])**2
            )
            if dist_to_goal < self.goal_reached_threshold:
                logger.debug(f"Goal reached: {self.current_goal}")
                self.current_goal = None
        
        # Jeśli nie ma celu, znajdź nowy
        if self.current_goal is None:
            self.current_goal = self._select_best_frontier(robot_x, robot_y)
        
        # Oblicz kierunek do celu
        if self.current_goal is not None:
            dx = self.current_goal[0] - robot_x
            dy = self.current_goal[1] - robot_y
            
            angle = math.atan2(dy, dx)
            magnitude = min(1.0, math.sqrt(dx**2 + dy**2) / 2.0)  # Normalized
            
            return angle, magnitude
        
        return None
    
    def _select_best_frontier(self, 
                             robot_x: float, 
                             robot_y: float) -> Optional[Tuple[float, float]]:
        """
        Wybiera najlepszy frontier do eksploracji.
        
        Scoring:
        - Bliskość (bliżej = lepiej)
        - Novelty (mało odwiedzony = lepiej)
        """
        frontiers = self.frontier_detector.find_frontiers(
            robot_x, robot_y, self.exploration_radius
        )
        
        if not frontiers:
            logger.debug("No frontiers found")
            return None
        
        best_frontier = None
        best_score = -float('inf')
        
        for fx, fy in frontiers:
            # Odległość
            dist = math.sqrt((fx - robot_x)**2 + (fy - robot_y)**2)
            
            # Novelty (czy mało odwiedzony?)
            visit_count = self.visited.get_visit_count(fx, fy)
            recency = self.visited.get_recency(fx, fy)
            
            # Score: bliżej = lepiej, mało odwiedzony = lepiej
            distance_score = 1.0 / (dist + 0.1)  # Unikaj dzielenia przez 0
            novelty_score = 1.0 / (visit_count + 1)  # Nieodwiedzone = wyższy score
            recency_score = min(recency / 100.0, 1.0)  # Dawno = wyższy score
            
            total_score = (distance_score + 
                          self.novelty_weight * novelty_score + 
                          recency_score)
            
            if total_score > best_score:
                best_score = total_score
                best_frontier = (fx, fy)
        
        if best_frontier:
            logger.debug(f"New exploration goal: {best_frontier}, score={best_score:.2f}")
        
        return best_frontier
    
    def get_exploration_bias(self) -> float:
        """
        Zwraca wagę biasu eksploracji (0-1).
        
        Im więcej nieznanego → wyższy bias (bardziej eksplorator)
        Im więcej znanego → niższy bias (więcej exploitation)
        """
        total_cells = self.occupancy.grid_width * self.occupancy.grid_height
        explored_cells = np.count_nonzero(self.occupancy.update_count > 0)
        
        exploration_ratio = explored_cells / total_cells if total_cells > 0 else 0.5
        
        # Im mniej zbadane → wyższy bias
        # 0% zbadane → bias = 0.8
        # 50% zbadane → bias = 0.4
        # 100% zbadane → bias = 0.0
        bias = max(0.0, 0.8 - exploration_ratio * 0.8)
        
        return bias


class SpatialMemory:
    """
    Główny moduł pamięci przestrzennej.
    
    Integruje:
    - Occupancy Grid (co widział LIDAR)
    - Visited Map (gdzie był)
    - Exploration Strategy (gdzie iść)
    """
    
    def __init__(self, 
                 world_width: float = 15.0,
                 world_height: float = 12.0):
        self.occupancy = OccupancyGrid(world_width, world_height, resolution=0.1)
        self.visited = VisitedMap(world_width, world_height, resolution=0.2)
        self.exploration = ExplorationStrategy(self.occupancy, self.visited)
        
        logger.info("SpatialMemory initialized")
    
    def update(self,
               robot_x: float,
               robot_y: float,
               robot_theta: float,
               lidar_points: List[Tuple[float, float]]):
        """
        Aktualizuje wszystkie mapy.
        
        Args:
            robot_x, robot_y: Pozycja robota
            robot_theta: Orientacja robota (rad)
            lidar_points: Lista (angle_deg, distance_m)
        """
        # Aktualizuj occupancy grid z LIDAR
        self.occupancy.update_from_lidar(
            robot_x, robot_y, robot_theta, lidar_points
        )
        
        # Oznacz pozycję jako odwiedzoną
        self.visited.mark_visited(robot_x, robot_y)
    
    def get_exploration_vector(self, 
                              robot_x: float,
                              robot_y: float) -> Optional[Tuple[float, float]]:
        """
        Zwraca kierunek eksploracji.
        
        Returns:
            (angle_rad, magnitude) lub None
        """
        return self.exploration.update(robot_x, robot_y)
    
    def get_exploration_bias_weight(self) -> float:
        """Waga biasu eksploracji (0-1)"""
        return self.exploration.get_exploration_bias()
    
    # ===== NOWE METODY DLA ZGODNOŚCI Z CHAOS ENGINE =====
    
    def get_explored_ratio(self) -> float:
        """Zwraca procent zbadanych komórek (0-1)"""
        total_cells = self.occupancy.grid_width * self.occupancy.grid_height
        explored_cells = np.count_nonzero(self.occupancy.update_count > 0)
        return explored_cells / total_cells if total_cells > 0 else 0.0
    
    def get_visited_ratio(self) -> float:
        """Zwraca procent odwiedzonych komórek (0-1)"""
        return self.visited.get_visited_ratio()
    
    def get_visited_cells(self) -> int:
        """Zwraca liczbę odwiedzonych komórek"""
        return self.visited.get_visited_cells()
    
    def get_stats(self) -> dict:
        """Statystyki dla debugowania"""
        total_cells = self.occupancy.grid_width * self.occupancy.grid_height
        explored_cells = np.count_nonzero(self.occupancy.update_count > 0)
        
        total_visit_cells = self.visited.grid_width * self.visited.grid_height
        visited_cells = np.count_nonzero(self.visited.visit_count > 0)
        
        return {
            'explored_ratio': explored_cells / total_cells if total_cells > 0 else 0.0,
            'explored_cells': int(explored_cells),
            'visited_ratio': visited_cells / total_visit_cells if total_visit_cells > 0 else 0.0,
            'visited_cells': int(visited_cells),
            'current_goal': self.exploration.current_goal,
            'exploration_bias': self.get_exploration_bias_weight()
        }