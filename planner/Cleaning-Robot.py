#!/usr/bin/env python3
#!/usr/bin/env python3
# sourcery skip: use-contextlib-suppress
"""
Full Robot Simulation Project
Features:
1. Grid-based Map Editor (Barriers, Dirt)
2. Path Planning (A*, Dijkstra)
3. Multi-Agent Swarm Intelligence (3+ Robots)
4. Reward-Shaped Q-Learning (Sequential "Clean All" Mode)
5. SLAM Exploration
6. Dynamic Obstacles
"""

import os
import sys
import random
import numpy as np
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False
    print("Warning: 'pygame' is not installed. GUI features will be disabled.")
    print("To use the full GUI, install pygame: pip3 install pygame")
import json
import time
import threading

# Ensure project root is on sys.path so `planner` package imports work
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

try:
    from planner import search as search_alg
    from planner import dwa as dwa_planner
except ImportError:
    pass # Handle if planner not found for standalone testing

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
WIDTH = 800
ROWS = 50
SIDEBAR = 260
WINDOW_WIDTH = WIDTH + SIDEBAR

# Colors
CELL_COLOR = (255, 255, 255, 0)
BARRIER_COLOR = (200, 0, 0)
DIRT_COLOR = (139, 69, 19)
GRID_COLOR = (200, 200, 200)
ROBOT_COLOR = (0, 180, 0)         # Original Robot (Green)
ROBOT_SWARM_COLOR = (0, 100, 255) # Swarm Robots (Blue)
DYNAMIC_COLOR = (0, 120, 255)     # Moving Obstacles
VISITED_OVERLAY = (200, 200, 20, 90)

# Constants
SENSOR_RADIUS = 5 
DYNAMIC_MOVE_INTERVAL = 0.5
MAP_IMG_NAME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'final-home (1).png'))

# --- NEW IMAGE CONFIGURATION ---
# IMPORTANT: Replace these with the actual paths to your PNG files
ROBOT_MAIN_IMG_PATH = "/Users/juju/Desktop/robot/planner/2d robot main.png" # Assuming these are in the same directory as the script
ROBOT_SWARM_IMG_PATH = "/Users/juju/Desktop/robot/planner/2d robot swarm.png"
# -------------------------------

# -----------------------------------------------------------------------------
# CLASSES
# -----------------------------------------------------------------------------

class Cell:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = col * width
        self.y = row * width
        self.width = width
        self.total_rows = total_rows
        self.is_barrier = False
        self.has_dirt = False
        self.was_visited = False
        self.explored = False
        # COMMUNICATION FLAG: Shared state for swarm coordination
        self.targeted = False 

    def get_pos(self):
        return self.row, self.col

    def make_barrier(self):
        self.is_barrier = True

    def remove_barrier(self):
        self.is_barrier = False

    def toggle_barrier(self):
        self.is_barrier = not self.is_barrier

    def make_dirt(self):
        self.has_dirt = True
        self.targeted = False

    def remove_dirt(self):
        self.has_dirt = False
        self.targeted = False

    def draw(self, win):
        rect = pygame.Rect(self.x, self.y, self.width, self.width)
            
        # Draw visited overlay (semi-transparent)
        if getattr(self, 'was_visited', False):
            s = pygame.Surface((self.width, self.width), pygame.SRCALPHA)
            s.fill(VISITED_OVERLAY)
            win.blit(s, (self.x, self.y))
            
        # Draw explored overlay (lighter)
        if getattr(self, 'explored', False) and not getattr(self, 'was_visited', False):
            s2 = pygame.Surface((self.width, self.width), pygame.SRCALPHA)
            s2.fill((180, 180, 180, 60))
            win.blit(s2, (self.x, self.y))
            
        # Draw barrier
        if self.is_barrier:
            pygame.draw.rect(win, BARRIER_COLOR, rect)
            
        # Draw dirt
        if self.has_dirt:
            cx = self.x + self.width // 2
            cy = self.y + self.width // 2
            r = max(2, self.width // 8)
            pygame.draw.circle(win, DIRT_COLOR, (cx, cy), r)
            
        # Draw dynamic obstacle
        if getattr(self, 'has_dynamic', False):
            pygame.draw.rect(win, DYNAMIC_COLOR, rect)


class Robot:
    def __init__(self, row, col, width, rows, color=ROBOT_COLOR, is_main=False, robot_img=None):
        # cell size (pixels per grid cell)
        gap = width // rows
        self.row = row
        self.col = col
        self.x = col * gap + gap // 2
        self.y = row * gap + gap // 2
        # Increase default robot visual radius so robots are more visible
        # Original used gap//3 with a min of 6; bump to gap//2 with a min of 10
        self.radius = max(10, gap // 2)
        self.color = color
        self.is_main = is_main # New flag to distinguish main/swarm
        self.image = robot_img # Image to be drawn
        self.image_rect = None # Rectangle for image position

        # Navigation State
        self.path_index = 0
        self.following = False
        # Increase speed fallback slightly so movement looks consistent with larger size
        self.speed = max(4, gap // 4)
        self.current_path = [] # Each robot stores its own path

    def draw(self, win):
        # --- UPDATED DRAW METHOD ---
        if self.image:
            # Calculate position to center the image on the robot's coordinates (self.x, self.y)
            if self.image_rect is None:
                self.image_rect = self.image.get_rect()
            
            # Center the image
            self.image_rect.center = (self.x, self.y)
            win.blit(self.image, self.image_rect)
        else:
            # Fallback to circle drawing if image is None
            pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)
        # ---------------------------

    def move_towards(self, tx, ty):
        """Move the robot towards target (tx,ty). Returns True if reached target."""
        dx = tx - self.x
        dy = ty - self.y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1e-3:
            return True
        step = min(self.speed, dist)
        self.x += int(round(dx / dist * step)) if dist != 0 else 0
        self.y += int(round(dy / dist * step)) if dist != 0 else 0
        return abs(self.x - tx) <= 1 and abs(self.y - ty) <= 1

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            cell = Cell(i, j, gap, rows)
            grid[i].append(cell)
    return grid

def draw_grid_lines(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GRID_COLOR, (0, i * gap), (width, i * gap))
        pygame.draw.line(win, GRID_COLOR, (i * gap, 0), (i * gap, width))

def draw(win, grid, rows, width, background, robots=[]):
    if background:
        win.blit(background, (0, 0))
    else:
        win.fill((255,255,255))
    
    for row in grid:
        for cell in row:
            cell.draw(win)
    draw_grid_lines(win, rows, width)
    
    # Draw paths for ALL robots
    gap = width // rows
    for i, robot in enumerate(robots):
        # Fallback for manual control of robot 0
        path_to_draw = getattr(robot, 'current_path', [])
        
        # If Robot 0 is being controlled manually via 'P' or Q-Learn, sync
        if i == 0 and hasattr(draw, 'current_path') and draw.current_path and not getattr(draw, 'swarm_active', False):
             path_to_draw = draw.current_path

        if path_to_draw:
            pts = []
            for (r, c) in path_to_draw:
                cx = c * gap + gap // 2
                cy = r * gap + gap // 2
                pts.append((cx, cy))
            if len(pts) > 1:
                # Use a neutral color for path if the robot's color is being replaced by an image
                path_color = (255, 100, 100) if robot.is_main else (100, 100, 255)
                pygame.draw.lines(win, path_color, False, pts, max(2, gap // 10))
    # Draw robots on top of paths
    for robot in robots:
        try:
            robot.draw(win)
        except Exception:
            # Fallback to drawing a simple circle at robot coordinates
            try:
                pygame.draw.circle(win, robot.color, (int(robot.x), int(robot.y)), robot.radius)
            except Exception:
                pass
    return

def get_clicked_pos(pos, rows, width):
    # sourcery skip: assign-if-exp, reintroduce-else
    x, y = pos
    gap = width // rows
    col = x // gap
    row = y // gap
    if row < 0 or row >= rows or col < 0 or col >= rows:
        return None
    return row, col

def place_random_dirt(grid, amount=20):
    rows = len(grid)
    placed = 0
    attempts = 0
    while placed < amount and attempts < amount * 10:
        r = random.randrange(rows)
        c = random.randrange(rows)
        if not grid[r][c].is_barrier and not grid[r][c].has_dirt:
            grid[r][c].make_dirt()
            placed += 1
        attempts += 1
    print(f"Placed {placed} dirt spots")


# ---------- Q-Learning hyperparameters / optimizer config ----------
DEFAULT_CONFIG = {
    'alpha': 0.5,
    'gamma': 0.95,
    'epsilon': 1.0,
    'epsilon_decay': 0.99,
    'min_epsilon': 0.01,
    'episodes': 300,
    'max_steps': 200,
    'optimizer': 'sgd',  # options: 'sgd', 'momentum', 'adam'
    # optimizer-specific
    'momentum_beta': 0.9,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_eps': 1e-8,
    # reward shaping / scaling
    'reward_scale': 1.0,
    'reward_clip': None,  # e.g. 100 to clip rewards to [-100,100]
}

# Global runtime config (can be updated from CLI args)
CONFIG = DEFAULT_CONFIG.copy()

# Allow overriding CONFIG from environment variable Q_CONFIG (JSON string).
# This is used by external tuning scripts to run experiments without editing the file.
try:
    qcfg_env = os.environ.get('Q_CONFIG')
    if qcfg_env:
        try:
            loaded = json.loads(qcfg_env)
            if isinstance(loaded, dict):
                CONFIG.update(loaded)
                print(f"Loaded CONFIG overrides from Q_CONFIG: {loaded}")
        except Exception as _e:
            print(f"Failed to parse Q_CONFIG environment variable: {_e}")
except Exception:
    pass

def find_nearest_dirt(grid, start):
    # sourcery skip: unwrap-iterable-construction
    """Standard BFS to find nearest dirt."""
    from collections import deque
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    q = deque()
    q.append(start)
    visited = set([start])
    while q:
        r, c = q.popleft()
        if grid[r][c].has_dirt:
            return (r, c)
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr,nc) not in visited and not grid[nr][nc].is_barrier:
                visited.add((nr,nc))
                q.append((nr,nc))
    return None

def find_nearest_untargeted_dirt(grid, start):
    # sourcery skip: unwrap-iterable-construction
    """
    SWARM BFS: Finds nearest dirt NOT targeted by another robot.
    Ensures robots don't chase the same goal.
    """
    from collections import deque
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    q = deque()
    q.append(start)
    visited = set([start])
    while q:
        r, c = q.popleft()
        # Key Swarm Logic: Check grid[r][c].targeted
        if grid[r][c].has_dirt and not grid[r][c].targeted:
            return (r, c)
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr,nc) not in visited and not grid[nr][nc].is_barrier:
                visited.add((nr,nc))
                q.append((nr,nc))
    return None

def save_map(grid, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = len(grid)
    occ = np.zeros((rows, rows), dtype=np.uint8)
    for i in range(rows):
        for j in range(rows):
            if grid[i][j].is_barrier:
                occ[i, j] = 1
    np.save(os.path.join(out_dir, 'occupancy.npy'), occ)
    dirt = np.zeros((rows, rows), dtype=np.uint8)
    for i in range(rows):
        for j in range(rows):
            if grid[i][j].has_dirt:
                dirt[i, j] = 1
    np.save(os.path.join(out_dir, 'dirt.npy'), dirt)
    print(f"Saved map to {out_dir}")

# -----------------------------------------------------------------------------
# Q-LEARNING WITH REWARD SHAPING
# -----------------------------------------------------------------------------
def run_qlearning_demo_blocking(grid, start_pos, goal_pos, update_status, config=None):
    """
    Run Q-learning with REWARD SHAPING and optimizer support.
    config: optional dict to override CONFIG values (alpha/gamma/epsilon/... and optimizer)
    """
    cfg = CONFIG.copy() if config is None else {**CONFIG, **config}

    rows = len(grid)
    cols = len(grid[0])

    print(f"Training Q-Learning for target {goal_pos} with optimizer={cfg.get('optimizer')}")

    # Q-Table
    Q = np.zeros((rows, cols, 4), dtype=np.float64)

    # Hyperparameters from config
    alpha = float(cfg.get('alpha', 0.5))
    gamma = float(cfg.get('gamma', 0.95))
    epsilon = float(cfg.get('epsilon', 1.0))
    epsilon_decay = float(cfg.get('epsilon_decay', 0.99))
    min_epsilon = float(cfg.get('min_epsilon', 0.01))
    episodes = int(cfg.get('episodes', 300))
    max_steps = int(cfg.get('max_steps', 200))
    optimizer = cfg.get('optimizer', 'sgd')

    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_dist(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # Optimizer internal state
    if optimizer == 'momentum':
        vel = np.zeros_like(Q)
        beta = float(cfg.get('momentum_beta', 0.9))
    elif optimizer == 'adam':
        m = np.zeros_like(Q)
        v = np.zeros_like(Q)
        beta1 = float(cfg.get('adam_beta1', 0.9))
        beta2 = float(cfg.get('adam_beta2', 0.999))
        eps = float(cfg.get('adam_eps', 1e-8))
        t = 0
    elif optimizer == 'sgd':
        # No extra state needed for SGD
        pass

    for ep in range(1, episodes + 1):
        # Pump pygame events if running with GUI; keeps window responsive.
        if PYGAME_AVAILABLE and pygame is not None:
            try:
                pygame.event.pump()
            except Exception:
                pass

        state = start_pos
        prev_dist = get_dist(state, goal_pos)

        for step in range(max_steps):
            r, c = state

            # Action selection
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)
            else:
                action = int(np.argmax(Q[r, c, :]))

            dr, dc = actions[action]
            nr, nc = r + dr, c + dc

            reward = -1
            done = False
            next_state = (nr, nc)

            # Reward shaping
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or grid[nr][nc].is_barrier:
                reward = -10
                next_state = state
            elif (nr, nc) == goal_pos:
                reward = 100
                done = True
            else:
                curr_dist = get_dist(next_state, goal_pos)
                if curr_dist < prev_dist:
                    reward += 1
                elif curr_dist > prev_dist:
                    reward -= 1.5
                prev_dist = curr_dist

            # Apply global reward scaling/clipping from config
            reward_scale = float(cfg.get('reward_scale', 1.0))
            reward_clip = cfg.get('reward_clip', None)
            reward = reward * reward_scale
            if reward_clip is not None:
                try:
                    rcv = float(reward_clip)
                    reward = max(-rcv, min(rcv, reward))
                except Exception:
                    pass

            old_value = Q[r, c, action]
            next_max = np.max(Q[next_state[0], next_state[1], :])
            td_target = reward + gamma * next_max
            td_error = td_target - old_value

            # Optimizer-style updates
            if optimizer == 'momentum':
                vel[r, c, action] = beta * vel[r, c, action] + alpha * td_error
                Q[r, c, action] = old_value + vel[r, c, action]
            elif optimizer == 'adam':
                t += 1
                m[r, c, action] = beta1 * m[r, c, action] + (1 - beta1) * td_error
                v[r, c, action] = beta2 * v[r, c, action] + (1 - beta2) * (td_error * td_error)
                m_hat = m[r, c, action] / (1 - beta1 ** t)
                v_hat = v[r, c, action] / (1 - beta2 ** t)
                lr = alpha
                Q[r, c, action] = old_value + lr * m_hat / (np.sqrt(v_hat) + eps)
            elif optimizer == 'sgd':
                Q[r, c, action] = old_value + alpha * td_error

            state = next_state
            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if ep % 50 == 0:
            update_status({'qlearn_msg': (f"Train Ep {ep}/{episodes}\n | "
                                f"Eps: {epsilon:.3f}\n | "
                                f"Opt: {optimizer}")})

    return Q

def extract_path_from_q(Q, start, goal, grid):
    """Extract best path from Q-table"""
    path = []
    curr = start
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    rows = len(grid)
    cols = len(grid[0])
    
    visited = set()
    visited.add(curr)
    path.append(curr)
    
    for _ in range(500): 
        r, c = curr
        action_idx = int(np.argmax(Q[r, c, :]))
        dr, dc = actions[action_idx]
        nr, nc = r + dr, c + dc
        
        if (nr, nc) == goal:
            path.append((nr, nc))
            return path
        
        if 0 <= nr < rows and 0 <= nc < cols and not grid[nr][nc].is_barrier:
            if (nr, nc) in visited: break
            curr = (nr, nc)
            visited.add(curr)
            path.append(curr)
        else:
            break
    return path

def qlearn_update_status(draw_obj, info_dict):
    # If no draw object provided (headless mode), print or ignore status updates
    if draw_obj is None:
        try:
            # Try to show a concise message
            if 'qlearn_msg' in info_dict:
                print(f"Q-Learn: {info_dict['qlearn_msg']}")
        except Exception:
            pass
        return

    if not hasattr(draw_obj, 'qlearn_status') or draw_obj.qlearn_status is None:
        draw_obj.qlearn_status = {}
    draw_obj.qlearn_status.update(info_dict)


# MAIN LOOP

def main():
    # Quick runtime check: pygame (and many GUI frameworks) must run on
    # the main thread on macOS and many windowing systems. If the
    # script is started from an IDE debugger or a background thread the
    # display may not appear even though pygame is installed. Detect
    # that early and print a helpful message.
    if threading.current_thread() is not threading.main_thread():
        print("Warning: Detected that the script is NOT running on the main thread.")
        print("Pygame display windows often require execution on the main thread (especially on macOS).")
        print("If you're running from an IDE (debugger) or via an embedded console, try running from a regular terminal:")
        print("  python3 /Users/juju/Desktop/robot/planner/Cleaning-Robot.py")
        print("Or run with --test-display to confirm whether your environment supports opening windows.")
        # continue anyway; main will try to initialize the display and will fall back to headless if it fails

    # Initialize pygame and try to open a window. Provide detailed
    # diagnostics if the display cannot be initialized (common in
    # headless/remote or misconfigured environments).
    try:
        pygame.init()
        # Print basic environment/display diagnostics
        try:
            print(f"Platform: {sys.platform}")
            print(f"PYGAME_AVAILABLE: {PYGAME_AVAILABLE}")
            print(f"pygame version: {getattr(pygame, '__version__', 'unknown')}")
            try:
                print(f"SDL version: {pygame.get_sdl_version()}")
            except Exception:
                pass
        except Exception:
            pass

        # Attempt to initialize the display subsystem and create a window.
        # Use a resizable window to improve compatibility with window managers.
        try:
            try:
                pygame.display.init()
            except Exception:
                # Some pygame builds auto-init the display on set_mode; ignore
                pass
            win = pygame.display.set_mode((WINDOW_WIDTH, WIDTH), pygame.RESIZABLE)
            pygame.display.set_caption("Sim Runner — Clean All Dirt (Q-Learning)")

            # Quick visible test so the window manager composes something and
            # the user can immediately verify that a window appeared. Render a
            # simple message and run a short pump/flip loop to help mapping.
            try:
                win.fill((50, 50, 150))
                try:
                    test_font = pygame.font.SysFont(None, 32)
                    msg = test_font.render('Cleaning Robot — Window Test', True, (255, 255, 255))
                    rect = msg.get_rect(center=(WINDOW_WIDTH // 2, WIDTH // 2))
                    win.blit(msg, rect)
                except Exception:
                    pass
                for _ in range(10):
                    try:
                        pygame.event.pump()
                    except Exception:
                        pass
                    try:
                        pygame.display.flip()
                    except Exception:
                        pass
                    pygame.time.wait(100)

                try:
                    print(f"Display driver: {pygame.display.get_driver()}")
                    print(f"Display initialized: {pygame.display.get_init()}")
                    info = pygame.display.Info()
                    print(f"Video Info: current_w={info.current_w}, current_h={info.current_h}")
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            # Let outer except handle fallback to headless
            raise
    except Exception as e:
        print(f"Pygame display initialization failed: {e}")
        print("The GUI cannot be started. Falling back to headless mode.")
        print("If you expected a GUI, ensure you run the script locally with an active desktop session, or run with --nogui to run headless.")
        try:
            return headless_smoke_test()
        finally:
            try:
                pygame.quit()
            except Exception:
                pass

    # Attempt to load background image if it exists and pygame is available.
    background = None
    try:
        if os.path.exists(MAP_IMG_NAME):
            # Print small diagnostic so user can see exactly which path is used
            print(f"Background image path: {MAP_IMG_NAME}")
            if PYGAME_AVAILABLE and pygame is not None:
                try:
                    background = pygame.image.load(MAP_IMG_NAME)
                    background = pygame.transform.scale(background, (WIDTH, WIDTH))
                except Exception as e:
                    print(f"Failed to load background image: {e}")
                    background = None
            else:
                print("Pygame not available — skipping background image load.")
        else:
            print(f"Background image not found at: {MAP_IMG_NAME}")
            background = None
    except Exception as e:
        print(f"Error while checking/loading background image: {e}")
        background = None

    # --- IMAGE LOADING ---
    robot_main_img = None
    robot_swarm_img = None
    # Keep image scaling consistent with Robot.radius (gap//2 with a min of 10)
    gap = WIDTH // ROWS
    robot_radius = max(10, gap // 2)
    
    if PYGAME_AVAILABLE and pygame is not None:
        try:
            # Load and scale main robot image
            main_img_original = pygame.image.load(ROBOT_MAIN_IMG_PATH).convert_alpha()
            # Scale to roughly 2x the circle radius for a reasonable display size
            img_size = robot_radius * 2
            robot_main_img = pygame.transform.scale(main_img_original, (img_size, img_size))
            print(f"Loaded and scaled main robot image: {ROBOT_MAIN_IMG_PATH}")
        except pygame.error as e:
            print(f"Error loading main robot image: {ROBOT_MAIN_IMG_PATH}. Falling back to circle. Error: {e}")
        
        try:
            # Load and scale swarm robot image
            swarm_img_original = pygame.image.load(ROBOT_SWARM_IMG_PATH).convert_alpha()
            img_size = robot_radius * 2
            robot_swarm_img = pygame.transform.scale(swarm_img_original, (img_size, img_size))
            print(f"Loaded and scaled swarm robot image: {ROBOT_SWARM_IMG_PATH}")
        except pygame.error as e:
            print(f"Error loading swarm robot image: {ROBOT_SWARM_IMG_PATH}. Falling back to circle. Error: {e}")
    # ---------------------

    grid = make_grid(ROWS, WIDTH)
    
    robots = []
    font = pygame.font.SysFont(None, 18)
    last_map_pos = None
    
    # Sidebar
    sidebar_buttons = {}
    btn_w = SIDEBAR - 40
    x0 = WIDTH + 20
    y = 40
    def add_btn(name, label):
        nonlocal y
        rect = pygame.Rect(x0, y, btn_w, 28)
        sidebar_buttons[name] = (rect, label)
        y += 36

    add_btn('add_robot', '1. Add Robot')
    add_btn('save_env', '2. Save Environment')
    add_btn('clear', '3. Clear')
    add_btn('restore', '4. Restore')
    add_btn('clean_astar', '5. Clean (A*)')
    add_btn('clean_dij', '6. Clean (Dijkstra)')
    add_btn('add_dirt', '7. Add Dirt (10)')
    add_btn('add_dynamic', '9. Add Dynamic Obstacle')
    add_btn('run_qlearn', '10. Run Q-Learn (Clean All)')
    add_btn('run_swarm', '11. Run Swarm (3 Bots)')
    y += 8
    add_btn('slam_explore', 'SLAM Explore (85%)')

    dynamic_obstacles = []
    
    # State Flags
    draw.algorithm = 'astar'
    draw.auto_clean = False
    draw.goal = None
    draw._current_run = None
    draw.last_run_str = None
    draw.slam_exploring = False
    draw.slam_explored = False
    draw.slam_requested = False
    draw._suppress_slam_until = 0.0
    draw.qlearn_status = {'qlearn_msg': 'idle'}
    draw.swarm_active = False
    draw.q_learning_active = False # New Flag for Q-Learning loop

    run = True
    clock = pygame.time.Clock()

    while run:
        dt = clock.tick(60) / 1000.0
        
        draw(win, grid, ROWS, WIDTH, background, robots)
        
        # Sidebar
        pygame.draw.rect(win, (40, 40, 40), pygame.Rect(WIDTH, 0, SIDEBAR, WIDTH))
        title = font.render('Controls & Planner', True, (255, 255, 255))
        win.blit(title, (WIDTH + 16, 8))

        for name, (rect, label) in sidebar_buttons.items():
            color = (200, 200, 200)
            if name == 'run_swarm' and draw.swarm_active: color = (50, 255, 100)
            if name == 'run_qlearn' and draw.q_learning_active: color = (50, 255, 100)
            pygame.draw.rect(win, color, rect)
            txt = font.render(label, True, (10, 10, 10))
            win.blit(txt, (rect.x + 6, rect.y + 6))

        # Status Info
        status_y = y + 10
        alg_text = font.render(f'Alg: {getattr(draw, "algorithm", "astar")}', True, (255, 255, 255))
        win.blit(alg_text, (WIDTH + 16, status_y))
        
        q_text = font.render(f'Q-Status: {draw.qlearn_status.get("qlearn_msg", "")}', True, (255, 255, 255))
        win.blit(q_text, (WIDTH + 16, status_y + 24))

        if len(robots) > 0:
            pos_text = font.render(f'Robot1: ({robots[0].row},{robots[0].col})', True, (255, 255, 255))
            win.blit(pos_text, (WIDTH + 16, status_y + 48))

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            # Map Editing
            if event.type == pygame.MOUSEMOTION:
                pos = event.pos
                if pos[0] < WIDTH:
                    rc = get_clicked_pos(pos, ROWS, WIDTH)
                    if rc:
                        if pygame.mouse.get_pressed()[0]: 
                            if not grid[rc[0]][rc[1]].is_barrier: grid[rc[0]][rc[1]].make_barrier()
                        if pygame.mouse.get_pressed()[2]: 
                            grid[rc[0]][rc[1]].remove_barrier()
                            grid[rc[0]][rc[1]].remove_dirt()
                            if hasattr(grid[rc[0]][rc[1]], 'has_dynamic'): grid[rc[0]][rc[1]].has_dynamic = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if pos[0] >= WIDTH: 
                    if getattr(event, 'button', 1) != 1: continue
                    for name, (rect, label) in sidebar_buttons.items():
                        if rect.collidepoint(pos):
                            if name == 'add_robot':
                                placed = False
                                while not placed:
                                    rr, rc = random.randrange(ROWS), random.randrange(ROWS)
                                    if not grid[rr][rc].is_barrier:
                                        # --- UPDATED ROBOT CREATION ---
                                        is_main = len(robots) == 0
                                        img = robot_main_img if is_main else robot_swarm_img
                                        color = ROBOT_COLOR if is_main else ROBOT_SWARM_COLOR
                                        robots.append(Robot(rr, rc, WIDTH, ROWS, color=color, is_main=is_main, robot_img=img))
                                        # ------------------------------
                                        placed = True
                            
                            elif name == 'save_env':
                                save_map(grid, os.path.join(os.path.dirname(__file__), '..', 'data'))
                            
                            elif name == 'clear':
                                for i in range(ROWS):
                                    for j in range(ROWS):
                                        grid[i][j].is_barrier = False
                                        grid[i][j].remove_dirt()
                                        if hasattr(grid[i][j], 'has_dynamic'): grid[i][j].has_dynamic = False
                                robots = []
                                draw.swarm_active = False
                                draw.q_learning_active = False
                                draw.goal = None
                                dynamic_obstacles.clear()

                            elif name == 'restore':
                                try:
                                    occ = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'saved_occupancy.npy'))
                                    for i in range(min(len(occ), ROWS)):
                                        for j in range(min(len(occ), ROWS)):
                                            grid[i][j].is_barrier = bool(occ[i, j])
                                except: pass
                                draw.slam_explored = False
                                draw.slam_requested = False
                                for r in robots: r.following = False

                            elif name == 'clean_astar': draw.algorithm = 'astar'; draw.auto_clean = True
                            elif name == 'clean_dij': draw.algorithm = 'dijkstra'; draw.auto_clean = True
                            elif name == 'add_dirt': place_random_dirt(grid, 10)
                            
                            elif name == 'add_dynamic':
                                for _ in range(100):
                                    rr, rc = random.randrange(ROWS), random.randrange(ROWS)
                                    if not grid[rr][rc].is_barrier:
                                        grid[rr][rc].has_dynamic = True
                                        dynamic_obstacles.append({'r': rr, 'c': rc, 'last_move': time.time()})
                                        break
                            
                            
                            # TRIGGER CONTINUOUS Q-LEARNING
                            elif name == 'run_qlearn':
                                if not robots: 
                                    print("Place robot first.")
                                else:
                                    print("Starting Sequential Q-Learning Mode...")
                                    draw.q_learning_active = True
                                    draw.swarm_active = False
                                    # Reset path
                                    robots[0].current_path = []
                                    robots[0].following = False

                            # TRIGGER SWARM
                            elif name == 'run_swarm':
                                print("Initializing Swarm...")
                                while len(robots) < 3:
                                    placed = False
                                    while not placed:
                                        rr, rc = random.randrange(ROWS), random.randrange(ROWS)
                                        if not grid[rr][rc].is_barrier:
                                            # --- UPDATED SWARM CREATION ---
                                            is_main = len(robots) == 0
                                            img = robot_main_img if is_main else robot_swarm_img
                                            color = ROBOT_COLOR if is_main else ROBOT_SWARM_COLOR
                                            robots.append(Robot(rr, rc, WIDTH, ROWS, color=color, is_main=is_main, robot_img=img))
                                            # ------------------------------
                                            placed = True
                                if sum(1 for r in grid for c in r if c.has_dirt) < 5:
                                    place_random_dirt(grid, 20)
                                draw.swarm_active = True
                                draw.q_learning_active = False
                                for r in robots: 
                                    r.current_path = []
                                    r.following = False

                            elif name == 'slam_explore':
                                draw.slam_requested = True
                                draw.slam_explored = False

                else: 
                    rc = get_clicked_pos(pos, ROWS, WIDTH)
                    if rc:
                        if event.button == 1: grid[rc[0]][rc[1]].make_barrier()
                        elif event.button == 3: grid[rc[0]][rc[1]].remove_barrier()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Place Robot
                    for _ in range(100):
                        rr, rc = random.randrange(ROWS), random.randrange(ROWS)
                        if not grid[rr][rc].is_barrier:
                            # --- UPDATED KEYDOWN ROBOT CREATION ---
                            is_main = len(robots) == 0
                            img = robot_main_img if is_main else robot_swarm_img
                            color = ROBOT_COLOR if is_main else ROBOT_SWARM_COLOR
                            robots.append(Robot(rr, rc, WIDTH, ROWS, color=color, is_main=is_main, robot_img=img))
                            # --------------------------------------
                            break
                if event.key == pygame.K_d: place_random_dirt(grid, 10)
                if event.key == pygame.K_g: 
                    pos = pygame.mouse.get_pos()
                    rc = get_clicked_pos(pos, ROWS, WIDTH)
                    if rc: draw.goal = rc
                if event.key == pygame.K_p: # Plan Path (Robot 0)
                    if robots and draw.goal:
                        path = search_alg.astar((robots[0].row, robots[0].col), draw.goal, grid)
                        if path:
                            robots[0].current_path = path
                            draw.current_path = path
                if event.key == pygame.K_SPACE: 
                    if robots: robots[0].following = not robots[0].following
                if event.key == pygame.K_c: 
                    robots = []; draw.swarm_active = False; dynamic_obstacles = []
                    for i in range(ROWS):
                        for j in range(ROWS):
                            grid[i][j].is_barrier = False
                            grid[i][j].remove_dirt()

        
        # Q-LEARNING AUTO-CLEAN LOGIC (Sequential)
        if draw.q_learning_active and robots:
            robot = robots[0]
            # If robot is idle (not currently moving on a path)
            if not robot.following:
                target = find_nearest_dirt(grid, (robot.row, robot.col))
                if target:
                    # 1. Train Q-Learning for this specific target
                    Q = run_qlearning_demo_blocking(grid, (robot.row, robot.col), target, lambda i: qlearn_update_status(draw, i))
                    # 2. Extract Path
                    path = extract_path_from_q(Q, (robot.row, robot.col), target, grid)
                    # 3. Move
                    if path:
                        robot.current_path = path
                        robot.path_index = 0
                        robot.following = True
                    else:
                        print("Q-Learning path failed. Skipping this target.")
                else:
                    print("All dirt cleaned! Q-Learning Mode Stopped.")
                    draw.q_learning_active = False
                    draw.qlearn_status = {'qlearn_msg': 'DONE'}

        
        # SWARM OPTIMIZER LOGIC (Parallel)
        if draw.swarm_active:
            if sum(1 for r in grid for c in r if c.has_dirt) == 0:
                print("Swarm Job Complete.")
                draw.swarm_active = False
            
            for bot in robots:
                if not bot.current_path or bot.path_index >= len(bot.current_path):
                    bot.following = False
                    target = find_nearest_untargeted_dirt(grid, (bot.row, bot.col))
                    if target:
                        path = search_alg.astar((bot.row, bot.col), target, grid)
                        if path:
                            bot.current_path = path
                            bot.path_index = 0
                            bot.following = True
                            grid[target[0]][target[1]].targeted = True

        
        # ACTUATION LOGIC (All Robots)
        for robot in robots:
            # We draw the robot here
            # robot.draw(win) # Drawing handled after all path lines are drawn (in draw function)
            path = getattr(robot, 'current_path', [])
            
            if path and robot.following:
                if robot.path_index < len(path):
                    target_rc = path[robot.path_index]
                    
                    if getattr(grid[target_rc[0]][target_rc[1]], 'has_dynamic', False):
                        pass 
                    else:
                        gap = WIDTH // ROWS
                        tx = target_rc[1] * gap + gap // 2
                        ty = target_rc[0] * gap + gap // 2
                        if robot.move_towards(tx, ty):
                            robot.row, robot.col = target_rc
                            robot.x, robot.y = tx, ty
                            
                            grid[robot.row][robot.col].was_visited = True
                            
                            # Clean if dirt present (and not just SLAM exploring)
                            if grid[robot.row][robot.col].has_dirt and not draw.slam_exploring:
                                grid[robot.row][robot.col].remove_dirt()
                            
                            robot.path_index += 1
                            if robot.path_index >= len(path):
                                robot.following = False

        
        # AUTO-CLEAN (Manual A*)
        if draw.auto_clean and robots and not draw.swarm_active and not draw.q_learning_active:
            robot = robots[0]
            if not robot.following:
                target = find_nearest_dirt(grid, (robot.row, robot.col))
                if target:
                    if draw.algorithm == 'dijkstra':
                        path = search_alg.dijkstra((robot.row, robot.col), target, grid)
                    else:
                        path = search_alg.astar((robot.row, robot.col), target, grid)
                    if path:
                        robot.current_path = path
                        robot.path_index = 0
                        robot.following = True
                else:
                    draw.auto_clean = False
                    print("Auto Clean Finished.")

        
        # SLAM EXPLORATION
        
        if draw.slam_requested:
            draw.slam_exploring = True
            if not robots: 
                for _ in range(100):
                    rr, rc = random.randrange(ROWS), random.randrange(ROWS)
                    if not grid[rr][rc].is_barrier:
                        # --- UPDATED SLAM ROBOT CREATION ---
                        robots.append(Robot(rr, rc, WIDTH, ROWS, is_main=True, robot_img=robot_main_img)); break
                        # -----------------------------------
            
            robot = robots[0]
            
            for rr in range(max(0, robot.row-SENSOR_RADIUS), min(ROWS, robot.row+SENSOR_RADIUS+1)):
                for cc in range(max(0, robot.col-SENSOR_RADIUS), min(ROWS, robot.col+SENSOR_RADIUS+1)):
                    if abs(rr-robot.row)+abs(cc-robot.col) <= SENSOR_RADIUS:
                        if not grid[rr][cc].is_barrier: grid[rr][cc].explored = True

            free_cells = sum(1 for r in grid for c in r if not c.is_barrier)
            explored = sum(1 for r in grid for c in r if c.explored and not c.is_barrier)
            if free_cells > 0 and (explored / free_cells) >= 0.85:
                draw.slam_exploring = False
                draw.slam_requested = False
                print("SLAM Complete.")
            
            if not robot.following:
                best = None
                best_dist = float('inf')
                for i in range(ROWS):
                    for j in range(ROWS):
                        if not grid[i][j].explored and not grid[i][j].is_barrier:
                            d = abs(i - robot.row) + abs(j - robot.col)
                            if d < best_dist: best = (i, j); best_dist = d
                if best:
                    path = search_alg.astar((robot.row, robot.col), best, grid)
                    if path:
                        robot.current_path = path
                        robot.path_index = 0
                        robot.following = True

        # DYNAMIC OBSTACLES
        if dynamic_obstacles:
            now = time.time()
            for d in dynamic_obstacles:
                if now - d['last_move'] > DYNAMIC_MOVE_INTERVAL:
                    rr, rc = d['r'], d['c']
                    if grid[rr][rc].has_dynamic: grid[rr][rc].has_dynamic = False
                    
                    moves = [(0,1), (0,-1), (1,0), (-1,0), (0,0)]
                    random.shuffle(moves)
                    for dr, dc in moves:
                        nr, nc = rr+dr, rc+dc
                        if 0<=nr<ROWS and 0<=nc<ROWS and not grid[nr][nc].is_barrier:
                            if any(r.row == nr and r.col == nc for r in robots): continue
                            if getattr(grid[nr][nc], 'has_dynamic', False): continue
                            
                            d['r'], d['c'] = nr, nc
                            grid[nr][nc].has_dynamic = True
                            d['last_move'] = now
                            break

        pygame.display.flip()

    pygame.quit()


def headless_smoke_test():
    """Run a minimal headless test that exercises Q-learning and path extraction."""
    print("Running headless smoke test...")
    # Diagnostic: show which background image path is used and whether it exists
    try:
        print(f"Background image path (MAP_IMG_NAME): {MAP_IMG_NAME}")
        print(f"Background exists: {os.path.exists(MAP_IMG_NAME)}")
    except Exception as e:
        print(f"Could not check background image path: {e}")
    rows = ROWS
    grid = make_grid(rows, WIDTH)
    place_random_dirt(grid, 5)
    # Place a robot at center
    start = (rows//2, rows//2)
    # Find a dirt
    target = find_nearest_dirt(grid, start)
    if not target:
        print("No dirt placed; aborting.")
        return 1
    Q = run_qlearning_demo_blocking(grid, start, target, lambda i: qlearn_update_status(None, i))
    path = extract_path_from_q(Q, start, target, grid)
    if path:
        print(f"Headless path length: {len(path)}")
        return 0
    else:
        print("Failed to extract path from Q-table.")
        return 2


def run_display_test():
    """Attempt to open a small window, show a message and exit.
    This is a minimal diagnostic that helps users confirm whether their
    environment allows pygame windows to be created.
    """
    try:
        pygame.init()
        try:
            pygame.display.init()
        except Exception:
            pass
        win = pygame.display.set_mode((640, 240), pygame.RESIZABLE)
        pygame.display.set_caption('Cleaning-Robot Display Test')
        try:
            win.fill((30, 80, 160))
            try:
                f = pygame.font.SysFont(None, 28)
                m = f.render('Display test OK — window will close in 1s', True, (255,255,255))
                r = m.get_rect(center=(320, 120))
                win.blit(m, r)
            except Exception:
                pass
            for _ in range(10):
                try:
                    pygame.event.pump()
                except Exception:
                    pass
                try:
                    pygame.display.flip()
                except Exception:
                    pass
                pygame.time.wait(100)
        finally:
            try:
                pygame.quit()
            except Exception:
                pass
        print('Display test completed successfully.')
        return 0
    except Exception as e:
        print(f'Display test failed: {e}')
        try:
            pygame.quit()
        except Exception:
            pass
        return 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cleaning Robot Simulator')
    parser.add_argument('--nogui', action='store_true', help='Run a headless smoke test without pygame')
    parser.add_argument('--test-display', action='store_true', help='Run a short display test (open a small window and exit)')
    args = parser.parse_args()
    if args.nogui or not PYGAME_AVAILABLE:
        sys.exit(headless_smoke_test())
    if args.test_display:
        sys.exit(run_display_test())
    else:
        main()