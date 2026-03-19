import pygame
import random
import numpy as np
import math

# ==========================================
# 1. GLOBAL LEARNING PARAMETERS
# ==========================================
ALPHA = 0.3  # Learning Rate: How fast it learns from new experiences
GAMMA = 0.95  # Discount Factor: How much it values future rewards
EPSILON = 1.0  # Starting Exploration: Chance of taking a random move
EPSILON_DECAY = 0.9999  # How fast to stop being random (0.999 = very slow)
MIN_EPSILON = 0.001  # Minimum randomness to keep the agent from getting stuck
GHOST_EPSILON = 0.3  # Maximum randomness to take from random movement for ghost

# ==========================================
# 2. GLOBAL GAME PARAMETERS
# ==========================================
TILE_SIZE = 24
TRAIN_FPS = 10000  # Increase this to 100+ to watch the agent learn at high speed
TEST_FPS = 7  # Increase this to 100+ to watch the agent learn at high speed
REWARD_DOT = 10
REWARD_POWER = 100
REWARD_DIE = -1000
REWARD_WIN = 1000
REWARD_STEP = -1  # Penalty for each move (encourages speed)

TRAIN_EPISODES = 100000
TEST_EPISODES = 1000

# Colors
BLACK, WALL_BLUE, YELLOW, RED, WHITE = (
    (0, 0, 0),
    (0, 0, 180),
    (255, 255, 0),
    (255, 0, 0),
    (255, 255, 255),
)

# 1: Wall, 0: Dot, 2: Power Pellet, 3: Empty Path
# MEDIUM_CLASSIC_MAZE = [
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#     [1, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 2, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 1, 1, 1, 0, 1, 1, 1, 3, 3, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#     [1, 0, 0, 0, 1, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 1, 0, 0, 0, 1],
#     [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
#     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# ]
# PAC_MAN_INIT_POS = (7, 9)
# GHOST_INIT_POS = (7, 1)
MEDIUM_CLASSIC_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 2, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 2, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
PAC_MAN_INIT_POS = [4, 9]
GHOST_INIT_POS = [7, 1]

MOVE_ACTION = [(1, 0), (0, 1), (-1, 0), (0, -1)]

MAZE_WIDTH = len(MEDIUM_CLASSIC_MAZE[0])
MAZE_HEIGHT = len(MEDIUM_CLASSIC_MAZE)
SCREEN_WIDTH = MAZE_WIDTH * TILE_SIZE
SCREEN_HEIGHT = (MAZE_HEIGHT * TILE_SIZE) + 60


def draw_ghost(x, y, color):
    """
    Draws a Pac-Man style ghost at a specific (x, y) position.
    The ghost is roughly 20 pixels wide.
    """
    radius = 8
    # Draw the main body (circle on top, rectangle on bottom)
    pygame.draw.circle(screen, color, (x, y - radius), radius)
    pygame.draw.rect(screen, color, (x - radius, y - radius, radius * 2, radius))

    # Draw the "legs" (simple jagged bottom using polygons)
    leg_height = 5
    num_legs = 4
    for i in range(num_legs):
        # Calculate points for each leg segment
        left_x = x - radius + (i * (2 * radius / num_legs))
        right_x = x - radius + ((i + 1) * (2 * radius / num_legs))
        top_y = y + leg_height
        bottom_y = y
        # Alternate top and bottom points for a jagged look
        if i % 2 == 0:
            points = [
                (left_x, bottom_y),
                (right_x, bottom_y),
                (right_x, top_y),
                (left_x, top_y),
            ]
        else:
            points = [
                (left_x, top_y),
                (right_x, top_y),
                (right_x, bottom_y),
                (left_x, bottom_y),
            ]
        pygame.draw.polygon(screen, color, points)

    # Draw the eyes (simple white circles with black pupils)
    eye_radius = 3
    pupil_radius = 1
    # Left eye
    pygame.draw.circle(screen, WHITE, (x - radius // 2, y - radius // 2), eye_radius)
    pygame.draw.circle(screen, BLACK, (x - radius // 2, y - radius // 2), pupil_radius)
    # Right eye
    pygame.draw.circle(screen, WHITE, (x + radius // 2, y - radius // 2), eye_radius)
    pygame.draw.circle(screen, BLACK, (x + radius // 2, y - radius // 2), pupil_radius)


class PacMan:
    def __init__(self, x, y, radius, color, direction):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mouth_angle = 45  # Angle for the mouth opening (half of total angle)
        self.direction = direction  # 0: right, 1: up, 2: left, 3: down

    def draw(self, surface):
        # The mouth animation can be controlled by changing self.mouth_angle over time

        # 1. Draw the main body (a circle)
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.radius)

        # 2. Determine the mouth angles based on direction
        # The angles for the "cutout" polygon
        if self.direction == 0:  # Right
            start_angle = -self.mouth_angle
            end_angle = self.mouth_angle
        elif self.direction == 1:  # Up
            start_angle = 90 - self.mouth_angle
            end_angle = 90 + self.mouth_angle
        elif self.direction == 2:  # Left
            start_angle = 180 - self.mouth_angle
            end_angle = 180 + self.mouth_angle
        elif self.direction == 3:  # Down
            start_angle = 270 - self.mouth_angle
            end_angle = 270 + self.mouth_angle

        # 3. Create the "mouth" as a polygon that matches the background color (e.g., BLACK)
        # The polygon will be a triangle with points at the center, and two points on the edge of the circle.

        center = (self.x, self.y)
        # Convert degrees to radians for math functions
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)

        # Calculate the two points on the edge of the circle that define the mouth
        point1 = (
            self.x + (self.radius + 3) * math.cos(start_rad),
            self.y + (self.radius + 3) * math.sin(start_rad),
        )
        point2 = (
            self.x + (self.radius + 3) * math.cos(end_rad),
            self.y + (self.radius + 3) * math.sin(end_rad),
        )

        # Draw the triangle (mouth) in the background color to create the open effect
        BACKGROUND_COLOR = (0, 0, 0)  # Assuming a black background
        pygame.draw.polygon(surface, BACKGROUND_COLOR, [center, point1, point2])


class GameEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.maze = [row[:] for row in MEDIUM_CLASSIC_MAZE]
        self.pacman = [PAC_MAN_INIT_POS[0], PAC_MAN_INIT_POS[1]]
        self.ghost = [GHOST_INIT_POS[0], GHOST_INIT_POS[1]]
        self.score = 0
        return self.get_state()

    def get_state(self):
        # We track pac_man, ghost's position and remaining dots and powers
        # remaining_dots = sum(row.count(0) for row in self.maze)
        # remaining_powers = sum(row.count(2) for row in self.maze)
        flatten_grid = [column for row in self.maze for column in row]
        flatten_grid_str = "".join(map(str, flatten_grid))
        return (
            self.pacman[0],
            self.pacman[1],
            self.ghost[0],
            self.ghost[1],
            flatten_grid_str,
        )

    def is_walkable(self, x, y):
        if 0 <= x < MAZE_WIDTH and 0 <= y < MAZE_HEIGHT:
            return self.maze[y][x] != 1
        return False

    def move_ghost(self):
        best_move = self.ghost
        min_dist = float("inf")
        for dx, dy in MOVE_ACTION:
            nx, ny = self.ghost[0] + dx, self.ghost[1] + dy
            if self.is_walkable(nx, ny):
                if random.random() < GHOST_EPSILON:
                    best_move = [nx, ny]
                    break
                else:
                    dist = abs(nx - self.pacman[0]) + abs(ny - self.pacman[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_move = [nx, ny]
        self.ghost = best_move

    def step(self, action):
        dx, dy = MOVE_ACTION[action]
        if self.is_walkable(self.pacman[0] + dx, self.pacman[1] + dy):
            self.pacman[0] += dx
            self.pacman[1] += dy

        if self.pacman == self.ghost:
            reward = REWARD_DIE
            return self.get_state(), reward, True

        reward = REWARD_STEP
        tile = self.maze[self.pacman[1]][self.pacman[0]]
        if tile == 0:
            self.maze[self.pacman[1]][self.pacman[0]] = 3
            reward = REWARD_DOT
        elif tile == 2:
            self.maze[self.pacman[1]][self.pacman[0]] = 3
            reward = REWARD_POWER

        self.score += reward
        self.move_ghost()

        done = False
        if self.pacman == self.ghost:
            reward = REWARD_DIE
            done = True
        elif not any(0 in row or 2 in row for row in self.maze):
            reward = REWARD_WIN
            done = True

        return self.get_state(), reward, done


# --- Main Logic ---
q_table = {}
current_epsilon = EPSILON


def get_q(s):
    if s not in q_table:
        q_table[s] = np.zeros(4)
    return q_table[s]


env = GameEnv()
pygame.init()


# --- Train Logic ---
for ep in range(TRAIN_EPISODES):
    state = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Epsilon-Greedy Choice
        if random.random() < current_epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(get_q(state))

        next_state, reward, done = env.step(action)

        # Q-Learning Equation
        target = reward + GAMMA * np.max(get_q(next_state))
        get_q(state)[action] += ALPHA * (target - get_q(state)[action])
        state = next_state

        pygame.time.Clock().tick(TRAIN_FPS)

    print(f"Train Episode-{ep} Score: {env.score}, Epsilon: {current_epsilon}")

    # Decay Epsilon after each episode
    current_epsilon = max(MIN_EPSILON, current_epsilon * EPSILON_DECAY)


# --- Test Logic ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snow Pacman - Q-Learning")
font = pygame.font.SysFont("Arial Black", 26)

current_epsilon = 0.03

for ep in range(TEST_EPISODES):
    state = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Epsilon-Greedy Choice
        if random.random() < current_epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(get_q(state))

        next_state, reward, done = env.step(action)
        state = next_state

        # Rendering
        screen.fill(BLACK)
        for r in range(MAZE_HEIGHT):
            for c in range(MAZE_WIDTH):
                tile = env.maze[r][c]
                x, y = c * TILE_SIZE, r * TILE_SIZE
                if tile == 1:
                    pygame.draw.rect(
                        screen,
                        WALL_BLUE,
                        (x + 1, y + 1, TILE_SIZE - 2, TILE_SIZE - 2),
                        1,
                    )
                elif tile == 0:
                    pygame.draw.circle(
                        screen, WHITE, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), 2
                    )
                elif tile == 2:
                    pygame.draw.circle(
                        screen, WHITE, (x + TILE_SIZE // 2, y + TILE_SIZE // 2), 6
                    )

        # Draw pac man
        pacman = PacMan(
            env.pacman[0] * TILE_SIZE + 12,
            env.pacman[1] * TILE_SIZE + 12,
            10,
            YELLOW,
            action,
        )
        pacman.draw(screen)

        # Draw ghost
        draw_ghost(
            env.ghost[0] * TILE_SIZE + 12,
            env.ghost[1] * TILE_SIZE + 15,
            RED,
        )

        # UI Info
        score_txt = font.render(f"SCORE: {int(env.score)}", True, YELLOW)
        eps_txt = font.render(f"EPS: {current_epsilon:.2f}", True, WHITE)
        screen.blit(score_txt, (10, SCREEN_HEIGHT - 40))
        screen.blit(eps_txt, (SCREEN_WIDTH - 110, SCREEN_HEIGHT - 40))

        pygame.display.flip()
        pygame.time.Clock().tick(TEST_FPS)

    print(f"Test Episode-{ep}: Score: {env.score}")

    pygame.time.delay(3000)

while True:
    pass

# pygame.quit()
