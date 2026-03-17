import pygame
import random
import numpy as np
import sys

# ==========================================
# 1. GLOBAL LEARNING PARAMETERS (The "Brain")
# ==========================================
ALPHA = 0.1  # Learning Rate: How fast it learns from new experiences
GAMMA = 0.9  # Discount Factor: How much it values future rewards
EPSILON = 0.5  # Starting Exploration: Chance of taking a random move
EPSILON_DECAY = 0.999  # How fast to stop being random (0.999 = very slow)
MIN_EPSILON = 0.01  # Minimum randomness to keep the agent from getting stuck

# ==========================================
# 2. GLOBAL GAME PARAMETERS (The "World")
# ==========================================
TILE_SIZE = 24
FPS = 10  # Increase this to 100+ to watch the agent learn at high speed
REWARD_DOT = 10
REWARD_POWER = 50
REWARD_DIE = -500
REWARD_WIN = 1000
REWARD_STEP = -1  # Penalty for each move (encourages speed)

# Colors
BLACK, WALL_BLUE, YELLOW, RED, WHITE = (
    (0, 0, 0),
    (0, 0, 180),
    (255, 255, 0),
    (255, 0, 0),
    (255, 255, 255),
)

# 1: Wall, 0: Dot, 2: Power Pellet, 3: Empty Path
MEDIUM_CLASSIC_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 2, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 3, 3, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

MAZE_WIDTH = len(MEDIUM_CLASSIC_MAZE[0])
MAZE_HEIGHT = len(MEDIUM_CLASSIC_MAZE)
SCREEN_WIDTH = MAZE_WIDTH * TILE_SIZE
SCREEN_HEIGHT = (MAZE_HEIGHT * TILE_SIZE) + 60


class GameEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.maze = [row[:] for row in MEDIUM_CLASSIC_MAZE]
        self.pacman = [10, 7]
        self.ghost = [10, 5]
        self.score = 0
        return self.get_state()

    def get_state(self):
        # We track relative distance to the ghost
        return (self.ghost[0] - self.pacman[0], self.ghost[1] - self.pacman[1])

    def is_walkable(self, x, y):
        if 0 <= x < MAZE_WIDTH and 0 <= y < MAZE_HEIGHT:
            return self.maze[y][x] != 1
        return False

    def move_ghost(self):
        best_move = self.ghost
        min_dist = float("inf")
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = self.ghost[0] + dx, self.ghost[1] + dy
            if self.is_walkable(nx, ny):
                dist = abs(nx - self.pacman[0]) + abs(ny - self.pacman[1])
                if dist < min_dist:
                    min_dist = dist
                    best_move = [nx, ny]
        self.ghost = best_move

    def step(self, action):
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        if self.is_walkable(self.pacman[0] + dx, self.pacman[1] + dy):
            self.pacman[0] += dx
            self.pacman[1] += dy

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
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snow Pacman - Q-Learning")
font = pygame.font.SysFont("Arial Black", 30)
env = GameEnv()
q_table = {}
current_epsilon = EPSILON


def get_q(s):
    if s not in q_table:
        q_table[s] = np.zeros(4)
    return q_table[s]


running = True
episode = 1

while running:
    state = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
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

        pygame.draw.circle(
            screen,
            YELLOW,
            (env.pacman[0] * TILE_SIZE + 11, env.pacman[1] * TILE_SIZE + 11),
            9,
        )
        pygame.draw.rect(
            screen,
            RED,
            (
                env.ghost[0] * TILE_SIZE + 4,
                env.ghost[1] * TILE_SIZE + 4,
                TILE_SIZE - 8,
                TILE_SIZE - 8,
            ),
        )

        # UI Info
        score_txt = font.render(f"SCORE: {int(env.score)}", True, YELLOW)
        eps_txt = font.render(f"EPS: {current_epsilon:.2f}", True, WHITE)
        screen.blit(score_txt, (10, SCREEN_HEIGHT - 40))
        screen.blit(eps_txt, (SCREEN_WIDTH - 110, SCREEN_HEIGHT - 40))

        pygame.display.flip()
        pygame.time.Clock().tick(FPS)

    # Decay Epsilon after each episode
    current_epsilon = max(MIN_EPSILON, current_epsilon * EPSILON_DECAY)
    episode += 1

    print(f"Total Score: {env.score}")

pygame.quit()
