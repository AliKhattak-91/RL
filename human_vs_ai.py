import pygame
import torch
import random
import numpy as np
from collections import namedtuple
from enum import Enum
from model import Linear_QNet

pygame.init()
font = pygame.font.SysFont('arial.ttf', 25)

# Direction Enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
GREEN1 = (0, 255, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 15

# Snake Class
class Snake:
    def __init__(self, x, y, color, is_ai=False, model=None):
        self.direction = Direction.RIGHT
        self.head = Point(x, y)
        self.snake = [self.head, Point(x - BLOCK_SIZE, y), Point(x - 2 * BLOCK_SIZE, y)]
        self.color = color
        self.score = 0
        self.alive = True
        self.is_ai = is_ai
        self.model = model

    def move(self, action):
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = directions[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = directions[(idx + 1) % 4]
        else:
            new_dir = directions[(idx - 1) % 4]

        self.direction = new_dir
        x, y = self.head.x, self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        self.snake.insert(0, self.head)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > 800 - BLOCK_SIZE or pt.x < 0 or pt.y > 600 - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def get_state(self, food):
        head = self.head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            food.x < head.x,
            food.x > head.x,
            food.y < head.y,
            food.y > head.y
        ]

        return np.array(state, dtype=int)

def generate_food(snakes):
    while True:
        x = random.randint(0, (800 - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (600 - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        food = Point(x, y)
        if not any(food in snake.snake for snake in snakes):
            return food

def draw_snake(display, snake):
    for pt in snake.snake:
        pygame.draw.rect(display, snake.color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(display, WHITE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

def show_scores(display, human, ai):
    text = font.render(f"Human: {human.score}    AI: {ai.score}", True, WHITE)
    display.blit(text, [10, 10])

def show_winner(human, ai):
    print("\nGAME OVER")
    if human.score == 3:
        print("Human wins!")
    elif ai.score == 10:
        print("AI wins!")
    else:
        print("It's a tie!")

def main():
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Snake: Human vs AI")
    clock = pygame.time.Clock()

    model = Linear_QNet(11, 256, 3)
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()

    human = Snake(100, 100, BLUE1)
    ai = Snake(500, 100, GREEN1, is_ai=True, model=model)

    snakes = [human, ai]
    food = generate_food(snakes)

    running = True
    while running:
        display.fill(BLACK)

       # Handle Human Input - Natural Directional Controls
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT] and human.direction != Direction.RIGHT:
            human.direction = Direction.LEFT
        elif keys[pygame.K_RIGHT] and human.direction != Direction.LEFT:
            human.direction = Direction.RIGHT
        elif keys[pygame.K_UP] and human.direction != Direction.DOWN:
            human.direction = Direction.UP
        elif keys[pygame.K_DOWN] and human.direction != Direction.UP:
            human.direction = Direction.DOWN

        # Always move straight in current direction
        human_action = [1, 0, 0]
        human.move(human_action)


        # AI Action
        state = ai.get_state(food)
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = ai.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        ai.move(final_move)

        # Eat food
        for snake in snakes:
            if snake.head == food:
                snake.score += 1
                food = generate_food(snakes)
            else:
                snake.snake.pop()

        # Win condition
        if human.score >= 20 or ai.score >= 20:
            show_winner(human, ai)
            running = False
            break

        # Check collisions
        for snake in snakes:
            if snake.is_collision():
                snake.alive = False

        if not human.alive or not ai.alive:
            show_winner(human, ai)
            running = False
            break 

        draw_snake(display, human)
        draw_snake(display, ai)
        pygame.draw.rect(display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        show_scores(display, human, ai)

        pygame.display.flip()
        clock.tick(SPEED)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

    pygame.time.wait(3000)
    pygame.quit()

if __name__ == "__main__":
    main()
