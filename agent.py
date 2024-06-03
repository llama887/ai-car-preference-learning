# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)
# Code Adapted for preference learning for Emerge Lab research by Franklin Yiu

import math
import random
import sys
import os
import glob

import neat
import pygame

import argparse

import pickle

from reward import TrajectoryRewardNet, prepare_single_trajectory
import torch

# Constants

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit

TRAJECTORY_LENGTH = 30 * 15

current_generation = 0  # Generation counter
saved_trajectory_count = 0  # Counter for saved trajectories
trajectory_path = "./trajectories/"
reward_network = None


class Car:
    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load("car.png").convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = [830, 920]  # Starting Position
        self.angle = 0
        self.speed = 0

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [
            self.position[0] + CAR_SIZE_X / 2,
            self.position[1] + CAR_SIZE_Y / 2,
        ]  # Calculate Center

        self.radars = []  # List For Sensors / Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.alive = True  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed

        self.trajectory = []  # All Positions of the Car

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        self.draw_radar(screen)  # OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + degree))) * length
        )
        y = int(
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + degree))) * length
        )

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + degree))) * length
            )
            y = int(
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + degree))) * length
            )

        # Calculate Distance To Border And Append To Radars List
        dist = int(
            math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2))
        )
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [
            int(self.position[0]) + CAR_SIZE_X / 2,
            int(self.position[1]) + CAR_SIZE_Y / 2,
        ]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length,
        ]
        right_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length,
        ]
        left_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length,
        ]
        right_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length,
        ]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

        self.trajectory.append(self.position.copy())

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def get_reward(self):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        if reward_network is not None:
            trajectory_tensor = prepare_single_trajectory(self.trajectory)
            reward = reward_network(trajectory_tensor)
            return reward.item()
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

    def save_trajectory(self, filename):
        with open(filename, "wb") as f:
            pickle.dump((self.get_reward(), self.trajectory), f)
            # self.distance is used to calculate the true reward


def generate_database(trajectory_path):
    # Load All Trajectories
    trajectories = []
    for file in glob.glob(f"{trajectory_path}/trajectory*.pkl"):
        with open(file, "rb") as f:
            reward, trajectory = pickle.load(f)
            trajectories.append((reward, trajectory))

    max_length, max_index = max(
        (len(trajectory), index)
        for index, (reward, trajectory) in enumerate(trajectories)
    )

    random.shuffle(trajectories)
    if len(trajectories) % 2 != 0:
        trajectories.pop()

    # Pads shorter tajectoires so there is a consistent input size
    def pad_trajectory(trajectory, max_length):
        return trajectory + [trajectory[-1]] * (max_length - len(trajectory))

    trajectory_pairs = [
        (
            pad_trajectory(trajectories[i][1], max_length),
            pad_trajectory(trajectories[i + 1][1], max_length),
            0 if trajectories[i][0] > trajectories[i + 1][0] else 1,
            trajectories[i][0],
            trajectories[i + 1][0],
        )
        for i in range(0, len(trajectories), 2)
    ]
    print(f"Generating Database with {len(trajectory_pairs)} trajectory pairs...")

    # Delete all trajectories
    print("Removing saved trajectories...")
    old_trajectories = glob.glob(trajectory_path + "trajectory*")
    for f in old_trajectories:
        os.remove(f)

    # Delete old database
    print("Removing old database...")
    old_trajectories = glob.glob(trajectory_path + "database*")
    for f in old_trajectories:
        os.remove(f)

    # Save To Database
    with open(trajectory_path + f"database_{len(trajectory_pairs)}.pkl", "wb") as f:
        pickle.dump(trajectory_pairs, f)


def run_simulation(genomes, config):

    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load("maps/map.png").convert()  # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0
    global saved_trajectory_count

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2  # Slow Down
            else:
                car.speed += 2  # Speed Up

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == TRAJECTORY_LENGTH:  # Stop After About 7 Seconds
            if (
                number_of_trajectories > 0
                and saved_trajectory_count < number_of_trajectories
            ):
                for i, car in enumerate(cars):
                    if saved_trajectory_count >= number_of_trajectories:
                        break
                    car.save_trajectory(
                        f"{trajectory_path}trajectory_{current_generation}_{i}.pkl"
                    )
                    print("Saved trajectory")
                    saved_trajectory_count += 1
            break
        if (
            number_of_trajectories > 0
            and saved_trajectory_count >= number_of_trajectories
        ):
            print(f"Saved {saved_trajectory_count} trajectories to {trajectory_path}.")
            generate_database(trajectory_path)
            sys.exit(0)
        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = generation_font.render(
            "Generation: " + str(current_generation), True, (0, 0, 0)
        )
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="AI Car Preference Learning")
    parse.add_argument(
        "-t",
        "--trajectories",
        type=int,
        nargs=1,
        default=[-1],
        help="Number of trajectories to save",
    )
    parse.add_argument(
        "-r",
        "--reward",
        type=str,
        help="Directory to reward function weights",
    )
    args = parse.parse_args()

    if args.reward and args.trajectories[0] > 0:
        print("Cannot save trajectories and train reward function at the same time")
        sys.exit(1)

    if args.reward is not None:
        print("Loading reward network...")
        reward_network = TrajectoryRewardNet(TRAJECTORY_LENGTH * 2)
        weights = torch.load(args.reward)
        reward_network.load_state_dict(weights)

    number_of_trajectories = [-1]
    if args.trajectories is not None:
        number_of_trajectories = args.trajectories[0]
        if number_of_trajectories > 0:
            print("Removing old trajectories...")
            old_trajectories = glob.glob(trajectory_path + "trajectory*")
            for f in old_trajectories:
                os.remove(f)
            print(f"Saving {number_of_trajectories} trajectories...")

    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run Simulation For A Maximum of 1000 Generations
    population.run(
        run_simulation, 1000 if number_of_trajectories < 0 else number_of_trajectories
    )
