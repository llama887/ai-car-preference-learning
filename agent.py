# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)
# Code Adapted for preference learning for Emerge Lab research by Franklin Yiu & Alex Tang

import argparse
import glob
import math
import os
import pickle
import random
import sys
import time
from collections import deque

import neat
import pygame
import torch
import yaml
from orientation.get_orientation import get_angle
import reward
import rules
from reward import Ensemble, TrajectoryRewardNet, prepare_single_trajectory
from segment import StateActionPair
from itertools import combinations
from collections import defaultdict

os.environ["SDL_AUDIODRIVER"] = "dummy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trajectories_path = "trajectories/"
master_database = None
paired_database = None

# Constants

WIDTH = 1920
HEIGHT = 1080

AGENTS_PER_GENERATION = 20

CAR_SIZE_X = 60
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit

TRAJECTORY_LENGTH = 30 * 6
train_trajectory_length = 1
NUM_RADARS = 5
STATE_ACTION_SIZE = 8

DEFAULT_MAX_GENERATIONS = 1000
ENSEMBLE_MULTIPLIER = 5

current_generation = 0  # Generation counter
reward_network = None
ensemble = None
num_pairs = 0
population = ""
run_type = "collect"
headless = False
subsample = True
saved_segments = []
saved_trajectories = []
big_car_best_distance = 0
big_car_distance = 0
game_map = None
rules_followed = []


class Trajectory:
    def __init__(self, s, t, r):
        self.num_satisfaction_segments = s
        self.traj = t.copy()
        self.total_reward = r

    def truncate(self, trajectory, truncate_length=3):
        trajectory_length = len(trajectory)
        truncated_trajectory = trajectory[:truncate_length]
        if trajectory_length > truncate_length:
            truncated_trajectory.append(
                f"...{trajectory_length - truncate_length} more StateActionPairs"
            )
        return truncated_trajectory

    def __repr__(self):
        return (
            "Trajectory:\n"
            + str(self.truncate(self.traj))
            + "\n"
            + "Number of Satisfaction Segments: "
            + str(self.num_satisfaction_segments)
            + "\n"
            + "Reward: "
            + str(self.total_reward)
            + "\n"
        )

class Car:
    def __init__(self, position=[830, 920], angle=0, color="blue"):
        # Load Car Sprite and Rotate
        if color == "red":
            self.sprite = pygame.image.load("sprites/red_car.png").convert()
        else:
            self.sprite = pygame.image.load(
                "sprites/car.png"
            ).convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.color = color
        self.position = position  # Starting Position
        self.angle = angle
        self.speed = 0
        self.rules_per_step = []
        self.num_satisfaction_segments = 0

        self.speed_set = False  # Flag For Default Speed Later on
        self.radar_max = 10000

        self.center = [
            self.position[0] + CAR_SIZE_X / 2,
            self.position[1] + CAR_SIZE_Y / 2,
        ]  # Calculate Center

        self.radars = []  # List For Sensors / Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.alive = True  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.reward = 0
        self.time = 0  # Time Passed
        self.id = 0

        self.trajectory = []  # All Positions of the Car

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        # self.draw_radar(screen)  # OPTIONAL FOR SENSORS

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
            global i
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                self.speed = 0
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
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < self.radar_max:
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
        return dist

    def update(self, game_map, action):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if self.alive:
            if not self.speed_set:
                self.speed = 20
                self.speed_set = True

                first_state_action = StateActionPair(
                    [self.check_radar(d, game_map) for d in range(-90, 120, 45)],
                    action,
                    [830, 920],
                    True,
                )
                self.trajectory.append(first_state_action)

            # Get Rotated Sprite And Move Into The Right X-Direction
            # Don't Let The Car Go Closer Than 20px To The Edge
            self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
            self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
            self.position[0] = max(self.position[0], 20)
            self.position[0] = min(self.position[0], WIDTH - 120)

            # Increase Distance and Time
            self.distance += self.speed
            self.reward += self.get_reward()
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
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + 30))) * length,
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + 30))) * length,
            ]
            right_top = [
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + 150))) * length,
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + 150))) * length,
            ]
            left_bottom = [
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + 210))) * length,
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + 210))) * length,
            ]
            right_bottom = [
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + 330))) * length,
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + 330))) * length,
            ]
            self.corners = [left_top, right_top, left_bottom, right_bottom]

            # Check Collisions And Clear Radars
            self.check_collision(game_map)
        self.radars.clear()

        radar_dists = []
        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            radar_dists.append(self.check_radar(d, game_map))
        next_state_action = StateActionPair(
            radar_dists, action, self.position.copy(), self.alive
        )
        self.trajectory.append(next_state_action)

        rules_satisfied, is_satisfaction, _ = rules.check_rules_one(
            self.trajectory[-2:], rules.NUMBER_OF_RULES
        )
        self.rules_per_step.append(rules_satisfied)
        self.num_satisfaction_segments += is_satisfaction
        return rules_followed

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
        if len(self.trajectory) < train_trajectory_length + 1:
            return 0
        if ensemble:
            trajectory_tensor = prepare_single_trajectory(
                self.trajectory, train_trajectory_length + 1
            )
            return ensemble(trajectory_tensor).item()
        elif reward_network:
            trajectory_tensor = prepare_single_trajectory(
                self.trajectory, train_trajectory_length + 1
            )
            return reward_network(trajectory_tensor).item()

        return rules.check_rules_one(self.trajectory[-2:], rules.NUMBER_OF_RULES)[1]

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

    def save_trajectory(self):
        global run_type, saved_segments, saved_trajectories
        if run_type == "collect":
            global num_pairs
            status = collection_status(saved_segments, num_pairs)
            new_segments = break_into_segments(
                self.trajectory, self.rules_per_step, status
            )
            for i in range(rules.NUMBER_OF_RULES + 1):
                saved_segments[i].extend(new_segments[i])
        else:
            saved_trajectories.append(
                (self.num_satisfaction_segments, self.trajectory, self.reward)
            )


def display_saved_segments(saved_segments):
    print("SAVED SEGMENTS CURRENTLY HAS:")
    if not saved_segments:
        print("NO SEGMENTS")
    for i, seg in enumerate(saved_segments):
        print(i, "RULE SEGMENTS:", len(seg))
    print()


def display_requested_segments(number_of_pairs):
    print(
        "SEGMENTS REQUESTED (Pairs Requested * 2 * distribution[i]) (x10 for ensemble)):"
    )
    for i in range(rules.NUMBER_OF_RULES + 1):
        print(
            i,
            "RULE SEGMENTS:",
            math.ceil(number_of_pairs * 2 * rules.SEGMENT_DISTRIBUTION_BY_RULES[i]),
        )
    print()


def break_into_segments(trajectory, rules_per_step, done):
    global train_trajectory_length
    trajectory_segments = [[] for _ in range(rules.NUMBER_OF_RULES + 1)]
    if len(trajectory) < train_trajectory_length + 1:
        return

    current_segment = trajectory[: train_trajectory_length + 1]
    current_rule_sum = sum(rules_per_step[:train_trajectory_length])
    current_rule_avg = int(round(current_rule_sum // train_trajectory_length))
    trajectory_segments[current_rule_avg].append(current_segment)
    current_segment = deque(current_segment)

    for i in range(train_trajectory_length + 1, len(trajectory)):
        current_segment.popleft()
        current_segment.append(trajectory[i])
        if trajectory[i].action == -1:
            break
        current_rule_sum += (
            rules_per_step[i - 1] - rules_per_step[i - 1 - train_trajectory_length]
        )
        current_rule_avg = int(round(current_rule_sum // train_trajectory_length))
        trajectory_segments[current_rule_avg].append(list(current_segment))

    for i in range(rules.NUMBER_OF_RULES + 1):
        if done[i]:
            trajectory_segments[i] = []
    return trajectory_segments


def dist(traj_segment):
    position_segment = [traj_segment[0].position, traj_segment[1].position]
    traj_segment_distance = math.sqrt(
        (position_segment[1][0] - position_segment[0][0]) ** 2
        + (position_segment[1][1] - position_segment[0][1]) ** 2
    )
    return traj_segment_distance


def sort_and_pair(trajectory_segments, clean=True):
    sorted_trajectory_segments = sorted(trajectory_segments, key=lambda x: dist(x))
    distDict = {}
    cleaned_segments = []
    num_limit = 4
    for trajectory_segment in sorted_trajectory_segments:
        trajectory_distance = round(dist(trajectory_segment))
        if (
            trajectory_distance not in distDict
            or distDict[trajectory_distance] < num_limit
        ):
            cleaned_segments.append(trajectory_segment)
        distDict[trajectory_distance] = 1 + distDict.get(trajectory_distance, 0)
    segments_to_return = cleaned_segments if clean else sorted_trajectory_segments
    for i in range(len(segments_to_return)):
        print(
            segments_to_return[i],
            "; DIST:",
            dist(segments_to_return[i]),
        )
    return segments_to_return


def calculate_new_point(point, distance, angle):
    x0, y0 = point
    # Convert angle from degrees to radians
    angle_rad = math.radians(angle)
    # Calculate new coordinates
    x1 = x0 + distance * math.cos(angle_rad)
    y1 = y0 + distance * math.sin(angle_rad)
    return [x1, y1]


def sample_segments(saved_segments):
    global num_pairs
    sampled_segments = [[] for _ in range(rules.NUMBER_OF_RULES + 1)]
    for i in range(rules.NUMBER_OF_RULES + 1):
        segments_needed = math.ceil(
            num_pairs * 2 * rules.SEGMENT_DISTRIBUTION_BY_RULES[i]
        )
        sampled_segments[i] = random.sample(saved_segments[i], segments_needed)
    return sampled_segments



# STALE
def generate_database_from_segments(segments, number_of_pairs):
    global trajectories_path
    trajectory_path = trajectories_path
    os.makedirs(trajectories_path, exist_ok=True)
    trajectory_pairs = []
    trajectory_segments = []
    print(
        "SEGMENT POOL FOR DATA COLLECTION:",
        list((f"{i}: {len(seg)}" for i, seg in enumerate(segments))),
    )
    for segments in segments:
        trajectory_segments.extend(segments)

    print(f"Collected {len(trajectory_segments)} segments.")
    if len(trajectory_segments) % 2 != 0:
        trajectory_segments.pop()

    random.shuffle(trajectory_segments)
    same_reward = 0
    for i in range(0, len(trajectory_segments), 2):
        _, reward_1, _ = rules.check_rules_long_segment(
            trajectory_segments[i], rules.NUMBER_OF_RULES
        )
        _, reward_2, _ = rules.check_rules_long_segment(
            trajectory_segments[i + 1], rules.NUMBER_OF_RULES
        )
        if reward_1 == reward_2:
            same_reward += 1
        trajectory_pairs.append(
            (
                list(trajectory_segments[i]),
                list(trajectory_segments[i + 1]),
                0 if reward_1 < reward_2 else 1,
                reward_1,
                reward_2,
            )
        )

    n = len(trajectory_pairs)
    print(
        "Generated",
        n - same_reward,
        f"pairs with different rewards ({(n - same_reward) / n}%)",
    )

    # Delete old database if it is redundant (same size)
    old_pairs_path = (
        trajectory_path
        + f"database_{number_of_pairs}_pairs_{rules.NUMBER_OF_RULES}_rules_{train_trajectory_length}_length.pkl"
    )
    if os.path.exists(old_pairs_path):
        print("Removing old database with same pairs and rules...")
        os.remove(old_pairs_path)

    database_to_save = (
        paired_database if
        paired_database else 
        trajectory_path
        + f"database_{len(trajectory_pairs)}_pairs_{rules.NUMBER_OF_RULES}_rules_{train_trajectory_length}_length.pkl"
    )
    
    # Save To Database
    with open(
        database_to_save,
        "wb",
    ) as f:
        pickle.dump(trajectory_pairs, f)

    return len(trajectory_segments)


def generate_database(trajectory_path, saved_data, datatype, segment_generation_mode='random'):
    trajectory_pairs = []

    global \
        run_type, \
        num_pairs, \
        train_trajectory_length, \
        master_database
    
    if datatype == "segments":
        # Break trajectories into trajectory segments
        trajectory_segments = []
        sampled_segments = sample_segments(saved_data)
        print(
            "SEGMENT POOL FOR DATA COLLECTION:",
            list((f"{i}: {len(seg)}" for i, seg in enumerate(sampled_segments))),
        )
        for segments in sampled_segments:
            trajectory_segments.extend(segments)

        if len(trajectory_segments) % 2 != 0:
            trajectory_segments.pop()

        if segment_generation_mode == "random":
            random.shuffle(trajectory_segments)
            same_reward = 0
            for i in range(0, num_pairs * 2, 2):
                _, reward_1, _ = rules.check_rules_long_segment(
                    trajectory_segments[i], rules.NUMBER_OF_RULES
                )
                _, reward_2, _ = rules.check_rules_long_segment(
                    trajectory_segments[i + 1], rules.NUMBER_OF_RULES
                )
                if reward_1 == reward_2:
                    same_reward += 1
                trajectory_pairs.append(
                    (
                        list(trajectory_segments[i]),
                        list(trajectory_segments[i + 1]),
                        0 if reward_1 < reward_2 else 1,
                        reward_1,
                        reward_2,
                    )
                )

            n = len(trajectory_pairs)
            print(
                "Generated",
                n - same_reward,
                f"pairs with different rewards ({(n - same_reward) / n}%)",
            )

            if n > num_pairs:
                print("too many pairs!")
                for i in range(n - num_pairs):
                    trajectory_pairs.pop()

        elif segment_generation_mode == "different":
            random.shuffle(trajectory_segments)
            same_reward = []
            for i in range(0, len(trajectory_segments), 2):
                _, reward_1, _ = rules.check_rules_long_segment(
                    trajectory_segments[i], rules.NUMBER_OF_RULES
                )
                _, reward_2, _ = rules.check_rules_long_segment(
                    trajectory_segments[i + 1], rules.NUMBER_OF_RULES
                )
                if reward_1 == reward_2:
                    same_reward.append(
                        (
                            list(trajectory_segments[i]),
                            list(trajectory_segments[i + 1]),
                            0 if reward_1 < reward_2 else 1,
                            reward_1,
                            reward_2,
                        )
                    )
                else:
                    trajectory_pairs.append(
                        (
                            list(trajectory_segments[i]),
                            list(trajectory_segments[i + 1]),
                            0 if reward_1 < reward_2 else 1,
                            reward_1,
                            reward_2,
                        )
                    )
            random.shuffle(same_reward)

            n = len(trajectory_pairs)
            print("Generated", n, "pairs with different reward.")
            for i in range(n - num_pairs):
                trajectory_pairs.pop()
            fill = min(num_pairs - n, len(same_reward))
            print("Remaining", fill, "pairs are between segments with same reward.")
            for i in range(fill):
                trajectory_pairs.append(same_reward[i])

        print(f"Generating Database with {len(trajectory_pairs)} trajectory pairs...")

        database_to_save = (
            paired_database if
            paired_database else 
            trajectory_path
            + f"database_{len(trajectory_pairs)}_pairs_{rules.NUMBER_OF_RULES}_rules_{train_trajectory_length}_length.pkl"
        )

        # Delete old database if it is redundant (same size)
        if os.path.exists(database_to_save):
            print("Removing old database with same pairs and rules...")
            os.remove(database_to_save)
        
        # Save To Database
        with open(
            database_to_save,
            "wb",
        ) as f:
            pickle.dump(trajectory_pairs, f)

        if master_database and not paired_database:
            # Create a temporary file path
            temp_database = f"{trajectory_path}_master_temp_{len(trajectory_pairs)}.pkl"

            # Save new master database to a temporary file
            with open(temp_database, "wb") as f:
                new_database = save_as_database(saved_data)
                pickle.dump(new_database, f)

            if os.path.exists(temp_database):
                try:
                    print("Replacing old master database with the new one...")
                    if os.path.exists(master_database):
                        print(f"Removing old master database... {master_database}")
                        os.remove(master_database)
                    os.rename(temp_database, master_database)
                    print("Master database updated successfully.")
                except Exception as e:
                    print(f"Error during database update: {e}")
            else:
                print("Failed to create the temporary database file.")

        return len(trajectory_pairs)
    
    else:
        trajectories = []
        for i in range(len(saved_data)):
            trajectories.append(
                Trajectory(
                    s=saved_data[i][0],
                    t=saved_data[i][1],
                    r=saved_data[i][2],
                )
            )

        trajectory_file_path = ""
        if run_type == "trueRF":
            os.makedirs("trueRF_trajectories/", exist_ok=True)
            trajectory_file_path = f"trueRF_trajectories/{run_type}_{len(trajectories)}_trajectories_{rules.NUMBER_OF_RULES}_rules.pkl"
        elif run_type == "trainedRF":
            trajectory_file_path = (
                trajectory_path
                + f"{run_type}_{len(trajectories)}_trajectories_{rules.NUMBER_OF_RULES}_rules.pkl"
            )
        else:
            raise Exception("Invalid run type.")

        if os.path.exists(trajectory_file_path):
            print(
                "Removing old agent database with same number of trajectories and rules..."
            )
            try:
                os.remove(trajectory_file_path)
                print("File removed successfully.")
            except PermissionError:
                print("Permission denied: could not remove the file.")
            except FileNotFoundError:
                print("File not found: it may have been removed by another process.")
            except Exception as e:
                print(f"Unexpected error: {e}")

        with open(trajectory_file_path, "wb") as f:
            if len(trajectories) > 0:
                pickle.dump(trajectories, f)
                print(
                    f"Saved {len(trajectories)} trajectories to {trajectory_file_path}"
                )

        return len(trajectories)


def run_simulation(genomes, config):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []
    global rules_followed
    rules_followed = []
    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
    with open("grid_points.pkl", "rb") as f:
        grid_points = pickle.load(f)
    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        random_trajectory_segment = random.choice(random.choice(grid_points))
        random_position = random_trajectory_segment[0].position
        cars.append(Car(position=random_position, angle=get_angle(random_position[0], random_position[1])) )
    for i, car in enumerate(cars):
        cars[i].id = i
    global run_type
    if run_type == "big_mode":
        big_car = Car(color="red")
    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    global game_map
    game_map = pygame.image.load("maps/map.png").convert()

    global current_generation, headless
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0
    reward_count = 0
    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        if run_type == "big_mode":
            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT]:
                big_car.angle += 2
            if keys[pygame.K_RIGHT]:
                big_car.angle -= 2
            if keys[pygame.K_x]:
                big_car.speed += 0.5
            if keys[pygame.K_z]:
                big_car.speed -= 0.5

        # For Each Car Get The Acton It Takes
        actions = []
        for i, car in enumerate(cars):
            action = -1
            if car.is_alive():
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
                action = choice
            actions.append(action)

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        speeds = []
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1

            reward_count += 1
            car_reward = car.get_reward()
            speeds.append(car.speed)
            rules_followed.append(car.update(game_map, actions[i]))
            genomes[i][1].fitness += car_reward

        global big_car_distance
        big_car_alive = False
        if run_type == "big_mode" and big_car.is_alive():
            big_car_alive = True
        if big_car_alive:
            big_car_distance = big_car.distance
            big_car.update(game_map)

        counter += 1
        if counter == TRAJECTORY_LENGTH:
            global saved_segments
            cars_alive = 0
            old_lengths = [len(seg) for seg in saved_segments]
            for i, car in enumerate(cars):
                cars_alive += car.is_alive()
                car.save_trajectory()
            print(f"{cars_alive} cars survived this generation.")
            if run_type == "collect":
                for i in range(rules.NUMBER_OF_RULES + 1):
                    print(
                        "THIS GENERATION PRODUCED",
                        len(saved_segments[i]) - old_lengths[i],
                        f"SEGMENTS SATISFYING {i} RULES ON AVERAGE.",
                    )
                display_saved_segments(saved_segments)
            break

        if not headless:
            # Draw Map And All Cars That Are Alive
            screen.blit(game_map, (0, 0))
            for car in cars:
                if car.is_alive():
                    car.draw(screen)
            if big_car_alive:
                big_car.draw(screen)

            # Display Info
            text = generation_font.render(
                "Generation: " + str(current_generation), True, (0, 0, 0)
            )
            text_rect = text.get_rect()
            text_rect.center = (900, 450)
            screen.blit(text, text_rect)

            text = alive_font.render(
                "Still Alive: " + str(still_alive), True, (0, 0, 0)
            )
            text_rect = text.get_rect()
            text_rect.center = (900, 490)
            screen.blit(text, text_rect)

            if run_type == "big_mode":
                pygame.draw.circle(screen, (255, 0, 0), big_car.position, 30)
                global big_car_best_distance
                text = alive_font.render(
                    "Time Left before generation ends: "
                    + str(TRAJECTORY_LENGTH - counter),
                    True,
                    (0, 0, 0),
                )
                text_rect = text.get_rect()
                text_rect.center = (900, 530)
                screen.blit(text, text_rect)

                text = alive_font.render(
                    "Left/Right Arrows for Steer | Z/X for Speed",
                    True,
                    (0, 0, 0),
                )
                text_rect = text.get_rect()
                text_rect.center = (900, 570)
                screen.blit(text, text_rect)

                text = alive_font.render(
                    "Current Distance: " + str(big_car_distance), True, (0, 0, 0)
                )
                text_rect = text.get_rect()
                text_rect.center = (900, 610)
                screen.blit(text, text_rect)

                text = alive_font.render(
                    "Best Distance: " + str(big_car_best_distance), True, (0, 0, 0)
                )
                text_rect = text.get_rect()
                text_rect.center = (900, 650)
                screen.blit(text, text_rect)

            pygame.display.flip()
        clock.tick(60)  # 60 FPS


def show_database_segments(database):
    print("DATABASE SEGMENTS:")
    for key in sorted(database.keys(), key=lambda x: len(x)):
        print(f"Rules: {key}, # Segments: {len(database[key])}")

def load_from_garg(database):
    try:
        with open(database, "rb") as file:
            data = pickle.load(file)
            print(f"USING MASTER DB: {database}")
            # if "subsampled" in master_database: # Franklin can fix ig?
            #     saved_segments = data[: rules.NUMBER_OF_RULES + 1]

        show_database_segments(data)

        loaded_segments = [[] for _ in range(rules.NUMBER_OF_RULES + 1)]
        for i in range(rules.NUMBER_OF_RULES + 1):
            for combination in combinations(rules.RULES_INCLUDED, i):
                loaded_segments[i].extend(data[combination])
        return loaded_segments
                
    except Exception:
        print(f"COULD NOT LOAD FROM MASTER DB: {master_database}")
        return [[] for _ in range(rules.NUMBER_OF_RULES + 1)]

def save_as_database(segments):
    database = defaultdict(list)
    for segment_i_rules in segments:
        for segment in segment_i_rules:
            _, _, rules_followed = rules.check_rules_one(segment, rules.NUMBER_OF_RULES)
            database[tuple(rules_followed)].append(segment)

    if os.path.exists(master_database):
        with open(master_database, "rb") as file:
            old_database = pickle.load(file)
            for key in database:
                if key not in old_database:
                    old_database[key] = database[key]
                elif len(database[key]) > len(old_database[key]):
                    old_database[key] = database[key]
        return old_database
    else:
        return database



def finished_collecting(saved_data, number_of_pairs):
    return collection_status(saved_data, number_of_pairs) == [True] * (rules.NUMBER_OF_RULES + 1)


def collection_status(saved_data, number_of_pairs):
    rule_finished = [False] * (rules.NUMBER_OF_RULES + 1)
    for i in range(rules.NUMBER_OF_RULES + 1):
        if (
            len(saved_data[i])
            >= number_of_pairs * 2 * rules.SEGMENT_DISTRIBUTION_BY_RULES[i]
        ):
            rule_finished[i] = True
    return rule_finished


def run_population(
    config_path,
    max_generations,
    number_of_pairs,
    runType,
    noHead=False,
    use_ensemble=False,
):
    global trajectories_path
    trajectory_path = trajectories_path
    os.makedirs(trajectories_path, exist_ok=True)
    try:
        # Load Config
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        global num_pairs
        num_pairs = number_of_pairs * (ENSEMBLE_MULTIPLIER if use_ensemble else 1)

        global run_type, headless
        run_type = runType
        datatype = "segments" if run_type == "collect" else "trajectories"

        headless = noHead
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            if "SDL_VIDEODRIVER" in os.environ:
                del os.environ["SDL_VIDEODRIVER"]

        if run_type == "collect":
            max_generations = number_of_pairs
        else:
            rules.PARTIAL_REWARD = False
        if run_type == "trainedRF":
            pass

        # Create Population And Add Reporters
        global \
            current_generation, \
            population, \
            saved_segments, \
            saved_trajectories, \
            master_database
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        if not master_database:
            master_database = f"database_gargantuar_{train_trajectory_length}_length.pkl"

        reward.INPUT_SIZE = STATE_ACTION_SIZE * (train_trajectory_length + 1)
        missing_segments = "subsampled" not in master_database
        if run_type == "collect":
            if subsample:
                saved_segments = load_from_garg(master_database)
            else:
                saved_segments = [[] for _ in range(rules.NUMBER_OF_RULES + 1)]

            display_saved_segments(saved_segments)
            display_requested_segments(num_pairs)

            if finished_collecting(saved_segments, num_pairs):
                missing_segments = False
                print("SEGMENT COUNT SATISFIED! Subsampling...")
            else:
                print("MISSING SEGMENTS! Generating more...")

        current_generation = 0
        saved_trajectories = []
        generation = 1

        start_time = time.time()
        if run_type != "collect" or missing_segments:
            while True:
                population.run(run_simulation, 1)
                if run_type == "collect" and finished_collecting(saved_segments, num_pairs):
                    print(f"Stopping after {generation} generations.")
                    pygame.display.quit()
                    pygame.quit()
                    break
                elif generation >= max_generations and (
                    len(saved_trajectories) >= config.pop_size * max_generations
                ):
                    while len(saved_trajectories) > config.pop_size * max_generations:
                        saved_trajectories.pop()
                    break
                generation += 1
                global big_car_distance, big_car_best_distance
                big_car_best_distance = max(big_car_distance, big_car_best_distance)
                big_car_distance = 0

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Total execution time for {generation} generations: {elapsed_time:.2f} seconds"
        )
        numTrajPairs = generate_database(trajectory_path, saved_segments, datatype)

        return numTrajPairs
    except KeyboardInterrupt:
        print(
            "\nKeyboardInterrupt detected. Do you want to save the database? (y/n): ",
            end="",
        )
        user_input = input().strip().lower()
        if user_input == "y":
            generate_database(trajectory_path, saved_trajectories, datatype)
        else:
            print("Exiting without saving the database.")
        sys.exit()


def load_models(reward_paths, hidden_size):
    global runType, reward_network, ensemble
    if len(reward_paths) == 1:
        print("\nLoading reward network...")
        reward_network = TrajectoryRewardNet(
            STATE_ACTION_SIZE * (train_trajectory_length + 1),
            hidden_size=hidden_size,
        ).to(device)
        weights = torch.load(reward_paths[0], map_location=torch.device(f"{device}"))
        reward_network.load_state_dict(weights)
    else:
        if reward_paths[0] == "QUICK":
            if len(reward_paths) > 2:
                raise Exception("REWARD PATH ERROR (QUICK MODE)")
            ensemble_dir = reward_paths[1] + "*"
            reward_paths = []
            for file in glob.glob(ensemble_dir):
                reward_paths.append(file)

        print(f"\nLoading ensemble of {len(reward_paths)} models...")
        ensemble_nets = [
            TrajectoryRewardNet(
                STATE_ACTION_SIZE * (train_trajectory_length + 1),
                hidden_size=hidden_size,
            ).to(device)
            for _ in range(len(reward_paths))
        ]
        ensemble_weights = []
        for reward_path in reward_paths:
            ensemble_weights.append(
                torch.load(reward_path, map_location=torch.device(f"{device}"))
            )
        for i in range(len(ensemble_nets)):
            ensemble_nets[i].load_state_dict(ensemble_weights[i])
            print(f"Loaded model #{i} from ensemble...")
        ensemble = Ensemble(
            STATE_ACTION_SIZE * (train_trajectory_length + 1),
            len(ensemble_nets),
            ensemble_nets,
        )

    runType = "trainedRF"


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="AI Car Preference Learning")
    parse.add_argument(
        "-t",
        "--trajectories",
        type=int,
        nargs=1,
        default=[0],
        help="Number of trajectories to save",
    )
    parse.add_argument(
        "-r",
        "--reward",
        type=str,
        action="append",
        help="Directory to reward function weights",
    )
    parse.add_argument(
        "--headless", action="store_true", help="Run simulation without GUI"
    )
    parse.add_argument(
        "-b",
        "--big",
        action="store_true",
        help="flag for big mode",
    )
    parse.add_argument(
        "-db",
        "--database",
        type=str,
        help="Optional path to saved database",
    )
    parse.add_argument(
        "--generate",
        action="store_true",
        help="flag for generating trajectories rather than sampling",
    )
    parse.add_argument(
        "-s",
        "--segment",
        type=int,
        help="Length of segments",
    )
    args = parse.parse_args()

    if args.reward and args.trajectories[0] > 0:
        print("Cannot save trajectories and train reward function at the same time")
        sys.exit(1)

    hidden_size = None
    parameters_path = "best_params.yaml"
    with open(parameters_path, "r") as file:
        data = yaml.safe_load(file)
        hidden_size = data["hidden_size"]

    if args.reward:
        load_models(args.reward, hidden_size)

    if args.big:
        runType = "big_mode"

    if args.generate:
        subsample = False

    if args.database:
        sampled_database = args.database

    config_path = (
        "config/data_collection_config.txt"
        if reward_network is None
        else "config/agent_config.txt"
    )

    if args.trajectories[0] > 0:
        number_of_pairs = args.trajectories[0]
        runType = "collect"

    if args.segment and args.segment < 1:
        raise Exception("Can not have segments with length < 2")
    train_trajectory_length = args.segment if args.segment else 1

    run_population(
        config_path=config_path,
        max_generations=DEFAULT_MAX_GENERATIONS,
        number_of_pairs=number_of_pairs,
        runType=runType,
        noHead=args.headless,
        use_ensemble=args.ensemble,
    )
