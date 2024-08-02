# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)
# Code Adapted for preference learning for Emerge Lab research by Franklin Yiu & Alex Tang

import argparse
import glob
import math
import os
import pickle
import random
import re
import sys

import neat
import pygame
import torch
import yaml

import reward
from reward import TrajectoryRewardNet, prepare_single_trajectory, scaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trajectories_path = "trajectories/"
os.makedirs(trajectories_path, exist_ok=True)
# Constants

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit

TRAJECTORY_LENGTH = 30 * 15
TRAIN_TRAJECTORY_LENGTH = 2
STATE_ACTION_SIZE = 7

DEFAULT_MAX_GENERATIONS = 1000
SEGMENTS_PER_PAIR = 5

current_generation = 0  # Generation counter
saved_trajectory_count = 0  # Counter for saved trajectories
trajectory_path = "./trajectories/"
reward_network = None
number_of_pairs = 0
population = ""
run_type = "collect"
headless = False
saved_segments = []
saved_trajectories = []
big_car_best_distance = 0
big_car_distance = 0

# need to save heading, there are 3dof


class state_action_pair:
    def __init__(self, radars, position):
        if len(radars) != 5:
            raise ValueError("radars must be 5 floats")
        if len(position) != 2:
            raise ValueError("position must be 2 floats")

        self.radars = radars
        self.position = position

    def __iter__(self):
        return iter(self.radars + self.position)

    def __getitem__(self, index):
        if 0 <= index < 5:
            return self.radars[index]
        elif 5 <= index < 7:
            return self.position[index - 5]
        else:
            raise IndexError("Index out of range")

    def __len__(self):
        return 7


class Car:
    def __init__(self, color="blue"):
        # Load Car Sprite and Rotate
        if color == "red":
            self.sprite = pygame.image.load("red_car.png").convert()
        else:
            self.sprite = pygame.image.load(
                "car.png"
            ).convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.color = color
        self.position = [830, 920]  # Starting Position
        self.angle = 0
        self.speed = 0

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

    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if self.alive:
            if not self.speed_set:
                self.speed = 20
                self.speed_set = True

                first_state_action = state_action_pair(
                    [self.check_radar(d, game_map) for d in range(-90, 120, 45)],
                    [830, 920],
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
        next_state_action = state_action_pair(radar_dists, self.position)
        self.trajectory.append(next_state_action)

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
            if len(self.trajectory) < 2:
                return 0
            trajectory_tensor = prepare_single_trajectory(self.trajectory)
            reward = reward_network(trajectory_tensor)
            return reward.item()
        # return self.distance / (CAR_SIZE_X / 2)
        return self.speed

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

    def save_trajectory(self, filename):
        global run_type, saved_segments, saved_trajectories
        if run_type != "collect":
            # with open(filename, "wb") as f:
            #     pickle.dump((self.distance, self.trajectory, self.reward), f)
            saved_trajectories.append((self.distance, self.trajectory, self.reward))
        saved_segments.extend(break_into_segments(self.trajectory))


def break_into_segments(trajectory):
    trajectory_segments = []
    prev = 0
    curr = 1
    while curr < len(trajectory):
        segment = [trajectory[prev], trajectory[curr]]
        # if dist(segment) == 0:
        #     break
        trajectory_segments.append(segment)
        prev += 1
        curr += 1

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


def generate_database(trajectory_path):
    trajectory_pairs = []
    global run_type, saved_segments, number_of_pairs, saved_trajectories

    def shuffle(trajectory_pairs):
        random.shuffle(trajectory_pairs)
        for i in range(len(trajectory_pairs)):
            swap = random.randint(0, 1)
            if swap:
                trajectory_pairs[i] = (
                    trajectory_pairs[i][1],
                    trajectory_pairs[i][0],
                    (trajectory_pairs[i][2] + 1) % 2,
                    trajectory_pairs[i][4],
                    trajectory_pairs[i][3],
                )

    # Break trajectories into trajectory segments
    trajectory_segments = saved_segments

    if len(trajectory_segments) % 2 != 0:
        trajectory_segments.pop()

    if run_type == "collect":
        segment_generation_mode = "random"
        if segment_generation_mode == "random":
            random.shuffle(trajectory_segments)
            close_distance = []
            for i in range(0, len(trajectory_segments), 2):
                distance_1 = dist(trajectory_segments[i])
                distance_2 = dist(trajectory_segments[i + 1])
                if abs(distance_1 - distance_2) < 0.001:
                    close_distance.append(
                        (
                            list(trajectory_segments[i]),
                            list(trajectory_segments[i + 1]),
                            0 if distance_1 < distance_2 else 1,
                            distance_1,
                            distance_2,
                        )
                    )
                else:
                    trajectory_pairs.append(
                        (
                            list(trajectory_segments[i]),
                            list(trajectory_segments[i + 1]),
                            0 if distance_1 < distance_2 else 1,
                            distance_1,
                            distance_2,
                        )
                    )
            random.shuffle(close_distance)
            n = len(trajectory_pairs)
            for i in range(n - number_of_pairs):
                trajectory_pairs.pop()
            fill = min(number_of_pairs - n, len(close_distance))
            for i in range(fill):
                trajectory_pairs.append(close_distance[i])

        elif segment_generation_mode == "sequential_pairing":
            trajectory_segments = sort_and_pair(trajectory_segments, False)
            n = len(trajectory_segments)
            for i in range(n // 2):
                j = n - i - 1
                distance_1 = dist(trajectory_segments[i])
                distance_2 = dist(trajectory_segments[j])
                if abs(distance_1 - distance_2) < 1:
                    continue
                trajectory_pairs.append(
                    (
                        trajectory_segments[i],
                        trajectory_segments[j],
                        0 if distance_1 > distance_2 else 1,
                        distance_1,
                        distance_2,
                    )
                )
        elif segment_generation_mode == "all_combinations":
            trajectory_segments = sort_and_pair(trajectory_segments, True)
            n = len(trajectory_segments)
            for i in range(len(trajectory_segments)):
                for j in range(i + 1, len(trajectory_segments)):
                    distance_1 = dist(trajectory_segments[i])
                    distance_2 = dist(trajectory_segments[j])
                    if abs(distance_1 - distance_2) < 1:
                        continue
                    trajectory_pairs.append(
                        (
                            trajectory_segments[i],
                            trajectory_segments[j],
                            0 if distance_1 > distance_2 else 1,
                            distance_1,
                            distance_2,
                        )
                    )
        elif segment_generation_mode == "artificial":
            n = 100
            trajectory_segments = []
            start_points = [
                [random.randint(-50, 50), random.randint(-50, 50)] for _ in range(100)
            ]
            for start in start_points:
                for i in range(100):
                    trajectory_segments.append(
                        [
                            start,
                            calculate_new_point(start, i, random.randint(0, 365)),
                        ]
                    )
            random.shuffle(trajectory_segments)
            for i in range(0, len(trajectory_segments), 2):
                distance_1 = dist(trajectory_segments[i])
                distance_2 = dist(trajectory_segments[i + 1])
                if abs(distance_1 - distance_2) < 0.01:
                    continue
                trajectory_pairs.append(
                    (
                        trajectory_segments[i],
                        trajectory_segments[i + 1],
                        0 if distance_1 > distance_2 else 1,
                        distance_1,
                        distance_2,
                    )
                )

        shuffle(trajectory_pairs)
    else:
        trajectories = saved_trajectories

        # Pads shorter tajectoires so there is a consistent input size
        def pad_trajectory(trajectory, max_length):
            return trajectory + [trajectory[-1]] * (max_length - len(trajectory))

        max_length, _ = max(
            (len(trajectory), index)
            for index, (distance, trajectory, reward) in enumerate(trajectories)
        )
        num_traj = (
            len(trajectory_pairs) * 2 if run_type == "collect" else len(trajectories)
        )
        for i in range(0, num_traj, 2):
            trajectory_pairs.append(
                (
                    pad_trajectory(trajectories[i][1], max_length),
                    pad_trajectory(trajectories[i + 1][1], max_length),
                    0 if trajectories[i][0] > trajectories[i + 1][0] else 1,
                    trajectories[i][0],
                    trajectories[i + 1][0],
                    trajectories[i][2],
                    trajectories[i + 1][2],
                )
            )
    # print(trajectory_pairs)
    print(f"Generating Database with {len(trajectory_pairs)} trajectory pairs...")

    # Delete all trajectories
    print("Removing saved trajectories...")
    old_trajectories = glob.glob(trajectory_path + "trajectory*")
    for f in old_trajectories:
        os.remove(f)

    prefix = "database" if run_type == "collect" else run_type
    # Delete old database if it is redundant (same size)
    print("Removing old database...")
    old_trajectories = glob.glob(
        trajectory_path + f"{prefix}_{len(trajectory_pairs)}.pkl"
    )
    for f in old_trajectories:
        os.remove(f)

    # Save To Database
    with open(trajectory_path + f"{prefix}_{len(trajectory_pairs)}.pkl", "wb") as f:
        pickle.dump(trajectory_pairs, f)

    # print("Done saving to database...")
    return len(trajectory_pairs)


def run_simulation(genomes, config):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []
    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())
    global run_type
    if run_type == "big_mode":
        big_car = Car(color="red")
    # print("THIS GENERATION HAS", len(cars), "CARS.")
    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load("maps/map.png").convert()  # Convert Speeds Up A Lot

    global current_generation, saved_trajectory_count, headless
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
        for i, car in enumerate(cars):
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

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        speeds = []
        rewards = []
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1

            reward_count += 1
            car_reward = car.get_reward()
            rewards.append(car_reward)
            speeds.append(car.speed)
            car.update(game_map)
            genomes[i][1].fitness += car_reward

        # print("SPEED:", speeds)
        # print("REWARDS:", rewards)
        # print("STILL ALIVE:", still_alive)
        # print("NON-ZERO:", sum([1 if r == 0 else 0 for r in rewards]))

        global big_car_distance
        big_car_alive = False
        if run_type == "big_mode" and big_car.is_alive():
            big_car_alive = True
        if big_car_alive:
            big_car_distance = big_car.distance
            big_car.update(game_map)

        global saved_segments
        counter += 1

        # if counter == TRAJECTORY_LENGTH or (still_alive == 0 and not big_car_alive and run_type == "collect"):
        if counter == TRAJECTORY_LENGTH:
            non_expert_traj = False
            num_expert_trajectory = 0
            # if still_alive == 0:
            #     maxCar = max(enumerate(cars), key=lambda x: len(x[1].trajectory))
            #     maxCar[1].save_trajectory(f"{trajectory_path}trajectory_{current_generation}_{maxCar[0]}.pkl")
            #     num_expert_trajectory = 1
            # else:

            for i, car in enumerate(cars):
                if saved_trajectory_count >= number_of_pairs and run_type == "collect":
                    break
                if not car.is_alive() and run_type == "collect" and non_expert_traj:
                    continue
                car.save_trajectory(
                    f"{trajectory_path}trajectory_{current_generation}_{i}.pkl"
                )
                saved_trajectory_count += 1
                num_expert_trajectory += 1
                non_expert_traj = True
            if run_type == "collect":
                print(
                    "THIS GENERATION PRODUCED",
                    num_expert_trajectory,
                    "EXPERT TRAJECTORIES.",
                )
                print("TOTAL SEGMENTS COLLECTED SO FAR:", len(saved_segments))
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
    print("GET REWARD CALLED:", reward_count, "TIMES THIS GENERATION.")


def run_population(
    config_path, max_generations, number_of_pairs, runType, noHead=False
):
    try:
        # Load Config
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        global run_type, saved_trajectory_count, headless
        run_type = runType
        headless = noHead
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            if "SDL_VIDEODRIVER" in os.environ:
                del os.environ["SDL_VIDEODRIVER"]

        if run_type == "collect":
            max_generations = number_of_pairs
        if run_type == "trainedRF":
            pass

        # Create Population And Add Reporters
        global current_generation, population, saved_segments, saved_trajectories
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        current_generation = 0
        saved_trajectories = []
        generation = 1
        while True:
            population.run(run_simulation, 1)
            if (
                run_type == "collect"
                and len(saved_segments) >= number_of_pairs * SEGMENTS_PER_PAIR
            ):
                print(f"Stopping after {generation} generations.")
                pygame.display.quit()
                pygame.quit()
                break
            elif generation == max_generations:
                break
            generation += 1
            global big_car_distance, big_car_best_distance
            big_car_best_distance = max(big_car_distance, big_car_best_distance)
            big_car_distance = 0

        global saved_trajectory_count
        print(f"Saved {saved_trajectory_count} trajectories to {trajectory_path}.")

        numTrajPairs = generate_database(trajectory_path)
        saved_trajectory_count = 0

        return numTrajPairs
    except KeyboardInterrupt:
        generate_database(trajectory_path)


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
    args = parse.parse_args()

    if args.reward and args.trajectories[0] > 0:
        print("Cannot save trajectories and train reward function at the same time")
        sys.exit(1)

    hidden_size = None
    parameters_path = "best_params.yaml"
    with open(parameters_path, "r") as file:
        data = yaml.safe_load(file)
        hidden_size = data["hidden_size"]

    if args.reward is not None:
        print("\nLoading reward network...")

        reward_network = TrajectoryRewardNet(
            STATE_ACTION_SIZE * 2,
            hidden_size=hidden_size,
        ).to(device)
        weights = torch.load(args.reward, map_location=torch.device(f"{device}"))
        reward_network.load_state_dict(weights)
        runType = "trainedRF"
        with open("scaler.pkl", "rb") as f:
            reward.scaler = pickle.load(f)
        if args.big:
            runType = "big_mode"

    config_path = (
        "config/data_collection_config.txt"
        if reward_network is None
        else "config/agent_config.txt"
    )
    # number_of_pairs = [-1]
    if args.trajectories[0] > 0:
        number_of_pairs = args.trajectories[0]
        runType = "collect"

    run_population(
        config_path=config_path,
        max_generations=DEFAULT_MAX_GENERATIONS,
        number_of_pairs=number_of_pairs,
        runType=runType,
        noHead=args.headless,
    )
