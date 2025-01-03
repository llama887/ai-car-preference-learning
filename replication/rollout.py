import pickle

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Load the trained model
model_path = "./models/final_ppo_pendulum.zip"
model = PPO.load(model_path)

# Create the environment
env = gym.make("Pendulum-v1", render_mode="human")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Initialize variables for video recording
obs = env.reset()
done = False

observation_action_reward_triples = []
# Run the agent in the environment and capture frames
while not done and len(observation_action_reward_triples) < 1000:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    observation_action_reward_triples.append((obs, action, reward))
    env.render(mode="human")

with open("pendulum_rollout.pkl", "wb") as f:
    pickle.dump(observation_action_reward_triples, f)
