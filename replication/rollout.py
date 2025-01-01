import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Define the output video file
output_video_file = "ppo_pendulum_performance.mp4"

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

# Run the agent in the environment and capture frames
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
