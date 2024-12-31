from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env_id = "Pendulum-v1"
env = make_vec_env(env_id, n_envs=1)

# Instantiate the agent
model = PPO(
    "MlpPolicy",
    env,
    gamma=0.98,
    # Using https://proceedings.mlr.press/v164/raffin22a.html
    use_sde=True,
    sde_sample_freq=4,
    learning_rate=1e-3,
    verbose=1,
)

# Train the agent
model.learn(total_timesteps=int(1e5))
