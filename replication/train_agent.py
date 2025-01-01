import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


# Load the hyperparameters from the YAML file
def load_hyperparameters(file_path, env_id):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    if env_id in config:
        return config[env_id]
    else:
        raise ValueError(
            f"Environment ID '{env_id}' not found in the hyperparameters file."
        )


# Specify the hyperparameter file and environment ID
hyperparameter_file = "replication/pendulum_params.yml"
env_id = "Pendulum-v1"
hyperparams = load_hyperparameters(hyperparameter_file, env_id)

# Extract relevant hyperparameters
n_envs = hyperparams.get("n_envs", 1)
n_timesteps = int(hyperparams.get("n_timesteps", 1e5))
policy = hyperparams.get("policy", "MlpPolicy")
n_steps = hyperparams.get("n_steps", 2048)
gae_lambda = hyperparams.get("gae_lambda", 0.95)
gamma = hyperparams.get("gamma", 0.99)
n_epochs = hyperparams.get("n_epochs", 10)
ent_coef = hyperparams.get("ent_coef", 0.0)
learning_rate = hyperparams.get("learning_rate", 3e-4)
clip_range = hyperparams.get("clip_range", 0.2)
use_sde = hyperparams.get("use_sde", False)
sde_sample_freq = hyperparams.get("sde_sample_freq", -1)

# Create the environment with logging
env = make_vec_env(env_id, n_envs=n_envs, monitor_dir="./logs")

# Set up a callback for saving the model at intervals
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # Save every 10,000 steps
    save_path="./models/",
    name_prefix="ppo_pendulum",
)

# Instantiate the agent with loaded hyperparameters
model = PPO(
    policy,
    env,
    n_steps=n_steps,
    gae_lambda=gae_lambda,
    gamma=gamma,
    n_epochs=n_epochs,
    ent_coef=ent_coef,
    learning_rate=learning_rate,
    clip_range=clip_range,
    use_sde=use_sde,
    sde_sample_freq=sde_sample_freq,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",  # Enable TensorBoard logging
)

# Train the agent with the callback
model.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)

# Save the final model
model.save("./models/final_ppo_pendulum")
