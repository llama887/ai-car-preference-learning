import pickle
import random

with open("trajectories/database_50.pkl", "rb") as f:
    database = pickle.load(f)

duplicate_data = []
for i in range(1):
    data = [database[i] for _ in range(50)]
    duplicate_data.extend(data)

random.shuffle(duplicate_data)

with open("trajectories/test.pkl", "wb") as f:
    pickle.dump(duplicate_data, f)
