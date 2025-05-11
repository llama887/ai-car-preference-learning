import pickle
import os

from agent import show_database_dict_segments

DATABASE = 'databases/database_gargantuar_1_length.pkl'
DATABASE_DIR = DATABASE.replace(".pkl", "/")
print(f"Database directory: {DATABASE_DIR}")
os.makedirs(DATABASE_DIR, exist_ok=True)

with open(DATABASE, 'rb') as f:
    database = pickle.load(f)
print(f"Database ({DATABASE}) loaded successfully.")

show_database_dict_segments(database)

for bucket in list(database.keys()):
    if isinstance(database[bucket], list):
        bucket_path = DATABASE_DIR + f"bucket_{str(list(bucket))}.pkl" 
        print(f"Saving bucket {bucket} to {bucket_path}")
        with open(bucket_path, 'wb') as f:
            pickle.dump(database[bucket], f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Database ({DATABASE}) bucketed.")

del database
