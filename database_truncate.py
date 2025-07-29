import pickle
from agent import show_database_dict_segments

import pickle
import os
import argparse

import rules
from itertools import combinations
from agent import show_database_segments, get_buckets

DATABASE_DIR = './databases/database_gargantuar_1_length/'
NEW_SIZE = 2000000
print(f"Database directory: {DATABASE_DIR}")


def load_from_bucket(database, bucket):
    bucket_file = os.path.join(database, "bucket_" + str(bucket) + ".pkl")
    try:
        with open(bucket_file, "rb") as file:
            data = pickle.load(file)
            print(f"BUCKET {bucket} LOADED. [{len(data)} segments]")
            return data
    except FileNotFoundError:
        print(f"File '{bucket_file}' not found in '{database}'.")
        return []

show_database_segments(DATABASE_DIR)
buckets = get_buckets(DATABASE_DIR)

for bucket in buckets:
    print("BUCKET:", bucket)
    violations = 0
    data = load_from_bucket(DATABASE_DIR, bucket)
    new_data = data[:NEW_SIZE]
    with open(os.path.join(DATABASE_DIR, f"bucket_{bucket}.pkl"), "wb") as file:
        pickle.dump(new_data, file, protocol=pickle.HIGHEST_PROTOCOL)

show_database_segments(DATABASE_DIR)

print(f"Database ({DATABASE_DIR}) truncated.")
