import pickle
import os

import rules
from agent import show_database_segments, get_buckets
from collections import defaultdict
import copy
import orientation.get_orientation

TRAIN_DATABASE_DIR = './databases/database_gargantuar_1_length'
TEST_DATABASE_DIR = './databases/database_gargantuar_testing_1_length/'
TRAIN_SIZE = 2000000
TEST_SIZE = 500000
print(f"Train Database directory: {TRAIN_DATABASE_DIR}")

os.makedirs(TEST_DATABASE_DIR, exist_ok=True)
print("Made test dir")

def get_min_test_size(database, buckets):
    min_size = float('inf')
    for b in buckets:
        bucket_file = os.path.join(database, "bucket_" + str(b) + ".pkl")
        try:
            with open(bucket_file, "rb") as file:
                data = pickle.load(file)
                min_size = min(min_size, len(data) - TRAIN_SIZE)
        except FileNotFoundError:
            print(f"File '{bucket_file}' not found in '{database}'.")
    return min_size

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

show_database_segments(TRAIN_DATABASE_DIR)
buckets = get_buckets(TRAIN_DATABASE_DIR)

# min_test_size = get_min_test_size(TRAIN_DATABASE_DIR, buckets)

# if min_test_size <= 0:
#     print("Not enough data to create a test set.")

# else:
#     print(min_test_size)

for bucket in buckets:
    print("BUCKET:", bucket)
    data = load_from_bucket(TRAIN_DATABASE_DIR, bucket)
    train_data, test_data = data[:TRAIN_SIZE], data[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
    with open(os.path.join(TRAIN_DATABASE_DIR, f"bucket_{bucket}.pkl"), "wb") as file:
        pickle.dump(train_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(TEST_DATABASE_DIR, f"bucket_{bucket}.pkl"), "wb") as file:
        pickle.dump(test_data, file, protocol=pickle.HIGHEST_PROTOCOL)

print("NEW TRAIN:")        
show_database_segments(TRAIN_DATABASE_DIR)
print()

print("NEW TEST:")
show_database_segments(TEST_DATABASE_DIR)   
