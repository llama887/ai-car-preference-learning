import pickle
import os
import argparse

import rules
from itertools import combinations
from agent import show_database_segments, get_buckets

DATABASE_DIR = './databases/database_gargantuar_testing_1_length/'
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

# show_database_segments(DATABASE_DIR)

buckets = get_buckets(DATABASE_DIR)

for bucket in buckets:
    print("BUCKET:", bucket)
    violations = 0
    data = load_from_bucket(DATABASE_DIR, bucket)
    new_data = []
    for segment in data:
        _, _, rules_followed = rules.check_rules_one(segment, rules.NUMBER_OF_RULES)
        if rules_followed == bucket:
            new_data.append(segment)
        else:
            violations += 1
    
    print(f"Total violations: {violations} out of {len(data)} segments")
    print(f"Total valid segments: {len(new_data)} out of {len(data)} segments")
    with open(os.path.join(DATABASE_DIR, f"bucket_{bucket}.pkl"), "wb") as file:
        pickle.dump(new_data, file, protocol=pickle.HIGHEST_PROTOCOL)


print(f"Database ({DATABASE_DIR}) checked.")
