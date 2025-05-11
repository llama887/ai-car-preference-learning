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

def test1():
    rules.RULES_INCLUDED = [1, 2, 3]
    rules.NUMBER_OF_RULES = len(rules.RULES_INCLUDED)
    for bucket in buckets:
        print("BUCKET:", bucket)
        violations = 0
        data = load_from_bucket(DATABASE_DIR, bucket)
        for segment in data:
            _, _, rules_followed = rules.check_rules_one(segment, rules.NUMBER_OF_RULES)
            violations += rules_followed != bucket
        
        print(f"Total violations: {violations} out of {len(data)} segments")


def test2():
    for bucket in buckets:
        print("BUCKET:", bucket)
        rules.RULES_INCLUDED = bucket
        rules.NUMBER_OF_RULES = len(rules.RULES_INCLUDED)
        violations = 0
        data = load_from_bucket(DATABASE_DIR, bucket)
        for segment in data:
            _, _, rules_followed = rules.check_rules_one(segment, rules.NUMBER_OF_RULES)
            violations += rules_followed != bucket
        
        print(f"Total violations: {violations} out of {len(data)} segments")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Preferences"
    )

    parse.add_argument(
        "-t",
        "--test",
        type=int,
        nargs=1,
        help="Test number (1 or 2)",
    )

    args = parse.parse_args()
    if args.test and args.test[0] == 1:
        print("Running test 1...")
        test1()
    elif args.test and args.test[0] == 2:
        print("Running test 2...")
        test2()
    print(f"Database ({DATABASE_DIR}) checked.")
