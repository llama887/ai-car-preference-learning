import argparse
import glob
import os
import pickle
import random
import rules
from collections import defaultdict
from agent import get_buckets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Folder containing database folders.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file for combined database.",
    )
    args = parser.parse_args()

    # Find all files in the directory with "database" in the name and ".pkl" extension
    folder_pattern = os.path.join(args.directory, "*master_database*")
    database_folders = [d for d in glob.glob(folder_pattern) if os.path.isdir(d)]

    if not database_folders:
        raise ValueError(
            f"No folders found in {args.directory} matching pattern '*master_database*.pkl'"
        )

    buckets = []
    for database_folder in database_folders:
        database_buckets = get_buckets(database_folder)
        for bucket in database_buckets:
            if bucket not in buckets:
                buckets.append(bucket)


    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)
    for bucket in buckets:
        bucket_path = os.path.join(output_folder, f"bucket_{bucket}.pkl")
        if os.path.exists(bucket_path):
            with open(bucket_path, "rb") as f:
                existing_data = pickle.load(f)
        else:
            existing_data = []

        for database_folder in database_folders:
            database_bucket_file = os.path.join(database_folder, f"bucket_{bucket}.pkl")
            if os.path.exists(database_bucket_file):
                with open(database_bucket_file, "rb") as f:
                    data = pickle.load(f)
                    existing_data.extend(data)

        with open(bucket_path, "wb") as f:
            pickle.dump(existing_data, f)


    print(f"Combined database written to {output_folder}")
