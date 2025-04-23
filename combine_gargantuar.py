import argparse
import glob
import os
import pickle
import random
import rules
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Folder containing database files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file for combined database.",
    )
    parser.add_argument(
        "-r",
        "--rules",
        type=int,
        help="Number of rules",
    )
    parser.add_argument(
        "-p",
        "--pair",
        action="store_true",
        help="Database is paired",
    )
    args = parser.parse_args()

    if args.pair and not args.rules:
        raise Exception("Need to provide number of rules if pairing!")

    # Find all files in the directory with "database" in the name and ".pkl" extension
    file_pattern = os.path.join(args.directory, "*master_database*.pkl")
    database_files = glob.glob(file_pattern)

    if not database_files:
        raise ValueError(
            f"No files found in {args.directory} matching pattern '*master_database*.pkl'"
        )

    databases = []
    for database_file in database_files:
        with open(database_file, "rb") as f:
            data = pickle.load(f)
            databases.append(data)

    print(f"Number of database files processed: {len(databases)}")

    if args.pair:
        rules.NUMBER_OF_RULES = args.rules
        combined_database = []
        for pairs in databases:
            combined_database.extend(pairs)
    else:
         # Combine the databases
        combined_database = defaultdict(list)
        for database in databases:
            for key in database:
                combined_database[key].extend(database[key])

    output_file = args.output
    if args.output == "database_test":
        output_file += f"_{args.rules}_rules.pkl"
    # Save the combined database
    with open(output_file, "wb") as f:
        pickle.dump(combined_database, f)

    print(f"Combined database written to {output_file}")
