import argparse
import glob
import os
import pickle

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
    args = parser.parse_args()

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
            databases.append(pickle.load(f))

    print(f"Number of database files processed: {len(databases)}")

    # Combine the databases
    combined_database = [[] for _ in range(len(databases[0]))]
    for rules_satisfied in range(len(databases[0])):
        for database in databases:
            combined_database[rules_satisfied].extend(database[rules_satisfied])

    # Save the combined database
    with open(args.output, "wb") as f:
        pickle.dump(combined_database, f)

    print(f"Combined database written to {args.output}")
