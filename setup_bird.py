"""
Setup script: checks BIRD Mini-Dev data availability.
Run this before the main pipeline.
"""

import os

from config import BIRD_DATA_DIR, BIRD_DB_DIR, BIRD_MYSQL_JSON, BIRD_SQLITE_JSON


EXPECTED_DBS = [
    "california_schools",
    "card_games",
    "codebase_community",
    "debit_card_specializing",
    "european_football_2",
    "financial",
    "formula_1",
    "student_club",
    "superhero",
    "thrombosis_prediction",
    "toxicology",
]


def check_setup():
    has_sqlite_json = os.path.exists(BIRD_SQLITE_JSON)
    has_mysql_json = os.path.exists(BIRD_MYSQL_JSON)

    print("=== BIRD Mini-Dev Setup Check ===\n")
    print(
        f"SQLite JSON (mini_dev_sqlite.json): {'FOUND' if has_sqlite_json else 'MISSING'}"
    )
    print(f"MySQL JSON  (mini_dev_mysql.json):  {'FOUND' if has_mysql_json else 'MISSING'}")

    print(f"\n--- Database files in {BIRD_DB_DIR} ---")
    missing_dbs = []
    found_dbs = []

    for db_name in EXPECTED_DBS:
        sqlite_path = os.path.join(BIRD_DB_DIR, db_name, f"{db_name}.sqlite")
        if os.path.exists(sqlite_path):
            size_mb = os.path.getsize(sqlite_path) / (1024 * 1024)
            print(f"  {db_name}: FOUND ({size_mb:.1f} MB)")
            found_dbs.append(db_name)
        else:
            print(f"  {db_name}: MISSING")
            missing_dbs.append(db_name)

    print("\n--- Summary ---")
    print(f"SQLite databases found: {len(found_dbs)}/11")

    if missing_dbs or not has_sqlite_json:
        print("\n--- ACTION REQUIRED ---")
        if not has_sqlite_json:
            print("You need mini_dev_sqlite.json (SQLite-dialect queries).")
        if missing_dbs:
            print(f"Missing .sqlite files for: {', '.join(missing_dbs)}")

        print("\nDownload the full package from:")
        print("  https://bird-bench.github.io/  (BIRD Mini-Dev Complete Package)")
        print("\nOr via HuggingFace CLI:")
        print("  pip install huggingface_hub")
        print(
            "  huggingface-cli download --repo-type dataset "
            "birdsql/bird_mini_dev --local-dir data/bird/mini_dev_data"
        )
        print("\nThen re-run this script to verify.")
    else:
        print("\nAll data present. Ready to run: python -u main.py")

    if not has_sqlite_json and not has_mysql_json:
        print("\nNeither SQLite nor MySQL JSON query file is present.")


if __name__ == "__main__":
    if not os.path.exists(BIRD_DATA_DIR):
        print("BIRD data directory not found.")
        print(f"Expected location: {BIRD_DATA_DIR}")
        print("Create it and place Mini-Dev files there, then re-run this script.")
    else:
        check_setup()
