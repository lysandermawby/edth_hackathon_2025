#!/usr/bin/env python3
"""
Data visualisation for tracking data stored in a SQLite database.
"""

import os
import argparse
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def main(): 
    parser = argparse.ArgumentParser(description="Data visualisation for tracking data stored in a SQLite database")
    parser.add_argument("--db-path", type=str, default="../../databases/tracking_data.db", 
                       help="Path to SQLite database file")
    args = parser.parse_args()

    # establishing database connection and cursor
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()



    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()