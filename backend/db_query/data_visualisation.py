#!/usr/bin/env python3
"""
Data visualisation for tracking data stored in a SQLite database.

The tables are created in the database_integration.py file. They are currently named tracking_sessions and tracked_objects.
"""

import os
import argparse
import sqlite3
from datetime import datetime
from tabulate import tabulate

def find_tables(cursor):
    """Find the tables available in the database"""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

def find_column_dict(cursor, tables):
    """Find the columns available in the tables"""
    column_dict = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        column_dict[table] = [row[1] for row in columns]
    return column_dict

def find_available_columns(cursor):
    """find the tables available in the database"""
    tables = find_tables(cursor)
    print("--------------------------------")
    print(f"Tables available in the database: {tables}")
    print("--------------------------------")

    # find the columns available in the tracking_sessions table
    columns = find_column_dict(cursor, tables)

    for table, column_list in columns.items():
        print(f"Columns in the {table} table: {column_list}")
        print("--------------------------------")

    return columns

def display_tracking_sessions(cursor):
    """display the tracking sessions in the database"""
    cursor.execute("SELECT * FROM tracking_sessions;")
    tracking_sessions = cursor.fetchall()
    
    # Get column names from the existing function
    columns_dict = find_column_dict(cursor, ['tracking_sessions'])
    columns = columns_dict['tracking_sessions']
    
    if not tracking_sessions:
        print("No tracking sessions found.")
        return tracking_sessions
        
    # Convert data to list of lists with column headers
    table_data = [list(session) for session in tracking_sessions]
    print("\n## Tracking Sessions")
    print(tabulate(table_data, headers=columns, tablefmt="pipe"))
    
    return tracking_sessions

def main(): 
    parser = argparse.ArgumentParser(description="Data visualisation for tracking data stored in a SQLite database")
    parser.add_argument("--db-path", type=str, default="../../databases/tracking_data.db", 
                       help="Path to SQLite database file")
    args = parser.parse_args()

    # establishing database connection and cursor
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    columns = find_available_columns(cursor)

    tracking_sessions = display_tracking_sessions(cursor)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()