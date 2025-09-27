#!/usr/bin/env python3
"""Utilities for inspecting tracking data stored in the SQLite database."""

import argparse
import sqlite3
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


def find_tables(cursor):
    """Return all table names available in the database."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]


def find_column_dict(cursor, tables):
    """Return a mapping of table name to its column names."""
    column_dict = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        column_dict[table] = [row[1] for row in columns]
    return column_dict


def find_available_columns(cursor):
    """Print and return the available tables and their columns."""
    tables = find_tables(cursor)
    print("--------------------------------")
    print(f"Tables available in the database: {tables}")
    print("--------------------------------")

    columns = find_column_dict(cursor, tables)

    for table, column_list in columns.items():
        print(f"Columns in the {table} table: {column_list}")
        print("--------------------------------")

    return columns


def display_tracking_sessions(cursor):
    """Pretty-print tracking sessions using a Markdown table."""
    cursor.execute("SELECT * FROM tracking_sessions;")
    tracking_sessions = cursor.fetchall()

    columns_dict = find_column_dict(cursor, ['tracking_sessions'])
    columns = columns_dict['tracking_sessions']

    if not tracking_sessions:
        print("No tracking sessions found.")
        return tracking_sessions

    table_data = [list(session) for session in tracking_sessions]
    print("\n## Tracking Sessions")
    print(tabulate(table_data, headers=columns, tablefmt="pipe"))
    print(f"\n**Total sessions: {len(tracking_sessions)}**")
    
    return tracking_sessions


def display_tracked_objects(cursor, limit=None, session_id=None):
    """Display the tracked objects in the database"""
    if session_id is not None:
        if limit:
            cursor.execute("SELECT * FROM tracked_objects WHERE session_id = ? ORDER BY id DESC LIMIT ?;", (session_id, limit))
            query_desc = f" (session {session_id}, showing latest {limit})"
        else:
            cursor.execute("SELECT * FROM tracked_objects WHERE session_id = ?;", (session_id,))
            query_desc = f" (session {session_id})"
    else:
        if limit:
            cursor.execute("SELECT * FROM tracked_objects ORDER BY id DESC LIMIT ?;", (limit,))
            query_desc = f" (showing latest {limit})"
        else:
            cursor.execute("SELECT * FROM tracked_objects;")
            query_desc = ""
    
    tracked_objects = cursor.fetchall()
    
    # Get column names from the existing function
    columns_dict = find_column_dict(cursor, ['tracked_objects'])
    columns = columns_dict['tracked_objects']
    
    if not tracked_objects:
        print("No tracked objects found.")
        return tracked_objects
        
    # Convert data to list of lists with column headers
    table_data = [list(obj) for obj in tracked_objects]
    print(f"\n## Tracked Objects{query_desc}")
    print(tabulate(table_data, headers=columns, tablefmt="pipe"))
    print(f"\n**Total objects shown: {len(tracked_objects)}**")
    
    # Get total count if we're showing a limited set
    if limit:
        if session_id is not None:
            cursor.execute("SELECT COUNT(*) FROM tracked_objects WHERE session_id = ?;", (session_id,))
            total_count = cursor.fetchone()[0]
            print(f"**Total objects in session {session_id}: {total_count}**")
        else:
            cursor.execute("SELECT COUNT(*) FROM tracked_objects;")
            total_count = cursor.fetchone()[0]
            print(f"**Total objects in database: {total_count}**")
    
    return tracked_objects


def _fetch_class_distribution(cursor, session_id: Optional[int]):
    """Return (class_name, detections) rows for the requested scope."""
    if session_id is None:
        cursor.execute(
            """
            SELECT class_name, COUNT(*) AS detections
            FROM tracked_objects
            GROUP BY class_name
            ORDER BY detections DESC
            """
        )
    else:
        cursor.execute(
            """
            SELECT class_name, COUNT(*) AS detections
            FROM tracked_objects
            WHERE session_id = ?
            GROUP BY class_name
            ORDER BY detections DESC
            """,
            (session_id,),
        )

    return cursor.fetchall()


def plot_class_distribution(rows: Iterable[tuple[str, int]], session_id: Optional[int]):
    """Render a simple bar chart of detection counts per class."""
    data = list(rows)
    if not data:
        print("No detections available for plotting.")
        return

    classes, counts = zip(*data)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(classes), palette="viridis")
    title = "Class distribution"
    if session_id is not None:
        title += f" (session {session_id})"
    plt.title(title)
    plt.xlabel("Detections")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Data visualisation for tracking data stored in a SQLite database")
    parser.add_argument(
        "--db-path",
        type=str,
        default="../../databases/tracking_data.db",
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--session-id",
        type=int,
        default=None,
        help="Limit analysis to a specific session identifier",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Render seaborn plots for class distributions",
        default=False
    )
    parser.add_argument("--show-sessions", action="store_true", default=True,
                       help="Show tracking sessions table")
    parser.add_argument("--show-objects", action="store_true", default=True,
                       help="Show tracked objects table")
    parser.add_argument("--object-limit", type=int, default=20, 
                       help="Limit number of objects to show (default: 20, use 0 for all)")
    parser.add_argument("--show-schema", action="store_true", default=True,
                       help="Show database schema")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    if args.show_schema:
        find_available_columns(cursor)
        
    if args.show_sessions:
        display_tracking_sessions(cursor)

    if args.show_plots:
        plot_rows = _fetch_class_distribution(cursor, args.session_id)
        plot_class_distribution(plot_rows, args.session_id)

    if args.show_objects:
        object_limit = None if args.object_limit == 0 else args.object_limit
        display_tracked_objects(cursor, object_limit, args.session_id)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
