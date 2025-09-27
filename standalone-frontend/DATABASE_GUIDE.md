# Database Interaction Guide (SQLite)

This guide provides instructions on how to interact with the SQLite database used in this project. 

**Disclaimer:** This guide is based on the files in the `standalone-frontend` directory. The project appears to have a `backend` directory that could not be accessed, so this guide may be incomplete.

## Database Location

The SQLite database is located at `databases/tracking_data.db`.

## Database Schema

Based on the queries in `server.js`, the database contains at least two tables:

### `tracking_sessions`

| Column | Type |
|---|---|
| `session_id` | INTEGER |
| `video_path` | TEXT |
| `start_time` | DATETIME |
| `end_time` | DATETIME |
| `total_frames` | INTEGER |
| `fps` | REAL |

### `tracked_objects`

| Column | Type |
|---|---|
| `id` | INTEGER |
| `session_id` | INTEGER |
| `frame_number` | INTEGER |
| `track_id` | INTEGER |
| `object_class` | TEXT |
| `x` | REAL |
| `y` | REAL |
| `w` | REAL |
| `h` | REAL |


## Node.js (server.js)

### Connecting to the Database

The connection to the database is managed in `server.js`. The `sqlite3` library is used to create a database object.

```javascript
import sqlite3 from "sqlite3";

const DB_PATH = "databases/tracking_data.db";

// Initialize SQLite database
const db = new sqlite3.Database(DB_PATH, (err) => {
  if (err) {
    console.error("Error connecting to SQLite database:", err.message);
  } else {
    console.log("Connected to SQLite database");
  }
});
```

### Reading from the Database

You can read data from the database using `db.all()` for multiple rows or `db.get()` for a single row.

#### Example: Reading all tracked objects

```javascript
app.get("/api/tracked-objects", (req, res) => {
  db.all("SELECT * FROM tracked_objects", [], (err, rows) => {
    if (err) {
      res.status(500).json({ error: err.message });
      return;
    }
    res.json({ data: rows });
  });
});
```

### Writing to the Database

You can write data to the database using `db.run()`.

#### Example: Inserting a new tracked object

```javascript
app.post("/api/tracked-objects", (req, res) => {
  const { session_id, track_id, object_class, x, y, w, h } = req.body;
  const sql = `INSERT INTO tracked_objects (session_id, track_id, object_class, x, y, w, h)
               VALUES (?, ?, ?, ?, ?, ?, ?)`;
  const params = [session_id, track_id, object_class, x, y, w, h];

  db.run(sql, params, function (err) {
    if (err) {
      res.status(400).json({ error: err.message });
      return;
    }
    res.json({
      message: "Success",
      data: { id: this.lastID },
    });
  });
});
```

## Python

Python can interact with the SQLite database using the built-in `sqlite3` module.

### Connecting to the Database

```python
import sqlite3

DB_PATH = "databases/tracking_data.db"

try:
    conn = sqlite3.connect(DB_PATH)
    print("Connected to SQLite database")
except sqlite3.Error as e:
    print(f"Error connecting to SQLite database: {e}")

# It's recommended to close the connection when you're done
# conn.close()
```

### Reading from the Database

You can fetch data using a cursor. `cursor.fetchall()` fetches all rows, and `cursor.fetchone()` fetches a single row.

#### Example: Reading all tracked objects

```python
import sqlite3

DB_PATH = "databases/tracking_data.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT * FROM tracked_objects")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
```

### Writing to the Database

You can execute `INSERT` statements and commit the changes to the database.

#### Example: Inserting a new tracked object

```python
import sqlite3

DB_PATH = "databases/tracking_data.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

sql = """INSERT INTO tracked_objects (session_id, track_id, object_class, x, y, w, h)
         VALUES (?, ?, ?, ?, ?, ?, ?)"""
params = ('some_session_id', 123, 'car', 100, 200, 50, 30)

cursor.execute(sql, params)
conn.commit()

print(f"Inserted row with id: {cursor.lastrowid}")

conn.close()
```

## Running Raw SQL Queries

You can use the `sqlite3` CLI to run raw SQL queries on the a inspection.

```bash
sqlite3 databases/tracking_data.db "SELECT COUNT(*) FROM tracked_objects;"
```
