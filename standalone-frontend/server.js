import express from "express";
import cors from "cors";
import sqlite3 from "sqlite3";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3001;

// Enable CORS for all routes
app.use(cors());
app.use(express.json());

// Path to the database and data directory
const DB_PATH = path.join(__dirname, "..", "databases", "tracking_data.db");
const DATA_DIR = path.join(__dirname, "..", "data");

// Initialize SQLite database
const db = new sqlite3.Database(DB_PATH, (err) => {
  if (err) {
    console.error("Error opening database:", err);
  } else {
    console.log("Connected to SQLite database");
  }
});

// API Routes

// Get all tracking sessions
app.get("/api/sessions", (req, res) => {
  const query = `
    SELECT session_id, video_path, start_time, end_time, total_frames, fps
    FROM tracking_sessions
    ORDER BY start_time DESC
  `;

  db.all(query, [], (err, rows) => {
    if (err) {
      console.error("Database error:", err);
      res.status(500).json({ error: "Failed to fetch sessions" });
      return;
    }
    res.json(rows);
  });
});

// Get detections for a specific session
app.get("/api/sessions/:sessionId/detections", (req, res) => {
  const sessionId = parseInt(req.params.sessionId);

  if (isNaN(sessionId)) {
    res.status(400).json({ error: "Invalid session ID" });
    return;
  }

  const query = `
    SELECT *
    FROM tracked_objects
    WHERE session_id = ?
    ORDER BY frame_number, id
  `;

  db.all(query, [sessionId], (err, rows) => {
    if (err) {
      console.error("Database error:", err);
      res.status(500).json({ error: "Failed to fetch detections" });
      return;
    }
    res.json(rows);
  });
});

// Serve video files
app.get("/api/video/:videoPath(*)", (req, res) => {
  const videoPath = req.params.videoPath;

  // Security check - ensure the path is safe
  if (videoPath.includes("..") || path.isAbsolute(videoPath)) {
    res.status(400).json({ error: "Invalid video path" });
    return;
  }

  // Remove 'data/' prefix if it exists, since DATA_DIR already points to the data directory
  const cleanPath = videoPath.startsWith("data/")
    ? videoPath.substring(5)
    : videoPath;
  const fullPath = path.join(DATA_DIR, cleanPath);

  console.log(`Serving video: ${videoPath} -> ${cleanPath} -> ${fullPath}`);

  // Set video-specific headers
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Range");
  res.header("Access-Control-Expose-Headers", "Content-Length, Content-Range");

  // Check if file exists and serve it
  res.sendFile(fullPath, (err) => {
    if (err) {
      console.error("Error serving video:", err);
      res.status(404).json({ error: "Video file not found" });
    }
  });
});

// Health check endpoint
app.get("/api/health", (req, res) => {
  res.json({ status: "OK", timestamp: new Date().toISOString() });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error("Server error:", err);
  res.status(500).json({ error: "Internal server error" });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`Database path: ${DB_PATH}`);
  console.log(`Data directory: ${DATA_DIR}`);
});
