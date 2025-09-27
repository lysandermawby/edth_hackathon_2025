import express from "express";
import cors from "cors";
import sqlite3 from "sqlite3";
import path from "path";
import fs from "fs";
import { spawn } from "child_process";
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

// Regenerate detections for a session using the Python tracking pipeline
app.post("/api/sessions/:sessionId/generate-detections", (req, res) => {
  const sessionId = parseInt(req.params.sessionId);

  if (isNaN(sessionId)) {
    res.status(400).json({ error: "Invalid session ID" });
    return;
  }

  const {
    videoPath,
    keepExisting = false,
    ignoreClasses = [],
    showLabels = false
  } = req.body || {};

  const scriptPath = path.join(
    __dirname,
    "..",
    "backend",
    "tracking",
    "generate_session_detections.py"
  );

  const args = [
    scriptPath,
    String(sessionId),
    "--db-path",
    DB_PATH
  ];

  if (videoPath && typeof videoPath === "string") {
    args.push("--video-path", videoPath);
  }

  if (Array.isArray(ignoreClasses) && ignoreClasses.length > 0) {
    args.push("--ignore-classes", ...ignoreClasses.map(String));
  }

  if (keepExisting) {
    args.push("--keep-existing");
  }

  if (showLabels) {
    args.push("--show-labels");
  }

  console.log("Generating detections via Python:", args.join(" "));

  const pythonProcess = spawn("python3", args, {
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stdout = "";
  let stderr = "";

  pythonProcess.stdout.on("data", (data) => {
    stdout += data.toString();
  });

  pythonProcess.stderr.on("data", (data) => {
    stderr += data.toString();
  });

  pythonProcess.on("close", (code) => {
    if (code !== 0) {
      console.error("Detection generation failed:", stderr);
      res.status(500).json({
        error: "Failed to generate detections",
        details: stderr.trim() || stdout.trim(),
      });
      return;
    }

    db.get(
      "SELECT COUNT(*) as count FROM tracked_objects WHERE session_id = ?",
      [sessionId],
      (countErr, row) => {
        if (countErr) {
          console.error("Database error after generation:", countErr);
          res.status(500).json({
            error: "Detections generated but counting results failed",
          });
          return;
        }

        res.json({
          message: "Detections generated successfully",
          detections: row?.count ?? 0,
          output: stdout.trim(),
        });
      }
    );
  });
});

// Get GPS metadata for a specific session
app.get("/api/sessions/:sessionId/metadata", (req, res) => {
  const sessionId = parseInt(req.params.sessionId);

  if (isNaN(sessionId)) {
    res.status(400).json({ error: "Invalid session ID" });
    return;
  }

  // First get the session to get the video path
  const sessionQuery = `
    SELECT video_path, fps
    FROM tracking_sessions
    WHERE session_id = ?
  `;

  db.get(sessionQuery, [sessionId], (err, session) => {
    if (err) {
      console.error("Database error:", err);
      res.status(500).json({ error: "Failed to fetch session" });
      return;
    }

    if (!session) {
      res.status(404).json({ error: "Session not found" });
      return;
    }

    // Extract session name from video path to find corresponding CSV files
    const videoPath = session.video_path;
    const sessionName = path.basename(videoPath, path.extname(videoPath));

    // Look for CSV files in the data directory that match this session
    const dataDir = path.join(__dirname, "..", "data");

    console.log(`Looking for GPS metadata for session: ${sessionName}`);
    console.log(`Video path: ${videoPath}`);

    // Try to find CSV files in subdirectories
    try {
      const findMetadataFiles = (dir) => {
        const csvFiles = [];
        const items = fs.readdirSync(dir);

        console.log(`Searching in directory: ${dir}`);
        console.log(`Found items: ${items.join(', ')}`);

        for (const item of items) {
          const itemPath = path.join(dir, item);
          const stat = fs.statSync(itemPath);

          if (stat.isDirectory()) {
            console.log(`Checking directory: ${item}`);

            // Check if this directory contains the video file
            if (videoPath.includes(item)) {
              console.log(`Found matching directory: ${item}`);

              // Look for CSV files in this directory and its subdirectories
              const searchInDir = (searchDir) => {
                try {
                  const subItems = fs.readdirSync(searchDir);
                  console.log(`Searching in subdirectory: ${searchDir}, items: ${subItems.join(', ')}`);

                  for (const subItem of subItems) {
                    const subItemPath = path.join(searchDir, subItem);
                    const subStat = fs.statSync(subItemPath);

                    if (subItem.endsWith('.csv')) {
                      console.log(`Found CSV file: ${subItemPath}`);
                      csvFiles.push(subItemPath);
                    } else if (subStat.isDirectory()) {
                      // Recursively search subdirectories
                      searchInDir(subItemPath);
                    }
                  }
                } catch (err) {
                  console.log(`Error reading directory ${searchDir}: ${err.message}`);
                }
              };

              searchInDir(itemPath);
            }
          }
        }
        return csvFiles;
      };

      const metadataFiles = findMetadataFiles(dataDir);

      if (metadataFiles.length === 0) {
        res.status(404).json({ error: "No GPS metadata found for this session" });
        return;
      }

      // Parse the first CSV file found (or combine multiple if needed)
      const csvFile = metadataFiles[0];
      console.log(`Loading GPS metadata from: ${csvFile}`);

      const csvContent = fs.readFileSync(csvFile, 'utf8');
      const lines = csvContent.trim().split('\n');

      if (lines.length < 2) {
        res.status(404).json({ error: "Invalid CSV file format" });
        return;
      }

      const headers = lines[0].split(',');
      const metadata = [];

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        if (values.length === headers.length) {
          const entry = {};
          headers.forEach((header, index) => {
            const value = values[index];
            // Convert numeric fields
            if (['timestamp', 'vfov', 'hfov', 'roll', 'pitch', 'yaw', 'latitude', 'longitude', 'altitude', 'gimbal_elevation', 'gimbal_azimuth'].includes(header)) {
              entry[header] = parseFloat(value);
            } else {
              entry[header] = value;
            }
          });
          metadata.push(entry);
        }
      }

      // Convert to the format expected by the frontend
      const convertedMetadata = metadata.map((entry, index) => ({
        timestamp: index, // Use frame index as timestamp for now
        latitude: entry.latitude,
        longitude: entry.longitude,
        altitude: entry.altitude,
        roll: entry.roll,
        pitch: entry.pitch,
        yaw: entry.yaw,
        gimbal_elevation: entry.gimbal_elevation,
        gimbal_azimuth: entry.gimbal_azimuth,
        vfov: entry.vfov,
        hfov: entry.hfov
      }));

      console.log(`Loaded ${convertedMetadata.length} GPS metadata entries`);
      res.json(convertedMetadata);

    } catch (error) {
      console.error("Error reading GPS metadata:", error);
      res.status(500).json({ error: "Failed to read GPS metadata files" });
    }
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
