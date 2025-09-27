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
    showLabels = false,
  } = req.body || {};

  const scriptPath = path.join(
    __dirname,
    "..",
    "backend",
    "tracking",
    "generate_session_detections.py"
  );

  const args = [scriptPath, String(sessionId), "--db-path", DB_PATH];

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
        console.log(`Found items: ${items.join(", ")}`);

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
                  console.log(
                    `Searching in subdirectory: ${searchDir}, items: ${subItems.join(
                      ", "
                    )}`
                  );

                  for (const subItem of subItems) {
                    const subItemPath = path.join(searchDir, subItem);
                    const subStat = fs.statSync(subItemPath);

                    if (subItem.endsWith(".csv")) {
                      console.log(`Found CSV file: ${subItemPath}`);
                      csvFiles.push(subItemPath);
                    } else if (subStat.isDirectory()) {
                      // Recursively search subdirectories
                      searchInDir(subItemPath);
                    }
                  }
                } catch (err) {
                  console.log(
                    `Error reading directory ${searchDir}: ${err.message}`
                  );
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
        res
          .status(404)
          .json({ error: "No GPS metadata found for this session" });
        return;
      }

      // Parse the first CSV file found (or combine multiple if needed)
      const csvFile = metadataFiles[0];
      console.log(`Loading GPS metadata from: ${csvFile}`);

      const csvContent = fs.readFileSync(csvFile, "utf8");
      const lines = csvContent.trim().split("\n");

      if (lines.length < 2) {
        res.status(404).json({ error: "Invalid CSV file format" });
        return;
      }

      const headers = lines[0].split(",");
      const metadata = [];

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(",");
        if (values.length === headers.length) {
          const entry = {};
          headers.forEach((header, index) => {
            const value = values[index];
            // Convert numeric fields
            if (
              [
                "timestamp",
                "vfov",
                "hfov",
                "roll",
                "pitch",
                "yaw",
                "latitude",
                "longitude",
                "altitude",
                "gimbal_elevation",
                "gimbal_azimuth",
              ].includes(header)
            ) {
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
        hfov: entry.hfov,
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

// Get enhanced telemetry data for a specific session
app.get("/api/sessions/:sessionId/telemetry", (req, res) => {
  const sessionId = parseInt(req.params.sessionId);

  if (isNaN(sessionId)) {
    res.status(400).json({ error: "Invalid session ID" });
    return;
  }

  // Get session details to find the video path and calculate duration
  db.get(
    "SELECT video_path, fps, total_frames FROM tracking_sessions WHERE session_id = ?",
    [sessionId],
    (err, session) => {
      if (err) {
        console.error("Database error:", err);
        res.status(500).json({ error: "Failed to fetch session" });
        return;
      }

      if (!session) {
        res.status(404).json({ error: "Session not found" });
        return;
      }

      // Calculate video duration
      const videoDuration =
        session.total_frames && session.fps
          ? session.total_frames / session.fps
          : 25.0; // fallback

      // Extract directory from video path to find CSV file
      const videoPath = session.video_path;
      const videoBasename = path.basename(videoPath, path.extname(videoPath));

      // Look for CSV file with the same name as the video in the same directory
      // Remove 'data/' prefix from video path if present since DATA_DIR already includes it
      const relativePath = videoPath.startsWith("data/")
        ? videoPath.substring(5)
        : videoPath;
      const csvFileName = `${videoBasename}.csv`;
      const csvPath = path.join(
        DATA_DIR,
        path.dirname(relativePath),
        csvFileName
      );

      console.log(`Looking for enhanced telemetry CSV at: ${csvPath}`);

      if (!fs.existsSync(csvPath)) {
        console.log(
          `CSV not found, sending empty telemetry for session ${sessionId}`
        );
        res.json({ telemetry: [], analytics: null });
        return;
      }

      try {
        // Use Python script to parse enhanced telemetry
        const python = spawn("python3", [
          "-c",
          `
import sys
import json
sys.path.append('${path.join(__dirname, "..", "backend", "metadata_process")}')
from enhanced_telemetry_parser import parse_telemetry_csv
from pathlib import Path

try:
    result = parse_telemetry_csv(Path('${csvPath}'), ${videoDuration})
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
        `,
        ]);

        let output = "";
        let error = "";

        python.stdout.on("data", (data) => {
          output += data.toString();
        });

        python.stderr.on("data", (data) => {
          error += data.toString();
        });

        python.on("close", (code) => {
          if (code !== 0) {
            console.error("Enhanced telemetry parsing error:", error);
            res
              .status(500)
              .json({ error: "Failed to parse enhanced telemetry" });
            return;
          }

          try {
            const telemetryData = JSON.parse(output);

            if (telemetryData.error) {
              console.error("Telemetry parser error:", telemetryData.error);
              res.status(500).json({ error: telemetryData.error });
              return;
            }

            console.log(
              `Loaded enhanced telemetry with ${telemetryData.telemetry.length} points for session ${sessionId}`
            );
            res.json(telemetryData);
          } catch (parseError) {
            console.error("Failed to parse telemetry output:", parseError);
            res.status(500).json({ error: "Invalid telemetry output" });
          }
        });
      } catch (error) {
        console.error("Error processing enhanced telemetry:", error);
        res.status(500).json({ error: "Failed to process telemetry file" });
      }
    }
  );
});

// Get available videos from data directory for import
app.get("/api/videos/available", (req, res) => {
  try {
    const videoExtensions = [
      ".mp4",
      ".avi",
      ".mov",
      ".mkv",
      ".wmv",
      ".flv",
      ".webm",
    ];
    const videos = [];

    function scanDirectory(dirPath, relativePath = "") {
      const items = fs.readdirSync(dirPath);

      for (const item of items) {
        const fullPath = path.join(dirPath, item);
        const itemRelativePath = path.join(relativePath, item);
        const stats = fs.statSync(fullPath);

        if (stats.isDirectory()) {
          // Skip hidden directories and node_modules
          if (!item.startsWith(".") && item !== "node_modules") {
            scanDirectory(fullPath, itemRelativePath);
          }
        } else if (stats.isFile()) {
          const ext = path.extname(item).toLowerCase();
          if (videoExtensions.includes(ext)) {
            videos.push({
              filename: item,
              path: itemRelativePath.replace(/\\/g, "/"), // Normalize path separators
              size: stats.size,
              modified: stats.mtime.toISOString(),
              sizeFormatted: formatFileSize(stats.size),
            });
          }
        }
      }
    }

    scanDirectory(DATA_DIR);

    // Sort by filename
    videos.sort((a, b) => a.filename.localeCompare(b.filename));

    res.json(videos);
  } catch (error) {
    console.error("Error scanning for videos:", error);
    res.status(500).json({ error: "Failed to scan video directory" });
  }
});

// Create a new session from an imported video
app.post("/api/sessions/import", (req, res) => {
  const { videoPath, autoProcess = true } = req.body;

  if (!videoPath) {
    res.status(400).json({ error: "Video path is required" });
    return;
  }

  // Verify the video file exists
  const fullVideoPath = path.join(DATA_DIR, videoPath);
  if (!fs.existsSync(fullVideoPath)) {
    res.status(404).json({ error: "Video file not found" });
    return;
  }

  // Create new session in database with current timestamp
  const relativePath = videoPath; // Already relative to DATA_DIR
  const insertQuery = `
    INSERT INTO tracking_sessions (video_path, fps, created_at)
    VALUES (?, ?, datetime('now'))
  `;

  // Get FPS using CV2 via Python if available, otherwise default to 30
  const getFpsScript = `
import cv2
import sys
try:
    cap = cv2.VideoCapture("${fullVideoPath.replace(/"/g, '\\"')}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    print(fps)
except:
    print(30.0)
  `;

  const pythonGetFps = spawn("python3", ["-c", getFpsScript]);
  let fpsOutput = "";

  pythonGetFps.stdout.on("data", (data) => {
    fpsOutput += data.toString();
  });

  pythonGetFps.on("close", (code) => {
    let fps = 30.0;
    try {
      fps = parseFloat(fpsOutput.trim()) || 30.0;
    } catch (e) {
      fps = 30.0;
    }

    db.run(insertQuery, [relativePath, fps], function (insertErr) {
      if (insertErr) {
        console.error("Database error:", insertErr);
        res.status(500).json({ error: "Failed to create session" });
        return;
      }

      const newSessionId = this.lastID;

      if (autoProcess) {
        // Automatically start processing the video
        const scriptPath = path.join(
          __dirname,
          "..",
          "backend",
          "tracking",
          "generate_session_detections.py"
        );
        const args = [
          scriptPath,
          String(newSessionId),
          "--db-path",
          DB_PATH,
          "--video-path",
          fullVideoPath,
        ];

        console.log("Auto-processing imported video:", args.join(" "));

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

        pythonProcess.on("close", (processCode) => {
          if (processCode === 0) {
            // Count detections
            db.get(
              "SELECT COUNT(*) as count FROM tracked_objects WHERE session_id = ?",
              [newSessionId],
              (countErr, row) => {
                const detectionCount = row?.count ?? 0;
                res.json({
                  message: "Video imported and processed successfully",
                  session_id: newSessionId,
                  video_path: relativePath,
                  fps: fps,
                  detections: detectionCount,
                  auto_processed: true,
                });
              }
            );
          } else {
            // Processing failed, but session was created
            res.json({
              message: "Video imported but processing failed",
              session_id: newSessionId,
              video_path: relativePath,
              fps: fps,
              detections: 0,
              auto_processed: false,
              processing_error: stderr.trim() || stdout.trim(),
            });
          }
        });
      } else {
        // Just create session without processing
        res.json({
          message: "Video imported successfully",
          session_id: newSessionId,
          video_path: relativePath,
          fps: fps,
          detections: 0,
          auto_processed: false,
        });
      }
    });
  });
});

// Utility function to format file sizes
function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

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
