// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::Command;
use tauri::command;

// Command to run Python tracking script
#[command]
async fn run_python_tracker(tracker_type: String) -> Result<String, String> {
    let script_path = match tracker_type.as_str() {
        "yolo" => "backend/tracking/realtime_tracker.py",
        "transformer" => "backend/tracking/realtime_tracker_transformer.py",
        "run_menu" => "backend/tracking/run_realtime_tracking.py",
        _ => return Err("Invalid tracker type".to_string()),
    };

    // Try python3 first, then python
    let python_commands = ["python3", "python"];

    for python_cmd in &python_commands {
        let output = Command::new(python_cmd)
            .current_dir("../..")  // Set working directory to project root (go up two levels from src-tauri)
            .arg(script_path)
            .output();

        match output {
            Ok(output) => {
                if output.status.success() {
                    return Ok(String::from_utf8_lossy(&output.stdout).to_string());
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    // Check for common errors and provide helpful messages
                    if stderr.contains("NumPy") || stderr.contains("_ARRAY_API") {
                        return Err(format!(
                            "NumPy compatibility error detected.\n\nTo fix:\n1. Activate virtual environment: source .venv/bin/activate\n2. Install compatible NumPy: pip install 'numpy<2.0'\n3. Reinstall packages: pip install --force-reinstall supervision==0.21.0\n\nOriginal error:\n{}",
                            stderr
                        ));
                    } else if stderr.contains("No module named") {
                        return Err(format!(
                            "Missing Python dependencies.\n\nTo fix:\n1. Activate virtual environment: source .venv/bin/activate\n2. Install dependencies:\n   poetry install\n   pip install trackers\n   pip install supervision==0.21.0\n\nOriginal error:\n{}",
                            stderr
                        ));
                    } else {
                        return Err(stderr.to_string());
                    }
                }
            }
            Err(_) => continue, // Try next python command
        }
    }

    Err(format!(
        "Failed to execute python script with any python command.\nMake sure Python is installed and available in PATH.\nScript: {}",
        script_path
    ))
}

// Command to run video processing
#[command]
async fn process_video(video_path: String) -> Result<String, String> {
    let script_path = "backend/tracking/trackers_test.py";

    // Try python3 first, then python
    let python_commands = ["python3", "python"];

    for python_cmd in &python_commands {
        let output = Command::new(python_cmd)
            .current_dir("../..")  // Set working directory to project root (go up two levels from src-tauri)
            .arg(script_path)
            .arg(&video_path)
            .output();

        match output {
            Ok(output) => {
                if output.status.success() {
                    return Ok(String::from_utf8_lossy(&output.stdout).to_string());
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    // Check for common errors and provide helpful messages
                    if stderr.contains("NumPy") || stderr.contains("_ARRAY_API") {
                        return Err(format!(
                            "NumPy compatibility error detected.\n\nTo fix:\n1. Activate virtual environment: source .venv/bin/activate\n2. Install compatible NumPy: pip install 'numpy<2.0'\n3. Reinstall packages: pip install --force-reinstall supervision==0.21.0\n\nOriginal error:\n{}",
                            stderr
                        ));
                    } else if stderr.contains("No module named") {
                        return Err(format!(
                            "Missing Python dependencies.\n\nTo fix:\n1. Activate virtual environment: source .venv/bin/activate\n2. Install dependencies:\n   poetry install\n   pip install trackers\n   pip install supervision==0.21.0\n\nOriginal error:\n{}",
                            stderr
                        ));
                    } else {
                        return Err(stderr.to_string());
                    }
                }
            }
            Err(_) => continue, // Try next python command
        }
    }

    Err(format!(
        "Failed to execute python script with any python command.\nMake sure Python is installed and available in PATH.\nScript: {}",
        script_path
    ))
}

// Command to get system info
#[command]
async fn get_system_info() -> Result<String, String> {
    let output = Command::new("python3")
        .current_dir("../..")  // Set working directory to project root (go up two levels from src-tauri)
        .arg("-c")
        .arg("import sys; import cv2; print(f'Python: {sys.version}'); print(f'OpenCV: {cv2.__version__}')")
        .output()
        .map_err(|e| format!("Failed to get system info: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err("Could not retrieve system information".to_string())
    }
}

// Command to list available video files
#[command]
async fn list_video_files() -> Result<Vec<String>, String> {
    use std::fs;

    let data_dir = "../../data";  // Relative to frontend/src-tauri (go up two levels to project root)
    if !std::path::Path::new(data_dir).exists() {
        return Ok(vec![]);
    }

    let entries = fs::read_dir(data_dir)
        .map_err(|e| format!("Failed to read data directory: {}", e))?;

    let mut video_files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
        let path = entry.path();

        if let Some(extension) = path.extension() {
            let ext = extension.to_string_lossy().to_lowercase();
            if matches!(ext.as_str(), "mp4" | "avi" | "mov" | "mkv" | "webm") {
                if let Some(filename) = path.file_name() {
                    video_files.push(filename.to_string_lossy().to_string());
                }
            }
        }
    }

    Ok(video_files)
}

// Command to open file dialog for video selection
#[command]
async fn open_video_file_dialog(app_handle: tauri::AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let file_path = app_handle
        .dialog()
        .file()
        .add_filter("Video Files", &["mp4", "avi", "mov", "mkv", "webm", "m4v", "flv"])
        .add_filter("MP4 Files", &["mp4"])
        .add_filter("All Files", &["*"])
        .set_title("Select Video File")
        .blocking_pick_file();

    match file_path {
        Some(path) => Ok(Some(path.to_string())),
        None => Ok(None), // User cancelled
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            run_python_tracker,
            process_video,
            get_system_info,
            list_video_files,
            open_video_file_dialog
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}