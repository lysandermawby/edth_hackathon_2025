# 📹 Multi-Camera Selection Guide

## 🎯 Overview

The real-time detection system now supports **multiple camera selection** with automatic detection and easy switching between available cameras.

## 🚀 How to Use

### 1. **Automatic Detection**

When you connect to the real-time server, cameras are automatically detected:

- System scans cameras 0-9 for availability
- Only working cameras are shown in the dropdown
- Camera count is displayed below the selector

### 2. **Manual Camera Selection**

```
Camera: [Camera 0 ▼] [🔄]
1 camera detected
[📹 Start Camera 0]
```

**Steps:**

1. **Connect** to the real-time server
2. **Select camera** from the dropdown menu
3. **Click** "📹 Start Camera X" to begin detection
4. **Switch cameras** anytime by selecting a different option

### 3. **Refresh Camera List**

- Click the **🔄 refresh button** to re-detect cameras
- Useful when plugging in new USB cameras
- Updates the dropdown with newly detected cameras

## 🔧 Technical Details

### Camera Detection Process

The system tests each camera by:

1. **Opening** the camera device (cv2.VideoCapture)
2. **Reading** a test frame to verify it works
3. **Adding** working cameras to the available list
4. **Closing** the camera cleanly

### Smart Camera Switching

- If your selected camera becomes unavailable, the system automatically switches to the first available camera
- Camera IDs typically correspond to:
  - **Camera 0**: Built-in laptop camera
  - **Camera 1**: First external USB camera
  - **Camera 2**: Second external USB camera
  - etc.

## 🎮 Use Cases

### **Single Camera Setup**

- Shows "Camera 0" in dropdown
- Displays "1 camera detected"
- Works exactly as before

### **Multiple USB Cameras**

```
Camera: [Camera 2 ▼]  [🔄]
3 cameras detected
[📹 Start Camera 2]
```

- Select different angles or camera qualities
- Switch between cameras without restarting
- Perfect for multi-angle surveillance or demos

### **Camera Troubleshooting**

- Use the **🔄 refresh** if cameras aren't detected
- Check camera isn't being used by another application
- Try different USB ports for external cameras

## 📊 Visual Indicators

| Element                | Meaning                                      |
| ---------------------- | -------------------------------------------- |
| `Camera: [Camera 1 ▼]` | Currently selected camera                    |
| `🔄`                   | Refresh/detect cameras button                |
| `2 cameras detected`   | Number of working cameras found              |
| `📹 Start Camera 1`    | Start button shows which camera will be used |

## 🔄 WebSocket Commands

### Frontend → Backend

```javascript
// Request available cameras
{ "command": "list_cameras" }

// Start specific camera
{ "command": "start_camera", "camera_id": 2 }
```

### Backend → Frontend

```javascript
// Camera list response
{
  "type": "camera_list",
  "cameras": [0, 1, 2],
  "message": "Found 3 available cameras"
}
```

## 🎉 Benefits

- **🎯 Flexibility**: Choose the best camera for your use case
- **🔄 Hot-swapping**: Switch cameras without restarting detection
- **📱 Multi-device**: Support for webcams, USB cameras, etc.
- **🛠️ Debugging**: Easy camera testing and troubleshooting
- **⚡ Auto-detection**: No manual configuration needed

Perfect for demos, surveillance setups, or any scenario where you have multiple camera options! 📹✨
