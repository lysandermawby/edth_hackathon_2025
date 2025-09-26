# 🎯 Interactive Detection Tooltips

## ✨ What's New

The video canvas now features **interactive hover tooltips** that appear when you hover your mouse over any detected object!

## 🖱️ How to Use

1. **Start the application** and select a tracking session
2. **Hover your mouse** over any bounding box on the video
3. **See instant details** about the detected object

## 📋 Tooltip Information

When you hover over a detection, you'll see:

### 🎯 **Object Class** 
- The type of object (car, bus, airplane, etc.)
- Displayed with relevant emoji

### 🆔 **Tracker ID**
- Unique identifier for tracking the object across frames
- Helps follow the same object through the video

### 📊 **Confidence Score**
- How confident the AI is about this detection (0-100%)
- Higher percentages = more confident detection

### 📍 **Position**
- Top-left corner coordinates (x, y) of the bounding box
- Useful for precise positioning

### 📏 **Size**
- Width × Height of the bounding box in pixels
- Shows the detected object's size

### 🎯 **Center Point**
- Exact center coordinates of the object
- Useful for tracking movement

## 🎨 Visual Features

- **Smart Positioning**: Tooltip automatically moves to avoid screen edges
- **Color-Coded Info**: Different colors for different types of information
- **Arrow Pointer**: Visual arrow points from tooltip to the detection
- **Crosshair Cursor**: Mouse cursor changes to crosshair when over canvas
- **Smooth Interaction**: Tooltip appears/disappears instantly

## 🔧 Technical Features

- **Precise Hit Detection**: Accurately detects when mouse is over a bounding box
- **Top Object Priority**: Shows info for the topmost detection when overlapping
- **Performance Optimized**: Efficient collision detection for smooth interaction
- **Responsive Design**: Works on different screen sizes

## 🎮 Example Usage

```
Hover over a car detection:
┌─────────────────────────┐
│ 🎯 car                  │
│ ID: #17                 │
│ Confidence: 95%         │
│ Position: (245, 180)    │
│ Size: 120 × 85          │
│ Center: (305, 223)      │
└─────────────────────────┘
            ↓
    [Car Detection Box]
```

## 💡 Pro Tips

- **Follow Objects**: Use the Tracker ID to follow the same object across frames
- **Quality Check**: Low confidence scores might indicate uncertain detections
- **Size Analysis**: Compare object sizes to understand scale and distance
- **Position Tracking**: Watch how center coordinates change to see movement patterns

This feature makes it easy to inspect and understand individual detections without cluttering the interface! 🚀
