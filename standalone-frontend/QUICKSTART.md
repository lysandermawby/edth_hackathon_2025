# 🚀 EDTH Object Tracker - Quick Start Guide

## Video Not Loading? Troubleshooting Steps

### 1. Check Video File Exists

```bash
ls -la /Users/kenny/GitHub/edth_hackathon_2025/data/Individual_2.mp4
```

### 2. Test API Endpoints

```bash
# Test API health
curl http://localhost:3001/api/health

# Test sessions endpoint
curl http://localhost:3001/api/sessions

# Test video endpoint
curl -I http://localhost:3001/api/video/data/Individual_2.mp4
```

### 3. Check Browser Console

Open your browser's Developer Tools (F12) and look for:

- **Console tab**: Check for JavaScript errors
- **Network tab**: Verify video requests are successful (200 status)
- **Console logs**: Look for video loading messages

### 4. Common Issues & Solutions

#### Issue: Black video canvas

**Symptoms**: Canvas appears but no video shows
**Solutions**:

1. Check browser console for CORS errors
2. Ensure video file exists at the correct path
3. Verify API server is running on port 3001

#### Issue: "Failed to load video"

**Symptoms**: Error messages in console
**Solutions**:

1. Check video file permissions: `chmod 644 data/Individual_2.mp4`
2. Restart both servers
3. Clear browser cache

#### Issue: No detection overlays

**Symptoms**: Video plays but no bounding boxes
**Solutions**:

1. Verify tracking data exists in database
2. Check console for detection data loading errors
3. Ensure session has completed tracking

### 5. Expected Behavior

When working correctly, you should see:

1. **Session List**: Left panel shows available tracking sessions
2. **Video Loading**: Console logs "Video load started" and "Video loaded"
3. **Detection Overlays**: Colored bounding boxes on detected objects
4. **Video Controls**: Play/pause and seek functionality works
5. **Statistics**: Frame count and detection count displayed

### 6. Debug Mode

The application includes debug logging. Check browser console for:

- `Video load started: [URL]`
- `Video loaded: [width] x [height]`
- `Generated video URL: [URL] for path: [path]`

### 7. Manual Testing

Test the video endpoint directly:

```bash
# Should return video data
curl http://localhost:3001/api/video/data/Individual_2.mp4 -o test.mp4
```

### 8. Restart Everything

```bash
# Kill all processes
pkill -f "node server.js"
pkill -f "vite"

# Restart from project root
./run-standalone-frontend.sh
```

## 📞 Need More Help?

1. Check that both servers are running:

   - API: http://localhost:3001/api/health
   - Frontend: http://localhost:5173

2. Verify you have tracking data:

   ```bash
   sqlite3 databases/tracking_data.db "SELECT COUNT(*) FROM tracked_objects;"
   ```

3. Ensure video file is accessible:
   ```bash
   file data/Individual_2.mp4
   ```

The application should work perfectly once all components are properly connected! 🎉
