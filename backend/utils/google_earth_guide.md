# Google Earth KML Image Download Guide

## 🚨 **Important: Google Earth Data Limitations**

Your KML file contains **Google Earth-specific data** that cannot be directly downloaded using standard methods. Here's what you need to know:

## 📋 **What's in Your KML File**

Your `gs_kml.xml` contains:
- **Standard KML elements**: Polygon coordinates, styles, placemarks
- **Google Earth data blob**: Base64-encoded proprietary data (line 84)
- **Authentication tokens**: Google Earth session data

## 🔧 **How to Get the Actual Images**

### **Option 1: Export from Google Earth Pro (Recommended)**

1. **Open your KML in Google Earth Pro**
2. **Right-click on your placemark** → "Save Place As..."
3. **Choose "Save as Image"** or "Export to Image"
4. **Set the resolution** (higher = better quality)
5. **Save the georeferenced image**

### **Option 2: Use Google Earth Engine (Advanced)**

If you have access to Google Earth Engine:
```python
# This requires Google Earth Engine API
import ee
ee.Initialize()

# Your coordinates from the KML
coordinates = [
    [11.32179297519343, 48.0838715636152],
    [11.32311472931257, 48.06686293222302],
    [11.35184127505622, 48.06747717510396],
    [11.34913616165074, 48.0847503226986]
]

# Create a polygon
polygon = ee.Geometry.Polygon(coordinates)

# Get satellite imagery
image = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2023-01-01', '2023-12-31').first()
```

### **Option 3: Manual Screenshot with Coordinates**

1. **Open Google Earth Web/Pro**
2. **Navigate to your coordinates**:
   - Center: 11.33567333424076, 48.07639779746007
   - Altitude: 583.2985999874493
3. **Take a screenshot** at the right zoom level
4. **Note the coordinates** for georeferencing

## 🛠 **What You Need to Save from Google Earth**

### **Essential Data to Capture:**

1. **Viewport Settings**:
   - Longitude: 11.33567333424076
   - Latitude: 48.07639779746007
   - Altitude: 583.2985999874493
   - Heading: 0
   - Tilt: 0
   - Field of View: 35°

2. **Polygon Coordinates**:
   ```
   11.32179297519343,48.0838715636152,0
   11.32311472931257,48.06686293222302,0
   11.35184127505622,48.06747717510396,0
   11.34913616165074,48.0847503226986,0
   ```

3. **Image Resolution**: At least 1920x1080 for good quality

## 🔄 **Alternative: Use Our Enhanced Parser**

Our updated parser will:
- ✅ Extract the Google Earth data blob
- ✅ Save it as `google_earth_data.bin`
- ✅ Provide coordinates for manual image capture
- ⚠️ **Cannot directly download the image** (Google's proprietary format)

## 📝 **Step-by-Step Process**

1. **Run the parser** to extract coordinates:
   ```bash
   python backend/utils/example_kml_download.py
   ```

2. **Use the coordinates** to navigate in Google Earth

3. **Export the image** using Google Earth Pro's export function

4. **Save with georeferencing data** using the coordinates provided

## 🎯 **Best Practices**

- **Use Google Earth Pro** (not web version) for best export options
- **Set high resolution** when exporting (at least 4K)
- **Save as GeoTIFF** if possible for better georeferencing
- **Note the exact viewport settings** for reproducibility

## ⚠️ **Limitations**

- Google Earth's data blob is **proprietary** and encrypted
- **Cannot be parsed** by standard KML tools
- Requires **Google Earth Pro** for proper image export
- **Authentication tokens** expire and are user-specific
