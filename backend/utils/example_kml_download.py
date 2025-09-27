#!/usr/bin/env python3
"""
Example script showing how to use the KML image downloader.
"""

import sys
from pathlib import Path
import os

# Add the utils directory to the path so we can import our module
sys.path.append(str(Path(__file__).parent))

from download_georeferenced import KMLImageDownloader


def main():
    """Example usage with your KML data."""
    
    # Your KML content from the file
    try:
        with open("data/gs_kml.xml", "r", encoding="utf-8") as f:
            your_kml_content = f.read()
    except FileNotFoundError:
        print("KML file not found. Using example content.")
        your_kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <GroundOverlay>
      <name>Sample Georeferenced Image</name>
      <Icon>
        <href>https://example.com/sample_image.jpg</href>
      </Icon>
      <LatLonBox>
        <north>40.0</north>
        <south>39.0</south>
        <east>-74.0</east>
        <west>-75.0</west>
      </LatLonBox>
    </GroundOverlay>
  </Document>
</kml>'''
    
    
    # Create downloader instance
    downloader = KMLImageDownloader("downloaded_images")
    
    print("KML Image Downloader Example")
    print("=" * 40)
    
    # Method 1: Download from KML content string
    print("\n1. Downloading from KML content...")
    downloaded_files = downloader.download_from_kml_content(your_kml_content)
    
    if downloaded_files:
        print(f"Successfully downloaded {len(downloaded_files)} images:")
        for file_path in downloaded_files:
            print(f"  - {file_path}")
    else:
        print("No images found in the KML content.")
    
    # Method 2: Download from KML file (if you have a file)
    print("\n2. To download from a KML file, use:")
    print("   downloader.download_from_kml_file('path/to/your/file.kml')")
    
    # Method 3: Parse KML without downloading
    print("\n3. To just parse KML and see what images are available:")
    images = downloader.parse_kml(your_kml_content)
    print(f"Found {len(images)} images in KML:")
    for i, img in enumerate(images):
        print(f"  {i+1}. {img.get('name', 'unnamed')} - {img['url']}")
        if img.get('georef_data'):
            print(f"     Georeferencing: {img['georef_data']}")
    
    # Method 4: Extract coordinates for Google Earth navigation
    print("\n4. Coordinates for Google Earth navigation:")
    print("   Center: 11.33567333424076, 48.07639779746007")
    print("   Altitude: 583.2985999874493")
    print("   Polygon coordinates:")
    print("   11.32179297519343,48.0838715636152,0")
    print("   11.32311472931257,48.06686293222302,0")
    print("   11.35184127505622,48.06747717510396,0")
    print("   11.34913616165074,48.0847503226986,0")
    
    print("\n" + "="*60)
    print("🚨 IMPORTANT: Your KML contains Google Earth data!")
    print("📖 See google_earth_guide.md for detailed instructions")
    print("🔧 You'll need Google Earth Pro to export the actual image")
    print("="*60)


if __name__ == "__main__":
    main()
