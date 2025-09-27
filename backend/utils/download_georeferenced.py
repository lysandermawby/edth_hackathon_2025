#!/usr/bin/env python3
"""
Utility for downloading georeferenced images from KML files.
Supports parsing KML files and extracting image URLs with georeferencing data.
"""

import os
import re
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, urljoin
import logging
import base64
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KMLImageDownloader:
    """Downloads georeferenced images from KML files."""
    
    def __init__(self, output_dir: str = "downloaded_images"):
        """
        Initialize the KML image downloader.
        
        Args:
            output_dir: Directory to save downloaded images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def parse_kml(self, kml_content: str) -> List[Dict]:
        """
        Parse KML content and extract image information.
        
        Args:
            kml_content: KML file content as string
            
        Returns:
            List of dictionaries containing image metadata
        """
        try:
            root = ET.fromstring(kml_content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse KML: {e}")
            return []
        
        images = []
        
        # Define namespaces
        namespaces = {
            'kml': 'http://www.opengis.net/kml/2.2',
            'gx': 'http://www.google.com/kml/ext/2.2'
        }
        
        # Find all GroundOverlay elements (common for georeferenced images)
        overlays = root.findall('.//kml:GroundOverlay', namespaces)
        
        for overlay in overlays:
            image_info = self._extract_overlay_info(overlay, namespaces)
            if image_info:
                images.append(image_info)
        
        # Also look for PhotoOverlay elements
        photos = root.findall('.//kml:PhotoOverlay', namespaces)
        
        for photo in photos:
            image_info = self._extract_photo_info(photo, namespaces)
            if image_info:
                images.append(image_info)
        
        # Look for any href elements that might contain image URLs
        hrefs = root.findall('.//kml:href', namespaces)
        for href in hrefs:
            url = href.text
            if url and self._is_image_url(url):
                images.append({
                    'url': url,
                    'name': f"image_{len(images)}",
                    'type': 'href'
                })
        
        # Check for Google Earth specific data
        google_earth_images = self._extract_google_earth_data(kml_content)
        images.extend(google_earth_images)
        
        return images
    
    def _extract_google_earth_data(self, kml_content: str) -> List[Dict]:
        """Extract Google Earth specific data from KML."""
        images = []
        
        # Look for Google Earth data blob
        earth_data_match = re.search(r'<\?earth data="([^"]+)"\?>', kml_content)
        if earth_data_match:
            earth_data = earth_data_match.group(1)
            logger.info("Found Google Earth data blob")
            
            try:
                # Try to decode the base64 data
                decoded_data = base64.b64decode(earth_data)
                logger.info(f"Decoded Google Earth data: {len(decoded_data)} bytes")
                
                # This is Google Earth's proprietary format - we can't directly parse it
                # But we can save it for manual inspection
                earth_data_path = self.output_dir / "google_earth_data.bin"
                with open(earth_data_path, 'wb') as f:
                    f.write(decoded_data)
                
                logger.warning("Google Earth data found but cannot be automatically parsed.")
                logger.warning("This requires Google Earth Pro or specific tools to extract images.")
                logger.info(f"Raw data saved to: {earth_data_path}")
                
            except Exception as e:
                logger.error(f"Failed to decode Google Earth data: {e}")
        
        return images
    
    def _extract_overlay_info(self, overlay, namespaces: Dict) -> Optional[Dict]:
        """Extract information from GroundOverlay element."""
        try:
            # Get image URL
            icon = overlay.find('kml:Icon', namespaces)
            if icon is not None:
                href = icon.find('kml:href', namespaces)
                if href is not None and href.text:
                    url = href.text
                    
                    # Get name
                    name_elem = overlay.find('kml:name', namespaces)
                    name = name_elem.text if name_elem is not None else f"overlay_{len(overlay)}"
                    
                    # Get LatLonBox for georeferencing
                    latlonbox = overlay.find('kml:LatLonBox', namespaces)
                    georef_data = {}
                    if latlonbox is not None:
                        georef_data = {
                            'north': latlonbox.find('kml:north', namespaces).text if latlonbox.find('kml:north', namespaces) is not None else None,
                            'south': latlonbox.find('kml:south', namespaces).text if latlonbox.find('kml:south', namespaces) is not None else None,
                            'east': latlonbox.find('kml:east', namespaces).text if latlonbox.find('kml:east', namespaces) is not None else None,
                            'west': latlonbox.find('kml:west', namespaces).text if latlonbox.find('kml:west', namespaces) is not None else None,
                            'rotation': latlonbox.find('kml:rotation', namespaces).text if latlonbox.find('kml:rotation', namespaces) is not None else None
                        }
                    
                    return {
                        'url': url,
                        'name': name,
                        'type': 'ground_overlay',
                        'georef_data': georef_data
                    }
        except Exception as e:
            logger.warning(f"Error extracting overlay info: {e}")
        
        return None
    
    def _extract_photo_info(self, photo, namespaces: Dict) -> Optional[Dict]:
        """Extract information from PhotoOverlay element."""
        try:
            # Get image URL
            icon = photo.find('kml:Icon', namespaces)
            if icon is not None:
                href = icon.find('kml:href', namespaces)
                if href is not None and href.text:
                    url = href.text
                    
                    # Get name
                    name_elem = photo.find('kml:name', namespaces)
                    name = name_elem.text if name_elem is not None else f"photo_{len(photo)}"
                    
                    # Get Point for georeferencing
                    point = photo.find('kml:Point', namespaces)
                    georef_data = {}
                    if point is not None:
                        coords = point.find('kml:coordinates', namespaces)
                        if coords is not None and coords.text:
                            # Parse coordinates (longitude,latitude,altitude)
                            coord_parts = coords.text.strip().split(',')
                            if len(coord_parts) >= 2:
                                georef_data = {
                                    'longitude': coord_parts[0],
                                    'latitude': coord_parts[1],
                                    'altitude': coord_parts[2] if len(coord_parts) > 2 else None
                                }
                    
                    return {
                        'url': url,
                        'name': name,
                        'type': 'photo_overlay',
                        'georef_data': georef_data
                    }
        except Exception as e:
            logger.warning(f"Error extracting photo info: {e}")
        
        return None
    
    def _is_image_url(self, url: str) -> bool:
        """Check if URL points to an image file."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        return any(path.endswith(ext) for ext in image_extensions)
    
    def download_image(self, image_info: Dict, base_url: str = None) -> Optional[str]:
        """
        Download a single image.
        
        Args:
            image_info: Dictionary containing image metadata
            base_url: Base URL for relative URLs
            
        Returns:
            Path to downloaded file or None if failed
        """
        url = image_info['url']
        
        # Handle relative URLs
        if not url.startswith(('http://', 'https://')):
            if base_url:
                url = urljoin(base_url, url)
            else:
                logger.warning(f"Relative URL without base URL: {url}")
                return None
        
        try:
            # Generate filename
            filename = self._generate_filename(image_info)
            filepath = self.output_dir / filename
            
            # Download the image
            logger.info(f"Downloading {url} to {filepath}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Save georeferencing data if available
            if image_info.get('georef_data'):
                self._save_georef_data(filepath, image_info['georef_data'])
            
            logger.info(f"Successfully downloaded: {filepath}")
            return str(filepath)
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return None
    
    def _generate_filename(self, image_info: Dict) -> str:
        """Generate a filename for the downloaded image."""
        name = image_info.get('name', 'image')
        url = image_info['url']
        
        # Clean the name
        name = re.sub(r'[^\w\-_\.]', '_', name)
        
        # Get file extension from URL
        parsed_url = urlparse(url)
        path = parsed_url.path
        if '.' in path:
            ext = os.path.splitext(path)[1]
        else:
            ext = '.jpg'  # Default extension
        
        return f"{name}{ext}"
    
    def _save_georef_data(self, image_path: Path, georef_data: Dict):
        """Save georeferencing data as a text file alongside the image."""
        georef_path = Path(str(image_path) + '.georef.txt')
        
        with open(georef_path, 'w') as f:
            f.write("# Georeferencing data\n")
            for key, value in georef_data.items():
                if value is not None:
                    f.write(f"{key}: {value}\n")
    
    def download_from_kml_file(self, kml_file_path: str) -> List[str]:
        """
        Download all images from a KML file.
        
        Args:
            kml_file_path: Path to the KML file
            
        Returns:
            List of paths to downloaded images
        """
        try:
            with open(kml_file_path, 'r', encoding='utf-8') as f:
                kml_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read KML file {kml_file_path}: {e}")
            return []
        
        return self.download_from_kml_content(kml_content, kml_file_path)
    
    def download_from_kml_content(self, kml_content: str, base_path: str = None) -> List[str]:
        """
        Download all images from KML content.
        
        Args:
            kml_content: KML content as string
            base_path: Base path for relative URLs
            
        Returns:
            List of paths to downloaded images
        """
        images = self.parse_kml(kml_content)
        downloaded_paths = []
        
        logger.info(f"Found {len(images)} images in KML")
        
        for i, image_info in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}: {image_info.get('name', 'unnamed')}")
            filepath = self.download_image(image_info, base_path)
            if filepath:
                downloaded_paths.append(filepath)
        
        logger.info(f"Successfully downloaded {len(downloaded_paths)} images")
        return downloaded_paths


def main():
    """Example usage of the KML image downloader."""
    # Example KML content (you can replace this with your actual KML)
    kml_content = '''<?xml version="1.0" encoding="UTF-8"?>
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
    
    # Create downloader
    downloader = KMLImageDownloader("downloaded_images")
    
    # Download images from KML content
    downloaded_files = downloader.download_from_kml_content(kml_content)
    
    print(f"Downloaded {len(downloaded_files)} images:")
    for file_path in downloaded_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()
