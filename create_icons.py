#!/usr/bin/env python3
"""
Create simple placeholder icons for the Tauri app
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename, bg_color=(102, 126, 234), text_color=(255, 255, 255)):
    """Create a simple icon with text"""
    image = Image.new('RGBA', (size, size), bg_color + (255,))
    draw = ImageDraw.Draw(image)

    # Draw text "ET" (EDTH Tracker)
    text = "ET"

    # Try to use a font, fallback to default if not available
    try:
        font_size = size // 3
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the text
    x = (size - text_width) // 2
    y = (size - text_height) // 2

    draw.text((x, y), text, fill=text_color + (255,), font=font)

    # Save the image
    image.save(filename, 'PNG')
    print(f"Created {filename} ({size}x{size})")

def main():
    """Create all required icons"""
    icons_dir = "src-tauri/icons"
    os.makedirs(icons_dir, exist_ok=True)

    # Create various sizes needed by Tauri
    icon_sizes = [
        (32, "32x32.png"),
        (128, "128x128.png"),
        (128, "128x128@2x.png"),
        (256, "icon.png"),
        (512, "icon@2x.png"),
    ]

    for size, filename in icon_sizes:
        create_icon(size, os.path.join(icons_dir, filename))

    # Create ICO file (Windows)
    try:
        ico_sizes = [32, 48, 64, 128, 256]
        images = []
        for size in ico_sizes:
            img = Image.new('RGBA', (size, size), (102, 126, 234, 255))
            draw = ImageDraw.Draw(img)

            try:
                font_size = size // 3
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            text = "ET"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            x = (size - text_width) // 2
            y = (size - text_height) // 2

            draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
            images.append(img)

        # Save as ICO
        images[0].save(os.path.join(icons_dir, "icon.ico"), format='ICO', sizes=[(img.width, img.height) for img in images])
        print("Created icon.ico")
    except Exception as e:
        print(f"Could not create ICO file: {e}")

    # Create ICNS file (macOS) - simplified version
    try:
        # Create a 512x512 version for ICNS
        create_icon(512, os.path.join(icons_dir, "icon.icns.png"))
        # Note: Real ICNS creation requires more complex tools
        print("Created icon.icns.png (rename to icon.icns for macOS)")
    except Exception as e:
        print(f"Could not create ICNS file: {e}")

if __name__ == "__main__":
    try:
        import PIL
        main()
    except ImportError:
        print("PIL (Pillow) not installed. Creating minimal placeholder icons...")

        # Create minimal placeholder files
        icons_dir = "src-tauri/icons"
        os.makedirs(icons_dir, exist_ok=True)

        # Create empty PNG files as placeholders
        for filename in ["32x32.png", "128x128.png", "128x128@2x.png", "icon.ico", "icon.icns"]:
            with open(os.path.join(icons_dir, filename), 'wb') as f:
                # Write minimal PNG header for PNG files
                if filename.endswith('.png'):
                    # Minimal 1x1 transparent PNG
                    f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00\x007n\xf9$\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00~\x1b\xdf\x00\x00\x00\x00IEND\xaeB`\x82')
                else:
                    # Empty file for ICO/ICNS
                    f.write(b'')
            print(f"Created minimal placeholder: {filename}")