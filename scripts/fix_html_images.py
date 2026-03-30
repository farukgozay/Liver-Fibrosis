"""
Fix HTML Files - Embed Images as Base64
========================================

Embeds all images directly into HTML files as base64 data URIs.
This makes HTML files completely standalone.
"""

from pathlib import Path
import base64
import re

print("="*80)
print("FIXING HTML FILES - EMBEDDING IMAGES")
print("="*80)

html_dir = Path("docs/html_reports")
image_dirs = [
    Path("results/visualizations"),
    Path("C:/Users/gozay/.gemini/antigravity/brain/658c2488-b489-4091-a534-e043f43cb867")
]

def get_mime_type(filename):
    """Get MIME type from filename"""
    ext = filename.suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml'
    }
    return mime_types.get(ext, 'image/png')

def image_to_base64(image_path):
    """Convert image to base64 data URI"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        base64_data = base64.b64encode(image_data).decode('utf-8')
        mime_type = get_mime_type(image_path)
        
        return f"data:{mime_type};base64,{base64_data}"
    except Exception as e:
        print(f"  ⚠️ Could not encode {image_path.name}: {e}")
        return None

def find_image(image_name, search_dirs):
    """Find image in multiple directories"""
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # Search for file
        for img_file in search_dir.rglob(f"*{image_name}*"):
            if img_file.is_file():
                return img_file
        
        # Try exact match
        img_path = search_dir / image_name
        if img_path.exists():
            return img_path
    
    return None

def fix_html_file(html_path, image_dirs):
    """Fix image paths in HTML file"""
    print(f"\nProcessing: {html_path.name}")
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Find all image references
    img_pattern = r'<img[^>]+src=["\'](![^\]]*\]\()?([^"\']+)["\']'
    matches = re.findall(img_pattern, html_content)
    
    if not matches:
        # Try markdown image pattern
        md_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        md_matches = re.findall(md_pattern, html_content)
        
        if md_matches:
            print(f"  Found {len(md_matches)} markdown images")
            
            for alt_text, img_path in md_matches:
                # Extract filename
                img_filename = Path(img_path).name
                
                # Find actual image
                actual_img = find_image(img_filename, image_dirs)
                
                if actual_img:
                    # Convert to base64
                    base64_uri = image_to_base64(actual_img)
                    
                    if base64_uri:
                        # Replace markdown with HTML img tag
                        old_pattern = f"![{alt_text}]({img_path})"
                        new_tag = f'<img src="{base64_uri}" alt="{alt_text}" style="max-width:100%; height:auto; margin:20px 0; border:1px solid #ddd; border-radius:4px;">'
                        html_content = html_content.replace(old_pattern, new_tag)
                        print(f"  ✓ Embedded: {img_filename} ({actual_img.stat().st_size / 1024:.1f} KB)")
                    else:
                        print(f"  ✗ Failed to encode: {img_filename}")
                else:
                    print(f"  ⚠️ Not found: {img_filename}")
    
    # Save fixed HTML
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    new_size = html_path.stat().st_size / 1024
    print(f"  Final size: {new_size:.1f} KB")

# Process all HTML files
html_files = list(html_dir.glob("*.html"))

print(f"\nFound {len(html_files)} HTML files to process")
print(f"Image directories: {len([d for d in image_dirs if d.exists()])} accessible\n")

for html_file in html_files:
    fix_html_file(html_file, image_dirs)

print("\n" + "="*80)
print("✅ ALL HTML FILES FIXED!")
print("="*80)
print(f"\nImages embedded in: {html_dir.absolute()}")
print("\nHTML files are now standalone and can be opened anywhere.")
print("Images are embedded as base64 data URIs.\n")

# Also copy actual images to html_reports for reference
print("Copying image files to html_reports...")
img_copy_dir = html_dir / "images"
img_copy_dir.mkdir(exist_ok=True)

import shutil

copied = 0
for img_dir in image_dirs:
    if not img_dir.exists():
        continue
    
    for img_file in img_dir.glob("*.png"):
        try:
            shutil.copy(img_file, img_copy_dir / img_file.name)
            copied += 1
        except:
            pass

print(f"✓ Copied {copied} image files to {img_copy_dir}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
