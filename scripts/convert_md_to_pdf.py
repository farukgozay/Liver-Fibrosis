"""
Simple MD to Printable HTML Converter
======================================

Converts markdown files to styled HTML that can be printed to PDF via browser.
No external dependencies required.
"""

from pathlib import Path
import re

print("="*80)
print("MARKDOWN TO PRINTABLE HTML CONVERTER")
print("="*80)

# Files to convert
md_files = [
    "README.md",
    "PROJE_OZET.md",
    "TEST_RESULTS.md",
    "VISUAL_TEST_RESULTS.md",
    "REAL_PROGRAM_OUTPUTS.md",
    "docs/SRS_DOCUMENT.md"
]

# Output directory
html_dir = Path("docs/html_reports")
html_dir.mkdir(parents=True, exist_ok=True)

print(f"\nOutput directory: {html_dir}")
print(f"Files to convert: {len(md_files)}\n")

# Simple markdown to HTML converter
def md_to_html(md_text):
    """Convert basic markdown to HTML"""
    html = md_text
    
    # Headers
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # Bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    html = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html)
    html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)
    
    # Code blocks
    html = re.sub(r'```(\w+)?\n(.*?)```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Lists
    lines = html.split('\n')
    in_ul = False
    in_ol = False
    result = []
    
    for line in lines:
        # Unordered list
        if re.match(r'^\s*[-*+]\s', line):
            if not in_ul:
                result.append('<ul>')
                in_ul = True
            content = re.sub(r'^\s*[-*+]\s', '', line)
            result.append(f'<li>{content}</li>')
        # Ordered list
        elif re.match(r'^\s*\d+\.\s', line):
            if not in_ol:
                result.append('<ol>')
                in_ol = True
            content = re.sub(r'^\s*\d+\.\s', '', line)
            result.append(f'<li>{content}</li>')
        else:
            if in_ul:
                result.append('</ul>')
                in_ul = False
            if in_ol:
                result.append('</ol>')
                in_ol = False
            result.append(line)
    
    if in_ul:
        result.append('</ul>')
    if in_ol:
        result.append('</ol>')
    
    html = '\n'.join(result)
    
    # Paragraphs
    html = re.sub(r'\n\n', '</p><p>', html)
    html = '<p>' + html + '</p>'
    
    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', html)
    
    # Horizontal rules
    html = re.sub(r'^---+$', '<hr>', html, flags=re.MULTILINE)
    html = re.sub(r'^\*\*\*+$', '<hr>', html, flags=re.MULTILINE)
    
    # Blockquotes
    html = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
    
    # Clean up empty paragraphs
    html = re.sub(r'<p>\s*</p>', '', html)
    html = re.sub(r'<p>(<h[1-6]>)', r'\1', html)
    html = re.sub(r'(</h[1-6]>)</p>', r'\1', html)
    html = re.sub(r'<p>(<ul>|<ol>|<pre>)', r'\1', html)
    html = re.sub(r'(</ul>|</ol>|</pre>)</p>', r'\1', html)
    
    return html

# HTML template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 30px;
            background: white;
        }}
        h1 {{
            color: #1a1a1a;
            font-size: 2.2em;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 12px;
            margin: 40px 0 20px 0;
            page-break-after: avoid;
        }}
        h1:first-child {{
            margin-top: 0;
        }}
        h2 {{
            color: #2c3e50;
            font-size: 1.8em;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 10px;
            margin: 30px 0 15px 0;
            page-break-after: avoid;
        }}
        h3 {{
            color: #34495e;
            font-size: 1.4em;
            margin: 25px 0 12px 0;
            page-break-after: avoid;
        }}
        h4 {{
            color: #555;
            font-size: 1.2em;
            margin: 20px 0 10px 0;
        }}
        p {{
            margin: 12px 0;
            text-align: justify;
        }}
        code {{
            background: #f5f5f5;
            padding: 3px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            color: #c7254e;
        }}
        pre {{
            background: #f8f8f8;
            border:1px solid #e0e0e0;
            border-left: 4px solid #0066cc;
            padding: 15px;
            overflow-x: auto;
            border-radius: 4px;
            margin: 20px 0;
            page-break-inside: avoid;
        }}
        pre code {{
            background: none;
            padding: 0;
            color: #333;
            font-size: 0.85em;
            line-height: 1.5;
        }}
        ul, ol {{
            margin: 15px 0;
            padding-left: 35px;
        }}
        li {{
            margin: 8px 0;
        }}
        strong {{
            color: #000;
            font-weight: 600;
        }}
        em {{
            font-style: italic;
            color: #555;
        }}
        blockquote {{
            border-left: 4px solid #0066cc;
            margin: 20px 0;
            padding: 15px 20px;
            background: #f9f9f9;
            color: #555;
            font-style: italic;
        }}
        hr {{
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 30px 0;
        }}
        a {{
            color: #0066cc;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #0066cc;
        }}
        .footer {{
            margin-top: 60px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #888;
            font-size: 0.9em;
            page-break-before: avoid;
        }}
        @media print {{
            body {{
                max-width: 100%;
                padding: 0;
            }}
            h1, h2, h3 {{
                page-break-after: avoid;
            }}
            pre, blockquote {{
                page-break-inside: avoid;
            }}
            a {{
                color: #000;
                text-decoration: underline;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="border:none; margin:0;">{title}</h1>
        <p style="margin:10px 0 0 0; color:#666;">Ankara Üniversitesi - Bilgisayar Mühendisliği</p>
    </div>
    
    {content}
    
    <div class="footer">
        <p><strong>Karaciğer Fibrozunun BT Görüntülerinden Non-İnvaziv Evrelendirilmesi</strong></p>
        <p>Bülent Tuğrul - 22290673 | 2026</p>
        <p style="font-size:0.85em; margin-top:10px;">Kaynak: {filename}</p>
    </div>
</body>
</html>
"""

print("Converting markdown to HTML...\n")

converted_files = []

for md_file in md_files:
    md_path = Path(md_file)
    
    if not md_path.exists():
        print(f"⚠️  Not found: {md_file}")
        continue
    
    try:
        # Read markdown
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = md_to_html(md_content)
        
        # Create title
        title = md_path.stem.replace('_', ' ').title()
        
        # Wrap in template
        full_html = HTML_TEMPLATE.format(
            title=title,
            content=html_content,
            filename=md_file
        )
        
        # Save HTML
        html_filename = md_path.stem + ".html"
        html_path = html_dir / html_filename
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        file_size = html_path.stat().st_size / 1024
        print(f"✓ {md_file:<50} → {html_filename} ({file_size:.1f} KB)")
        
        converted_files.append(html_path)
        
    except Exception as e:
        print(f"✗ Failed: {md_file} - {e}")

print(f"\n{'='*80}")
print(f"✅ CONVERSION COMPLETE!")
print(f"{'='*80}\n")
print(f"Created {len(converted_files)} HTML files in: {html_dir.absolute()}\n")

print("📄 Generated HTML files:")
for html_file in converted_files:
    print(f"   • {html_file.name}")

print(f"\n{'='*80}")
print("HOW TO CREATE PDF FILES:")
print(f"{'='*80}\n")
print("1. Open each HTML file in your web browser")
print("2. Press Ctrl+P (Print)")
print("3. Select 'Save as PDF' or 'Microsoft Print to PDF'")
print("4. Click Save\n")
print("OR use this PowerShell command:")
print(f'   Get-ChildItem "{html_dir.absolute()}\\*.html" | ForEach-Object {{ Start-Process chrome --headless --print-to-pdf="$($_.DirectoryName)\\pdfs\\$($_.BaseName).pdf" $_.FullName }}') 
print(f"\n{'='*80}")
print("✅ All HTML files are ready for PDF conversion!")
print(f"{'='*80}")
