import re
import os

# Paths
md_path = r'd:\deepfake project\deepfake video detection project\PROJECT_REPORT.md'
html_path = r'd:\deepfake project\deepfake video detection project\FINAL_TECHNICAL_REPORT.html'

with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

# ROBUST MARKDOWN CONVERTER v3 (Layout Focused)
def convert_to_html(md):
    # 1. Preformatted Blocks (Save them first to avoid paragraph wrapping)
    code_blocks = []
    def save_code(match):
        code_blocks.append(match.group(1))
        return f"<!--CODE_BLOCK_{len(code_blocks)-1}-->"
    md = re.sub(r'```(?:.*?)\n(.*?)```', save_code, md, flags=re.DOTALL)
    
    # 2. Headers
    md = re.sub(r'^# (.*)', r'<h1>\1</h1>', md, flags=re.MULTILINE)
    md = re.sub(r'^## (.*)', r'<h2>\1</h2>', md, flags=re.MULTILINE)
    md = re.sub(r'^### (.*)', r'<h3>\1</h3>', md, flags=re.MULTILINE)
    
    # 3. Tables (Zebra + No Ghost Rows)
    def table_replacer(match):
        rows = match.group(0).strip().split('\n')
        html = '<table>'
        row_ct = 0
        for r in rows:
            if re.match(r'^[|\s:\-]+$', r): continue
            cells = [c.strip() for c in r.split('|') if c.strip()]
            if not cells: continue
            tag = 'th' if row_ct == 0 else 'td'
            row_cls = ' class="zebra"' if row_ct % 2 == 0 and row_ct > 0 else ''
            html += f'<tr{row_cls}>' + ''.join([f'<{tag}>{c}</{tag}>' for c in cells]) + '</tr>'
            row_ct += 1
        return html + '</table>'
    md = re.sub(r'(\|.*\|(?:\n\|.*\|)+)', table_replacer, md)

    # 4. Images
    md = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<div class="img-container"><img src="\2" alt="\1"><p class="caption">\1</p></div>', md)

    # 5. Lists (Handle \n properly)
    md = re.sub(r'^\- (.*)', r'<li>\1</li>', md, flags=re.MULTILINE)
    md = re.sub(r'^\d+\. (.*)', r'<li>\1</li>', md, flags=re.MULTILINE)
    md = re.sub(r'(?:<li>.*</li>\n?)+', r'<ul>\g<0></ul>', md)

    # 6. Formatting
    md = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', md)
    md = re.sub(r'\*(.*?)\*', r'<em>\1</em>', md)
    
    # 7. Admonitions
    md = re.sub(r'> \[!(.*?)\]\s*(.*?)\s*(?=\n\n|\Z)', r'<div class="admonition \1"><strong>\1:</strong> \2</div>', md, flags=re.DOTALL | re.IGNORECASE)

    # 8. Paragraph Spacing & Line Breaks
    # Replace double-newline with <p> wrap
    segments = md.split('\n\n')
    output = []
    for s in segments:
        if s.startswith('<h') or s.startswith('<ul') or s.startswith('<table') or s.startswith('<div') or s.startswith('<hr'):
            output.append(s)
        else:
            # Preserve single line breaks as <br> for readability in long lists/text
            s = s.replace('\n', '<br>')
            output.append(f'<p>{s}</p>')
    md = "\n".join(output)

    # 9. Restore Code Blocks
    for i, code in enumerate(code_blocks):
        md = md.replace(f"<!--CODE_BLOCK_{i}-->", f'<pre><code>{code}</code></pre>')

    md = md.replace('---', '<hr>')
    return md

html_body = convert_to_html(text)

styled_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Technical Report - Re-Styled</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px; color: #24292f; line-height: 1.6; }}
        h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; text-align: center; }}
        h2 {{ color: #2E86AB; border-left: 6px solid #2E86AB; padding-left: 15px; background: #f6f8fa; margin-top: 40px; }}
        .img-container {{ text-align: center; margin: 30px 0; }}
        img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #f6f8fa; }}
        tr.zebra {{ background: #fcfcfc; }}
        pre {{ background: #0d1117; color: #e6edf3; padding: 20px; border-radius: 8px; overflow-x: auto; font-size: 14px; margin: 20px 0; }}
        .admonition {{ border: 1px solid #d0d7de; border-left: 10px solid #2F81F7; background: #f8f9ff; padding: 20px; margin: 30px 0; border-radius: 6px; }}
        ul {{ margin: 15px 0; }}
        p {{ margin: 15px 0; }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>
"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(styled_html)

print(f"✅ SUCCESSFULLY FIXED LAYOUT: Newlines preserved, Training Loop simplified.")
