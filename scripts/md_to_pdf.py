"""Convierte informe_TFE_completo.md a PDF usando Chrome headless."""
import subprocess
import sys
import os
import markdown

MD_FILE   = os.path.join(os.path.dirname(__file__), '..', 'docs', 'informe_TFE_completo.md')
HTML_FILE = os.path.join(os.path.dirname(__file__), '..', 'docs', 'informe_TFE_completo.html')
PDF_FILE  = os.path.join(os.path.dirname(__file__), '..', 'docs', 'informe_TFE_completo.pdf')
CHROME    = r'C:\Program Files\Google\Chrome\Application\chrome.exe'

CSS = """
@page {
    size: A4;
    margin: 0 1.8cm;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #000;
    max-width: 900px;
    margin: 0 auto;
    padding: 1.8cm 30px;
}
h1 {
    font-size: 22pt;
    color: #000;
    border-bottom: 3px solid #000;
    padding-bottom: 8px;
    margin: 30px 0 16px 0;
}
h2 {
    font-size: 16pt;
    color: #000;
    border-bottom: 1.5px solid #000;
    padding-bottom: 5px;
    margin: 28px 0 12px 0;
    page-break-after: avoid;
}
h3 {
    font-size: 13pt;
    color: #000;
    margin: 20px 0 8px 0;
    page-break-after: avoid;
}
h4 {
    font-size: 11pt;
    color: #000;
    font-style: italic;
    margin: 16px 0 6px 0;
    page-break-after: avoid;
}
p { margin: 0 0 10px 0; }
ul, ol { margin: 6px 0 10px 20px; }
li { margin-bottom: 3px; }
blockquote {
    border-left: 4px solid #000;
    background: #fff;
    padding: 10px 16px;
    margin: 12px 0;
    font-style: italic;
    color: #000;
    border-radius: 0 4px 4px 0;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 10pt;
    page-break-inside: avoid;
}
th {
    background-color: #fff;
    color: #000;
    padding: 7px 10px;
    text-align: left;
    font-weight: 700;
    border: 2px solid #000;
}
td {
    border: 1px solid #000;
    padding: 6px 10px;
    background-color: #fff;
}
pre {
    background-color: #fff;
    color: #000;
    border: 2px solid #000;
    border-radius: 4px;
    padding: 14px 16px;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.5;
    margin: 10px 0 16px 0;
    page-break-inside: avoid;
}
code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    background-color: #f5f5f5;
    color: #000;
    padding: 1px 4px;
    border-radius: 3px;
}
pre code {
    background: transparent;
    color: #000;
    padding: 0;
    border-radius: 0;
    font-size: 9pt;
}
hr {
    border: none;
    border-top: 2px solid #000;
    margin: 28px 0;
}
strong { color: #000; }
img {
    display: block;
    max-width: 100%;
    height: auto;
    margin: 10px auto 16px auto;
}
.fig-title {
    text-align: center;
    font-weight: 700;
    font-size: 10pt;
    margin: 16px 0 4px 0;
    color: #000;
}
.page-break { page-break-before: always; }
@media print {
    body { padding: 0; }
    pre { white-space: pre-wrap; word-wrap: break-word; }
    h1, h2, h3 { page-break-after: avoid; }
}
"""

def main():
    # 1. Leer Markdown
    with open(MD_FILE, encoding='utf-8') as f:
        md_content = f.read()

    # 2. Convertir a HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
    body_html = md.convert(md_content)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TFE — Análisis de Supervivencia en Desastres Marítimos</title>
  <style>{CSS}</style>
</head>
<body>
{body_html}
</body>
</html>"""

    with open(HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML generado: {os.path.abspath(HTML_FILE)}')

    # 3. Convertir a PDF con Chrome headless
    html_abs = os.path.abspath(HTML_FILE)
    pdf_abs  = os.path.abspath(PDF_FILE)

    cmd = [
        CHROME,
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        '--allow-file-access-from-files',
        '--run-all-compositor-stages-before-draw',
        '--print-to-pdf-no-header',
        f'--print-to-pdf={pdf_abs}',
        f'file:///{html_abs}'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode == 0 and os.path.exists(pdf_abs):
        size_kb = os.path.getsize(pdf_abs) / 1024
        print(f'PDF generado:  {pdf_abs}')
        print(f'Tamaño:        {size_kb:.1f} KB')
    else:
        print('ERROR al generar el PDF:')
        print(result.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
