"""Gera entrega/TechChallenge_Fase2_Entrega.html (imagens embutidas) a partir do
RELATORIO_TECNICO.md, no mesmo formato da entrega da Fase 1. O PDF é impresso a partir
do HTML (Edge/Chrome headless):

    python scripts/build_entrega.py
    msedge --headless --print-to-pdf=entrega/TechChallenge_Fase2_Entrega.pdf entrega/TechChallenge_Fase2_Entrega.html
"""

import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "entrega" / "TechChallenge_Fase2_Entrega.html"

REPO_URL = "https://github.com/WesleySardi/TechChallenge_Fase2_POSTECH"   # ajustar se o nome for outro
VIDEO_URL = None                                                          # preencher após o upload

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a1a; line-height: 1.5; font-size: 10.5pt;
       max-width: 900px; margin: 0 auto; padding: 0 12px; background: #fff; }
h1 { font-size: 21pt; margin-bottom: 4px; color: #0d1b2a; }
h2 { font-size: 14.5pt; margin-top: 26px; margin-bottom: 8px; color: #0d1b2a;
     border-bottom: 2px solid #2c7fb8; padding-bottom: 4px; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 16px; margin-bottom: 6px; color: #16324f; page-break-after: avoid; }
p, li { text-align: justify; }
p { margin: 6px 0; }
.cover-box { background: #f2f6fa; border: 1px solid #cfe0ee; border-radius: 6px; padding: 14px 18px; margin: 18px 0 26px; }
.cover-box p { margin: 4px 0; }
.cover-box a { color: #1a5a96; word-break: break-all; }
.placeholder { background: #fff3cd; border: 1px dashed #d6a419; padding: 2px 6px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 16px; font-size: 8.8pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #eef3f8; }
img { max-width: 100%; display: block; margin: 12px auto; page-break-inside: avoid; }
code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; padding: 10px; border-radius: 5px; font-size: 8.3pt;
      white-space: pre-wrap; word-break: break-word; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #2c7fb8; margin: 10px 0; padding: 4px 12px; color: #333; background: #f8fbfd; }
"""


def embed_images(html: str) -> str:
    def repl(m):
        path = ROOT / m.group(1)
        if not path.exists():
            return m.group(0)
        data = base64.b64encode(path.read_bytes()).decode()
        return f'src="data:image/png;base64,{data}"'
    return re.sub(r'src="(outputs/figures/[^"]+\.png)"', repl, html)


def main() -> None:
    md_text = (ROOT / "RELATORIO_TECNICO.md").read_text(encoding="utf-8")
    title, body_md = md_text.split("\n", 1)
    body = markdown.markdown(body_md, extensions=["tables", "fenced_code"])
    body = body.replace('href="README.md', f'href="{REPO_URL}/blob/main/README.md')
    video = (f'<a href="{VIDEO_URL}">{VIDEO_URL}</a>' if VIDEO_URL
             else '<span class="placeholder">[inserir link do vídeo (YouTube/Vimeo, até 15 min)]</span>')
    cover = f"""
<h1>Tech Challenge — Fase 2</h1>
<p style="font-size:12.5pt;color:#444">Otimização de modelos de diagnóstico com algoritmos genéticos + interpretação com LLMs</p>
<div class="cover-box">
  <p><strong>Aluno:</strong> Wesley Sardi (entrega individual)</p>
  <p><strong>Repositório Git (código-fonte, README, notebook de demonstração, IaC, testes e resultados):</strong><br>
     <a href="{REPO_URL}">{REPO_URL}</a></p>
  <p><strong>Vídeo de demonstração:</strong><br>{video}</p>
</div>"""
    html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tech Challenge Fase 2 — Entrega</title><style>{CSS}</style></head>
<body>{cover}
{embed_images(body)}
</body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
