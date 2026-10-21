from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from ui_rag.generation_eval.human_audit import (
    RUBRIC,
    select_blinded_audit_cases,
    write_blinded_manifest,
    write_private_key,
)


def _esc(x: str) -> str:
    return html.escape(x, quote=True)


def _rubric_html() -> str:
    blocks = []
    labels = {
        "faithfulness": "Faithfulness",
        "relevance": "Relevance",
        "unsupported": "Unsupported-claim severity",
        "completeness": "Completeness",
    }
    for key, title in labels.items():
        parts = [f"<strong>{title}</strong>"]
        for score in [0, 1, 2]:
            parts.append(f"{score}: {_esc(RUBRIC[key][score])}")
        blocks.append("<div class='rubric-block'>" + "<br>".join(parts) + "</div>")
    return "".join(blocks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--judged", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--size", type=int, default=12)
    args = p.parse_args()

    rows = [json.loads(x) for x in Path(args.judged).read_text(encoding="utf-8").splitlines() if x.strip()]
    cases = select_blinded_audit_cases(rows, total=args.size)
    out = Path(args.output_dir)
    assets = out / "assets"
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    for c in cases:
        src = Path(c.rank1_path)
        if not src.exists():
            raise FileNotFoundError(f"missing RICO evidence image: {src}")
        shutil.copy2(src, assets / f"{c.audit_id}_{c.rank1_filename}")

    write_blinded_manifest(cases, out / "audit_manifest.json")
    write_private_key(cases, out / "audit_key.json")

    cards = []
    for c in cases:
        img = f"assets/{c.audit_id}_{c.rank1_filename}"
        cards.append(f"""
<section class="case" data-audit-id="{_esc(c.audit_id)}">
  <h2>{_esc(c.audit_id)}</h2>
  <div class="grid">
    <div><img src="{_esc(img)}" alt="Evidence screenshot"></div>
    <div>
      <h3>Query</h3><p>{_esc(c.query)}</p>
      <h3>Generated answer</h3><p>{_esc(c.answer)}</p>
      <div class="scores">
        <label>Faithfulness <select data-field="human_faithfulness_0_2"><option value="">—</option><option>0</option><option>1</option><option>2</option></select></label>
        <label>Relevance <select data-field="human_relevance_0_2"><option value="">—</option><option>0</option><option>1</option><option>2</option></select></label>
        <label>Unsupported <select data-field="human_unsupported_0_2"><option value="">—</option><option>0</option><option>1</option><option>2</option></select></label>
        <label>Completeness <select data-field="human_completeness_0_2"><option value="">—</option><option>0</option><option>1</option><option>2</option></select></label>
      </div>
      <label>Notes<textarea data-field="human_notes" rows="3"></textarea></label>
    </div>
  </div>
</section>
""")

    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>M7.2 blinded human audit</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1180px;margin:24px auto;padding:0 18px;line-height:1.45}}
.notice{{padding:14px;border:1px solid #999;border-radius:10px;margin-bottom:18px}}
.rubric{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
.rubric-block{{padding:10px;border:1px solid #ddd;border-radius:8px}}
.case{{border-top:2px solid #222;padding:24px 0}}
.grid{{display:grid;grid-template-columns:minmax(260px,420px) 1fr;gap:24px;align-items:start}}
img{{width:100%;height:auto;border:1px solid #aaa}}
.scores{{display:grid;grid-template-columns:repeat(2,minmax(180px,1fr));gap:12px;margin:18px 0}}
label{{display:flex;flex-direction:column;gap:5px}}
select,textarea{{font:inherit;padding:7px}}
button{{font:inherit;padding:10px 16px;margin:16px 8px 16px 0}}
#status{{font-weight:600}}
@media(max-width:760px){{.grid,.rubric,.scores{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>M7.2 blinded human audit</h1>
<div class="notice"><strong>Blinding:</strong> retrieval slice, target filename, and automated judge score are intentionally hidden. Score only the query, generated answer, and displayed E1 screenshot. Complete every field, then export the CSV.</div>
<div class="rubric">{_rubric_html()}</div>
<div><button onclick="exportCSV()">Save completed CSV</button><button onclick="copyCSV()">Copy CSV fallback</button><span id="status"></span></div>
{''.join(cards)}
<script>
function q(s){{return '"'+String(s).replaceAll('"','""')+'"';}}
function buildCSV(){{
  const headers=['audit_id','human_faithfulness_0_2','human_relevance_0_2','human_unsupported_0_2','human_completeness_0_2','human_notes'];
  const rows=[headers];
  let missing=[];
  document.querySelectorAll('.case').forEach(card=>{{
    const aid=card.dataset.auditId;
    const vals={{audit_id:aid}};
    card.querySelectorAll('[data-field]').forEach(el=>vals[el.dataset.field]=el.value);
    for(const k of headers.slice(1,5)) if(vals[k]==='') missing.push(aid+': '+k);
    rows.push(headers.map(h=>vals[h]??''));
  }});
  if(missing.length) throw new Error('Missing scores: '+missing.slice(0,4).join(', ')+(missing.length>4?' …':''));
  return rows.map(r=>r.map(q).join(',')).join('\n');
}}
async function exportCSV(){{
  const status=document.getElementById('status');
  try{{
    const csv=buildCSV();
    if(location.protocol==='http:' || location.protocol==='https:'){{
      status.textContent='Saving…';
      const resp=await fetch('/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{csv}})}});
      const body=await resp.json();
      if(!resp.ok || !body.ok) throw new Error(body.error || 'save failed');
      status.textContent='Saved directly: '+body.path;
      return;
    }}
    const blob=new Blob([csv],{{type:'text/csv;charset=utf-8'}});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;a.download='human_audit_completed.csv';a.style.display='none';
    document.body.appendChild(a);a.click();
    setTimeout(()=>{{URL.revokeObjectURL(url);a.remove();}},1500);
    status.textContent='Download requested. If the browser blocks it, use Copy CSV fallback or run make m7-human-audit-serve.';
  }}catch(e){{status.textContent=String(e.message || e);}}
}}
async function copyCSV(){{
  const status=document.getElementById('status');
  try{{
    const csv=buildCSV();
    await navigator.clipboard.writeText(csv);
    status.textContent='CSV copied to clipboard. Paste it into human_audit_completed.csv if needed.';
  }}catch(e){{
    status.textContent='Clipboard blocked too. Run make m7-human-audit-serve.';
  }}
}}
</script></body></html>"""
    (out / "audit.html").write_text(page, encoding="utf-8")
    print("M7.2 blinded human audit bundle created")
    print(f"cases={len(cases)}")
    print(f"html={out / 'audit.html'}")
    print(f"manifest={out / 'audit_manifest.json'}")
    print(f"private_key={out / 'audit_key.json'}")
    print("Recommended: run make m7-human-audit-serve, score all cases, then press Save completed CSV. file:// download remains a fallback only.")


if __name__ == "__main__":
    main()
