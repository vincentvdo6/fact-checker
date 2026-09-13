"""
Label the harvested pair and role files in a browser, one item at a time, with the keyboard.

The two files under `labels/` are the only measurement of the judge and the role typer on
news that this project can have, and they are labelled by a person, not a model. This serves
them on localhost and writes each answer straight back into the file, so labelling 98 pairs
and 96 sentences is a few minutes of keystrokes rather than an afternoon in a JSON editor.
Nothing leaves the machine; the page is plain HTML with no framework and no network access
beyond this server. The rubric from the file header is on screen the whole time.

    python -m scripts.label_pairs                       # both files, http://127.0.0.1:8765
    python -m scripts.label_pairs --labels labels/pairs-frozen-2026-09-10.json --port 9000

Keys: 1-4 choose a relation (pairs) or toggle a role (roles), Enter saves and advances,
Backspace goes back, q adds a qualifier from the sentence (pairs), n edits the note.
"""

from __future__ import annotations

import argparse
import html
import json
import webbrowser
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from src.verdict.pair_judgment import RELATIONS
from src.verdict.reading import ROLES

LABELS = Path("labels")

PAGE = """<!doctype html><meta charset="utf-8"><title>Label __KIND__</title>
<style>
body{font:15px/1.45 system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#1b1b1b}
.box{border:1px solid #ccc;border-radius:8px;padding:12px 16px;margin:12px 0;background:#fafafa}
.assertion{font-weight:600}.sentence{font-size:17px}.def{color:#555;font-size:14px;margin-left:12px}
.meta{color:#666;font-size:13px}.keys button{font:inherit;padding:6px 10px;margin:4px 6px 4px 0;border:1px solid #888;border-radius:6px;background:#fff;cursor:pointer}
.keys button.on{background:#285f3f;color:#fff;border-color:#285f3f}.rubric{font-size:13px;color:#444;white-space:pre-wrap}
input,textarea{font:inherit;width:100%;box-sizing:border-box;padding:6px}.q{display:inline-block;background:#eef;padding:2px 6px;border-radius:4px;margin:2px}
progress{width:100%}
</style>
<h2>__TITLE__ <span class="meta" id="pos"></span></h2>
<progress id="bar" max="__N__" value="0"></progress>
<p class="meta">__QUESTION__ &nbsp;·&nbsp; labeller: <input id="who" style="width:200px;display:inline" placeholder="your name"></p>
<div class="box" id="item"></div>
<div class="keys" id="keys"></div>
<p id="quals"></p>
<p><textarea id="note" rows="2" placeholder="note (n)"></textarea></p>
<p><button id="prev">← back (Backspace)</button> <button id="save">save &amp; next (Enter)</button> <button id="skip">skip →</button></p>
<details><summary>Rubric</summary><div class="rubric">__RUBRIC__</div></details>
<script>
const KIND="__KIND__", FILE="__FILE__", CHOICES=__CHOICES__; let items=[], i=0, chosen=[];
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function load(){items=await (await fetch("/items?file="+encodeURIComponent(FILE))).json(); i=items.findIndex(x=>KIND==="pairs"?!x.relation:!(x.role&&x.role.length)); if(i<0)i=0; show();}
function show(){const x=items[i]; document.getElementById("pos").textContent=`${i+1} / ${items.length}`; document.getElementById("bar").value=items.filter(y=>KIND==="pairs"?y.relation:(y.role&&y.role.length)).length;
 let h=`<p class="meta">${esc(x.case)} · ${esc(x.url||"control")} ${x.published_at?"· "+esc(x.published_at):""} · typer hint: ${esc((x.role_hint||[]).join(", "))}</p>`;
 if(KIND==="pairs"){h+=`<p class="assertion">Assertion: ${esc(x.assertion)}</p><p class="sentence">${esc(x.sentence)}</p>`+(x.definitions||[]).map(d=>`<p class="def">Definition in the same paragraph: ${esc(d)}</p>`).join(""); chosen=x.relation?[x.relation]:[];}
 else{h+=`<p class="sentence">${esc(x.sentence)}</p><p class="def">Paragraph: ${esc(x.paragraph)}</p>`; chosen=[...(x.role||[])];}
 document.getElementById("item").innerHTML=h; document.getElementById("note").value=x.note||""; keys(); quals();}
function keys(){document.getElementById("keys").innerHTML=CHOICES.map((c,k)=>`<button data-c="${c}" class="${chosen.includes(c)?"on":""}">${k+1} ${c}</button>`).join("");
 for(const b of document.querySelectorAll("#keys button")) b.onclick=()=>pick(b.dataset.c);}
function pick(c){if(KIND==="pairs")chosen=[c]; else chosen=chosen.includes(c)?chosen.filter(z=>z!==c):[...chosen,c]; keys();}
function quals(){if(KIND!=="pairs")return; const x=items[i]; document.getElementById("quals").innerHTML="Qualifiers (q): "+(x.qualifiers||[]).map((q,k)=>`<span class="q">${esc(q)} <a href="#" data-k="${k}">×</a></span>`).join(" ");
 for(const a of document.querySelectorAll("#quals a")) a.onclick=e=>{e.preventDefault(); x.qualifiers.splice(+a.dataset.k,1); quals();};}
function addQual(){const x=items[i]; const q=window.getSelection().toString().trim()||prompt("exact substring of the sentence:"); if(!q)return; if(!x.sentence.includes(q)){alert("must be an exact substring of the sentence");return;} (x.qualifiers=x.qualifiers||[]).push(q); quals();}
async function save(advance){const x=items[i]; if(KIND==="pairs"){x.relation=chosen[0]||"";} else {x.role=[...chosen];} x.note=document.getElementById("note").value;
 await fetch("/save",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({file:FILE,id:x.id,item:x,labeller:document.getElementById("who").value})});
 if(advance&&i<items.length-1)i++; show();}
document.getElementById("save").onclick=()=>save(true); document.getElementById("skip").onclick=()=>{if(i<items.length-1){i++;show();}};
document.getElementById("prev").onclick=()=>{if(i>0){i--;show();}};
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA"||e.target.tagName==="INPUT"){if(e.key==="Escape")e.target.blur();return;}
 if(e.key>="1"&&e.key<=String(CHOICES.length)){pick(CHOICES[+e.key-1]);e.preventDefault();}
 else if(e.key==="Enter"){save(true);e.preventDefault();} else if(e.key==="Backspace"){document.getElementById("prev").onclick();e.preventDefault();}
 else if(e.key==="q"&&KIND==="pairs"){addQual();e.preventDefault();} else if(e.key==="n"){document.getElementById("note").focus();e.preventDefault();}});
load();
</script>"""

INDEX = """<!doctype html><meta charset="utf-8"><title>Label</title><body style="font:16px system-ui;margin:40px">
<h2>Label files</h2><ul>{links}</ul></body>"""


class Files:
    """Label files keyed by stem; the kind (pairs or roles) is the stem's prefix."""

    def __init__(self, paths: dict[str, Path]) -> None:
        self.paths = paths

    @staticmethod
    def kind_of(stem: str) -> str:
        return "pairs" if stem.startswith("pairs-") else "roles"

    def load(self, stem: str) -> dict:
        return json.loads(self.paths[stem].read_text(encoding="utf-8"))

    def save(self, stem: str, identity: str, item: dict, labeller: str) -> int:
        kind = self.kind_of(stem)
        payload = self.load(stem)
        for index, existing in enumerate(payload["items"]):
            if existing["id"] == identity:
                allowed = {"relation", "qualifiers", "note"} if kind == "pairs" else {"role", "note"}
                update = {key: value for key, value in item.items() if key in allowed}     # harvested text is fixed
                if kind == "pairs":
                    if update.get("relation") not in ("", *RELATIONS):
                        return 400
                    if any(q not in existing["sentence"] for q in update.get("qualifiers", [])):
                        return 400
                elif any(role not in ROLES for role in update.get("role", [])):
                    return 400
                payload["items"][index] = {**existing, **update}
                break
        else:
            return 404
        if labeller and not payload.get("labeller"):
            payload["labeller"] = labeller
        if not payload.get("labelled_on"):
            payload["labelled_on"] = date.today().isoformat()
        self.paths[stem].write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        return 200


def handler(files: Files):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:      # noqa: N802 - http.server API
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            stem = query.get("file", [""])[0]
            if url.path == "/" and not stem:
                links = "".join(f'<li><a href="/?file={html.escape(k)}">{html.escape(str(p))}</a></li>' for k, p in files.paths.items())
                return self._send(200, INDEX.format(links=links).encode("utf-8"))
            if stem not in files.paths:
                return self._send(404, b"unknown label file")
            payload = files.load(stem)
            if url.path == "/items":
                # The judge's own answer (`judged`, from harvest_clicks) stays on disk: the labeller never receives it.
                items = [{key: value for key, value in item.items() if key != "judged"} for item in payload["items"]]
                return self._send(200, json.dumps(items).encode("utf-8"), "application/json")
            kind = files.kind_of(stem)
            choices = list(RELATIONS) if kind == "pairs" else list(ROLES)
            page = (PAGE.replace("__KIND__", kind).replace("__FILE__", html.escape(stem)).replace("__TITLE__", html.escape(files.paths[stem].name))
                    .replace("__N__", str(len(payload["items"]))).replace("__QUESTION__", html.escape(payload["question"]))
                    .replace("__RUBRIC__", html.escape(json.dumps(payload["rubric"], indent=1, ensure_ascii=False)))
                    .replace("__CHOICES__", json.dumps(choices)))
            return self._send(200, page.encode("utf-8"))

        def do_POST(self) -> None:     # noqa: N802 - http.server API
            if urlsplit(self.path).path != "/save":
                return self._send(404, b"")
            length = int(self.headers.get("content-length", "0"))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                status = files.save(body["file"], body["id"], body["item"], str(body.get("labeller", ""))[:80])
            except (ValueError, KeyError, TypeError):
                status = 400
            return self._send(status, b"ok" if status == 200 else b"rejected", "text/plain")

        def log_message(self, *args: object) -> None:
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", action="append", type=Path, default=None, help="label file(s); default: every pairs-*/roles-* file under labels/")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    paths = args.labels or sorted(LABELS.glob("pairs-*.json")) + sorted(LABELS.glob("roles-*.json"))
    files = Files({path.stem: path for path in paths})
    server = HTTPServer(("127.0.0.1", args.port), handler(files))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"labelling {', '.join(str(p) for p in paths)} at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
