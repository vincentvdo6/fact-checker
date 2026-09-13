"""The labelling server writes only valid labels back, into the item they belong to, and nothing else."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import HTTPServer

from scripts.label_pairs import Files, handler


def files(tmp_path):
    pairs = {"labeller": "", "labelled_on": "", "question": "q", "rubric": {"states": "s"},
             "items": [{"id": "a/1", "case": "a", "assertion": "It rose.", "sentence": "The rate rose in April.", "judged": "states",
                        "definitions": [], "url": "", "published_at": "", "role_hint": [], "relation": "", "qualifiers": [], "note": ""}]}
    roles = {"labeller": "", "labelled_on": "", "question": "q", "rubric": {"definition": "d"},
             "items": [{"id": "a/u1", "case": "a", "sentence": "The rate rose in April.", "paragraph": "The rate rose in April.",
                        "url": "", "role_hint": [], "role": [], "note": ""}]}
    (tmp_path / "pairs-x.json").write_text(json.dumps(pairs), encoding="utf-8")
    (tmp_path / "roles-x.json").write_text(json.dumps(roles), encoding="utf-8")
    return Files({"pairs-x": tmp_path / "pairs-x.json", "roles-x": tmp_path / "roles-x.json"})


def test_saves_validate_relations_qualifiers_and_roles_and_record_the_labeller(tmp_path):
    store = files(tmp_path)
    item = store.load("pairs-x")["items"][0]
    assert store.save("pairs-x", "a/1", item | {"relation": "proves"}, "V") == 400
    assert store.save("pairs-x", "a/1", item | {"relation": "states", "qualifiers": ["in May"]}, "V") == 400, "not a substring"
    assert store.save("pairs-x", "missing", item | {"relation": "states"}, "V") == 404
    assert store.save("pairs-x", "a/1", item | {"relation": "states", "qualifiers": ["in April"], "note": "ok", "sentence": "changed"}, "V") == 200
    saved = store.load("pairs-x")
    assert saved["items"][0]["relation"] == "states" and saved["items"][0]["qualifiers"] == ["in April"]
    assert saved["items"][0]["sentence"] == "The rate rose in April.", "only labels change; the harvested text is fixed"
    assert saved["items"][0]["judged"] == "states", "the judge's answer survives a save it was never sent with"
    assert saved["labeller"] == "V" and saved["labelled_on"]
    role = store.load("roles-x")["items"][0]
    assert store.save("roles-x", "a/u1", role | {"role": ["definition", "invented"]}, "") == 400
    assert store.save("roles-x", "a/u1", role | {"role": ["reported_observation", "definition"]}, "") == 200
    assert store.load("roles-x")["items"][0]["role"] == ["reported_observation", "definition"]


def test_server_serves_items_and_accepts_a_save_over_http(tmp_path):
    store = files(tmp_path)
    server = HTTPServer(("127.0.0.1", 0), handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = HTTPConnection("127.0.0.1", server.server_address[1])
        client.request("GET", "/items?file=pairs-x")
        served = json.loads(client.getresponse().read())[0]
        assert served["id"] == "a/1" and "judged" not in served, "the judge's answer never reaches the labeller"
        client.request("GET", "/?file=pairs-x")
        page = client.getresponse().read().decode("utf-8")
        assert "Label pairs" in page and '"states"' in page and "The rate rose" not in page, "items load over /items, not inline"
        body = json.dumps({"file": "pairs-x", "id": "a/1", "labeller": "V",
                           "item": store.load("pairs-x")["items"][0] | {"relation": "bears_on"}})
        client.request("POST", "/save", body=body, headers={"content-type": "application/json"})
        assert client.getresponse().status == 200
        assert store.load("pairs-x")["items"][0]["relation"] == "bears_on"
        client.request("POST", "/save", body="{", headers={"content-type": "application/json"})
        assert client.getresponse().status == 400
        client.request("GET", "/items?file=nope")
        assert client.getresponse().status == 404
    finally:
        server.shutdown()
