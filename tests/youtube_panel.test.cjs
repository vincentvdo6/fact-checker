const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");
const { spawnSync } = require("node:child_process");

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.style = {};
    this.textContent = "";
    this.attributes = {};
    this.parentElement = null;
  }
  append(...children) {
    for (const child of children) {
      child.remove();
      child.parentElement = this;
    }
    this.children.push(...children);
  }
  insertBefore(child, reference) {
    child.remove();
    this.children.splice(this.children.indexOf(reference), 0, child);
    child.parentElement = this;
  }
  remove() {
    if (this.parentElement) {
      const siblings = this.parentElement.children;
      siblings.splice(siblings.indexOf(this), 1);
      this.parentElement = null;
    }
  }
  get nextElementSibling() {
    const siblings = this.parentElement?.children || [];
    return siblings[siblings.indexOf(this) + 1] || null;
  }
  get isConnected() {
    return this.tag === "html" || Boolean(this.parentElement?.isConnected);
  }
  scrollIntoView() {}
  focus() {
    this.focused = true;
  }
  replaceChildren(...children) {
    this.children = children;
  }
  setAttribute(name, value) {
    this.attributes[name] = value;
  }
  attachShadow() {
    this.shadow = new Element("shadow");
    return this.shadow;
  }
}
const descendants = (element) => [
  element,
  ...element.children.flatMap(descendants),
  ...(element.shadow ? descendants(element.shadow) : []),
];
function fixture(initialStorage = {}) {
  const root = new Element("html"),
    below = new Element("div"),
    sidebar = new Element("div"),
    video = { currentTime: 10 };
  let panels = new Element("div");
  panels.id = "panels";
  root.append(below, sidebar);
  sidebar.append(panels);
  const location = {
    href: "https://www.youtube.com/watch?v=abcdefghijk",
    pathname: "/watch",
  };
  let tick,
    listener,
    resolve,
    caption = "";
  const requests = [];
  const stored = { ...initialStorage };
  const context = vm.createContext({
    URL,
    location,
    stored,
    document: {
      documentElement: root,
      addEventListener() {},
      createElement: (tag) => new Element(tag),
      querySelector: (selector) =>
        selector === "video.html5-main-video" ? video : panels,
      getElementById: () => ({ classList: { contains: () => false } }),
      querySelectorAll: () => (caption ? [{ textContent: caption }] : []),
    },
    setInterval: (callback) => {
      tick = callback;
    },
    chrome: {
      storage: {
        local: {
          set: (values) => Object.assign(stored, values),
          get: (key, callback) => callback({ [key]: stored[key] }),
        },
      },
      runtime: {
        onMessage: {
          addListener: (callback) => {
            listener = callback;
          },
        },
        sendMessage: (message) => {
          requests.push(message);
          return new Promise((done) => {
            resolve = done;
          });
        },
      },
    },
  });
  vm.runInContext(
    readFileSync(join(__dirname, "../extension/youtube/panel.js"), "utf8"),
    context,
  );
  tick();
  return {
    root,
    video,
    location,
    requests,
    stored,
    below,
    sidebar,
    panels,
    movePanels: () => below.append(panels),
    replacePanels: () => {
      panels.remove();
      panels = new Element("div");
      panels.id = "panels";
      sidebar.append(panels);
    },
    tick: () => tick(),
    caption: (text) => {
      caption = text;
    },
    open: () => listener({ type: "open-checker" }),
    reply: (data) => resolve(data),
  };
}
const text = (root) =>
  descendants(root)
    .filter((element) => element.tag !== "style")
    .map((element) => element.textContent)
    .join(" ");
const flush = () => new Promise((resolve) => setImmediate(resolve));
const result = (rows) => ({
  result: {
    video_id: "abcdefghijk",
    title: "Fixture",
    start: 1,
    end: 8,
    clicked_at: 10,
    excerpt: "Caption excerpt.",
    source: "track",
    rows,
  },
});

test("checker styles stay inside its shadow root and native panels remain untouched", () => {
  const app = fixture();
  app.panels.setAttribute("visibility", "ENGAGEMENT_PANEL_VISIBILITY_EXPANDED");
  app.panels.textContent = "YouTube transcript";
  const original = { ...app.panels.attributes };
  const host = app.sidebar.children[0];
  host.scrollIntoView = () =>
    assert.fail("opening the checker scrolled YouTube");
  app.open();
  descendants(host)
    .find((element) => element.className === "dismiss")
    .onclick();
  app.tick();
  const lightElements = (element) => [
    element,
    ...element.children.flatMap(lightElements),
  ];
  assert.equal(
    lightElements(app.root).some((element) => element.tag === "style"),
    false,
  );
  assert.equal(
    descendants(host.shadow).some((element) => element.tag === "style"),
    true,
  );
  assert.deepEqual(app.panels.attributes, original);
  assert.equal(app.panels.hidden, undefined);
  assert.equal(app.panels.textContent, "YouTube transcript");
  assert.equal(app.panels.isConnected, true);
  assert.equal(app.panels.parentElement, app.sidebar);
});

test("closed checker tab follows the transcript area across layout changes and navigation", () => {
  const app = fixture();
  const host = app.sidebar.children[0];
  assert.equal(host.id, "youtube-fact-checker");
  assert.equal(host.nextElementSibling.id, "panels");
  assert.equal(host.attributes["data-layout"], "sidebar");
  assert.equal(app.requests.length, 0);
  const panel = descendants(host).find(
    (element) => element.className === "panel",
  );
  assert.equal(panel.hidden, true);
  app.replacePanels();
  app.tick();
  app.tick();
  assert.equal(app.sidebar.children.length, 2);
  assert.equal(app.sidebar.children[0], host);
  app.movePanels();
  app.tick();
  assert.equal(app.below.children.length, 2);
  assert.equal(app.below.children[0], host);
  app.location.href = "https://www.youtube.com/";
  app.location.pathname = "/";
  app.tick();
  assert.equal(host.isConnected, false);
  app.location.href = "https://www.youtube.com/watch?v=abcdefghijk";
  app.location.pathname = "/watch";
  app.tick();
  assert.equal(app.below.children[0], host);
});

test("opening the tab does not check a claim and closing preserves results", async () => {
  const app = fixture();
  const elements = descendants(app.root);
  const check = elements.find((element) => element.className === "check");
  const launcher = elements.find((element) => element.className === "launcher");
  const panel = elements.find((element) => element.className === "panel");
  const dismiss = elements.find((element) => element.className === "dismiss");
  launcher.onclick();
  assert.equal(panel.hidden, false);
  assert.equal(app.requests.length, 0);
  check.onclick();
  assert.equal(launcher.attributes["aria-expanded"], "true");
  assert.equal(check.disabled, true);
  app.reply(result([{ text: "Saved result.", result: { status: "skipped" } }]));
  await flush();
  dismiss.onclick();
  assert.equal(panel.hidden, true);
  assert.equal(launcher.focused, true);
  assert.equal(launcher.attributes["aria-expanded"], "false");
  assert.equal(check.disabled, false);
  launcher.onclick();
  assert.equal(panel.hidden, false);
  assert.match(text(app.root), /Saved result/);
  assert.equal(app.requests.length, 1);
});

test("closing during a check keeps the panel closed when the reply arrives", async () => {
  const app = fixture();
  app.open();
  const panel = descendants(app.root).find(
    (element) => element.className === "panel",
  );
  panel.onkeydown({ key: "Escape", stopPropagation() {} });
  app.reply(
    result([{ text: "Saved while closed.", result: { status: "skipped" } }]),
  );
  await flush();
  assert.equal(panel.hidden, true);
  assert.match(text(app.root), /Saved while closed/);
});

test("Shorts keep an icon-opened panel and do not retain it on other pages", () => {
  const app = fixture();
  app.location.pathname = "/shorts/0123456789_";
  app.location.href = "https://www.youtube.com/shorts/0123456789_";
  app.tick();
  assert.equal(
    descendants(app.root).some(
      (element) => element.id === "youtube-fact-checker",
    ),
    false,
  );
  app.open();
  const host = app.root.children.find(
    (element) => element.id === "youtube-fact-checker",
  );
  assert.equal(host.attributes["data-layout"], "shorts");
  assert.equal(app.requests.length, 1);
  descendants(host)
    .find((element) => element.className === "dismiss")
    .onclick();
  app.tick();
  assert.equal(host.isConnected, false);
  app.open();
  assert.equal(host.isConnected, true);
  app.location.pathname = "/";
  app.location.href = "https://www.youtube.com/";
  app.tick();
  assert.equal(host.isConnected, false);
});

test("panel labels abstention, NEI and filtered speech separately and keeps text literal", async () => {
  const app = fixture();
  app.open();
  app.reply(
    result([
      {
        text: "<img src=x>",
        result: {
          status: "verified",
          outcome: "declined_both",
          predicted: "supported",
          verdict: null,
          evidence: [],
        },
      },
      {
        text: "Unclear claim.",
        result: {
          status: "verified",
          outcome: "answered",
          verdict: "not_enough_evidence",
          evidence: [],
        },
      },
      { text: "Hello.", result: { status: "skipped" } },
    ]),
  );
  await flush();
  const content = text(app.root);
  assert.match(content, /No verdict/);
  assert.match(content, /Not enough evidence/);
  assert.match(content, /retrieval was not run/);
  assert.doesNotMatch(content, /Supported/);
  assert.equal(
    descendants(app.root).some((element) => element.tag === "img"),
    false,
  );
});

test("navigation invalidates an in-flight result", async () => {
  const app = fixture();
  app.open();
  app.location.href = "https://www.youtube.com/watch?v=0123456789_";
  app.tick();
  app.reply(
    result([{ text: "Old video claim.", result: { status: "skipped" } }]),
  );
  await flush();
  assert.doesNotMatch(text(app.root), /Old video claim/);
  assert.match(text(app.root), /Video position changed/);
});

test("compound claims show one card with an explicit limitation and no verdict", async () => {
  const app = fixture();
  app.open();
  app.reply(
    result([
      {
        text: "We don't have a labor shortage. We have a good job shortage.",
        result: {
          status: "verified",
          outcome: "declined_compound_claim",
          verdict: null,
          evidence: [
            [
              "Secondary_labor_market",
              0,
              "The labor market includes part-time work.",
            ],
          ],
        },
      },
    ]),
  );
  await flush();
  assert.equal(
    descendants(app.root).filter((element) => element.tag === "article").length,
    1,
  );
  assert.match(text(app.root), /No verdict/);
  assert.match(text(app.root), /cannot yet assess both together/);
  assert.doesNotMatch(text(app.root), /Contradicted|Supported/);
});

test("visible buffer submits only finished captions and clears on seeking", () => {
  const app = fixture();
  app.caption("First caption.");
  app.tick();
  app.video.currentTime = 11;
  app.caption("Second caption.");
  app.tick();
  app.open();
  assert.equal(app.requests[0].visible.length, 1);
  assert.equal(app.requests[0].visible[0].text, "First caption.");
  assert.equal(app.requests[0].visible[0].end, 11);
  app.video.currentTime = 2;
  app.tick();
  app.open();
  assert.equal(app.requests[1].visible.length, 0);
});

test("web research keeps one unresolved claim and honest dates with safe source links", async () => {
  const app = fixture();
  app.open();
  app.reply(result([{text: "A contrast.", result: {
    status: "verified", outcome: "declined_web_review", verdict: null,
    notes: ["Sentence-level reading is off: the pair judge is not installed."],
    research: {
      assertions: [{text: "First assertion.", needs: "Matching period.", query: "first", source_ids: ["source-1"]}],
      gaps: ["Country is unknown."], errors: ["One page timed out."],
      search_cutoff: "2025-07-20", cutoff_basis: "Video publication date; the speech may be older",
      sources: [
        {id: "source-1", title: "<img src=x>", url: "https://example.org/report", published_at: "",
         feed_published_at: "2026-09-08T05:00:00+00:00",
         retrieved_at: "2026-09-08", temporal_note: "Claim date unknown.", temporal_status: "later_publication",
         role: "Research candidate.", source_context: "A measure of the labor force <img src=x>.", excerpts: ["Literal page text was [24.3%](https://c212.net/c/link/?t=0&l=en) see https://x.y/z now."]},
        {id: "unsafe", title: "Unsafe source", url: "javascript:alert(1)", excerpts: []},
      ],
    },
  }}]));
  await flush();
  const content = text(app.root);
  assert.match(content, /No verdict/);
  assert.match(content, /Published: unknown/);
  assert.match(content, /News feed date: 2026-09-08T05:00:00\+00:00/);
  assert.match(content, /not confirmed publication or event time/);
  assert.match(content, /Country is unknown/);
  assert.match(content, /Sentence-level reading is off: the pair judge is not installed\./);
  assert.match(content, /One page timed out/);
  assert.match(content, /A measure of the labor force <img src=x>/);
  assert.match(content, /Literal page text was 24\.3% see {2}now\./);
  assert.doesNotMatch(content, /c212\.net|x\.y\/z/);
  assert.equal(descendants(app.root).some(element => element.tag === "img"), false);
  assert.match(content, /Source search boundary: 2025-07-20/);
  assert.match(content, /excluded from the assertion’s candidate sources/);
  assert.doesNotMatch(content, /Supported|Contradicted|Unsafe source|Excerpts are from June 2017/);
  const links = descendants(app.root).filter(element => element.tag === "a");
  assert.equal(links.length, 1);
  assert.equal(links[0].href, "https://example.org/report");
  assert.equal(links[0].rel, "noopener noreferrer");
  assert.equal(descendants(app.root).filter(element => element.tag === "article").length, 1);
});

test("rolling visible captions become one assertion through the Python adapter", () => {
  const app = fixture();
  for (const [time, caption] of [
    [10, "The law"],
    [11, "The law passed"],
    [12, "The law passed in 2010."],
    [13, ""],
  ]) {
    app.video.currentTime = time;
    app.caption(caption);
    app.tick();
  }
  app.open();
  assert.equal(app.requests[0].visible.length, 1);
  const payload = {
    video_id: "abcdefghijk",
    title: "Fixture",
    time: 13,
    language: "en",
    source: "visible",
    captions: app.requests[0].visible,
  };
  const python =
    process.platform === "win32"
      ? join(__dirname, "../.venv/Scripts/python.exe")
      : "python";
  const checked = spawnSync(
    python,
    [
      "-c",
      "import json,sys; from src.pipeline.youtube import caption_excerpt; print(caption_excerpt(json.load(sys.stdin)).text)",
    ],
    {
      cwd: join(__dirname, ".."),
      input: JSON.stringify(payload),
      encoding: "utf8",
      env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" },
    },
  );
  assert.equal(checked.status, 0, checked.stderr);
  assert.equal(checked.stdout.trim(), "The law passed in 2010.");
});

test("sentence-level draft shows quoted sentences and its status without a verdict badge", async () => {
  const app = fixture();
  app.open();
  app.reply(result([{text: "We don't have a labor shortage. We have a good job shortage.", result: {
    status: "verified", outcome: "declined_web_review", verdict: null,
    research: {assertions: [], gaps: [], errors: [], sources: []},
    decomposed: {
      status: "draft: rule-composed from an unmeasured judge; not a verified verdict",
      text: "Claim: ...",
      verdict: {
        relationship: "insufficient", summary: "unresolved",
        scope_limits: ["The claim's country has not been established."],
        judge_note: "When this judge counted a sentence on 35 unseen claims (2026-09-12), it was wrong 10 of 18 times; a counted sentence is a lead to read, not a finding.",
        withheld: {"attributed opinion": 12, "instruction or navigation text": 3},
        assertions: [
          {text: "We don't have a labor shortage.", status: "qualified_contradiction", judged: 3, eligible: 3,
           limits: ["every counted sentence states a narrower scope"], basis: "",
           evidence: [{unit_id: "wpr:p1:u1", url: "https://www.wpr.org/news/x", published_at: "2025-06-25", qualifiers: ["Wisconsin"],
                       relation: "states_negation", text: "Wisconsin’s labor shortage is a major barrier <img src=x>.",
                       definitions: [{text: "Defined by [UW](https://uw.example/x) staff."}]}],
           relevant: []},
          {text: "We have a good job shortage.", status: "insufficient", judged: 3, eligible: 3, evidence: [],
           figures_confirmed: [{claimed: "around 25%", read_as: "24.3%", relation: "near", unit_id: "yahoo:p1:u0",
                                source_id: "s2", url: "https://finance.yahoo.com/a", independent_source: "finance.yahoo.com", origin: "assertion"},
                               {claimed: "around 25%", read_as: "24%", relation: "near", unit_id: "pr:p1:u1",
                                source_id: "s7", url: "https://www.prnewswire.com/r", independent_source: "finance.yahoo.com", origin: "assertion"}],
           relevant: [{unit_id: "wpr:p1:u1", url: "https://www.wpr.org/news/x", published_at: "2025-06-25", qualifiers: [],
                       text: "Wisconsin’s labor shortage is a major barrier <img src=x>.", definitions: []},
                      ...Array.from({length: 8}, (_, index) => ({unit_id: `yahoo:p1:u${index}`, url: "https://finance.yahoo.com/a", published_at: "",
             qualifiers: [], text: `Relevant sentence ${index}.`, origin: index === 0 ? "context" : "assertion",
             period: index === 1 ? ["June 2022"] : undefined, relation: "bears_on",
             not_counted: index === 2 ? "states: this page is read both as stating and as denying the assertion; a misreading, not a dispute" : undefined,
             figures: index === 1 ? [{claimed: "around 25%", relation: "near", read_as: "24.3%"}] : [],
             definitions: [{text: "Defined as the share of the labor force."}]}))]},
        ],
      },
    },
  }}]));
  await flush();
  const content = text(app.root);
  assert.match(content, /Sentence-level reading \(draft\)/);
  assert.match(content, /Not established\. draft: rule-composed from an unmeasured judge/);
  assert.equal(descendants(app.root).find(element => element.className?.startsWith("badge")).textContent, "No verdict",
    "an unresolved reading keeps the plain badge");
  assert.match(content, /contradicted but not established: every counted sentence states a narrower scope\./);
  assert.match(content, /“We have a good job shortage\.” is not established by the sources\./);
  assert.match(content, /wpr\.org \(published 2025-06-25\) \[read as denying it\]: “Wisconsin’s labor shortage is a major barrier <img src=x>\.”/);
  assert.equal((content.match(/\[read as/g) || []).length, 1, "only counted sentences carry a direction");
  assert.equal((content.match(/Shown, not counted: states: this page is read both as stating and as denying the assertion; a misreading, not a dispute\./g) || []).length, 1);
  assert.match(content, /Scope stated in the sentence: Wisconsin\./);
  assert.match(content, /Definition in the same paragraph: “Defined by UW staff\.”/);
  assert.doesNotMatch(content, /uw\.example/);
  assert.match(content, /finance\.yahoo\.com \(publication date unconfirmed\)/);
  assert.match(content, /Definition in the same paragraph: “Defined as the share of the labor force\.”/);
  assert.match(content, /… and 2 more relevant sentences not shown\./);
  assert.match(content, /1 relevant sentence already shown above\./);
  assert.match(content, /Its figure checks: “around 25%” is given as 24\.3% by finance\.yahoo\.com — in sentences that bear on the claim without stating it\./);
  assert.doesNotMatch(content, /prnewswire\.com/, "a figure from a sentence the card does not quote is not cited");
  assert.match(content, /Gives the claim’s figure: 24\.3% fits “around 25%”\./);
  assert.equal((content.match(/Wisconsin’s labor shortage is a major barrier/g) || []).length, 1);
  assert.match(content, /Read as context, never as evidence: 12 attributed opinion, 3 instruction or navigation text\./);
  assert.match(content, /it was wrong 10 of 18 times; a counted sentence is a lead to read, not a finding\./, "the judge's record sits on the card");
  assert.equal((content.match(/Relevant sentence/g) || []).length, 6);
  assert.equal((content.match(/Found by caption-concept research, which does not resolve the claim\./g) || []).length, 1);
  assert.equal((content.match(/Dated to June 2022, outside the claim's stated period; shown, not counted\./g) || []).length, 1);
  assert.equal(descendants(app.root).some(element => element.tag === "img"), false);
  assert.doesNotMatch(content, /Supported|Contradicted\b/);
});

test("a viewer-declared claim country travels with the check and is shown as supplied by the viewer", async () => {
  const app = fixture();
  app.open();
  assert.equal(app.requests[0].claim_country, "", "nothing is inferred; unset until the viewer says");
  assert.equal(app.requests[0].spoken_at, "");
  const dateInput = descendants(app.root).find(element => element.tag === "input" && element.type === "date");
  assert.equal(dateInput.attributes["aria-label"], "Date the claims were spoken, if known");
  const input = descendants(app.root).find(element => element.tag === "input" && element.type !== "date");
  assert.equal(input.attributes["aria-label"], "Country the claims in this video are about");
  input.value = "  United   States ";
  app.reply(result([]));
  await flush();
  app.open();
  assert.equal(app.requests[1].claim_country, "United States", "whitespace is collapsed, nothing else is changed");
  input.value = "<script>";
  dateInput.value = "2025-07-15";
  app.reply(result([]));
  await flush();
  app.open();
  assert.equal(app.requests[2].claim_country, "", "a value that is not a place name is sent as unset");
  assert.equal(app.requests[2].spoken_at, "2025-07-15");
  dateInput.value = "15/07/2025";
  app.reply(result([]));
  await flush();
  app.open();
  assert.equal(app.requests[3].spoken_at, "", "only a calendar day is sent");
  app.reply({result: {...result([]).result, claim_country: "United States"}});
  await flush();
  assert.match(text(app.root), /Claims read as about United States \(supplied by you\)\./);
  app.open();
  app.open();
  app.reply({result: {...result([]).result, claim_country: ""}});
  await flush();
  assert.match(text(app.root), /Claim country not set; nothing can be established without it\./);
  input.value = "Canada";
  input.onchange();
  assert.equal(app.stored.claim_country, "Canada", "the declaration is kept locally");
  input.value = "<nope>";
  input.onchange();
  assert.equal(app.stored.claim_country, "", "only a valid place name is kept");
});

test("a stored claim country is restored into the field when the panel is built", () => {
  const app = fixture({ claim_country: "Canada" });
  app.open();
  assert.equal(descendants(app.root).find(element => element.tag === "input" && element.type !== "date").value, "Canada");
  assert.equal(app.requests[0].claim_country, "Canada");
});

test("a reading with a direction becomes the row's badge, marked as a reading", async () => {
  const app = fixture();
  app.open();
  const draft = (summary, relationship, assertions = []) => ({
    status: "verified", outcome: "declined_web_review", verdict: null,
    research: {assertions: [], gaps: [], errors: [], sources: []},
    decomposed: {status: "draft", text: "Claim: ...", verdict: {relationship, summary, scope_limits: [], withheld: {}, assertions}},
  });
  const half = (id, status) => ({id, text: id, negated: false, status, direction: "none", limits: [], evidence: [], relevant: [],
                                 figures_confirmed: [], judged: 0, eligible: 0});
  app.reply(result([
    {text: "We don't have a labor shortage. We have a good job shortage.", result: draft("partially_contradicted", "qualified")},
    {text: "The rate rose.", result: draft("partially_established", "qualified")},
    {text: "The rate fell.", result: draft("established", "contradicted")},
    {text: "Nothing here.", result: draft("unresolved", "insufficient")},
    {text: "Prices fell.", result: draft("qualified_contradiction", "qualified", [half("a", "qualified_contradiction")])},
    {text: "Rates rose.", result: draft("established", "supported", [{...half("a", "supported"), basis: "one source: rbc.com (published 2025-07-09)"}])},
    {text: "We lost jobs. We gained hours.", result: draft("qualified_support", "qualified",
                                                          [half("a", "qualified_support"), half("b", "insufficient")])},
  ]));
  await flush();
  const badges = descendants(app.root).filter(element => element.className?.startsWith("badge"));
  assert.deepEqual(badges.map(element => [element.textContent, element.className]), [
    ["Partially contradicted (reading)", "badge reading-against"],
    ["Partially established (reading)", "badge reading-for"],
    ["Contradicted (reading)", "badge reading-against"],
    ["No verdict", "badge declined"],
    ["Contradicted, not established (reading)", "badge reading-against"],
    ["Supported (reading)", "badge reading-for"],
    ["Supported, not established (reading)", "badge reading-for"],
  ]);
  assert.match(text(app.root), /“a” is supported \(one source: rbc\.com \(published 2025-07-09\)\)\./, "an established assertion names its basis");
  assert.match(text(app.root), /Contradicted, not established\. /, "one assertion: no 'partially'");
  assert.match(text(app.root), /One part supported, not established\. /, "a contrast names the part");
  assert.match(text(app.root), /The verifier gives no verdict\. The sentence-level reading below composes one/);
});
