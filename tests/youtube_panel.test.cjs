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
function fixture(initialStorage = {}, deferStorage = false) {
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
  const timers = new Set();
  const timeouts = [];
  // Viewing mode and the player's box on screen; the defaults leave the page in normal mode.
  const view = { theater: false, fullscreen: null, player: null, picture: null, width: 1280, height: 900 };
  video.getBoundingClientRect = () => view.picture;
  const storageReads = [];
  const stored = { ...initialStorage };
  const context = vm.createContext({
    URL,
    location,
    stored,
    get innerWidth() {
      return view.width;
    },
    get innerHeight() {
      return view.height;
    },
    document: {
      documentElement: root,
      get fullscreenElement() {
        return view.fullscreen;
      },
      addEventListener() {},
      createElement: (tag) => new Element(tag),
      querySelector: (selector) =>
        selector === "video.html5-main-video" ? video
          : selector === "ytd-watch-flexy" ? { hasAttribute: (name) => name === "theater" && view.theater }
          : panels,
      getElementById: (id) => ({
        classList: { contains: () => false },
        getBoundingClientRect: id === "movie_player" && view.player ? () => view.player : undefined,
      }),
      querySelectorAll: () => (caption ? [{ textContent: caption }] : []),
    },
    // The caption sampler registers first and is what the tests drive; a check's elapsed-time
    // counter registers later and is cleared when the reply arrives.
    setInterval: (callback) => {
      if (!tick) tick = callback;
      timers.add(callback);
      return callback;
    },
    clearInterval: (callback) => timers.delete(callback),
    setTimeout: (callback, ms) => timeouts.push({ callback, ms }),
    chrome: {
      storage: {
        local: {
          set: (values) => Object.assign(stored, values),
          get: (key, callback) => deferStorage ? storageReads.push(() => callback({[key]:stored[key]})) : callback({ [key]: stored[key] }),
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
    storageReads,
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
    timers,
    view,
    // Runs every pending timeout that was set for exactly this delay.
    elapse: (ms) => {
      const due = timeouts.filter((item) => item.ms === ms);
      timeouts.splice(0, timeouts.length, ...timeouts.filter((item) => item.ms !== ms));
      for (const item of due) item.callback();
    },
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
const blipLayer = (app) => [app.root, ...app.root.children].flatMap((element) => [element, ...element.children])
  .find((element) => element.id === "youtube-fact-checker-blips");
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

test("the Details button opens the panel without checking and closing it keeps the results", async () => {
  const app = fixture();
  const elements = descendants(app.root);
  const check = elements.find((element) => element.className === "check");
  const more = elements.find((element) => element.className === "more");
  const panel = elements.find((element) => element.className === "panel");
  const dismiss = elements.find((element) => element.className === "dismiss");
  assert.equal(panel.hidden, true, "results arrive as blips; the panel stays closed until asked for");
  assert.equal(elements.some((element) => element.className === "launcher"), false, "no tab sits in YouTube's sidebar");
  more.onclick();
  assert.equal(panel.hidden, false);
  assert.equal(more.attributes["aria-expanded"], "true");
  assert.equal(app.requests.length, 0);
  check.onclick();
  assert.equal(check.disabled, true);
  app.reply(result([{ text: "Saved result.", result: { status: "skipped" } }]));
  await flush();
  dismiss.onclick();
  assert.equal(panel.hidden, true);
  assert.equal(more.focused, true);
  assert.equal(more.attributes["aria-expanded"], "false");
  assert.equal(check.disabled, false);
  more.onclick();
  assert.equal(panel.hidden, false);
  assert.match(text(app.root), /Saved result/);
  more.onclick();
  assert.equal(panel.hidden, true, "the same button closes it");
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

test("on Shorts the icon checks without opening a panel, and Details opens one that leaves with the page", () => {
  const app = fixture();
  app.location.pathname = "/shorts/0123456789_";
  app.location.href = "https://www.youtube.com/shorts/0123456789_";
  app.tick();
  const panelHost = () => app.root.children.find((element) => element.id === "youtube-fact-checker");
  assert.equal(panelHost(), undefined);
  app.open();
  assert.equal(app.requests.length, 1);
  assert.equal(panelHost(), undefined, "a check alone opens no panel");
  const more = descendants(blipLayer(app)).find((element) => element.className === "more");
  more.onclick();
  const host = panelHost();
  assert.equal(host.attributes["data-layout"], "shorts");
  descendants(host)
    .find((element) => element.className === "dismiss")
    .onclick();
  app.tick();
  assert.equal(host.isConnected, false);
  more.onclick();
  assert.equal(host.isConnected, true);
  app.location.pathname = "/";
  app.location.href = "https://www.youtube.com/";
  app.tick();
  assert.equal(host.isConnected, false);
  assert.equal(blipLayer(app), undefined, "the blips leave with the video");
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
  assert.match(content, /“First assertion\.”: 1 candidate page\./, "one assertion that is not the whole claim is still named");
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
  assert.equal(descendants(app.root).find(element => element.className?.startsWith("badge")).textContent, "Not established",
    "an unresolved reading says so plainly");
  assert.equal((content.match(/Draft readings of quoted sources by fixed rules; not verified verdicts\./g) || []).length, 1);
  const folded = descendants(app.root).filter(element => element.tag === "details").map(element => element.className);
  assert.ok(folded.includes("related") && folded.includes("how"), "related sentences and the method fold away");
  const related = descendants(app.root).find(element => element.tag === "details" && element.className === "related");
  assert.equal(related.children[0].tag, "summary");
  assert.equal(related.children[0].textContent, "Relevant, not counted (9)", "the folded section names its own count");
  assert.match(content, /contradicted but not established: every counted sentence states a narrower scope\./);
  assert.match(content, /“We have a good job shortage\.” is not established by the sources\./);
  assert.match(content, /wpr\.org\s+\(published 2025-06-25\) \[read as denying it\]: “Wisconsin’s labor shortage is a major barrier <img src=x>\.”/);
  assert.equal((content.match(/\[read as/g) || []).length, 1, "only counted sentences carry a direction");
  assert.equal((content.match(/Shown, not counted: states: this page is read both as stating and as denying the assertion; a misreading, not a dispute\./g) || []).length, 1);
  assert.match(content, /Scope stated in the sentence: Wisconsin\./);
  assert.match(content, /Definition in the same paragraph: “Defined by UW staff\.”/);
  assert.doesNotMatch(content, /uw\.example/);
  assert.match(content, /finance\.yahoo\.com\s+\(publication date unconfirmed\)/);
  assert.match(content, /Definition in the same paragraph: “Defined as the share of the labor force\.”/);
  assert.match(content, /… and 3 more relevant sentences not shown\./);
  assert.doesNotMatch(content, /already shown above/);
  assert.doesNotMatch(content, /Its figure checks|Gives the claim’s figure/, "legacy arithmetic annotations cannot imply a shared measure");
  assert.doesNotMatch(content, /prnewswire\.com/, "a figure from a sentence the card does not quote is not cited");
  assert.equal((content.match(/Wisconsin’s labor shortage is a major barrier/g) || []).length, 2);
  assert.match(content, /Read as context, never as evidence: 12 attributed opinion, 3 instruction or navigation text\./);
  assert.match(content, /it was wrong 10 of 18 times; a counted sentence is a lead to read, not a finding\./, "the judge's record sits on the card");
  assert.equal((content.match(/Relevant sentence/g) || []).length, 5);
  assert.equal((content.match(/Found by caption-concept research, which does not resolve the claim\./g) || []).length, 1);
  assert.equal((content.match(/Dated to June 2022, outside the claim's stated period; shown, not counted\./g) || []).length, 1);
  assert.equal(descendants(app.root).some(element => element.tag === "img"), false);
  assert.doesNotMatch(content, /Supported|Contradicted\b/);
  // The publisher's name is the link, so a quotation never carries an anonymous "open" control.
  const citations = descendants(app.root).filter(element => element.tag === "a" && element.className !== "source");
  assert.deepEqual(citations.map(link => [link.textContent, link.href]), [
    ...Array(2).fill(["wpr.org", "https://www.wpr.org/news/x"]), ...Array(5).fill(["finance.yahoo.com", "https://finance.yahoo.com/a"])]);
  assert.doesNotMatch(content, /Open source/);
  assert.ok(citations.every(link => link.target === "_blank" && link.rel === "noopener noreferrer"));
});

test("invalid citation destinations stay as literal quotations without active links", async () => {
  const urls = ["javascript:alert(1)", "data:text/html,hi", "http://example.org/report",
    "https://user:password@example.org/report", "https://user@example.org/report", "not a URL"];
  const app = fixture();
  app.open();
  app.reply(result([{text: "A factual claim.", result: {
    status: "verified", outcome: "declined_web_review", verdict: null,
    research: {assertions: [], gaps: [], errors: [], sources: []},
    decomposed: {status: "draft", verdict: {relationship: "insufficient", summary: "unresolved", assertions: [{
      text: "A factual claim.", status: "insufficient", evidence: [],
      relevant: urls.map((url, index) => ({unit_id: `u${index}`, url, text: `Literal quotation ${index}.`, relation: "bears_on"})),
    }]}},
  }}]));
  await flush();
  assert.equal(descendants(app.root).filter(element => element.tag === "a").length, 0);
  for (const index of urls.keys()) assert.ok(text(app.root).includes(`Literal quotation ${index}.`));
});

test("each assertion keeps its ranked citations and its own reading regardless of assertion order", async () => {
  const quote = (unit_id, text) => ({unit_id, text, relation: "bears_on", qualifiers: [], definitions: [],
    url: "https://example.org/hours", published_at: ""});
  const shared = quote("hours", "The library and museum open on weekdays.");
  const notices = Array.from({length: 7}, (_, i) => quote(`notice-${i}`, `Opening-hours notice ${i}.`));
  const first = {id: "a1", text: "The library is open.", status: "supported",
    evidence: [{...shared, relation: "states", qualifiers: ["weekdays"]}], relevant: notices.slice(0, 3)};
  const second = {id: "a2", text: "The museum is open.", status: "insufficient", evidence: [],
    relevant: [notices[2], {...shared, qualifiers: ["museum"]}, notices[0], ...notices.slice(3)]};
  const original = JSON.stringify([first, second]);
  let alone;
  for (const assertions of [[second], [first, second], [second, first]]) {
    const app = fixture();
    app.open();
    app.reply(result([{text: "The library is open. The museum is open.", result: {
      status: "verified", outcome: "declined_web_review", verdict: null,
      research: {assertions: [], gaps: [], errors: [], sources: []},
      decomposed: {status: "draft", verdict: {relationship: "insufficient", summary: "unresolved", assertions}},
    }}]));
    await flush();
    const heading = descendants(app.root).find(element => element.textContent.startsWith("“The museum is open.” is"));
    const section = [];
    for (let element = heading.nextElementSibling; element; element = element.nextElementSibling) {
      if (element.textContent.startsWith("“The library is open.” is")) break;
      section.push(element);
    }
    const quotes = section.flatMap(element => [element, ...descendants(element)])
      .filter(element => element.className === "evidence").map(text);
    alone ??= quotes;
    assert.deepEqual(quotes, alone, "earlier assertions cannot fill this assertion's slots with lower-ranked citations");
    assert.equal(quotes.length, 6);
    for (const [index, row] of second.relevant.slice(0, 6).entries()) assert.ok(quotes[index].includes(row.text));
    assert.match(quotes[1], /Scope stated in the sentence: museum\./);
    assert.doesNotMatch(quotes[1], /Scope stated in the sentence: weekdays\.|\[read as stating it\]/);
    assert.match(section.map(text).join(" "), /1 more relevant sentences not shown/);
    assert.doesNotMatch(section.map(text).join(" "), /already shown above|Opening-hours notice 6/);
  }
  assert.equal(JSON.stringify([first, second]), original);
});

test("a viewer-declared claim country travels with the check and is shown as supplied by the viewer", async () => {
  const app = fixture();
  app.open();
  assert.equal(app.requests[0].claim_country, "", "an empty correction lets the background detect context");
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
  assert.match(text(app.root), /Country could not be detected confidently/);
  input.value = "Canada";
  input.onchange();
  assert.equal(app.stored["context:abcdefghijk"].country, "Canada", "the declaration is kept for this video");
  input.value = "<nope>";
  input.onchange();
  assert.equal(app.stored["context:abcdefghijk"].country, "", "only a valid place name is kept");
});

test("corrections are restored only for their video and old global country settings are ignored", () => {
  const app = fixture({ claim_country: "Canada", "context:abcdefghijk": {country:"Curaçao",spoken:"2025-07-15"} });
  app.open();
  const inputs = descendants(app.root).filter(element => element.tag === "input");
  assert.equal(inputs[0].value, "Curaçao");
  assert.equal(app.requests[0].claim_country, "Curaçao");
  assert.equal(app.requests[0].spoken_at, "2025-07-15");
  app.location.href = "https://www.youtube.com/watch?v=0123456789_";
  app.tick();
  app.open();
  assert.equal(app.requests[1].claim_country, "");
  assert.equal(app.requests[1].spoken_at, "");
  assert.equal(inputs[0].value, "");
  assert.equal(inputs[1].value, "");
  app.location.href = "https://www.youtube.com/watch?v=abcdefghijk";
  app.tick();
  assert.equal(inputs[0].value, "Curaçao");
  assert.equal(inputs[1].value, "2025-07-15");
});

test("an immediate check waits for saved context and late storage cannot overwrite a correction", async () => {
  const app = fixture({"context:abcdefghijk":{country:"Canada",spoken:"2025-07-15"}},true);
  app.open();
  assert.equal(app.requests.length,0);
  app.storageReads[0]();
  await flush();
  assert.equal(app.requests[0].claim_country,"Canada");
  assert.equal(app.requests[0].spoken_at,"2025-07-15");
  const pending = fixture({"context:abcdefghijk":{country:"Canada"}},true);
  const input = descendants(pending.root).find(element=>element.tag === "input");
  input.value = "France";
  input.oninput();
  pending.storageReads[0]();
  assert.equal(input.value,"France");
});

test("automatic context is attributed and does not become a saved manual correction", async () => {
  const app = fixture();
  app.open();
  app.reply({result:{...result([]).result,claim_country:"India",country_basis:"detected from captions",
    source_published_at:"2025-07-20",spoken_at:"2025-07-15",spoken_at_basis:"recording date stated in the video description or title"}});
  await flush();
  assert.match(text(app.root),/India \(detected from captions\)/);
  assert.match(text(app.root),/Video published: 2025-07-20/);
  assert.match(text(app.root),/Spoken on: 2025-07-15 \(recording date stated/);
  assert.equal(descendants(app.root).find(element=>element.tag === "input").value,"");
  assert.deepEqual(app.stored,{});
  app.open();
  assert.equal(app.requests[1].claim_country,"");
  assert.equal(app.requests[1].spoken_at,"");
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
    {text: "Rates rose.", result: draft("established", "supported", [{...half("a", "supported"), text: "Rates rose.", basis: "one source: rbc.com (published 2025-07-09)"}])},
    {text: "We lost jobs. We gained hours.", result: draft("qualified_support", "qualified",
                                                          [half("a", "qualified_support"), half("b", "insufficient")])},
  ]));
  await flush();
  const badges = descendants(app.root).filter(element => element.className?.startsWith("badge"));
  // Newest first: the last sentence spoken is the claim the viewer just heard.
  assert.deepEqual(badges.map(element => [element.textContent, element.className]), [
    ["Supported, not established", "badge reading-for"],
    ["Supported", "badge reading-for"],
    ["Contradicted, not established", "badge reading-against"],
    ["Not established", "badge declined"],
    ["Contradicted", "badge reading-against"],
    ["Partially established", "badge reading-for"],
    ["Partially contradicted", "badge reading-against"],
  ]);
  assert.match(text(app.root), /Contradicted, not established\. /, "one assertion: no 'partially'");
  assert.match(text(app.root), /One part supported, not established\. /, "a contrast names the part");
  assert.doesNotMatch(text(app.root), /The verifier gives no verdict/, "news checks do not describe the offline verifier");
  assert.equal((text(app.root).match(/Draft readings of quoted sources by fixed rules; not verified verdicts\./g) || []).length, 1,
    "the draft caveat is stated once per panel, not once per card");
  assert.match(text(app.root), /Rates rose\. Supported \(one source: rbc\.com \(published 2025-07-09\)\)\./,
    "an assertion that is the whole claim is not quoted again");
});

test("the claim just heard comes first and skipped sentences are listed once, not as cards", async () => {
  const app = fixture();
  app.open();
  const checked = (claim) => ({status: "verified", outcome: "declined_web_review", verdict: null,
    research: {assertions: [], gaps: [], errors: [], sources: []},
    decomposed: {status: "draft", verdict: {relationship: "insufficient", summary: "unresolved", scope_limits: [], withheld: {},
      assertions: [{id: "a", text: claim, status: "insufficient", evidence: [], relevant: [], judged: 0, eligible: 0}]}}});
  app.reply(result([
    {text: "Earlier claim about rates.", result: checked("Earlier claim about rates.")},
    {text: "Okay?", result: {status: "skipped"}},
    {text: "No. Yes, we do.", result: {status: "skipped"}},
    {text: "He said they're fine.", result: {status: "skipped",
      filter_reason: "reported speech: quotations are never counted as evidence"}},
    {text: "Latest claim about jobs.", result: checked("Latest claim about jobs.")},
  ]));
  await flush();
  const claims = descendants(app.root).filter(element => element.className === "claim").map(element => element.textContent);
  assert.deepEqual(claims, ["Latest claim about jobs.", "Earlier claim about rates."]);
  const skipped = descendants(app.root).filter(element => element.tag === "details" && element.className === "skipped");
  assert.equal(skipped.length, 1);
  assert.match(text(skipped[0]), /3 sentences not checked.*Okay\?.*No\. Yes, we do\./s);
  assert.match(text(skipped[0]), /He said they're fine\. \(reports what someone said\)/, "a known skip names its reason");
  assert.doesNotMatch(text(skipped[0]), /Okay\? \(/, "a detector skip carries no invented reason");
  assert.equal(descendants(app.root).filter(element => element.tag === "article").length, 2);
});

test("a running check shows its elapsed time and the counter stops with the reply", async () => {
  const app = fixture();
  const status = () => descendants(app.root).find(element => element.className === "status").textContent;
  app.open();
  await flush();
  assert.match(status(), /checking… 0 s\./);
  const counters = [...app.timers].filter(callback => callback !== [...app.timers][0]);
  assert.equal(counters.length, 1, "one counter per running check");
  counters[0]();
  assert.match(status(), /checking… \d+ s\./);
  app.reply(result([{text: "A factual claim.", result: {status: "verified", outcome: "declined_web_review", verdict: null,
    research: {assertions: [], gaps: [], errors: [], sources: []}}}]));
  await flush();
  assert.equal(app.timers.size, 1, "the counter is cleared when the reply arrives");
  const shown = status();
  counters[0]();
  assert.equal(status(), shown, "a late tick cannot overwrite the finished check's status");
});

test("a counted near miss says the second model confirmed it; other rows do not", async () => {
  const app = fixture();
  app.open();
  const row = (unit_id, relation, nli) => ({unit_id, relation, nli, text: `Sentence ${unit_id}.`, qualifiers: [], definitions: [],
    url: "https://www.cnbc.com/jobs", published_at: "2025-09-05"});
  app.reply(result([{text: "Jobs grew.", result: {status: "verified", outcome: "declined_web_review", verdict: null,
    research: {assertions: [], gaps: [], errors: [], sources: []},
    decomposed: {status: "draft", verdict: {relationship: "supported", summary: "established", scope_limits: [], withheld: {}, assertions: [{
      id: "a1", text: "Jobs grew.", status: "supported", limits: [], basis: "one source: cnbc.com (published 2025-09-05)",
      evidence: [row("confirmed", "states", {entailment: 0.97}), row("direct", "states"), row("denial", "states_negation", {entailment: 0.97})],
      relevant: [row("context", "bears_on", {entailment: 0.97})], judged: 3, eligible: 3}]}}}}]));
  await flush();
  const rows = descendants(app.root).filter(element => element.className?.startsWith("evidence")).map(text);
  assert.match(rows[0], /\[read as stating it; confirmed by a second model\]: “Sentence confirmed\.”/);
  assert.match(rows[1], /\[read as stating it\]: “Sentence direct\.”/);
  assert.match(rows[2], /\[read as denying it\]: “Sentence denial\.”/, "confirmation only ever adds supports");
  assert.doesNotMatch(rows[3], /confirmed by a second model/, "a reading that does not count names no confirmation");
});

test("web research names each searched assertion's yield and summarises unreadable pages", async () => {
  const app = fixture();
  app.open();
  app.reply(result([{text: "A factual claim.", result: {status: "verified", outcome: "declined_web_review", verdict: null,
    research: {assertions: [{text: "First half.", query: "first half query", source_ids: ["W1", "W2"]},
                            {text: "Second half.", source_ids: []}],
               gaps: [], errors: ["HTTP 403.", "HTTP 403.", "timed out"], sources: []}}},
    {text: "Rates rose.", result: {status: "verified", outcome: "declined_web_review", verdict: null,
     research: {assertions: [{text: "Rates rose.", query: "rates rose query", source_ids: ["W3"]}], gaps: [], errors: [], sources: []}}}]));
  await flush();
  const content = text(app.root);
  assert.match(content, /“First half\.”: 2 candidate pages\./, "each part of a claim is named with its yield");
  assert.match(content, /“Second half\.”: no usable candidate excerpts\./);
  assert.match(content, /News search: 1 candidate page\./, "an assertion that is the whole claim is not quoted again");
  assert.doesNotMatch(content, /Searched “/, "the visible line names the assertion, not the generated query");
  const queries = descendants(app.root).filter(element => element.tag === "details" && element.children[0]?.textContent === "Search context")
    .map(text).join(" ");
  assert.match(queries, /first half query/, "the exact query stays under Search context");
  assert.match(content, /3 pages could not be read: HTTP 403; timed out\./, "each failure is counted, each reason named once");
  assert.doesNotMatch(content, /\.;/);
});

const counted = (relation, extra = {}) => ({unit_id: "u1", relation, url: "https://www.cnbc.com/2026/03/06/jobs.html",
  published_at: "2026-03-06", text: "It was the third time in five months that the economy lost jobs.", qualifiers: [],
  definitions: [], ...extra});
const reading = (summary, relationship, assertions = []) => ({status: "verified", outcome: "declined_web_review", verdict: null,
  research: {assertions: [], gaps: [], errors: [], sources: []},
  decomposed: {status: "draft", verdict: {relationship, summary, scope_limits: [], withheld: {},
                                          judge_note: "Judge record line.", assertions}}});
const supported = (basis = "one source: cnbc.com (published 2026-03-06)", row = counted("states", {nli: {entailment: 0.99}})) =>
  reading("established", "supported", [{id: "a1", text: "The economy lost jobs.", status: "supported", basis, evidence: [row], relevant: []}]);
const contradicted = (basis) => reading("established", "contradicted", [{id: "a1", text: "We don't have a labor shortage.",
  status: "contradicted", basis, relevant: [],
  evidence: [counted("states", {url: "https://www.rbc.com/insights/labor", published_at: "2025-07-09",
                                text: "At the core of this labor shortage is an aging US population."})]}]);
const stackOf = (app) => descendants(blipLayer(app)).find((element) => element.className === "stack");
const cards = (app) => descendants(blipLayer(app)).filter((element) => element.attributes?.role === "dialog");

test("in normal mode a check waits in a blip, then each claim gets a compact blip and quiet ones fold into a count", async () => {
  const app = fixture();
  app.open();
  assert.match(text(stackOf(app)), /Checking what you just heard · 0 s/);
  assert.equal(descendants(app.root).find((element) => element.className === "panel").hidden, true,
    "nothing opens beside the video while it plays");
  assert.equal(descendants(blipLayer(app)).find((element) => element.className === "trigger").disabled, true);
  app.reply(result([{text: "The economy lost jobs.", result: supported()},
                    {text: "This is one of the ugliest jobs reports.", result: reading("unresolved", "insufficient")},
                    {text: "Okay?", result: {status: "skipped"}}]));
  await flush();
  const blips = () => descendants(stackOf(app)).filter((element) => element.className?.startsWith("blip"));
  assert.deepEqual(blips().map((element) => element.className), ["blip quiet", "blip for"], "newest first, skipped left out");
  assert.match(text(blips()[1]), /Supported · cnbc\.com “The economy lost jobs\.”/);
  assert.match(text(blips()[0]), /Not established “This is one of the ugliest jobs reports\.”/);
  assert.equal(cards(app).length, 0, "normal mode keeps results compact until tapped");
  assert.equal(descendants(blipLayer(app)).filter((element) => element.tag === "a").length, 0);
  app.elapse(6000);
  assert.deepEqual(blips().map((element) => element.className), ["blip for"]);
  const count = descendants(stackOf(app)).find((element) => element.className === "count");
  assert.equal(count.textContent, "1 not established");
  count.onclick();
  assert.equal(descendants(app.root).find((element) => element.className === "panel").hidden, false, "the count opens the details");
  descendants(blips()[0]).find((element) => element.className === "body").onclick();
  const [card] = cards(app);
  assert.match(text(card), /The source says “It was the third time in five months that the economy lost jobs\.”/);
  assert.match(text(card), /one source: cnbc\.com \(published 2026-03-06\)/, "the basis is the composed basis, verbatim");
  assert.match(text(card), /published 2026-03-06 · read as stating it; confirmed by a second model/);
  const record = text(card).indexOf("Judge record line.");
  assert.ok(record >= 0 && record < text(card).indexOf("The source says"),
    "the record says the sentences are quoted below, so it comes before the quotation");
  assert.deepEqual(descendants(card).filter((element) => element.tag === "a").map((link) => [link.textContent, link.href]),
    [["cnbc.com", "https://www.cnbc.com/2026/03/06/jobs.html"], ["Read the source", "https://www.cnbc.com/2026/03/06/jobs.html"]]);
  descendants(card).find((element) => element.className === "collapse").onclick();
  assert.equal(cards(app).length, 0);
  descendants(blips()[0]).find((element) => element.className === "close").onclick();
  assert.equal(blips().length, 0, "a dismissed blip stays dismissed");
});

test("blips sit beside the player in normal mode and below it when YouTube stacks the page", () => {
  const app = fixture();
  app.view.player = {top: 64, right: 854, bottom: 531, width: 830, height: 467};
  app.view.picture = {right: 854, width: 830};
  app.tick();
  assert.equal(blipLayer(app).attributes["data-mode"], "normal");
  assert.deepEqual({...stackOf(app).style}, {top: "72px", right: "24px", width: "344px", maxHeight: ""});
  app.view.width = 1000;
  app.view.player = {top: 64, right: 976, bottom: 600, width: 952, height: 536};
  app.view.picture = {right: 976, width: 952};
  app.tick();
  assert.deepEqual({...stackOf(app).style}, {top: "612px", right: "24px", width: "344px", maxHeight: "208px"},
    "with no room beside the video the stack moves below it, not over it, and stops above the Check button");
  assert.equal(blipLayer(app).attributes["data-corner"], "false");
  app.view.height = 700;
  app.tick();
  assert.deepEqual({...stackOf(app).style}, {top: "76px", right: "36px", width: "344px", maxHeight: "464px"},
    "with no room below either, it goes to the player's corner rather than off screen");
  assert.equal(blipLayer(app).attributes["data-corner"], "true");
});

test("in theater mode the newest directional reading opens inside the black bar beside the video", async () => {
  const app = fixture();
  app.view.theater = true;
  app.view.player = {top: 56, right: 1280, bottom: 607, width: 1280, height: 551};
  app.view.picture = {right: 1130, width: 980};
  app.tick();
  assert.equal(blipLayer(app).attributes["data-mode"], "theater");
  assert.equal(blipLayer(app).attributes["data-narrow"], "true");
  assert.deepEqual({...stackOf(app).style}, {top: "68px", right: "8px", width: "134px", maxHeight: "479px"});
  app.open();
  app.reply(result([{text: "The economy lost jobs.", result: supported()},
                    {text: "Rates rose.", result: supported("one source: cnbc.com (published 2026-03-06)", counted("states"))}]));
  await flush();
  const [card] = cards(app);
  assert.match(text(card), /Rates rose\./, "the claim just heard opens");
  assert.doesNotMatch(text(card), /confirmed by a second model/, "only a confirmed near miss says so");
  assert.equal(cards(app).length, 1, "older readings stay compact");
  app.view.width = 1264;
  app.view.picture = {right: 1122, width: 980};
  app.tick();
  assert.deepEqual([stackOf(app).style.right, stackOf(app).style.width], ["8px", "126px"],
    "a player wider than the visible page is clipped to it, so the stack never runs off screen");
  app.view.picture = {right: 1220, width: 1160};
  app.tick();
  assert.equal(stackOf(app).style.top, "619px", "with no black bar the stack goes below the player");
  assert.equal(stackOf(app).style.maxHeight, "201px", "and stops above the Check button");
  assert.equal(blipLayer(app).attributes["data-narrow"], "false");
});

test("in theater mode with narrow black bars and no room below, the reading opens over the player's corner and folds", async () => {
  // The live 1440x900 geometry: a 16:9 picture leaves 70 px bars and the player ends 113 px above the bottom.
  const app = fixture();
  app.view.theater = true;
  app.view.width = 1440;
  app.view.player = {top: 56, right: 1440, bottom: 787, width: 1440, height: 731};
  app.view.picture = {right: 1370, width: 1300};
  app.tick();
  assert.deepEqual({...stackOf(app).style}, {top: "68px", right: "12px", width: "344px", maxHeight: "659px"});
  assert.equal(blipLayer(app).attributes["data-corner"], "true", "styled as see-through chips");
  assert.equal(blipLayer(app).attributes["data-mode"], "theater", "the Check button stays, unlike fullscreen");
  app.open();
  app.reply(result([{text: "Rates rose.", result: supported("one source: cnbc.com (published 2026-03-06)", counted("states"))}]));
  await flush();
  assert.equal(cards(app).length, 1, "the reading opens by itself");
  app.elapse(15000);
  assert.equal(cards(app).length, 0, "over the picture it folds back after 15 s");
  assert.ok(descendants(stackOf(app)).some((element) => element.className === "blip for"));
});

test("in fullscreen the reading opens in the corner, folds into a chip after 15 s, and a single contradiction is flagged", async () => {
  const app = fixture();
  const screen = new Element("div");
  app.root.append(screen);
  app.view.fullscreen = screen;
  app.tick();
  assert.equal(blipLayer(app).parentElement, screen, "only the fullscreen element is shown, so the blips move into it");
  assert.equal(blipLayer(app).attributes["data-mode"], "fullscreen");
  assert.equal(blipLayer(app).attributes["data-corner"], "true");
  assert.deepEqual({...stackOf(app).style}, {top: "20px", right: "20px", width: "344px", maxHeight: ""});
  app.open();
  app.reply(result([{text: "We don't have a labor shortage.", result: contradicted("one source: rbc.com (published 2025-07-09)")}]));
  await flush();
  const [card] = cards(app);
  assert.match(text(card), /Contradicted/);
  assert.match(text(card), /One source only\. Read it before you repeat this\./);
  app.elapse(15000);
  assert.equal(cards(app).length, 0);
  const blip = descendants(stackOf(app)).find((element) => element.className === "blip against");
  assert.match(text(blip), /Contradicted · rbc\.com/);
  assert.match(text(blip), /One source only/, "the warning survives folding");
  app.view.fullscreen = null;
  app.tick();
  assert.equal(blipLayer(app).parentElement, app.root);
});

test("a contradiction from two independent sources carries no single-source warning", async () => {
  const app = fixture();
  app.open();
  app.reply(result([{text: "We don't have a labor shortage.",
    result: contradicted("2 independent sources: rbc.com (published 2025-07-09), shrm.org (published 2025-01-13)")}]));
  await flush();
  assert.doesNotMatch(text(stackOf(app)), /One source only/);
});

test("qualified readings and the offline verifier's abstentions stay quiet", async () => {
  const app = fixture();
  app.open();
  app.reply(result([{text: "Prices fell.", result: reading("qualified_contradiction", "qualified")},
                    {text: "Unclear claim.", result: {status: "verified", outcome: "answered", verdict: "not_enough_evidence", evidence: []}}]));
  await flush();
  const blips = descendants(stackOf(app)).filter((element) => element.className?.startsWith("blip"));
  assert.deepEqual(blips.map((element) => element.className), ["blip quiet", "blip quiet"]);
  assert.match(text(blips[1]), /Contradicted, not established/, "the exact reading label is kept");
});

test("a failed check offers a retry and Stop ignores the reply that arrives afterwards", async () => {
  const app = fixture();
  app.open();
  app.reply({error: "YouTube refused the caption request (HTTP 429)."});
  await flush();
  const failure = descendants(stackOf(app)).find((element) => element.attributes?.role === "alert");
  assert.match(text(failure), /HTTP 429/);
  descendants(failure).find((element) => element.className === "text-button retry").onclick();
  assert.equal(app.requests.length, 2);
  descendants(stackOf(app)).find((element) => element.textContent === "Stop").onclick();
  assert.equal(descendants(blipLayer(app)).find((element) => element.className === "trigger").disabled, false);
  app.reply(result([{text: "Late claim.", result: supported()}]));
  await flush();
  assert.doesNotMatch(text(app.root), /Late claim/);
});

test("nothing checkable says so briefly and leaves no count", async () => {
  const app = fixture();
  app.open();
  app.reply(result([{text: "Okay?", result: {status: "skipped"}}]));
  await flush();
  assert.match(text(stackOf(app)), /No checkable claim in the last 30 seconds of captions\./);
  app.elapse(6000);
  assert.equal(descendants(stackOf(app)).length, 1, "the note folds away without becoming a count");
});
