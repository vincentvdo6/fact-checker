(() => {
  if (globalThis.youtubeFactCheckerLoaded) return;
  globalThis.youtubeFactCheckerLoaded = true;
  let videoId = "",
    lastTime = 0,
    visible = [],
    activeCue = null,
    busy = false,
    version = 0,
    shortsOpen = false,
    host = null,
    ui = null;
  const currentId = () =>
    new URL(location.href).searchParams.get("v") ||
    location.pathname.match(/^\/shorts\/([\w-]{11})/)?.[1] ||
    "";
  document.addEventListener(
    "click",
    (event) => {
      if (event.target.closest?.(".ytp-settings-menu, .ytp-subtitles-button")) {
        visible = [];
        activeCue = null;
      }
    },
    true,
  );
  const clock = (value) =>
    `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, "0")}`;
  const node = (tag, text, cls) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (cls) element.className = cls;
    return element;
  };
  function reset() {
    visible = [];
    activeCue = null;
    version++;
    busy = false;
    if (ui) {
      ui.cards.replaceChildren();
      ui.status.textContent =
        "Video position changed. Check the claim you just heard.";
      ui.check.disabled = false;
    }
  }
  function expand(open) {
    ui.panel.hidden = !open;
    ui.launcher.setAttribute("aria-expanded", String(open));
  }
  function mount() {
    const panels =
      location.pathname === "/watch" && currentId()
        ? document.querySelector("ytd-watch-flexy #panels")
        : null;
    if (panels?.parentElement) {
      if (!host) build();
      host.setAttribute("data-layout", "sidebar");
      if (host.nextElementSibling !== panels)
        panels.parentElement.insertBefore(host, panels);
    } else if (shortsOpen && location.pathname.startsWith("/shorts/")) {
      if (!host) build();
      host.setAttribute("data-layout", "shorts");
      if (host.parentElement !== document.documentElement)
        document.documentElement.append(host);
    } else {
      host?.remove();
    }
  }
  setInterval(() => {
    const video = document.querySelector("video.html5-main-video"),
      id = currentId();
    const time = video?.currentTime || 0;
    if (id !== videoId) {
      shortsOpen = false;
      if (ui) expand(false);
    }
    if (id !== videoId || time < lastTime - 0.5 || time > lastTime + 3) reset();
    videoId = id;
    lastTime = time;
    mount();
    const player = document.getElementById("movie_player");
    if (!video || player?.classList.contains("ad-showing")) {
      visible = [];
      activeCue = null;
      return;
    }
    const text = [...document.querySelectorAll(".ytp-caption-segment")]
      .map((span) => span.textContent)
      .join(" ")
      .trim();
    if (activeCue && text.startsWith(activeCue.text + " ")) {
      activeCue.text = text;
    } else if (text !== activeCue?.text) {
      if (activeCue) visible.push({ ...activeCue, end: time });
      activeCue =
        text && text.length <= 4000 ? { start: time, end: time, text } : null;
    }
    visible = visible.filter((cue) => cue.start >= time - 35).slice(-200);
  }, 250);

  function build() {
    host = document.createElement("div");
    host.id = "youtube-fact-checker";
    const shadow = host.attachShadow({ mode: "closed" });
    const style = node("style");
    style.textContent = `
      :host {
        all:initial;
        display:block;
        width:100%;
        margin:0 0 16px;
        color-scheme:light dark;
        --surface:var(--yt-spec-base-background,#fff);
        --text:var(--yt-spec-text-primary,#0f0f0f);
        --muted:var(--yt-spec-text-secondary,#606060);
        --border:var(--yt-spec-10-percent-layer,#ddd);
        --soft:var(--yt-spec-badge-chip-background,#f2f2f2);
        --accent:#285f3f
      }
      :host-context(html[dark]) {
        --surface:#0f0f0f;
        --text:#f1f1f1;
        --muted:#aaa;
        --border:#383838;
        --soft:#272727;
        --accent:#9aceab
      }
      :host([data-layout="shorts"]) {
        position:fixed;
        right:16px;
        top:72px;
        width:370px;
        max-width:calc(100vw - 32px);
        z-index:2147483647
      }
      [hidden] {
        display:none !important
      }
      * {
        box-sizing:border-box
      }
      .panel {
        display:flex;
        flex-direction:column;
        border:1px solid var(--border);
        border-radius:12px;
        background:var(--surface);
        color:var(--text);
        font:13px/1.5 'Segoe UI',Arial,sans-serif;
        margin-top:10px
      }
      
      header {
        display:flex;
        align-items:center;
        flex-wrap:wrap;
        gap:8px 12px;
        padding:10px 14px
      }
      .brand {
        margin-right:auto;
        font-size:15px;
        font-weight:750;
        letter-spacing:-.3px
      }
      button {
        font:inherit;
        cursor:pointer;
        border:1px solid var(--border);
        border-radius:18px;
        background:transparent;
        color:var(--accent);
        padding:8px 11px
      }
      button:disabled {
        opacity:.5;
        cursor:default
      }
      .dismiss {
        border:0;
        color:var(--muted);
        font-size:20px;
        line-height:1
      }
      .launcher {
        font:600 14px/1.5 'Segoe UI',Arial,sans-serif;
        color:var(--text);
        background:var(--soft)
      }
      .launcher[aria-expanded="true"] {
        border-color:var(--accent);
        color:var(--accent)
      }
      :host([data-layout="shorts"]) .launcher {
        display:none
      }
      .body {
        overflow-y:auto;
        max-height:60vh;
        padding:14px;
        border-top:1px solid var(--border)
      }
      .check {
        width:100%;
        background:#285f3f;
        color:white;
        border-color:#285f3f;
        font-weight:600
      }
      .status,.scope,.reason {
        font-size:12px;
        color:var(--muted)
      }
      .country {
        display:flex;
        align-items:center;
        gap:8px;
        margin:10px 0 0;
        font-size:12px;
        color:var(--muted)
      }
      .country input {
        flex:1;
        min-width:0;
        font:inherit;
        padding:4px 6px;
        border:1px solid var(--border);
        border-radius:6px;
        background:var(--surface);
        color:inherit
      }
      .status {
        white-space:pre-wrap;
        overflow-wrap:anywhere
      }
      .scope {
        border-top:1px solid var(--border);
        padding-top:13px;
        margin-bottom:0
      }
      
      article {
        background:var(--surface);
        border:1px solid var(--border);
        border-radius:8px;
        margin:12px 0;
        padding:14px
      }
      .claim {
        font:18px/1.45 Georgia,serif;
        overflow-wrap:anywhere
      }
      .badge {
        display:inline-block;
        border-radius:12px;
        background:var(--soft);
        font-size:11px;
        padding:3px 8px
      }
      .supported {
        background:#e1efe3;
        color:#28633e
      }
      .contradicted {
        background:#f8e6de;
        color:#98472f
      }
      .reading-for {
        background:#e1efe3;
        color:#28633e;
        border:1px dashed #28633e
      }
      .reading-against {
        background:#f8e6de;
        color:#98472f;
        border:1px dashed #98472f
      }
      .reading-mixed {
        background:#f3ecd7;
        color:#6b5416;
        border:1px dashed #6b5416
      }
      details {
        border-top:1px solid var(--border);
        padding-top:10px;
        font-size:12px
      }
      summary {
        cursor:pointer;
        color:var(--accent)
      }
      .evidence {
        border-left:2px solid var(--border);
        padding-left:10px;
        margin:13px 0;
        overflow-wrap:anywhere
      }
      a {
        color:var(--accent)
      }
      .excerpt {
        font-size:12px;
        white-space:pre-wrap;
        color:var(--muted);
        overflow-wrap:anywhere
      }
      h3 {
        font-size:13px;
        margin:8px 0
      }
      button:focus-visible,summary:focus-visible,a:focus-visible {
        outline:2px solid #438853;
        outline-offset:2px
      }
    `;
    const panel = node("section", undefined, "panel"),
      header = node("header"),
      launcher = node("button", "Fact checker", "launcher");
    panel.id = "fact-checker-panel";
    launcher.setAttribute("aria-controls", panel.id);
    launcher.onclick = () => expand(ui.panel.hidden);
    panel.setAttribute("aria-label", "YouTube fact checker");
    const body = node("div", undefined, "body"),
      check = node("button", "Check what I just heard", "check");
    const dismiss = node("button", "×", "dismiss");
    dismiss.setAttribute("aria-label", "Close fact checker");
    dismiss.onclick = () => {
      expand(false);
      shortsOpen = false;
      mount();
      if (host.isConnected) launcher.focus();
    };
    panel.onkeydown = (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        dismiss.onclick();
      }
    };
    header.append(node("span", "Fact checker", "brand"), dismiss);
    check.onclick = run;
    // The claim's country is never inferred from sources or the news edition; the viewer, who knows
    // what the video is about, may declare it. It travels with each check and is shown on every card.
    const country = node("label", undefined, "country");
    const countryInput = node("input");
    countryInput.type = "text";
    countryInput.placeholder = "not set";
    countryInput.maxLength = 40;
    countryInput.setAttribute("list", "fact-checker-countries");
    countryInput.setAttribute("aria-label", "Country the claims in this video are about");
    const suggestions = node("datalist");
    suggestions.id = "fact-checker-countries";
    for (const name of ["United States", "United Kingdom", "Canada", "Australia", "Ireland", "New Zealand", "India",
                        "South Africa", "Nigeria", "Philippines", "European Union", "France", "Germany", "Spain",
                        "Italy", "Mexico", "Brazil", "Japan", "China", "Israel", "Ukraine", "Russia"])
      suggestions.append(node("option", name));
    country.append(node("span", "Claims are about"), countryInput, suggestions);
    // The speech date, when the viewer knows it, bounds the search: a debate uploaded weeks after it
    // was recorded should not be checked against pages published in between.
    const spoken = node("label", undefined, "country");
    const spokenInput = node("input");
    spokenInput.type = "date";
    spokenInput.setAttribute("aria-label", "Date the claims were spoken, if known");
    spoken.append(node("span", "Spoken on"), spokenInput);
    // The declaration is the viewer's setting, so it survives reloads: stored locally in the
    // extension, never sent anywhere but the local checker with each check.
    countryInput.onchange = () => {
      try { chrome.storage?.local?.set({ claim_country: claimCountry() }); } catch {}
    };
    try {
      chrome.storage?.local?.get("claim_country", (stored) => {
        if (typeof stored?.claim_country === "string" && !countryInput.value) countryInput.value = stored.claim_country;
      });
    } catch {}
    const status = node("p", "Ready to read recent captions.", "status");
    status.setAttribute("role", "status");
    const cards = node("div");
    const scope = node(
      "p",
      "Uses completed English captions from the preceding 30 seconds, checking up to three recent claims. Contrasts stay together. Captions may contain errors. Experimental news research reads public pages; coverage and YouTube accuracy are unmeasured. Matching passages do not establish a verdict.",
      "scope",
    );
    body.append(check, country, spoken, status, cards, scope);
    panel.append(header, body);
    shadow.append(style, launcher, panel);
    ui = { check, status, cards, panel, launcher, scope, country: countryInput, spoken: spokenInput };
    expand(false);
  }

  // The sentence-level reading's summary, when it has a direction, is what the viewer should see
  // first. It is a draft composed by rules over quoted sentences, and the badge says so; an
  // unresolved reading keeps the plain "No verdict".
  const READING_BADGE = {
    partially_established: ["Partially established (reading)", "reading-for"],
    partially_contradicted: ["Partially contradicted (reading)", "reading-against"],
    qualified_support: ["Supported, not established (reading)", "reading-for"],
    qualified_contradiction: ["Contradicted, not established (reading)", "reading-against"],
    contested: ["Contested (reading)", "reading-mixed"],
  };
  function outcome(result) {
    if (result.status === "skipped") return ["Not selected", "skipped"];
    if (result.outcome !== "answered") {
      const verdict = result.decomposed?.verdict;
      if (verdict?.summary === "established")
        return verdict.relationship === "contradicted" ? ["Contradicted (reading)", "reading-against"] : ["Supported (reading)", "reading-for"];
      if (verdict && READING_BADGE[verdict.summary]) return READING_BADGE[verdict.summary];
      return ["No verdict", "declined"];
    }
    const labels = {
      supported: "Supported",
      contradicted: "Contradicted",
      not_enough_evidence: "Not enough evidence",
    };
    return [
      labels[result.verdict] || "No verdict",
      result.verdict || "declined",
    ];
  }
  function showResult(data) {
    if (data.scope) ui.scope.textContent = data.scope;
    ui.status.textContent = `${data.title}\nCaption excerpt ${clock(data.start)}–${clock(data.end)} · clicked at ${clock(data.clicked_at)}\n${data.source === "visible" ? "Observed on-screen captions; timing is approximate." : "YouTube captions · English"}${data.claim_country ? `\nClaims read as about ${data.claim_country} (supplied by you).` : "\nClaim country not set; nothing can be established without it."}`;
    const transcript = node("details");
    transcript.append(
      node("summary", "Captured caption excerpt"),
      node("p", data.excerpt, "excerpt"),
    );
    ui.cards.append(transcript);
    for (const row of data.rows) {
      const result = row.result,
        [label, cls] = outcome(result),
        card = node("article");
      card.append(
        node("span", label, "badge " + cls),
        node("p", row.text, "claim"),
      );
      const reasons = {
        declined_web_review:
          "The verifier gives no verdict. The sentence-level reading below composes one from quoted sentences by fixed rules; it is a draft.",
        declined_no_relevant_evidence:
          "No retrieved excerpts passed the topic match check. The verdict model was not run.",
        declined_compound_claim:
          "This claim contrasts two assertions. The verifier cannot yet assess both together. Retrieved excerpts are background, not a verdict on the claim.",
        declined_low_confidence:
          "The model did not meet the confidence threshold.",
        declined_insufficient_evidence:
          "The evidence did not pass the sufficiency gate.",
        declined_both: "Both confidence and evidence checks fell short.",
      };
      if (reasons[result.outcome])
        card.append(node("p", reasons[result.outcome], "reason"));
      if (result.status === "skipped")
        card.append(
          node(
            "p",
            "The detector did not select this sentence; evidence retrieval was not run.",
            "reason",
          ),
        );
      else if (result.research) {
        if (result.decomposed) showDecomposed(card, result.decomposed);
        for (const note of result.notes || []) card.append(node("p", note, "reason"));
        showResearch(card, result.research);
      } else {
        const details = node("details");
        details.append(node("summary", "Inspect evidence"));
        for (const [title, , text] of result.evidence || []) {
          let decoded = title.replaceAll("_", " ");
          for (const [code, value] of [
            ["-LRB-", "("],
            ["-RRB-", ")"],
            ["-LSB-", "["],
            ["-RSB-", "]"],
            ["-COLON-", ":"],
          ])
            decoded = decoded.replaceAll(code, value);
          const link = node("a", decoded);
          link.href =
            "https://en.wikipedia.org/wiki/" +
            encodeURIComponent(decoded.replaceAll(" ", "_"));
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          const evidence = node("div", undefined, "evidence");
          evidence.append(link, node("p", text));
          details.append(evidence);
        }
        if (!result.evidence?.length)
          details.append(node("p", "No evidence was returned.", "reason"));
        details.append(
          node(
            "p",
            "Excerpts are from June 2017; linked pages may have changed.",
            "reason",
          ),
        );
        card.append(details);
      }
      ui.cards.append(card);
    }
    if (data.omitted_claims)
      ui.cards.append(
        node(
          "p",
          `${data.omitted_claims} earlier passages in the excerpt were not checked.`,
          "reason",
        ),
      );
  }

  const DRAFT_STATUS = {
    supported: "supported", contradicted: "contradicted",
    qualified_support: "supported but not established",
    qualified_contradiction: "contradicted but not established",
    contested: "contested by the sources", insufficient: "not established by the sources",
  };
  const DRAFT_SUMMARY = {
    established: "Established", partially_established: "Partially established",
    partially_contradicted: "Partially contradicted", qualified_support: "Supported, not established",
    qualified_contradiction: "Contradicted, not established", contested: "Contested", unresolved: "Not established",
  };
  // Mirrors summary_text in src/verdict/span_render.py: a qualified direction on a contrast names the part.
  function summaryText(verdict) {
    const text = DRAFT_SUMMARY[verdict.summary] || verdict.summary;
    if ((verdict.summary === "qualified_support" || verdict.summary === "qualified_contradiction") && (verdict.assertions || []).length > 1)
      return "One part " + text[0].toLowerCase() + text.slice(1);
    return text;
  }
  const MAX_RELEVANT = 6;

  function publisher(url) {
    try {
      const host = new URL(url).hostname || "";
      return host.startsWith("www.") ? host.slice(4) : host || "unnamed source";
    } catch { return "unnamed source"; }
  }

  // A quoted sentence as a reader sees it: markdown link labels without their destinations and no
  // bare URLs. The record keeps the exact span; this mirrors display_text in src/verdict/span_render.py.
  function displayText(text) {
    return String(text ?? "").replace(/\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/https?:\/\/\S+/g, "");
  }

  const READ_AS = { states: "read as stating it", states_negation: "read as denying it" };
  function draftRow(row) {
    const entry = node("div", undefined, "evidence");
    const when = row.published_at ? `published ${row.published_at}` : "publication date unconfirmed";
    // A counted sentence says which way it was read; a contested assertion is unreadable otherwise.
    const direction = READ_AS[row.relation] ? ` [${READ_AS[row.relation]}]` : "";
    entry.append(node("p", `${publisher(row.url)} (${when})${direction}: “${displayText(row.text)}”`));
    if (row.qualifiers?.length)
      entry.append(node("p", `Scope stated in the sentence: ${row.qualifiers.join("; ")}.`, "reason"));
    for (const definition of row.definitions || [])
      entry.append(node("p", `Definition in the same paragraph: “${displayText(definition.text)}”`, "reason"));
    if (row.period?.length)
      entry.append(node("p", `Dated to ${row.period.join(", ")}, outside the claim's stated period; shown, not counted.`, "reason"));
    if (row.origin === "context")
      entry.append(node("p", "Found by caption-concept research, which does not resolve the claim.", "reason"));
    else if (row.not_counted && !row.period?.length)
      entry.append(node("p", `Shown, not counted: ${row.not_counted}.`, "reason"));
    return entry;
  }

  // The sentence-level draft: verbatim sentences the gate let through, judged one at a time and
  // composed by fixed rules. Everything shown is a quoted sentence or a source record; the
  // judge's reading is not a verified verdict and the card says so.
  function showDecomposed(card, draft) {
    const verdict = draft.verdict;
    card.append(node("h3", "Sentence-level reading (draft)"));
    card.append(node("p", `${summaryText(verdict)}. ${draft.status}`, "reason"));
    for (const limit of verdict.scope_limits || []) card.append(node("p", limit, "reason"));
    for (const note of verdict.scope_notes || []) card.append(node("p", note, "reason"));
    if (verdict.judge_note) card.append(node("p", verdict.judge_note, "reason"));

    // A sentence quoted under one half of a contrast is not quoted again under the next; the
    // verdict data keeps every row, the card counts the repeats.
    const shown = new Set();
    for (const assertion of verdict.assertions || []) {
      const limits = assertion.limits?.length ? `: ${assertion.limits.join("; ")}` : "";
      // An established assertion names the dated source it rests on, so one source is never read as many.
      const basis = assertion.basis ? ` (${assertion.basis})` : "";
      card.append(node("p", `“${assertion.text}” is ${DRAFT_STATUS[assertion.status] || assertion.status}${basis}${limits}.`));
      const fresh = (assertion.relevant || []).filter(row => !shown.has(row.unit_id)).slice(0, MAX_RELEVANT);
      if (assertion.evidence?.length) {
        card.append(node("p", "Counted:", "reason"));
        for (const row of assertion.evidence) { card.append(draftRow(row)); shown.add(row.unit_id); }
      }
      if (fresh.length) card.append(node("p", "Relevant, not counted:", "reason"));
      if (assertion.relevant?.length) {
        const repeated = assertion.relevant.filter(row => shown.has(row.unit_id)).length;
        for (const row of fresh) { card.append(draftRow(row)); shown.add(row.unit_id); }
        if (repeated) card.append(node("p", `${repeated} relevant sentence${repeated === 1 ? "" : "s"} already shown above.`, "reason"));
        const more = assertion.relevant.length - repeated - fresh.length;
        if (more > 0) card.append(node("p", `… and ${more} more relevant sentences not shown.`, "reason"));
      }
      if (!assertion.evidence?.length && !assertion.relevant?.length)
        card.append(node("p", `No eligible sentence addresses it (${assertion.judged} of ${assertion.eligible} judged).`, "reason"));
    }
    const withheld = Object.entries(verdict.withheld || {});
    if (withheld.length)
      card.append(node("p", `Read as context, never as evidence: ${withheld.map(([reason, count]) => `${count} ${reason}`).join(", ")}.`, "reason"));
  }

  function showResearch(card, research) {
    if (research.search_cutoff)
      card.append(node("p", `Source search boundary: ${research.search_cutoff}. ${research.cutoff_basis}.`, "reason"));
    card.append(node("h3", "What needs checking"));
    for (const assertion of research.assertions || []) {
      card.append(node("p", assertion.text), node("p", assertion.needs, "reason"));
      card.append(node("p", assertion.source_ids?.length
        ? `Candidate sources: ${assertion.source_ids.join(", ")}. Assertion remains unresolved.`
        : "No usable candidate excerpts found for this assertion.", "reason"));
    }
    for (const gap of research.gaps || []) card.append(node("p", gap, "reason"));
    for (const error of research.errors || []) card.append(node("p", error, "reason"));
    const details = node("details");
    details.append(node("summary", "Inspect web sources"));
    for (const source of research.sources || []) {
      let url;
      try { url = new URL(source.url); } catch { continue; }
      if (url.protocol !== "https:" || url.username || url.password) continue;
      const entry = node("div", undefined, "evidence");
      const link = node("a", `${source.id}: ${source.title}`);
      link.href = url.href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      entry.append(link, node("p", source.role, "reason"));
      if (source.source_context)
        entry.append(node("p", source.source_context, "excerpt"));
      entry.append(node("p", `Published: ${source.published_at || "unknown"} · Retrieved: ${source.retrieved_at}`, "reason"));
      if (source.feed_published_at)
        entry.append(node("p", `News feed date: ${source.feed_published_at}. Reported by the feed; not confirmed publication or event time.`, "reason"));
      entry.append(node("p", source.temporal_note, "reason"));
      if (source.archive_url)
        entry.append(node("p", `Found in publisher archive · listed publication: ${source.listed_publication}.`, "reason"));
      if (source.temporal_status === "later_publication")
        entry.append(node("p", "Later publication: excluded from the assertion’s candidate sources.", "reason"));
      for (const excerpt of source.excerpts || []) entry.append(node("p", displayText(excerpt)));
      details.append(entry);
    }
    const search = node("details");
    search.append(node("summary", "Search context"));
    for (const assertion of research.assertions || []) search.append(node("p", assertion.query, "excerpt"));
    if (research.context) search.append(node("p", research.context, "excerpt"));
    details.append(search);
    card.append(details);
  }

  // A short place name or nothing; anything else is sent as unset rather than rejected mid-check.
  function claimCountry() {
    const value = (ui?.country?.value || "").trim().replace(/\s+/g, " ");
    return /^[A-Za-z][A-Za-z .'-]{1,39}$/.test(value) ? value : "";
  }

  function spokenAt() {
    const value = (ui?.spoken?.value || "").trim();
    return /^\d{4}-\d{2}-\d{2}$/.test(value) ? value : "";
  }

  async function run() {
    expand(true);
    if (busy) return;
    if (currentId() !== videoId) {
      reset();
      videoId = currentId();
    }
    busy = true;
    ui.check.disabled = true;
    ui.cards.replaceChildren();
    ui.status.textContent =
      "Reading recent captions and checking… The first check loads the models. Research and reading take up to a minute.";
    const requestedId = currentId(),
      requestVersion = ++version;
    try {
      const response = await chrome.runtime.sendMessage({
        type: "check-captions",
        video_id: requestedId,
        visible: [...visible],
        claim_country: claimCountry(),
        spoken_at: spokenAt(),
      });
      if (requestVersion !== version || currentId() !== requestedId) return;
      if (response.error) throw new Error(response.error);
      if (response.result.video_id !== requestedId)
        throw new Error("The video changed. Check again.");
      showResult(response.result);
    } catch (error) {
      if (requestVersion === version) ui.status.textContent = error.message;
    } finally {
      if (requestVersion === version) {
        busy = false;
        ui.check.disabled = false;
      }
    }
  }
  chrome.runtime.onMessage.addListener((message) => {
    if (message.type !== "open-checker") return;
    if (currentId() !== videoId) {
      reset();
      videoId = currentId();
    }
    shortsOpen = true;
    mount();
    if (!host?.isConnected) return;
    run();
  });
})();
