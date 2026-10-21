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
    contextReady = null,
    contextVersion = 0,
    host = null,
    ui = null,
    blipHost = null,
    blipUi = null,
    corner = false,
    entries = [],
    folded = 0,
    failure = "",
    started = 0;
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
    entries = [];
    folded = 0;
    failure = "";
    if (ui) {
      ui.cards.replaceChildren();
      ui.status.textContent =
        "Video position changed. Check the claim you just heard.";
      ui.check.disabled = false;
    }
    renderBlips();
  }
  function expand(open) {
    ui.panel.hidden = !open;
    blipUi?.more.setAttribute("aria-expanded", String(open));
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
      if (ui) { expand(false); restoreContext(id); }
    }
    if (id !== videoId || time < lastTime - 0.5 || time > lastTime + 3) reset();
    videoId = id;
    lastTime = time;
    mount();
    place();
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
        margin:0;
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
        margin:0 0 16px
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
      .caption {
        font-size:11px;
        color:var(--muted);
        margin:6px 0 0
      }
      .verdict-line {
        margin:12px 0 4px
      }
      .counted {
        border-left-color:var(--accent)
      }
      details.related,details.how,details.skipped {
        margin-top:12px
      }
      button:focus-visible,summary:focus-visible,a:focus-visible {
        outline:2px solid #438853;
        outline-offset:2px
      }
    `;
    // The details: everything a check found, and the context corrections. It opens only when asked,
    // from a blip or the floating Details button; results arrive as blips beside the video.
    const panel = node("section", undefined, "panel"),
      header = node("header");
    panel.id = "fact-checker-panel";
    panel.setAttribute("aria-label", "YouTube fact checker");
    const body = node("div", undefined, "body"),
      check = node("button", "Check what I just heard", "check");
    const dismiss = node("button", "×", "dismiss");
    dismiss.setAttribute("aria-label", "Close fact checker");
    dismiss.onclick = () => {
      expand(false);
      shortsOpen = false;
      mount();
      blipUi?.more.focus();
    };
    panel.onkeydown = (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        dismiss.onclick();
      }
    };
    header.append(node("span", "Fact checker", "brand"), dismiss);
    check.onclick = run;
    const context = node("details");
    context.append(node("summary", "Context · automatic, with optional corrections"));
    const country = node("label", undefined, "country");
    const countryInput = node("input");
    countryInput.type = "text";
    countryInput.placeholder = "Detect automatically";
    countryInput.maxLength = 60;
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
    const saveContext = () => {
      contextVersion++;
      try { chrome.storage?.local?.set({ [`context:${currentId()}`]: { country: claimCountry(), spoken: spokenAt() } }); } catch {}
    };
    countryInput.onchange = spokenInput.onchange = saveContext;
    countryInput.oninput = spokenInput.oninput = () => { contextVersion++; };
    context.append(country, spoken, node("p", "Leave blank to detect context from this video. Corrections apply only to this video.", "scope"));
    const status = node("p", "Ready to read recent captions.", "status");
    status.setAttribute("role", "status");
    const cards = node("div");
    const scope = node(
      "p",
      "Uses completed English captions from the preceding 30 seconds, checking up to three recent claims. Contrasts stay together. Captions may contain errors. Experimental news research reads public pages; coverage and YouTube accuracy are unmeasured. Matching passages do not establish a verdict.",
      "scope",
    );
    body.append(check, context, status, cards, scope);
    panel.append(header, body);
    shadow.append(style, panel);
    ui = { check, status, cards, panel, scope, country: countryInput, spoken: spokenInput };
    restoreContext(currentId());
    expand(false);
  }

  function restoreContext(id) {
    ui.country.value = ui.spoken.value = "";
    ui.country.placeholder = "Detect automatically";
    const revision = ++contextVersion;
    let pending = true;
    const ready = new Promise(resolve => {
      const done = stored => {
        if (currentId() === id && contextVersion === revision) {
          const saved = stored?.[`context:${id}`];
          if (typeof saved?.country === "string") ui.country.value = saved.country;
          if (typeof saved?.spoken === "string") ui.spoken.value = saved.spoken;
        }
        pending = false;
        resolve();
      };
      try {
        if (chrome.storage?.local) chrome.storage.local.get(`context:${id}`, done);
        else done();
      } catch { done(); }
    });
    contextReady = pending ? ready : null;
  }

  // Results arrive as blips beside the video rather than in a sidebar. Nothing covers the picture
  // while there is room elsewhere: in normal mode the stack sits beside the player, in theater mode in
  // the black bar beside the video, and otherwise below the player when it fits above the Check button.
  // When nothing fits -- fullscreen, or a theater player whose bars are too narrow and whose bottom is
  // near the screen's -- the stack sits over the player's corner as see-through chips. A directional
  // reading opens by itself in theater and fullscreen (over the corner it folds back after 15 s);
  // "not established" shows for 6 s, then becomes a count.
  const OPEN_FOR_MS = 15000,
    QUIET_FOR_MS = 6000;
  const BLIP_CSS = `
    :host{all:initial;position:fixed;top:0;left:0;width:0;height:0;z-index:2147483647;
      --font:Roboto,'Segoe UI',Arial,sans-serif;font:13px/1.4 var(--font);color:#f1f1f1}
    *{box-sizing:border-box}
    [hidden]{display:none!important}
    p{margin:0}
    .stack{position:fixed;display:flex;flex-direction:column;gap:8px;pointer-events:none;overflow-x:hidden;overflow-y:auto}
    .stack>*{pointer-events:auto}
    .blip,.card{position:relative;background:#212121;border:1px solid #3a3a3a;border-radius:12px;
      box-shadow:0 8px 24px rgba(0,0,0,.45)}
    .blip{display:flex;align-items:flex-start;gap:10px;padding:10px 4px 10px 12px}
    .card{display:flex;flex-direction:column;gap:10px;padding:12px 14px 14px}
    .against{border-color:#5a4330}
    .quiet{background:#1a1a1a;border-color:#303030;align-items:center;padding:8px 12px}
    .mark{flex:0 0 24px;height:24px;border-radius:12px;display:flex;align-items:center;justify-content:center;
      font:700 13px/1 var(--font);color:#0f0f0f}
    .for .mark{background:#3ea6ff}.against .mark{background:#ffab5c}.mixed .mark{background:#e3cf73}
    .quiet .mark{flex-basis:18px;height:18px;border:2px solid #aaa;color:#aaa;font-size:10px}
    .body{flex:1;min-width:0;display:flex;flex-direction:column;gap:3px;text-align:left;background:none;border:0;
      padding:0;color:inherit;font:inherit;cursor:pointer}
    .label{font:500 12px/1.4 var(--font)}
    .for .label{color:#3ea6ff}.against .label{color:#ffab5c}.mixed .label{color:#e3cf73}.quiet .label{color:#ccc}
    .said{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;overflow-wrap:anywhere}
    .quiet .said{-webkit-line-clamp:1;color:#aaa;font-size:12px}
    .note{font-size:12px;color:#aaa;overflow-wrap:anywhere}
    .caution{font-size:12px;color:#ffcf9f}
    .close,.collapse{flex:0 0 32px;height:32px;border:0;background:none;color:#aaa;border-radius:16px;
      font:18px/1 var(--font);cursor:pointer}
    .checking{align-items:center}
    .checking .wait{flex:1}
    .spin{flex:0 0 18px;width:18px;height:18px;border-radius:50%;border:3px solid #3a3a3a;border-top-color:#f1f1f1;
      animation:spin 1s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    @media (prefers-reduced-motion:reduce){.spin{animation:none}}
    .text-button{border:0;background:none;color:#aaa;font:500 12px/1 var(--font);height:32px;padding:0 8px;
      border-radius:16px;cursor:pointer}
    .retry{color:#3ea6ff}
    .count{align-self:flex-end;display:flex;align-items:center;gap:6px;background:#212121;border:1px solid #3a3a3a;
      border-radius:14px;padding:4px 10px;color:#aaa;font:12px/1.4 var(--font);cursor:pointer}
    .count::before{content:"";width:8px;height:8px;border-radius:4px;border:1.5px solid #aaa;box-sizing:border-box}
    .head{display:flex;align-items:center;gap:10px}
    .head .label{flex:1;min-width:0;font-size:14px;overflow-wrap:anywhere}
    .kicker{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#aaa}
    .said-full{font-size:15px;line-height:1.45;overflow-wrap:anywhere}
    .source{background:#2c2c2c;border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:6px}
    .quote{font-size:14px;line-height:1.5;overflow-wrap:anywhere}
    .actions{display:flex;flex-wrap:wrap;gap:8px}
    a{color:#3ea6ff}
    .primary{display:inline-flex;align-items:center;height:36px;padding:0 14px;border-radius:18px;background:#f1f1f1;
      color:#0f0f0f;font:500 13px/1 var(--font);text-decoration:none}
    .secondary{height:36px;padding:0 14px;border-radius:18px;border:0;background:#303030;color:#f1f1f1;
      font:500 13px/1 var(--font);cursor:pointer}
    .controls{position:fixed;right:24px;bottom:24px;display:flex;gap:6px}
    .trigger,.more{height:44px;border:1px solid #3a3a3a;background:#212121;color:#f1f1f1;border-radius:22px;
      padding:0 16px;font:500 13px/1 var(--font);box-shadow:0 6px 18px rgba(0,0,0,.45);cursor:pointer}
    .more{width:44px;padding:0;font-size:18px}
    .trigger:disabled{opacity:.6;cursor:default}
    button:focus-visible,a:focus-visible{outline:2px solid #3ea6ff;outline-offset:2px}
    :host([data-narrow="true"]) .blip{flex-wrap:wrap;gap:6px;padding:8px 4px 8px 9px}
    :host([data-narrow="true"]) .blip .body{order:3;flex-basis:100%;padding-right:5px}
    :host([data-narrow="true"]) .blip .close{margin-left:auto}
    :host([data-narrow="true"]) .card{padding:10px;gap:8px}
    :host([data-narrow="true"]) .collapse,:host([data-narrow="true"]) .record,
    :host([data-narrow="true"]) .basis.single{display:none}
    :host([data-narrow="true"]) .mark{flex-basis:20px;height:20px;font-size:11px}
    :host([data-narrow="true"]) .close{flex-basis:28px;height:28px}
    :host([data-narrow="true"]) .said-full{display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
    :host([data-narrow="true"]) .actions>*{height:32px;justify-content:center}
    :host([data-narrow="true"]) .head{gap:6px}
    :host([data-narrow="true"]) .head .mark{display:none}
    :host([data-narrow="true"]) .head .label{font-size:11px;overflow-wrap:break-word}
    :host([data-narrow="true"]) .said-full,:host([data-narrow="true"]) .quote{font-size:12px}
    :host([data-narrow="true"]) .source{padding:0;background:none}
    :host([data-narrow="true"]) .actions{flex-direction:column}
    :host([data-mode="fullscreen"]) .controls,:host([data-mode="fullscreen"]) .secondary{display:none}
    :host([data-corner="true"]) .blip{align-self:flex-end;align-items:center;border:0;border-radius:18px;
      background:rgba(28,28,28,.82);box-shadow:none;padding:6px 4px 6px 6px}
    :host([data-corner="true"]) .blip .said{display:none}
    :host([data-corner="true"]) .card{border:0;background:rgba(28,28,28,.94)}
    :host([data-corner="true"]) .count{border:0;background:rgba(28,28,28,.82)}
  `;

  function viewMode() {
    if (document.fullscreenElement) return "fullscreen";
    if (document.querySelector("ytd-watch-flexy")?.hasAttribute?.("theater")) return "theater";
    return "normal";
  }

  function buildBlips() {
    blipHost = document.createElement("div");
    blipHost.id = "youtube-fact-checker-blips";
    const shadow = blipHost.attachShadow({ mode: "closed" });
    const style = node("style");
    style.textContent = BLIP_CSS;
    const stack = node("div", undefined, "stack");
    stack.setAttribute("aria-live", "polite");
    stack.setAttribute("aria-label", "Fact checks");
    const controls = node("div", undefined, "controls");
    const trigger = node("button", "Check · Alt+Shift+F", "trigger");
    trigger.setAttribute("aria-label", "Check what I just heard (Alt+Shift+F)");
    trigger.onclick = run;
    const more = node("button", "⋯", "more");
    more.setAttribute("aria-label", "Fact checker details and context");
    more.setAttribute("aria-expanded", "false");
    more.onclick = () => (ui && !ui.panel.hidden ? closeDetails() : openDetails());
    controls.append(trigger, more);
    shadow.append(style, stack, controls);
    blipUi = { stack, trigger, more, wait: null };
  }

  // The stack is placed from the player's box on screen, never by moving or restyling YouTube.
  function place() {
    const watching = (location.pathname === "/watch" || location.pathname.startsWith("/shorts/")) && currentId();
    if (!watching) {
      blipHost?.remove();
      return;
    }
    if (!blipHost) buildBlips();
    const parent = document.fullscreenElement || document.documentElement;
    if (blipHost.parentElement !== parent) parent.append(blipHost);
    // The visible width, without a scrollbar; a box that runs past it is clipped to it.
    const mode = viewMode(),
      width = document.documentElement.clientWidth || globalThis.innerWidth || 1280,
      height = document.documentElement.clientHeight || globalThis.innerHeight || 720;
    const player = document.getElementById("movie_player")?.getBoundingClientRect?.();
    const picture = document.querySelector("video.html5-main-video")?.getBoundingClientRect?.();
    let box = mode === "fullscreen" ? { top: 20, right: 20, width: 344, corner: true } : { top: 72, right: 24, width: 344 };
    if (mode !== "fullscreen" && player?.width && picture?.width) {
      const edge = Math.min(player.right, width),
        bar = edge - picture.right,
        // Room below the player that the fixed Check button (24 px margin, 44 px tall) leaves free.
        below = height - (player.bottom + 12) - 80;
      if (mode === "theater" && bar >= 140)
        box = { top: player.top + 12, right: width - edge + 8, width: Math.min(344, bar - 16),
                maxHeight: player.height - 72 };
      else if ((mode === "theater" || width - player.right < 368) && below >= 120)
        box = { top: player.bottom + 12, right: 24, width: 344, maxHeight: below };
      else if (mode === "theater" || width - player.right < 368)
        box = { top: player.top + 12, right: width - edge + 12, width: 344, maxHeight: player.height - 72,
                corner: true };
    }
    corner = Boolean(box.corner);
    blipHost.setAttribute("data-mode", mode);
    blipHost.setAttribute("data-corner", String(corner));
    blipHost.setAttribute("data-narrow", String(box.width < 240));
    Object.assign(blipUi.stack.style, {
      top: `${Math.max(8, Math.round(box.top))}px`, right: `${Math.round(box.right)}px`,
      width: `${Math.round(box.width)}px`, maxHeight: box.maxHeight ? `${Math.round(box.maxHeight)}px` : "",
    });
  }

  // Which way a reading points, for the blip's colour and whether it speaks up. A qualified reading
  // ("supported, not established") is a lead, so it stays as quiet as no reading at all.
  function blipKind(result) {
    if (result.outcome === "answered") return { supported: "for", contradicted: "against" }[result.verdict] || "quiet";
    const verdict = result.decomposed?.verdict;
    if (verdict?.summary === "established") return verdict.relationship === "contradicted" ? "against" : "for";
    return { partially_established: "for", partially_contradicted: "against", contested: "mixed" }[verdict?.summary] || "quiet";
  }

  function entryFor(row) {
    const result = row.result,
      verdict = result.decomposed?.verdict,
      assertions = verdict?.assertions || [],
      kind = blipKind(result);
    // The quotation a direction rests on: the first counted sentence of an assertion that has one.
    const assertion = kind === "quiet" ? null
      : assertions.find((item) => /^(supported|contradicted)$/.test(item.status) && item.evidence?.length) ||
        assertions.find((item) => item.evidence?.length) || null;
    return {
      text: row.text, kind, label: outcome(result)[0], assertion, quote: assertion?.evidence?.[0] || null,
      basis: assertion?.basis || "", parts: assertions.length, judgeNote: verdict?.judge_note || "",
      // Every wrong reading on the held-out batches was one source contradicting a true claim.
      caution: kind === "against" && /^one source/.test(assertion?.basis || "")
        ? "One source only. Read it before you repeat this." : "",
      open: false, folded: false, dismissed: false,
    };
  }

  const MARK = { for: "✓", against: "✕", mixed: "!", quiet: "–" };
  function closeButton(entry) {
    const close = node("button", "×", "close");
    close.setAttribute("aria-label", "Dismiss");
    close.onclick = () => {
      entry.dismissed = true;
      renderBlips();
    };
    return close;
  }

  function blipFor(entry) {
    const item = node("div", undefined, `blip ${entry.kind}`);
    item.setAttribute("role", "status");
    const mark = node("span", MARK[entry.kind], "mark");
    mark.setAttribute("aria-hidden", "true");
    const body = node("button", undefined, "body");
    body.append(node("span", entry.quote ? `${entry.label} · ${publisher(entry.quote.url)}` : entry.label, "label"));
    if (entry.text) body.append(node("span", `“${entry.text}”`, "said"));
    if (entry.caution) body.append(node("span", entry.caution, "note"));
    body.onclick = () => {
      if (entry.kind === "quiet") return openDetails();
      for (const other of entries) other.open = false;
      entry.open = true;
      renderBlips();
    };
    item.append(mark, body);
    if (entry.kind !== "quiet") item.append(closeButton(entry));
    return item;
  }

  function cardFor(entry) {
    const card = node("div", undefined, `card ${entry.kind}`);
    card.setAttribute("role", "dialog");
    card.setAttribute("aria-label", `Fact check: ${entry.label}`);
    const head = node("div", undefined, "head"),
      mark = node("span", MARK[entry.kind], "mark"),
      collapse = node("button", "⌃", "collapse");
    mark.setAttribute("aria-hidden", "true");
    collapse.setAttribute("aria-label", "Collapse");
    collapse.onclick = () => {
      entry.open = false;
      renderBlips();
    };
    head.append(mark, node("span", entry.label, "label"), collapse, closeButton(entry));
    card.append(head);
    // A single source's basis repeats the source line below, so the narrow bar can leave it out.
    if (entry.basis) card.append(node("p", entry.basis, /^one source/.test(entry.basis) ? "note basis single" : "note basis"));
    card.append(node("p", "Just heard", "kicker"), node("p", entry.text, "said-full"));
    // A contrast is read part by part; the quotation answers this part, not the whole sentence.
    if (entry.parts > 1 && entry.assertion) card.append(node("p", `About “${entry.assertion.text}”`, "note"));
    // The judge's record says the sentences are "quoted below", so it precedes the quotation.
    if (entry.judgeNote) card.append(node("p", entry.judgeNote, "note record"));
    const quote = entry.quote;
    if (quote) {
      const source = node("div", undefined, "source"),
        line = node("p", undefined, "note");
      const read = [READ_AS[quote.relation], quote.relation === "states" && quote.nli ? "confirmed by a second model" : ""]
        .filter(Boolean).join("; ");
      line.append(sourceLink(quote.url, publisher(quote.url)) || node("span", publisher(quote.url)),
                  node("span", ` · ${quote.published_at ? `published ${quote.published_at}` : "publication date unconfirmed"}${read ? ` · ${read}` : ""}`));
      source.append(node("p", "The source says", "kicker"), node("p", `“${displayText(quote.text)}”`, "quote"), line);
      card.append(source);
    }
    if (entry.caution) card.append(node("p", entry.caution, "caution"));
    card.append(node("p", "A draft reading by fixed rules, not a verified verdict.", "note"));
    const actions = node("div", undefined, "actions"),
      link = quote && sourceLink(quote.url, "Read the source"),
      details = node("button", "Details", "secondary");
    if (link) {
      link.className = "primary";
      actions.append(link);
    }
    details.onclick = openDetails;
    actions.append(details);
    card.append(actions);
    return card;
  }

  function renderBlips() {
    if (!blipUi) return;
    const items = [];
    blipUi.wait = null;
    if (busy) {
      const item = node("div", undefined, "blip checking"),
        spin = node("span", undefined, "spin"),
        stop = node("button", "Stop", "text-button");
      item.setAttribute("role", "status");
      spin.setAttribute("aria-hidden", "true");
      blipUi.wait = node("span", `Checking what you just heard · ${Math.round((Date.now() - started) / 1000)} s`, "wait");
      // Stops waiting; the local check may still finish, and its reply is ignored.
      stop.onclick = stopWaiting;
      item.append(spin, blipUi.wait, stop);
      items.push(item);
    }
    if (failure) {
      const item = node("div", undefined, "blip quiet failure"),
        retry = node("button", "Retry", "text-button retry");
      item.setAttribute("role", "alert");
      retry.onclick = run;
      item.append(node("span", failure, "note"), retry);
      items.push(item);
    }
    for (const entry of entries)
      if (!entry.dismissed && !entry.folded) items.push(entry.open ? cardFor(entry) : blipFor(entry));
    if (folded) {
      const count = node("button", `${folded} not established`, "count");
      count.setAttribute("aria-label", `${folded} not established. Show details`);
      count.onclick = openDetails;
      items.push(count);
    }
    blipUi.stack.replaceChildren(...items);
    blipUi.trigger.disabled = busy;
  }

  function showBlips(checked) {
    entries = checked.map(entryFor);
    folded = 0;
    failure = "";
    if (!entries.length)
      entries = [{ text: "", kind: "quiet", label: "No checkable claim in the last 30 seconds of captions.", nothing: true,
                   open: false, folded: false, dismissed: false }];
    const lead = entries.find((entry) => entry.kind !== "quiet"),
      mode = viewMode();
    if (lead && mode !== "normal") {
      lead.open = true;
      if (corner)
        setTimeout(() => {
          if (!entries.includes(lead)) return;
          lead.open = false;
          renderBlips();
        }, OPEN_FOR_MS);
    }
    for (const entry of entries)
      if (entry.kind === "quiet")
        setTimeout(() => {
          if (!entries.includes(entry) || entry.folded) return;
          entry.folded = true;
          if (!entry.nothing) folded++;
          renderBlips();
        }, QUIET_FOR_MS);
    renderBlips();
  }

  function openDetails() {
    if (!host) build();
    shortsOpen = true;
    mount();
    expand(true);
  }

  function closeDetails() {
    expand(false);
    shortsOpen = false;
    mount();
  }

  function stopWaiting() {
    version++;
    busy = false;
    if (ui) {
      ui.check.disabled = false;
      ui.status.textContent = "Stopped waiting for that check.";
    }
    renderBlips();
  }

  // The sentence-level reading's summary is what the viewer should see first. It is a draft
  // composed by rules over quoted sentences: the dashed badge and the caption under the claim say
  // so. Without a reading (the judge is not installed) the plain "No verdict" stays.
  const READING_BADGE = {
    partially_established: ["Partially established", "reading-for"],
    partially_contradicted: ["Partially contradicted", "reading-against"],
    qualified_support: ["Supported, not established", "reading-for"],
    qualified_contradiction: ["Contradicted, not established", "reading-against"],
    contested: ["Contested", "reading-mixed"],
  };
  function outcome(result) {
    if (result.status === "skipped") return ["Not selected", "skipped"];
    if (result.outcome !== "answered") {
      const verdict = result.decomposed?.verdict;
      if (verdict?.summary === "established")
        return verdict.relationship === "contradicted" ? ["Contradicted", "reading-against"] : ["Supported", "reading-for"];
      if (verdict && READING_BADGE[verdict.summary]) return READING_BADGE[verdict.summary];
      if (verdict) return ["Not established", "declined"];
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
    const countryBasis = data.country_basis && data.country_basis !== "supplied by the viewer" ? data.country_basis : "supplied by you";
    ui.country.placeholder = data.claim_country ? `Auto: ${data.claim_country}` : "Country unclear · optional correction";
    ui.status.textContent = `${data.title}\nCaption excerpt ${clock(data.start)}–${clock(data.end)} · clicked at ${clock(data.clicked_at)}\n${data.source === "visible" ? "Observed on-screen captions; timing is approximate." : "YouTube captions · English"}${data.claim_country ? `\nClaims read as about ${data.claim_country} (${countryBasis}).` : "\nCountry could not be detected confidently; you can correct it under Context."}\nVideo published: ${data.source_published_at || "unknown"}.\nSpoken on: ${data.spoken_at ? `${data.spoken_at} (${data.spoken_at_basis || "supplied by you"})` : "unknown; publication date is not a recording date"}.`;
    const transcript = node("details");
    transcript.append(
      node("summary", "Captured caption excerpt"),
      node("p", data.excerpt, "excerpt"),
    );
    ui.cards.append(transcript);
    // The clicked claim is the last one spoken, so cards read back from it; sentences the detector
    // passed over are listed once at the end rather than as cards of their own.
    const checked = data.rows.filter((row) => row.result.status !== "skipped").reverse();
    const skipped = data.rows.filter((row) => row.result.status === "skipped");
    // Said once for every card below; each card's dashed badge marks it as a draft too.
    if (checked.some((row) => row.result.decomposed))
      ui.cards.append(node("p", "Draft readings of quoted sources by fixed rules; not verified verdicts.", "caption"));
    for (const row of checked) {
      const result = row.result,
        [label, cls] = outcome(result),
        card = node("article");
      card.append(
        node("span", label, "badge " + cls),
        node("p", row.text, "claim"),
      );
      // News checks are read sentence by sentence below; the offline verifier's outcomes keep their reasons.
      const reasons = {
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
      if (result.research) {
        for (const note of result.notes || []) card.append(node("p", note, "reason"));
        const how = node("details", undefined, "how");
        how.append(node("summary", "How this was checked"));
        if (result.decomposed) showDecomposed(card, how, result.decomposed, row.text);
        showResearch(how, result.research, row.text);
        card.append(how);
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
    if (skipped.length) {
      const list = node("details", undefined, "skipped");
      list.append(node("summary", `${skipped.length} ${skipped.length === 1 ? "sentence" : "sentences"} not checked`));
      list.append(node("p", "The detector did not select these sentences, or they could not be checked as heard; evidence retrieval was not run.", "reason"));
      const why = { fragment: "names nothing to check", "unnamed subject": "who it is about was said earlier",
                    "reported speech": "reports what someone said", unfinished: "still being spoken" };
      for (const row of skipped) {
        const reason = why[String(row.result.filter_reason || "").split(":")[0]];
        list.append(node("p", reason ? `${row.text} (${reason})` : row.text, "excerpt"));
      }
      ui.cards.append(list);
    }
    if (data.omitted_claims)
      ui.cards.append(
        node(
          "p",
          `${data.omitted_claims} earlier passages in the excerpt were not checked.`,
          "reason",
        ),
      );
    showBlips(checked);
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

  function sourceLink(value, label) {
    let url;
    try { url = new URL(value); } catch { return null; }
    if (url.protocol !== "https:" || url.username || url.password) return null;
    const link = node("a", label);
    link.href = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  }

  // A quoted sentence as a reader sees it: markdown link labels without their destinations and no
  // bare URLs. The record keeps the exact span; this mirrors display_text in src/verdict/span_render.py.
  function displayText(text) {
    return String(text ?? "").replace(/\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/https?:\/\/\S+/g, "");
  }

  const READ_AS = { states: "read as stating it", states_negation: "read as denying it" };
  function draftRow(row, counted = false) {
    const entry = node("div", undefined, counted ? "evidence counted" : "evidence");
    const when = row.published_at ? `published ${row.published_at}` : "publication date unconfirmed";
    // A counted sentence says which way it was read; a contested assertion is unreadable otherwise.
    // A near miss the second model confirmed says so, because it is why that sentence counts.
    const confirmed = row.relation === "states" && row.nli ? "; confirmed by a second model" : "";
    const direction = READ_AS[row.relation] ? ` [${READ_AS[row.relation]}${confirmed}]` : "";
    // The publisher's name is the link to the validated source record; a bare name when there is none.
    const line = node("p");
    line.append(sourceLink(row.url, publisher(row.url)) || node("span", publisher(row.url)),
                node("span", ` (${when})${direction}: “${displayText(row.text)}”`));
    entry.append(line);
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
  // judge's reading is not a verified verdict and the card says so. The verdict lines and the
  // counted sentences stay in view; related sentences and the method fold away, nothing is dropped.
  function showDecomposed(card, how, draft, claim) {
    const verdict = draft.verdict;
    how.append(node("h3", "Sentence-level reading (draft)"));
    how.append(node("p", `${summaryText(verdict)}. ${draft.status}`, "reason"));
    for (const limit of verdict.scope_limits || []) how.append(node("p", limit, "reason"));
    for (const note of verdict.scope_notes || []) how.append(node("p", note, "reason"));

    const assertions = verdict.assertions || [];
    // The reading's record introduces the sentences a count rests on; with nothing counted it is method.
    const counted = assertions.some((assertion) => assertion.evidence?.length);
    if (verdict.judge_note) (counted ? card : how).append(node("p", verdict.judge_note, "reason"));
    for (const assertion of assertions) {
      const limits = assertion.limits?.length ? `: ${assertion.limits.join("; ")}` : "";
      // An established assertion names the dated source it rests on, so one source is never read as many.
      const basis = assertion.basis ? ` (${assertion.basis})` : "";
      const status = DRAFT_STATUS[assertion.status] || assertion.status;
      // An assertion that is the claim printed just above is not quoted again; any other part is named.
      card.append(node("p", assertions.length === 1 && assertion.text.trim() === String(claim ?? "").trim()
        ? `${status[0].toUpperCase()}${status.slice(1)}${basis}${limits}.`
        : `“${assertion.text}” is ${status}${basis}${limits}.`, "verdict-line"));
      // Rank and scope belong to this assertion; earlier quotations cannot replace its best rows.
      const selected = (assertion.relevant || []).slice(0, MAX_RELEVANT);
      if (assertion.evidence?.length) {
        card.append(node("p", "Counted:", "reason"));
        for (const row of assertion.evidence) card.append(draftRow(row, true));
      }
      if (assertion.relevant?.length) {
        const related = node("details", undefined, "related");
        related.append(node("summary", `Relevant, not counted (${assertion.relevant.length})`));
        for (const row of selected) related.append(draftRow(row));
        const more = assertion.relevant.length - selected.length;
        if (more > 0) related.append(node("p", `… and ${more} more relevant sentences not shown.`, "reason"));
        card.append(related);
      }
      if (!assertion.evidence?.length && !assertion.relevant?.length)
        card.append(node("p", `No eligible sentence addresses it (${assertion.judged} of ${assertion.eligible} judged).`, "reason"));
    }
    const withheld = Object.entries(verdict.withheld || {});
    if (withheld.length)
      how.append(node("p", `Read as context, never as evidence: ${withheld.map(([reason, count]) => `${count} ${reason}`).join(", ")}.`, "reason"));
  }

  function showResearch(card, research, claim) {
    if (research.search_cutoff)
      card.append(node("p", `Source search boundary: ${research.search_cutoff}. ${research.cutoff_basis}.`, "reason"));
    card.append(node("h3", "Web research"));
    const searched = research.assertions || [];
    for (const assertion of searched) {
      const pages = assertion.source_ids?.length || 0;
      const found = pages ? `${pages} candidate ${pages === 1 ? "page" : "pages"}` : "no usable candidate excerpts";
      // The exact queries are under Search context; this line names what was searched for.
      const solo = searched.length === 1 && assertion.text.trim() === String(claim ?? "").trim();
      card.append(node("p", solo ? `News search: ${found}.` : `“${assertion.text}”: ${found}.`, "reason"));
    }
    // Pages that failed are counted with their distinct reasons; each attempted address is listed below.
    const errors = research.errors || [];
    const reasons = [...new Set(errors.map((error) => String(error).replace(/\.$/, "")))];
    if (errors.length)
      card.append(node("p", `${errors.length} ${errors.length === 1 ? "page" : "pages"} could not be read: ${reasons.join("; ")}.`, "reason"));
    for (const gap of research.gaps || []) card.append(node("p", gap, "reason"));
    const details = node("details");
    details.append(node("summary", "Inspect web sources"));
    for (const source of research.sources || []) {
      const link = sourceLink(source.url, `${source.id}: ${source.title}`);
      if (!link) continue;
      const entry = node("div", undefined, "evidence");
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
    return /^[\p{L}][\p{L} .’'&-]{1,59}$/u.test(value) ? value : "";
  }

  function spokenAt() {
    const value = (ui?.spoken?.value || "").trim();
    return /^\d{4}-\d{2}-\d{2}$/.test(value) ? value : "";
  }

  async function run() {
    if (busy) return;
    if (!host) build();
    if (currentId() !== videoId) {
      reset();
      videoId = currentId();
      restoreContext(videoId);
    }
    busy = true;
    ui.check.disabled = true;
    ui.cards.replaceChildren();
    entries = [];
    folded = 0;
    failure = "";
    const waiting = (seconds) =>
      `Reading recent captions and checking… ${seconds} s. The first check loads the models; research and reading take up to a minute.`;
    ui.status.textContent = waiting(0);
    const requestedId = currentId(),
      requestVersion = ++version;
    started = Date.now();
    renderBlips();
    // A minute of silence reads as a hang; the counter shows the check is still running.
    const ticker = setInterval(() => {
      if (requestVersion !== version || !busy) return;
      const seconds = Math.round((Date.now() - started) / 1000);
      ui.status.textContent = waiting(seconds);
      if (blipUi?.wait) blipUi.wait.textContent = `Checking what you just heard · ${seconds} s`;
    }, 1000);
    try {
      if (contextReady) await contextReady;
      if (requestVersion !== version || currentId() !== requestedId) return;
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
      if (requestVersion === version) ui.status.textContent = failure = error.message;
    } finally {
      clearInterval(ticker);
      if (requestVersion === version) {
        busy = false;
        ui.check.disabled = false;
        renderBlips();
      }
    }
  }
  // The toolbar icon and the keyboard shortcut check what was just heard; results arrive as blips.
  chrome.runtime.onMessage.addListener((message) => {
    if (message.type !== "open-checker" || !currentId()) return;
    if (currentId() !== videoId) {
      reset();
      videoId = currentId();
      if (ui) restoreContext(videoId);
    }
    place();
    run();
  });
})();
