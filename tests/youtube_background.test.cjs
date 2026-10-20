const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const code = readFileSync(join(__dirname, "../extension/youtube/background.js"), "utf8")
  .replace(/^import .*;\r?\n/gm, "");
const contextCode = readFileSync(join(__dirname, "../extension/youtube/context.js"), "utf8").replaceAll("export function", "function");
const flush = () => new Promise((resolve) => setImmediate(resolve));

function event() {
  const listeners = [];
  return {
    addListener: (listener) => listeners.push(listener),
    emit: (...args) => listeners.map((listener) => listener(...args)),
  };
}

function fixture({ postError = null } = {}) {
  const captures = [], ports = [], timers = new Map(), scheduled = [], sent = [];
  let sequence = 0, timerId = 0;
  const chrome = {
    action: { onClicked: event(), setBadgeText: async () => {}, setTitle: async () => {} },
    commands: { onCommand: event() },
    tabs: { sendMessage: async (tabId, message) => { sent.push([tabId, message]); } },
    runtime: {
      onMessage: event(),
      connectNative: (name) => {
        assert.equal(name, "com.factchecker.youtube");
        const port = {
          onMessage: event(), onDisconnect: event(), posts: [], disconnects: 0,
          postMessage: (message) => {
            if (postError) {
              const error = postError;
              postError = null;
              throw error;
            }
            port.posts.push(message);
          },
          disconnect: () => { port.disconnects++; port.onDisconnect.emit(); },
        };
        ports.push(port);
        return port;
      },
    },
    scripting: {
      executeScript: (options) => new Promise((resolve, reject) => captures.push({ options, resolve, reject })),
    },
  };
  vm.runInNewContext(contextCode + "\n" + code, {
    chrome, readCaptions: () => {},
    crypto: { randomUUID: () => `request-${++sequence}` },
    setTimeout: (callback, ms) => {
      const id = ++timerId;
      timers.set(id, callback);
      scheduled.push({ id, callback, ms });
      return id;
    },
    clearTimeout: (id) => timers.delete(id),
  });
  return {
    captures, ports, timers, scheduled, sent,
    command: (name, tab) => chrome.commands.onCommand.emit(name, tab),
    send(tabId = 1, type = "check-captions", fields = {}) {
      const replies = [];
      let resolve;
      const response = new Promise((done) => { resolve = done; });
      const accepted = chrome.runtime.onMessage.emit(
        { type, video_id: "abcdefghijk", visible: [], ...fields },
        { tab: { id: tabId }, frameId: 0, url: "https://www.youtube.com/watch?v=abcdefghijk" },
        (reply) => { replies.push(reply); resolve(reply); },
      );
      return { response, replies, accepted };
    },
    captured(index, text = "A completed claim.", fields = {}) {
      captures[index].resolve([{ result: {
        video_id: "abcdefghijk", captions: [{ start: 1, end: 2, text }], ...fields,
      } }]);
    },
    dropped(port, message) {
      chrome.runtime.lastError = { message };
      port.onDisconnect.emit();
      delete chrome.runtime.lastError;
    },
  };
}

test("a new same-tab check rejects old caption capture before it starts native work", async () => {
  const app = fixture();
  const old = app.send(), current = app.send();
  assert.match((await old.response).error, /newer check/);
  app.captured(0, "Old claim.");
  await flush();
  assert.equal(app.ports.length, 0);
  app.captured(1, "Current claim.");
  await flush();
  assert.equal(app.ports.length, 1);
  const [port] = app.ports, [request] = port.posts;
  assert.equal(request.payload.captions[0].text, "Current claim.");
  port.onMessage.emit({ id: request.id, result: { checked: true } });
  assert.equal((await current.response).result.checked, true);
  assert.equal(old.replies.length, 1);
  assert.equal(app.timers.size, 0);
});

test("automatic context and explicit corrections reach the native host with their basis", async () => {
  for (const [correction,country,date,basis] of [
    [{},"Canada","2025-07-15","detected from captions"],
    [{claim_country:"France",spoken_at:"2025-07-14"},"France","2025-07-14","supplied by the viewer"],
  ]) {
    const app = fixture(), check = app.send(1,"check-captions",correction);
    app.captured(0,"Canada passed the law.",{source_published_at:"2025-07-20",video_metadata:{description:"Recorded on July 15, 2025."}});
    await flush();
    const [port] = app.ports, [request] = port.posts, {payload} = request;
    assert.equal(payload.claim_country,country);
    assert.equal(payload.country_basis,basis);
    assert.equal(payload.spoken_at,date);
    assert.equal(payload.spoken_at_basis,correction.spoken_at ? basis : "recording date stated in the video description or title");
    assert.equal(payload.video_metadata,undefined,"the full description is not sent to the local checker");
    port.onMessage.emit({id:request.id,result:{}});
    await check.response;
  }
});

test("late old capture failures cannot reject a newer native request", async () => {
  const app = fixture();
  const old = app.send(), current = app.send();
  await old.response;
  app.captured(1);
  await flush();
  app.captures[0].reject(new Error("Old capture failed"));
  await flush();
  assert.equal(current.replies.length, 0);
  const [port] = app.ports;
  port.onMessage.emit({ id: port.posts[0].id, result: {} });
  assert.ok((await current.response).result);
});

test("superseding native work disconnects its port and ignores every stale callback", async () => {
  const app = fixture(), old = app.send();
  app.captured(0);
  await flush();
  const previous = app.ports[0], oldTimer = app.scheduled[0];
  const current = app.send();
  assert.match((await old.response).error, /newer check/);
  assert.equal(previous.disconnects, 1);
  assert.equal(app.timers.size, 0);
  app.captured(1);
  await flush();
  const active = app.ports[1], id = active.posts[0].id;
  previous.onMessage.emit({ id, result: { stale: true } });
  app.dropped(previous, "Old pipe closed");
  oldTimer.callback();
  assert.equal(active.disconnects, 0);
  assert.equal(current.replies.length, 0);
  active.onMessage.emit({ id, result: { current: true } });
  assert.equal((await current.response).result.current, true);
  assert.equal(old.replies.length, 1);
  assert.equal(app.timers.size, 0);
});

for (const phase of ["capture", "native"]) {
  test(`another tab cannot replace the ${phase} owner`, async () => {
    const app = fixture(), owner = app.send(1);
    if (phase === "native") { app.captured(0); await flush(); }
    const other = app.send(2);
    assert.match((await other.response).error, /Another claim/);
    assert.equal(app.captures.length, 1);
    assert.equal(owner.replies.length, 0);
    if (phase === "capture") { app.captured(0); await flush(); }
    const [port] = app.ports;
    assert.equal(port.disconnects, 0);
    port.onMessage.emit({ id: port.posts[0].id, result: {} });
    await owner.response;
    const next = app.send(2);
    app.captured(1);
    await flush();
    assert.equal(app.ports.length, 1);
    port.onMessage.emit({ id: port.posts[1].id, result: {} });
    await next.response;
  });
}

test("native replies require the current request id and release ownership on errors", async () => {
  const app = fixture(), first = app.send();
  app.captured(0);
  await flush();
  const [port] = app.ports;
  port.onMessage.emit(null);
  port.onMessage.emit({ id: "unknown", result: {} });
  assert.equal(first.replies.length, 0);
  assert.equal(app.timers.size, 1);
  port.onMessage.emit({ id: port.posts[0].id, error: "Model unavailable" });
  assert.equal((await first.response).error, "Model unavailable");
  assert.equal(app.timers.size, 0);
  const second = app.send();
  app.captured(1);
  await flush();
  port.onMessage.emit({ id: port.posts[0].id, result: { stale: true } });
  assert.equal(second.replies.length, 0);
  port.onMessage.emit({ id: port.posts[1].id, result: {} });
  await second.response;
  assert.equal(port.disconnects, 0);
});

test("the unchanged 120-second timeout closes the port and permits a fresh check", async () => {
  const app = fixture(), first = app.send();
  app.captured(0);
  await flush();
  assert.equal(app.scheduled[0].ms, 120000);
  app.scheduled[0].callback();
  assert.match((await first.response).error, /timed out/);
  assert.equal(app.ports[0].disconnects, 1);
  assert.equal(app.timers.size, 0);
  const second = app.send();
  app.captured(1);
  await flush();
  const port = app.ports[1];
  port.onMessage.emit({ id: port.posts[0].id, result: {} });
  await second.response;
  assert.equal(app.timers.size, 0);
});

test("disconnect and posting failures clean up resources before retry", async () => {
  const app = fixture({ postError: new Error("Cannot post") }), first = app.send();
  app.captured(0);
  assert.match((await first.response).error, /Cannot post/);
  assert.equal(app.ports[0].disconnects, 1);
  assert.equal(app.timers.size, 0);
  const second = app.send();
  app.captured(1);
  await flush();
  app.dropped(app.ports[1], "Native pipe closed");
  assert.match((await second.response).error, /Native pipe closed/);
  assert.equal(app.timers.size, 0);
  const third = app.send();
  app.captured(2);
  await flush();
  const port = app.ports[2];
  port.onMessage.emit({ id: port.posts[0].id, result: {} });
  await third.response;
});

test("an idle-port disconnect does not fail caption capture and non-check messages do not cancel", async () => {
  const app = fixture(), first = app.send();
  app.captured(0);
  await flush();
  app.ports[0].onMessage.emit({ id: app.ports[0].posts[0].id, result: {} });
  await first.response;
  const second = app.send();
  app.dropped(app.ports[0], "Idle host closed");
  assert.equal(second.replies.length, 0);
  assert.equal(app.send(1, "close-checker").accepted[0], undefined);
  app.captured(1);
  await flush();
  assert.equal(app.ports.length, 2);
  app.ports[1].onMessage.emit({ id: app.ports[1].posts[0].id, result: {} });
  await second.response;
});

test("the keyboard shortcut checks the current video the way the toolbar icon does", async () => {
  const app = fixture();
  app.command("check-captions", { id: 7, url: "https://www.youtube.com/watch?v=abcdefghijk" });
  app.command("another-command", { id: 8, url: "https://www.youtube.com/watch?v=abcdefghijk" });
  await flush();
  assert.equal(JSON.stringify(app.sent), JSON.stringify([[7, { type: "open-checker" }]]));
});
