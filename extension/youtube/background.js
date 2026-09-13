import { readCaptions } from "./captions.js";

let port = null,
  pending = null;
function finish(request, error, result) {
  if (pending !== request) return;
  pending = null;
  clearTimeout(request.timer);
  error ? request.reject(error) : request.resolve(result);
}

function disconnect(connected) {
  if (port === connected) port = null;
  connected.disconnect();
}

function localCheck(payload, request) {
  if (!port) {
    port = chrome.runtime.connectNative("com.factchecker.youtube");
    const connected = port;
    port.onMessage.addListener((message) => {
      if (port !== connected || pending?.port !== connected || message?.id !== pending.id) return;
      finish(pending, message.error ? new Error(message.error) : null, message.result);
    });
    port.onDisconnect.addListener(() => {
      if (port !== connected) return;
      const detail = chrome.runtime.lastError?.message || "The local checker disconnected.";
      port = null;
      if (pending?.port === connected)
        finish(pending, new Error(
          detail +
            " Run .venv\\Scripts\\python.exe -m scripts.install_youtube_extension in the repository to connect the local model.",
        ));
    });
  }
  request.port = port;
  request.timer = setTimeout(() => {
    if (pending !== request) return;
    disconnect(request.port);
    finish(request, new Error("The local check timed out. Try again; the first request loads the models."));
  }, 120000);
  try {
    request.port.postMessage({ id: request.id, payload });
  } catch (error) {
    disconnect(request.port);
    finish(request, error);
  }
}

function checkCaptions(message, tabId) {
  if (pending && pending.tabId !== tabId)
    return Promise.reject(new Error("Another claim is being checked. Wait for it to finish."));
  if (pending) {
    const previous = pending;
    if (previous.port) disconnect(previous.port);
    finish(previous, new Error("A newer check replaced this request."));
  }
  return new Promise((resolve, reject) => {
    const request = { id: crypto.randomUUID(), tabId, resolve, reject };
    pending = request;
    (async () => {
      const [{ result }] = await chrome.scripting.executeScript({
        target: { tabId },
        world: "MAIN",
        func: readCaptions,
        args: [{ video_id: message.video_id, cues: message.visible || [] }],
      });
      if (pending !== request) return;
      if (result?.error) throw new Error(result.error);
      if (!result?.captions?.length)
        throw new Error(
          "No recent completed subtitles were accessible. Try again after the spoken sentence finishes; CC can stay off.",
        );
      if (typeof message.claim_country === "string" && message.claim_country) result.claim_country = message.claim_country;
      if (typeof message.spoken_at === "string" && message.spoken_at) result.spoken_at = message.spoken_at;
      localCheck(result, request);
    })().catch((error) => finish(request, error));
  });
}

chrome.action.onClicked.addListener(async (tab) => {
  if (!/^https:\/\/www\.youtube\.com\/(watch\?|shorts\/)/.test(tab.url || "")) {
    await chrome.action.setBadgeText({ tabId: tab.id, text: "YT" });
    await chrome.action.setTitle({
      tabId: tab.id,
      title: "Open a YouTube video to check its captions",
    });
    return;
  }
  await chrome.action.setBadgeText({ tabId: tab.id, text: "" });
  try {
    await chrome.tabs.sendMessage(tab.id, { type: "open-checker" });
  } catch {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["panel.js"],
    });
    await chrome.tabs.sendMessage(tab.id, { type: "open-checker" });
  }
});

chrome.runtime.onMessage.addListener((message, sender, reply) => {
  if (
    message.type !== "check-captions" ||
    !sender.tab ||
    sender.frameId !== 0 ||
    !sender.url?.startsWith("https://www.youtube.com/")
  )
    return;
  checkCaptions(message, sender.tab.id).then(
    (result) => reply({ result }),
    (error) => reply({ error: error.message }),
  );
  return true;
});
