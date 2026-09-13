const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");
const code = readFileSync(
  join(__dirname, "../extension/youtube/captions.js"),
  "utf8",
).replace("export async", "async");

function reader({
  events = [],
  tracks = [],
  native = [],
  language = "en",
  ad = false,
  changed = false,
  fetchError = false,
  initial = undefined,
  legacy = [],
  background = null,
  onBackground = () => {},
  published = "",
  isLiveContent = false,
  liveBroadcastDetails = undefined,
  audioTracks = [],
} = {}) {
  const videoId = "abcdefghijk";
  const player = {
    classList: { contains: () => ad },
    getOption: () => ({ languageCode: language }),
    getPlayerResponse: () => ({
      videoDetails: {
        videoId: changed ? "0123456789_" : videoId,
        title: "Fixture",
        isLiveContent,
      },
      captions: { playerCaptionsTracklistRenderer: { captionTracks: tracks, audioTracks } },
      microformat: { playerMicroformatRenderer: { publishDate: published, liveBroadcastDetails } },
    }),
  };
  const requests = [];
  const posts = [];
  const clicks = [];
  const expand = { click: () => clicks.push("expand description") };
  const showTranscript = {
    textContent: "Show transcript",
    click: () => clicks.push("show transcript"),
  };
  const context = vm.createContext({
    URL,
    AbortSignal,
    setTimeout,
    atob,
    btoa,
    location: {
      href: `https://www.youtube.com/watch?v=${videoId}`,
      pathname: "/watch",
    },
    window: {
      ytInitialData: initial,
      ytcfg: {
        get: () => ({
          client: { clientName: "WEB", clientVersion: "fixture" },
        }),
      },
    },
    document: {
      getElementById: () => player,
      querySelectorAll: (selector) =>
        selector === "ytd-transcript-segment-renderer"
          ? legacy.map((cue) => ({
              querySelector: (part) => ({
                textContent:
                  part === ".segment-timestamp" ? cue.timestamp : cue.text,
              }),
            }))
          : selector === "button"
            ? [showTranscript]
            : [],
      querySelector: (selector) =>
        selector === "video.html5-main-video"
          ? { currentTime: 20, textTracks: native }
          : selector === "ytd-transcript-renderer #footer"
            ? { textContent: "English" }
            : selector === "#description-inline-expander #expand"
              ? expand
              : null,
    },
    fetch: async (url, options) => {
      requests.push(String(url));
      if (options.method === "POST") {
        posts.push({
          url: String(url),
          ...options,
          body: JSON.parse(options.body),
        });
        onBackground(context);
        return {
          ok: background !== null,
          text: async () => JSON.stringify(background),
        };
      }
      if (fetchError) throw new Error("Caption endpoint failed");
      return { ok: true, text: async () => JSON.stringify({ events }) };
    },
  });
  vm.runInContext(code, context);
  return { read: context.readCaptions, requests, posts, clicks };
}

const track = {
  baseUrl: "https://www.youtube.com/api/timedtext?v=abcdefghijk",
  languageCode: "en",
};
test("track parsing retains only completed recent cues", async () => {
  const app = reader({
    tracks: [track],
    events: [
      {
        tStartMs: 10000,
        dDurationMs: 5000,
        segs: [{ utf8: "The law passed." }],
      },
      {
        tStartMs: 18000,
        dDurationMs: 10000,
        segs: [{ utf8: "Future words." }],
      },
    ],
  });
  const result = await app.read();
  assert.equal(result.captions.length, 1);
  assert.equal(result.captions[0].text, "The law passed.");
  assert.equal(result.source, "track");
  assert.equal(result.time, 20);
});

test("publication metadata follows the video without becoming a speech date", async () => {
  for (const [published, expected] of [["2025-07-20T08:00:05-07:00", "2025-07-20"], ["2025-02-30", ""], ["yesterday", ""], ["2025-07-20Tnot-a-time", ""]]) {
    const app = reader({tracks: [track], published, events: [{tStartMs:10000,dDurationMs:5000,segs:[{utf8:"The law passed."}]}]});
    const result = await app.read();
    assert.equal(result.source_published_at, expected);
    assert.equal(result.spoken_at, undefined);
  }
});

test("capture failures explain missing metadata without exposing endpoint URLs", async () => {
  const app = reader({tracks:[track], fetchError:true});
  const result = await app.read();
  assert.match(result.error, /Caption track request failed or timed out/);
  assert.match(result.error, /Transcript metadata is not ready/);
  assert.doesNotMatch(result.error, /api\/timedtext|abcdefghijk/);
  assert.equal(app.clicks.length, 0);
});

test("live-origin publication time cannot bound speech later in a stream", async () => {
  for (const fields of [{isLiveContent:true}, {liveBroadcastDetails:{startTimestamp:"2025-07-20T12:00:00Z"}}]) {
    const app = reader({tracks:[track],published:"2025-07-20",...fields,
      events:[{tStartMs:10000,dDurationMs:5000,segs:[{utf8:"The law passed."}]}]});
    assert.equal((await app.read()).source_published_at, "");
  }
});

function modernData(language = "en", endpointId = "abcdefghijk") {
  const field = (number, value) =>
    Buffer.concat([
      Buffer.from([number * 8 + 2, Buffer.byteLength(value)]),
      Buffer.from(value),
    ]);
  const track = Buffer.concat([
    field(1, ""),
    field(2, language),
    field(3, ""),
  ]).toString("base64");
  const params = Buffer.concat([
    field(1, endpointId),
    field(2, track),
  ]).toString("base64");
  return {
    currentVideoEndpoint: { watchEndpoint: { videoId: "abcdefghijk" } },
    engagementPanels: [
      { getTranscriptEndpoint: { params } },
      ...[
        ["0:10", "The law passed in 2010."],
        ["0:15", "The next claim."],
        ["0:25", "Future."],
      ].map(([timestamp, simpleText]) => ({
        transcriptSegmentViewModel: { timestamp, simpleText },
      })),
    ],
  };
}

test("modern transcript uses video-bound language metadata and completed intervals", async () => {
  const result = await reader({ initial: modernData() }).read();
  assert.equal(result.source, "transcript");
  assert.equal(result.captions.length, 1);
  assert.equal(result.captions[0].text, "The law passed in 2010.");
  assert.equal(result.captions[0].end, 15);
  assert.match(
    (await reader({ initial: modernData("es") }).read()).error,
    /Captions could not/,
  );
  assert.match(
    (await reader({ initial: modernData("en", "0123456789_") }).read()).error,
    /Captions could not/,
  );
  const old = modernData();
  old.currentVideoEndpoint.watchEndpoint.videoId = "0123456789_";
  assert.match(
    (await reader({ initial: old }).read()).error,
    /Captions could not/,
  );
});

test("legacy transcript cannot label stale DOM as the new video", async () => {
  const initial = modernData();
  initial.currentVideoEndpoint.watchEndpoint.videoId = "0123456789_";
  const legacy = [
    { timestamp: "0:10", text: "Old video claim." },
    { timestamp: "0:15", text: "Another old claim." },
  ];
  assert.match(
    (await reader({ initial, legacy }).read()).error,
    /Captions could not/,
  );
  assert.match((await reader({ legacy }).read()).error, /Captions could not/);
});

test("request YouTube's English translation for translatable captions", async () => {
  const app = reader({
    tracks: [{ ...track, languageCode: "es", isTranslatable: true }],
    events: [
      {
        tStartMs: 10000,
        dDurationMs: 5000,
        segs: [{ utf8: "English translation." }],
      },
    ],
  });
  assert.equal((await app.read()).language, "en");
  assert.equal(new URL(app.requests[0]).searchParams.get("tlang"), "en");
});

test("use native text tracks when timedtext is unavailable", async () => {
  const app = reader({
    tracks: [track],
    fetchError: true,
    native: [
      {
        language: "en",
        cues: [{ startTime: 10, endTime: 15, text: "The law passed." }],
      },
    ],
  });
  assert.equal((await app.read()).source, "text-track");
});

test("visible-caption fallback is restricted to the same video and English", async () => {
  const observed = {
    video_id: "abcdefghijk",
    cues: [{ start: 10, end: 15, text: "The law passed." }],
  };
  assert.equal((await reader().read(observed)).source, "visible");
  assert.match(
    (await reader().read({ ...observed, video_id: "0123456789_" })).error,
    /Captions could not/,
  );
  assert.match(
    (await reader({ language: "es" }).read(observed)).error,
    /English subtitle track/,
  );
});

function unloadedPanel(videoId = "abcdefghijk") {
  const initial = modernData();
  initial.engagementPanels = initial.engagementPanels.slice(0, 1);
  const nested = Buffer.concat([
    Buffer.from([10, videoId.length]),
    Buffer.from(videoId),
    Buffer.from([24, 1]),
  ]);
  const params = Buffer.concat([
    Buffer.from([170, 9, nested.length]),
    nested,
  ]).toString("base64");
  initial.engagementPanels.push({
    updateEngagementPanelContentCommand: {
      contentSourcePanelIdentifier: { tag: "PAmodern_transcript_view" },
      globalConfiguration: { params },
    },
  });
  return initial;
}

test("read background subtitles with CC off and transcript closed without clicking native controls", async () => {
  const app = reader({
    initial: unloadedPanel(),
    language: "",
    tracks: [track],
    background: modernData(),
  });
  const result = await app.read();
  assert.equal(result.source, "transcript");
  assert.equal(result.language, "en");
  assert.equal(result.captions.length, 1);
  assert.equal(result.captions[0].text, "The law passed in 2010.");
  assert.equal(result.captions[0].end, 15);
  assert.equal(app.posts.length, 1);
  assert.equal(app.posts[0].url, "/youtubei/v1/get_panel?prettyPrint=false");
  assert.equal(app.posts[0].body.panelId, "PAmodern_transcript_view");
  assert.equal(app.posts[0].credentials, "include");
  assert.equal(app.posts[0].redirect, "error");
  assert.deepEqual(app.clicks, []);
});

test("legacy background endpoint uses a video-bound English track when the default track is ambiguous", async () => {
  const app = reader({
    initial: unloadedPanel(),
    tracks: [track, { ...track, languageCode: "es" }],
    language: "",
    background: modernData(),
  });
  assert.equal((await app.read()).source, "transcript");
  assert.equal(app.posts.length, 1);
  assert.equal(
    app.posts[0].url,
    "/youtubei/v1/get_transcript?prettyPrint=false",
  );
  assert.equal(app.posts[0].body.panelId, undefined);
});

test("dubbed tracks with an agreed English default use the modern panel without enabling CC", async () => {
  const app = reader({initial:unloadedPanel(),language:"",
    tracks:[{...track,languageCode:"nl-NL"},track,{...track,languageCode:"fr-FR"}],
    audioTracks:[{hasDefaultTrack:true,defaultCaptionTrackIndex:1,captionTrackIndices:[0,1,2]},
      {hasDefaultTrack:true,defaultCaptionTrackIndex:1,captionTrackIndices:[0,1,2]}],background:modernData()});
  const result=await app.read();
  assert.equal(result.source,"transcript");
  assert.equal(result.language,"en");
  assert.equal(app.posts[0].url,"/youtubei/v1/get_panel?prettyPrint=false");
  assert.equal(app.posts.length,1);
  assert.deepEqual(app.clicks,[]);
});

test("conflicting or invalid defaults cannot label a multilingual panel English", async () => {
  const good={hasDefaultTrack:true,defaultCaptionTrackIndex:1,captionTrackIndices:[0,1]};
  for(const bad of [{...good,defaultCaptionTrackIndex:0},{...good,defaultCaptionTrackIndex:99},
    {...good,defaultCaptionTrackIndex:"1"},{...good,hasDefaultTrack:false},{...good,captionTrackIndices:[0]}]) {
    const app=reader({initial:unloadedPanel(),tracks:[{...track,languageCode:"es"},track],
      audioTracks:[good,bad],background:modernData()});
    await app.read();
    assert.ok(app.posts.every(post=>!post.url.includes('get_panel')));
  }
  const app=reader({initial:unloadedPanel(),tracks:[{...track,languageCode:"es"},track],
    language:"es",audioTracks:[good],background:modernData()});
  await app.read();
  assert.ok(app.posts.every(post=>!post.url.includes('get_panel')));
  const initial=unloadedPanel();
  initial.engagementPanels[0]=modernData('es').engagementPanels[0];
  const mismatch=reader({initial,tracks:[{...track,languageCode:"es"},track],audioTracks:[good],background:modernData()});
  assert.ok((await mismatch.read()).error);
  assert.ok(mismatch.posts.every(post=>!post.url.includes('get_panel')));
});

test("background fetch rejects stale panel metadata and stale replies", async () => {
  const stale = reader({
    initial: unloadedPanel("0123456789_"),
    tracks: [track],
    background: modernData(),
  });
  assert.equal((await stale.read()).source, "transcript");
  assert.ok(stale.posts.every((post) => !post.url.includes("get_panel")));
  const changed = reader({
    initial: unloadedPanel(),
    tracks: [track],
    background: modernData(),
    onBackground: (context) => {
      context.location.href = "https://www.youtube.com/watch?v=0123456789_";
    },
  });
  assert.match((await changed.read()).error, /video changed/i);
  assert.deepEqual(changed.clicks, []);
});

test("unavailable background endpoints fail without enabling captions or opening panels", async () => {
  const app = reader({
    initial: unloadedPanel(),
    tracks: [track],
    language: "",
  });
  assert.match((await app.read()).error, /CC can stay off/);
  assert.equal(app.posts.length, 2);
  assert.deepEqual(app.clicks, []);
});

test("background fallback parses legacy cue endpoints and ignores advertised non-English tracks", async () => {
  const initial = modernData("es");
  initial.engagementPanels = [
    initial.engagementPanels[0],
    modernData().engagementPanels[0],
  ];
  const background = {
    actions: [
      {
        transcriptRenderer: {
          body: {
            segments: [
              {
                transcriptSegmentRenderer: {
                  startMs: "10000",
                  endMs: "15000",
                  snippet: { runs: [{ text: "Completed claim." }] },
                },
              },
              {
                transcriptSegmentRenderer: {
                  startMs: "18000",
                  endMs: "23000",
                  snippet: { runs: [{ text: "Unfinished claim." }] },
                },
              },
            ],
          },
        },
      },
    ],
  };
  const app = reader({ initial, background, language: "" });
  const result = await app.read();
  assert.equal(result.language, "en");
  assert.equal(result.captions.length, 1);
  assert.equal(result.captions[0].text, "Completed claim.");
  assert.equal(result.captions[0].end, 15);
  assert.equal(app.posts.length, 1);
  assert.equal(
    app.posts[0].body.params,
    modernData().engagementPanels[0].getTranscriptEndpoint.params,
  );
});

test("background attempts reserve the second request for the legacy endpoint", async () => {
  const initial = unloadedPanel();
  const duplicate = JSON.parse(JSON.stringify(initial.engagementPanels[1]));
  const command = duplicate.updateEngagementPanelContentCommand;
  command.globalConfiguration.params = encodeURIComponent(
    command.globalConfiguration.params,
  );
  initial.engagementPanels.push(duplicate);
  const app = reader({ initial, tracks: [track] });
  await app.read();
  assert.equal(app.posts.length, 2);
  assert.match(app.posts[0].url, /get_panel/);
  assert.match(app.posts[1].url, /get_transcript/);
});

test("unavailable caption tracks never expand the description or open the transcript", async () => {
  const app = reader({ tracks: [track], fetchError: true });
  const observed = {
    video_id: "abcdefghijk",
    cues: [{ start: 10, end: 15, text: "The law passed." }],
  };
  assert.equal((await app.read(observed)).source, "visible");
  assert.deepEqual(app.clicks, []);
  assert.match((await app.read()).error, /Captions could not/);
  assert.deepEqual(app.clicks, []);
});

test("never fetch arbitrary caption URLs or check advertisements and stale players", async () => {
  const app = reader({
    tracks: [{ ...track, baseUrl: "https://other.example/captions" }],
  });
  assert.match((await app.read()).error, /Captions could not/);
  assert.equal(app.requests.length, 0);
  assert.match((await reader({ ad: true }).read()).error, /advertisement/);
  assert.match((await reader({ changed: true }).read()).error, /video changed/);
});
