// This function runs in the YouTube page world. Treat every returned field as untrusted.
export async function readCaptions(observed = {}) {
  try {
    return await capture();
  } catch (error) {
    return { error: error.message };
  }

  async function capture() {
    const failures = new Set();
    const visible = Array.isArray(observed.cues) ? observed.cues : [];
    const videoId =
      new URL(location.href).searchParams.get("v") ||
      location.pathname.match(/^\/shorts\/([\w-]{11})/)?.[1];
    const video = document.querySelector("video.html5-main-video"),
      player = document.getElementById("movie_player");
    if (!videoId || !video || !player)
      throw new Error("Open a YouTube video and play the claim first.");
    if (player.classList.contains("ad-showing"))
      throw new Error("Wait until the advertisement has finished.");
    const time = video.currentTime;
    const response =
      player.getPlayerResponse?.() || window.ytInitialPlayerResponse;
    if (
      response?.videoDetails?.videoId &&
      response.videoDetails.videoId !== videoId
    )
      throw new Error("The video changed. Try again once it has loaded.");
    const title = (response?.videoDetails?.title || document.title).slice(
      0,
      500,
    );
    const liveOrigin = response?.videoDetails?.isLiveContent
      || response?.videoDetails?.isLive
      || response?.microformat?.playerMicroformatRenderer?.liveBroadcastDetails;
    const published = response?.videoDetails?.videoId === videoId && !liveOrigin
      ? response.microformat?.playerMicroformatRenderer?.publishDate
      : "";
    const publishedDate = typeof published === "string" ? published.slice(0, 10) : "";
    const sourcePublishedAt = /^\d{4}-\d{2}-\d{2}$/.test(publishedDate)
      && (published === publishedDate || /^\d{4}-\d{2}-\d{2}T/.test(published))
      && !Number.isNaN(Date.parse(published))
      && !Number.isNaN(Date.parse(publishedDate))
      && new Date(publishedDate).toISOString().slice(0, 10) === publishedDate
      ? publishedDate : "";
    const active = player.getOption?.("captions", "track");
    const activeLanguage =
      active?.translationLanguage?.languageCode || active?.languageCode || "";
    const tracklist = response?.captions?.playerCaptionsTracklistRenderer;
    const tracks = tracklist?.captionTracks || [];
    const english = (language) => /^en(?:-|$)/.test(language || "");
    const recent = (cues) =>
      cues
        .filter(
          (cue) =>
            cue.start >= Math.max(0, time - 30) &&
            cue.end <= time &&
            cue.end >= cue.start &&
            cue.text.trim(),
        )
        .slice(-500);
    function result(cues, language, source) {
      const current =
        new URL(location.href).searchParams.get("v") ||
        location.pathname.match(/^\/shorts\/([\w-]{11})/)?.[1];
      if (current !== videoId || player.classList.contains("ad-showing"))
        throw new Error("The video changed during caption capture. Try again.");
      return {
        video_id: videoId,
        title,
        time,
        captions: recent(cues),
        language,
        source,
        source_published_at: sourcePublishedAt,
      };
    }
    const candidates = [...tracks].sort(
      (a, b) =>
        Number(english(b.languageCode)) - Number(english(a.languageCode)),
    );
    for (const track of candidates.slice(0, 3)) {
      if (!english(track.languageCode) && !track.isTranslatable) continue;
      try {
        const url = new URL(track.baseUrl);
        if (
          url.origin !== "https://www.youtube.com" ||
          url.pathname !== "/api/timedtext" ||
          url.searchParams.get("v") !== videoId
        )
          continue;
        url.searchParams.set("fmt", "json3");
        const language = english(track.languageCode)
          ? track.languageCode
          : "en";
        if (!english(track.languageCode)) url.searchParams.set("tlang", "en");
        const response = await fetch(url, {
          credentials: "include",
          signal: AbortSignal.timeout(3500),
        });
        if (!response.ok) {
          failures.add(`Caption track request: HTTP ${response.status}.`);
          continue;
        }
        const body = await response.text();
        if (!body || body.length > 8000000) {
          failures.add(body ? "Caption track exceeded the size limit." : "Caption track returned no text.");
          continue;
        }
        const data = JSON.parse(body),
          cues = [];
        for (const event of data.events || []) {
          if (!event.segs) continue;
          const start = event.tStartMs / 1000,
            end = start + (event.dDurationMs || 0) / 1000;
          cues.push({
            start,
            end,
            text: event.segs.map((part) => part.utf8 || "").join(""),
          });
        }
        if (recent(cues).length) return result(cues, language, "track");
      } catch {
        failures.add("Caption track request failed or timed out.");
        /* The player's visible captions may remain available when timedtext fails. */
      }
    }
    for (const track of video.textTracks || []) {
      if (!english(track.language) || !track.cues) continue;
      const cues = [...track.cues].map((cue) => ({
        start: cue.startTime,
        end: cue.endTime,
        text: cue.text,
      }));
      if (recent(cues).length)
        return result(cues, track.language, "text-track");
    }
    async function modernTranscript() {
      const watch = document.querySelector("ytd-watch-flexy")?.data;
      const initial =
        watch?.currentVideoEndpoint?.watchEndpoint?.videoId === videoId
          ? watch
          : window.ytInitialData;
      if (initial?.currentVideoEndpoint?.watchEndpoint?.videoId !== videoId) {
        failures.add("Transcript metadata is not ready for this video.");
        return null;
      }
      const items = [],
        endpoints = [],
        panels = [];
      let visited = 0;
      function walk(value, depth = 0) {
        if (
          !value ||
          typeof value !== "object" ||
          depth > 30 ||
          ++visited > 100000
        )
          return;
        if (value.transcriptSegmentViewModel)
          items.push(value.transcriptSegmentViewModel);
        const segment = value.transcriptSegmentRenderer;
        if (
          segment &&
          /^\d+$/.test(segment.startMs) &&
          /^\d+$/.test(segment.endMs)
        )
          items.push({
            start: Number(segment.startMs) / 1000,
            end: Number(segment.endMs) / 1000,
            simpleText:
              segment.snippet?.simpleText ||
              (segment.snippet?.runs || [])
                .map((run) => run.text || "")
                .join(""),
          });
        if (value.getTranscriptEndpoint?.params)
          endpoints.push(value.getTranscriptEndpoint.params);
        const panel = value.updateEngagementPanelContentCommand;
        if (
          panel?.contentSourcePanelIdentifier?.tag ===
            "PAmodern_transcript_view" &&
          typeof panel.globalConfiguration?.params === "string"
        )
          panels.push(panel.globalConfiguration.params);
        for (const child of Object.values(value)) walk(child, depth + 1);
      }
      walk(initial.engagementPanels);
      // The transcript endpoint carries the video ID and its selected caption track.
      function fields(encoded) {
        if (typeof encoded !== "string" || encoded.length > 16384)
          throw new Error("Invalid transcript metadata");
        const raw = atob(
          decodeURIComponent(encoded).replaceAll("-", "+").replaceAll("_", "/"),
        );
        let position = 0;
        const values = new Map();
        function integer() {
          let value = 0,
            shift = 0;
          while (position < raw.length && shift < 35) {
            const byte = raw.charCodeAt(position++);
            value += (byte & 127) * 2 ** shift;
            if (!(byte & 128)) return value;
            shift += 7;
          }
          throw new Error("Invalid transcript metadata");
        }
        while (position < raw.length) {
          const tag = integer(),
            type = tag & 7;
          if (type === 0) integer();
          else if (type === 2) {
            const length = integer();
            if (length > raw.length - position)
              throw new Error("Invalid transcript metadata");
            values.set(tag >> 3, raw.slice(position, position + length));
            position += length;
          } else throw new Error("Unsupported transcript metadata");
        }
        return values;
      }
      const languages = new Set();
      const englishEndpoints = new Map();
      for (const endpoint of endpoints) {
        try {
          const outer = fields(endpoint);
          if (outer.get(1) !== videoId) continue;
          const track = fields(outer.get(2));
          if (track.get(3)) continue;
          if (track.get(2)) languages.add(track.get(2));
          if (english(track.get(2)))
            englishEndpoints.set(endpoint, track.get(2));
        } catch {
          /* Unknown metadata cannot establish the transcript language. */
        }
      }
      let selectedLanguage =
        languages.size === 1 && english([...languages][0])
          ? [...languages][0]
          : "";
      if (!selectedLanguage) items.length = 0;
      if (!items.length && selectedLanguage) {
        for (const row of document.querySelectorAll(
          "transcript-segment-view-model",
        )) {
          items.push({
            timestamp: row
              .querySelector(".ytwTranscriptSegmentViewModelTimestamp")
              ?.textContent.trim(),
            simpleText: row
              .querySelector(".ytAttributedStringHost")
              ?.textContent.trim(),
          });
        }
      }
      if (!items.length) {
        const context = window.ytcfg?.get?.("INNERTUBE_CONTEXT");
        const requests = [];
        // Dubbed audio may advertise many subtitle languages while every audio track defaults to English.
        const audioTracks = tracklist?.audioTracks || [];
        const defaults = audioTracks.map((audio) => {
          const index = audio.defaultCaptionTrackIndex;
          return Number.isInteger(index) && index >= 0
            && audio.hasDefaultTrack === true && audio.captionTrackIndices?.includes(index)
            ? tracks[index]?.languageCode : "";
        });
        const defaultLanguage = defaults.length && defaults.every((language) => english(language))
          && new Set(defaults).size === 1 ? defaults[0] : "";
        const panelLanguage = tracks.length && tracks.every((track) => english(track.languageCode))
          ? tracks[0].languageCode
          : response?.videoDetails?.videoId === videoId && defaultLanguage
            && selectedLanguage === defaultLanguage && (!activeLanguage || activeLanguage === defaultLanguage)
            ? defaultLanguage : "";
        if (panelLanguage) {
          for (const params of new Set(panels)) {
            try {
              const nested = fields(params).get(149);
              if (nested && fields(btoa(nested)).get(1) === videoId) {
                requests.push({
                  api: "get_panel",
                  params,
                  panelId: "PAmodern_transcript_view",
                  language: panelLanguage,
                });
                break;
              }
            } catch {
              /* Unknown panel metadata cannot establish video ownership. */
            }
          }
        }
        for (const [params, language] of englishEndpoints) {
          requests.push({ api: "get_transcript", params, language });
          break;
        }
        if (context?.client?.clientName && context.client.clientVersion) {
          if (!requests.length) failures.add("No video-bound English transcript endpoint was available.");
          for (const { api, language, ...payload } of requests) {
            try {
              const response = await fetch(
                `/youtubei/v1/${api}?prettyPrint=false`,
                {
                  method: "POST",
                  credentials: "include",
                  redirect: "error",
                  headers: { "Content-Type": "application/json" },
                  signal: AbortSignal.timeout(5000),
                  body: JSON.stringify({ context, ...payload }),
                },
              );
              if (!response.ok) {
                failures.add(`Transcript request: HTTP ${response.status}.`);
                continue;
              }
              const body = await response.text();
              if (!body || body.length > 8000000) {
                failures.add(body ? "Transcript exceeded the size limit." : "Transcript request returned no text.");
                continue;
              }
              visited = 0;
              walk(JSON.parse(body));
              if (items.length) {
                selectedLanguage = language;
                break;
              }
              failures.add("Transcript response contained no caption segments.");
            } catch {
              failures.add("Transcript request failed or timed out.");
              /* Keep capture read-only when the background endpoint is unavailable. */
            }
          }
        } else failures.add("YouTube's transcript request context is not ready.");
      }
      if (!selectedLanguage) return null;
      const cues = items
        .map((item) => ({
          start: Number.isFinite(item.start)
            ? item.start
            : /^\d+(?::[0-5]\d){1,2}$/.test(item.timestamp)
              ? item.timestamp
                  .split(":")
                  .map(Number)
                  .reduce((total, part) => total * 60 + part, 0)
              : NaN,
          end: item.end,
          text: item.simpleText || "",
        }))
        .filter((cue) => Number.isFinite(cue.start))
        .sort((a, b) => a.start - b.start);
      // Modern captions need the next timestamp; legacy captions carry explicit end times.
      return {
        language: selectedLanguage,
        cues: cues.map((cue, index) => ({
          ...cue,
          end: Number.isFinite(cue.end)
            ? cue.end
            : (cues[index + 1]?.start ?? Infinity),
        })),
      };
    }
    const modern = await modernTranscript();
    if (modern && recent(modern.cues).length)
      return result(modern.cues, modern.language, "transcript");
    function transcript() {
      return [...document.querySelectorAll("ytd-transcript-segment-renderer")]
        .map((row) => {
          const clock = row
            .querySelector(".segment-timestamp")
            ?.textContent.trim();
          const parts = clock?.split(":").map(Number);
          const start = parts?.reduce((total, value) => total * 60 + value, 0);
          return {
            start,
            text: row.querySelector(".segment-text")?.textContent.trim() || "",
          };
        })
        .filter((cue) => Number.isFinite(cue.start))
        .map((cue, index, cues) => ({
          ...cue,
          end: cues[index + 1]?.start ?? Infinity,
        }));
    }
    const cues = transcript();
    const transcriptLabel =
      document.querySelector("ytd-transcript-renderer #footer")?.textContent ||
      "";
    const transcriptLanguage = /\bEnglish\b/i.test(transcriptLabel) ? "en" : "";
    if (modern && recent(cues).length && transcriptLanguage)
      return result(cues, transcriptLanguage, "transcript");
    if (
      observed.video_id === videoId &&
      english(activeLanguage) &&
      recent(visible).length
    )
      return result(visible, activeLanguage, "visible");
    if (activeLanguage && !english(activeLanguage))
      throw new Error(
        "An English subtitle track or translation could not be read in the background for this video.",
      );
    throw new Error(
      "Captions could not be read in the background at this position. This video needs accessible English subtitles or a translation. Try again after the spoken sentence finishes; CC can stay off."
      + (failures.size ? "\n" + [...failures].join(" ") : ""),
    );
  }
}
