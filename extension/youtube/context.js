// Context comes from this video's words and metadata, never the viewer's locale or channel country.
const regions = new Intl.DisplayNames(["en"], { type: "region", fallback: "none" });
const countries = new Map();
for (let a = 65; a <= 90; a++) for (let b = 65; b <= 90; b++) {
  const code = String.fromCharCode(a, b), name = regions.of(code);
  if (name && name.length > 3 && !["EU", "EZ", "UN", "XA", "XB", "ZZ", "GE", "JO", "TD", "TR"].includes(code)) countries.set(name, name);
}
for (const [alias, country] of Object.entries({
  "United States of America": "United States", "U.S.": "United States", "USA": "United States",
  "US": "United States", "U.K.": "United Kingdom", "UK": "United Kingdom",
  "Britain": "United Kingdom", "Great Britain": "United Kingdom",
})) countries.set(alias, country);
const names = [...countries.keys()].sort((a, b) => b.length - a.length);
const canonical = new Map([...countries].map(([name, country]) => [name.toLowerCase(), country]));
const places = new RegExp(`(?<![\\p{L}])(${names.map(n => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})(?![\\p{L}])`, "giu");

function countryIn(text) {
  // ASR often omits capitals; the pronoun "us" still cannot establish a country.
  const found = new Set([...text.matchAll(places)]
    .filter(match => match[1].toLowerCase() !== "us" || match[1] === "US")
    .map(match => canonical.get(match[1].toLowerCase())));
  return { value: found.size === 1 ? [...found][0] : "", ambiguous: found.size > 1 };
}

export function calendarDay(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return "";
  const parsed = new Date(value);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value ? value : "";
}

function recordingDate(text, published) {
  const months = "January February March April May June July August September October November December".split(" ");
  const pattern = new RegExp(`\\b(?:recorded|filmed|originally (?:aired|broadcast)|speech delivered)(?:\\s+on)?\\s*[:–-]?\\s*(\\d{4}-\\d{2}-\\d{2}|(?:${months.join("|")})\\s+\\d{1,2},?\\s+\\d{4}|\\d{1,2}\\s+(?:${months.join("|")})\\s+\\d{4})\\b`, "gi");
  const days = new Set();
  for (const match of text.matchAll(pattern)) {
    let day = match[1];
    if (!/^\d{4}-/.test(day)) {
      const parts = day.replace(",", "").split(/\s+/), month = months.findIndex(m => m.toLowerCase() === parts[0].toLowerCase());
      const index = month >= 0 ? month : months.findIndex(m => m.toLowerCase() === parts[1].toLowerCase());
      day = `${parts[2]}-${String(index + 1).padStart(2, "0")}-${(month >= 0 ? parts[1] : parts[0]).padStart(2, "0")}`;
    }
    day = calendarDay(day);
    if (day && (!published || day <= published)) days.add(day);
  }
  return days.size === 1 ? [...days][0] : "";
}

export function videoContext(payload) {
  const metadata = payload.video_metadata || {};
  const captionText = (payload.captions || []).map(cue => cue.text).join(" ");
  let country = countryIn(captionText), basis = country.value ? "detected from captions" : "";
  // Conflicting countries in the excerpt must not be resolved by the title or a previous video.
  if (!country.value && !country.ambiguous) {
    country = countryIn(payload.title || "");
    if (country.value) basis = "detected from the video title";
  }
  const spoken = recordingDate(`${payload.title || ""}\n${metadata.description || ""}`, payload.source_published_at);
  const stream = calendarDay(metadata.stream_day);
  return {
    claim_country: country.value,
    country_basis: basis,
    spoken_at: spoken || stream,
    spoken_at_basis: spoken ? "recording date stated in the video description or title" : stream ? "YouTube broadcast metadata (UTC)" : "",
  };
}
