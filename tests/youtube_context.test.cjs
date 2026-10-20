const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");
const code = readFileSync(join(__dirname, "../extension/youtube/context.js"), "utf8").replaceAll("export function", "function");
const scope = vm.createContext({});
vm.runInContext(code, scope);
const context = fields => scope.videoContext({title:"A discussion",captions:[],...fields});

test("country names come from current speech before the title, across topics and countries", () => {
  for (const country of ["Canada", "India", "Argentina", "Nigeria", "United States", "New Zealand"]) {
    const result=context({title:"United Kingdom news",captions:[{text:`The policy changed in ${country}.`}]});
    assert.equal(result.claim_country,country);
    assert.equal(result.country_basis,"detected from captions");
  }
  assert.equal(context({title:"Canada election results"}).country_basis,"detected from the video title");
  assert.equal(context({captions:[{text:"The US unemployment rate fell."}]}).claim_country,"United States");
  assert.equal(context({captions:[{text:"in new zealand the rate fell"}]}).claim_country,"New Zealand");
  assert.equal(context({captions:[{text:"Curaçao passed the law."}]}).claim_country,"Curaçao");
});

test("comparisons, ambiguous names, user locale and channel location do not supply claim country", () => {
  for (const text of ["Canada and France signed the agreement.","They asked us to polish it.","Jordan spoke to Georgia."]) {
    const result=context({captions:[{text}],locale:"en-US",channel_country:"US"});
    assert.equal(result.claim_country,"");
  }
  assert.equal(context({title:"United States news",captions:[{text:"Canada and France signed the agreement."}]}).claim_country,"");
});

test("America and Americans name the United States, but the continents do not", () => {
  for (const text of ["We are going to make America affordable again.", "american families are paying more",
                      "Americans deserve better wages."]) {
    const result = context({captions:[{text}]});
    assert.equal(result.claim_country, "United States", text);
    assert.equal(result.country_basis, "detected from captions");
  }
  for (const text of ["Exports to Latin America rose.", "South American growth slowed.", "the North American trade pact",
                      "Central America saw record rain.", "The American people and Canada agreed."]) {
    assert.equal(context({captions:[{text}]}).claim_country, "", text);
  }
  assert.equal(context({captions:[{text:"Prices in the Americas rose."}]}).claim_country, "", "the plural continent is not a country");
});

test("publication date alone never becomes the date speech was recorded", () => {
  const result=context({source_published_at:"2025-07-20",title:"Election results, July 15, 2025"});
  assert.equal(result.spoken_at,"");
  for (const date of ["2025-07-15", "July 15, 2025", "15 July 2025"]) {
    const result=context({source_published_at:"2025-07-20",video_metadata:{description:`Recorded on ${date}.`}});
    assert.equal(result.spoken_at,"2025-07-15");
    assert.match(result.spoken_at_basis,/recording date stated/);
  }
});

test("conflicting, invalid or later recording dates stay unknown", () => {
  for (const description of ["Recorded on 2025-02-30", "Recorded on 2025-07-21", "Recorded on 2025-07-10. Filmed on 2025-07-12."]) {
    assert.equal(context({source_published_at:"2025-07-20",video_metadata:{description}}).spoken_at,"");
  }
  assert.equal(context({video_metadata:{stream_day:"2025-07-15"}}).spoken_at_basis,"YouTube broadcast metadata (UTC)");
  assert.equal(context({video_metadata:{stream_day:"2025-02-30"}}).spoken_at,"");
});
