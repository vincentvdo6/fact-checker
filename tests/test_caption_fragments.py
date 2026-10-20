"""Caption sentences split at speaker changes, and fragments that name nothing are recognised."""

from __future__ import annotations

from src.pipeline.caption_claims import caption_claims, names_a_subject, names_whom, reports_speech


def test_a_speaker_change_marker_is_a_boundary_and_never_part_of_a_sentence():
    text = "the economy grew three percent >> And you called Ukraine a disgrace for attacking Russia. Well, no."
    sentences = caption_claims(text)
    assert [sentence.text for sentence in sentences] == [
        "the economy grew three percent", "And you called Ukraine a disgrace for attacking Russia.", "Well, no."]
    assert all(">>" not in sentence.text and text[sentence.start:sentence.end] == sentence.text for sentence in sentences)
    assert [sentence.index for sentence in sentences] == [0, 1, 2]


def test_a_contrast_is_kept_together_only_within_one_speakers_turn():
    same = caption_claims("We don't have a labor shortage. We have a good job shortage.")
    assert [sentence.text for sentence in same] == ["We don't have a labor shortage. We have a good job shortage."]
    split = caption_claims("We don't have a labor shortage. >> We have a good job shortage.")
    assert [sentence.text for sentence in split] == ["We don't have a labor shortage.", "We have a good job shortage."]


def test_sentences_made_only_of_pronouns_numbers_and_times_name_nothing_to_check():
    for fragment in ("Last month it was 32,000", "Last month it was thirty two thousand.", "He won.",
                     "They did it again in 2024.", "It was the third time this year."):
        assert not names_a_subject(fragment), fragment
    for claim in ("Inflation hit 9%.", "Biden won.", "25,000 soldiers on average a month being killed.",
                  "We don't have a labor shortage. We have a good job shortage."):
        assert names_a_subject(claim), claim


def test_a_third_person_pronoun_subject_names_no_one_unless_a_name_follows():
    for unnamed in ("They don't have equity in their home.", "His brother's a great guy, he's a friend of mine.",
                    "And she said, I want the plant closed.", "Uh, they cut it again in 2024."):
        assert not names_whom(unnamed), unnamed
    for named in ("He fought bravely in the famous Battle of Manila.", "They raised rates in March, the Fed said.",
                  "We don't have a labor shortage.", "It's also the third time in the last 5 months our economy has lost jobs.",
                  "Inflation hit 9%.", "they cut rates again in 2024"):
        assert names_whom(named), named


def test_a_report_of_what_someone_said_is_reported_speech_but_a_cited_report_is_not():
    for speech in ("He said they're just having headaches.", "And I said, \"Please extend my best wishes.\"",
                   "well she told them the plant would close", "They claimed inflation was falling."):
        assert reports_speech(speech), speech
    for claim in ("The report said unemployment rose.", "Jobs report says unemployment rose.", "Inflation hit 9%.",
                  "The Labor Department said unemployment rose to 4.2 percent.",
                  "We got over 100,000 Americans out of Afghanistan.", "A new study says sleep matters.",
                  # A named subject may be a company whose announcement is the event itself.
                  "Amazon announced about 14,000 corporate job cuts in October 2025.",
                  "Netflix said it would buy Warner Bros. Discovery's studios."):
        assert not reports_speech(claim), claim
