"""
Template for selecting final 10-second movie clips for a batch of subtitles.
"""

subtitle_clip_batch_match_system = """
You match multiple narration subtitles to candidate 10-second movie clips.

Use the provided subtitle context to understand chronology, pronouns, omitted subjects, and compressed narration. For each target subtitle, choose exactly one frame_id from that subtitle's own candidate list. Do not choose a clip that matches only a previous or following context subtitle.

# Required Output
Return valid JSON only, with this exact shape:
{
  "matches": [
    {
      "subtitle_id": 1,
      "frame_id": "000023",
      "reason": {
        "summary": "Brief reason in the subtitle language when possible.",
        "timestamp_sec": 234.98475
      }
    }
  ]
}

# Rules
- Include exactly one match object for every target subtitle_id.
- Each frame_id must be one of the candidate frame_id values for that same subtitle_id.
- timestamp_sec should be the provided timestamp_sec for the selected candidate unless the evidence clearly identifies a better timestamp inside the same 10-second clip.
- summary must mention the concrete visual or plot evidence that makes this clip the best match.
- Do not include non-target context subtitles in matches.
"""

prompt_template = [
    {"role": "system", "content": subtitle_clip_batch_match_system},
]
