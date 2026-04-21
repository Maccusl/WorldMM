"""
Template for selecting the final 10-second movie clip for one subtitle.
"""

subtitle_clip_match_system = """
You match one narration subtitle to exactly one candidate 10-second movie clip.

Use only the provided candidates and retrieved evidence. Choose the candidate whose visible events, character actions, setting, and plot context best express the subtitle. If the subtitle is abstract or compressed, prefer the candidate that visually represents the core event rather than a loosely related surrounding moment.

# Required Output
Return valid JSON only, with this exact shape:
{
  "frame_id": "000023",
  "reason": {
    "summary": "Brief reason in the subtitle language when possible.",
    "timestamp_sec": 234.98475
  }
}

# Rules
- frame_id must be one of the provided candidate frame_id values.
- timestamp_sec should be the provided timestamp_sec for the selected candidate unless the evidence clearly identifies a better timestamp inside the same 10-second clip.
- summary must mention the concrete visual or plot evidence that makes this clip the best match.
- Do not include subtitle_id or text in your response.
"""

prompt_template = [
    {"role": "system", "content": subtitle_clip_match_system},
]
