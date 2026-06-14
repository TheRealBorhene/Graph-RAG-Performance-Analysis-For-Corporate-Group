from openai import OpenAI
import json

ENTITY_FINDER_PROMPT = """You are reading a chunk of a financial document.
Find ALL potentially relevant named entities in the text.

Be PERMISSIVE — include anything that might be:
- A company or organization name
- A geographic location: a specific country, state, province, or city
  (e.g. United States, France, Texas, London, Canada — not regions, bodies of water, or blocs)
- A specific dollar value or percentage (e.g. $8,589, $1.2 billion, 26.6%)
- A raw number or percentage from a financial table row (e.g. 75.0, 25.0%, 100.0)
- A named business division or segment

Do NOT classify or filter — just find and list them as they appear in the text.

Return ONLY valid JSON:
{"candidates": ["entity1", "entity2", ...]}
If nothing found, return: {"candidates": []}"""

MODEL                = "gpt-4.1-mini"

def find_entities(client: OpenAI, text: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ENTITY_FINDER_PROMPT},
            {"role": "user",   "content": text}
        ],
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"}
    )
    try:
        candidates = json.loads(response.choices[0].message.content).get("candidates", [])
    except json.JSONDecodeError:
        candidates = []
    return candidates