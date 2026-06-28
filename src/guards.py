import re
from logger import info, drop

# ─────────────────────────────────────────────
# PATTERNS
# ─────────────────────────────────────────────

# Matches financial values for auto-registration:
#   $116,193 / $1.4 billion    -  dollar with optional commas/suffix
#   116,193 / 14,304           -  comma-thousands format, no $ sign
#   75.0% / 25.0 / 0.5         -  bare number or percentage
#   Negative variants of all the above
FINANCIAL_VALUE_PATTERN = re.compile(
    r'^\$[\d,]+(\.\d+)?(\s*(billion|million|trillion|thousand))?$'
    r'|^\$\(\d[\d,]*(\.\d+)?\)$'
    r'|^\d{1,3}(,\d{3})*(\.\d+)?$'
    r'|^\d+(\.\d+)?%?$'
    r'|^-\$[\d,]+(\.\d+)?(\s*(billion|million|trillion|thousand))?$'
    r'|^-\d{1,3}(,\d{3})*(\.\d+)?$'
    r'|^-\d+(\.\d+)?%?$',
    re.IGNORECASE
)

# Matches a parenthetical negative financial value — accounting notation for negatives.
# Examples: (72,880)  ($1,200)  (1.2 billion)  (0.5%)
# Capture group 1 holds the inner value so we can rewrite as -<value>.
PAREN_NEGATIVE_PATTERN = re.compile(
    r'^\((\$?[\d,]+(\.\d+)?(\s*(billion|million|trillion|thousand))?%?)\)$',
    re.IGNORECASE
)

# Geography names that are too broad/vague to be useful in the graph.
TOO_BROAD_GEO_PATTERN = re.compile(
    r'^(north america|south america|latin america|central america|'
    r'europe|asia|africa|middle east|oceania|pacific|'
    r'apac|emea|americas|worldwide|global|international|'
    r'rest of world|other|western europe|eastern europe|'
    r'southeast asia|east asia|south asia|sub-saharan africa|'
    r'gulf coast|midwest|northeast|northwest|southeast|southwest)$',
    re.IGNORECASE
)

# Legal suffix pattern — stripped when normalising subsidiary names for fuzzy matching.
_LEGAL_SUFFIX = re.compile(
    r',?\s*(lp|llc|inc\.?|corp\.?|corporation|ltd\.?|co\.?|plc|llp|'
    r'holding\s+corp(?:oration)?|holding|holdings?|'
    r'partners?|group|company|companies|&\s*co\.?)\s*$',
    re.IGNORECASE
)

# Characters either side of a Business Segment name to search for GAAP phrases.
SEGMENT_WINDOW = 200

# Matches date-like strings that should never appear as Financial Item values.
DATE_STRING_PATTERN = re.compile(
    r'^\d{4}$'
    r'|Q[1-4]\s*\d{4}'
    r'|fiscal\s+year\s+\d{4}'
    r'|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+\d{1,2},?\s+\d{4}',
    re.IGNORECASE
)

# Ownership language required to validate a Subsidiary entity.
# NOTE: "acquired" is intentionally excluded — it is ambiguous (e.g. "ACE acquired Chubb"
# does not mean Loews owns either). Directional acquisition is handled separately below.
OWNERSHIP_PHRASES = (
    "wholly-owned subsidiary", "wholly owned subsidiary",
    "owned by", "controlled by", "owns",
    "owned subsidiary", "% owned", "percent owned",
)

# Directional acquisition — only validates when the filing company (or first-person
# pronoun) is the subject doing the acquiring, not a third party.
DIRECTIONAL_ACQUISITION = re.compile(
    r'\b(we|our company|the company)\s+(acquired|completed the acquisition of|'
    r'purchase[sd]? all|purchase[sd]? the remaining)\b',
    re.IGNORECASE
)

# Peer / stock-comparison context — entities appearing near these phrases are
# competitors or index constituents, not subsidiaries.
PEER_CONTEXT_PATTERN = re.compile(
    r'\b(peer group|stock performance|comparison group|composite index|'
    r'stock price performance|cumulative total return|'
    r'the following graph compares|published industry)\b',
    re.IGNORECASE
)

# GAAP phrases required to validate a Business Segment entity.
# Broader set to match segment reporting tables, narratives, and headers.
GAAP_PHRASES = (
    "reportable segment", "operating segment", "business segment",
    "segment result", "segment income", "segment loss", "segment revenue",
    "segment operating", "by segment", "our segment", "segment information",
    "segment data", "segment performance",
)

# Keywords in a Financial Item's name that immediately identify it as a non-financial metric.
NON_FINANCIAL_NAME = re.compile(
    r'\b(employees?|workforce|headcount|male|female|gender|'
    r'countries|nations|referral|hires?|turnover|attrition)\b',
    re.IGNORECASE
)

# Keywords used to detect a non-financial context window around a Financial Item value.
NON_FINANCIAL_CONTEXT = re.compile(
    r'\b(employees?|workforce|headcount|diversity|inclusion|gender|'
    r'hiring|retention|turnover|attrition|referral|'
    r'emissions?|carbon|renewable|sustainability|environmental|'
    r'suppliers?|engagement)\b',
    re.IGNORECASE
)

# Generic Financial Item validity check.
FINANCIAL_ITEM_VALID = re.compile(
    r'\$[\d,]'
    r'|\$\('
    r'|^\('
    r'|^\d'
    r'|%'
    r'|\b(billion|million|trillion|thousand)\b',
    re.IGNORECASE
)

# Matches per-share EPS values: $2.94, $0.17, $1.19
EPS_VALUE_PATTERN = re.compile(r'^\$\d{1,3}\.\d{1,4}$')

# Keywords confirming a per-share context around a potential EPS value
EPS_CONTEXT_PATTERN = re.compile(
    r'\b(per\s+share|diluted|basic|eps|earnings\s+per\s+share)\b',
    re.IGNORECASE
)

# Inner-scale form of an accounting negative, e.g. "(1.2) billion" -> capture for "(1.2 billion)".
_PAREN_SCALE_OUTSIDE = re.compile(
    r'^(\$?)\((\d[\d,]*(?:\.\d+)?)\)\s*(billion|million|trillion|thousand)',
    re.IGNORECASE
)


def _normalize_paren_negative(value) -> str | None:
    """
    Convert an accounting parenthetical negative to a leading-minus form, or return None
    if the value is not a parenthetical negative. Shared by the entity-name guard and the
    relationship-target guard so the logic lives in one place.
      "(72,880)"      → "-72,880"
      "$(1,200)"      → "-$1,200"
      "(1.2 billion)" → "-1.2 billion"
    """
    name = str(value).strip()
    m_outside = _PAREN_SCALE_OUTSIDE.match(name)
    if m_outside:
        prefix, digits, scale = m_outside.groups()
        name = f"({prefix}{digits} {scale})"
    elif re.match(r'^\$\(', name):
        name = "($" + name[2:]
    m = PAREN_NEGATIVE_PATTERN.match(name)
    return "-" + m.group(1) if m else None


# ─────────────────────────────────────────────
# SUBSIDIARY NAME NORMALISATION
# ─────────────────────────────────────────────

def _normalize_sub(name: str) -> str:
    """
    Normalise a subsidiary name for fuzzy canonical matching.
    Steps:
      1. Lowercase
      2. Repeatedly strip legal suffixes until stable
         (handles multi-word suffixes like "Holding Corporation")
      3. Strip stray punctuation/connectors left by suffix removal
      4. Normalise plural/variant forms
    Examples:
      "Apple Inc."                          → "apple"
      "General Electric Company"            → "general electric"
      "JPMorgan Chase & Co."                → "jpmorgan chase"
      "Berkshire Hathaway Holdings, Inc."   → "berkshire hathaway"   (multi-suffix)
      "Blackstone Real Estate Partners, LP" → "blackstone real estate" (multi-suffix)
      "Loews Hotels Holding Corporation"    → "loews hotels"
      "Boardwalk Pipeline Partners, LP"     → "boardwalk pipeline"
    """
    n = name.strip().lower()
    while True:
        stripped = _LEGAL_SUFFIX.sub('', n).strip().rstrip('&,. ').strip()
        if stripped == n:
            break
        n = stripped
    n = re.sub(r'\bpipelines\b', 'pipeline', n)
    n = re.sub(r'\bholdings\b',  'holding',  n)
    n = re.sub(r'\bpartners\b',  'partner',  n)
    n = re.sub(r'\bcompanies\b', 'company',  n)
    return n.strip()


def _segment_canonical(subsidiary_name: str) -> str:
    """
    Derive the canonical Business Segment short name from a Subsidiary's legal name.
    Same logic as _normalize_sub but preserves original casing — strips legal suffixes
    iteratively until stable, without lowercasing.
    Examples:
      "CNA Financial Corporation"        → "CNA Financial"
      "Diamond Offshore Drilling, Inc."  → "Diamond Offshore Drilling"
      "Boardwalk Pipeline Partners, LP"  → "Boardwalk Pipeline"
      "Loews Hotels Holding Corporation" → "Loews Hotels"
      "Altium Packaging LLC"             → "Altium Packaging"
    """
    n = subsidiary_name.strip()
    while True:
        stripped = _LEGAL_SUFFIX.sub('', n).strip().rstrip('&,. ').strip()
        if stripped == n or not stripped:
            break
        n = stripped
    return n


# Generic descriptors stripped from a GENERATED source label before matching it to a
# subsidiary by leading token — e.g. "CNA segment" → "CNA", "Diamond operations" → "Diamond".
_GENERIC_SOURCE_WORDS = {
    "segment", "segments", "business", "operations", "operation",
    "division", "divisions", "results", "result", "unit",
}


def _resolve_source_by_content(src: str, sub_names: list[str]) -> str | None:
    """
    Resolve a GENERATED source label to a confirmed Subsidiary using the LABEL'S OWN
    content — not the surrounding chunk text.

    Agent 3 sometimes attributes a metric to an abbreviated / segment form ("CNA segment")
    whose canonical name never appears in the chunk. The whole-text re-route below then
    misfires onto whatever subsidiary IS named nearby (e.g. Diamond Offshore), producing a
    misattribution. Matching the source string itself avoids that: strip generic descriptors
    ("segment"...), normalise, then compare the leading token to each confirmed subsidiary's
    leading token. Resolve only on a single unambiguous match; otherwise return None and let
    the caller fall through (so an ambiguous source is dropped, never guessed).

    Examples (generic — derived from the confirmed-subsidiary list, nothing hardcoded):
      "CNA segment"        + [CNA Financial Corporation, Diamond ...] → "CNA Financial Corporation"
      "Boardwalk Pipelines"+ [Boardwalk Pipeline Partners, LP, ...]   → "Boardwalk Pipeline Partners, LP"
      "Corporate"          + [...]                                    → None  (no leading-token match)
    """
    tokens = [t for t in re.split(r'\s+', src.strip())
              if t and t.lower() not in _GENERIC_SOURCE_WORDS]
    core_norm = _normalize_sub(" ".join(tokens))
    if not core_norm:
        return None
    core_first = core_norm.split()[0]
    matches: list[str] = []
    for sub in sub_names:
        sub_norm = _normalize_sub(sub)
        if sub_norm and sub_norm.split() and sub_norm.split()[0] == core_first:
            matches.append(sub)
    matches = list(dict.fromkeys(matches))
    return matches[0] if len(matches) == 1 else None


def find_confirmed_subs_in_text(text: str, confirmed_subs: list[str]) -> list[str]:
    """
    Return confirmed subsidiaries whose distinctive leading token appears as a whole word
    in the chunk text. Used to re-inject a GENERATED source when Agent 2 dropped the
    subsidiary entirely (the chunk-53 "source starvation": Financial Items present but no
    subsidiary/segment for Agent 3 to attribute them to).

    Derived entirely from the confirmed-subsidiary list — nothing hardcoded. Over-matches
    (a token that appears but isn't the real owner) are harmless: Agent 3's grounding rule
    only attributes values adjacent to a name, and drop_orphaned_financial_items removes any
    injected subsidiary that ends up with no edge.
    """
    text_lower = text.lower()
    found: list[str] = []
    for sub in confirmed_subs:
        sub_norm = _normalize_sub(sub)
        if not sub_norm:
            continue
        lead = sub_norm.split()[0]
        if re.search(r'\b' + re.escape(lead) + r'\b', text_lower):
            found.append(sub)
    return list(dict.fromkeys(found))


# ─────────────────────────────────────────────
# DETERMINISTIC FINANCIAL ITEM PROMOTION
# ─────────────────────────────────────────────

def promote_financial_items(candidates: list, entities: list[dict]) -> list[dict]:
    """
    Deterministically recover Financial Items that Agent 2 (LLM) dropped.

    Agent 2 is non-deterministic even at temperature=0 and occasionally discards
    Financial Items it should keep (e.g. chunk 53 collapsing from ~20 entities to 1).
    Financial values are highly regular, so any Agent 1 candidate that matches
    FINANCIAL_VALUE_PATTERN and that Agent 2 did NOT already classify is promoted to a
    Financial Item here. This is the upstream counterpart to the existing post-Agent-3
    auto-registration in validate_relationships.

    Safety nets that still apply afterwards:
      - apply_entity_guards drops dates, EPS, non-financial and malformed values
      - drop_orphaned_financial_items removes any promoted item with no GENERATED edge

    Returns only the NEW Financial Item dicts (caller appends them to its entity list).
    """
    existing = {e["name"] for e in entities}
    promoted: list[dict] = []
    for cand in candidates:
        name = str(cand).strip()
        if not name or name in existing:
            continue
        # FINANCIAL_VALUE_PATTERN covers $/bare/percentage/$()-negative forms;
        # PAREN_NEGATIVE_PATTERN additionally covers bare accounting negatives like
        # "(211)" — the apply_entity_guards pass then normalises those to "-211".
        if FINANCIAL_VALUE_PATTERN.match(name) or PAREN_NEGATIVE_PATTERN.match(name):
            promoted.append({"name": name, "type": "Financial Item"})
            existing.add(name)
    return promoted


# ─────────────────────────────────────────────
# POST-AGENT-2 ENTITY GUARDS
# ─────────────────────────────────────────────

def _is_non_financial(entity: dict, text: str) -> bool:
    """Return True if the Financial Item entity is a workforce/ESG metric, not a financial value."""
    if entity["type"] != "Financial Item":
        return False
    name = entity["name"]
    if NON_FINANCIAL_NAME.search(name):
        return True
    has_dollar = name.strip().startswith("$")
    has_suffix = bool(re.search(r'\b(billion|million|trillion|thousand)\b', name, re.IGNORECASE))
    if has_dollar or has_suffix:
        return False
    pattern     = re.compile(re.escape(name.strip()), re.IGNORECASE)
    occurrences = list(pattern.finditer(text))
    if not occurrences:
        return False
    return all(
        NON_FINANCIAL_CONTEXT.search(text[max(0, m.start() - 200): m.end() + 200])
        for m in occurrences
    )


def _is_eps_value(entity: dict, text: str) -> bool:
    """Return True if the Financial Item is a per-share EPS value."""
    if entity["type"] != "Financial Item":
        return False
    name = entity["name"].strip()
    if not EPS_VALUE_PATTERN.match(name):
        return False
    pattern     = re.compile(re.escape(name), re.IGNORECASE)
    occurrences = list(pattern.finditer(text))
    if not occurrences:
        return False
    return any(
        EPS_CONTEXT_PATTERN.search(text[max(0, m.start() - 300): m.end() + 300])
        for m in occurrences
    )


def _drop_malformed(entities: list[dict]) -> list[dict]:
    """Drop entities with a null, empty, or NUL-containing name."""
    before   = len(entities)
    entities = [e for e in entities if e.get("name") and '\x00' not in e["name"] and e["name"].strip()]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} malformed entity(ies) with null/empty names")
    return entities


def _normalise_parent_name(entities: list[dict], filing_company: str) -> None:
    """In place: snap a Parent whose name case-matches the filing company to the canonical form."""
    for entity in entities:
        if entity["type"] == "Parent" and entity["name"].lower() == filing_company.lower():
            if entity["name"] != filing_company:
                info(f"Parent name normalised: '{entity['name']}' → '{filing_company}'")
                entity["name"] = filing_company


def _drop_too_broad_geos(entities: list[dict]) -> list[dict]:
    """Drop Geography entities that are regions/blocs rather than a country/state/city."""
    before   = len(entities)
    entities = [
        e for e in entities
        if not (e["type"] == "Geography" and TOO_BROAD_GEO_PATTERN.match(e["name"].strip()))
    ]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Geography entity(ies) that are too broad (region/bloc)")
    return entities


def _drop_date_like_fis(entities: list[dict]) -> list[dict]:
    """Drop Financial Items whose value is actually a date string."""
    before   = len(entities)
    entities = [e for e in entities if not (e["type"] == "Financial Item" and DATE_STRING_PATTERN.search(e["name"]))]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Financial Item(s) that looked like date strings")
    return entities


def _drop_non_financial_fis(entities: list[dict], text: str) -> list[dict]:
    """Drop Financial Items that are workforce/ESG metrics rather than financial values."""
    before   = len(entities)
    entities = [e for e in entities if not _is_non_financial(e, text)]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Financial Item(s) identified as non-financial metrics (workforce/ESG)")
    return entities


def _drop_invalid_fi_format(entities: list[dict]) -> list[dict]:
    """Drop Financial Items that are labels/references, not numeric values."""
    before   = len(entities)
    entities = [e for e in entities if not (e["type"] == "Financial Item" and not FINANCIAL_ITEM_VALID.search(e["name"].strip()))]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Financial Item(s) that are labels/references, not numeric values")
    return entities


def _drop_eps_values(entities: list[dict], text: str) -> list[dict]:
    """Drop Financial Items identified as per-share EPS values."""
    before   = len(entities)
    entities = [e for e in entities if not _is_eps_value(e, text)]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Financial Item(s) identified as per-share EPS values")
    return entities


def _validate_subsidiaries(entities: list[dict], text: str, filing_company: str,
                           confirmed: set[str]) -> list[dict]:
    """
    Keep a Subsidiary only if it is re-validated from a prior chunk (matches a confirmed
    name) or supported by ownership language in the text. Confirmed subsidiaries are also
    canonicalised in place to their previously confirmed name.
    """
    validated_subsidiaries: set[str] = set()
    for ent in entities:
        if ent["type"] != "Subsidiary":
            continue
        ent_lower = ent["name"].lower()
        ent_norm  = _normalize_sub(ent["name"])
        matched_canonical = next(
            (conf for conf in confirmed if ent_lower in conf.lower() or conf.lower() in ent_lower),
            None
        )
        if matched_canonical is None:
            matched_canonical = next(
                (conf for conf in confirmed
                 if ent_norm and _normalize_sub(conf) and
                    (ent_norm in _normalize_sub(conf) or _normalize_sub(conf) in ent_norm)),
                None
            )
        if matched_canonical:
            if ent["name"] != matched_canonical:
                info(f"Name normalized: '{ent['name']}' → '{matched_canonical}'")
                ent["name"] = matched_canonical
            validated_subsidiaries.add(matched_canonical)
            info(f"Subsidiary '{matched_canonical}' re-validated from prior chunk (no ownership phrase needed)")
            continue
        filing_lower = filing_company.lower()
        pattern = re.compile(re.escape(ent["name"]), re.IGNORECASE)
        for match in pattern.finditer(text):
            start  = max(0, match.start() - 150)
            end    = min(len(text), match.end() + 150)
            window = text[start:end].lower()

            # Peer / stock-comparison context guard — discard immediately if the
            # entity appears inside a peer comparison or stock performance table.
            wide_start = max(0, match.start() - 500)
            wide_end   = min(len(text), match.end() + 500)
            wide_window = text[wide_start:wide_end]
            if PEER_CONTEXT_PATTERN.search(wide_window):
                break  # treat this occurrence as invalid; check next occurrence

            # Standard ownership phrases in the 150-char window
            if any(phrase in window for phrase in OWNERSHIP_PHRASES):
                # Accept first-person possessive ("our", "ours", "we") as
                # attribution — covers "a wholly owned subsidiary of ours".
                if re.search(r'\b(our|ours|we)\b', window):
                    validated_subsidiaries.add(ent["name"])
                    break
                if filing_lower in wide_window.lower():
                    validated_subsidiaries.add(ent["name"])
                    break

            # Directional acquisition — "we acquired X" / "our company acquired X"
            if DIRECTIONAL_ACQUISITION.search(wide_window):
                validated_subsidiaries.add(ent["name"])
                break
    before   = len(entities)
    entities = [e for e in entities if e["type"] != "Subsidiary" or e["name"] in validated_subsidiaries]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Subsidiary entity(ies) lacking ownership language in text")
    return entities


def _drop_segment_subcomponents(entities: list[dict]) -> list[dict]:
    """Drop a Business Segment whose name is a substring of another segment (a sub-component)."""
    segment_names  = [e["name"] for e in entities if e["type"] == "Business Segment"]
    valid_segments = {
        seg for seg in segment_names
        if not any(seg != other and seg.lower() in other.lower() for other in segment_names)
    }
    before   = len(entities)
    entities = [e for e in entities if e["type"] != "Business Segment" or e["name"] in valid_segments]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Business Segment sub-component(s)")
    return entities


def _validate_segment_adjacency(entities: list[dict], text: str) -> list[dict]:
    """Keep a Business Segment only if a GAAP segment phrase appears near a mention of it
    (in a 200-char window, or within a 400-char table lookback when the name is in a table cell)."""
    TABLE_LOOKBACK     = 400
    validated_segments: set[str] = set()
    for ent in entities:
        if ent["type"] != "Business Segment":
            continue
        pattern = re.compile(re.escape(ent["name"]), re.IGNORECASE)
        for match in pattern.finditer(text):
            start  = max(0, match.start() - SEGMENT_WINDOW)
            end    = min(len(text), match.end() + SEGMENT_WINDOW)
            window = text[start:end].lower()
            if any(phrase in window for phrase in GAAP_PHRASES):
                validated_segments.add(ent["name"])
                break
            pre = text[max(0, match.start() - 20):match.start()]
            if "|" in pre:
                table_start  = max(0, match.start() - TABLE_LOOKBACK)
                table_window = text[table_start:match.end()].lower()
                if any(phrase in table_window for phrase in GAAP_PHRASES):
                    validated_segments.add(ent["name"])
                    break
    before   = len(entities)
    entities = [e for e in entities if e["type"] != "Business Segment" or e["name"] in validated_segments]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Business Segment(s) failing adjacency check")
    return entities


def _canonicalise_segments(entities: list[dict], confirmed: set[str]) -> None:
    """
    In place: normalise each Business Segment name to a stable canonical form derived from
    its matching confirmed subsidiary. This ensures that "CNA Financial", "CNA Financial
    Corporation", and "CNA Financial Corp" all converge to one node name across chunks.
    """
    for ent in entities:
        if ent["type"] != "Business Segment":
            continue
        ent_norm = _normalize_sub(ent["name"])
        matched_sub = next(
            (sub for sub in confirmed
             if ent_norm and _normalize_sub(sub)
             and (ent_norm in _normalize_sub(sub) or _normalize_sub(sub) in ent_norm)),
            None,
        )
        if matched_sub:
            canonical = _segment_canonical(matched_sub)
            if ent["name"] != canonical:
                info(f"Segment name canonicalised: '{ent['name']}' → '{canonical}'")
                ent["name"] = canonical


def _normalise_paren_negatives(entities: list[dict]) -> None:
    """In place: convert Financial Item parenthetical negatives to a leading-minus form."""
    for entity in entities:
        if entity["type"] == "Financial Item":
            normalised = _normalize_paren_negative(entity["name"])
            if normalised:
                info(f"Normalised parenthetical negative: '{entity['name']}' → '{normalised}'")
                entity["name"] = normalised


def apply_entity_guards(entities: list[dict], text: str, filing_company: str,
                        confirmed_subsidiaries: set[str] | None = None) -> list[dict]:
    """
    Apply all post-Agent-2 guards in sequence and return the cleaned entity list.
    Each guard is a small helper above. The order matters: later guards (subsidiary /
    segment validation and canonicalisation) depend on names that earlier guards normalise.
    """
    confirmed = confirmed_subsidiaries or set()

    entities = _drop_malformed(entities)
    _normalise_parent_name(entities, filing_company)
    entities = _drop_too_broad_geos(entities)
    entities = _drop_date_like_fis(entities)
    entities = _drop_non_financial_fis(entities, text)
    entities = _drop_invalid_fi_format(entities)
    entities = _drop_eps_values(entities, text)
    entities = _validate_subsidiaries(entities, text, filing_company, confirmed)
    entities = _drop_segment_subcomponents(entities)
    entities = _validate_segment_adjacency(entities, text)
    _canonicalise_segments(entities, confirmed)
    _normalise_paren_negatives(entities)

    return entities


# ─────────────────────────────────────────────
# POST-AGENT-3 RELATIONSHIP VALIDATION
# ─────────────────────────────────────────────
# `validate_relationships` and its 6 per-step helpers were extracted into
# relationship_validator.py to keep this module focused on entity-side guards.
# Consumers should import from there:
#     from relationship_validator import validate_relationships
