import re
from logger import info, drop, warn, new, log_error

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


def apply_entity_guards(entities: list[dict], text: str, filing_company: str,
                        confirmed_subsidiaries: set[str] | None = None) -> list[dict]:
    """Apply all post-Agent-2 guards and return the cleaned entity list."""

    # Malformed entity guard
    before   = len(entities)
    entities = [e for e in entities if e.get("name") and '\x00' not in e["name"] and e["name"].strip()]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} malformed entity(ies) with null/empty names")

    # Parent name normalisation
    for entity in entities:
        if entity["type"] == "Parent" and entity["name"].lower() == filing_company.lower():
            if entity["name"] != filing_company:
                info(f"Parent name normalised: '{entity['name']}' → '{filing_company}'")
                entity["name"] = filing_company

    # Geography too-broad guard
    before   = len(entities)
    entities = [
        e for e in entities
        if not (e["type"] == "Geography" and TOO_BROAD_GEO_PATTERN.match(e["name"].strip()))
    ]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Geography entity(ies) that are too broad (region/bloc)")

    # Financial Item date-string guard
    before   = len(entities)
    entities = [e for e in entities if not (e["type"] == "Financial Item" and DATE_STRING_PATTERN.search(e["name"]))]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Financial Item(s) that looked like date strings")

    # Financial Item non-financial metrics guard
    before   = len(entities)
    entities = [e for e in entities if not _is_non_financial(e, text)]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Financial Item(s) identified as non-financial metrics (workforce/ESG)")

    # Financial Item format guard
    before   = len(entities)
    entities = [e for e in entities if not (e["type"] == "Financial Item" and not FINANCIAL_ITEM_VALID.search(e["name"].strip()))]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Financial Item(s) that are labels/references, not numeric values")

    # EPS per-share value guard
    before   = len(entities)
    entities = [e for e in entities if not _is_eps_value(e, text)]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Financial Item(s) identified as per-share EPS values")

    # Subsidiary ownership language guard
    _confirmed = confirmed_subsidiaries or set()
    validated_subsidiaries: set[str] = set()
    for ent in entities:
        if ent["type"] != "Subsidiary":
            continue
        ent_lower = ent["name"].lower()
        ent_norm  = _normalize_sub(ent["name"])
        matched_canonical = next(
            (conf for conf in _confirmed if ent_lower in conf.lower() or conf.lower() in ent_lower),
            None
        )
        if matched_canonical is None:
            matched_canonical = next(
                (conf for conf in _confirmed
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

    # Business Segment sub-component guard
    segment_names  = [e["name"] for e in entities if e["type"] == "Business Segment"]
    valid_segments = {
        seg for seg in segment_names
        if not any(seg != other and seg.lower() in other.lower() for other in segment_names)
    }
    before   = len(entities)
    entities = [e for e in entities if e["type"] != "Business Segment" or e["name"] in valid_segments]
    if len(entities) < before:
        drop(f"Dropped {before - len(entities)} Business Segment sub-component(s)")

    # Business Segment adjacency guard
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

    # Business Segment canonical name resolution
    # After the adjacency guard, normalize each segment name to a stable canonical
    # form derived from its matching confirmed subsidiary. This ensures that "CNA
    # Financial", "CNA Financial Corporation", and "CNA Financial Corp" all converge
    # to the same node name ("CNA Financial") across chunks.
    _conf_list = list(_confirmed)
    for ent in entities:
        if ent["type"] != "Business Segment":
            continue
        ent_norm = _normalize_sub(ent["name"])
        matched_sub = next(
            (sub for sub in _conf_list
             if ent_norm and _normalize_sub(sub)
             and (ent_norm in _normalize_sub(sub) or _normalize_sub(sub) in ent_norm)),
            None,
        )
        if matched_sub:
            canonical = _segment_canonical(matched_sub)
            if ent["name"] != canonical:
                info(f"Segment name canonicalised: '{ent['name']}' → '{canonical}'")
                ent["name"] = canonical

    # Parenthetical negative normalisation
    for entity in entities:
        if entity["type"] == "Financial Item":
            name = entity["name"].strip()
            m_outside = re.match(
                r'^(\$?)\((\d[\d,]*(?:\.\d+)?)\)\s*(billion|million|trillion|thousand)',
                name, re.IGNORECASE
            )
            if m_outside:
                prefix, digits, scale = m_outside.groups()
                name = f"({prefix}{digits} {scale})"
            elif re.match(r'^\$\(', name):
                name = "($" + name[2:]
            m = PAREN_NEGATIVE_PATTERN.match(name)
            if m:
                original = entity["name"]
                entity["name"] = "-" + m.group(1)
                info(f"Normalised parenthetical negative: '{original}' → '{entity['name']}'")

    return entities


# ─────────────────────────────────────────────
# POST-AGENT-3 RELATIONSHIP VALIDATION
# ─────────────────────────────────────────────

def validate_relationships(relationships: list[dict], entity_names: set,
                           entity_type_map: dict, chunk_id: int,
                           file: str, page_number: int,
                           run_timestamp: str, all_entities: list[dict],
                           chunk_text: str = "") -> list[dict]:
    """Validate Agent 3 relationships: fix $ prefixes, auto-register FIs, check types."""

    dollar_normalised = {
        name.lstrip("$").strip(): name
        for name in entity_names if name.startswith("$")
    }

    valid = []
    for rel in relationships:
        src      = rel.get("source")
        tgt      = rel.get("target")
        rel_type = rel.get("type")

        # $ prefix correction on target
        if tgt and tgt not in entity_names:
            canonical = dollar_normalised.get(str(tgt).lstrip("$").strip())
            if canonical:
                rel["target"] = canonical
                tgt = canonical

        # $-value → -$value normalisation on target
        if tgt and str(tgt).startswith("$-"):
            tgt = "-$" + str(tgt)[2:]
            rel["target"] = tgt

        # Parenthetical negative normalisation on target
        if tgt and tgt not in entity_names:
            tgt_str   = str(tgt).strip()
            m_outside = re.match(
                r'^(\$?)\((\d[\d,]*(?:\.\d+)?)\)\s*(billion|million|trillion|thousand)',
                tgt_str, re.IGNORECASE
            )
            if m_outside:
                prefix, digits, scale = m_outside.groups()
                tgt_str = f"({prefix}{digits} {scale})"
            elif re.match(r'^\$\(', tgt_str):
                tgt_str = "($" + tgt_str[2:]
            m_paren = PAREN_NEGATIVE_PATTERN.match(tgt_str)
            if m_paren:
                normalised = "-" + m_paren.group(1)
                info(f"Target parenthetical normalised: '{tgt}' → '{normalised}'")
                rel["target"] = normalised
                tgt = normalised

        # Source alias resolution
        if src and src not in entity_names:
            src_norm = _normalize_sub(src)
            canonical_src = next(
                (name for name in entity_names
                 if src_norm and _normalize_sub(name) == src_norm),
                None
            )
            if canonical_src:
                info(f"Source alias resolved: '{src}' → '{canonical_src}'")
                rel["source"] = canonical_src
                src = canonical_src

        # Fix 1 — Sub-segment GENERATED fallback
        # Agent 3 sometimes uses a sub-segment row header (e.g. "Specialty",
        # "Commercial", "International") as the GENERATED source instead of the
        # owning subsidiary.  These names are not in entity_names so the normal
        # alias resolver above can't help.
        #
        # Strategy: search the chunk TEXT for mentions of known Subsidiary names.
        # The text is the ground truth — even if the LLM failed to extract a
        # subsidiary as an entity this run, the subsidiary's name still appears
        # in the text.  If exactly one confirmed Subsidiary is mentioned in the
        # chunk text, it unambiguously owns the sub-segment.  If multiple are
        # found, attribution would be a guess so we fall through and let the
        # existence check below drop the relationship.
        #
        # This fixes the regression where chunk-entity scan picked Diamond Offshore
        # (the only Subsidiary the LLM happened to extract in chunk 53) instead of
        # CNA Financial Corporation (whose name appears throughout the chunk text).
        if src and src not in entity_names and rel_type == "GENERATED" and chunk_text:
            text_lower = chunk_text.lower()
            text_matched_subs = list(dict.fromkeys(
                name for name in entity_names
                if entity_type_map.get(name) == "Subsidiary"
                and (name.lower() in text_lower
                     or (_normalize_sub(name) and _normalize_sub(name) in text_lower))
            ))
            if len(text_matched_subs) == 1:
                info(f"Sub-segment source re-routed: '{src}' → '{text_matched_subs[0]}'")
                rel["source"] = text_matched_subs[0]
                src = text_matched_subs[0]

        src_in = src in entity_names
        tgt_in = tgt in entity_names

        # Financial value auto-registration (GENERATED only — REPORTED and HAS_METRIC removed)
        if (rel_type == "GENERATED"
                and src_in and not tgt_in
                and FINANCIAL_VALUE_PATTERN.match(str(tgt or ""))):
            all_entities.append({"name": tgt, "type": "Financial Item",
                                  "chunk_id": chunk_id, "file": file, "page_number": page_number})
            entity_names.add(tgt)
            tgt_in = True
            new(f"Auto-registered Financial Item: {tgt}")

        # Source / target existence check
        if not src_in or not tgt_in:
            warn(f"Dropped invalid relationship: [{src}] --{rel_type} --> [{tgt}]")
            log_error(run_timestamp, chunk_id, f"Invalid relationship dropped: source='{src}' target='{tgt}' type='{rel_type}'")
            continue

        # Relationship type checks
        src_type = entity_type_map.get(src, "")

        # Inverted PARENT_OF guard
        if rel_type == "PARENT_OF" and src_type == "Subsidiary" and entity_type_map.get(tgt, "") == "Parent":
            warn(f"Inverted PARENT_OF corrected: '{src}' ↔ '{tgt}'")
            rel["source"], rel["target"] = tgt, src
            src, tgt = rel["source"], rel["target"]
            src_type = entity_type_map.get(src, "")

        # Fix 2 — Business Segment OPERATES_IN re-routing
        # When a chunk yields a canonical segment name (e.g. "Loews Hotels",
        # "Boardwalk Pipeline") Agent 3 sometimes uses that segment as the
        # OPERATES_IN source instead of the legal subsidiary.  OPERATES_IN only
        # allows Parent or Subsidiary sources, so we handle two sub-cases:
        #
        #   2a: Dual-type conflict — same entity name registered as both Subsidiary
        #       and Business Segment (e.g. "CNA Financial Corporation").
        #       entity_type_map last-write-wins may have set it to Business Segment.
        #       If all_entities contains it as a Subsidiary anywhere, Subsidiary wins.
        #
        #   2b: Short segment name — "Loews Hotels", "Boardwalk Pipeline" etc.
        #       These are canonical segment names distinct from the subsidiary's
        #       legal name.  Use _normalize_sub matching to find the real subsidiary.
        if rel_type == "OPERATES_IN" and src_type == "Business Segment":
            # 2a: dual-type conflict — same name exists as Subsidiary in all_entities
            if any(e["name"] == src and e["type"] == "Subsidiary" for e in all_entities):
                info(f"Dual-type resolved: '{src}' treated as Subsidiary for OPERATES_IN")
                src_type = "Subsidiary"
            else:
                # 2b: short segment name — find matching subsidiary by normalised name
                src_norm = _normalize_sub(src)
                matched_sub = next(
                    (name for name in entity_names
                     if entity_type_map.get(name) == "Subsidiary"
                     and src_norm
                     and _normalize_sub(name)
                     and (src_norm in _normalize_sub(name) or _normalize_sub(name) in src_norm)),
                    None
                )
                if matched_sub:
                    info(f"Segment OPERATES_IN source re-routed: '{src}' → '{matched_sub}'")
                    rel["source"] = matched_sub
                    src = matched_sub
                    src_type = "Subsidiary"
                else:
                    warn(f"Dropped OPERATES_IN: source '{src}' is [Business Segment] and no matching subsidiary found")
                    log_error(run_timestamp, chunk_id, f"Invalid OPERATES_IN: source='{src}' type=[{src_type}]")
                    continue

        if rel_type == "OPERATES_IN" and src_type not in ("Parent", "Subsidiary"):
            warn(f"Dropped OPERATES_IN: source '{src}' is [{src_type}]  -  must be Parent or Subsidiary")
            log_error(run_timestamp, chunk_id, f"Invalid OPERATES_IN: source='{src}' type=[{src_type}]")
            continue
        tgt_type = entity_type_map.get(tgt, "")
        if rel_type == "OPERATES_IN" and tgt_type != "Geography":
            warn(f"Dropped OPERATES_IN: target '{tgt}' is [{tgt_type}]  -  must be Geography")
            log_error(run_timestamp, chunk_id, f"Invalid OPERATES_IN: target='{tgt}' type=[{tgt_type}]")
            continue
        if rel_type == "GENERATED" and src_type not in ("Business Segment", "Subsidiary"):
            warn(f"Dropped GENERATED: source '{src}' is [{src_type}]  -  must be Business Segment or Subsidiary")
            log_error(run_timestamp, chunk_id, f"Invalid GENERATED: source='{src}' type=[{src_type}]")
            continue

        rel["chunk_id"]    = chunk_id
        rel["file"]        = file
        rel["page_number"] = page_number
        valid.append(rel)

    return valid
