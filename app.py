#!/usr/bin/env python3
"""
app.py — Ethnobotanical AI Assistant
====================================

Turns a JSON database of medicinal plants (scientific name, family, local names,
plant parts, traditional uses, preparation methods, source) into:

  * a fast keyword/semantic-lite search engine over the records
  * an "Ask AI" interface that grounds an LLM in the retrieved records (RAG)
  * a prompt builder so you can copy the grounded prompt into ANY chatbot
  * a browsable / filterable / exportable data explorer

Run as a web app:
    streamlit run app.py

Run as a terminal chat:
    python app.py                 # interactive REPL
    python app.py --ask "What is Kalanchoe densiflora used for?"
    python app.py --search "snake bite"

Data file resolution order:
    --data PATH  >  $PLANT_DATA  >  plants.json / data.json / ethnobotany.json
    >  first *.txt file next to this script

Optional LLM backend (any OpenAI-compatible endpoint):
    export OPENAI_API_KEY=sk-...
    export OPENAI_BASE_URL=https://api.openai.com/v1   # or Groq / OpenRouter / Ollama
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

APP_TITLE = "🌿 Ethnobotanical AI Assistant"

SAFETY_NOTE = (
    "This is documented traditional knowledge, provided for research and "
    "educational purposes only. Many of these plants and preparations are toxic, "
    "abortifacient, or otherwise dangerous. **Do not attempt any preparation "
    "without qualified medical guidance.**"
)

SYSTEM_PROMPT = """You are an ethnobotanical research assistant.

You answer questions using ONLY the RECORDS supplied in the user message.
Those records come from a documented source book of African medicinal plants.

Rules you must follow:
1. Base every statement on the supplied RECORDS. Never invent a plant, a use,
   a preparation, a local name, or a page number.
2. Always name the scientific name (and family) of the plant you are describing.
3. Cite the source page for each plant, e.g. "(Source book, p. 109)".
4. If the records do not contain the answer, say so plainly and suggest what
   the database does cover.
5. Do NOT give dosages, quantities, or step-by-step instructions for preparing
   or consuming any plant. Describe what the source documents, in the past
   tense / reported voice ("the source records that ...").
6. Never encourage self-treatment, abortion, or the use of poisons.
7. End every answer with a one-line safety caveat reminding the reader that
   this is documented traditional use and requires qualified medical guidance.

Formatting: short paragraphs or bullets. Be precise and compact.
"""

USER_PROMPT_TEMPLATE = """RECORDS
=======
{context}
=======

QUESTION: {question}

Answer using only the records above, following your rules."""

# Field weights for the retrieval scorer
FIELD_WEIGHTS: Dict[str, float] = {
    "scientificName": 6.0,
    "family": 1.5,
    "localNames": 3.0,
    "plantPartsUsed": 2.0,
    "traditionalUses": 4.0,
    "preparationMethods": 2.0,
    "notes": 1.0,
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "what", "which", "who", "with", "used", "use", "uses", "plant", "plants",
    "medicine", "medicinal", "how", "do", "does", "can", "i", "you", "me", "my",
    "there", "their", "about", "into", "any", "some", "all", "list", "tell",
}

# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #

DATA_FILENAMES = [
    "plants.json", "data.json", "ethnobotany.json", "medicinal_plants.json",
    "plants.txt", "data.txt",
]


def find_data_file(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate the dataset, trying several sensible locations."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"Data file not found: {p}")

    env = os.environ.get("PLANT_DATA")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    here = Path(__file__).resolve().parent
    for name in DATA_FILENAMES:
        p = here / name
        if p.exists():
            return p

    # fall back to the first .txt file sitting next to the script
    for p in sorted(here.glob("*.txt")):
        return p

    return None


def _as_str_list(v: Any) -> List[str]:
    if v is None or v == []:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        return [str(x) for x in v if x]
    return [str(v)]


def _pick_name_locale(n: Any) -> Dict[str, Any]:
    """Normalize one local-name entry from any schema."""
    if not isinstance(n, dict):
        return {"name": str(n), "language": None}
    lang = n.get("language") or n.get("language_tribe")
    if not lang:
        eg = n.get("ethnic_groups")
        lang = eg[0] if isinstance(eg, list) and eg else None
    return {
        "name": n.get("name") or n.get("vernacular_name") or n.get("localName") or "",
        "language": lang,
    }


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Map every known schema variant onto the canonical record shape.

    Handles:
      * json1.txt..json10.txt, json 4.txt  -> scientificName, localNames,
        plantPartsUsed, traditionalUses, preparationMethods, source{}
      * json11.txt                         -> scientific_name, local_names[{ethnic_groups}],
        uses (string), source_page, figure_reference
      * plants_combined.json               -> partsUsed, uses (list), preparations, sourcePage
    """
    out: Dict[str, Any] = {}
    out["scientificName"] = (
        rec.get("scientificName")
        or rec.get("scientific_name")
        or rec.get("species")
        or "Unknown species"
    )
    out["family"] = rec.get("family") or "Unknown family"

    ln = rec.get("localNames") or rec.get("local_names")
    if not isinstance(ln, list):
        names_in = _as_str_list(ln)
        names = [_pick_name_locale(x) for x in names_in]
    else:
        names = [_pick_name_locale(x) for x in ln if isinstance(x, dict)]
    out["localNames"] = names

    out["plantPartsUsed"] = _as_str_list(rec.get("plantPartsUsed") or rec.get("partsUsed"))
    out["traditionalUses"] = _as_str_list(rec.get("traditionalUses") or rec.get("uses"))
    out["preparationMethods"] = _as_str_list(
        rec.get("preparationMethods") or rec.get("preparations")
    )

    src = rec.get("source")
    if isinstance(src, dict):
        out["source"] = {
            "page": src.get("page"),
            "reference": src.get("reference", "Source book") or "Source book",
        }
    else:
        page = rec.get("source_page")
        if page is None:
            page = rec.get("sourcePage")
        out["source"] = {"page": page, "reference": "Source book"}

    out["notes"] = rec.get("notes")
    out["safetyNote"] = rec.get("safetyNote", SAFETY_NOTE)
    return out


def _unwrap(data: Any) -> List[Any]:
    """Return the list payload from a dict wrapper or pass through."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "data", "plants", "items", "entries", "compounds"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return []


def _parse_json_lenient(raw: str) -> Any:
    """Parse JSON, recovering truncated arrays (e.g. json5.txt) element by element."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        s = raw
        if s.startswith("["):
            s = s[1:]
        elif s.startswith("{"):
            try:
                obj, _ = decoder.raw_decode(s)
                return obj
            except json.JSONDecodeError:
                return []
        objs: List[Any] = []
        i, n = 0, len(s)
        while i < n:
            while i < n and s[i] in " \t\r\n,":
                i += 1
            if i >= n or s[i] != "{":
                break
            try:
                obj, end = decoder.raw_decode(s[i:])
                objs.append(obj)
                i += end
            except json.JSONDecodeError:
                nxt = s.find(",", i)
                if nxt == -1:
                    break
                i = nxt
        return objs


def _read_json_lenient(path: Path) -> Any:
    """Read a JSON file, recovering truncated arrays when possible."""
    return _parse_json_lenient(path.read_text(encoding="utf-8").strip())


def _read_json_lenient_text(text: str) -> Any:
    """Parse JSON from a string (e.g. an uploaded file's contents)."""
    return _parse_json_lenient(text)


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Read the JSON dataset (tolerant of a wrapping object)."""
    data = _read_json_lenient(path)
    data = _unwrap(data)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON structure in {path.name}: expected a list.")
    return [normalize_record(r) for r in data if isinstance(r, dict)]


def _merge_unique(target: List[Any], items: Iterable[Any], key: Any = None) -> None:
    seen = {key(i) if key else i for i in target}
    for i in items:
        k = key(i) if key else i
        if k not in seen:
            target.append(i)
            seen.add(k)


def _merge_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge records that refer to the same species, keeping all known detail."""
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for rec in records:
        n = normalize_record(rec) if not _looks_normalized(rec) else rec
        key = n["scientificName"]
        if key not in merged:
            merged[key] = {
                "scientificName": n["scientificName"],
                "family": n.get("family") or "Unknown family",
                "localNames": list(n.get("localNames") or []),
                "plantPartsUsed": list(n.get("plantPartsUsed") or []),
                "traditionalUses": list(n.get("traditionalUses") or []),
                "preparationMethods": list(n.get("preparationMethods") or []),
                "source": n.get("source", {"page": None, "reference": "Source book"}),
                "notes": n.get("notes"),
                "safetyNote": n.get("safetyNote", SAFETY_NOTE),
            }
            order.append(key)
        else:
            base = merged[key]
            if base["family"] in ("", "Unknown family") and n.get("family") not in ("", "Unknown family"):
                base["family"] = n["family"]
            _merge_unique(base["localNames"], n.get("localNames") or [], key=lambda x: (x.get("name"), x.get("language")))
            _merge_unique(base["plantPartsUsed"], n.get("plantPartsUsed") or [])
            _merge_unique(base["traditionalUses"], n.get("traditionalUses") or [])
            _merge_unique(base["preparationMethods"], n.get("preparationMethods") or [])
            if n.get("notes") and not base.get("notes"):
                base["notes"] = n["notes"]
            bpage = base.get("source", {}).get("page")
            npage = n.get("source", {}).get("page")
            if npage and not bpage:
                base["source"]["page"] = npage
    return [merged[k] for k in order]


def _looks_normalized(rec: Dict[str, Any]) -> bool:
    return all(k in rec for k in ("scientificName", "family")) and not (
        "scientific_name" in rec or "partsUsed" in rec
    )


def discover_data_files(directory: Path) -> Tuple[List[Path], Optional[Path]]:
    """Find all JSON/TXT 'notepad' data files next to the script."""
    directory = directory or Path(__file__).resolve().parent
    txt_files = sorted(directory.glob("*.txt"))
    json_files = sorted(directory.glob("*.json"))
    return txt_files, json_files[0] if json_files else None


def load_all_data(directory: Optional[Path] = None) -> Dict[str, Any]:
    """Load every local data source and return a unified dataset.

    Keys returned:
      * records        -- merged, de-duplicated plant records (canonical schema)
      * compounds      -- phytochemical compound entries (present_compounds)
      * disease_plants -- list[{disease, plants}]
      * vernacular     -- list[{vernacular_name, language_tribe, botanical_equivalent}]
      * medicinal_names-- list[str] known scientific names
      * sources        -- list[str] of file names actually loaded
    """
    directory = Path(directory or Path(__file__).resolve().parent)

    records_in: List[Dict[str, Any]] = []
    compounds: List[Dict[str, Any]] = []
    disease_plants: List[Dict[str, Any]] = []
    vernacular: List[Dict[str, Any]] = []
    medicinal_names: List[str] = []
    loaded: List[str] = []
    record_source_counts: Dict[str, int] = {}

    def _tally(name: str, rows: Any) -> None:
        if not isinstance(rows, list):
            return
        record_source_counts[name] = sum(
            1
            for r in rows
            if isinstance(r, dict)
            and (r.get("scientificName") or r.get("scientific_name") or r.get("species"))
        )

    # --- primary plant record sources (JSON files + txt notepads) ---
    # 1. plants_combined.json (the large, sparse-but-complete catalog)
    combined_path = directory / "plants_combined.json"
    if combined_path.exists():
        rows = _unwrap(_read_json_lenient(combined_path))
        records_in.extend(rows)
        loaded.append(combined_path.name)
        _tally(combined_path.name, rows)

    # 2. medicinal_plant.json (list of names) -> kept as a lookup
    mp_path = directory / "medicinal_plant.json"
    if mp_path.exists():
        rows = _unwrap(_read_json_lenient(mp_path))
        medicinal_names = [str(r) for r in rows if isinstance(r, str)] or [
            str(r) for r in rows if isinstance(r, dict) for _ in [r]
        ]
        loaded.append(mp_path.name)

    # 3. every json*.txt notepad (json1..json11, "json 4")
    for p in sorted(directory.glob("json*.txt")):
        if p.name in loaded:
            continue
        rows = _unwrap(_read_json_lenient(p))
        if rows:
            records_in.extend(rows)
            loaded.append(p.name)
            _tally(p.name, rows)

    # 4. vernacular & disease notepads
    vd_path = directory / "vernacular_data.json"
    if vd_path.exists():
        vernacular = _unwrap(_read_json_lenient(vd_path))
        loaded.append(vd_path.name)

    dp_path = directory / "disease_plants.json"
    if dp_path.exists():
        disease_plants = _unwrap(_read_json_lenient(dp_path))
        loaded.append(dp_path.name)

    # 5. phytochemical compounds notepad
    pc_path = directory / "present_compounds.txt"
    if pc_path.exists():
        payload = _read_json_lenient(pc_path)
        compounds = _unwrap(payload)
        loaded.append(pc_path.name)

    records = _merge_records(records_in) if records_in else []

    return {
        "records": records,
        "compounds": compounds,
        "disease_plants": disease_plants,
        "vernacular": vernacular,
        "medicinal_names": medicinal_names,
        "sources": loaded,
        "record_source_counts": record_source_counts,
    }


# --------------------------------------------------------------------------- #
#  Retrieval index
# --------------------------------------------------------------------------- #


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9']+", (text or "").lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


class PlantIndex:
    """A small weighted keyword index over the plant records."""

    def __init__(self, records: Sequence[Dict[str, Any]]):
        self.records: List[Dict[str, Any]] = list(records)
        self.docs: List[Dict[str, str]] = [self._doc(r) for r in self.records]

    # -- indexing ---------------------------------------------------------- #
    @staticmethod
    def _doc(rec: Dict[str, Any]) -> Dict[str, str]:
        local_names = " ".join(
            f'{n.get("name", "")} {n.get("language", "")}'
            for n in rec.get("localNames", [])
        )
        return {
            "scientificName": rec.get("scientificName", ""),
            "family": rec.get("family", ""),
            "localNames": local_names,
            "plantPartsUsed": " ".join(rec.get("plantPartsUsed", [])),
            "traditionalUses": " ".join(rec.get("traditionalUses", [])),
            "preparationMethods": " ".join(rec.get("preparationMethods", [])),
            "notes": rec.get("notes", "") or "",
        }

    # -- scoring ----------------------------------------------------------- #
    def _score(self, tokens: Sequence[str], doc: Dict[str, str]) -> float:
        lowered = {k: (v or "").lower() for k, v in doc.items()}
        score = 0.0
        for tok in tokens:
            for field, text in lowered.items():
                if not text:
                    continue
                weight = FIELD_WEIGHTS.get(field, 1.0)
                if tok in text.split():
                    score += weight
                elif len(tok) >= 4 and tok in text:
                    score += weight * 0.6
        return score

    # -- public API -------------------------------------------------------- #
    def search(
        self,
        query: str,
        top_k: int = 5,
        predicate=None,
        min_score: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return [(record, score), ...] sorted best-first."""
        tokens = tokenize(query)

        candidates = [
            (rec, doc) for rec, doc in zip(self.records, self.docs)
            if predicate is None or predicate(rec)
        ]

        if not tokens:
            # no usable query -> return the (filtered) records in file order
            return [(rec, 0.0) for rec, _ in candidates][:top_k]

        scored: List[Tuple[Dict[str, Any], float]] = []
        q_lower = query.lower().strip()
        for rec, doc in candidates:
            score = self._score(tokens, doc)
            if score <= 0:
                continue
            # phrase bonus
            joined = " ".join(doc.values()).lower()
            if len(q_lower) >= 5 and q_lower in joined:
                score += 8.0
            if score >= min_score:
                scored.append((rec, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0].get("scientificName", "")))
        return scored[:top_k]

    # -- convenience ------------------------------------------------------- #
    def families(self) -> List[str]:
        return sorted({r.get("family", "") for r in self.records if r.get("family")})

    def languages(self) -> List[str]:
        langs = set()
        for r in self.records:
            for n in r.get("localNames", []):
                if n.get("language"):
                    langs.add(n["language"])
        return sorted(langs)

    def plant_parts(self) -> List[str]:
        parts = set()
        for r in self.records:
            parts.update(r.get("plantPartsUsed", []))
        return sorted(parts)

    def uses(self) -> List[str]:
        uses = set()
        for r in self.records:
            uses.update(r.get("traditionalUses", []))
        return sorted(uses)


# --------------------------------------------------------------------------- #
#  Context / prompt construction
# --------------------------------------------------------------------------- #


def format_record(rec: Dict[str, Any], index: Optional[int] = None) -> str:
    """Render one record as compact text for an LLM context block."""
    head = f"[{index}] " if index is not None else ""
    lines = [f"{head}{rec.get('scientificName')} ({rec.get('family')})"]

    local = rec.get("localNames") or []
    if local:
        lines.append(
            "  Local names: "
            + "; ".join(f'{n.get("name")} [{n.get("language")}]' for n in local)
        )

    if rec.get("plantPartsUsed"):
        lines.append("  Parts used: " + ", ".join(rec["plantPartsUsed"]))
    if rec.get("traditionalUses"):
        lines.append("  Traditional uses: " + ", ".join(rec["traditionalUses"]))
    if rec.get("preparationMethods"):
        lines.append("  Preparation: " + " | ".join(rec["preparationMethods"]))

    src = rec.get("source") or {}
    if src:
        page = src.get("page")
        ref = src.get("reference", "Source")
        lines.append(f"  Source: {ref}" + (f", p. {page}" if page is not None else ""))

    if rec.get("notes"):
        lines.append(f"  Notes: {rec['notes']}")

    return "\n".join(lines)


def build_context(records: Sequence[Dict[str, Any]]) -> str:
    if not records:
        return "(no matching records found in the database)"
    return "\n\n".join(format_record(r, i + 1) for i, r in enumerate(records))


def build_prompt(question: str, records: Sequence[Dict[str, Any]]) -> str:
    """Full prompt (system + user) ready to paste into any chatbot."""
    return (
        SYSTEM_PROMPT.strip()
        + "\n\n"
        + USER_PROMPT_TEMPLATE.format(
            context=build_context(records), question=question.strip()
        )
    )


# --------------------------------------------------------------------------- #
#  Extractive fallback "answer" (no LLM required)
# --------------------------------------------------------------------------- #


def extractive_answer(question: str, records: Sequence[Dict[str, Any]]) -> str:
    if not records:
        return (
            "No records in the database match that question.\n\n"
            "Try a plant name (e.g. *Kalanchoe densiflora*), a family "
            "(e.g. *Cucurbitaceae*), a use (e.g. *snake bite*), or a language "
            "(e.g. *Luo*)."
        )

    out = [f"**{len(records)} matching record(s)** for: _{question.strip()}_\n"]
    for rec in records:
        out.append(format_record(rec).replace("\n", "\n> "))
        out.append("")
    out.append("---")
    out.append(SAFETY_NOTE)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
#  LLM backend
# --------------------------------------------------------------------------- #


def call_llm(
    question: str,
    records: Sequence[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> str:
    """Call any OpenAI-compatible chat endpoint. Raises on failure."""
    try:
        from openai import OpenAI  # imported lazily so it stays optional
    except ImportError as exc:
        raise RuntimeError(
            "The `openai` package is not installed. Run: pip install openai"
        ) from exc

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set OPENAI_API_KEY, or use the local "
            "extractive answer / copy the prompt into your own chatbot."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        timeout=timeout,
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    context=build_context(records), question=question.strip()
                ),
            },
        ],
    )
    return response.choices[0].message.content or "(empty response)"


# --------------------------------------------------------------------------- #
#  Shared "ask" pipeline
# --------------------------------------------------------------------------- #


def answer_question(
    index: PlantIndex,
    question: str,
    top_k: int = 5,
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve, then answer. Returns (answer_text, retrieved_records)."""
    hits = index.search(question, top_k=top_k)
    records = [rec for rec, _ in hits]

    if use_llm:
        try:
            return call_llm(
                question, records, model=model, api_key=api_key, base_url=base_url
            ), records
        except Exception as exc:  # noqa: BLE001 - surface any backend error
            fallback = extractive_answer(question, records)
            return f"⚠️ LLM unavailable ({exc}).\n\n---\n\n{fallback}", records

    return extractive_answer(question, records), records


# --------------------------------------------------------------------------- #
#  Charting helpers
# --------------------------------------------------------------------------- #


def _value_counts(items: Iterable[Any], top_n: int = 15) -> "pandas.DataFrame":
    import pandas as pd  # lazy

    counts = pd.Series(list(items)).value_counts().rename_axis("term").reset_index(name="count")
    counts = counts[counts["term"].notna() & (counts["term"].astype(str) != "")]
    return counts.head(top_n)


def _bar(counts_df, x="term", y="count", title="", orientation="v"):
    import plotly.express as px  # lazy

    fig = px.bar(
        counts_df,
        x=None if orientation == "h" else x,
        y=None if orientation == "v" else y,
        color=y,
        orientation=orientation,
        title=title,
        text_auto=".2s",
    )
    if orientation == "h":
        fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
    return fig


def chart_records_per_family(records: Sequence[Dict[str, Any]], top_n: int = 20):
    import pandas as pd

    fams = _value_counts([r.get("family", "") for r in records], top_n)
    if fams.empty:
        return None
    fams = fams.sort_values("count", ascending=True)
    return _bar(fams, x="count", y="term", title="Records per family", orientation="h")


def chart_top_uses(records: Sequence[Dict[str, Any]], top_n: int = 15):
    uses = _value_counts(
        (u for r in records for u in (r.get("traditionalUses") or [])), top_n
    )
    if uses.empty:
        return None
    uses = uses.sort_values("count", ascending=True)
    return _bar(uses, x="count", y="term", title="Top traditional uses", orientation="h")


def chart_records_per_page(records: Sequence[Dict[str, Any]]):
    pages = _value_counts(
        (r.get("source", {}).get("page") for r in records), 30
    )
    if pages.empty:
        return None
    pages["term"] = pages["term"].astype(str)
    pages = pages.sort_values("count", ascending=True)
    return _bar(pages, x="count", y="term", title="Records per source page", orientation="h")


def chart_languages(records: Sequence[Dict[str, Any]], top_n: int = 15):
    langs = _value_counts(
        (n.get("language") for r in records for n in (r.get("localNames") or [])),
        top_n,
    )
    if langs.empty:
        return None
    langs = langs.sort_values("count", ascending=True)
    return _bar(langs, x="count", y="term", title="Local-name languages", orientation="h")


def chart_compounds_per_family(compounds: Sequence[Dict[str, Any]], top_n: int = 20):
    families = _value_counts(
        (c.get("family", "") for c in compounds if c.get("family")), top_n
    )
    if families.empty:
        return None
    families = families.sort_values("count", ascending=True)
    return _bar(families, x="count", y="term", title="Compounds per family", orientation="h")


def chart_compounds_per_species(compounds: Sequence[Dict[str, Any]], top_n: int = 20):
    species = _value_counts(
        (c.get("species", "") for c in compounds if c.get("species")), top_n
    )
    if species.empty:
        return None
    species = species.sort_values("count", ascending=True)
    return _bar(species, x="count", y="term", title="Compounds per species", orientation="h")


def chart_mw_histogram(compounds: Sequence[Dict[str, Any]]):
    import pandas as pd
    import plotly.express as px

    weights = pd.to_numeric(
        pd.Series([c.get("molecularWeight") for c in compounds]), errors="coerce"
    ).dropna()
    if weights.empty:
        return None
    fig = px.histogram(
        weights,
        x=weights,
        nbins=40,
        title="Distribution of compound molecular weights",
        labels={"x": "Molecular weight", "y": "Count"},
    )
    fig.update_layout(showlegend=False)
    return fig


def chart_novelty(compounds: Sequence[Dict[str, Any]]):
    nov = _value_counts((c.get("novelty") for c in compounds), 20)
    if nov.empty:
        return None
    nov = nov.sort_values("count", ascending=True)
    return _bar(nov, x="term", y="count", title="Compound novelty breakdown", orientation="v")


def chart_diseases(disease_plants: Sequence[Dict[str, Any]], top_n: int = 20):
    plants_per_disease = (
        len(e.get("plants", [])) if isinstance(e, dict) else 0 for e in disease_plants
    )
    counts = _value_counts(
        (
            e.get("disease", "")
            for e in disease_plants
            if isinstance(e, dict) and e.get("disease")
        ),
        top_n,
    )
    if counts.empty:
        return None
    counts = counts.sort_values("count", ascending=True)
    return _bar(counts, x="count", y="term", title="Plants indexed per disease", orientation="h")


def chart_vernacular(vernacular: Sequence[Dict[str, Any]], top_n: int = 15):
    langs = _value_counts(
        (v.get("language_tribe", "") for v in vernacular if isinstance(v, dict)),
        top_n,
    )
    if langs.empty:
        return None
    langs = langs.sort_values("count", ascending=True)
    return _bar(langs, x="count", y="term", title="Vernacular names per language/tribe", orientation="h")


def _chart_records_per_source(source_counts: Mapping[str, int]):
    import pandas as pd

    if not source_counts:
        return None
    df = (
        pd.Series(source_counts, name="count")
        .rename_axis("term")
        .reset_index()
        .astype({"term": str, "count": int})
    )
    df = df.sort_values("count", ascending=True)
    return _bar(df, x="count", y="term", title="Raw records per source file", orientation="h")


def _empty_chart(st, title: str) -> None:
    st.info(f"No data available to chart: {title}.")


# --------------------------------------------------------------------------- #
#  Streamlit UI
# --------------------------------------------------------------------------- #


def _load_local_dataset(here: Path) -> Dict[str, Any]:
    """Load every local JSON/TXT source into a unified dataset."""
    try:
        return load_all_data(here)
    except Exception as exc:  # noqa: BLE001
        import streamlit as st  # noqa: F811
        st.error(f"Could not load local data files: {exc}")
        st.stop()


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Ethnobotanical AI Assistant", page_icon="🌿",
                       layout="wide")

    here = Path(__file__).resolve().parent

    # ---------- sidebar ---------- #
    with st.sidebar:
        st.title("🌿 Settings")

        if "dataset" not in st.session_state:
            dataset = _load_local_dataset(here)
            if not dataset["records"]:
                st.error(
                    "No dataset found. Place plant JSON/TXT files next to app.py, "
                    "or upload one below."
                )
                st.stop()
            st.session_state["dataset"] = dataset
            st.session_state["data_name"] = "all local sources"

        dataset = st.session_state["dataset"]
        records = dataset["records"]
        index = PlantIndex(records)

        st.caption(
            f"Dataset: **{st.session_state.get('data_name', '?')}** — "
            f"{len(records)} records · "
            f"{len(dataset.get('compounds', []))} compounds · "
            f"{len(dataset.get('disease_plants', []))} diseases · "
            f"{len(dataset.get('vernacular', []))} vernacular names"
        )
        with st.expander("Data sources loaded"):
            for s in dataset.get("sources", []):
                st.markdown(f"- `{s}`")

        uploaded = st.file_uploader(
            "Or upload an extra JSON/TXT dataset to merge",
            type=["json", "txt"],
            key="extra_uploader",
        )
        if uploaded is not None:
            try:
                payload = _read_json_lenient_text(
                    uploaded.read().decode("utf-8")
                )
                rows = _unwrap(payload)
                extra = _merge_records(rows)
                records = _merge_records(list(records) + extra)
                dataset["records"] = records
                index = PlantIndex(records)
                st.session_state["records"] = records
                st.session_state["data_name"] = f"{uploaded.name} (+ local)"
                st.success(f"Merged {len(extra)} records from {uploaded.name}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read uploaded file: {exc}")

        st.divider()
        st.subheader("AI backend")
        use_llm = st.checkbox(
            "Use an LLM",
            value=bool(os.environ.get("OPENAI_API_KEY")),
            help="Requires the `openai` package and an API key. "
                 "Uncheck for a purely local, extractive answer.",
        )
        model = st.text_input("Model", value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
        base_url = st.text_input(
            "Base URL (optional)",
            value=os.environ.get("OPENAI_BASE_URL", ""),
            help="Point at Groq, OpenRouter, Together, a local Ollama server, etc.",
        )
        api_key = st.text_input(
            "API key", value=os.environ.get("OPENAI_API_KEY", ""), type="password"
        )
        top_k = st.slider("Records retrieved per question", 1, 12, 5)

        st.divider()
        st.warning(SAFETY_NOTE)

    # ---------- header ---------- #
    st.title(APP_TITLE)
    st.caption(
        "Grounded question answering over a documented database of African "
        "medicinal plants. Answers are restricted to the records in the dataset."
    )

    tab_browse, tab_ask, tab_prompt, tab_charts, tab_data = st.tabs(
        ["🔎 Browse & Filter", "💬 Ask AI", "📋 Prompt Builder", "📈 Charts", "📊 Dataset"]
    )

    # ================= BROWSE ================= #
    with tab_browse:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            fam = st.multiselect("Family", index.families())
        with c2:
            lang = st.multiselect("Language", index.languages())
        with c3:
            part = st.multiselect("Plant part", index.plant_parts())
        with c4:
            use = st.multiselect("Traditional use", index.uses())

        q = st.text_input("Keyword search", placeholder="e.g. rheumatism, snake bite, Kalanchoe")

        def keep(rec: Dict[str, Any]) -> bool:
            if fam and rec.get("family") not in fam:
                return False
            if lang and not any(n.get("language") in lang for n in rec.get("localNames", [])):
                return False
            if part and not set(part) & set(rec.get("plantPartsUsed", [])):
                return False
            if use and not set(use) & set(rec.get("traditionalUses", [])):
                return False
            return True

        if q.strip():
            hits = index.search(q, top_k=200, predicate=keep)
            results = [rec for rec, _ in hits]
        else:
            results = [r for r in records if keep(r)]

        st.markdown(f"**{len(results)}** record(s)")

        for rec in results:
            with st.expander(f"{rec['scientificName']} — *{rec['family']}*"):
                left, right = st.columns([3, 2])
                with left:
                    if rec.get("localNames"):
                        st.markdown(
                            "**Local names:** "
                            + "; ".join(
                                f"{n.get('name')} ({n.get('language')})"
                                for n in rec["localNames"]
                            )
                        )
                    st.markdown("**Parts used:** " + (", ".join(rec["plantPartsUsed"]) or "—"))
                    st.markdown("**Traditional uses:** " + (", ".join(rec["traditionalUses"]) or "—"))
                with right:
                    st.markdown("**Preparation**")
                    for p in rec.get("preparationMethods", []) or ["—"]:
                        st.markdown(f"- {p}")
                    src = rec.get("source") or {}
                    st.caption(
                        f"Source: {src.get('reference', 'n/a')}"
                        + (f", p. {src['page']}" if src.get("page") is not None else "")
                    )
                if rec.get("notes"):
                    st.info(rec["notes"])
                st.caption(rec.get("safetyNote", SAFETY_NOTE))

        if results:
            st.download_button(
                "⬇️ Download results (JSON)",
                data=json.dumps(results, indent=2, ensure_ascii=False),
                file_name="filtered_plants.json",
                mime="application/json",
            )

    # ================= ASK ================= #
    with tab_ask:
        st.markdown(
            "Ask about a plant, a family, a use, a language, or a symptom. "
            "The assistant retrieves the most relevant records first and then "
            "answers **only** from them."
        )

        examples = [
            "What is Kalanchoe densiflora used for and how is it prepared?",
            "Which plants treat snake bites?",
            "List the Cucurbitaceae in the database with their uses.",
            "Which plants are used for rheumatism?",
            "What is Ipomoea pes-caprae used for?",
            "Which plants are documented as poisonous or hazardous?",
            "What plants are used to increase lactation?",
        ]
        st.markdown("**Try:** " + " · ".join(f"`{e}`" for e in examples[:4]))

        with st.form("ask_form", clear_on_submit=False):
            question = st.text_input(
                "Your question",
                placeholder="e.g. Which plants are used to treat stomach ache?",
            )
            submitted = st.form_submit_button("Ask", type="primary")

        if submitted and question.strip():
            with st.spinner("Retrieving records and composing an answer…"):
                answer, retrieved = answer_question(
                    index,
                    question,
                    top_k=top_k,
                    use_llm=use_llm,
                    model=model,
                    api_key=api_key,
                    base_url=base_url or None,
                )
            st.session_state["last_answer"] = answer
            st.session_state["last_retrieved"] = retrieved
            st.session_state["last_question"] = question

        if st.session_state.get("last_answer"):
            st.markdown("### Answer")
            st.markdown(st.session_state["last_answer"])

            with st.expander(
                f"📚 Retrieved records ({len(st.session_state.get('last_retrieved', []))})"
            ):
                for rec in st.session_state.get("last_retrieved", []):
                    st.markdown(
                        f"**{rec['scientificName']}** ({rec['family']}) — "
                        f"p. {(rec.get('source') or {}).get('page', 'n/a')}"
                    )
                    st.markdown(
                        "> "
                        + format_record(rec).replace("\n", "\n> ")
                    )
                    st.divider()

    # ================= PROMPT BUILDER ================= #
    with tab_prompt:
        st.markdown(
            "Build a grounded prompt and paste it into any chatbot "
            "(ChatGPT, Claude, Gemini, a local model…). No API key needed."
        )
        pq = st.text_area(
            "Question",
            value="Which plants are used for snake bites and how are they prepared?",
            height=90,
        )
        if st.button("Build prompt", type="primary"):
            hits = index.search(pq, top_k=top_k)
            recs = [r for r, _ in hits]
            st.session_state["built_prompt"] = build_prompt(pq, recs)
            st.session_state["built_records"] = recs

        if st.session_state.get("built_prompt"):
            st.caption(
                f"Prompt built from {len(st.session_state.get('built_records', []))} record(s)."
            )
            st.code(st.session_state["built_prompt"], language="markdown")
            st.download_button(
                "⬇️ Download prompt (.txt)",
                data=st.session_state["built_prompt"],
                file_name="grounded_prompt.txt",
                mime="text/plain",
            )

    # ================= CHARTS ================= #
    with tab_charts:
        recs = records
        cmps = dataset.get("compounds", [])
        dp = dataset.get("disease_plants", [])
        vd = dataset.get("vernacular", [])

        st.markdown(
            "Visualise the merged dataset built from every local JSON file and "
            "notepad. Choose a dimension below, or scroll through the gallery."
        )

        option = st.selectbox(
            "What to chart",
            [
                "Records per family",
                "Top traditional uses",
                "Records per source page",
                "Local-name languages",
                "Compounds per family",
                "Compounds per species",
                "Compound molecular-weight distribution",
                "Compound novelty breakdown",
                "Plants per disease",
                "Vernacular names per language/tribe",
                "Records per data source",
            ],
            index=0,
        )

        st.markdown("### Selected chart")
        if option == "Records per family":
            fig = chart_records_per_family(recs)
        elif option == "Top traditional uses":
            fig = chart_top_uses(recs)
        elif option == "Records per source page":
            fig = chart_records_per_page(recs)
        elif option == "Local-name languages":
            fig = chart_languages(recs)
        elif option == "Compounds per family":
            fig = chart_compounds_per_family(cmps)
        elif option == "Compounds per species":
            fig = chart_compounds_per_species(cmps)
        elif option == "Compound molecular-weight distribution":
            fig = chart_mw_histogram(cmps)
        elif option == "Compound novelty breakdown":
            fig = chart_novelty(cmps)
        elif option == "Plants per disease":
            fig = chart_diseases(dp)
        elif option == "Vernacular names per language/tribe":
            fig = chart_vernacular(vd)
        elif option == "Records per data source":
            fig = _chart_records_per_source(dataset.get("record_source_counts", {}))
        else:
            fig = None

        if fig is not None:
            st.plotly_chart(fig, width="stretch")
        else:
            _empty_chart(st, option)

        st.markdown("---")
        st.markdown("### Chart gallery")

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Plant records**")
            for fig, hdr in (
                (chart_records_per_family(recs), "Records per family"),
                (chart_top_uses(recs), "Top traditional uses"),
                (chart_records_per_page(recs), "Records per source page"),
                (chart_languages(recs), "Local-name languages"),
            ):
                if fig is not None:
                    st.subheader(hdr, divider="gray")
                    st.plotly_chart(fig, width="stretch", key=hdr)
                else:
                    _empty_chart(st, hdr)

        with cols[1]:
            st.markdown("**Compounds, diseases & vernacular**")
            for fig, hdr in (
                (chart_compounds_per_family(cmps), "Compounds per family"),
                (chart_compounds_per_species(cmps), "Compounds per species"),
                (chart_mw_histogram(cmps), "Molecular-weight distribution"),
                (chart_novelty(cmps), "Compound novelty"),
                (chart_diseases(dp), "Plants per disease"),
                (chart_vernacular(vd), "Vernacular names per language"),
                (_chart_records_per_source(dataset.get("record_source_counts", {})), "Records per data source"),
            ):
                if fig is not None:
                    st.subheader(hdr, divider="gray")
                    st.plotly_chart(fig, width="stretch", key=hdr)
                else:
                    _empty_chart(st, hdr)

    # ================= DATASET ================= #
    with tab_data:
        st.markdown("### Dataset overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Records", len(records))
        m2.metric("Families", len(index.families()))
        m3.metric("Languages", len(index.languages()))
        m4.metric("Distinct uses", len(index.uses()))

        st.markdown("#### Records per family")
        fam_counts: Dict[str, int] = {}
        for r in records:
            fam_counts[r.get("family", "?")] = fam_counts.get(r.get("family", "?"), 0) + 1
        st.bar_chart(
            {k: v for k, v in sorted(fam_counts.items(), key=lambda kv: -kv[1])}
        )

        st.markdown("#### Raw JSON")
        st.download_button(
            "⬇️ Download full dataset",
            data=json.dumps(records, indent=2, ensure_ascii=False),
            file_name="plants_export.json",
            mime="application/json",
        )
        st.json(records[:3], expanded=False)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def run_cli(args: argparse.Namespace) -> int:
    data_path = find_data_file(args.data)
    if data_path is None:
        print("❌ No dataset found. Use --data PATH or set $PLANT_DATA.", file=sys.stderr)
        return 2

    try:
        records = load_records(data_path)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    index = PlantIndex(records)
    print(f"🌿 Loaded {len(records)} records from {data_path.name}\n")

    # one-shot search
    if args.search:
        for rec, score in index.search(args.search, top_k=args.top_k):
            print(format_record(rec))
            print(f"  (score {score:.1f})\n")
        return 0

    # one-shot ask
    if args.ask:
        answer, _ = answer_question(
            index,
            args.ask,
            top_k=args.top_k,
            use_llm=not args.no_llm,
            model=args.model,
            base_url=args.base_url,
        )
        print(answer)
        print("\n" + "-" * 70)
        print(SAFETY_NOTE)
        return 0

    # interactive REPL
    print("Type a question, or a bare keyword to search.")
    print("Commands: :search <kw>   :record <name>   :quit\n")
    while True:
        try:
            line = input("🌿 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {":quit", ":q", "exit", "quit"}:
            break
        if line.startswith(":search "):
            for rec, _ in index.search(line[8:], top_k=args.top_k):
                print(format_record(rec), "\n")
            continue
        if line.startswith(":record "):
            hits = index.search(line[8:], top_k=1)
            print(format_record(hits[0][0]) if hits else "Not found.")
            print()
            continue

        answer, _ = answer_question(
            index,
            line,
            top_k=args.top_k,
            use_llm=not args.no_llm,
            model=args.model,
            base_url=args.base_url,
        )
        print("\n" + answer + "\n")
        print("-" * 70)
        print(SAFETY_NOTE + "\n")

    return 0


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #


def running_under_streamlit() -> bool:
    try:
        from streamlit.runtime import exists  # type: ignore
        return bool(exists())
    except Exception:  # noqa: BLE001
        return False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ethnobotanical AI Assistant over a JSON plant database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  streamlit run app.py\n"
            "  python app.py --search 'snake bite'\n"
            "  python app.py --ask 'What is Kalanchoe densiflora used for?'\n"
        ),
    )
    p.add_argument("--data", help="Path to the JSON dataset")
    p.add_argument("--ask", help="Ask one question and exit")
    p.add_argument("--search", help="Keyword search and exit")
    p.add_argument("--top-k", type=int, default=5, help="Records to retrieve (default 5)")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or None)
    p.add_argument("--no-llm", action="store_true",
                   help="Skip the LLM; return retrieved records only")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    if running_under_streamlit():
        run_streamlit_app()
        return 0
    return run_cli(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())