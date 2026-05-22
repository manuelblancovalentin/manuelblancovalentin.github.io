#!/usr/bin/env python3
# coding: utf-8
"""
Generate publication markdown files for AcademicPages from a BibTeX file.

Usage:
  python3 pubsFromBib.py --bib /path/to/citations.bib --out ../_publications
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urlencode
from typing import Dict, List, Tuple, Optional

from pybtex.database import BibliographyData
from pybtex.database.input import bibtex
from pybtex.database.output.bibtex import Writer
from pybtex.richtext import Text

MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}

CATEGORY_MAP = {
    "book": "books",
    "inbook": "books",
    "incollection": "books",
    "booklet": "books",
    "inproceedings": "conferences",
    "proceedings": "conferences",
    "conference": "conferences",
}

TYPE_LABELS = {
    "article": "Journal article",
    "inproceedings": "Conference paper",
    "proceedings": "Conference proceedings",
    "phdthesis": "PhD thesis",
    "mastersthesis": "Master's thesis",
    "techreport": "Technical report",
    "misc": "Miscellaneous",
    "book": "Book",
    "inbook": "Book section",
    "incollection": "Book chapter",
    "booklet": "Booklet",
}

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

TAG_KEYWORDS = [
    ("hls4ml", "hls"),
    ("hls", "hls"),
    ("fpga", "fpga"),
    ("fpga", "hardware"),
    ("reconfigurable", "reconfigurable"),
    ("asic", "asic"),
    ("asic", "hardware"),
    ("hardware", "hardware"),
    ("sensor", "sensors"),
    ("pixel", "pixel"),
    ("detector", "detectors"),
    ("quantum", "quantum"),
    ("cryogenic", "cryogenic"),
    ("deep learning", "deep-learning"),
    ("neural network", "neural-networks"),
    ("machine learning", "machine-learning"),
    ("computer vision", "computer-vision"),
    ("image", "imaging"),
    ("borehole", "geoscience"),
    ("geological", "geoscience"),
    ("petroleum", "geoscience"),
    ("lensing", "astrophysics"),
    ("arxiv", "arxiv"),
    ("patent", "patent"),
    ("thesis", "thesis"),
]


def latex_to_text(value: str) -> str:
    if value is None:
        return ""
    try:
        return Text.from_latex(str(value)).render_as("text")
    except Exception:
        return str(value)


def escape_yaml(value: str) -> str:
    text = latex_to_text(value).replace("\n", " ").strip()
    text = text.replace("&", "&amp;").replace('"', "&quot;").replace("'", "&apos;")
    return text.encode("ascii", "xmlcharrefreplace").decode("ascii")


def escape_table_value(value: str) -> str:
    return escape_yaml(value).replace("|", "\\|")


def normalize_ascii(value: str) -> str:
    text = latex_to_text(value)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def clean_title_text(value: str) -> str:
    text = latex_to_text(value)
    text = text.replace("\\backslash", "")
    text = re.sub(r"\$+", "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    text = re.sub(r"\btextit\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_ascii(value).lower())


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_ascii(value).lower())


def truncate_text(text: str, limit: int = 280) -> str:
    trimmed = text.strip()
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[: limit - 1].rstrip() + "…"


def fetch_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def fetch_scholar_citations(profile_url: str) -> Dict[str, int]:
    citations: Dict[str, int] = {}
    parsed = urllib.parse.urlparse(profile_url)
    query = urllib.parse.parse_qs(parsed.query)
    user = query.get("user", [None])[0]
    if not user:
        raise ValueError("Scholar profile URL missing user parameter.")

    cstart = 0
    pagesize = 100
    while True:
        page_url = f"https://scholar.google.com/citations?user={user}&hl=en&cstart={cstart}&pagesize={pagesize}"
        html_text = fetch_url(page_url)
        rows = re.findall(r'<tr class="gsc_a_tr".*?>.*?</tr>', html_text, re.DOTALL)
        if not rows:
            break
        for row in rows:
            title_match = re.search(r'class="gsc_a_at"[^>]*>(.*?)</a>', row, re.DOTALL)
            if not title_match:
                continue
            title = html.unescape(title_match.group(1)).strip()
            cite_match = re.search(r'class="gsc_a_ac[^"]*"[^>]*>(.*?)</a>', row, re.DOTALL)
            cite_text = cite_match.group(1).strip() if cite_match else "0"
            cite_text = re.sub(r"[^0-9]", "", cite_text)
            count = int(cite_text) if cite_text else 0
            key = normalize_title_key(title)
            if key:
                citations[key] = max(count, citations.get(key, 0))
        if len(rows) < pagesize:
            break
        cstart += pagesize
        time.sleep(1)
    return citations


def extract_arxiv_id(fields: Dict[str, str]) -> Optional[str]:
    candidates = [
        fields.get("eprint", ""),
        fields.get("journal", ""),
        fields.get("note", ""),
        fields.get("url", ""),
        fields.get("title", ""),
        fields.get("abstract", ""),
    ]
    for candidate in candidates:
        text = latex_to_text(candidate)
        # Try arXiv URL format: arxiv.org/abs/XXXX.XXXXX or arxiv.org/abs/category/XXXXXXX
        match = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+|[a-z\-]+/[0-9]+)", text, re.IGNORECASE)
        if match:
            return match.group(1)
        # Try arXiv: prefix
        match = re.search(r"arxiv:\s*([0-9]+\.[0-9]+|[a-z\-]+/[0-9]+)", text, re.IGNORECASE)
        if match:
            return match.group(1)
        # Try standalone arXiv ID
        match = re.search(r"\b([0-9]+\.[0-9]{4,5})\b", text)
        if match and "arxiv" in text.lower():
            return match.group(1)
    return None


def extract_doi(fields: Dict[str, str]) -> Optional[str]:
    if fields.get("doi"):
        return latex_to_text(fields.get("doi")).strip()
    url_text = latex_to_text(fields.get("url", ""))
    match = re.search(r"doi\.org/([^\s]+)", url_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def lookup_doi_by_title(title: str, authors: str) -> Optional[str]:
    """Look up DOI by title via Crossref API."""
    # First try exact title match
    query = urlencode({"query.title": title, "rows": 5})
    url = f"https://api.crossref.org/works?{query}"
    try:
        response = fetch_url(url)
        payload = json.loads(response)
        items = payload.get("message", {}).get("items", [])
        for item in items:
            item_title = " ".join(item.get("title", []))
            # Allow fuzzy match if 80%+ of title words are present
            if (normalize_title_key(item_title) == normalize_title_key(title) or
                _fuzzy_title_match(title, item_title)):
                return item.get("DOI")
        return None
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"WARNING: Crossref DOI lookup failed for '{title}': {exc}")
        return None


def _fuzzy_title_match(title1: str, title2: str, threshold: float = 0.8) -> bool:
    """Check if titles match with fuzzy matching."""
    words1 = set(normalize_ascii(w).lower() for w in re.split(r"\W+", title1) if len(w) > 2)
    words2 = set(normalize_ascii(w).lower() for w in re.split(r"\W+", title2) if len(w) > 2)
    if not words1 or not words2:
        return False
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union >= threshold


def fetch_crossref_abstract(doi: str) -> Optional[str]:
    url = f"https://api.crossref.org/works/{doi}"
    try:
        response = fetch_url(url)
        payload = json.loads(response)
        abstract = payload.get("message", {}).get("abstract")
        if not abstract:
            return None
        abstract = re.sub(r"<[^>]+>", "", abstract)
        return html.unescape(abstract).strip()
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"WARNING: Crossref abstract fetch failed for DOI {doi}: {exc}")
        return None


def fetch_arxiv_abstract(arxiv_id: str) -> Optional[str]:
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        response = fetch_url(url)
        root = ET.fromstring(response)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", namespace)
        if entry is None:
            return None
        summary = entry.findtext("atom:summary", default="", namespaces=namespace)
        return " ".join(summary.split()).strip()
    except (urllib.error.URLError, ET.ParseError) as exc:
        print(f"WARNING: arXiv abstract fetch failed for {arxiv_id}: {exc}")
        return None


def derive_tags(title: str, venue: str, entry_type: str, note: str, doi: Optional[str], arxiv_id: Optional[str]) -> List[str]:
    tag_set: List[str] = []
    haystack = f"{title} {venue} {note} {entry_type}".lower()
    for keyword, tag in TAG_KEYWORDS:
        if keyword in haystack and tag not in tag_set:
            tag_set.append(tag)
    if arxiv_id and "arxiv" not in tag_set:
        tag_set.append("arxiv")
    if doi and "doi" not in tag_set:
        tag_set.append("doi")
    if entry_type in ("phdthesis", "mastersthesis") and "thesis" not in tag_set:
        tag_set.append("thesis")
    if entry_type == "misc" and "patent" in haystack and "patent" not in tag_set:
        tag_set.append("patent")
    if not tag_set:
        tag_set.append("research")
    return tag_set


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def author_rank(authors: List[str], variants: List[str]) -> int:
    normalized_variants = {normalize_name(v) for v in variants if v.strip()}
    for idx, author in enumerate(authors, start=1):
        if normalize_name(author) in normalized_variants:
            return idx
    return 999

def slugify(value: str) -> str:
    ascii_title = normalize_ascii(value).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title).strip("-")
    return slug or "untitled"


def parse_date(fields: Dict[str, str]) -> Tuple[str, str]:
    raw_year = str(fields.get("year", "")).strip()
    year_match = re.search(r"\d{4}", raw_year)
    year = year_match.group(0) if year_match else "1900"

    month = "01"
    day = "01"
    month_text = latex_to_text(fields.get("month", "")).lower()
    if month_text:
        month_match = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", month_text)
        if month_match:
            month = MONTHS[month_match.group(1)]
        else:
            month_digits = re.search(r"\b(\d{1,2})\b", month_text)
            if month_digits:
                month_value = int(month_digits.group(1))
                if 1 <= month_value <= 12:
                    month = f"{month_value:02d}"

    day_text = latex_to_text(fields.get("day", "")).lower()
    if not day_text and month_text:
        day_text = month_text
    day_digits = re.search(r"\b(\d{1,2})\b", day_text)
    if day_digits:
        day_value = int(day_digits.group(1))
        if 1 <= day_value <= 31:
            day = f"{day_value:02d}"

    return f"{year}-{month}-{day}", year


def format_person(person) -> str:
    first = " ".join([latex_to_text(n) for n in person.first_names + person.middle_names]).strip()
    last = " ".join([latex_to_text(n) for n in person.prelast_names + person.last_names + person.lineage_names]).strip()
    full = " ".join([first, last]).strip()
    return full if full else latex_to_text(str(person))


def format_authors(entry) -> str:
    authors = entry.persons.get("author", [])
    return ", ".join(format_person(author) for author in authors if format_person(author))


def list_authors(entry) -> List[str]:
    authors = entry.persons.get("author", [])
    return [format_person(author) for author in authors if format_person(author)]


def pick_venue(fields: Dict[str, str]) -> str:
    for key in ("journal", "booktitle", "publisher", "organization", "institution", "school"):
        value = fields.get(key)
        if value and str(value).strip():
            return latex_to_text(value)
    return "Unpublished"


def entry_score(entry) -> int:
    score_fields = [
        "year",
        "month",
        "day",
        "journal",
        "booktitle",
        "volume",
        "number",
        "pages",
        "publisher",
        "organization",
        "institution",
        "school",
        "doi",
        "url",
        "note",
        "abstract",
        "isbn",
        "issn",
    ]
    fields = entry.fields
    score = sum(1 for field in score_fields if fields.get(field))
    score += len(entry.persons.get("author", []))
    return score


def build_citation(authors: str, title: str, venue: str, year: str, fields: Dict[str, str]) -> str:
    parts: List[str] = []
    if authors:
        parts.append(authors)
    if year:
        parts.append(f"({year}).")
    if title:
        parts.append(f"\"{title}.\"")
    if venue and venue != "Unpublished":
        parts.append(f"<i>{venue}</i>.")

    volume = latex_to_text(fields.get("volume", "")).strip()
    number = latex_to_text(fields.get("number", "")).strip()
    pages = latex_to_text(fields.get("pages", "")).strip()
    issue_text = ""
    if volume:
        issue_text = volume
        if number:
            issue_text += f"({number})"
    elif number:
        issue_text = f"({number})"
    if pages:
        issue_text = f"{issue_text}, {pages}" if issue_text else pages
    if issue_text:
        parts.append(f"{issue_text}.")

    return " ".join(parts).strip()


def unique_path(out_dir: str, base_filename: str) -> str:
    filename = base_filename
    counter = 2
    while os.path.exists(os.path.join(out_dir, filename)):
        filename = base_filename.replace(".md", f"-{counter}.md")
        counter += 1
    return filename


def build_detail_rows(
    entry_type: str,
    authors: str,
    venue: str,
    year: str,
    fields: Dict[str, str],
    citation_count: int,
    tags: List[str],
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    rows.append(("Publication type", TYPE_LABELS.get(entry_type, entry_type.title())))
    if authors:
        rows.append(("Authors", authors))
    if venue:
        rows.append(("Venue", venue))
    if year:
        rows.append(("Year", year))
    rows.append(("Citations", str(citation_count)))
    if tags:
        rows.append(("Tags", ", ".join(tags)))
    for label, key in [
        ("Volume", "volume"),
        ("Number", "number"),
        ("Pages", "pages"),
        ("Publisher", "publisher"),
        ("Organization", "organization"),
        ("Institution", "institution"),
        ("School", "school"),
        ("ISBN", "isbn"),
        ("ISSN", "issn"),
        ("DOI", "doi"),
        ("URL", "url"),
        ("Note", "note"),
        ("Abstract", "abstract"),
    ]:
        value = fields.get(key)
        if value and str(value).strip():
            rows.append((label, latex_to_text(value)))
    return rows


def load_bibtex_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    seen: Dict[str, int] = {}
    lines: List[str] = []
    entry_re = re.compile(r"^\s*@\w+\s*{\s*([^,]+)\s*,", re.IGNORECASE)
    for line in text.splitlines():
        match = entry_re.match(line)
        if match:
            key = match.group(1)
            count = seen.get(key, 0)
            if count:
                new_key = f"{key}-{count + 1}"
                line = line.replace(key, new_key, 1)
            seen[key] = count + 1
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication markdown files from a BibTeX file.")
    parser.add_argument("--bib", required=True, help="Path to a BibTeX file (.bib)")
    parser.add_argument("--out", default="../_publications", help="Output directory for markdown files")
    parser.add_argument("--scholar", help="Google Scholar profile URL for citation counts")
    parser.add_argument(
        "--author",
        action="append",
        default=[],
        help="Author name variant to detect primary/secondary author ordering (repeatable)",
    )
    parser.add_argument("--keep-duplicates", action="store_true", help="Keep duplicate titles")
    parser.add_argument("--no-abstracts", action="store_true", help="Skip fetching abstracts from arXiv/DOI")
    args = parser.parse_args()

    if not os.path.exists(args.bib):
        raise SystemExit(f"BibTeX file not found: {args.bib}")

    os.makedirs(args.out, exist_ok=True)

    parser_bib = bibtex.Parser()
    bib_text = load_bibtex_text(args.bib)
    bibdata = parser_bib.parse_string(bib_text)

    citation_lookup: Dict[str, int] = {}
    if args.scholar:
        try:
            citation_lookup = fetch_scholar_citations(args.scholar)
        except Exception as exc:
            raise SystemExit(f"Failed to fetch Google Scholar citations: {exc}") from exc

    entries: Dict[str, Tuple[str, object]] = {}
    for key, entry in bibdata.entries.items():
        raw_title = entry.fields.get("title", "")
        normalized_title = normalize_title_key(raw_title)
        if not normalized_title:
            normalized_title = key.lower()
        if args.keep_duplicates:
            entries[f"{key}-{normalized_title}"] = (key, entry)
            continue
        if normalized_title not in entries:
            entries[normalized_title] = (key, entry)
            continue
        existing_key, existing_entry = entries[normalized_title]
        if entry_score(entry) > entry_score(existing_entry):
            entries[normalized_title] = (key, entry)

    writer = Writer()
    doi_cache: Dict[str, Optional[str]] = {}
    abstract_cache: Dict[str, Optional[str]] = {}
    for _, (bib_id, entry) in entries.items():
        fields = entry.fields
        entry_type = entry.type.lower()

        pub_date, pub_year = parse_date(fields)
        title_text = clean_title_text(fields.get("title", "")).strip()
        if not title_text:
            title_text = "Untitled"

        slug = slugify(title_text)
        md_filename = f"{pub_date}-{slug}.md"
        md_filename = unique_path(args.out, md_filename)
        html_filename = md_filename.replace(".md", "")

        authors_list = list_authors(entry)
        authors_text = ", ".join(authors_list)
        venue_text = pick_venue(fields)

        category = CATEGORY_MAP.get(entry_type, "manuscripts")
        citation_text = build_citation(authors_text, title_text, venue_text, pub_year, fields)

        note_text = latex_to_text(fields.get("note", "")).strip()
        abstract_text = latex_to_text(fields.get("abstract", "")).strip()

        arxiv_id = extract_arxiv_id(fields)
        doi = extract_doi(fields)

        paper_url = latex_to_text(fields.get("url", "")).strip()
        if not paper_url and is_url(latex_to_text(fields.get("journal", ""))):
            paper_url = latex_to_text(fields.get("journal", "")).strip()

        if not doi:
            cache_key = normalize_title_key(title_text)
            if cache_key not in doi_cache:
                doi_cache[cache_key] = lookup_doi_by_title(title_text, authors_text)
            doi = doi_cache[cache_key]

        if doi and not paper_url:
            paper_url = f"https://doi.org/{doi}"
        if arxiv_id and not paper_url:
            paper_url = f"https://arxiv.org/abs/{arxiv_id}"

        if not args.no_abstracts and not abstract_text:
            if arxiv_id:
                abstract_text = fetch_arxiv_abstract(arxiv_id) or ""
            if not abstract_text and doi:
                if doi not in abstract_cache:
                    abstract_cache[doi] = fetch_crossref_abstract(doi)
                abstract_text = abstract_cache.get(doi) or ""
            # If still no abstract, try to find DOI first if we haven't already
            if not abstract_text and not doi:
                cache_key = normalize_title_key(title_text)
                if cache_key not in doi_cache:
                    doi_cache[cache_key] = lookup_doi_by_title(title_text, authors_text)
                doi = doi_cache[cache_key]
                if doi:
                    if doi not in abstract_cache:
                        abstract_cache[doi] = fetch_crossref_abstract(doi)
                    abstract_text = abstract_cache.get(doi) or ""

        description_text = abstract_text or note_text
        # No fallback to title - leave blank if no abstract/note found
        excerpt_text = truncate_text(description_text) if description_text else ""

        if not paper_url:
            search_query = quote_plus(title_text)
            paper_url = f"https://www.google.com/search?q={search_query}"

        citation_count = citation_lookup.get(normalize_title_key(title_text), 0)
        current_rank = author_rank(authors_list, args.author)
        priority = 0 if current_rank <= 2 else 1
        date_int = int(pub_date.replace("-", "")) if pub_date else 0
        date_inverse = 99991231 - date_int
        citations_inverse = 999999 - min(citation_count, 999999)
        sort_key = f"{priority}-{citations_inverse:06d}-{date_inverse:08d}-{slug}"

        tags = derive_tags(title_text, venue_text, entry_type, note_text, doi, arxiv_id)

        md = f"---\ntitle: \"{escape_yaml(title_text)}\"\n"
        md += "collection: publications"
        md += f"\ncategory: {category}"
        md += f"\npermalink: /publication/{html_filename}"
        if excerpt_text.strip():
            md += f"\nexcerpt: '{escape_yaml(excerpt_text)}'"
        if description_text.strip():
            md += f"\ndescription: '{escape_yaml(description_text)}'"
        md += f"\ndate: {pub_date}"
        md += f"\nvenue: '{escape_yaml(venue_text)}'"
        md += f"\npaperurl: '{paper_url}'"
        md += f"\nauthor_rank: {current_rank}"
        md += f"\ncitation_count: {citation_count}"
        md += f"\nsort_key: '{sort_key}'"
        if tags:
            md += f"\ntags: [{', '.join(tags)}]"
        if citation_text:
            md += f"\ncitation: '{escape_yaml(citation_text)}'"
        md += "\n---\n"

        detail_rows = build_detail_rows(entry_type, authors_text, venue_text, pub_year, fields, citation_count, tags)
        if detail_rows:
            md += "\n| Field | Value |\n| --- | --- |\n"
            for label, value in detail_rows:
                md += f"| {escape_table_value(label)} | {escape_table_value(value)} |\n"

        if description_text.strip():
            md += f"\n**Abstract**\n\n{escape_yaml(description_text)}\n"

        md += f"\n[View publication]({paper_url}){{:target=\"_blank\"}}\n"

        bibtex_str = writer.to_string(BibliographyData(entries={bib_id: entry}))
        md += "\n**BibTeX**\n\n```bibtex\n" + bibtex_str.strip() + "\n```\n"

        with open(os.path.join(args.out, md_filename), "w", encoding="utf-8") as file:
            file.write(md)


if __name__ == "__main__":
    main()
