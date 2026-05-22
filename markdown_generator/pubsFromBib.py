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
import os
import re
import unicodedata
from urllib.parse import quote_plus
from typing import Dict, List, Tuple

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


def build_detail_rows(entry_type: str, authors: str, venue: str, year: str, fields: Dict[str, str]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    rows.append(("Publication type", TYPE_LABELS.get(entry_type, entry_type.title())))
    if authors:
        rows.append(("Authors", authors))
    if venue:
        rows.append(("Venue", venue))
    if year:
        rows.append(("Year", year))
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
    parser.add_argument("--keep-duplicates", action="store_true", help="Keep duplicate titles")
    args = parser.parse_args()

    if not os.path.exists(args.bib):
        raise SystemExit(f"BibTeX file not found: {args.bib}")

    os.makedirs(args.out, exist_ok=True)

    parser_bib = bibtex.Parser()
    bib_text = load_bibtex_text(args.bib)
    bibdata = parser_bib.parse_string(bib_text)

    entries: Dict[str, Tuple[str, object]] = {}
    for key, entry in bibdata.entries.items():
        raw_title = entry.fields.get("title", "")
        normalized_title = re.sub(r"[^a-z0-9]+", "", normalize_ascii(raw_title).lower())
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

        authors_text = format_authors(entry)
        venue_text = pick_venue(fields)

        category = CATEGORY_MAP.get(entry_type, "manuscripts")
        citation_text = build_citation(authors_text, title_text, venue_text, pub_year, fields)

        excerpt_text = ""
        if fields.get("abstract"):
            excerpt_text = latex_to_text(fields.get("abstract", ""))
        elif fields.get("note"):
            excerpt_text = latex_to_text(fields.get("note", ""))

        paper_url = fields.get("url", "")
        if not paper_url and fields.get("doi"):
            paper_url = f"https://doi.org/{latex_to_text(fields.get('doi'))}"

        md = f"---\ntitle: \"{escape_yaml(title_text)}\"\n"
        md += "collection: publications"
        md += f"\ncategory: {category}"
        md += f"\npermalink: /publication/{html_filename}"
        if excerpt_text.strip():
            md += f"\nexcerpt: '{escape_yaml(excerpt_text)}'"
        md += f"\ndate: {pub_date}"
        md += f"\nvenue: '{escape_yaml(venue_text)}'"
        if paper_url:
            md += f"\npaperurl: '{paper_url}'"
        if citation_text:
            md += f"\ncitation: '{escape_yaml(citation_text)}'"
        md += "\n---\n"

        detail_rows = build_detail_rows(entry_type, authors_text, venue_text, pub_year, fields)
        if detail_rows:
            md += "\n| Field | Value |\n| --- | --- |\n"
            for label, value in detail_rows:
                md += f"| {escape_table_value(label)} | {escape_table_value(value)} |\n"

        if excerpt_text.strip():
            md += f"\n{escape_yaml(excerpt_text)}\n"

        if paper_url:
            md += f"\n[Access paper here]({paper_url}){{:target=\"_blank\"}}\n"
        else:
            query = quote_plus(title_text)
            md += f"\nUse [Google Scholar](https://scholar.google.com/scholar?q={query}){{:target=\"_blank\"}} for full citation.\n"

        bibtex_str = writer.to_string(BibliographyData(entries={bib_id: entry}))
        md += "\n**BibTeX**\n\n```bibtex\n" + bibtex_str.strip() + "\n```\n"

        with open(os.path.join(args.out, md_filename), "w", encoding="utf-8") as file:
            file.write(md)


if __name__ == "__main__":
    main()
