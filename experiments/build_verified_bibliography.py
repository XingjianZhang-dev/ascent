#!/usr/bin/env python3
"""Build an arXiv-verified bibliography fragment for the manuscript.

Every entry is read from the citation metadata on the corresponding official
arXiv abstract page.  The output deliberately uses stable identifier-based
keys so manuscript citations remain unchanged if title capitalization changes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path


# Existing hand-curated entries in paper/references.bib are excluded here.
# The retained set covers architecture, long-context evaluation, retrieval,
# memory, prompt compression, positional extrapolation, efficient inference,
# scaling, model provenance, and confirmatory evaluation.
ARXIV_IDS = (
    "1706.03762", "1901.02860", "1911.05507", "1410.3916",
    "1503.08895", "1410.5401", "1605.06065", "1907.05242",
    "2004.05150", "2007.14062", "2001.04451", "2009.14794",
    "2006.04768", "2102.03902", "2003.05997", "2004.08483",
    "2312.00752", "2405.21060", "2302.10866", "2305.13048",
    "2307.08621", "2306.07174", "2310.08560", "2305.10250",
    "2005.11401", "2002.08909", "2004.04906", "1911.00172",
    "2208.03299", "2301.12652", "2310.11511", "2401.15884",
    "2401.18059", "2212.10496", "2112.09118", "2004.12832",
    "2007.01282", "2112.01488", "2310.05736", "2310.06839",
    "2304.12102", "2307.03172", "2308.14508", "2307.11088",
    "2402.13718", "2407.11963", "2402.13753", "2309.00071",
    "2306.15595", "2104.09864", "2108.12409", "2205.14135",
    "2307.08691", "2309.06180", "2306.14048", "2309.17453",
    "2404.14469", "2406.02069", "2406.10774", "2305.17118",
    "2402.04617", "2401.03462", "2308.16137", "2206.07682",
    "2102.01293", "2305.16264", "2405.04324", "2302.13971",
    "2307.09288", "2407.21783", "1502.05698", "2011.04006",
    "2206.04615", "2211.17192", "2302.01318", "2310.01889",
    "2309.12307", "2407.02490", "2109.08668", "2202.07856",
)


META_RE = re.compile(
    r'<meta name="citation_([^"\s]+)" content="([^"]*)"\s*/?>',
    re.IGNORECASE,
)


def cite_key(arxiv_id: str) -> str:
    return "a" + re.sub(r"\D", "", arxiv_id)


def fetch_metadata(arxiv_id: str, attempts: int = 4) -> dict[str, object]:
    url = f"https://arxiv.org/abs/{arxiv_id}"
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "ASCENT-reference-audit/1.0"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8", "replace")
            pairs = [(name, html.unescape(value)) for name, value in META_RE.findall(body)]
            authors = [value for name, value in pairs if name == "author"]
            scalar = {name: value for name, value in pairs if name != "author"}
            title = scalar.get("title")
            date = scalar.get("date")
            observed_id = scalar.get("arxiv_id")
            if not title or not date or observed_id != arxiv_id or not authors:
                raise RuntimeError(f"incomplete citation metadata for {arxiv_id}")
            return {
                "arxiv_id": arxiv_id,
                "key": cite_key(arxiv_id),
                "title": title,
                "authors": authors,
                "year": int(str(date)[:4]),
                "url": url,
            }
        except Exception as error:  # retry only transport/transient parse failures
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"unable to verify {arxiv_id}: {last_error}")


def latex_escape(value: str) -> str:
    # BibTeX in the IEEE/Tectonic toolchain is not reliably UTF-8 clean.
    # Retain the exact Unicode metadata in the verification manifest, but emit
    # a conservative ASCII transliteration in the .bib fragment.
    special = str.maketrans({
        "Ł": "L", "ł": "l", "Đ": "D", "đ": "d", "Ð": "D", "ð": "d",
        "Þ": "Th", "þ": "th", "Æ": "AE", "æ": "ae", "Œ": "OE",
        "œ": "oe", "Ø": "O", "ø": "o", "ß": "ss",
    })
    value = value.replace("–", "--").replace("—", "---")
    value = unicodedata.normalize("NFKD", value.translate(special))
    value = value.encode("ascii", "ignore").decode("ascii")
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def render_entry(item: dict[str, object]) -> str:
    authors = item["authors"]
    assert isinstance(authors, list)
    author_text = " and ".join(str(author) for author in authors)
    return "\n".join(
        (
            f"@article{{{item['key']},",
            f"  author  = {{{latex_escape(author_text)}}},",
            f"  title   = {{{latex_escape(str(item['title']))}}},",
            f"  journal = {{arXiv preprint arXiv:{item['arxiv_id']}}},",
            f"  year    = {{{item['year']}}},",
            f"  url     = {{{item['url']}}}",
            "}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(fetch_metadata, ARXIV_IDS))
    if len(records) < 80 or len({record["key"] for record in records}) != len(records):
        raise RuntimeError("verified reference count or key uniqueness gate failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(render_entry(record) for record in records) + "\n")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"count": len(records), "records": records}, indent=2))


if __name__ == "__main__":
    main()
