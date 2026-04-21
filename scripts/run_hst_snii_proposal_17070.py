#!/usr/bin/env python3
"""
Fetch HST proposal 17070 target table from MAST Archive HTML, dedupe by target
name, then run hst123 per object under a dedicated work directory.

Default base directory: /data/ckilpatrick/hst/supernovae/SNII/<TargetName>
"""
from __future__ import annotations

import argparse
import html
import os
import re
import shlex
import subprocess
import sys
import urllib.request

from astropy.coordinates import SkyCoord
import astropy.units as u

DEFAULT_URL = "https://archive.stsci.edu/proposal_search.php?id=17070&mission=hst"
DEFAULT_BASE = "/data/ckilpatrick/hst/supernovae/SNII"


def _cell_text(td_fragment: str) -> str:
    t = re.sub(r"<[^>]+>", " ", td_fragment)
    t = html.unescape(t)
    return " ".join(t.split()).strip()


def parse_archive_targets(page_html: str) -> list[tuple[str, str, str]]:
    """Return list of (target_name, ra_sex, dec_sex) from proposal_search tbody."""
    m = re.search(r"<tbody>(.*?)</tbody>", page_html, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError("No <tbody> found in archive HTML (table layout changed?)")
    body = m.group(1)
    out: list[tuple[str, str, str]] = []
    for trm in re.finditer(r"<tr>(.*?)</tr>", body, flags=re.DOTALL | re.IGNORECASE):
        tr = trm.group(1)
        if "pageexample" in tr or "stdads_mark" not in tr:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.DOTALL | re.IGNORECASE)
        if len(tds) < 5:
            continue
        target = _cell_text(tds[2])
        ra_s = _cell_text(tds[3])
        dec_s = _cell_text(tds[4])
        if not target or not ra_s or not dec_s:
            continue
        if target.casefold() == "target name":
            continue
        out.append((target, ra_s, dec_s))
    return out


def sexagesimal_to_deg(ra_s: str, dec_s: str) -> tuple[float, float]:
    s = f"{ra_s} {dec_s}"
    c = SkyCoord(s, unit=(u.hourangle, u.deg), frame="icrs")
    return float(c.ra.deg), float(c.dec.deg)


def unique_targets(rows: list[tuple[str, str, str]]) -> list[tuple[str, float, float]]:
    """One entry per target name; warn on coordinate mismatches."""
    best: dict[str, tuple[float, float, str, str]] = {}
    for name, ra_s, dec_s in rows:
        ra_d, dec_d = sexagesimal_to_deg(ra_s, dec_s)
        if name not in best:
            best[name] = (ra_d, dec_d, ra_s, dec_s)
            continue
        old_ra, old_dec, old_rs, old_ds = best[name]
        if abs(old_ra - ra_d) > 1e-6 or abs(old_dec - dec_d) > 1e-6:
            print(
                f"warning: duplicate target {name!r} with different coords "
                f"({old_rs} {old_ds} vs {ra_s} {dec_s}); keeping first",
                file=sys.stderr,
            )
    return [(n, best[n][0], best[n][1]) for n in sorted(best)]


def fs_safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-+]", "_", name)


def build_hst123_argv(
    ra_deg: float,
    dec_deg: float,
    work_dir: str,
    object_name: str,
    extra: list[str],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "hst123",
        str(ra_deg),
        str(dec_deg),
        "--work-dir",
        work_dir,
        "--object",
        object_name,
        "--download",
        "--drizzle-all",
        "--align-with",
        "jhat",
        "--run-dolphot",
        "--scrape-dolphot",
        "--scrape-all",
        "--scrape-radius",
        "10",
        *extra,
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL, help="MAST proposal_search URL")
    ap.add_argument(
        "--base-dir",
        default=DEFAULT_BASE,
        help="Parent directory for per-target work dirs",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned hst123 commands without running",
    )
    ap.add_argument(
        "extra_hst123_args",
        nargs="*",
        help="Extra arguments forwarded to each hst123 invocation (e.g. --token …)",
    )
    args = ap.parse_args()

    print(f"fetching {args.url}", file=sys.stderr)
    with urllib.request.urlopen(args.url, timeout=120) as resp:
        page = resp.read().decode(resp.headers.get_content_charset() or "utf-8", errors="replace")

    rows = parse_archive_targets(page)
    targets = unique_targets(rows)
    print(f"parsed {len(rows)} table rows, {len(targets)} unique targets", file=sys.stderr)

    for name, ra_deg, dec_deg in targets:
        safe = fs_safe_name(name)
        work_dir = os.path.join(args.base_dir, safe)
        argv = build_hst123_argv(ra_deg, dec_deg, work_dir, name, args.extra_hst123_args)
        if args.dry_run:
            print(shlex_join(argv))
            continue
        os.makedirs(work_dir, exist_ok=True)
        print(f"=== {name} -> {work_dir}", file=sys.stderr)
        subprocess.run(argv, check=True)
    return 0


def shlex_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


if __name__ == "__main__":
    raise SystemExit(main())
