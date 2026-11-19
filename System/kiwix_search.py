import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup
from libzim.reader import Archive
from libzim.search import Query, Searcher
from libzim.suggestion import SuggestionSearcher

_ARCHIVE_LOCK = threading.Lock()
_ARCHIVE_CACHE: Dict[str, Tuple[int, Archive]] = {}


def _get_archives(zim_dir: str) -> List[Tuple[str, Archive]]:
    """Returns cached open libzim Archive handles for all valid .zim files in zim_dir."""
    root = Path(zim_dir)
    if not root.exists() or not root.is_dir():
        return []
    out: List[Tuple[str, Archive]] = []
    with _ARCHIVE_LOCK:
        for p in sorted(root.glob("*.zim")):
            key = str(p)
            try:
                mtime = p.stat().st_mtime_ns
                cached = _ARCHIVE_CACHE.get(key)
                if cached and cached[0] == mtime:
                    out.append((p.name, cached[1]))
                    continue
                arch = Archive(key)
                _ARCHIVE_CACHE[key] = (mtime, arch)
                out.append((p.name, arch))
            except Exception:
                continue
    return out


def _extract_article_snippet(arch: Archive, path: str, max_chars: int = 750) -> Tuple[str, str]:
    """Follows redirects and extracts clean readable text from a ZIM HTML entry."""
    entry = arch.get_entry_by_path(path)
    if entry.is_redirect:
        entry = entry.get_redirect_entry()
    title = entry.title or path
    raw = bytes(entry.get_item().content).decode("utf-8", errors="ignore")
    if "<" not in raw:
        return title, re.sub(r"\s+", " ", raw).strip()[:max_chars]

    soup = BeautifulSoup(raw, "html.parser")
    for bad in soup.select("script, style, nav, .navbox, .wikia-navbox, .mw-editsection, .toc, .printfooter"):
        bad.decompose()

    chunks: List[str] = []
    for node in soup.select("p, li, .pi-data, h2, h3"):
        txt = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if len(txt) >= 20:
            chunks.append(txt)
        if sum(len(c) for c in chunks) >= max_chars + 150:
            break

    text = "\n".join(chunks) if chunks else re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return title, text[:max_chars].strip()


def search_kiwix_zims(query: str, zim_dir: str, max_results: int = 4) -> List[Dict[str, str]]:
    """
    Searches all valid .zim archives in zim_dir using title suggestions and fulltext index.
    Prioritizes domain-specific archives alongside Wikipedia.
    """
    q = (query or "").strip()
    if not q or not zim_dir:
        return []
    archives = _get_archives(zim_dir)
    if not archives:
        return []

    candidates: List[Tuple[int, str, Archive, str]] = []
    seen: set = set()

    # 1. Fast title suggestion pass
    for name, arch in archives:
        if not arch.has_title_index:
            continue
        is_wiki = name.startswith("wikipedia_")
        try:
            res = SuggestionSearcher(arch).suggest(q)
            for path in res.get_results(0, 2):
                key = (name, path)
                if key not in seen:
                    seen.add(key)
                    candidates.append((1 if is_wiki else 0, name, arch, path))
        except Exception:
            continue

    # 2. Fulltext search pass
    for name, arch in archives:
        if not arch.has_fulltext_index:
            continue
        is_wiki = name.startswith("wikipedia_")
        try:
            res = Searcher(arch).search(Query().set_query(q))
            if res.get_estimated_matches() <= 0:
                continue
            for path in res.get_results(0, 2):
                key = (name, path)
                if key not in seen:
                    seen.add(key)
                    candidates.append((3 if is_wiki else 2, name, arch, path))
        except Exception:
            continue

    candidates.sort(key=lambda x: x[0])

    results: List[Dict[str, str]] = []
    used_resolved: set = set()
    for _, name, arch, path in candidates:
        if len(results) >= max_results:
            break
        try:
            title, snippet = _extract_article_snippet(arch, path)
            dedup_key = (name, title.lower())
            if dedup_key in used_resolved or len(snippet) < 40:
                continue
            used_resolved.add(dedup_key)
            results.append({"source": name, "title": title, "snippet": snippet})
        except Exception:
            continue
    return results
