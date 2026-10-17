#!/usr/bin/env python3
"""
group_similar.py — distribute photos so that SIMILAR ones end up in DIFFERENT
folders.  Every output folder then holds only photos that are visibly DISTINCT
from one another (no two near-duplicates in the same folder).

This is the inverse of "collect each burst of repeated takes into one folder".
A photoshoot shoots a model standing / smiling / sitting again and again, so most
frames have near-twins.  Here those near-twins are SPREAD APART: similar photos
are forced into separate folders, and the unique shots are used as filler to keep
the folders evenly sized.

The effect is that the number of folders is driven UP by how many near-identical
takes exist: if some pose was shot 6 times, you get at least 6 folders so all 6
copies land in different places.

Similarity is measured 100% locally (no cloud, no cost, no API key) using a
blend of FOUR complementary, content-aware signals:

  1. pHash (perceptual hash) ......... coarse, rotation-normalised, cheap.
  2. Grayscale STRUCTURE correlation . the dominant signal: the whole frame is
                                       downscaled and contrast-normalised, so
                                       COMPOSITION is compared directly.
  3. Edge correlation ................ captures outlines / object boundaries.
  4. Colour correlation on a grid .... colour-aware detail (backdrop, outfit).

Two photos are treated as "alike" (and therefore MUST NOT share a folder) when
EITHER:
  * their COMBINED similarity — a weighted blend of the four measures above,
    weighted by --w-hash / --w-gray / --w-edge / --w-color — is
    >= --similar-threshold (default 0.60),  -- or
  * their grayscale structure correlation alone is
    >= --strong-gray-threshold (default 0.80) (near-identical composition).

These "alike" pairs form a CONFLICT GRAPH.  The folders are the colours of that
graph, found with DSATUR colouring so two alike photos never share a colour:

    build conflict graph  ->  DSATUR colouring  ->  (optional) balanced fill

DSATUR uses roughly the minimum number of colours, and that minimum equals the
size of the largest set of mutually-alike photos — so a 6-take burst guarantees
6 folders.  By default every NON-alike ("unique") photo is simply allowed to
pool together, so you get one folder holding all the genuinely distinct shots
plus one folder per take of each repeated pose.  If you would rather spread the
unique photos evenly so the folders end up a similar size, pass --balance (off
by default).

Why structure dominates the blend: on a studio shoot every frame shares the
backdrop and lighting, so pHash and colour barely discriminate (a same-backdrop
pair already scores ~0.5 identical-ness on those alone) — weighting them heavily
would mark DIFFERENT poses as alike.  The grayscale structure signal is the one
that actually tracks composition, hence its high default weight.

Usage:
  # just give a folder path: every folder under it that holds images gets its
  # own set of group folders, created INSIDE that folder (in "_groups/")
  python3 group_similar.py "input_folder"

  # or point at one shoot folder
  python3 group_similar.py "input_folder/Paltu/Edit"

Tuning (lower --similar-threshold to force MORE separation / more folders; raise
it to tolerate more similarity within a folder / fewer folders).
"""

import argparse
import csv
import logging
import os
import shutil
import sys
import time

from PIL import Image, ImageOps, ExifTags
import imagehash
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# Windows-safe console: force UTF-8 with a lossy fallback so non-ASCII filenames
# never crash printing on a cp1252 console.  line_buffering makes every print
# show up immediately (no "looks stuck" while a long phase runs silently).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

# Name of the per-folder output directory created inside each image folder.
DEFAULT_OUT_NAME = "_groups"

# EXIF tag ids
TAG_DATETIME = 306             # DateTime
TAG_DATETIME_ORIGINAL = 36867  # DateTimeOriginal (capture time)

# Colours used to tint each group in the Excel sheet
PALETTE = ["D9E1F2", "FCE4D6", "E2EFDA", "FFF2CC", "EDEDED",
           "DDEBF7", "FDE9D9", "E4DFEC", "DCE6F1", "F2DCDB"]

# Module logger.  Console progress is still printed directly; the log file
# records the full story: what was scanned, which photos conflict (must be
# separated), and how the folders were balanced.
log = logging.getLogger("group_similar")


def setup_logging(log_path):
    """Send detailed DEBUG logging to `log_path` (created/overwritten).

    Returns the path on success, or None if the file could not be opened.
    """
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.propagate = False
    try:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    except Exception:
        return None
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(fh)
    return log_path


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def list_images(folder, recursive=False):
    """Return a sorted list of image file paths inside `folder`."""
    out = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() in IMG_EXTS:
                    out.append(os.path.join(root, name))
    else:
        if not os.path.isdir(folder):
            return []
        for name in sorted(os.listdir(folder)):
            p = os.path.join(folder, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMG_EXTS:
                out.append(p)
    return out


def read_capture_time(path):
    """Return the capture timestamp string (DateTimeOriginal) or None."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            try:
                ifd = exif.get_ifd(ExifTags.IFD.Exif)
            except Exception:
                ifd = {}
            dto = ifd.get(TAG_DATETIME_ORIGINAL)
            if dto:
                return str(dto)
            dt = exif.get(TAG_DATETIME)
            if dt:
                return str(dt)
    except Exception:
        return None
    return None


def _nz(a):
    """Flatten `a` and return it zero-mean / unit-std (a normalised vector)."""
    a = a.ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def features_of(path, hash_size, struct_grid, color_grid):
    """Return (phash_bits, gray_vec, edge_vec, color_vec) for one image.

    Returns (None, None, None, None) on failure.

    Orientation is normalised (EXIF transpose) so rotated copies still match.
    Every vector is zero-mean / unit-std so a plain normalised dot product IS
    the correlation (a value in [-1, 1]).

    * gray_vec  — the downscaled grayscale frame.  This is the main signal:
                  it compares actual COMPOSITION, so an extra/missing person or
                  object, or a different crop, drops the score.
    * edge_vec  — gradient magnitude, emphasising outlines / object boundaries.
    * color_vec — downscaled RGB, colour-aware (backdrop / outfit differences).
    """
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            g = im.convert("L")
            ph = imagehash.phash(g, hash_size=hash_size)
            gray = np.asarray(g.resize((struct_grid, struct_grid), Image.BILINEAR),
                              dtype=np.float32)
            gy, gx = np.gradient(gray)
            edge = np.hypot(gx, gy)
            col = np.asarray(
                im.convert("RGB").resize((color_grid, color_grid), Image.BILINEAR),
                dtype=np.float32)
        gray_v = _nz(gray)
        edge_v = _nz(edge)
        col_v = np.concatenate([_nz(col[:, :, c]) for c in range(3)])
        bits = np.asarray(ph.hash, dtype=np.int8).ravel()
        return bits, gray_v, edge_v, col_v
    except Exception as e:
        log.warning("cannot read '%s': %s: %s", path, type(e).__name__, e)
        return None, None, None, None


# --------------------------------------------------------------------------- #
# Folder discovery
# --------------------------------------------------------------------------- #
def find_image_folders(root, out_name):
    """Every folder under `root` that directly holds at least one image.

    Traversal skips our own output folders (the `out_name` directory and the
    `group_*` / `_singletons` folders inside it) so a second run does not try to
    re-separate the copies it produced the first time.
    """
    folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames)
                       if d != out_name
                       and not d.startswith("group_")
                       and d != "_singletons"]
        if any(os.path.splitext(f)[1].lower() in IMG_EXTS for f in filenames):
            folders.append(dirpath)
    return folders


# --------------------------------------------------------------------------- #
# Separation core: conflict graph -> colouring -> balancing
# --------------------------------------------------------------------------- #
def build_conflict(paths, bits_list, gray_list, edge_list, col_list,
                   w_hash, w_gray, w_edge, w_color,
                   sim_threshold, strong_gray_threshold,
                   chunk=256, progress=False):
    """Build the pairwise similarity matrices and the conflict graph.

    Returns (conflict, S, Gs):
      * conflict — bool (n x n): True where two photos are "alike" and therefore
                   MUST NOT share a folder.
      * S        — float32 (n x n): combined similarity (weighted blend).
      * Gs       — float32 (n x n): grayscale structure correlation.

    A pair conflicts when its combined similarity >= `sim_threshold` OR its
    structure correlation >= `strong_gray_threshold`.
    """
    n = len(paths)
    H = np.stack(bits_list).astype(np.int32)     # n x bits (0/1)
    GR = np.stack(gray_list).astype(np.float32)  # n x dg (normalised rows)
    ED = np.stack(edge_list).astype(np.float32)  # n x de
    CO = np.stack(col_list).astype(np.float32)   # n x dc
    bits = H.shape[1]
    dg, de, dc = GR.shape[1], ED.shape[1], CO.shape[1]

    # Precompute the transposed hash matrices once for the chunked products.
    notH = (1 - H).T        # bits x n
    Ht = H.T                # bits x n

    S = np.empty((n, n), dtype=np.float32)
    Gs = np.empty((n, n), dtype=np.float32)
    conflict = np.zeros((n, n), dtype=bool)
    t0 = time.time()
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        A = H[i0:i1]                     # c x bits
        notA = 1 - A                     # c x bits
        Dh = A @ notH + notA @ Ht        # c x n (Hamming distance)
        hash_sim = 1.0 - Dh / bits       # c x n, in [0, 1]
        Cg = (GR[i0:i1] @ GR.T) / dg
        Ce = (ED[i0:i1] @ ED.T) / de
        Cc = (CO[i0:i1] @ CO.T) / dc
        S[i0:i1] = w_hash * hash_sim + w_gray * Cg + w_edge * Ce + w_color * Cc
        Gs[i0:i1] = Cg
        conflict[i0:i1] = ((S[i0:i1] >= sim_threshold)
                           | (Gs[i0:i1] >= strong_gray_threshold))
        if progress and (i1 % (chunk * 8) == 0 or i1 == n):
            pct = 100 * i1 / n
            rate = i1 / max(time.time() - t0, 1e-6)
            eta = (n - i1) / rate if rate > 0 else 0
            print(f"    compared {i1}/{n} ({pct:.0f}%), ~{eta:.0f}s left")
    np.fill_diagonal(S, 1.0)
    np.fill_diagonal(Gs, 1.0)
    np.fill_diagonal(conflict, False)
    return conflict, S, Gs


def dsatur_colour(conflict):
    """DSATUR graph colouring of the conflict graph.

    Each vertex (photo) gets the smallest colour not used by its neighbours
    (its conflicting "alike" photos).  Vertices are chosen in order of
    DECREASING saturation (number of distinct colours among neighbours), which
    keeps the colour count close to the minimum.  Returns a list of colour
    classes, each a list of vertex indices.
    """
    n = conflict.shape[0]
    if n == 0:
        return []
    colors = [-1] * n
    neighbour_colors = [set() for _ in range(n)]
    degrees = conflict.sum(axis=1)
    uncoloured = set(range(n))
    while uncoloured:
        best = None
        best_key = None
        for v in uncoloured:
            key = (len(neighbour_colors[v]), degrees[v], -v)
            if best_key is None or key > best_key:
                best_key = key
                best = v
        v = best
        used = neighbour_colors[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c
        uncoloured.discard(v)
        for u in np.nonzero(conflict[v])[0]:
            u = int(u)
            if colors[u] == -1:
                neighbour_colors[u].add(c)
    k = max(colors) + 1
    groups = [[] for _ in range(k)]
    for v, c in enumerate(colors):
        groups[c].append(v)
    return [g for g in groups if g]


def balance_groups(groups, conflict):
    """Even out folder sizes without ever breaking the no-alike-pair rule.

    Repeatedly move a photo from the largest folder to the smallest folder that
    contains NO conflicting (alike) photo.  Every move reduces the spread, so
    this terminates.  Returns the (possibly reordered) group list.
    """
    groups = [list(g) for g in groups]
    k = len(groups)
    if k <= 1:
        return groups
    while True:
        order = sorted(range(k), key=lambda i: len(groups[i]))
        largest = order[-1]
        smallest = order[0]
        if len(groups[largest]) - len(groups[smallest]) <= 1:
            break
        moved = False
        # Prefer the truly smallest folder; fall back to any folder that is
        # still meaningfully smaller than the largest.
        for cand in order:
            if cand == largest:
                continue
            if len(groups[largest]) - len(groups[cand]) <= 1:
                continue
            for idx in list(groups[largest]):
                clash = conflict[idx]
                if any(clash[other] for other in groups[cand]):
                    continue
                groups[largest].remove(idx)
                groups[cand].append(idx)
                moved = True
                break
            if moved:
                break
        if not moved:
            break
    return groups


def separate_groups(paths, bits_list, gray_list, edge_list, col_list,
                    w_hash, w_gray, w_edge, w_color,
                    sim_threshold, strong_gray_threshold,
                    chunk=256, progress=False, balance=False):
    """Full pipeline: conflict graph -> DSATUR colouring -> (optional) balance.

    Returns (groups, S, Gs) where `groups` is a list of index lists (one per
    folder) and `S`/`Gs` are the pairwise matrices used for reporting.

    By default the unique (non-conflicting) photos are left pooled together, so
    folders are as FEW and as PURE as possible.  Pass `balance=True` to spread
    them out so the folders end up similar in size.
    """
    conflict, S, Gs = build_conflict(
        paths, bits_list, gray_list, edge_list, col_list,
        w_hash, w_gray, w_edge, w_color, sim_threshold, strong_gray_threshold,
        chunk=chunk, progress=progress)
    groups = dsatur_colour(conflict)
    if balance:
        groups = balance_groups(groups, conflict)
    return groups, S, Gs


def _log_decisions(paths, groups, S, Gs, sim_threshold, strong_gray_threshold):
    """Write a human-readable reason per folder and per forced-apart pair."""
    n_conflicts = int(np.count_nonzero(np.triu(S >= sim_threshold, 1))
                      + np.count_nonzero(np.triu(Gs >= strong_gray_threshold, 1)))
    log.info("separation summary: %d folder(s) from %d forced-apart pair(s)",
             len(groups), n_conflicts)
    for rank, members in enumerate(sorted(groups, key=len, reverse=True), 1):
        names = ", ".join(os.path.basename(paths[i]) for i in sorted(members))
        log.info("FOLDER %02d  size=%d  members: %s", rank, len(members), names)


# --------------------------------------------------------------------------- #
# Per-folder processing
# --------------------------------------------------------------------------- #
def process_folder(folder, out_dir, args):
    t0 = time.time()
    paths = list_images(folder, recursive=False)
    if not paths:
        print(f"  [skip] no images in '{folder}'")
        return None

    print(f"  folder : {folder}")
    print(f"  images : {len(paths)}")
    log.info("")
    log.info("FOLDER %s  (%d image(s))", folder, len(paths))

    # ---- Phase 1: read each image and build its fingerprint ---------------- #
    print(f"  [1/3] fingerprinting {len(paths)} image(s) ...")
    keep, bits_list, gray_list, edge_list, col_list = [], [], [], [], []
    total = len(paths)
    t1 = time.time()
    for idx, p in enumerate(paths, 1):
        name = os.path.basename(p)
        b, gv, ev, cv = features_of(p, args.hash_size, args.struct_grid,
                                    args.grid)
        if b is None:
            print(f"    [{idx}/{total}] [warn] unreadable: {name}")
            log.warning("SKIP unreadable [%d/%d]: %s", idx, total, name)
            continue
        keep.append(p)
        bits_list.append(b)
        gray_list.append(gv)
        edge_list.append(ev)
        col_list.append(cv)
        if idx % 25 == 0 or idx == total:
            rate = idx / max(time.time() - t1, 1e-6)
            eta = (total - idx) / rate if rate > 0 else 0
            print(f"    [{idx}/{total}] {rate:.1f} img/s, ~{eta:.0f}s left "
                  f"- {name}")
    paths = keep
    if not paths:
        print("  [skip] nothing readable")
        return None
    print(f"  [1/3] done in {time.time() - t1:.1f}s "
          f"({len(paths)} fingerprint(s))")

    # ---- Phase 2: build conflict graph, colour, balance -------------------- #
    pairs = len(paths) * (len(paths) - 1) // 2
    print(f"  [2/3] separating {len(paths)} image(s) ({pairs} pairs) "
          f"(alike if combined>= {args.similar_threshold} with weights "
          f"h/g/e/c = {args.w_hash}/{args.w_gray}/{args.w_edge}/{args.w_color}, "
          f"or struct>= {args.strong_gray_threshold}) ...")
    t2 = time.time()
    groups, S, Gs = separate_groups(
        paths, bits_list, gray_list, edge_list, col_list,
        args.w_hash, args.w_gray, args.w_edge, args.w_color,
        args.similar_threshold, args.strong_gray_threshold,
        progress=True, balance=args.balance)
    print(f"  [2/3] done in {time.time() - t2:.1f}s "
          f"({len(groups)} folder(s))")
    log.info("[2/3] separated %d image(s) into %d folder(s) in %.1fs",
             len(paths), len(groups), time.time() - t2)
    _log_decisions(paths, groups, S, Gs, args.similar_threshold,
                   args.strong_gray_threshold)

    # capture-time cache (used for ordering + report)
    ctime = {}

    def ct(i):
        if i not in ctime:
            ctime[i] = read_capture_time(paths[i]) or ""
        return ctime[i]

    # Order folders: biggest first, then earliest capture time, then name.
    def gkey(members):
        times = [ct(i) or "~" for i in members]
        names = sorted(os.path.basename(paths[i]) for i in members)
        return (-len(members), min(times), names[0])

    groups.sort(key=gkey)

    os.makedirs(out_dir, exist_ok=True)

    # collision-safe copying (matters only when --recursive flattens sub-folders)
    used = {}

    def place(src, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        base = os.path.basename(src)
        if used.get((dest_dir, base)) not in (None, src):
            stem, ext = os.path.splitext(base)
            n = 1
            while used.get((dest_dir, f"{stem}_{n}{ext}")) not in (None, src):
                n += 1
            base = f"{stem}_{n}{ext}"
        used[(dest_dir, base)] = src
        dest = os.path.join(dest_dir, base)
        shutil.copy2(src, dest)

    print(f"  [3/3] writing {len(groups)} folder(s) to '{out_dir}' ...")
    rows = []          # (group_id, file, group_size, max_similarity, capture_time)
    n_groups = 0
    sizes = []

    for members in groups:
        size = len(members)
        n_groups += 1
        gid = f"{n_groups:04d}"
        dest_dir = os.path.join(out_dir, f"group_{gid}")
        sizes.append(size)
        # order within the folder by capture time, then name
        ordered = sorted(members, key=lambda i: (ct(i) or "~",
                                                 os.path.basename(paths[i])))
        for m in members:
            if size > 1:
                best = max(float(S[m, o]) for o in members if o != m)
            else:
                best = 0.0
            rows.append((gid, os.path.basename(paths[m]), size,
                         round(best, 3), ct(m)))
        for m in ordered:
            place(paths[m], dest_dir)

    # ---- CSV report ------------------------------------------------------ #
    csv_path = os.path.join(out_dir, "groups.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "file", "group_size", "max_similarity",
                    "capture_time"])
        w.writerows(rows)

    # ---- Excel report ---------------------------------------------------- #
    xlsx_path = os.path.join(out_dir, "groups.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "Groups"
    headers = ["group_id", "file", "group_size", "max_similarity",
               "capture_time"]
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="305496")
        cell.alignment = Alignment(horizontal="center")

    color_for = {}
    for gid, fname, size, sim, ctv in rows:
        if gid not in color_for:
            try:
                idx = int(gid) - 1
            except ValueError:
                idx = 0
            color_for[gid] = PALETTE[idx % len(PALETTE)]
        ws.append([gid, fname, size, sim, ctv])
        fill = color_for[gid]
        ws.cell(row=ws.max_row, column=1).fill = PatternFill("solid",
                                                             fgColor=fill)

    for i, w in enumerate([14, 34, 12, 14, 22], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    wb.save(xlsx_path)

    largest = max(sizes, default=0)
    smallest = min(sizes, default=0)
    print(f"  folders          : {n_groups}  "
          f"(largest: {largest}, smallest: {smallest})")
    print(f"  output           : {out_dir}")
    print(f"  report           : {csv_path}")
    print(f"  done in {time.time() - t0:.1f}s")
    log.info("[3/3] wrote %d folder(s) to %s", n_groups, out_dir)

    return {
        "folder": folder,
        "out": out_dir,
        "images": len(paths),
        "groups": n_groups,
        "largest": largest,
        "smallest": smallest,
        "csv": csv_path,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Distribute photos so SIMILAR ones land in DIFFERENT "
                    "folders: every output folder ends up holding only "
                    "visibly distinct photos. Give one folder path; every "
                    "folder under it that holds images gets its own group "
                    "folders, created inside it. Local, free, non-destructive.",
    )
    p.add_argument("path",
                   help="folder to scan; every nested folder holding images is "
                        "separated, with a '<out-name>/' folder created inside "
                        "it")
    p.add_argument("--out-name", default=DEFAULT_OUT_NAME,
                   help="name of the group folder created inside each image "
                        f"folder (default: {DEFAULT_OUT_NAME})")
    p.add_argument("--hash-size", type=int, default=8,
                   help="pHash bits per side (default 8 -> 64-bit hash)")
    p.add_argument("--similar-threshold", type=float, default=0.60,
                   help="min COMBINED similarity (weighted blend of pHash, "
                        "structure, edge and colour) at which two photos are "
                        "treated as alike and forced into DIFFERENT folders; "
                        "lower = more separation / more folders (default 0.60)")
    p.add_argument("--w-hash", type=float, default=0.15,
                   help="weight of pHash similarity in the combined score. "
                        "Keep this LOW: on a same-backdrop shoot pHash barely "
                        "discriminates and would falsely merge different poses "
                        "(default 0.15)")
    p.add_argument("--w-gray", type=float, default=0.6,
                   help="weight of grayscale STRUCTURE correlation — the main "
                        "signal that tracks composition (default 0.6)")
    p.add_argument("--w-edge", type=float, default=0.1,
                   help="weight of edge/outline correlation in the combined "
                        "score (default 0.1)")
    p.add_argument("--w-color", type=float, default=0.15,
                   help="weight of colour correlation in the combined score. "
                        "Keep this LOW: a shared backdrop gives a high colour "
                        "score even for different poses (default 0.15)")
    p.add_argument("--strong-gray-threshold", type=float, default=0.80,
                   help="min grayscale structure correlation at which two "
                        "photos are alike on its own (forced apart) even if the "
                        "combined score is low (default 0.80)")
    p.add_argument("--grid", type=int, default=16,
                   help="colour correlation grid size, grid x grid (default 16)")
    p.add_argument("--struct-grid", type=int, default=48,
                   help="grayscale structure grid size, grid x grid "
                        "(default 48)")
    p.add_argument("--balance", action="store_true", default=False,
                   help="spread the unique photos evenly so all folders end up "
                        "a similar size; off by default, which pools unique "
                        "photos together and keeps the folder count minimal")
    return p


def main():
    args = build_parser().parse_args()
    root = args.path
    if not os.path.isdir(root):
        sys.exit(f"error: '{root}' is not a directory")

    folders = find_image_folders(root, args.out_name)
    if not folders:
        sys.exit(f"error: no folders containing images found under '{root}'")

    log_path = setup_logging(os.path.join(root, "group_similar.log"))
    log.info("=== group_similar run start ===")
    log.info("root=%s  out_name=%s  python=%s", os.path.abspath(root),
             args.out_name, sys.version.split()[0])
    log.info("settings: hash_size=%d alike_if_combined>=%.2f weights h/g/e/c="
             "%.2f/%.2f/%.2f/%.2f strong_gray>=%.2f grid=%d struct_grid=%d "
             "balance=%s",
             args.hash_size, args.similar_threshold, args.w_hash, args.w_gray,
             args.w_edge, args.w_color, args.strong_gray_threshold, args.grid,
             args.struct_grid, args.balance)
    log.info("image folders found (%d):", len(folders))
    for f in folders:
        log.info("  - %s", os.path.relpath(f, root))

    print(f"Found {len(folders)} folder(s) with images under '{root}'")
    if log_path:
        print(f"  log              : {log_path}")
    for i, f in enumerate(folders, 1):
        print(f"  {i:>3}. {os.path.relpath(f, root)}")

    summaries = []
    n_folders = len(folders)
    run_t0 = time.time()
    for fi, folder in enumerate(folders, 1):
        # Group folders are created INSIDE each image folder.
        out_dir = os.path.join(folder, args.out_name)
        rel = os.path.relpath(folder, root)
        print()
        print("=" * 66)
        print(f" [{fi}/{n_folders}] {rel}")
        print("=" * 66)
        s = process_folder(folder, out_dir, args)
        if s:
            summaries.append(s)
        elapsed = time.time() - run_t0
        print(f"  progress: {fi}/{n_folders} folder(s) done, "
              f"{elapsed:.1f}s elapsed")

    print("\n=== Done ===")
    tot_images = sum(s["images"] for s in summaries)
    tot_groups = sum(s["groups"] for s in summaries)
    log.info("=== run done: %d image(s) -> %d folder(s) across %d folder(s) ===",
             tot_images, tot_groups, len(summaries))
    print(f"{tot_images} image(s) -> {tot_groups} folder(s) "
          f"across {len(summaries)} folder(s).")


if __name__ == "__main__":
    main()