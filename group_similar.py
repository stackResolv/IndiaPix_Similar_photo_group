#!/usr/bin/env python3
"""
group_similar.py — split a folder of photos into exactly TWO output folders:

  distinct/    ONE representative of every look-alike (burst) group, plus every
               photo that has no look-alike at all.  Nothing in here is a near
               duplicate of anything else in here.
  duplicates/  every extra take — the near-twins of the representative that was
               kept in distinct/.

So a folder of 10 photos in which "standing" was shot 3x, "smiling" 2x and the
other 5 are unique becomes:

  distinct/    standing #1, smiling #1, plus the 5 unique shots   (7 photos)
  duplicates/  standing #2, standing #3, smiling #2               (3 photos)

The representative of each look-alike group is its EARLIEST shot (by EXIF
capture time, falling back to file name), so the kept frame is the first take;
the later takes are the ones moved aside.

Similarity is measured 100% locally (no cloud, no cost, no API key) using a
blend of FOUR complementary, content-aware signals:

  1. pHash (perceptual hash) ......... coarse, rotation-normalised, cheap.
  2. Grayscale STRUCTURE correlation . the dominant signal: the whole frame is
                                       downscaled and contrast-normalised, so
                                       COMPOSITION is compared directly.
  3. Edge correlation ................ captures outlines / object boundaries.
  4. Colour correlation on a grid .... colour-aware detail (backdrop, outfit).

Two photos are treated as "alike" (near duplicates) when EITHER:
  * their COMBINED similarity — a weighted blend of the four measures above,
    weighted by --w-hash / --w-gray / --w-edge / --w-color — is
    >= --similar-threshold (default 0.60),  -- or
  * their grayscale structure correlation alone is
    >= --strong-gray-threshold (default 0.80) (near-identical composition).

These "alike" pairs link the photos into LOOK-ALIKE GROUPS (connected
components): every group is one burst of repeated takes of the same shot.
Exactly one member of each group is kept in distinct/, the rest go to
duplicates/.

Why structure dominates the blend: on a studio shoot every frame shares the
backdrop and lighting, so pHash and colour barely discriminate (a same-backdrop
pair already scores ~0.5 identical-ness on those alone) — weighting them heavily
would mark DIFFERENT poses as alike.  The grayscale structure signal is the one
that actually tracks composition, hence its high default weight.

Usage:
  # just give a folder path: every folder under it that holds images gets its
  # own distinct/ + duplicates/ pair, created INSIDE that folder
  python3 group_similar.py "input_folder"

  # or point at one shoot folder
  python3 group_similar.py "input_folder/Paltu/Edit"

Tuning (lower --similar-threshold to treat more photos as duplicates; raise it
to keep more of them in distinct/).
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
DEFAULT_OUT_NAME = "_separated"

# The two output folders produced for every image folder.
GROUP_DISTINCT = "distinct"       # one representative per look-alike group + uniques
GROUP_DUPLICATES = "duplicates"   # all remaining extra takes

# Colours used to tint the two folders in the Excel sheet
GROUP_FILL = {
    GROUP_DISTINCT: "E2EFDA",
    GROUP_DUPLICATES: "FCE4D6",
}

# EXIF tag ids
TAG_DATETIME = 306             # DateTime
TAG_DATETIME_ORIGINAL = 36867  # DateTimeOriginal (capture time)

# Module logger.  Console progress is still printed directly; the log file
# records the full story: what was scanned, which photos are look-alikes, which
# one of each group was kept, and which takes were moved to duplicates/.
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


def _nz(a):
    """Flatten `a` and return it zero-mean / unit-std (a normalised vector)."""
    a = a.ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def features_of(path, hash_size, struct_grid, color_grid):
    """Return (phash_bits, gray_vec, edge_vec, color_vec, capture_time).

    Returns (None, None, None, None, "") on failure.

    Orientation is normalised (EXIF transpose) so rotated copies still match.
    Every vector is zero-mean / unit-std so a plain normalised dot product IS
    the correlation (a value in [-1, 1]).

    * gray_vec  — the downscaled grayscale frame.  This is the main signal:
                  it compares actual COMPOSITION, so an extra/missing person or
                  object, or a different crop, drops the score.
    * edge_vec  — gradient magnitude, emphasising outlines / object boundaries.
    * color_vec — downscaled RGB, colour-aware (backdrop / outfit differences).
    * capture_time — EXIF DateTimeOriginal/DateTime as a string, or "".
    """
    ctime = ""
    try:
        with Image.open(path) as im:
            try:
                exif = im.getexif()
                try:
                    ifd = exif.get_ifd(ExifTags.IFD.Exif)
                except Exception:
                    ifd = {}
                dto = ifd.get(TAG_DATETIME_ORIGINAL) or exif.get(TAG_DATETIME)
                if dto:
                    ctime = str(dto)
            except Exception:
                pass
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
        return bits, gray_v, edge_v, col_v, ctime
    except Exception as e:
        log.warning("cannot read '%s': %s: %s", path, type(e).__name__, e)
        return None, None, None, None, ""


# --------------------------------------------------------------------------- #
# Folder discovery
# --------------------------------------------------------------------------- #
def find_image_folders(root, out_name):
    """Every folder under `root` that directly holds at least one image.

    Traversal skips our own output folders (the `out_name` directory, plus the
    `group_*` / `_singletons` names left behind by older runs) so a second run
    does not try to re-separate the copies it produced the first time.
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
# Similarity -> look-alike groups -> distinct / duplicates split
# --------------------------------------------------------------------------- #
def build_conflict(paths, bits_list, gray_list, edge_list, col_list,
                   w_hash, w_gray, w_edge, w_color,
                   sim_threshold, strong_gray_threshold,
                   chunk=256, progress=False):
    """Build the pairwise similarity matrices and the conflict graph.

    Returns (conflict, S, Gs):
      * conflict — bool (n x n): True where two photos are "alike" and therefore
                   are near duplicates.
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


def look_alike_groups(conflict):
    """Connected components of the conflict graph (union-find).

    Every returned component is one burst of repeated takes: all its photos are
    linked (directly or transitively) by "alike" edges.  Photos with no
    look-alike at all come back as components of size 1.
    """
    n = conflict.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, j in zip(*np.nonzero(np.triu(conflict, 1))):
        union(int(i), int(j))

    comps = {}
    for v in range(n):
        comps.setdefault(find(v), []).append(v)
    return list(comps.values())


def split_photos(paths, bits_list, gray_list, edge_list, col_list, ctimes,
                 w_hash, w_gray, w_edge, w_color,
                 sim_threshold, strong_gray_threshold,
                 chunk=256, progress=False):
    """Split the photos into (distinct, duplicates).

    Builds the conflict graph, links it into look-alike groups, then keeps ONE
    representative per group in `distinct` (its earliest shot) and sends the
    remaining takes to `duplicates`.  Photos with no look-alike go to
    `distinct` unchanged.

    Returns (distinct, duplicates, info, comps, conflict, S, Gs) where
      * distinct / duplicates — index lists, in shot order,
      * info    — index -> (group_id, is_representative, representative_index),
      * comps   — the look-alike groups (list of index lists).
    """
    conflict, S, Gs = build_conflict(
        paths, bits_list, gray_list, edge_list, col_list,
        w_hash, w_gray, w_edge, w_color, sim_threshold, strong_gray_threshold,
        chunk=chunk, progress=progress)

    def shot_key(i):
        return (ctimes[i] or "~", os.path.basename(paths[i]))

    comps = [sorted(c, key=shot_key) for c in look_alike_groups(conflict)]
    comps.sort(key=lambda c: shot_key(c[0]))

    distinct, duplicates = [], []
    info = {}
    for c, members in enumerate(comps, 1):
        gid = f"LA_{c:04d}"
        rep = members[0]                      # earliest shot of the group
        for m in members:
            info[m] = (gid, m == rep, rep)
        distinct.append(rep)
        duplicates.extend(m for m in members if m != rep)

    return distinct, duplicates, info, comps, conflict, S, Gs


def _log_decisions(paths, comps, info, conflict):
    """Write a human-readable account of every look-alike group."""
    n_pairs = int(np.count_nonzero(np.triu(conflict, 1)))
    multi = [c for c in comps if len(c) > 1]
    log.info("look-alike groups: %d group(s) with >1 take, %d unique photo(s); "
             "%d alike pair(s)", len(multi), len(comps) - len(multi), n_pairs)
    for comp in sorted(multi, key=len, reverse=True):
        gid, _is_rep, rep = info[comp[0]]
        extras = ", ".join(os.path.basename(paths[i])
                           for i in comp if i != rep)
        log.info("  GROUP %s size=%d  distinct: %s  duplicates: %s",
                 gid, len(comp), os.path.basename(paths[rep]), extras)


def _reset_output(out_dir):
    """Remove this run's own previously produced folders so no stale copies stay.

    Only our output directory is touched: `distinct/`, `duplicates/` and the
    `group_*` folders written by older versions.  Source photos are never
    touched.
    """
    if not os.path.isdir(out_dir):
        return
    for name in (GROUP_DISTINCT, GROUP_DUPLICATES):
        p = os.path.join(out_dir, name)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
    for name in os.listdir(out_dir):
        p = os.path.join(out_dir, name)
        if name.startswith("group_") and os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


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
    ctimes = []
    total = len(paths)
    t1 = time.time()
    for idx, p in enumerate(paths, 1):
        name = os.path.basename(p)
        b, gv, ev, cv, ct = features_of(p, args.hash_size, args.struct_grid,
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
        ctimes.append(ct)
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

    # ---- Phase 2: find look-alike groups, split distinct / duplicates ------ #
    pairs = len(paths) * (len(paths) - 1) // 2
    print(f"  [2/3] comparing {len(paths)} image(s) ({pairs} pairs) "
          f"(alike if combined>= {args.similar_threshold} with weights "
          f"h/g/e/c = {args.w_hash}/{args.w_gray}/{args.w_edge}/{args.w_color}, "
          f"or struct>= {args.strong_gray_threshold}) ...")
    t2 = time.time()
    (distinct, duplicates, info, comps, conflict,
     S, Gs) = split_photos(
        paths, bits_list, gray_list, edge_list, col_list, ctimes,
        args.w_hash, args.w_gray, args.w_edge, args.w_color,
        args.similar_threshold, args.strong_gray_threshold, progress=True)
    n_multi = sum(1 for c in comps if len(c) > 1)
    print(f"  [2/3] done in {time.time() - t2:.1f}s "
          f"({n_multi} look-alike group(s) -> {len(distinct)} distinct, "
          f"{len(duplicates)} duplicate(s))")
    log.info("[2/3] split %d image(s) in %.1fs: %d look-alike group(s), "
             "%d kept in distinct/, %d in duplicates/",
             len(paths), time.time() - t2, n_multi, len(distinct),
             len(duplicates))
    _log_decisions(paths, comps, info, conflict)

    # ---- Phase 3: copy the two folders + write the reports ----------------- #
    _reset_output(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # collision-safe copying (matters only when two folders share file names)
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

    print(f"  [3/3] writing 2 folder(s) to '{out_dir}' ...")
    for label, members in ((GROUP_DISTINCT, distinct),
                           (GROUP_DUPLICATES, duplicates)):
        dest_dir = os.path.join(out_dir, label)
        os.makedirs(dest_dir, exist_ok=True)
        for m in members:
            place(paths[m], dest_dir)

    # highest similarity of each photo to the other takes of its own group
    # (size-1 groups: to any other photo in the folder)
    comp_size = {}
    max_sim = {}
    for comp in comps:
        cid = info[comp[0]][0]
        comp_size[cid] = len(comp)
        for m in comp:
            if len(comp) > 1:
                max_sim[m] = max(float(S[m, o]) for o in comp if o != m)
            else:
                max_sim[m] = max((float(S[m, o]) for o in range(len(paths))
                                  if o != m), default=0.0)

    rows = []          # [group, file, group_id, group_size, max_sim, dup_of, time]
    for label, members in ((GROUP_DISTINCT, distinct),
                           (GROUP_DUPLICATES, duplicates)):
        for m in members:
            cid, is_rep, rep = info[m]
            rows.append([label,
                         os.path.basename(paths[m]),
                         cid,
                         comp_size[cid],
                         round(max_sim[m], 3),
                         "" if is_rep else os.path.basename(paths[rep]),
                         ctimes[m]])

    headers = ["group", "file", "look_alike_group", "look_alike_size",
               "max_similarity", "duplicate_of", "capture_time"]

    # ---- CSV report ------------------------------------------------------ #
    csv_path = os.path.join(out_dir, "report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    # ---- Excel report ---------------------------------------------------- #
    xlsx_path = os.path.join(out_dir, "report.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "Distinct vs duplicates"
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="305496")
        cell.alignment = Alignment(horizontal="center")

    for label, fname, cid, size, sim, dup_of, ctv in rows:
        ws.append([label, fname, cid, size, sim, dup_of, ctv])
        ws.cell(row=ws.max_row, column=1).fill = PatternFill(
            "solid", fgColor=GROUP_FILL[label])

    for i, w in enumerate([13, 34, 17, 15, 14, 34, 22], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    wb.save(xlsx_path)

    print(f"  distinct         : {len(distinct)} image(s)")
    print(f"  duplicates       : {len(duplicates)} image(s) "
          f"({n_multi} look-alike group(s) trimmed)")
    print(f"  output           : {out_dir}")
    print(f"  report           : {csv_path}")
    print(f"  done in {time.time() - t0:.1f}s")
    log.info("[3/3] wrote distinct/ (%d) + duplicates/ (%d) to %s",
             len(distinct), len(duplicates), out_dir)

    return {
        "folder": folder,
        "out": out_dir,
        "images": len(paths),
        "distinct": len(distinct),
        "duplicates": len(duplicates),
        "groups": n_multi,
        "csv": csv_path,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Split photos into TWO folders: 'distinct' (one "
                    "representative of every look-alike group + every unique "
                    "photo) and 'duplicates' (all remaining extra takes). Give "
                    "one folder path; every folder under it that holds images "
                    "gets its own distinct/ + duplicates/ pair, created inside "
                    "it. Local, free, non-destructive.",
    )
    p.add_argument("path",
                   help="folder to scan; every nested folder holding images is "
                        "split, with a '<out-name>/' folder created inside it")
    p.add_argument("--out-name", default=DEFAULT_OUT_NAME,
                   help="name of the output folder created inside each image "
                        f"folder (default: {DEFAULT_OUT_NAME})")
    p.add_argument("--hash-size", type=int, default=8,
                   help="pHash bits per side (default 8 -> 64-bit hash)")
    p.add_argument("--similar-threshold", type=float, default=0.60,
                   help="min COMBINED similarity (weighted blend of pHash, "
                        "structure, edge and colour) at which two photos are "
                        "treated as the same shot; lower = more photos counted "
                        "as duplicates / fewer kept as distinct (default 0.60)")
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
                        "photos are duplicates on its own, even if the combined "
                        "score is low (default 0.80)")
    p.add_argument("--grid", type=int, default=16,
                   help="colour correlation grid size, grid x grid (default 16)")
    p.add_argument("--struct-grid", type=int, default=48,
                   help="grayscale structure grid size, grid x grid "
                        "(default 48)")
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
             "%.2f/%.2f/%.2f/%.2f strong_gray>=%.2f grid=%d struct_grid=%d",
             args.hash_size, args.similar_threshold, args.w_hash, args.w_gray,
             args.w_edge, args.w_color, args.strong_gray_threshold, args.grid,
             args.struct_grid)
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
        # distinct/ + duplicates/ are created INSIDE each image folder.
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
    tot_distinct = sum(s["distinct"] for s in summaries)
    tot_dups = sum(s["duplicates"] for s in summaries)
    log.info("=== run done: %d image(s) -> %d distinct + %d duplicates across "
             "%d folder(s) ===", tot_images, tot_distinct, tot_dups,
             len(summaries))
    print(f"{tot_images} image(s) -> {tot_distinct} distinct + {tot_dups} "
          f"duplicates across {len(summaries)} folder(s).")


if __name__ == "__main__":
    main()