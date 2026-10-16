#!/usr/bin/env python3
"""
group_similar.py — group visually similar photos (bursts / repeated takes) into
folders, the way a photoshoot produces many near-identical frames of the same
shot (a model standing or smiling clicked several times).

Given a folder of photos, this finds groups of near-duplicate / same-scene
photos and copies each group into its own sub-folder, plus a CSV/XLSX report.
The source files are never moved, renamed or modified — they are only copied.

Similarity is measured 100% locally (no cloud, no cost, no API key) using a
blend of FOUR complementary, content-aware signals:

  1. pHash (perceptual hash) ......... coarse, rotation-normalised, cheap.
  2. Grayscale STRUCTURE correlation . the dominant signal: the whole frame is
                                       downscaled and contrast-normalised, so
                                       COMPOSITION is compared directly.  An
                                       extra person, a missing prop or a
                                       different crop changes the structure and
                                       lowers this score — exactly the "same
                                       scene but not the same shot" case.
  3. Edge correlation ................ captures outlines / object boundaries.
  4. Colour correlation on a grid .... colour-aware detail (backdrop, outfit).

Two photos join the same group when EITHER:
  * their COMBINED similarity — a weighted blend of the four measures above,
    weighted by --w-hash / --w-gray / --w-edge / --w-color — is
    >= --sim-threshold (default 0.55),  -- or
  * their grayscale structure correlation alone is
    >= --strong-gray-threshold (default 0.80) (near-identical composition).

Blending the measures (instead of AND-ing hard cutoffs) avoids the old
"AND-gate" failure where a genuine same-shot pair was split because one coarse
measure was a hair too far even though the others agreed.

Groups are built with AVERAGE-LINKAGE (UPGMA) agglomerative clustering: two
clusters merge only while the AVERAGE similarity between their members is high
enough.  This is what prevents "chaining" — a frame similar to just ONE member
of a group no longer drags a run of loosely-related frames together; such fringe
frames drop out as singletons.

Usage:
  # just give a folder path: every folder under it that holds images gets its
  # own set of group folders, created INSIDE that folder (in "_groups/")
  python3 group_similar.py "input_folder"

  # or point at one shoot folder
  python3 group_similar.py "input_folder/Paltu/Edit"

Tuning (lower --sim-threshold to group more aggressively; raise it to be
stricter).
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
# records the full story: what was scanned, which photos matched, and WHY (or
# why a photo ended up a singleton).
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
    re-group the copies it produced the first time.
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
# Grouping core
# --------------------------------------------------------------------------- #
def _weights(args):
    return (args.w_hash, args.w_gray, args.w_edge, args.w_color)


def _log_cluster_reasons(paths, groups, GR, args):
    """Write a human-readable "why" for every group and every singleton."""
    d = GR.shape[1]
    multi = sorted((g for g in groups if len(g) > 1), key=len, reverse=True)
    singles = [g[0] for g in groups if len(g) == 1]

    log.info("cluster summary: %d multi-photo group(s), %d singleton(s)",
             len(multi), len(singles))
    for rank, members in enumerate(multi, 1):
        names = ", ".join(os.path.basename(paths[i]) for i in members)
        log.info("GROUP %02d  size=%d  members: %s", rank, len(members), names)

    for i in singles:
        sims = (GR @ GR[i]) / d
        sims[i] = -2.0
        j = int(np.argmax(sims))
        c = float(sims[j])
        misses = []
        if c < args.strong_gray_threshold:
            misses.append(f"structure corr {c:.3f} < "
                          f"{args.strong_gray_threshold} (strong)")
        log.debug("SINGLE  %s  | closest %s: structure corr %.3f "
                  "-> not grouped because %s",
                  os.path.basename(paths[i]), os.path.basename(paths[j]),
                  c, "; ".join(misses) or "no qualifying partner")


def build_groups(paths, bits_list, gray_list, edge_list, col_list,
                 w_hash=0.3, w_gray=0.4, w_edge=0.1, w_color=0.2,
                 sim_threshold=0.55, strong_gray_thresh=0.80, chunk=256,
                 progress=False, log_decisions=False):
    """Average-linkage (UPGMA) agglomerative clustering of the "similar" relation.

    Unlike single-linkage union-find — which merges two clusters as soon as the
    single closest pair crosses the threshold — average linkage merges only
    while the AVERAGE similarity between the two clusters' members is high
    enough.  That is what stops "chaining": a frame that is similar to just ONE
    member of a group (an intermediate pose) no longer drags a whole run of
    loosely-related frames into the same folder.  Such fringe frames fall out as
    singletons instead.

    The merge score is the COMBINED similarity (weighted blend of pHash,
    grayscale structure, edge and colour); two clusters are also merged when
    their average grayscale structure correlation alone is very high
    (`strong_gray_thresh`), which keeps near-identical frames together even when
    the combined blend dips.

    Returns (groups, GR) where groups is a list of index lists and GR is the
    (n x d) matrix of normalised grayscale vectors (used again for reporting).
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

    # Full pairwise matrices (float32): combined similarity and structure corr.
    S = np.empty((n, n), dtype=np.float32)
    Gs = np.empty((n, n), dtype=np.float32)
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
        if progress and (i1 % (chunk * 8) == 0 or i1 == n):
            pct = 100 * i1 / n
            rate = i1 / max(time.time() - t0, 1e-6)
            eta = (n - i1) / rate if rate > 0 else 0
            print(f"    compared {i1}/{n} ({pct:.0f}%), ~{eta:.0f}s left")
    np.fill_diagonal(S, 1.0)
    np.fill_diagonal(Gs, 1.0)

    # Average-linkage (UPGMA).  Sc/Gc hold the current average similarity
    # between the clusters represented by active indices; Lance–Williams update.
    Sc = S.copy()
    Gc = Gs.copy()
    size = np.ones(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    members = [[i] for i in range(n)]

    while True:
        M = np.where(active[:, None] & active[None, :], Sc, -np.inf)
        np.fill_diagonal(M, -np.inf)
        Sgt = np.where(active[:, None] & active[None, :], Gc, -np.inf)
        np.fill_diagonal(Sgt, -np.inf)
        valid = (M >= sim_threshold) | (Sgt >= strong_gray_thresh)
        score = np.where(valid, np.maximum(M, Sgt), -np.inf)
        k = int(np.argmax(score))
        a, b = divmod(k, n)
        best = score[a, b]
        if not np.isfinite(best):
            break
        mc, mg = float(M[a, b]), float(Sgt[a, b])
        na, nb = size[a], size[b]
        Sc[a, :] = (na * Sc[a, :] + nb * Sc[b, :]) / (na + nb)
        Sc[:, a] = Sc[a, :]
        Gc[a, :] = (na * Gc[a, :] + nb * Gc[b, :]) / (na + nb)
        Gc[:, a] = Gc[a, :]
        size[a] = na + nb
        members[a] = members[a] + members[b]
        active[b] = False
        if log_decisions:
            rule = ("combined avg %.3f >= %.2f" % (mc, sim_threshold)
                    if mc >= sim_threshold
                    else "structure avg %.3f >= %.2f" % (mg, strong_gray_thresh))
            log.debug("MERGE  -> cluster size %d  (%s)", int(size[a]), rule)

    groups = [members[i] for i in range(n) if active[i]]

    if log_decisions:
        _log_cluster_reasons(paths, groups, GR,
                             argparse.Namespace(strong_gray_threshold=
                                                strong_gray_thresh))
    return groups, GR


def pick_representative(members, GR):
    """Return the medoid index: the member most similar to all the others."""
    if len(members) == 1:
        return members[0]
    sub = GR[members]                             # m x d
    sims = (sub @ sub.T) / sub.shape[1]           # m x m correlations
    np.fill_diagonal(sims, -2.0)
    return members[int(np.argmax(sims.mean(axis=1)))]


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

    # ---- Phase 2: cluster -------------------------------------------------- #
    pairs = len(paths) * (len(paths) - 1) // 2
    print(f"  [2/3] clustering {len(paths)} image(s) ({pairs} pairs) "
          f"(combined>= {args.sim_threshold}, weights h/g/e/c = "
          f"{args.w_hash}/{args.w_gray}/{args.w_edge}/{args.w_color}, "
          f"or struct>= {args.strong_gray_threshold}) ...")
    t2 = time.time()
    groups, GR = build_groups(paths, bits_list, gray_list, edge_list, col_list,
                              args.w_hash, args.w_gray, args.w_edge,
                              args.w_color, args.sim_threshold,
                              args.strong_gray_threshold, progress=True,
                              log_decisions=True)
    print(f"  [2/3] done in {time.time() - t2:.1f}s "
          f"({len(groups)} raw cluster(s))")
    log.info("[2/3] clustered %d image(s) into %d raw cluster(s) in %.1fs",
             len(paths), len(groups), time.time() - t2)

    # capture-time cache (used for ordering + report)
    ctime = {}

    def ct(i):
        if i not in ctime:
            ctime[i] = read_capture_time(paths[i]) or ""
        return ctime[i]

    # Order: biggest groups first, then earliest capture time, then name.
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

    print(f"  [3/3] writing {len(groups)} cluster(s) to '{out_dir}' ...")
    rows = []          # (group_id, file, group_size, similarity, capture_time)
    n_groups = 0
    n_singletons = 0
    group_index = 0

    for members in groups:
        size = len(members)
        rep = pick_representative(members, GR)
        is_single = size < args.min_group_size

        if is_single:
            n_singletons += 1
            if args.singletons == "skip":
                continue
            if args.singletons == "separate":
                gid = "_singletons"
                dest_dir = os.path.join(out_dir, "_singletons")
            else:  # "group": give each singleton its own numbered folder
                group_index += 1
                n_groups += 1
                gid = f"{group_index:04d}"
                dest_dir = os.path.join(out_dir, f"group_{gid}")
        else:
            group_index += 1
            n_groups += 1
            gid = f"{group_index:04d}"
            dest_dir = os.path.join(out_dir, f"group_{gid}")

        for m in members:
            sim = 1.0 if m == rep else float((GR[m] @ GR[rep]) / GR.shape[1])
            place(paths[m], dest_dir)
            rows.append((gid, os.path.basename(paths[m]), size,
                         round(sim, 3), ct(m)))

    # ---- CSV report ------------------------------------------------------ #
    csv_path = os.path.join(out_dir, "groups.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "file", "group_size", "similarity",
                    "capture_time"])
        w.writerows(rows)

    # ---- Excel report ---------------------------------------------------- #
    xlsx_path = os.path.join(out_dir, "groups.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "Groups"
    headers = ["group_id", "file", "group_size", "similarity", "capture_time"]
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="305496")
        cell.alignment = Alignment(horizontal="center")

    color_for = {}
    for gid, fname, size, sim, ctv in rows:
        if gid not in color_for:
            if gid == "_singletons":
                color_for[gid] = "FFF2CC"
            else:
                try:
                    idx = int(gid) - 1
                except ValueError:
                    idx = 0
                color_for[gid] = PALETTE[idx % len(PALETTE)]
        ws.append([gid, fname, size, sim, ctv])
        fill = color_for[gid]
        ws.cell(row=ws.max_row, column=1).fill = PatternFill("solid",
                                                             fgColor=fill)

    for i, w in enumerate([14, 34, 12, 12, 22], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    wb.save(xlsx_path)

    multi = [m for m in groups if len(m) >= args.min_group_size]
    largest = max((len(m) for m in multi), default=0)
    print(f"  groups (>= {args.min_group_size}) : {n_groups}  "
          f"(largest: {largest})")
    print(f"  singletons       : {n_singletons}  (mode: {args.singletons})")
    print(f"  output           : {out_dir}")
    print(f"  report           : {csv_path}")
    print(f"  done in {time.time() - t0:.1f}s")
    log.info("[3/3] wrote %d group(s) + %d singleton(s) to %s",
             n_groups, n_singletons, out_dir)

    return {
        "folder": folder,
        "out": out_dir,
        "images": len(paths),
        "groups": n_groups,
        "singletons": n_singletons,
        "largest": largest,
        "csv": csv_path,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Group visually similar photos (bursts / repeated takes) "
                    "into folders. Give one folder path; every folder under it "
                    "that holds images gets its own group folders, created "
                    "inside it. Local, free, non-destructive.",
    )
    p.add_argument("path",
                   help="folder to scan; every nested folder holding images is "
                        "grouped, with a '<out-name>/' folder created inside it")
    p.add_argument("--out-name", default=DEFAULT_OUT_NAME,
                   help="name of the group folder created inside each image "
                        f"folder (default: {DEFAULT_OUT_NAME})")
    p.add_argument("--hash-size", type=int, default=8,
                   help="pHash bits per side (default 8 -> 64-bit hash)")
    p.add_argument("--sim-threshold", type=float, default=0.55,
                   help="min COMBINED similarity (weighted blend of pHash, "
                        "structure, edge and colour) to group two photos; "
                        "lower = group more (default 0.55)")
    p.add_argument("--w-hash", type=float, default=0.3,
                   help="weight of pHash similarity in the combined score "
                        "(default 0.3)")
    p.add_argument("--w-gray", type=float, default=0.4,
                   help="weight of grayscale STRUCTURE correlation (the main "
                        "signal) in the combined score (default 0.4)")
    p.add_argument("--w-edge", type=float, default=0.1,
                   help="weight of edge/outline correlation in the combined "
                        "score (default 0.1)")
    p.add_argument("--w-color", type=float, default=0.2,
                   help="weight of colour correlation in the combined score "
                        "(default 0.2)")
    p.add_argument("--strong-gray-threshold", type=float, default=0.80,
                   help="min grayscale structure correlation to group two "
                        "photos on its own, even if the combined score is low "
                        "(default 0.80)")
    p.add_argument("--grid", type=int, default=16,
                   help="colour correlation grid size, grid x grid (default 16)")
    p.add_argument("--struct-grid", type=int, default=48,
                   help="grayscale structure grid size, grid x grid "
                        "(default 48)")
    p.add_argument("--min-group-size", type=int, default=2,
                   help="a group must have at least this many photos "
                        "(default 2)")
    p.add_argument("--singletons", default="separate",
                   choices=["separate", "group", "skip"],
                   help="what to do with photos that have no similar partner "
                        "(default: separate)")
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
    log.info("settings: hash_size=%d sim>=%.2f weights h/g/e/c="
             "%.2f/%.2f/%.2f/%.2f strong_gray>=%.2f grid=%d struct_grid=%d "
             "min_group_size=%d singletons=%s",
             args.hash_size, args.sim_threshold, args.w_hash, args.w_gray,
             args.w_edge, args.w_color, args.strong_gray_threshold, args.grid,
             args.struct_grid, args.min_group_size, args.singletons)
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
    tot_single = sum(s["singletons"] for s in summaries)
    log.info("=== run done: %d image(s) -> %d group(s), %d singleton(s) "
             "across %d folder(s) ===",
             tot_images, tot_groups, tot_single, len(summaries))
    print(f"{tot_images} image(s) -> {tot_groups} group(s), "
          f"{tot_single} singleton(s) across {len(summaries)} folder(s).")


if __name__ == "__main__":
    main()