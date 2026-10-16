#!/usr/bin/env python3
"""
group_similar.py — group visually similar photos (bursts / repeated takes) into
folders, the way a photoshoot produces many near-identical frames of the same
shot (a model standing or smiling clicked several times).

Given a folder of photos, this finds groups of near-duplicate / same-scene
photos and copies each group into its own sub-folder, plus a CSV/XLSX report.
The source files are never moved, renamed or modified — they are only copied.

Similarity is measured 100% locally (no cloud, no cost, no API key):

  1. pHash (perceptual hash) .......... coarse, rotation-normalised, cheap.
  2. Colour correlation on a grid ..... finer, colour-aware detail, used to
                                        CONFIRM a candidate really is the
                                        same shot (not just a similar scene).

Two photos join the same group when EITHER:
  * their pHash Hamming distance <= --phash-threshold AND
    their colour correlation      >= --corr-threshold (default 0.80),  -- or
  * their colour correlation      >= --strong-corr-threshold (default 0.95).

The first (tier A) is the conservative default. The second (tier B) lets two
frames that are near-identical in colour layout group even when the coarse pHash
disagrees, which makes burst grouping robust to small framing/sensor changes.

Groups are the connected components (union-find) of that "similar" relation.

Usage:
  # just give a folder path: every folder under it that holds images gets its
  # own set of group folders, created INSIDE that folder (in "_groups/")
  python3 group_similar.py "input_folder"

  # or point at one shoot folder
  python3 group_similar.py "input_folder/Paltu/Edit"

Tuning (raise --phash-threshold / lower --corr-threshold to group more
aggressively; do the opposite to be stricter).
"""

import argparse
import csv
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
# never crash printing on a cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
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


def features_of(path, hash_size, grid):
    """Return (phash_bits, color_vec) for one image, or (None, None) on failure.

    Orientation is normalised (EXIF transpose) so rotated copies still match.

    color_vec is a COLOUR-aware vector: the image is downscaled to grid x grid
    RGB and each channel is zero-mean / unit-std before flattening, so
    correlation is a plain normalised dot product. Using colour (not just
    grayscale) matters here: two otherwise similar compositions with a
    different backdrop or outfit are *different takes*, not one group.
    """
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            g = im.convert("L")
            ph = imagehash.phash(g, hash_size=hash_size)
            small = im.convert("RGB").resize((grid, grid), Image.BILINEAR)
            a = np.asarray(small, dtype=np.float32)  # grid x grid x 3
        chans = []
        for c in range(3):
            ch = a[:, :, c].ravel()
            chans.append((ch - ch.mean()) / (ch.std() + 1e-6))
        vec = np.concatenate(chans)
        bits = np.asarray(ph.hash, dtype=np.int8).ravel()
        return bits, vec
    except Exception:
        return None, None


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
def build_groups(paths, bits_list, vecs_list, phash_thresh, corr_thresh,
                 strong_corr_thresh=0.95, chunk=256):
    """Union-find clustering of the "similar" relation.

    Returns (groups, G) where groups is a list of index lists and G is the
    (n x d) matrix of normalised colour vectors (used again for reporting).
    """
    n = len(paths)
    H = np.stack(bits_list).astype(np.int32)        # n x bits (0/1)
    G = np.stack(vecs_list).astype(np.float32)      # n x d  (normalised rows)
    d = G.shape[1]

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Precompute the transposed matrices once for the chunked distance products.
    notH = (1 - H).T        # bits x n
    Ht = H.T               # bits x n

    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        A = H[i0:i1]                     # c x bits
        notA = 1 - A                     # c x bits
        # Hamming distance for every pair in this row-chunk.
        Dh = A @ notH + notA @ Ht        # c x n
        # Colour correlation for every pair in this row-chunk.
        C = (G[i0:i1] @ G.T) / d         # c x n
        # Tier A: pHash and colour agree.  Tier B: colour is near-identical.
        mask = ((Dh <= phash_thresh) & (C >= corr_thresh)) | (C >= strong_corr_thresh)
        ys, xs = np.nonzero(mask)
        for di, j in zip(ys.tolist(), xs.tolist()):
            i = i0 + di
            if j > i:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()), G


def pick_representative(members, G):
    """Return the medoid index: the member most similar to all the others."""
    if len(members) == 1:
        return members[0]
    sub = G[members]                              # m x d
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

    keep, bits_list, vecs_list = [], [], []
    for p in paths:
        b, v = features_of(p, args.hash_size, args.grid)
        if b is None:
            print(f"    [warn] unreadable: {os.path.basename(p)}")
            continue
        keep.append(p)
        bits_list.append(b)
        vecs_list.append(v)
    paths = keep
    if not paths:
        print("  [skip] nothing readable")
        return None

    print(f"  clustering {len(paths)} image(s) "
          f"(phash<= {args.phash_threshold} & corr>= {args.corr_threshold}, "
          f"or corr>= {args.strong_corr_threshold}) ...")
    groups, G = build_groups(paths, bits_list, vecs_list,
                             args.phash_threshold, args.corr_threshold,
                             args.strong_corr_threshold)

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

    rows = []          # (group_id, file, group_size, similarity, capture_time)
    n_groups = 0
    n_singletons = 0
    group_index = 0

    for members in groups:
        size = len(members)
        rep = pick_representative(members, G)
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
            sim = 1.0 if m == rep else float((G[m] @ G[rep]) / G.shape[1])
            place(paths[m], dest_dir)
            rows.append((gid, os.path.basename(paths[m]), size, round(sim, 3), ct(m)))

    # ---- CSV report ------------------------------------------------------ #
    csv_path = os.path.join(out_dir, "groups.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "file", "group_size", "similarity", "capture_time"])
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
        ws.cell(row=ws.max_row, column=1).fill = PatternFill("solid", fgColor=fill)

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
    p.add_argument("--phash-threshold", type=int, default=10,
                   help="max pHash Hamming distance to be a candidate (default 10)")
    p.add_argument("--corr-threshold", type=float, default=0.80,
                   help="min colour correlation when pHash also agrees (default 0.80)")
    p.add_argument("--strong-corr-threshold", type=float, default=0.95,
                   help="min colour correlation to group on its own, even if pHash "
                        "disagrees (default 0.95)")
    p.add_argument("--grid", type=int, default=16,
                   help="colour grid size for correlation, grid x grid (default 16)")
    p.add_argument("--min-group-size", type=int, default=2,
                   help="a group must have at least this many photos (default 2)")
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

    print(f"Found {len(folders)} folder(s) with images under '{root}'")

    summaries = []
    for folder in folders:
        # Group folders are created INSIDE each image folder.
        out_dir = os.path.join(folder, args.out_name)
        rel = os.path.relpath(folder, root)
        print()
        print("=" * 66)
        print(f" {rel}")
        print("=" * 66)
        s = process_folder(folder, out_dir, args)
        if s:
            summaries.append(s)

    print("\n=== Done ===")
    tot_images = sum(s["images"] for s in summaries)
    tot_groups = sum(s["groups"] for s in summaries)
    tot_single = sum(s["singletons"] for s in summaries)
    print(f"{tot_images} image(s) -> {tot_groups} group(s), "
          f"{tot_single} singleton(s) across {len(summaries)} folder(s).")


if __name__ == "__main__":
    main()