#!/usr/bin/env python3
"""
group_similar.py — split a folder of photos into exactly TWO output folders:

  distinct/    ONE representative of every look-alike (burst) group, plus every
               photo that has no look-alike at all.  Nothing in here is a near
               duplicate of anything else in here.
  duplicates/  every extra take — the near-twins of the representative that was
               kept in distinct/.

Similarity is measured 100% locally (no cloud, no cost, no API key) from image
content, with THREE deliberate design choices that matter on a large shoot:

  1. BACKGROUND-RESIDUAL comparison.  The median frame of the folder is
     estimated and subtracted before anything is compared, so the constant
     studio wall / floor / lighting cancels out and what is left is the SUBJECT
     (the pose).  Raw-frame comparison on a studio shoot mostly measures "same
     backdrop", which is why different poses used to get merged.

  2. CANDIDATE GENERATION instead of all-pairs.  Two photos are only compared
     when they are plausible candidates:
       * close in shot order / capture time (a burst is shot seconds apart), or
       * share most of their 64-bit pHash (multi-index hashing), or
       * the folder is small (<= --exact-max images), in which case every pair
         is compared exactly.
     That turns n*(n-1)/2 pairs (392 million for 28k photos) into roughly
     n * 20, with no dense n x n matrix in memory.

  3. COMPLETE-LINKAGE grouping, NOT connected components.  A photo may only
     join a group if it is alike to EVERY member already in it.  Single-linkage
     (connected components) chains A~B~C~D... and collapses an entire shoot
     into one group; complete linkage stops the chain, so genuinely different
     poses stay in distinct/.

Four complementary, content-aware signals are blended per candidate pair:

  1. pHash ............................ coarse, rotation-normalised, cheap;
                                        also drives candidate generation.
  2. Grayscale STRUCTURE correlation .. the dominant signal: the background
                                        residual is compared, so composition
                                        (the pose) is what is measured.
  3. Edge correlation ................. outlines of the residual subject.
  4. Colour correlation ............... residual colour detail (outfit, props).

Two photos are treated as "alike" (near duplicates) when EITHER:
  * their COMBINED similarity — a weighted blend of the four measures above,
    weighted by --w-hash / --w-gray / --w-edge / --w-color — is
    >= --similar-threshold (default 0.62),  -- or
  * their grayscale structure correlation alone is
    >= --strong-gray-threshold (default 0.78) (near-identical composition).

The representative of each group is its EARLIEST shot (by EXIF capture time,
falling back to file name) unless --keeper sharpest is given.

Usage:
  # every folder under it that holds images gets its own distinct/ + duplicates/
  python3 group_similar.py "input_folder"

  # or point at one shoot folder
  python3 group_similar.py "input_folder/Paltu/Edit"

Tuning (lower --similar-threshold to treat more photos as duplicates; raise it
to keep more of them in distinct/).
"""

import argparse
import csv
import errno
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Cached per-image fingerprints, kept inside the output folder so a re-run
# (e.g. while tuning a threshold, or after an interruption) does not re-decode
# thousands of images.  A DIRECTORY of incremental .npz shards, not one big
# file, so a long first pass is resumable.
CACHE_DIR = "_fingerprint_cache"

# How many fingerprints to buffer before writing a shard.  Smaller = less work
# lost if the run is killed; 1000 is a good balance.
CACHE_SHARD = 1000

# Also force a cache flush after this many seconds, so a slow I/O-bound run still
# checkpoints regularly (1000 big camera files can take minutes to read).
CACHE_FLUSH_SECONDS = 60.0

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


# --------------------------------------------------------------------------- #
# Per-image fingerprint (runs in worker processes)
# --------------------------------------------------------------------------- #
def _fingerprint(path, hash_size, struct_grid, color_grid, draft_px):
    """Return a dict describing one image, or ('error', reason).

    Decoding is deliberately cheap: JPEGs are decoded with DCT scaling
    (``Image.draft``) straight down to ~`draft_px`, so a 24 MP frame costs a
    fraction of a full decode.  Everything downstream works on small grids.

    Stored per image (all raw, background subtraction happens later so it can
    use the whole folder):
      * bits  — pHash bit array (hash_size*hash_size)
      * gray  — struct_grid x struct_grid float32 luminance
      * col   — 3 x color_grid x color_grid float32 RGB
      * sharp — variance of the Laplacian on `gray` (focus measure)
      * ctime — EXIF DateTimeOriginal/DateTime string, or ""
    """
    try:
        with Image.open(path) as im:
            ctime = ""
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

            # Cheap decode: let the JPEG decoder scale down as it works.
            try:
                im.draft("RGB", (draft_px, draft_px))
            except Exception:
                pass
            im = ImageOps.exif_transpose(im)
            if max(im.size) > draft_px:
                im = im.copy()
                im.thumbnail((draft_px, draft_px), Image.BILINEAR)

            g = im.convert("L")
            ph = imagehash.phash(g, hash_size=hash_size)
            gray = np.asarray(
                g.resize((struct_grid, struct_grid), Image.BILINEAR),
                dtype=np.float32)
            col = np.asarray(
                im.convert("RGB").resize((color_grid, color_grid), Image.BILINEAR),
                dtype=np.float32)

        gy, gx = np.gradient(gray)
        lap = np.gradient(gy, axis=0) + np.gradient(gx, axis=1)
        return {
            "bits": np.asarray(ph.hash, dtype=np.uint8).ravel(),
            "gray": gray.astype(np.float32),
            "col": col.astype(np.float32),
            "sharp": float(lap.var()),
            "ctime": ctime,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _fingerprint_job(args):
    """ProcessPoolExecutor adapter (single tuple argument)."""
    return _fingerprint(*args)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _cache_dir(out_dir):
    return os.path.join(out_dir, CACHE_DIR)


def _meta_path(cdir):
    return os.path.join(cdir, "meta.json")


def _shard_paths(cdir):
    try:
        names = sorted(n for n in os.listdir(cdir)
                       if n.startswith("shard_") and n.endswith(".npz"))
    except OSError:
        return []
    return [os.path.join(cdir, n) for n in names]


def _u8(a):
    """Store a 0-255 float grid as uint8 (the JPEG source is 8-bit anyway)."""
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def _load_cache(out_dir, meta):
    """Load every cached fingerprint that matches `meta`; {} otherwise.

    Shards are read with np.load — a few vectorised reads, far cheaper than
    parsing the single multi-hundred-megabyte JSON this replaced.  Because
    shards are written as we go, a run that was stopped at photo 20 000 of
    22 815 still contributes those 20 000 fingerprints.
    """
    cdir = _cache_dir(out_dir)
    try:
        with open(_meta_path(cdir), "r", encoding="utf-8") as f:
            if json.load(f) != meta:
                return {}
    except Exception:
        return {}

    out = {}
    for sp in _shard_paths(cdir):
        try:
            with np.load(sp, allow_pickle=False) as z:
                paths = z["paths"]
                bits, gray, col = z["bits"], z["gray"], z["col"]
                sharp, ctime = z["sharp"], z["ctime"]
                size, mtime = z["size"], z["mtime"]
                s2 = gray.shape[1]
                for k in range(len(paths)):
                    out[str(paths[k])] = {
                        "bits": bits[k].astype(np.uint8),
                        "gray": gray[k].astype(np.float32),
                        "col": col[k].astype(np.float32),
                        "sharp": float(sharp[k]),
                        "ctime": str(ctime[k]),
                        "stamp": (int(size[k]), int(mtime[k])),
                        "grid2": s2,
                    }
        except Exception as e:
            log.warning("cache shard '%s' unreadable: %s", sp, e)
    return out


class ShardWriter:
    """Append fingerprints to .npz shards, flushed every CACHE_SHARD photos.

    This periodic flush is what makes the long first pass RESUMABLE: stop the
    run part-way and the work already done is on disk.
    """

    def __init__(self, out_dir, meta, shard_size=CACHE_SHARD,
                 flush_seconds=CACHE_FLUSH_SECONDS):
        self.dir = _cache_dir(out_dir)
        self.shard_size = max(1, shard_size)
        self.flush_seconds = flush_seconds
        self.buf = []
        self.t_last = time.time()
        self.n = len(_shard_paths(self.dir))
        try:
            os.makedirs(self.dir, exist_ok=True)
            with open(_meta_path(self.dir), "w", encoding="utf-8") as f:
                json.dump(meta, f)
        except OSError as e:
            log.warning("cache directory unavailable: %s", e)

    def add(self, path, rec):
        self.buf.append((path, rec))
        if (len(self.buf) >= self.shard_size
                or time.time() - self.t_last >= self.flush_seconds):
            self.flush()

    def flush(self):
        if not self.buf:
            return
        rows = self.buf
        try:
            np.savez_compressed(
                os.path.join(self.dir, f"shard_{self.n:05d}.npz"),
                paths=np.array([p for p, _ in rows]),
                bits=np.stack([r["bits"] for _, r in rows]).astype(np.uint8),
                gray=np.stack([_u8(r["gray"]).ravel() for _, r in rows]),
                col=np.stack([_u8(r["col"]).ravel() for _, r in rows]),
                sharp=np.array([r["sharp"] for _, r in rows], dtype=np.float32),
                ctime=np.array([r["ctime"] for _, r in rows]),
                size=np.array([r["stamp"][0] for _, r in rows], dtype=np.int64),
                mtime=np.array([r["stamp"][1] for _, r in rows], dtype=np.int64),
            )
            self.n += 1
            log.debug("cache shard %d written (%d fingerprint(s))",
                      self.n, len(rows))
        except Exception as e:
            log.warning("cache shard write failed: %s", e)
        self.buf = []
        self.t_last = time.time()


def _stamp(path):
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)


# --------------------------------------------------------------------------- #
# Descriptors: background residual + normalisation
# --------------------------------------------------------------------------- #
def _z_rows(a, copy=True):
    """Zero-mean / unit-std every ROW of a 2-D array (each photo separately).

    With copy=False the mean is subtracted IN PLACE, which keeps peak memory
    flat on very large folders (the caller owns and discards the array).
    """
    a = np.array(a, dtype=np.float32, copy=copy)
    a -= a.mean(axis=1, keepdims=True)
    return a / (a.std(axis=1, keepdims=True) + 1e-6)


def _residual_z(raw, bg, eps=1e-3, chunk=2048):
    """Normalise each row of (raw - bg), falling back to the RAW row when the
    residual is degenerate.

    A row whose residual has no variance is one that *is* the background — the
    pathological case being a folder of identical photos, or a folder with a
    single photo.  There the background estimate equals every frame, the
    residual cancels to zero, and a straight normalisation would give a zero
    vector that matches nothing.  Falling back to the raw frame keeps such
    photos comparable (they are then trivially alike — which is correct).
    """
    n = raw.shape[0]
    out = np.empty_like(raw)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        block = raw[i0:i1] - bg
        bad = block.std(axis=1) < eps
        if bad.any():
            block[bad] = raw[i0:i1][bad]
        block -= block.mean(axis=1, keepdims=True)
        out[i0:i1] = block / (block.std(axis=1, keepdims=True) + 1e-6)
    return out


def build_descriptors(grays, cols, struct_grid, color_grid, sample=400, seed=0):
    """Turn raw per-image grids into background-residual descriptors.

    `grays` is (n, s*s) luminance, `cols` is (n, c*c*3) RGB, both raw.

    The folder's shared background is estimated as the per-pixel MEDIAN over a
    random sample of frames (bounded, so memory stays flat on huge folders).
    Subtracting it means the correlation that follows measures the SUBJECT —
    the pose — instead of "everyone shares this studio wall".

    Returns (R_gray, R_edge, R_col) — all row-normalised, ready for dot products.
    """
    n = grays.shape[0]
    s2, c2 = struct_grid * struct_grid, color_grid * color_grid
    if n == 0:
        return (np.zeros((0, s2), np.float32), np.zeros((0, s2), np.float32),
                np.zeros((0, c2), np.float32))

    rng = np.random.default_rng(seed)
    idx = (np.arange(n) if n <= sample
           else rng.choice(n, size=sample, replace=False))
    bg_gray = np.median(grays[idx], axis=0)
    bg_col = np.median(cols[idx], axis=0)

    # Structure: residual luminance.  Chunked so peak memory stays flat, and
    # with a raw-frame fallback for the degenerate rows described above.
    R_gray = _residual_z(grays, bg_gray)

    # Edges: gradient magnitude of the RESIDUAL, so backdrop seams vanish.
    g3 = R_gray.reshape(-1, struct_grid, struct_grid)
    dy, dx = np.gradient(g3, axis=(1, 2))
    R_edge = _z_rows(np.hypot(dx, dy).reshape(n, s2), copy=False)

    # Colour: residual per channel, each channel normalised, then concatenated.
    R_col = np.concatenate(
        [_residual_z(cols[:, ch::3], bg_col[ch::3]) for ch in range(3)],
        axis=1)

    return (R_gray.astype(np.float32, copy=False),
            R_edge.astype(np.float32, copy=False),
            R_col.astype(np.float32, copy=False))


# --------------------------------------------------------------------------- #
# Candidate generation (avoids the O(n^2) full comparison)
# --------------------------------------------------------------------------- #
def _pair_key(a, b):
    return (a, b) if a < b else (b, a)


def candidate_pairs(paths, bits, ctimes, order, window, max_seconds,
                    hash_bytes_keep, bucket_cap, exact_max):
    """Return a sorted list of index pairs that deserve an exact comparison.

    Three cheap sources, unioned:

      1. LOCAL BURST WINDOW — photos within `window` positions of each other in
         shot order, or within `max_seconds` of capture time.  Bursts are shot
         back to back, so this catches the overwhelming majority of true
         near-duplicates at n*window cost instead of n^2/2.

      2. pHASH MULTI-INDEX — split the 64-bit pHash into 8 bytes.  Pairs that
         agree on >= `hash_bytes_keep` of those bytes are candidates, which
         finds repeated shots that are far apart in time without comparing
         everything.

      3. EXACT MODE — when the folder is small (n <= exact_max) just compare
         every pair; the heuristic is only there to survive huge folders.
    """
    n = len(paths)
    if n <= 1:
        return []
    if n <= exact_max:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    pairs = set()

    # --- 1. local window in shot order -------------------------------- #
    for p in range(n):
        i = order[p]
        lo = order[max(0, p - window): p]
        for j in lo:
            pairs.add(_pair_key(i, j))
    # --- 1b. capture-time window -------------------------------------- #
    if max_seconds > 0:
        secs = [ctimes[i] for i in range(n)]
        timeable = [i for i in range(n) if secs[i]]
        if len(timeable) > 1:
            timeable.sort(key=lambda i: secs[i])
            for a in range(len(timeable)):
                i = timeable[a]
                for b in range(a + 1, len(timeable)):
                    j = timeable[b]
                    if _seconds_between(secs[i], secs[j]) > max_seconds:
                        break
                    pairs.add(_pair_key(i, j))

    # --- 2. pHash multi-index ----------------------------------------- #
    if hash_bytes_keep > 0 and bits is not None and bits.size:
        packed = np.packbits(bits.astype(np.uint8), axis=1)  # n x 8 bytes
        nb = packed.shape[1]
        keep = int(min(max(hash_bytes_keep, 1), nb))
        buckets = {}
        for i in range(n):
            row = packed[i]
            for k in range(nb):
                buckets.setdefault((k, int(row[k])), []).append(i)

        # Smallest buckets first: a byte value shared by thousands of photos
        # carries no signal, and the vote budget stops the expensive ones, so
        # this stage can never explode into an O(n^2) pair set on a 28k shoot.
        votes = {}
        budget = max(200_000, 40 * n)
        for members in sorted(buckets.values(), key=len):
            m = len(members)
            if m < 2 or m > bucket_cap:
                continue
            cost = m * (m - 1) // 2
            if cost > budget:
                continue
            budget -= cost
            for a in range(m):
                i = members[a]
                for b in range(a + 1, m):
                    key = _pair_key(i, members[b])
                    v = votes.get(key)
                    if v is None:
                        votes[key] = 1
                    elif v < keep:
                        votes[key] = v + 1
            if budget <= 0:
                break
        for key, v in votes.items():
            if v >= keep:
                pairs.add(key)

    return sorted(pairs)


def _seconds_between(a, b):
    """Rough EXIF timestamp distance in seconds (0 when unparseable)."""
    try:
        ta = time.strptime(a, "%Y:%m:%d %H:%M:%S")
        tb = time.strptime(b, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return 0.0
    return abs(time.mktime(ta) - time.mktime(tb))


# --------------------------------------------------------------------------- #
# Pairwise similarity on candidate pairs only
# --------------------------------------------------------------------------- #
def _scatter_max(n, idx, val):
    """best[i] = max(val) over every entry where idx == i.  Vectorised."""
    best = np.full(n, -np.inf, dtype=np.float32)
    if idx.size == 0:
        return best
    o = np.lexsort((-val, idx))
    idx_s = idx[o]
    first = np.ones(idx_s.size, dtype=bool)
    first[1:] = idx_s[1:] != idx_s[:-1]
    best[idx_s[first]] = val[o][first]
    return best


def score_pairs(pairs, RG, RE, RC, bits, weights, bits_per_hash,
                sim_threshold, strong_gray_threshold):
    """Score candidate pairs, keeping only what the grouping actually needs.

    Returns (alike, best_any, combined):
      * alike    -- {(i, j): (combined, gray, edge, color, hash)} for the pairs
                    that PASS the alike test.  Only this small subset is kept,
                    so a 28k shoot does not hold ~840k score tuples in RAM.
      * best_any -- float32 (n,): each photo's best combined score against
                    anything it was compared with (used by the report).
      * combined -- float32 (m,): every candidate's combined score, only used
                    to print the distribution for calibration.

    No n x n matrix is ever built; pairs are scored in ~4096-pair buffers.
    """
    w_hash, w_gray, w_edge, w_color = weights
    n = RG.shape[0]
    dg = RG.shape[1]
    de = RE.shape[1]
    dc = RC.shape[1]
    alike = {}
    I_all, J_all, S_all = [], [], []
    buf = 4096
    for s in range(0, len(pairs), buf):
        chunk = pairs[s:s + buf]
        I = np.fromiter((p[0] for p in chunk), dtype=np.int64, count=len(chunk))
        J = np.fromiter((p[1] for p in chunk), dtype=np.int64, count=len(chunk))
        Cg = np.einsum("ij,ij->i", RG[I], RG[J]) / dg
        Ce = np.einsum("ij,ij->i", RE[I], RE[J]) / de
        Cc = np.einsum("ij,ij->i", RC[I], RC[J]) / dc
        if bits is not None and bits.size:
            Dh = np.bitwise_xor(bits[I], bits[J]).sum(axis=1)
            Ch = 1.0 - Dh.astype(np.float32) / bits_per_hash
        else:
            Ch = np.zeros(len(chunk), dtype=np.float32)
        S = (w_hash * Ch + w_gray * Cg + w_edge * Ce + w_color * Cc)

        I_all.append(I)
        J_all.append(J)
        S_all.append(S)

        keep = (S >= sim_threshold) | (Cg >= strong_gray_threshold)
        if keep.any():
            for k in np.nonzero(keep)[0]:
                alike[(int(I[k]), int(J[k]))] = (
                    float(S[k]), float(Cg[k]), float(Ce[k]),
                    float(Cc[k]), float(Ch[k]))

    if I_all:
        idx = np.concatenate(I_all + J_all)
        val = np.concatenate(S_all + S_all)
        best_any = _scatter_max(n, idx, val)
    else:
        best_any = np.full(n, -np.inf, dtype=np.float32)
    combined = (np.concatenate(S_all) if S_all
                else np.zeros(0, dtype=np.float32))
    return alike, best_any, combined


# --------------------------------------------------------------------------- #
# Grouping: complete linkage (no chaining)
# --------------------------------------------------------------------------- #
def cluster_photos(paths, order, alike, link_mode):
    """Group photos into look-alike sets without single-linkage chaining.

    `alike` already holds ONLY the pairs that passed the similarity test (see
    score_pairs), so a non-duplicate pair is never seen here.  How those alike
    edges are turned into groups decides everything:

      * complete (default) — a photo joins a group ONLY if it is alike to EVERY
        member already in that group.  Stops A~B~C~D chains from collapsing an
        entire shoot into one group.
      * leader — a photo joins if it is alike to the group's representative.
        Cheaper, but allows slow drift.
      * single — classic connected components (old behaviour); kept only for
        comparison, because it is what merges different poses.

    Groups are built in shot order, so each group's first member is its
    earliest take — the natural keeper.
    """
    nbrs = {}
    for (i, j) in alike:
        nbrs.setdefault(i, set()).add(j)
        nbrs.setdefault(j, set()).add(i)

    cluster_of = [-1] * len(paths)
    members = []
    reps = []

    for i in order:
        cands = nbrs.get(i, ())
        if not cands:
            cluster_of[i] = len(members)
            members.append([i])
            reps.append(i)
            continue

        # group the candidate neighbours by the cluster they already sit in
        by_cluster = {}
        for j in cands:
            cj = cluster_of[j]
            if cj >= 0:
                by_cluster.setdefault(cj, []).append(j)
        if not by_cluster:
            cluster_of[i] = len(members)
            members.append([i])
            reps.append(i)
            continue

        best_c, best_score = None, -1.0
        for cj, js in by_cluster.items():
            sims = [alike[_pair_key(i, j)] for j in js]
            if not sims:
                continue
            if link_mode == "complete":
                # Must be alike to EVERY current member.  A member that is not
                # in `js` was never a candidate for `i`, i.e. not alike, so a
                # partial overlap is not enough — this is exactly what stops
                # A~B~C chaining from swallowing a whole shoot.
                if len(js) != len(members[cj]):
                    continue
                quality = min(s[0] for s in sims)
            elif link_mode == "leader":
                rep_key = _pair_key(i, reps[cj])
                if rep_key not in alike:
                    continue
                quality = alike[rep_key][0]
            else:  # single (connected components) — legacy behaviour
                quality = max(s[0] for s in sims)
            if quality > best_score:
                best_c, best_score = cj, quality

        if best_c is None:
            cluster_of[i] = len(members)
            members.append([i])
            reps.append(i)
        else:
            cluster_of[i] = best_c
            members[best_c].append(i)

    return members, cluster_of


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


def _same_file(a, b):
    """True when `a` and `b` are the same entry on disk (not two copies)."""
    try:
        sa, sb = os.stat(a), os.stat(b)
    except OSError:
        return False
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


def _same_content(a, b):
    """Byte-for-byte comparison, size short-circuit first.

    Used only to recognise a photo that an EARLIER run already placed in the
    output folder, so a re-run in move mode finishes the move (drops the now
    redundant source copy) instead of leaving the folder half full.
    """
    if _same_file(a, b):
        return True
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
        with open(a, "rb") as fa, open(b, "rb") as fb:
            while True:
                ba, bb = fa.read(1 << 20), fb.read(1 << 20)
                if ba != bb:
                    return False
                if not ba:
                    return True
    except OSError:
        return False


def _transfer(src, dest, move):
    """Put one photo at `dest` — by moving it (move=True) or copying it.

    Moving is the DEFAULT: a 28k-photo shoot would otherwise be stored twice.
      * same volume  -> os.rename, which is instant and needs no extra space;
      * other volume -> shutil.move (copy, then unlink the source).
    Copying (--copy) leaves the originals where they are.

    `dest` is never allowed to be silently overwritten: the caller has already
    picked a free name, and this re-checks so a race or a stale name can never
    destroy the only copy of a photo.
    """
    if os.path.exists(dest) and not _same_file(src, dest):
        raise FileExistsError(errno.EEXIST,
                              f"refusing to overwrite {dest}", dest)
    if not move:
        shutil.copy2(src, dest)
        return
    try:
        os.rename(src, dest)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        shutil.move(src, dest)


def place_photo(src, dest_dir, move=True):
    """Put one photo into `dest_dir` and report what actually happened.

    Returns one of:
      'moved'  — the file was transferred and the source entry is gone
                 (os.rename, or copy+unlink across volumes);
      'copied' — copy mode: the source was left alone;
      'deduped'— an IDENTICAL copy was already in `dest_dir` from an earlier
                 run, so the now-redundant source entry was deleted (this is
                 what finishes a move that a previous copy-run left half done);
      'kept'   — copy mode, identical copy already present, nothing to do.

    A photo is never overwritten: an occupied name with different content gets
    a numeric suffix, and an occupied name with the SAME content is treated as
    "already placed" rather than being replaced.
    """
    os.makedirs(dest_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(src))
    k = 0
    while True:
        name = f"{stem}{ext}" if k == 0 else f"{stem}_{k}{ext}"
        target = os.path.join(dest_dir, name)
        if not os.path.exists(target):
            _transfer(src, target, move)
            return "moved" if move else "copied"
        if _same_content(src, target):
            # Already here from an earlier run.  In move mode the source copy is
            # now redundant, so drop it — that completes the interrupted move.
            if move and not _same_file(src, target):
                os.remove(src)
                return "deduped"
            return "kept"
        k += 1
        if k > 9999:
            raise RuntimeError(f"too many name collisions for {src}")


def _prune_stale(out_dir, intended, move, dry):
    """Delete a photo's copy when it sits in the WRONG output folder.

    Move mode never wipes `distinct/` or `duplicates/` (they may be the only
    copy left), so a re-run that classifies a photo differently — or a
    copy-run followed by a move-run — would otherwise leave the photo in both
    folders.  A file is only removed when the correct placement exists AND the
    two files are byte-identical, so this can never delete a different photo.
    """
    if dry or not move:
        return 0
    removed = 0
    for src, label in intended.items():
        other = (GROUP_DUPLICATES if label == GROUP_DISTINCT
                 else GROUP_DISTINCT)
        base = os.path.basename(src)
        stale = os.path.join(out_dir, other, base)
        good = os.path.join(out_dir, label, base)
        if (os.path.isfile(stale) and os.path.isfile(good)
                and _same_content(stale, good)):
            try:
                os.remove(stale)
                removed += 1
                log.info("pruned stale placement: %s/%s (moved to %s/)",
                         other, base, label)
            except OSError as e:
                log.warning("could not prune %s: %s", stale, e)
    return removed


def _reset_output(out_dir, keep_existing=False, source_has_images=True):
    """Remove this run's own previously produced folders so no stale copies stay.

    Only our output directory is touched: `distinct/`, `duplicates/` and the
    `group_*` folders written by older versions.  Source photos and the
    fingerprint cache are never touched.

    Two data-loss guards, because a re-run must never destroy photos:
      * `keep_existing`  -- set whenever this run will MOVE photos, since the
        folders may already hold originals moved by an earlier run.
      * `source_has_images` -- if the source folder no longer holds any photos,
        the output folders ARE the collection (a previous move emptied the
        source), so they are left untouched and only reports are refreshed.
    """
    if keep_existing or not os.path.isdir(out_dir):
        return
    if not source_has_images:
        print("  [safe] output folders already hold the moved photos and the "
              "source folder is empty\n"
              "         -> they are NOT cleared; only the reports are "
              "refreshed")
        log.info("output kept: source folder has no images left (moved state)")
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
# Fingerprinting phase (parallel + cached)
# --------------------------------------------------------------------------- #
def fingerprint_folder(paths, out_dir, args):
    """Return (kept_paths, records) — decoding in parallel, with a disk cache."""
    meta = {
        "hash_size": args.hash_size,
        "struct_grid": args.struct_grid,
        "grid": args.grid,
        "draft_px": args.draft_px,
        "version": 2,
    }
    cache = {} if args.no_cache else _load_cache(out_dir, meta)
    records = {}
    todo = []
    t0 = time.time()
    for p in paths:
        ap = os.path.abspath(p)
        try:
            stamp = _stamp(ap)
            nbytes = os.path.getsize(ap)
        except OSError:
            continue
        hit = cache.get(ap)
        if hit and hit["stamp"] == stamp:
            records[ap] = hit
        else:
            todo.append((ap, stamp, nbytes))

    cached = len(records)
    if cached:
        print(f"  [1/3] {cached} fingerprint(s) from cache, "
              f"{len(todo)} to decode ...")
    else:
        print(f"  [1/3] fingerprinting {len(todo)} image(s) ...")

    # Bytes still to read.  This is what governs wall time: the decode is cheap
    # (DCT-scaled), so on big camera files the phase is bound by disk throughput.
    bytes_todo = sum(b for _, _, b in todo)
    bytes_done = 0
    if todo:
        print(f"  [1/3] {bytes_todo / 1073741824:.1f} GB to read for "
              f"{len(todo)} image(s); decoding on "
              f"{max(1, args.jobs)} process(es)")

    writer = None if args.no_cache else ShardWriter(out_dir, meta)

    def _done(k, ap, rec, stamp, nbytes):
        nonlocal bytes_done
        bytes_done += nbytes
        rec["stamp"] = stamp
        records[ap] = rec
        if writer is not None:
            writer.add(ap, rec)
        if k % 25 == 0 or k == len(todo):
            el = max(time.time() - t0, 1e-6)
            rate = k / el
            mbs = bytes_done / 1048576 / el
            print(f"    [{k}/{len(todo)}] {rate:.1f} img/s, "
                  f"{mbs:.0f} MB/s, ~{(len(todo) - k) / rate:.0f}s left - "
                  f"{os.path.basename(ap)}")

    if todo:
        jobs = max(1, args.jobs)
        if jobs == 1:
            for k, (ap, stamp, nbytes) in enumerate(todo, 1):
                rec = _fingerprint(ap, args.hash_size, args.struct_grid,
                                   args.grid, args.draft_px)
                if "error" in rec:
                    print(f"    [warn] unreadable: {os.path.basename(ap)}")
                    log.warning("SKIP unreadable: %s (%s)", ap, rec["error"])
                    continue
                _done(k, ap, rec, stamp, nbytes)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(_fingerprint_job,
                                  (ap, args.hash_size, args.struct_grid,
                                   args.grid, args.draft_px)):
                        (ap, stamp, nbytes)
                        for ap, stamp, nbytes in todo}
                done = 0
                for fut in as_completed(futs):
                    ap, stamp, nbytes = futs[fut]
                    done += 1
                    rec = fut.result()
                    if rec.get("error"):
                        print(f"    [warn] unreadable: {os.path.basename(ap)}")
                        log.warning("SKIP unreadable: %s (%s)",
                                    ap, rec["error"])
                        continue
                    _done(done, ap, rec, stamp, nbytes)

    if writer is not None:
        writer.flush()

    if todo:
        el = max(time.time() - t0, 1e-6)
        mbs = bytes_done / 1048576 / el
        print(f"  [1/3] throughput: {len(todo) / el:.1f} img/s, "
              f"{mbs:.0f} MB/s over {bytes_done / 1073741824:.1f} GB")
        log.info("fingerprint throughput: %.1f img/s, %.0f MB/s, %.1f GB read",
                 len(todo) / el, mbs, bytes_done / 1073741824)
        if mbs < 60:
            print("  [1/3] note: that is slow for a local disk, so this phase "
                  "is I/O-bound and raising --jobs will NOT help much. It is "
                  "paid ONCE - the cache makes every later run skip it.")

    kept = [p for p in paths if os.path.abspath(p) in records]
    return kept, records


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

    if args.max_images and len(paths) > args.max_images:
        print(f"  [limit] --max-images {args.max_images}: analysing the first "
              f"{args.max_images} of {len(paths)} photo(s)")
        log.info("LIMIT --max-images: %d of %d photo(s)", args.max_images,
                 len(paths))
        paths = paths[:args.max_images]

    # ---- Phase 1: fingerprint every image (parallel, cached) -------------- #
    paths, records = fingerprint_folder(paths, out_dir, args)
    if not paths:
        print("  [skip] nothing readable")
        return None
    print(f"  [1/3] done in {time.time() - t0:.1f}s "
          f"({len(paths)} fingerprint(s))")

    n = len(paths)
    recs = [records[os.path.abspath(p)] for p in paths]
    ctimes = [r["ctime"] for r in recs]
    bits = np.stack([r["bits"] for r in recs]).astype(np.uint8)
    grays = np.stack([r["gray"].ravel() for r in recs]).astype(np.float32)
    cols = np.stack([r["col"].reshape(-1) for r in recs]).astype(np.float32)
    sharp = np.array([r["sharp"] for r in recs], dtype=np.float32)

    # shot order: EXIF capture time, then file name
    order = sorted(range(n),
                   key=lambda i: (ctimes[i] or "~", os.path.basename(paths[i])))

    # ---- Phase 2: residual descriptors + candidate comparison ------------- #
    pairs_all = n * (n - 1) // 2
    print(f"  [2/3] building descriptors ({n} image(s), "
          f"{pairs_all} pairs if exhaustive) ...")
    t2 = time.time()
    RG, RE, RC = build_descriptors(grays, cols, args.struct_grid, args.grid,
                                   seed=0)
    del grays, cols

    pairs = candidate_pairs(paths, bits, ctimes, order,
                            window=args.cand_window,
                            max_seconds=args.cand_seconds,
                            hash_bytes_keep=args.cand_hash_bytes,
                            bucket_cap=args.cand_bucket_cap,
                            exact_max=args.exact_max)
    print(f"  [2/3] comparing {len(pairs)} candidate pair(s) "
          f"({100.0 * len(pairs) / max(pairs_all, 1):.2f}% of exhaustive); "
          f"alike if combined >= {args.similar_threshold} "
          f"(w h/g/e/c = {args.w_hash}/{args.w_gray}/{args.w_edge}/{args.w_color}) "
          f"or structure >= {args.strong_gray_threshold}, "
          f"link={args.link_mode}")

    weights = (args.w_hash, args.w_gray, args.w_edge, args.w_color)
    alike, best_any, comb = score_pairs(
        pairs, RG, RE, RC, bits, weights, args.hash_size * args.hash_size,
        args.similar_threshold, args.strong_gray_threshold)
    del RG, RE, RC

    if comb.size:
        print(f"  [2/3] {len(alike)} alike pair(s); candidate similarity "
              f"combined p50={np.percentile(comb, 50):.2f} "
              f"p90={np.percentile(comb, 90):.2f} "
              f"p99={np.percentile(comb, 99):.2f} max={comb.max():.2f}")
        log.info("candidate similarity: combined p50=%.3f p90=%.3f p99=%.3f "
                 "max=%.3f over %d candidate pair(s); %d alike pair(s)",
                 float(np.percentile(comb, 50)),
                 float(np.percentile(comb, 90)),
                 float(np.percentile(comb, 99)), float(comb.max()),
                 comb.size, len(alike))

    members, cluster_of = cluster_photos(paths, order, alike, args.link_mode)

    # keeper per group: earliest shot (default) or sharpest
    rank = {i: p for p, i in enumerate(order)}
    distinct, duplicates, info = [], [], {}
    comps = []
    for gi, mem in enumerate(members, 1):
        mem_sorted = sorted(mem, key=lambda i: (ctimes[i] or "~",
                                                os.path.basename(paths[i])))
        if args.keeper == "sharpest":
            rep = max(mem, key=lambda i: (float(sharp[i]), -rank[i]))
        else:
            rep = mem_sorted[0]
        gid = f"LA_{gi:04d}"
        for m in mem_sorted:
            info[m] = (gid, m == rep, rep)
        distinct.append(rep)
        duplicates.extend(m for m in mem_sorted if m != rep)
        comps.append(mem_sorted)

    comps.sort(key=lambda c: (ctimes[c[0]] or "~", os.path.basename(paths[c[0]])))
    distinct.sort(key=lambda i: (ctimes[i] or "~", os.path.basename(paths[i])))
    duplicates.sort(key=lambda i: (ctimes[i] or "~", os.path.basename(paths[i])))
    n_multi = sum(1 for c in comps if len(c) > 1)
    print(f"  [2/3] done in {time.time() - t2:.1f}s "
          f"({n_multi} look-alike group(s) -> {len(distinct)} distinct, "
          f"{len(duplicates)} duplicate(s))")
    log.info("[2/3] split %d image(s) in %.1fs: %d look-alike group(s), "
             "%d kept in distinct/, %d in duplicates/",
             n, time.time() - t2, n_multi, len(distinct), len(duplicates))
    _log_decisions(paths, comps, info)

    # ---- Phase 3: place the photos + write the reports -------------------- #
    # MOVE is the default: a separated shoot should not exist twice on disk.
    # --copy opts back into the old non-destructive behaviour.
    move = not bool(getattr(args, "copy", False))
    dry = bool(getattr(args, "dry_run", False))

    if dry:
        print("  [3/3] DRY RUN - nothing is written or moved "
              f"({GROUP_DISTINCT}/ and {GROUP_DUPLICATES}/ untouched)")
    elif move:
        # NEVER delete here.  In move mode these folders can contain the only
        # remaining copy of photos moved by an earlier run; a re-run must add
        # to them, never clear them.
        os.makedirs(out_dir, exist_ok=True)
        print(f"  [3/3] MOVING photos into 2 folder(s) to '{out_dir}' "
              f"(originals leave the source folder; use --copy to keep them "
              f"in place) ...")
        log.info("MOVE mode (default): existing output kept, nothing deleted")
    else:
        _reset_output(out_dir, source_has_images=bool(paths))
        os.makedirs(out_dir, exist_ok=True)
        print(f"  [3/3] COPYING photos into 2 folder(s) to '{out_dir}' "
              f"(originals stay where they are) ...")
        log.info("COPY mode (--copy): source folder left untouched")

    intended = {}
    stats = {}
    for label, mem in ((GROUP_DISTINCT, distinct),
                       (GROUP_DUPLICATES, duplicates)):
        dest_dir = os.path.join(out_dir, label)
        if not dry:
            os.makedirs(dest_dir, exist_ok=True)
        for m in mem:
            src = paths[m]
            intended[src] = label
            if dry:
                continue
            st = place_photo(src, dest_dir, move)
            stats[st] = stats.get(st, 0) + 1

    if move and not dry:
        placed = stats.get("moved", 0) + stats.get("deduped", 0)
        left = sum(1 for p in paths if os.path.exists(p))
        print(f"  moved            : {placed} photo(s) now live only in "
              f"{os.path.basename(out_dir)}/")
        if stats.get("deduped"):
            print(f"  finished moves   : {stats['deduped']} photo(s) were "
                  f"already in the output folder - the leftover source copy "
                  f"was removed")
        if left:
            print(f"  [warn] {left} source file(s) could not be removed")
            log.warning("%d source file(s) remain after the move", left)
        pruned = _prune_stale(out_dir, intended, move, dry)
        if pruned:
            print(f"  corrected        : {pruned} photo(s) removed from the "
                  f"wrong output folder (re-classified by this run)")
        log.info("MOVE mode: %d photo(s) live in the output folder only, "
                 "%d stale placement(s) pruned", placed, pruned)
    elif not dry:
        log.info("COPY mode: %d photo(s) copied, originals untouched",
                 stats.get("copied", 0))

    # Highest similarity seen for each photo: against the other takes of its own
    # group, or — for a photo with no second take — against anything it was
    # compared with.  `best_any` already came out of the scoring pass, so this
    # stays O(pairs) and never scans all n photos per image.
    comp_size = {}
    max_sim = {}
    for comp in comps:
        cid = info[comp[0]][0]
        comp_size[cid] = len(comp)
        for m in comp:
            if len(comp) > 1:
                max_sim[m] = max(
                    (alike[_pair_key(m, o)][0] for o in comp
                     if o != m and _pair_key(m, o) in alike), default=0.0)
            else:
                max_sim[m] = float(best_any[m]) if np.isfinite(best_any[m]) else 0.0

    rows = []          # [group, file, group_id, group_size, max_sim, dup_of, time]
    for label, mem in ((GROUP_DISTINCT, distinct),
                       (GROUP_DUPLICATES, duplicates)):
        for m in mem:
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

    csv_path = os.path.join(out_dir, "report.csv")
    xlsx_path = os.path.join(out_dir, "report.xlsx")

    if dry:
        # A dry run touches nothing at all - it only reports the plan.
        print(f"  (dry run) would write {csv_path}")
        print(f"  (dry run) would write {xlsx_path}")
    else:
        # ---- CSV report -------------------------------------------------- #
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)

        # ---- Excel report ------------------------------------------------ #
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
        "images": n,
        "distinct": len(distinct),
        "duplicates": len(duplicates),
        "groups": n_multi,
        "pairs": len(pairs),
        "csv": csv_path,
    }


def _log_decisions(paths, comps, info):
    """Write a human-readable account of every look-alike group."""
    multi = [c for c in comps if len(c) > 1]
    log.info("look-alike groups: %d group(s) with >1 take, %d unique photo(s)",
             len(multi), len(comps) - len(multi))
    for comp in sorted(multi, key=len, reverse=True):
        gid, _is_rep, rep = info[comp[0]]
        extras = ", ".join(os.path.basename(paths[i])
                           for i in comp if i != rep)
        log.info("  GROUP %s size=%d  distinct: %s  duplicates: %s",
                 gid, len(comp), os.path.basename(paths[rep]), extras)


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
    p.add_argument("--similar-threshold", type=float, default=0.62,
                   help="min COMBINED similarity (blend of pHash, structure, "
                        "edge and colour, measured on the background residual) "
                        "at which two photos are treated as the same shot; "
                        "lower = more photos counted as duplicates / fewer kept "
                        "as distinct (default 0.62)")
    p.add_argument("--strong-gray-threshold", type=float, default=0.78,
                   help="min grayscale structure correlation at which two "
                        "photos are duplicates on its own (default 0.78)")
    p.add_argument("--w-hash", type=float, default=0.05,
                   help="weight of pHash similarity in the combined score "
                        "(default 0.05)")
    p.add_argument("--w-gray", type=float, default=0.65,
                   help="weight of grayscale structure correlation on the "
                        "background residual — the main signal that tracks the "
                        "pose (default 0.65)")
    p.add_argument("--w-edge", type=float, default=0.15,
                   help="weight of residual edge/outline correlation "
                        "(default 0.15)")
    p.add_argument("--w-color", type=float, default=0.15,
                   help="weight of residual colour correlation (default 0.15)")
    p.add_argument("--grid", type=int, default=16,
                   help="colour correlation grid size, grid x grid (default 16)")
    p.add_argument("--struct-grid", type=int, default=48,
                   help="grayscale structure grid size, grid x grid "
                        "(default 48)")
    p.add_argument("--link-mode", choices=("complete", "leader", "single"),
                   default="complete",
                   help="how groups are formed: 'complete' (default) joins a "
                        "photo only if it is alike to EVERY member (no "
                        "chaining); 'leader' joins if alike to the group's "
                        "keeper (allows drift); 'single' = connected components "
                        "(old behaviour, chains different poses together)")
    p.add_argument("--keeper", choices=("earliest", "sharpest"),
                   default="earliest",
                   help="which take is kept in distinct/ (default: earliest)")
    p.add_argument("--cand-window", type=int, default=12,
                   help="compare each photo with the N photos around it in shot "
                        "order (default 12)")
    p.add_argument("--cand-seconds", type=float, default=30.0,
                   help="also compare photos taken within this many seconds of "
                        "each other (default 30; 0 disables)")
    p.add_argument("--cand-hash-bytes", type=int, default=5,
                   help="pHash multi-index: compare pairs agreeing on at least "
                        "this many of the 8 hash bytes, to catch repeated shots "
                        "far apart in time (default 5; 0 disables)")
    p.add_argument("--cand-bucket-cap", type=int, default=400,
                   help="ignore pHash byte buckets larger than this many photos "
                        "(they carry no signal and would explode the pair "
                        "count) (default 400)")
    p.add_argument("--exact-max", type=int, default=600,
                   help="folders with at most this many photos are compared "
                        "exhaustively (every pair), no candidate heuristics "
                        "(default 600)")
    p.add_argument("--jobs", type=int,
                   default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                   help="parallel decoding processes (default: CPU count - 1, "
                        "capped at 8)")
    p.add_argument("--draft-px", type=int, default=384,
                   help="decode JPEGs straight down to this size for "
                        "fingerprinting (default 384; bigger = slower, no real "
                        "gain)")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore the fingerprint cache and re-decode everything "
                        "(normally NOT needed: shards are reused automatically)")
    p.add_argument("--max-images", type=int, default=0,
                   help="fingerprint/analyse only the first N photos of each "
                        "folder (default 0 = all). Useful for a quick "
                        "threshold trial on a huge folder")
    p.add_argument("--copy", action="store_true",
                   help="COPY the photos into distinct/ + duplicates/ and "
                        "leave the originals in the source folder. NOT the "
                        "default: moving is, because a separated shoot should "
                        "not be stored twice on disk")
    p.add_argument("--move", action="store_true",
                   help="(kept for compatibility; moving is already the "
                        "default)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="skip the confirmation prompt (moving is the default, "
                        "so this makes the run non-interactive)")
    p.add_argument("--dry-run", action="store_true",
                   help="analyse and print what WOULD happen, writing and "
                        "moving nothing. Use this to sanity-check a move")
    return p


def main():
    args = build_parser().parse_args()
    root = args.path
    if not os.path.isdir(root):
        sys.exit(f"error: '{root}' is not a directory")

    folders = find_image_folders(root, args.out_name)
    if not folders:
        # A previous --move leaves no images in the source folders, so this is
        # a state to explain rather than a plain error.
        emptied = []
        for dirpath, dirnames, _files in os.walk(root):
            out = os.path.join(dirpath, args.out_name)
            if os.path.basename(dirpath) == args.out_name:
                continue
            if any(os.path.isdir(os.path.join(out, n))
                   for n in (GROUP_DISTINCT, GROUP_DUPLICATES)):
                emptied.append(out)
        if emptied:
            print(f"Nothing left to split under '{root}': the photos have "
                  f"already been moved out of the source folder(s):")
            for out in emptied:
                print(f"  - {out}")
            print("\nThe split is already done - the photos now live in "
                  "distinct/ and duplicates/ inside those folders.\n"
                  "To re-tune thresholds, move them back into the source "
                  "folder first (or re-run on a fresh copy of the originals)," 
                  " then use --dry-run before committing with --move.")
            sys.exit(0)
        sys.exit(f"error: no folders containing images found under '{root}'")

    if not args.copy and not args.dry_run:
        # Moving is irreversible for the source folder, so make the user say so.
        print("MOVE mode (default): photos will be MOVED out of the source "
              "folders into distinct/ and duplicates/.\n"
              "Nothing is deleted, but the originals will no longer be in "
              "their original folder. Use --copy to keep them in place, or "
              "--dry-run to preview the grouping first.")
        if args.yes:
            print("  --yes given: continuing without a prompt")
            log.info("move confirmed via --yes")
        else:
            try:
                answer = input("Continue? [Y/n] ").strip().lower()
            except EOFError:
                answer = "y"
                print("  no console available -> continuing (pass --yes to "
                      "silence this, or --dry-run to preview)")
            if answer not in ("", "y", "yes"):
                sys.exit("aborted: nothing was moved")

    log_path = setup_logging(os.path.join(root, "group_similar.log"))
    log.info("=== group_similar run start ===")
    log.info("root=%s  out_name=%s  python=%s", os.path.abspath(root),
             args.out_name, sys.version.split()[0])
    log.info("settings: hash_size=%d alike_if_combined>=%.2f weights h/g/e/c="
             "%.2f/%.2f/%.2f/%.2f strong_gray>=%.2f grid=%d struct_grid=%d "
             "link=%s keeper=%s draft_px=%d jobs=%d cand_window=%d "
             "cand_seconds=%.0f cand_hash_bytes=%d exact_max=%d place=%s",
             args.hash_size, args.similar_threshold, args.w_hash, args.w_gray,
             args.w_edge, args.w_color, args.strong_gray_threshold, args.grid,
             args.struct_grid, args.link_mode, args.keeper, args.draft_px,
             args.jobs, args.cand_window, args.cand_seconds,
             args.cand_hash_bytes, args.exact_max,
             "copy" if args.copy else "move")
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
