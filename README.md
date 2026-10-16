# group_similar

Group **visually similar photos** — the repeated takes a photoshoot produces
frame after frame (a model standing or smiling, clicked again and again) — so
every take of the same shot sits together in its own folder.

- **Runs 100% locally** — no cloud, no API key, no per-image fees. Cost is zero.
- **Non-destructive** — photos are never moved, renamed, or modified. They are
  only *copied* into a new output folder.
- **Reusable** — give it any folder; it walks every nested folder and groups the
  photos inside each one.

---

## How it works

Similarity is measured entirely with Pillow + numpy:

1. **pHash** — a coarse, rotation-normalised perceptual hash (fast to compare).
2. **Colour correlation on a grid** — the image is downscaled to a small RGB
   grid and each channel is contrast-normalised; similarity is the normalised
   dot product (correlation). This keeps far more real detail than a hash and is
   colour-aware, so a different backdrop/outfit counts as a different take.

Two photos join the same group when **either**:

- their pHash Hamming distance is `<= --phash-threshold` **and** their colour
  correlation is `>= --corr-threshold` (default `0.80`), **or**
- their colour correlation is `>= --strong-corr-threshold` (default `0.95`) — a
  near-identical colour layout groups even when the coarse pHash disagrees.

Groups are the connected components of that relation (union-find), so a whole
burst ends up in one folder even if the "bridge" frames aren't identical.

---

## Requirements

- Python 3
- Pillow, numpy, imagehash, openpyxl

Install everything at once (`python` on Windows, `python3` on macOS/Linux):

```bash
# macOS / Linux
python3 -m pip install -r requirements.txt

# Windows PowerShell
python -m pip install -r requirements.txt
```

---

## Usage

```bash
# just give a folder path — every folder under it that holds images is grouped
python3 group_similar.py "input_folder"

# or point at a single folder
python3 group_similar.py "input_folder/Paltu/Edit"
```

### Windows PowerShell

On Windows the command is `python` (not `python3`), and you run it from the
folder that contains `group_similar.py`. Backslashes (`\`) work in paths.

```powershell
# 1. go to the project folder (where group_similar.py lives)
cd "C:\Users\<you>\Documents\Work\IndiaPix_Similar_photo_group"

# 2. install the dependencies once
python -m pip install -r requirements.txt

# 3. run it — just pass the folder you want scanned, in quotes
python group_similar.py "input_folder"

# point at a single folder instead
python group_similar.py "input_folder\Paltu\Edit"

# current folder as the target
python group_similar.py "."
```

Notes for PowerShell:

- **Always quote the path** (`"input_folder"`). If a folder name contains a
  space (`"_sort images"`), the quotes are required, exactly as on macOS.
- `python` must be on your `PATH`. Check with `python --version`. If it prints
  `Python 3.x` you're ready; if Windows opens the Microsoft Store, install
  Python from <https://www.python.org/downloads/> and tick **"Add Python to
  PATH"** during setup.
- A full example with a space in the path:
  `python group_similar.py "C:\Photos\_sort images"`
- Each image folder gets its own `_groups\` output folder, e.g.
  `input_folder\Paltu\JPEG\_groups\group_0001\`.

### Options

| Flag | Default | Purpose |
|---|---|---|
| `path` | *(required)* | Folder to scan. Every nested folder that directly holds images is grouped, and a `<out-name>/` folder of groups is created **inside** it. |
| `--out-name` | `_groups` | Name of the group folder created inside each image folder |
| `--hash-size` | `8` | pHash bits per side (64-bit hash) |
| `--phash-threshold` | `10` | Max pHash Hamming distance for a candidate pair |
| `--corr-threshold` | `0.80` | Min colour correlation (when pHash also agrees) |
| `--strong-corr-threshold` | `0.95` | Min colour correlation to group on its own |
| `--grid` | `16` | Colour correlation grid size (grid × grid) |
| `--min-group-size` | `2` | A group must have at least this many photos |
| `--singletons` | `separate` | Where unmatched photos go: `separate`, `group`, or `skip` |

> **Tuning:** to group more aggressively, raise `--phash-threshold` or lower
> `--corr-threshold`. To be stricter, do the opposite. Start with the defaults.

---

## Output

For every folder that holds images, groups are created in a `_groups/`
sub-folder **inside that same folder**. Each group gets its own numbered
folder; frames with no similar partner go to `_singletons/` by default:

```
input_folder/
  Paltu/
    JPEG/                     ← a folder of photos
      _groups/
        group_0001/           ← first shot, every take together
          STAND_0.JPG
          STAND_1.JPG
          STAND_3.JPG
          STAND_2.JPG
        group_0002/
          SEAT_0.JPG
          SEAT_1.JPG
          ...
        _singletons/          ← frames that matched nothing
          misc_0.JPG
        groups.csv            ← one row per photo
        groups.xlsx           ← same, colour-coded per group
    Edit/                     ← another folder of photos -> its own _groups/
      _groups/
        ...
```

Source files are **only copied**, never moved, renamed or modified.

**`groups.csv` columns**

| Column | Meaning |
|---|---|
| `group_id` | Group folder number (`0001`, `0002`, …) or `_singletons` |
| `file` | File name |
| `group_size` | How many photos are in this group |
| `similarity` | Correlation of this photo to the group's representative (1.0 = the representative) |
| `capture_time` | EXIF `DateTimeOriginal`/`DateTime` when present (blank otherwise) |

Within a group the photos are ordered by capture time, so a burst reads in the
order it was shot.

---

## Why metadata isn't used for grouping

Grouping is purely visual so it needs no EXIF and works on any folder of images.
(If two files share a name, the later copy gets a numeric suffix so nothing is
overwritten.)