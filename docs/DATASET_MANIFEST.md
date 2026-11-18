# Dataset manifest specification (for the research assistants)

This page defines **exactly** how the photographs and geological journals must be delivered so the pipeline can read them without code changes. Before handing over, run `corelog validate`. It checks everything below and writes a report (`outputs/reports/validation.md`).

## 1. Folder layout

Deliver one folder (any name; it becomes `paths.data_root`):

```
<dataset_root>/
├── manifests/
│   ├── rows.csv              # one line per core ROW in a photo (required)
│   └── log_intervals.csv     # geological journal intervals (required for training)
└── photos/                   # any sub-folder structure is fine
    ├── HOLE_A/
    │   ├── HOLE_A_box001-004.jpg
    │   └── ...
    └── ...
```

- Paths in the CSVs are **relative to `<dataset_root>`** and use forward slashes (`photos/HOLE_A/xyz.jpg`).
- Accepted image formats: **TIFF or JPEG** (preferred; PNG also works). Canon `.CR2` can be read directly only if the `raw` extra is installed; converting to 8-bit TIFF with **identical settings for every photo** (same white balance, no auto-exposure) is safer.
- Do not edit or crop the original photographs. Pixel boxes in `rows.csv` do the cropping.
- Units: **metres**, depth measured **along hole** (as in the journals). Decimal point `.`, no thousands separators.
- CSV: UTF-8, comma-separated, header line exactly as below. Excel "CSV UTF-8" export is fine.

## 2. `rows.csv`: one line per core row

A core box usually holds several rows ("runs") of core. Each row visible in a photograph gets one line.

| Column | Required | Description |
|---|---|---|
| `hole_id` | yes | Drill hole, exactly as in the journals, e.g. `HOLE_A` |
| `photo_id` | yes | Unique ID of the photograph, e.g. `HOLE_A_P0001` |
| `image_path` | yes | Path relative to dataset root |
| `row_id` | yes | Row identifier within the photo, e.g. `1`, `2`, `3` (or `B2R1` for box 2 row 1). Unique within a photo. |
| `depth_from_m` | yes | Depth at the **start** of the row (shallow end) |
| `depth_to_m` | yes | Depth at the **end** of the row (deep end); must be > `depth_from_m` |
| `x0`,`y0`,`x1`,`y1` | recommended | Pixel box of the row: left, top, right, bottom (x to the right, y downwards, origin top-left). Draw it tightly around the core, excluding tray walls and labels. Leave all four empty if the image file is already a cropped single row. |
| `direction` | optional | `ltr` (default) if depth increases left → right in the image, `rtl` if right → left |
| `exclude` | optional | `1` to skip this row (photo blurred, core missing, wrong box…), otherwise `0` |
| `notes` | optional | Free text (core loss, broken core, wet surface, etc.) |

Example:

```csv
hole_id,photo_id,image_path,row_id,depth_from_m,depth_to_m,x0,y0,x1,y1,direction,exclude,notes
HOLE_A,HOLE_A_P0001,photos/HOLE_A/IMG_0412.tif,1,102.40,103.85,212,388,5620,560,ltr,0,
HOLE_A,HOLE_A_P0001,photos/HOLE_A/IMG_0412.tif,2,103.85,105.30,214,640,5618,812,ltr,0,
HOLE_A,HOLE_A_P0001,photos/HOLE_A/IMG_0412.tif,3,105.30,106.70,210,895,5616,1068,ltr,0,core loss 106.2-106.4
```

**Depth rules**
- Rows of one hole must not overlap in depth (validation error). Gaps are allowed and reported.
- If only box depths are known, split each box's interval over its rows **in proportion to the length of core in each row**, and write a note. Do not split equally when rows are not full.
- Depth is interpolated linearly along each row. Core loss inside a row cannot be represented, so describe it in `notes`. If it is large, split the row into two lines with two pixel boxes.

**How to get pixel boxes.** Any image viewer that shows cursor coordinates works (ImageJ/Fiji, GIMP, IrfanView). Record the top-left and bottom-right corners. The validator checks that each box lies inside the image.

## 3. `log_intervals.csv`: geological journal

| Column | Required | Description |
|---|---|---|
| `hole_id` | yes | Same IDs as `rows.csv` |
| `depth_from_m` | yes | Interval top |
| `depth_to_m` | yes | Interval bottom (> top) |
| `lithology_code` | yes | Code exactly as used in the journal (e.g. `GRN`, `SCH`) |
| `description` | optional | Free text |

```csv
hole_id,depth_from_m,depth_to_m,lithology_code,description
HOLE_A,102.40,104.10,SCH,chlorite schist
HOLE_A,104.10,104.65,QV,quartz vein with pyrite
HOLE_A,104.65,110.00,SCH,
```

- Intervals of one hole must not overlap. Unlogged gaps are fine; tiles there become "unlabeled".
- Use **one consistent code list**. Every code must be mapped to a model class in the experiment config (`labels.class_map`). The validator lists codes that are not mapped.
- If the journals have several attributes (lithology, alteration, mineralization), deliver one file per attribute (e.g. `log_alteration.csv` with the same columns) and say which one to use. The pipeline trains one attribute at a time (`paths.log_intervals`).

## 4. Before delivery: checklist

1. `rows.csv` and `log_intervals.csv` open in a text editor with the exact headers above.
2. Every `image_path` exists (case-sensitive on Linux!).
3. Hole IDs are spelled identically in both files.
4. `corelog validate -c configs/lab.yaml` reports **PASS**. Warnings are acceptable but should be read.
5. Note anything unusual (camera changes, wet versus dry core, re-photographed boxes) in a `README.txt` inside the dataset folder.

Empty templates: [`templates/rows_template.csv`](templates/rows_template.csv), [`templates/log_intervals_template.csv`](templates/log_intervals_template.csv).

## 5. What stays out of Git

The dataset folder **never** goes into the repository: no photos, CR2 files, journals or manifests with company data. Keep it on the lab machine's data disk and point `paths.data_root` to it.
