r"""
Before running this file, run the following lines


build_labels_csv.py
Build a self-contained CSV that pairs every image in the survey-line inference
layout with (a) its survey-line name and (b) its ground-truth label.

INPUTS (edit the paths below if yours differ):
  NORMAL_TXT: list of filenames currently in Inference_with_separate_folders/normal
  TUNNEL_TXT: list of filenames currently in Inference_with_separate_folders/tunnel
  LINE_TXT  : list of FULL PATHS to files in Inference_with_line_folders/,
              generated with `Get-ChildItem ... | ExpandProperty FullName`

OUTPUT: inference_labels.csv with columns:
    survey_line, filename, label
where label is 0 for normal and 1 for tunnel.

The eval script will read this CSV and evaluate every image, grouping by
survey_line for spatial aggregation.

At the end the script prints a summary so you can verify counts match your
File Explorer numbers.
"""

from pathlib import Path, PureWindowsPath
import csv


SCRIPT_DIR = Path(__file__).resolve().parent

# EDIT THESE if you put the txt files somewhere else.
NORMAL_TXT = SCRIPT_DIR / "normal_files.txt"
TUNNEL_TXT = SCRIPT_DIR / "tunnel_files.txt"
LINE_TXT   = SCRIPT_DIR / "line_folder_files.txt"
OUTPUT_CSV = SCRIPT_DIR / "inference_labels.csv"


def read_lines(txt_path: Path):
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing input file: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    print("=" * 70)
    print("Building inference_labels.csv")
    print("=" * 70)
    print(f"Normal listing : {NORMAL_TXT}")
    print(f"Tunnel listing : {TUNNEL_TXT}")
    print(f"Line listing   : {LINE_TXT}")
    print(f"Output CSV     : {OUTPUT_CSV}")
    print("=" * 70)

    normal_names = read_lines(NORMAL_TXT)
    tunnel_names = read_lines(TUNNEL_TXT)
    line_paths   = read_lines(LINE_TXT)

    print(f"\nRead from input files:")
    print(f"  normal filenames   : {len(normal_names)}")
    print(f"  tunnel filenames   : {len(tunnel_names)}")
    print(f"  line-folder paths  : {len(line_paths)}")

    # -------------------------------------------------------------------------
    # 1. Build filename -> label lookup.
    #    Compare case-insensitively so 'Foo.JPG' and 'foo.jpg' match.
    # -------------------------------------------------------------------------
    label_by_name = {}
    for n in normal_names:
        label_by_name[n.lower()] = 0
    conflicting = []
    for t in tunnel_names:
        key = t.lower()
        if key in label_by_name and label_by_name[key] != 1:
            conflicting.append(t)
        label_by_name[key] = 1
    if conflicting:
        print(f"\nWARNING: {len(conflicting)} filenames appear in BOTH normal and tunnel listings.")
        print("The label 1 (tunnel) wins for these. First few:")
        for c in conflicting[:5]:
            print(f"  {c}")

    # -------------------------------------------------------------------------
    # 2. Walk the line-folder paths, look up label by filename, write CSV rows.
    #    Use PureWindowsPath because the paths in LINE_TXT use backslashes.
    # -------------------------------------------------------------------------
    rows = []
    unmatched = []
    survey_lines_seen = set()
    normal_counted = 0
    tunnel_counted = 0

    for raw_path in line_paths:
        p = PureWindowsPath(raw_path)
        filename = p.name
        survey_line = p.parent.name  # subfolder name = survey line

        key = filename.lower()
        if key not in label_by_name:
            unmatched.append(raw_path)
            continue

        label = label_by_name[key]
        rows.append({
            "survey_line": survey_line,
            "filename": filename,
            "label": label,
        })
        survey_lines_seen.add(survey_line)
        if label == 0:
            normal_counted += 1
        else:
            tunnel_counted += 1

    # Sort rows by survey_line then filename for tidy CSV output.
    def sort_key(r):
        # Try to extract a number from survey_line for natural sort ("Line 2" before "Line 10")
        sl = r["survey_line"]
        digits = "".join(c for c in sl if c.isdigit())
        num = int(digits) if digits else 10**9
        return (num, sl, r["filename"].lower())
    rows.sort(key=sort_key)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["survey_line", "filename", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # -------------------------------------------------------------------------
    # 3. Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY (verify these against your File Explorer)")
    print("=" * 70)
    print(f"Line-folder images total      : {len(line_paths)}")
    print(f"Rows written to CSV           : {len(rows)}")
    print(f"  labelled normal (0)         : {normal_counted}  "
          f"(from normal listing: {len(normal_names)})")
    print(f"  labelled tunnel (1)         : {tunnel_counted}  "
          f"(from tunnel listing: {len(tunnel_names)})")
    print(f"Unmatched (no label found)    : {len(unmatched)}")
    print(f"Distinct survey lines in CSV  : {len(survey_lines_seen)}")
    print(f"\nCSV path: {OUTPUT_CSV}")

    if unmatched:
        print(f"\nFirst 10 unmatched files (present in line folders but not in normal/tunnel listings):")
        for u in unmatched[:10]:
            print(f"  {u}")

    ok = (normal_counted == len(normal_names)
          and tunnel_counted == len(tunnel_names)
          and len(unmatched) == 0
          and not conflicting)
    if ok:
        print("\n[OK] All counts match perfectly.")
    else:
        print("\n[CHECK] One or more counts do not match. See details above.")


if __name__ == "__main__":
    main()
