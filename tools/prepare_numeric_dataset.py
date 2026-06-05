#!/usr/bin/env python3
"""Convert numeric training rows to the transformer's .tnum dataset format.

Input rows are expected as:
    x0 x1 ... xN y0 y1 ... yM

The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CSV/TSV/whitespace numeric data to TRNUM1 .tnum format."
    )
    p.add_argument("input", help="Input table: CSV, TSV, or whitespace-separated text")
    p.add_argument("output", help="Output .tnum file")
    p.add_argument("--inputs", type=int, required=True, help="Number of input columns X")
    p.add_argument("--outputs", type=int, required=True, help="Number of output columns Y")
    p.add_argument(
        "--delimiter",
        choices=["auto", "comma", "tab", "space"],
        default="auto",
        help="Input delimiter. Default: auto",
    )
    p.add_argument("--has-header", action="store_true", help="Skip the first non-empty row")
    p.add_argument(
        "--categorical",
        default="",
        help=(
            "Categorical input specs as zero-based input_index:cardinality,"
            " e.g. '2:5,7:3'. Other inputs are continuous."
        ),
    )
    return p.parse_args()


def detect_delimiter(line: str, mode: str) -> str | None:
    if mode == "comma":
        return ","
    if mode == "tab":
        return "\t"
    if mode == "space":
        return None
    if "," in line:
        return ","
    if "\t" in line:
        return "\t"
    return None


def clean_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def read_rows(path: Path, delimiter_mode: str, has_header: bool) -> list[list[str]]:
    lines = clean_lines(path)
    if not lines:
        raise SystemExit(f"{path}: no data rows")
    if has_header:
        lines = lines[1:]
    if not lines:
        raise SystemExit(f"{path}: no data rows after header")

    delim = detect_delimiter(lines[0], delimiter_mode)
    if delim is None:
        return [line.split() for line in lines]
    return list(csv.reader(lines, delimiter=delim))


def parse_categorical(spec: str, n_inputs: int) -> dict[int, int]:
    out: dict[int, int] = {}
    if not spec.strip():
        return out
    for part in spec.split(","):
        if ":" not in part:
            raise SystemExit(f"Bad --categorical item {part!r}; expected index:cardinality")
        idx_s, card_s = part.split(":", 1)
        idx = int(idx_s)
        card = int(card_s)
        if idx < 0 or idx >= n_inputs:
            raise SystemExit(f"Categorical index {idx} outside input range 0..{n_inputs - 1}")
        if card <= 0:
            raise SystemExit(f"Categorical cardinality must be > 0 for input {idx}")
        out[idx] = card
    return out


def parse_float(token: str, row: int, col: int) -> float:
    try:
        v = float(token)
    except ValueError as e:
        raise SystemExit(f"Row {row}, column {col}: not a number: {token!r}") from e
    if not math.isfinite(v):
        raise SystemExit(f"Row {row}, column {col}: value is not finite: {token!r}")
    return v


def format_f32ish(v: float) -> str:
    return format(v, ".9g")


def build_specs(n_inputs: int, categorical: dict[int, int]) -> list[str]:
    specs = []
    for i in range(n_inputs):
        if i in categorical:
            specs.append(f"K:{categorical[i]}")
        else:
            specs.append("C")
    return specs


def validate_categories(rows: Iterable[list[float]], categorical: dict[int, int]) -> None:
    eps = 1e-4
    for r, row in enumerate(rows):
        for idx, cardinality in categorical.items():
            raw = row[idx]
            rounded = round(raw)
            if abs(raw - rounded) >= eps:
                raise SystemExit(
                    f"Row {r}, input {idx}: categorical code must be integer-like, got {raw}"
                )
            if rounded < 0 or rounded >= cardinality:
                raise SystemExit(
                    f"Row {r}, input {idx}: category {rounded} outside [0, {cardinality})"
                )


def main() -> None:
    args = parse_args()
    n_inputs = args.inputs
    n_outputs = args.outputs
    if n_inputs <= 0 or n_outputs <= 0:
        raise SystemExit("--inputs and --outputs must be > 0")

    categorical = parse_categorical(args.categorical, n_inputs)
    raw_rows = read_rows(Path(args.input), args.delimiter, args.has_header)
    expected_cols = n_inputs + n_outputs

    rows: list[list[float]] = []
    for r, tokens in enumerate(raw_rows):
        tokens = [t.strip() for t in tokens if t.strip()]
        if len(tokens) != expected_cols:
            raise SystemExit(
                f"Row {r}: expected {expected_cols} columns "
                f"({n_inputs} inputs + {n_outputs} outputs), got {len(tokens)}"
            )
        rows.append([parse_float(tok, r, c) for c, tok in enumerate(tokens)])

    validate_categories(rows, categorical)
    specs = build_specs(n_inputs, categorical)

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("TRNUM1\n")
        f.write(f"inputs {n_inputs}\n")
        f.write(f"outputs {n_outputs}\n")
        f.write("specs " + " ".join(specs) + "\n")
        f.write(f"rows {len(rows)}\n")
        f.write("data\n")
        for row in rows:
            f.write(" ".join(format_f32ish(v) for v in row) + "\n")

    print(
        f"Wrote {out_path} with {len(rows)} rows, "
        f"{n_inputs} inputs, {n_outputs} outputs"
    )


if __name__ == "__main__":
    main()
