#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path


TARGET_PROFILE = "NACA-0012"
G = 9.80665
INTERPOLATION_STEP_DEG = 0.5
OUTPUT_BY_SPEED = "resultats_par_vitesse"
COLUMNS = ["profil", "vitesse_ms", "angle_deg", "L_kg", "D_kg", "L_N", "D_N"]


def parse_float(value: str) -> float | None:
    value = value.strip().replace(",", ".")
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    return sum(values) / len(values)


def read_csv_rows(path: Path) -> list[list[str]]:
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return [row for row in csv.reader(handle, delimiter="\t")]
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Impossible de lire {path}")


def extract_airfoil(row: list[str], current_airfoil: str) -> str:
    text = " ".join(cell.strip() for cell in row if cell.strip())
    if "NACA" not in text.upper():
        return current_airfoil

    for token in text.replace("_", "-").split():
        if "NACA" in token.upper():
            return token.strip(" :;")
    return current_airfoil


def is_speed_row(row: list[str]) -> bool:
    return "vitesse" in " ".join(row).lower()


def is_force_header(row: list[str]) -> bool:
    labels = [cell.lower() for cell in row]
    return any("lift" in label for label in labels) and any("drag" in label for label in labels)


def row_has_new_block(row: list[str]) -> bool:
    text = " ".join(row).lower()
    return "donnees" in text or "données" in text or "vitesse" in text or is_force_header(row)


def non_empty(row: list[str]) -> list[str]:
    return [cell for cell in row if cell.strip()]


def add_force_rows(
    rows: list[list[str]],
    start_index: int,
    airfoil: str,
    speeds_ms: list[float],
    records: dict[tuple[str, float, float], dict[str, list[float]]],
) -> int:
    header = rows[start_index]
    force_columns: list[tuple[int, int, float]] = []

    speed_id = 0
    for col in range(1, len(header) - 1, 2):
        left = header[col].lower()
        right = header[col + 1].lower()
        if "lift" in left and "drag" in right and speed_id < len(speeds_ms):
            force_columns.append((col, col + 1, speeds_ms[speed_id]))
            speed_id += 1

    i = start_index + 1
    while i < len(rows):
        row = rows[i]
        if not non_empty(row):
            break
        if row_has_new_block(row):
            break

        angle = parse_float(row[0]) if row else None
        if angle is None:
            break

        if airfoil == TARGET_PROFILE:
            for lift_col, drag_col, speed_ms in force_columns:
                lift_kg = parse_float(row[lift_col]) if lift_col < len(row) else None
                drag_kg = parse_float(row[drag_col]) if drag_col < len(row) else None
                if lift_kg is None or drag_kg is None:
                    continue

                key = (airfoil, speed_ms, angle)
                records[key]["L_kg"].append(lift_kg)
                records[key]["D_kg"].append(drag_kg)

        i += 1
    return i


def parse_data_file(csv_path: Path) -> list[dict[str, float | str | None]]:
    rows = read_csv_rows(csv_path)
    records = defaultdict(lambda: {"L_kg": [], "D_kg": []})

    airfoil = ""
    last_speeds: list[float] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        airfoil = extract_airfoil(row, airfoil)

        if is_speed_row(row):
            speeds = [number for number in (parse_float(cell) for cell in row[1:]) if number is not None]
            if not speeds and i + 1 < len(rows):
                speeds = [number for number in (parse_float(cell) for cell in rows[i + 1][1:]) if number is not None]
                i += 1
            last_speeds = speeds

        if is_force_header(row):
            i = add_force_rows(rows, i, airfoil, last_speeds, records)
            continue

        i += 1

    output = []
    for (profile, speed_ms, angle), values in records.items():
        lift_kg = mean(values["L_kg"])
        drag_kg = mean(values["D_kg"])
        if lift_kg is None or drag_kg is None:
            continue

        output.append(
            {
                "profil": profile,
                "vitesse_ms": speed_ms,
                "angle_deg": angle,
                "L_kg": lift_kg,
                "D_kg": drag_kg,
                "L_N": lift_kg * G,
                "D_N": drag_kg * G,
            }
        )
    return sorted_rows(output)


def interpolate_group(rows: list[dict[str, float | str | None]]) -> list[dict[str, float | str | None]]:
    rows = sorted(rows, key=lambda row: float(row["angle_deg"]))
    if len(rows) < 2:
        return rows

    measured_angles = [float(row["angle_deg"]) for row in rows]
    out = []
    angle = measured_angles[0]
    max_angle = measured_angles[-1]

    while angle <= max_angle + 1.0e-9:
        lower_i = max(index for index, value in enumerate(measured_angles) if value <= angle + 1.0e-9)
        upper_i = min(index for index, value in enumerate(measured_angles) if value >= angle - 1.0e-9)
        lower = rows[lower_i]
        upper = rows[upper_i]
        a0 = float(lower["angle_deg"])
        a1 = float(upper["angle_deg"])

        new_row = {
            "profil": TARGET_PROFILE,
            "vitesse_ms": lower["vitesse_ms"],
            "angle_deg": round(angle, 6),
        }

        for column in ("L_kg", "D_kg", "L_N", "D_N"):
            v0 = float(lower[column])
            v1 = float(upper[column])
            if abs(a1 - a0) < 1.0e-12:
                new_row[column] = v0
            else:
                t = (angle - a0) / (a1 - a0)
                new_row[column] = v0 + t * (v1 - v0)

        out.append(new_row)
        angle += INTERPOLATION_STEP_DEG

    return out


def interpolate_all(measured_rows: list[dict[str, float | str | None]]) -> list[dict[str, float | str | None]]:
    groups = defaultdict(list)
    for row in measured_rows:
        groups[float(row["vitesse_ms"])].append(row)

    interpolated = []
    for group_rows in groups.values():
        interpolated.extend(interpolate_group(group_rows))
    return sorted_rows(interpolated)


def sorted_rows(rows: list[dict[str, float | str | None]]) -> list[dict[str, float | str | None]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["profil"]),
            float(row["vitesse_ms"]),
            float(row["angle_deg"]),
        ),
    )


def format_cell(value: float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def safe_filename(value: float | str | None) -> str:
    text = "inconnu" if value is None else str(value)
    text = text.replace(",", ".").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def write_output(path: Path, rows: list[dict[str, float | str | None]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in sorted_rows(rows):
            writer.writerow({column: format_cell(row.get(column)) for column in COLUMNS})


def write_by_speed(root: Path, measured: list[dict[str, float | str | None]], interpolated: list[dict[str, float | str | None]]) -> None:
    output_dir = root / OUTPUT_BY_SPEED
    output_dir.mkdir(exist_ok=True)

    grouped_measured = defaultdict(list)
    grouped_interpolated = defaultdict(list)

    for row in measured:
        grouped_measured[float(row["vitesse_ms"])].append(row)
    for row in interpolated:
        grouped_interpolated[float(row["vitesse_ms"])].append(row)

    for speed_ms, rows in grouped_measured.items():
        write_output(output_dir / f"vitesse_{safe_filename(speed_ms)}_ms_mesures.csv", rows)

    for speed_ms, rows in grouped_interpolated.items():
        write_output(output_dir / f"vitesse_{safe_filename(speed_ms)}_ms_interpoles.csv", rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    measured = []
    for csv_path in root.rglob("data.csv"):
        measured.extend(parse_data_file(csv_path))

    measured = sorted_rows(measured)
    interpolated = interpolate_all(measured)
    write_by_speed(root, measured, interpolated)

    speeds = sorted({float(row["vitesse_ms"]) for row in measured})
    print(f"{len(measured)} lignes mesurees {TARGET_PROFILE}")
    print(f"{len(interpolated)} lignes interpolees {TARGET_PROFILE}")
    print(f"Vitesses traitees: {', '.join(format_cell(speed) for speed in speeds)} m/s")
    print(f"Fichiers -> {OUTPUT_BY_SPEED}/")


if __name__ == "__main__":
    main()
