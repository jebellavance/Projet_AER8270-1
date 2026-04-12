#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


RHO_AIR = 1.225
G = 9.80665
S_REF_M2 = 1.0

INTERPOLATION_STEP_DEG = 0.5
OUTPUT_MEASURES = "resultats_data_mesures.csv"
OUTPUT_INTERPOLATED = "resultats_data_interpoles.csv"


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


def first_number(values: list[str]) -> float | None:
    for value in values:
        number = parse_float(value)
        if number is not None:
            return number
    return None


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
    upper_text = text.upper()
    if "NACA" not in upper_text:
        return current_airfoil

    for token in text.replace("_", "-").split():
        if "NACA" in token.upper():
            return token.strip(" :;")
    return current_airfoil


def detect_block(header: list[str]) -> str | None:
    labels = [cell.lower() for cell in header]
    if any(label.startswith("p1") for label in labels):
        return "pressure"
    if any("lift" in label for label in labels) and any("drag" in label for label in labels):
        return "force"
    return None


def non_empty(row: list[str]) -> list[str]:
    return [cell for cell in row if cell.strip()]


def add_pressure_rows(
    rows: list[list[str]],
    start_index: int,
    csv_path: Path,
    airfoil: str,
    speed_ms: float | None,
    records: dict[tuple[str, str, float | None, float], dict[str, list[float]]],
) -> int:
    i = start_index + 1
    while i < len(rows):
        row = rows[i]
        values = non_empty(row)
        if not values:
            break
        if detect_block(row) or "vitesse" in " ".join(row).lower() or "donnees" in " ".join(row).lower():
            break

        angle = parse_float(row[0]) if row else None
        pressure_values = [parse_float(value) for value in row[1:]]
        pressure_values = [value for value in pressure_values if value is not None]

        if angle is None or len(pressure_values) < 2:
            break

        key = (str(csv_path), airfoil, speed_ms, angle)
        records[key]["pression_PSI"].append(mean(pressure_values))
        i += 1
    return i


def add_force_rows(
    rows: list[list[str]],
    start_index: int,
    csv_path: Path,
    airfoil: str,
    speeds_ms: list[float],
    records: dict[tuple[str, str, float | None, float], dict[str, list[float]]],
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
        values = non_empty(row)
        if not values:
            break
        if detect_block(row) or "vitesse" in " ".join(row).lower() or "donnees" in " ".join(row).lower():
            break

        angle = parse_float(row[0]) if row else None
        if angle is None:
            break

        for lift_col, drag_col, speed_ms in force_columns:
            lift_kg = parse_float(row[lift_col]) if lift_col < len(row) else None
            drag_kg = parse_float(row[drag_col]) if drag_col < len(row) else None
            if lift_kg is None or drag_kg is None or speed_ms <= 0.0:
                continue

            q = 0.5 * RHO_AIR * speed_ms * speed_ms
            key = (str(csv_path), airfoil, speed_ms, angle)
            records[key]["CL"].append(lift_kg * G / (q * S_REF_M2))
            records[key]["CD"].append(drag_kg * G / (q * S_REF_M2))
        i += 1
    return i


def parse_data_file(csv_path: Path) -> list[dict[str, float | str | None]]:
    rows = read_csv_rows(csv_path)
    records = defaultdict(lambda: {"CL": [], "CD": [], "pression_PSI": []})

    airfoil = ""
    last_speeds: list[float] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        airfoil = extract_airfoil(row, airfoil)
        text = " ".join(cell.lower() for cell in row)

        if "vitesse" in text:
            speeds = [number for number in (parse_float(cell) for cell in row[1:]) if number is not None]
            if not speeds and i + 1 < len(rows):
                speeds = [number for number in (parse_float(cell) for cell in rows[i + 1][1:]) if number is not None]
                i += 1
            last_speeds = speeds

        block_type = detect_block(row)
        if block_type == "pressure":
            i = add_pressure_rows(rows, i, csv_path, airfoil, first_number([str(v) for v in last_speeds]), records)
            continue
        if block_type == "force":
            i = add_force_rows(rows, i, csv_path, airfoil, last_speeds, records)
            continue
        i += 1

    output = []
    for (source, foil, speed_ms, angle), values in sorted(records.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or -1.0, item[0][3])):
        output.append(
            {
                "source": source,
                "profil": foil,
                "vitesse_ms": speed_ms,
                "angle_deg": angle,
                "CL": mean(values["CL"]),
                "CD": mean(values["CD"]),
                "pression_PSI": mean(values["pression_PSI"]),
            }
        )
    return output


def interpolate_group(rows: list[dict[str, float | str | None]]) -> list[dict[str, float | str | None]]:
    rows = sorted(rows, key=lambda row: float(row["angle_deg"]))
    if len(rows) < 2:
        return rows

    angles = [float(row["angle_deg"]) for row in rows]
    out = []
    angle = angles[0]
    max_angle = angles[-1]

    while angle <= max_angle + 1.0e-9:
        lower_i = max(index for index, value in enumerate(angles) if value <= angle + 1.0e-9)
        upper_i = min(index for index, value in enumerate(angles) if value >= angle - 1.0e-9)
        lower = rows[lower_i]
        upper = rows[upper_i]
        a0 = float(lower["angle_deg"])
        a1 = float(upper["angle_deg"])

        new_row = {
            "source": lower["source"],
            "profil": lower["profil"],
            "vitesse_ms": lower["vitesse_ms"],
            "angle_deg": round(angle, 6),
            "CL": None,
            "CD": None,
            "pression_PSI": None,
        }

        for column in ("CL", "CD", "pression_PSI"):
            v0 = lower[column]
            v1 = upper[column]
            if v0 is None or v1 is None:
                continue
            if abs(a1 - a0) < 1.0e-12:
                new_row[column] = v0
            else:
                t = (angle - a0) / (a1 - a0)
                new_row[column] = float(v0) + t * (float(v1) - float(v0))
        out.append(new_row)
        angle += INTERPOLATION_STEP_DEG
    return out


def interpolate_all(measured_rows: list[dict[str, float | str | None]]) -> list[dict[str, float | str | None]]:
    groups = defaultdict(list)
    for row in measured_rows:
        groups[(row["source"], row["profil"], row["vitesse_ms"])].append(row)

    interpolated = []
    for group_rows in groups.values():
        interpolated.extend(interpolate_group(group_rows))
    return sorted(interpolated, key=lambda row: (str(row["source"]), str(row["profil"]), float(row["vitesse_ms"] or -1), float(row["angle_deg"])))


def format_cell(value: float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def write_output(path: Path, rows: list[dict[str, float | str | None]]) -> None:
    columns = ["profil", "vitesse_ms", "angle_deg", "CL", "CD", "pression_PSI"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_cell(row.get(column)) for column in columns})


def main() -> None:
    root = Path(__file__).resolve().parent
    measured = []
    for csv_path in root.rglob("data.csv"):
        measured.extend(parse_data_file(csv_path))

    interpolated = interpolate_all(measured)
    write_output(root / OUTPUT_MEASURES, measured)
    write_output(root / OUTPUT_INTERPOLATED, interpolated)

    print(f"{len(measured)} lignes mesurees -> {OUTPUT_MEASURES}")
    print(f"{len(interpolated)} lignes interpolees -> {OUTPUT_INTERPOLATED}")
    print(f"Surface de reference utilisee pour CL/CD: {S_REF_M2} m2")


if __name__ == "__main__":
    main()
