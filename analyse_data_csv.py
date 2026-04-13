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
ANGLE_MIN_DEG = -5.0
ANGLE_MAX_DEG = 15.0
INTERPOLATION_STEP_DEG = 0.5

FORCE_DIR = "force"
PRESSURE_DIR = "pression"

FORCE_COLUMNS = ["profil", "vitesse_ms", "angle_deg", "L_kg", "D_kg", "L_N", "D_N"]
PRESSURE_COLUMNS = ["profil", "vitesse_ms", "angle_deg", "pression_PSI"]


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


def is_pressure_header(row: list[str]) -> bool:
    return any(cell.strip().lower().startswith("p1") for cell in row)


def row_has_new_block(row: list[str]) -> bool:
    text = " ".join(row).lower()
    return "donn" in text or "vitesse" in text or is_force_header(row) or is_pressure_header(row)


def non_empty(row: list[str]) -> list[str]:
    return [cell for cell in row if cell.strip()]


def angle_in_range(angle: float) -> bool:
    return ANGLE_MIN_DEG <= angle <= ANGLE_MAX_DEG


def frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    value = start
    while value <= stop + 1.0e-9:
        values.append(round(value, 6))
        value += step
    return values


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

        if airfoil == TARGET_PROFILE and angle_in_range(angle):
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


def add_pressure_rows(
    rows: list[list[str]],
    start_index: int,
    airfoil: str,
    speed_ms: float | None,
    records: dict[tuple[str, float, float], list[float]],
) -> int:
    i = start_index + 1
    while i < len(rows):
        row = rows[i]
        if not non_empty(row):
            break
        if row_has_new_block(row):
            break

        angle = parse_float(row[0]) if row else None
        pressure_values = [parse_float(value) for value in row[1:]]
        pressure_values = [value for value in pressure_values if value is not None]

        if angle is None or speed_ms is None or len(pressure_values) < 2:
            break

        if airfoil == TARGET_PROFILE and angle_in_range(angle):
            key = (airfoil, speed_ms, angle)
            records[key].append(mean(pressure_values))

        i += 1
    return i


def parse_data_file(csv_path: Path) -> tuple[list[dict[str, float | str | None]], list[dict[str, float | str | None]]]:
    rows = read_csv_rows(csv_path)
    force_records = defaultdict(lambda: {"L_kg": [], "D_kg": []})
    pressure_records = defaultdict(list)

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
            i = add_force_rows(rows, i, airfoil, last_speeds, force_records)
            continue

        if is_pressure_header(row):
            speed_ms = last_speeds[0] if last_speeds else None
            i = add_pressure_rows(rows, i, airfoil, speed_ms, pressure_records)
            continue

        i += 1

    force_rows = []
    for (profile, speed_ms, angle), values in force_records.items():
        lift_kg = mean(values["L_kg"])
        drag_kg = mean(values["D_kg"])
        if lift_kg is None or drag_kg is None:
            continue

        force_rows.append(
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

    pressure_rows = []
    for (profile, speed_ms, angle), values in pressure_records.items():
        pressure = mean(values)
        if pressure is None:
            continue

        pressure_rows.append(
            {
                "profil": profile,
                "vitesse_ms": speed_ms,
                "angle_deg": angle,
                "pression_PSI": pressure,
            }
        )

    return sorted_rows(force_rows), sorted_rows(pressure_rows)


def aggregate_force_rows(rows: list[dict[str, float | str | None]]) -> list[dict[str, float | str | None]]:
    aggregated = defaultdict(lambda: {"L_kg": [], "D_kg": []})

    for row in rows:
        key = (
            str(row["profil"]),
            float(row["vitesse_ms"]),
            float(row["angle_deg"]),
        )
        aggregated[key]["L_kg"].append(float(row["L_kg"]))
        aggregated[key]["D_kg"].append(float(row["D_kg"]))

    output = []
    for (profile, speed_ms, angle_deg), values in aggregated.items():
        lift_kg = mean(values["L_kg"])
        drag_kg = mean(values["D_kg"])
        if lift_kg is None or drag_kg is None:
            continue

        output.append(
            {
                "profil": profile,
                "vitesse_ms": speed_ms,
                "angle_deg": angle_deg,
                "L_kg": lift_kg,
                "D_kg": drag_kg,
                "L_N": lift_kg * G,
                "D_N": drag_kg * G,
            }
        )

    return sorted_rows(output)


def aggregate_pressure_rows(rows: list[dict[str, float | str | None]]) -> list[dict[str, float | str | None]]:
    aggregated = defaultdict(list)

    for row in rows:
        key = (
            str(row["profil"]),
            float(row["vitesse_ms"]),
            float(row["angle_deg"]),
        )
        aggregated[key].append(float(row["pression_PSI"]))

    output = []
    for (profile, speed_ms, angle_deg), values in aggregated.items():
        pressure = mean(values)
        if pressure is None:
            continue

        output.append(
            {
                "profil": profile,
                "vitesse_ms": speed_ms,
                "angle_deg": angle_deg,
                "pression_PSI": pressure,
            }
        )

    return sorted_rows(output)


def linear_between(points: list[tuple[float, float]], x: float) -> float | None:
    points = sorted(points)
    for point_x, point_y in points:
        if abs(point_x - x) < 1.0e-9:
            return point_y

    lower = [point for point in points if point[0] < x]
    upper = [point for point in points if point[0] > x]
    if not lower or not upper:
        return None

    x0, y0 = lower[-1]
    x1, y1 = upper[0]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def interpolate_same_speed(rows: list[dict[str, float | str | None]], speed: float, angle: float, column: str) -> float | None:
    points = [
        (float(row["angle_deg"]), float(row[column]))
        for row in rows
        if abs(float(row["vitesse_ms"]) - speed) < 1.0e-9 and row[column] is not None
    ]
    return linear_between(points, angle)


def interpolate_same_angle(rows: list[dict[str, float | str | None]], speed: float, angle: float, column: str) -> float | None:
    points = [
        (float(row["vitesse_ms"]), float(row[column]))
        for row in rows
        if abs(float(row["angle_deg"]) - angle) < 1.0e-9 and row[column] is not None
    ]
    return linear_between(points, speed)


def interpolate_inverse_distance(
    rows: list[dict[str, float | str | None]],
    speed: float,
    angle: float,
    column: str,
) -> float | None:
    speeds = [float(row["vitesse_ms"]) for row in rows]
    angles = [float(row["angle_deg"]) for row in rows]
    speed_scale = max(max(speeds) - min(speeds), 1.0)
    angle_scale = max(max(angles) - min(angles), 1.0)

    weighted_points = []
    for row in rows:
        if row[column] is None:
            continue
        du = (float(row["vitesse_ms"]) - speed) / speed_scale
        da = (float(row["angle_deg"]) - angle) / angle_scale
        distance = math.hypot(du, da)
        if distance < 1.0e-12:
            return float(row[column])
        weighted_points.append((distance, float(row[column])))

    if not weighted_points:
        return None

    nearest = sorted(weighted_points, key=lambda point: point[0])[:8]
    weights = [1.0 / (distance * distance) for distance, _ in nearest]
    values = [value for _, value in nearest]
    return sum(weight * value for weight, value in zip(weights, values)) / sum(weights)


def interpolate_value(rows: list[dict[str, float | str | None]], speed: float, angle: float, column: str) -> float | None:
    exact_values = [
        float(row[column])
        for row in rows
        if row[column] is not None
        and abs(float(row["vitesse_ms"]) - speed) < 1.0e-9
        and abs(float(row["angle_deg"]) - angle) < 1.0e-9
    ]
    if exact_values:
        return mean(exact_values)

    candidates = [
        interpolate_same_speed(rows, speed, angle, column),
        interpolate_same_angle(rows, speed, angle, column),
    ]
    candidates = [value for value in candidates if value is not None]
    if candidates:
        return sum(candidates) / len(candidates)

    return interpolate_inverse_distance(rows, speed, angle, column)


def interpolate_force_rows(measured_rows: list[dict[str, float | str | None]]) -> list[dict[str, float | str | None]]:
    if not measured_rows:
        return []

    speeds = sorted({float(row["vitesse_ms"]) for row in measured_rows})
    angles = frange(ANGLE_MIN_DEG, ANGLE_MAX_DEG, INTERPOLATION_STEP_DEG)

    interpolated = []
    for speed in speeds:
        for angle in angles:
            lift_kg = interpolate_value(measured_rows, speed, angle, "L_kg")
            drag_kg = interpolate_value(measured_rows, speed, angle, "D_kg")
            if lift_kg is None or drag_kg is None:
                continue

            interpolated.append(
                {
                    "profil": TARGET_PROFILE,
                    "vitesse_ms": speed,
                    "angle_deg": angle,
                    "L_kg": lift_kg,
                    "D_kg": drag_kg,
                    "L_N": lift_kg * G,
                    "D_N": drag_kg * G,
                }
            )

    return sorted_rows(interpolated)


def interpolate_pressure_rows(measured_rows: list[dict[str, float | str | None]]) -> list[dict[str, float | str | None]]:
    if not measured_rows:
        return []

    speeds = sorted({float(row["vitesse_ms"]) for row in measured_rows})
    angles = frange(ANGLE_MIN_DEG, ANGLE_MAX_DEG, INTERPOLATION_STEP_DEG)

    interpolated = []
    for speed in speeds:
        for angle in angles:
            pressure = interpolate_value(measured_rows, speed, angle, "pression_PSI")
            if pressure is None:
                continue

            interpolated.append(
                {
                    "profil": TARGET_PROFILE,
                    "vitesse_ms": speed,
                    "angle_deg": angle,
                    "pression_PSI": pressure,
                }
            )

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


def write_output(path: Path, rows: list[dict[str, float | str | None]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted_rows(rows):
            writer.writerow({column: format_cell(row.get(column)) for column in columns})


def clear_csv_outputs(output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    for path in output_dir.glob("*.csv"):
        path.unlink()


def write_by_speed(
    root: Path,
    output_dir_name: str,
    measured: list[dict[str, float | str | None]],
    interpolated: list[dict[str, float | str | None]],
    columns: list[str],
) -> None:
    output_dir = root / output_dir_name
    clear_csv_outputs(output_dir)

    grouped_measured = defaultdict(list)
    grouped_interpolated = defaultdict(list)

    for row in measured:
        grouped_measured[float(row["vitesse_ms"])].append(row)
    for row in interpolated:
        grouped_interpolated[float(row["vitesse_ms"])].append(row)

    for speed_ms, rows in grouped_measured.items():
        write_output(output_dir / f"vitesse_{safe_filename(speed_ms)}_ms_mesures.csv", rows, columns)

    for speed_ms, rows in grouped_interpolated.items():
        write_output(output_dir / f"vitesse_{safe_filename(speed_ms)}_ms_interpoles.csv", rows, columns)


def main() -> None:
    root = Path(__file__).resolve().parent
    force_raw = []
    pressure_raw = []

    for csv_path in root.rglob("data.csv"):
        force_rows, pressure_rows = parse_data_file(csv_path)
        force_raw.extend(force_rows)
        pressure_raw.extend(pressure_rows)

    force_measured = aggregate_force_rows(force_raw)
    pressure_measured = aggregate_pressure_rows(pressure_raw)
    force_interpolated = interpolate_force_rows(force_measured)
    pressure_interpolated = interpolate_pressure_rows(pressure_measured)

    write_by_speed(root, FORCE_DIR, force_measured, force_interpolated, FORCE_COLUMNS)
    write_by_speed(root, PRESSURE_DIR, pressure_measured, pressure_interpolated, PRESSURE_COLUMNS)

    speeds_force = sorted({float(row["vitesse_ms"]) for row in force_measured})
    speeds_pressure = sorted({float(row["vitesse_ms"]) for row in pressure_measured})
    print(f"Forces brutes lues: {len(force_raw)} lignes {TARGET_PROFILE}")
    print(f"Forces mesurees moyennees: {len(force_measured)} lignes {TARGET_PROFILE}")
    print(f"Forces interpolees: {len(force_interpolated)} lignes entre {ANGLE_MIN_DEG:g} et {ANGLE_MAX_DEG:g} deg")
    print(f"Pressions brutes lues: {len(pressure_raw)} lignes {TARGET_PROFILE}")
    print(f"Pressions mesurees moyennees: {len(pressure_measured)} lignes {TARGET_PROFILE}")
    print(f"Pressions interpolees: {len(pressure_interpolated)} lignes entre {ANGLE_MIN_DEG:g} et {ANGLE_MAX_DEG:g} deg")
    print(f"Vitesses force: {', '.join(format_cell(speed) for speed in speeds_force)} m/s")
    print(f"Vitesses pression: {', '.join(format_cell(speed) for speed in speeds_pressure)} m/s")
    print(f"Fichiers force -> {FORCE_DIR}/")
    print(f"Fichiers pression -> {PRESSURE_DIR}/")


if __name__ == "__main__":
    main()
