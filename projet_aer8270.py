#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trace temporaire pour verifier les forces L et D des profils NACA.

Ce bloc est volontairement simple et pourra etre supprime/remplace plus tard
quand le fichier principal du projet sera construit.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent / "resultats_par_vitesse"
OUTPUT_FILE = Path(__file__).resolve().parent / "verification_L_D_NACA.png"


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_aero_file(path: Path) -> list[dict[str, float | str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            alpha = parse_float(raw_row["angle_deg"])
            lift = parse_float(raw_row.get("L_N", raw_row.get("CL", "")))
            drag = parse_float(raw_row.get("D_N", raw_row.get("CD", "")))
            speed = parse_float(raw_row["vitesse_ms"])

            if alpha is None or speed is None:
                continue
            if lift is None and drag is None:
                continue

            rows.append(
                {
                    "profil": raw_row["profil"],
                    "vitesse_ms": speed,
                    "angle_deg": alpha,
                    "L_N": lift,
                    "D_N": drag,
                }
            )
    return rows


def read_aero_data(directory: Path) -> list[dict[str, float | str]]:
    rows = []
    for path in sorted(directory.glob("*_interpoles.csv")):
        rows.extend(read_aero_file(path))
    return rows


def mean_by_angle(rows: list[dict[str, float | str]], column: str) -> tuple[list[float], list[float]]:
    grouped = defaultdict(list)
    for row in rows:
        value = row[column]
        if value is not None:
            grouped[float(row["angle_deg"])].append(float(value))

    angles = sorted(grouped)
    means = [sum(grouped[angle]) / len(grouped[angle]) for angle in angles]
    return angles, means


def plot_profile_data(ax, rows: list[dict[str, float | str]], profile: str, column: str) -> None:
    profile_rows = [row for row in rows if row["profil"] == profile]
    point_rows = [row for row in profile_rows if row[column] is not None]

    ax.scatter(
        [float(row["angle_deg"]) for row in point_rows],
        [float(row[column]) for row in point_rows],
        s=18,
        alpha=0.35,
        label=f"{profile} mesures",
    )

    angles, means = mean_by_angle(profile_rows, column)
    ax.plot(angles, means, marker="o", linewidth=2.0, label=f"{profile} moyenne")


def main() -> None:
    rows = read_aero_data(DATA_DIR)
    profiles = sorted({str(row["profil"]) for row in rows if str(row["profil"]).startswith("NACA")})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for profile in profiles:
        plot_profile_data(axes[0], rows, profile, "L_N")
        plot_profile_data(axes[1], rows, profile, "D_N")

    axes[0].set_title("Portance en fonction de l'angle d'attaque")
    axes[0].set_xlabel("Angle d'attaque [deg]")
    axes[0].set_ylabel("L [N]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title("Trainee en fonction de l'angle d'attaque")
    axes[1].set_xlabel("Angle d'attaque [deg]")
    axes[1].set_ylabel("D [N]")
    axes[1].grid(True)
    axes[1].legend()

    fig.savefig(OUTPUT_FILE, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
