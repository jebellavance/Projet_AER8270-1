#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import importlib
import sys
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "force"
DEPEND_HSPM = ROOT / "dependances" / "HSPM"
DEPEND_VLM = ROOT / "dependances" / "VLM"

OUTPUT_FORCE_FILE = ROOT / "verification_L_D_vs_angle_NACA0012.png"
OUTPUT_2D_Q1_FILE = ROOT / "partie2_question1_hspm_couche_limite.png"
OUTPUT_3D_CL_ALPHA = ROOT / "partie3_CL_alpha_rectangulaire.png"
OUTPUT_3D_CD_CL = ROOT / "partie3_CD_CL_rectangulaire.png"

ALPHA_MIN = -5.0
ALPHA_MAX = 18.0
ALPHA_STEP = 0.5
ALPHA_RANGE = [float(alpha) for alpha in np.arange(ALPHA_MIN, ALPHA_MAX + 0.5 * ALPHA_STEP, ALPHA_STEP)]
VALAREZO_CRITERION = 14.0

CHORD = 0.1524
FULL_SPAN = 0.6096
SEMI_SPAN = FULL_SPAN / 2.0
S_REF = 0.09290
SREF_HALF = S_REF / 2.0
RHO_AIR = 1.17

NI_RECT = 6
NJ_RECT = 50
NU_AIR = 1.5e-5
XTR_UPPER = 0.10
XTR_LOWER = 0.10
H_SEP_TURB = 3.0
H_TR = 1.4


def import_vlm_solver():
    dependency_path = str(DEPEND_VLM)
    if dependency_path in sys.path:
        sys.path.remove(dependency_path)
    sys.path.insert(0, dependency_path)

    for module_name in ("Vector3", "vortexRing", "vlm"):
        if module_name in sys.modules:
            del sys.modules[module_name]

    return importlib.import_module("vlm").VLM


def import_hspm_modules():
    dependency_path = str(DEPEND_HSPM)
    if dependency_path in sys.path:
        sys.path.remove(dependency_path)
    sys.path.insert(0, dependency_path)

    for module_name in ("Vector3", "sourcePanel", "geometryGenerator", "HSPM"):
        if module_name in sys.modules:
            del sys.modules[module_name]

    geometry_generator_module = importlib.import_module("geometryGenerator")
    hspm_module = importlib.import_module("HSPM")
    return geometry_generator_module, hspm_module


VLM = import_vlm_solver()
geometryGenerator, HSPM = import_hspm_modules()


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_force_file(path: Path) -> list[dict[str, float | str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            alpha = parse_float(raw_row["angle_deg"])
            speed = parse_float(raw_row["vitesse_ms"])
            lift = parse_float(raw_row["L_N"])
            drag = parse_float(raw_row["D_N"])

            if alpha is None or speed is None or lift is None or drag is None:
                continue

            rows.append(
                {
                    "profil": raw_row["profil"],
                    "vitesse_ms": speed,
                    "angle_deg": alpha,
                    "L_N": lift,
                    "D_N": drag,
                    "CL": lift / (0.5 * RHO_AIR * speed * speed * S_REF),
                    "CD": drag / (0.5 * RHO_AIR * speed * speed * S_REF),
                }
            )
    return rows


def read_force_data(directory: Path) -> list[dict[str, float | str]]:
    rows = []
    for path in sorted(directory.glob("*_interpoles.csv")):
        rows.extend(read_force_file(path))
    return rows


def mean_by_angle_for_speed(rows: list[dict[str, float | str]], column: str, speed: float) -> tuple[list[float], list[float]]:
    grouped = defaultdict(list)
    for row in rows:
        if abs(float(row["vitesse_ms"]) - speed) < 1.0e-9:
            grouped[float(row["angle_deg"])].append(float(row[column]))

    angles = sorted(grouped)
    means = [sum(grouped[angle]) / len(grouped[angle]) for angle in angles]
    return angles, means


def plot_force_vs_angle(ax, rows: list[dict[str, float | str]], column: str) -> None:
    speeds = sorted({float(row["vitesse_ms"]) for row in rows})
    for speed in speeds:
        angles, values = mean_by_angle_for_speed(rows, column, speed)
        if len(angles) < 2:
            continue
        ax.plot(angles, values, marker="o", linewidth=1.8, label=f"U = {speed:g} m/s")


def plot_2d_force_curves() -> None:
    rows = read_force_data(DATA_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    plot_force_vs_angle(axes[0], rows, "L_N")
    plot_force_vs_angle(axes[1], rows, "D_N")

    axes[0].set_title("NACA0012 - Portance en fonction de l'angle")
    axes[0].set_xlabel("Angle d'attaque [deg]")
    axes[0].set_ylabel("L [N]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title("NACA0012 - Trainee en fonction de l'angle")
    axes[1].set_xlabel("Angle d'attaque [deg]")
    axes[1].set_ylabel("D [N]")
    axes[1].grid(True)
    axes[1].legend()

    fig.savefig(OUTPUT_FORCE_FILE, dpi=200)


def build_naca_panels(profile: str, points_per_surface: int = 300):
    if profile == "NACA-0012":
        return geometryGenerator.GenerateNACA4digit(
            maxCamber=0.0,
            positionOfMaxCamber=0.0,
            thickness=12.0,
            pointsPerSurface=points_per_surface,
        )
    if profile == "NACA-4412":
        return geometryGenerator.GenerateNACA4digit(
            maxCamber=4.0,
            positionOfMaxCamber=4.0,
            thickness=12.0,
            pointsPerSurface=points_per_surface,
        )
    raise ValueError(f"Profil non supporte: {profile}")


def cumulative_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


def d_ds_forward(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    out = np.empty_like(y)
    ds = np.diff(s)
    out[:-1] = (y[1:] - y[:-1]) / ds
    out[-1] = out[-2]
    return out


def H_thwaites(lam: float) -> float:
    if lam >= 0.0:
        return 2.61 - 3.75 * lam + 5.24 * lam**2
    return 0.0731 / (0.14 + lam) + 2.088


def ell_thwaites(lam: float) -> float:
    if lam >= 0.0:
        return 0.22 + 1.57 * lam - 1.8 * lam**2
    return 0.22 + 1.402 * lam + (0.018 * lam) / (0.107 + lam)


def thwaites_laminar_theta(s: np.ndarray, ue: np.ndarray, nu: float):
    integral = cumulative_trapz(ue**5, s)
    theta = np.zeros_like(ue)
    mask = ue > 1.0e-14
    theta[mask] = np.sqrt(nu * (0.45 * integral[mask]) / (ue[mask] ** 6))
    d_ueds = d_ds_forward(ue, s)

    lam = np.zeros_like(ue)
    lam[mask] = (theta[mask] ** 2 / nu) * d_ueds[mask]

    H = np.array([H_thwaites(li) for li in lam])
    ell = np.array([ell_thwaites(li) for li in lam])

    re_theta = np.zeros_like(ue)
    re_theta[mask] = ue[mask] * theta[mask] / nu

    cf = np.zeros_like(ue)
    cf[mask] = 2.0 * ell[mask] / np.maximum(re_theta[mask], 1e-30)
    return theta, H, cf, d_ueds


def turbulent_march_profile(
    s: np.ndarray,
    ue: np.ndarray,
    d_ueds: np.ndarray,
    nu: float,
    i_tr: int,
    theta_tr: float,
    H_tr: float = H_TR,
    sep_H: float = H_SEP_TURB,
):
    n = len(s)
    theta = np.full(n, np.nan)
    H = np.full(n, np.nan)
    cf = np.zeros(n)

    theta[i_tr] = theta_tr
    H[i_tr] = H_tr

    for i in range(i_tr, n - 1):
        ds = s[i + 1] - s[i]
        if ue[i] <= 1.0e-14 or theta[i] <= 1.0e-20:
            break

        re_theta = max(ue[i] * theta[i] / nu, 1e-30)
        cf_i = 0.246 * (10.0 ** (-0.678 * H[i])) * (re_theta ** (-0.268))
        cf[i] = cf_i

        rhs_theta = cf_i / 2.0 - (H[i] + 2.0) * (theta[i] / ue[i]) * d_ueds[i]
        term1 = -H[i] * (H[i] - 1.0) * (3.0 * H[i] - 1.0) * (theta[i] / ue[i]) * d_ueds[i]
        term2 = H[i] * (3.0 * H[i] - 1.0) * (cf_i / 2.0)
        term3 = -(3.0 * H[i] - 1.0) ** 2 * (0.0056 / 2.0) * (re_theta ** (-1.0 / 6.0))
        rhs_H = (term1 + term2 + term3) / max(theta[i], 1e-30)

        theta[i + 1] = theta[i] + ds * rhs_theta
        H[i + 1] = H[i] + ds * rhs_H

        if H[i + 1] >= sep_H:
            cf[i + 1 :] = 0.0
            break

    cf[:i_tr] = np.nan
    return theta, H, cf


def build_surface_arrays(points_coordinate, vtang):
    x = np.array([point[0] for point in points_coordinate], dtype=float)
    z = np.array([point[2] for point in points_coordinate], dtype=float)
    ue = np.asarray(vtang, dtype=float)

    ds = np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)
    s = np.zeros_like(x)
    s[1:] = np.cumsum(ds)
    x_over_c = x / CHORD
    return x, z, s, x_over_c, ue


def compute_surface_drag_from_vtang(points_coordinate, vtang, xtr: float, nu: float) -> float:
    x, _z, s, x_over_c, ue = build_surface_arrays(points_coordinate, vtang)

    theta_lam, _H_lam, cf_lam, d_ueds = thwaites_laminar_theta(s, ue, nu)

    candidates = np.where(x_over_c >= xtr)[0]
    if len(candidates) == 0:
        i_tr = len(x_over_c) - 1
    else:
        i_tr = int(candidates[0])

    theta_tr = theta_lam[i_tr]
    _theta_turb, _H_turb, cf_turb = turbulent_march_profile(s, ue, d_ueds, nu, i_tr, theta_tr)

    cf_full = np.array(cf_lam, copy=True)
    if i_tr < len(cf_full):
        cf_full[i_tr:] = np.nan_to_num(cf_turb[i_tr:], nan=0.0)

    dx_abs = np.abs(np.diff(x))
    cf_mid = 0.5 * (cf_full[1:] + cf_full[:-1])
    return float(np.nansum(cf_mid * dx_abs) / CHORD)


def compute_profile_hspm_bl_curves(profile: str) -> dict[str, list[float] | str]:
    cl_hspm = []
    cd_hspm = []
    cd_visc = []

    for alpha in ALPHA_RANGE:
        panels = build_naca_panels(profile)
        prob = HSPM.HSPM(listOfPanels=panels, alphaRange=[alpha], referencePoint=[0.25, 0.0, 0.0])
        prob.run()

        cl_hspm.append(float(prob.CL[-1]))
        cd_hspm.append(float(prob.CD[-1]))
        lower_coords, lower_v = prob.getLowerVtangential()
        upper_coords, upper_v = prob.getUpperVtangential()

        cd_lower = compute_surface_drag_from_vtang(lower_coords, lower_v, XTR_LOWER, NU_AIR)
        cd_upper = compute_surface_drag_from_vtang(upper_coords, upper_v, XTR_UPPER, NU_AIR)
        cd_visc.append(cd_lower + cd_upper)

    return {
        "profile": profile,
        "alpha_range": list(ALPHA_RANGE),
        "cl_hspm": cl_hspm,
        "cd_hspm": cd_hspm,
        "cd_bl": cd_visc,
    }


def plot_partie2_question1(results: list[dict[str, list[float] | str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axis_map = {
        "NACA-0012": (axes[0][0], axes[0][1]),
        "NACA-4412": (axes[1][0], axes[1][1]),
    }

    for result in results:
        profile = str(result["profile"])
        alpha_range = result["alpha_range"]
        cl_hspm = result["cl_hspm"]
        cd_hspm = result["cd_hspm"]
        cd_bl = result["cd_bl"]
        ax_cl, ax_cd = axis_map[profile]

        ax_cl.plot(alpha_range, cl_hspm, marker="o", linewidth=1.8, label="HSPM")
        ax_cl.set_title(f"{profile} - CL en fonction de l'angle")
        ax_cl.set_xlabel("Angle d'attaque [deg]")
        ax_cl.set_ylabel("CL [-]")
        ax_cl.grid(True)
        ax_cl.legend()

        ax_cd.plot(cl_hspm, cd_hspm, linestyle="--", linewidth=1.6, label="HSPM")
        ax_cd.plot(cl_hspm, cd_bl, marker="o", linewidth=1.8, label="Couche limite")
        ax_cd.set_title(f"{profile} - CD en fonction de CL")
        ax_cd.set_xlabel("CL [-]")
        ax_cd.set_ylabel("CD [-]")
        ax_cd.grid(True)
        ax_cd.legend()

    fig.savefig(OUTPUT_2D_Q1_FILE, dpi=200)


def print_partie2_question1(results: list[dict[str, list[float] | str]]) -> None:
    print()
    print("=== Partie 2 - Question 1 ===")
    print("Methode utilisee :")
    print("1. HSPM pour calculer CL(alpha) sur les profils NACA0012 et NACA4412")
    print("2. Extraction de Ue(s) sur extrados et intrados")
    print("3. Marche de couche limite inspiree des TD2 (Thwaites + turbulent)")
    print("4. Integration de Cf pour estimer le CD visqueux")
    print()
    for result in results:
        profile = str(result["profile"])
        cl_hspm = result["cl_hspm"]
        cd_bl = result["cd_bl"]
        print(
            f"{profile}: CL(0 deg) = {np.interp(0.0, ALPHA_RANGE, cl_hspm):.4f}, "
            f"CD_BL(0 deg) = {np.interp(0.0, ALPHA_RANGE, cd_bl):.5f}"
        )
    print(f"Figure sauvegardee : {OUTPUT_2D_Q1_FILE.name}")
    print()


def compute_2d_valarezo(profile: str, criterion: float = VALAREZO_CRITERION) -> dict[str, object]:
    panels = build_naca_panels(profile)
    prob = HSPM.HSPM(listOfPanels=panels, alphaRange=ALPHA_RANGE, referencePoint=[0.25, 0.0, 0.0])
    prob.run()
    alpha_max, cl_max = prob.findAlphaMaxClMax(valarezoCriterion=criterion)

    return {
        "profile": profile,
        "alpha_range": list(ALPHA_RANGE),
        "cl_curve_2d": list(prob.CL),
        "delta_cp_valarezo": list(prob.deltaCPvalarezo),
        "alpha_stall_2d": float(alpha_max),
        "clmax_2d": float(cl_max),
    }


def run_rectangular_wing_3d(alpha_range: list[float]):
    prob = VLM(
        ni=NI_RECT,
        nj=NJ_RECT,
        chordRoot=CHORD,
        chordTip=CHORD,
        twistRoot=0.0,
        twistTip=0.0,
        span=SEMI_SPAN,
        sweep=0.0,
        Sref=SREF_HALF,
        referencePoint=[0.25 * CHORD, 0.0, 0.0],
        wingType=1,
        alphaRange=alpha_range,
    )
    prob.rho = RHO_AIR
    prob.run()
    return prob


def interpolate_curve(x_values: list[float], y_values: list[float], x_target: float) -> float:
    return float(np.interp(x_target, x_values, y_values))


def build_mean_viscous_cd_curve() -> tuple[list[float], list[float]]:
    rows = read_force_data(DATA_DIR)
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["angle_deg"])].append(float(row["CD"]))

    alphas = sorted(grouped)
    mean_cd = [sum(grouped[alpha]) / len(grouped[alpha]) for alpha in alphas]
    return alphas, mean_cd


def solve_partie3_question1() -> dict[str, dict[str, float]]:
    valarezo_0012 = compute_2d_valarezo("NACA-0012")
    valarezo_4412 = compute_2d_valarezo("NACA-4412")

    wing_prob = run_rectangular_wing_3d(ALPHA_RANGE)
    cl_curve_3d = [float(value) for value in wing_prob.CL]

    results = {}
    for result in (valarezo_0012, valarezo_4412):
        profile = str(result["profile"])
        alpha_stall_2d = float(result["alpha_stall_2d"])
        clmax_2d = float(result["clmax_2d"])
        clmax_3d_estime = interpolate_curve(ALPHA_RANGE, cl_curve_3d, alpha_stall_2d)

        results[profile] = {
            "alpha_stall_2d": alpha_stall_2d,
            "clmax_2d": clmax_2d,
            "clmax_3d_estime": clmax_3d_estime,
        }

    return results


def print_partie3_question1(results: dict[str, dict[str, float]]) -> None:
    print()
    print("=== Partie 3 - Question 1 ===")
    print("Methode utilisee :")
    print("1. Calcul 2D avec HSPM")
    print("2. Extraction de alpha_stall avec la methode de Valarezo")
    print("3. Transposition de cet angle au cas 3D")
    print("4. Lecture du CL de l'aile rectangulaire 3D a cet angle")
    print()

    for profile, values in results.items():
        print(f"Profil : {profile}")
        print(f"  alpha_stall_2D_Valarezo = {values['alpha_stall_2d']:.3f} deg")
        print(f"  CLmax_2D_Valarezo      = {values['clmax_2d']:.4f}")
        print(f"  CLmax_3D_estime        = {values['clmax_3d_estime']:.4f}")
        print()


def solve_partie3_question2() -> dict[str, list[float]]:
    wing_prob = run_rectangular_wing_3d(ALPHA_RANGE)
    cl_curve_3d = [float(value) for value in wing_prob.CL]
    cdi_curve_3d = [float(value) for value in wing_prob.CD]

    alpha_viscous, cd_viscous = build_mean_viscous_cd_curve()
    cdv_curve_2d = [interpolate_curve(alpha_viscous, cd_viscous, alpha) for alpha in ALPHA_RANGE]
    cd_total_curve = [cdi + cdv for cdi, cdv in zip(cdi_curve_3d, cdv_curve_2d)]

    return {
        "alpha_range": list(ALPHA_RANGE),
        "CL_3D": cl_curve_3d,
        "CDi_3D": cdi_curve_3d,
        "CDv_2D": cdv_curve_2d,
        "CD_total": cd_total_curve,
    }


def plot_partie3_question2(results: dict[str, list[float]]) -> None:
    alpha_range = results["alpha_range"]
    cl_curve = results["CL_3D"]
    cdi_curve = results["CDi_3D"]
    cdv_curve = results["CDv_2D"]
    cd_total_curve = results["CD_total"]

    fig1, ax1 = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax1.plot(alpha_range, cl_curve, marker="o", linewidth=2.0, label="Aile rectangulaire 3D")
    ax1.set_title("Partie 3 - CL en fonction de l'angle")
    ax1.set_xlabel("Angle d'attaque [deg]")
    ax1.set_ylabel("CL [-]")
    ax1.grid(True)
    ax1.legend()
    fig1.savefig(OUTPUT_3D_CL_ALPHA, dpi=200)

    fig2, ax2 = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax2.plot(cl_curve, cd_total_curve, marker="o", linewidth=2.0, label="CD total")
    ax2.plot(cl_curve, cdi_curve, linestyle="--", linewidth=1.6, label="CDi 3D")
    ax2.plot(cl_curve, cdv_curve, linestyle=":", linewidth=1.6, label="CDv 2D moyen")
    ax2.set_title("Partie 3 - CD en fonction de CL")
    ax2.set_xlabel("CL [-]")
    ax2.set_ylabel("CD [-]")
    ax2.grid(True)
    ax2.legend()
    fig2.savefig(OUTPUT_3D_CD_CL, dpi=200)


def print_partie3_question2(results: dict[str, list[float]]) -> None:
    print("=== Partie 3 - Question 2 ===")
    print("Methode utilisee :")
    print("1. Calcul du CL et du CDi de l'aile rectangulaire avec le VLM")
    print("2. Estimation de la trainee visqueuse a partir des donnees 2D experimentales du NACA-0012")
    print("3. Somme des contributions pour obtenir CD_total = CDi_3D + CDv_2D")
    print()
    print(f"Figure CL-alpha sauvegardee : {OUTPUT_3D_CL_ALPHA.name}")
    print(f"Figure CD-CL sauvegardee    : {OUTPUT_3D_CD_CL.name}")
    print(
        f"Exemple au premier angle ({results['alpha_range'][0]:.1f} deg) : "
        f"CL = {results['CL_3D'][0]:.4f}, "
        f"CDi = {results['CDi_3D'][0]:.5f}, "
        f"CDv = {results['CDv_2D'][0]:.5f}, "
        f"CDtotal = {results['CD_total'][0]:.5f}"
    )
    print()


def main() -> None:
    task = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "all"

    if task in ("courbes_2d", "2d", "all"):
        plot_2d_force_curves()

    if task in ("question_2d_1", "q2d1", "partie2_q1", "all"):
        results_2d_q1 = [
            compute_profile_hspm_bl_curves("NACA-0012"),
            compute_profile_hspm_bl_curves("NACA-4412"),
        ]
        plot_partie2_question1(results_2d_q1)
        print_partie2_question1(results_2d_q1)

    if task in ("question_1", "q1", "partie3_q1", "all"):
        partie3_q1_results = solve_partie3_question1()
        print_partie3_question1(partie3_q1_results)

    if task in ("question_2", "q2", "partie3_q2", "all"):
        partie3_q2_results = solve_partie3_question2()
        print_partie3_question2(partie3_q2_results)
        plot_partie3_question2(partie3_q2_results)

    if task not in (
        "courbes_2d",
        "2d",
        "question_2d_1",
        "q2d1",
        "partie2_q1",
        "question_1",
        "q1",
        "partie3_q1",
        "question_2",
        "q2",
        "partie3_q2",
        "all",
    ):
        print("Argument non reconnu.")
        print("Utilise par exemple :")
        print("  python projet_aer8270.py question_2d_1")
        print("  python projet_aer8270.py question_1")
        print("  python projet_aer8270.py question_2")
        print("  python projet_aer8270.py courbes_2d")
        print("  python projet_aer8270.py all")
        return

    plt.show()


if __name__ == "__main__":
    main()
