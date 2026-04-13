#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import importlib
import sys
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEPEND_HSPM = ROOT / "dependances" / "HSPM"

OUTPUT_2D_Q1_FILE = ROOT / "partie2_question1_CL_alpha.png"
OUTPUT_2D_CD_ALPHA_FILE = ROOT / "partie2_CD_alpha.png"
OUTPUT_2D_CD_CL_FILE = ROOT / "partie2_CD_CL.png"
OUTPUT_2D_Q3_ALPHA_CL_FILE = ROOT / "partie2_question3_sep_alpha_cl.png"
OUTPUT_2D_Q4_VALAREZO_FILE = ROOT / "partie2_question4_valarezo.png"
CACHE_DIR = ROOT / "cache_hspm"

ALPHA_MIN = -18.0
ALPHA_MAX = 18.0
ALPHA_STEP = 0.5
ALPHA_RANGE = [float(alpha) for alpha in np.arange(ALPHA_MIN, ALPHA_MAX + 0.5 * ALPHA_STEP, ALPHA_STEP)]
VALAREZO_CRITERION = 5.0

CHORD = 0.1524
FULL_SPAN = 0.6096
SEMI_SPAN = FULL_SPAN / 2.0
S_REF = 0.09290
NU_AIR = 1.5e-5
XTR_DEFAULT = 0.10
H_TR = 1.4
H_SEP_TURB = 3.0
CF_MAX = 0.05
PROFILE_XTR = {
    "NACA-0012": 0.50,
    "NACA-4412": 0.45,
}
PROFILE_CD0_BASE = {
    "NACA-0012": 0.020,
    "NACA-4412": 0.023,
}
CD_SEP_GAIN = 0.060
AIRFOIL_TOOLS_FILES = {
    "NACA-0012": {
        "cl_alpha": ROOT / "CL(alpha)_NACA0012_Airfoil_tools.csv",
        "cl_cd": ROOT / "CL(CD)_NACA0012_Airfoil_tools.csv",
    },
    "NACA-4412": {
        "cl_alpha": ROOT / "CL(alpha)_NACA4412_Airfoil_tools.csv",
        "cl_cd": ROOT / "CL(CD)_NACA4412_Airfoil_tools.csv",
    },
}


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

geometryGenerator, HSPM = import_hspm_modules()


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_xy_csv(path: Path) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    if not path.exists():
        return xs, ys

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) < 2:
            return xs, ys
        x_name, y_name = fieldnames[0], fieldnames[1]
        for row in reader:
            x = parse_float(row[x_name])
            y = parse_float(row[y_name])
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
    return xs, ys


def build_naca_panels(profile: str, points_per_surface: int = 200):
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


def profile_cache_slug(profile: str) -> str:
    return profile.lower().replace("-", "").replace(" ", "_")


def alpha_cache_slug(alpha: float) -> str:
    sign = "m" if alpha < 0.0 else "p"
    return f"{sign}{abs(alpha):04.1f}".replace(".", "p")


def get_profile_cache_dir(profile: str) -> Path:
    return CACHE_DIR / profile_cache_slug(profile)


def get_profile_summary_path(profile: str) -> Path:
    return get_profile_cache_dir(profile) / "courbe_CL.csv"


def get_surface_cache_path(profile: str, alpha: float, side: str) -> Path:
    return get_profile_cache_dir(profile) / f"Ue_{side}_alpha_{alpha_cache_slug(alpha)}.csv"


def get_cp_solution_path(alpha: float) -> Path:
    return ROOT.parent / f"CPsol_A{alpha:.2f}.dat"


def write_surface_cache(path: Path, points_coordinate, vtang) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "z", "ue"])
        for point, ue in zip(points_coordinate, vtang):
            writer.writerow([f"{point[0]:.12e}", f"{point[2]:.12e}", f"{float(ue):.12e}"])


def read_surface_cache(path: Path) -> tuple[list[list[float]], list[float]]:
    points_coordinate = []
    vtang = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x = parse_float(row["x"])
            z = parse_float(row["z"])
            ue = parse_float(row["ue"])
            if x is None or z is None or ue is None:
                continue
            points_coordinate.append([x, 0.0, z])
            vtang.append(ue)
    return points_coordinate, vtang


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


def build_surface_arrays(points_coordinate, vtang):
    x = np.array([point[0] for point in points_coordinate], dtype=float)
    z = np.array([point[2] for point in points_coordinate], dtype=float)
    ue = np.asarray(vtang, dtype=float)

    ds = np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)
    s = np.zeros_like(x)
    s[1:] = np.cumsum(ds)
    x_over_c = x / CHORD
    return x, z, s, x_over_c, ue


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
    cf[mask] = 2.0 * ell[mask] / np.maximum(re_theta[mask], 1.0e-30)
    return theta, H, cf, d_ueds, lam


def turbulent_march_profile(
    s: np.ndarray,
    ue: np.ndarray,
    d_ueds: np.ndarray,
    nu: float,
    i_tr: int,
    theta_tr: float,
    H_tr_value: float = H_TR,
    sep_H: float = H_SEP_TURB,
):
    n = len(s)
    theta = np.full(n, np.nan)
    H = np.full(n, np.nan)
    cf = np.zeros(n)
    separated = False
    sep_index = None

    theta[i_tr] = theta_tr
    H[i_tr] = H_tr_value

    for i in range(i_tr, n - 1):
        ds = s[i + 1] - s[i]
        if ue[i] <= 1.0e-14 or theta[i] <= 1.0e-20:
            break

        re_theta = max(ue[i] * theta[i] / nu, 1.0e-30)
        cf_i = 0.246 * (10.0 ** (-0.678 * H[i])) * (re_theta ** (-0.268))
        cf_i = min(max(cf_i, 0.0), CF_MAX)
        cf[i] = cf_i

        rhs_theta = cf_i / 2.0 - (H[i] + 2.0) * (theta[i] / ue[i]) * d_ueds[i]
        term1 = -H[i] * (H[i] - 1.0) * (3.0 * H[i] - 1.0) * (theta[i] / ue[i]) * d_ueds[i]
        term2 = H[i] * (3.0 * H[i] - 1.0) * (cf_i / 2.0)
        term3 = -(3.0 * H[i] - 1.0) ** 2 * (0.0056 / 2.0) * (re_theta ** (-1.0 / 6.0))
        rhs_H = (term1 + term2 + term3) / max(theta[i], 1.0e-30)

        theta[i + 1] = theta[i] + ds * rhs_theta
        H[i + 1] = H[i] + ds * rhs_H

        if not np.isfinite(theta[i + 1]) or not np.isfinite(H[i + 1]):
            break
        if H[i + 1] >= sep_H:
            cf[i + 1 :] = 0.0
            theta[i + 1 :] = theta[i + 1]
            H[i + 1 :] = H[i + 1]
            separated = True
            sep_index = i + 1
            break

    cf[:i_tr] = np.nan
    if not separated:
        last_valid_theta = np.where(np.isfinite(theta))[0]
        last_valid_H = np.where(np.isfinite(H))[0]
        if len(last_valid_theta) > 0 and last_valid_theta[-1] < n - 1:
            theta[last_valid_theta[-1] + 1 :] = theta[last_valid_theta[-1]]
        if len(last_valid_H) > 0 and last_valid_H[-1] < n - 1:
            H[last_valid_H[-1] + 1 :] = H[last_valid_H[-1]]
    return theta, H, cf, sep_index


def compute_surface_boundary_layer_from_ue_file(
    path: Path,
    xtr: float = XTR_DEFAULT,
    nu: float = NU_AIR,
) -> dict[str, float | np.ndarray]:
    points_coordinate, vtang = read_surface_cache(path)
    x, _z, s, x_over_c, ue = build_surface_arrays(points_coordinate, vtang)
    ue = np.maximum(ue, 1.0e-12)

    theta_lam, _H_lam, cf_lam, d_ueds, lam = thwaites_laminar_theta(s, ue, nu)

    candidates = np.where((x_over_c >= xtr) & (x_over_c <= 0.95))[0]
    if len(candidates) == 0:
        candidates = np.where(s >= xtr * s[-1])[0]
        if len(candidates) == 0:
            return {"cd_surface": float("nan"), "theta_te": float("nan"), "xtr": float(xtr), "xsep": 1.0}

    i_tr = int(candidates[0])
    theta_tr = theta_lam[i_tr]
    if not np.isfinite(theta_tr) or theta_tr <= 1.0e-12:
        return {"cd_surface": float("nan"), "theta_te": float("nan"), "xtr": float(xtr), "xsep": 1.0}

    lam_sep_candidates = np.where(lam[: i_tr + 1] <= -0.09)[0]
    if len(lam_sep_candidates) > 0:
        xsep_lam = float(np.clip(x_over_c[int(lam_sep_candidates[0])], 0.0, 1.0))
        theta_te = float(theta_lam[int(lam_sep_candidates[0])])
        cd_surface = float("nan")
        return {
            "cd_surface": cd_surface,
            "theta_te": theta_te,
            "xtr": float(xtr),
            "xsep": xsep_lam,
            "theta": theta_lam,
            "cf": np.clip(np.nan_to_num(cf_lam, nan=0.0, posinf=0.0, neginf=0.0), 0.0, CF_MAX),
            "H": _H_lam,
        }

    theta_turb, H_turb, cf_turb, sep_index = turbulent_march_profile(s, ue, d_ueds, nu, i_tr, theta_tr)

    cf_full = np.array(cf_lam, copy=True)
    cf_full = np.clip(np.nan_to_num(cf_full, nan=0.0, posinf=0.0, neginf=0.0), 0.0, CF_MAX)
    if i_tr < len(cf_full):
        cf_full[i_tr:] = np.clip(np.nan_to_num(cf_turb[i_tr:], nan=0.0, posinf=0.0, neginf=0.0), 0.0, CF_MAX)

    theta_full = np.array(theta_lam, copy=True)
    if i_tr < len(theta_full):
        theta_full[i_tr:] = np.nan_to_num(theta_turb[i_tr:], nan=theta_tr)

    theta_te = float(theta_full[-1])
    cd_surface = float(2.0 * theta_te / CHORD)
    xsep = float(np.clip(x_over_c[int(sep_index)], 0.0, 1.0)) if sep_index is not None else 1.0

    if not np.isfinite(cd_surface) or cd_surface < 0.0 or cd_surface > 1.0:
        cd_surface = float("nan")

    return {
        "cd_surface": cd_surface,
        "theta_te": theta_te,
        "xtr": float(xtr),
        "xsep": xsep,
        "theta": theta_full,
        "cf": cf_full,
        "H": np.nan_to_num(H_turb, nan=H_TR),
    }


def write_profile_summary(profile: str, rows: list[dict[str, float]]) -> None:
    summary_path = get_profile_summary_path(profile)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha_deg", "CL"])
        for row in rows:
            writer.writerow(
                [
                    f"{row['alpha_deg']:.2f}",
                    f"{row['CL']:.12e}",
                ]
            )


def read_profile_summary(profile: str) -> list[dict[str, float]]:
    summary_path = get_profile_summary_path(profile)
    rows = []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            alpha = parse_float(row["alpha_deg"])
            cl = parse_float(row["CL"])
            if None in (alpha, cl):
                continue
            rows.append(
                {
                    "alpha_deg": alpha,
                    "CL": cl,
                }
            )
    return rows


def cleanup_cp_solution_file(alpha: float) -> None:
    cp_path = get_cp_solution_path(alpha)
    if cp_path.exists():
        cp_path.unlink()


def clear_profile_cache(profile: str) -> None:
    profile_dir = get_profile_cache_dir(profile)
    if not profile_dir.exists():
        return
    for path in profile_dir.glob("*"):
        if path.is_file():
            path.unlink()


def hspm_cache_is_complete(profile: str) -> bool:
    summary_path = get_profile_summary_path(profile)
    if not summary_path.exists():
        return False

    rows = read_profile_summary(profile)
    if len(rows) != len(ALPHA_RANGE):
        return False

    return all(
        get_surface_cache_path(profile, alpha, "upper").exists()
        and get_surface_cache_path(profile, alpha, "lower").exists()
        for alpha in ALPHA_RANGE
    )


def ensure_hspm_cache(profile: str) -> list[dict[str, float]]:
    if hspm_cache_is_complete(profile):
        for alpha in ALPHA_RANGE:
            cleanup_cp_solution_file(alpha)
        return read_profile_summary(profile)

    clear_profile_cache(profile)
    summary_rows = []
    for alpha in ALPHA_RANGE:
        panels = build_naca_panels(profile)
        prob = HSPM.HSPM(listOfPanels=panels, alphaRange=[alpha], referencePoint=[0.25, 0.0, 0.0])
        prob.run()
        cleanup_cp_solution_file(alpha)

        upper_coords, upper_vtang = prob.getUpperVtangential()
        lower_coords, lower_vtang = prob.getLowerVtangential()
        write_surface_cache(get_surface_cache_path(profile, alpha, "upper"), upper_coords, upper_vtang)
        write_surface_cache(get_surface_cache_path(profile, alpha, "lower"), lower_coords, lower_vtang)

        summary_rows.append(
            {
                "alpha_deg": float(alpha),
                "CL": float(prob.CL[-1]),
            }
        )

    write_profile_summary(profile, summary_rows)
    return summary_rows

def compute_profile_hspm_cl_curve(profile: str) -> dict[str, list[float] | str]:
    summary_rows = ensure_hspm_cache(profile)

    cl_hspm = [row["CL"] for row in summary_rows]

    return {
        "profile": profile,
        "alpha_range": list(ALPHA_RANGE),
        "cl_hspm": cl_hspm,
    }


def get_recommended_xtr(profile: str) -> float:
    return float(PROFILE_XTR.get(profile, XTR_DEFAULT))


def get_profile_cd0_base(profile: str) -> float:
    return float(PROFILE_CD0_BASE.get(profile, 0.020))


def compute_profile_cd_alpha_from_cache(profile: str, xtr: float | None = None) -> dict[str, list[float] | str | float]:
    if xtr is None:
        xtr = get_recommended_xtr(profile)
    summary_rows = ensure_hspm_cache(profile)
    cd_values = []
    theta_upper_values = []
    theta_lower_values = []
    xsep_upper_values = []
    xsep_lower_values = []
    cd_friction_values = []

    for row in summary_rows:
        alpha = float(row["alpha_deg"])
        upper_path = get_surface_cache_path(profile, alpha, "upper")
        lower_path = get_surface_cache_path(profile, alpha, "lower")

        upper_bl = compute_surface_boundary_layer_from_ue_file(upper_path, xtr=xtr)
        lower_bl = compute_surface_boundary_layer_from_ue_file(lower_path, xtr=xtr)

        cd_upper = float(upper_bl["cd_surface"])
        cd_lower = float(lower_bl["cd_surface"])
        xsep_upper = float(upper_bl["xsep"])
        xsep_lower = float(lower_bl["xsep"])

        cd_friction = 0.0
        if np.isfinite(cd_upper):
            cd_friction += cd_upper
        if np.isfinite(cd_lower):
            cd_friction += cd_lower

        cd_total = (
            get_profile_cd0_base(profile)
            + 0.10 * cd_friction
            + CD_SEP_GAIN * (1.0 - xsep_upper) ** 2
            + CD_SEP_GAIN * (1.0 - xsep_lower) ** 2
        )

        if not np.isfinite(cd_total) or cd_total < 0.0:
            cd_total = float("nan")
        cd_values.append(cd_total)
        theta_upper_values.append(float(upper_bl["theta_te"]))
        theta_lower_values.append(float(lower_bl["theta_te"]))
        xsep_upper_values.append(xsep_upper)
        xsep_lower_values.append(xsep_lower)
        cd_friction_values.append(cd_friction)

    return {
        "profile": profile,
        "alpha_range": [float(row["alpha_deg"]) for row in summary_rows],
        "cd_alpha": cd_values,
        "theta_te_upper": theta_upper_values,
        "theta_te_lower": theta_lower_values,
        "xsep_upper": xsep_upper_values,
        "xsep_lower": xsep_lower_values,
        "cd_friction_raw": cd_friction_values,
        "xtr": float(xtr),
    }


def compute_profile_cl_cd_from_cache(profile: str, xtr: float | None = None) -> dict[str, object]:
    if xtr is None:
        xtr = get_recommended_xtr(profile)

    summary_rows = ensure_hspm_cache(profile)
    cd_result = compute_profile_cd_alpha_from_cache(profile, xtr=xtr)

    alpha_range = np.asarray([float(row["alpha_deg"]) for row in summary_rows], dtype=float)
    cl_curve = np.asarray([float(row["CL"]) for row in summary_rows], dtype=float)
    cd_curve = np.asarray(cd_result["cd_alpha"], dtype=float)
    mask = np.isfinite(cl_curve) & np.isfinite(cd_curve)

    return {
        "profile": profile,
        "alpha_range": alpha_range,
        "cl_curve": cl_curve,
        "cd_curve": cd_curve,
        "cl_valid": cl_curve[mask],
        "cd_valid": cd_curve[mask],
        "xtr": float(xtr),
    }


def plot_cd_alpha_from_cache(results: list[dict[str, list[float] | str | float]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
    axis_map = {
        "NACA-0012": axes[0],
        "NACA-4412": axes[1],
    }

    for result in results:
        profile = str(result["profile"])
        alpha_range = result["alpha_range"]
        cd_alpha = result["cd_alpha"]
        xtr = float(result["xtr"])
        ax = axis_map[profile]

        ax.plot(alpha_range, cd_alpha, marker="o", linewidth=1.8, label=f"Couche limite cachee, xtr={xtr:.2f}")
        ax.set_title(f"{profile} - CD en fonction de l'angle")
        ax.set_xlabel("Angle d'attaque [deg]")
        ax.set_ylabel("CD [-]")
        ax.grid(True)
        ax.legend()

    fig.savefig(OUTPUT_2D_CD_ALPHA_FILE, dpi=200)


def print_cd_alpha_from_cache(results: list[dict[str, list[float] | str | float]]) -> None:
    print()
    print("=== Partie 2 - CD(alpha) depuis le cache Ue ===")
    print("Methode utilisee :")
    print("1. Lecture des fichiers Ue upper/lower deja stockes dans le cache HSPM")
    print("2. Calcul de couche limite par Thwaites puis marche turbulente")
    print("3. Somme des contributions extrados et intrados pour obtenir CD(alpha)")
    print()
    for result in results:
        profile = str(result["profile"])
        alpha_range = np.asarray(result["alpha_range"], dtype=float)
        cd_alpha = np.asarray(result["cd_alpha"], dtype=float)
        theta_u = np.asarray(result["theta_te_upper"], dtype=float)
        theta_l = np.asarray(result["theta_te_lower"], dtype=float)
        xsep_u = np.asarray(result["xsep_upper"], dtype=float)
        xsep_l = np.asarray(result["xsep_lower"], dtype=float)
        mask = np.isfinite(cd_alpha)
        valid = cd_alpha[mask]
        cd0 = float(np.interp(0.0, alpha_range[mask], cd_alpha[mask])) if np.count_nonzero(mask) >= 2 else float("nan")
        theta_u0 = float(np.interp(0.0, alpha_range[mask], theta_u[mask])) if np.count_nonzero(mask) >= 2 else float("nan")
        theta_l0 = float(np.interp(0.0, alpha_range[mask], theta_l[mask])) if np.count_nonzero(mask) >= 2 else float("nan")
        xsep_u0 = float(np.interp(0.0, alpha_range[mask], xsep_u[mask])) if np.count_nonzero(mask) >= 2 else float("nan")
        xsep_l0 = float(np.interp(0.0, alpha_range[mask], xsep_l[mask])) if np.count_nonzero(mask) >= 2 else float("nan")
        print(
            f"{profile}: "
            f"points valides = {len(valid)}/{len(cd_alpha)}, "
            f"CD(0 deg) = {cd0:.5f}, "
            f"theta_TE upper = {theta_u0:.6e}, "
            f"theta_TE lower = {theta_l0:.6e}, "
            f"xsep upper = {xsep_u0:.3f}, "
            f"xsep lower = {xsep_l0:.3f}, "
            f"xtr = {float(result['xtr']):.2f}"
        )
    print(f"Figure sauvegardee : {OUTPUT_2D_CD_ALPHA_FILE.name}")
    print()


def plot_cl_cd_from_cache(results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
    axis_map = {
        "NACA-0012": axes[0],
        "NACA-4412": axes[1],
    }

    for result in results:
        profile = str(result["profile"])
        ax = axis_map[profile]
        ax.plot(result["cd_valid"], result["cl_valid"], marker="o", linewidth=1.8, label=f"xtr={float(result['xtr']):.2f}")
        at_cd, at_cl = read_xy_csv(AIRFOIL_TOOLS_FILES[profile]["cl_cd"])
        if at_cd and at_cl:
            ax.plot(at_cd, at_cl, linestyle="--", linewidth=1.8, label="Airfoil Tools")
        ax.set_title(f"{profile} - CL en fonction de CD")
        ax.set_xlabel("CD [-]")
        ax.set_ylabel("CL [-]")
        ax.grid(True)
        ax.legend()

    fig.savefig(OUTPUT_2D_CD_CL_FILE, dpi=200)


def print_cl_cd_from_cache(results: list[dict[str, object]]) -> None:
    print()
    print("=== Partie 2 - Polar CL(CD) ===")
    print("Methode utilisee :")
    print("1. Lecture de CL(alpha) et des fichiers Ue dans le cache HSPM")
    print("2. Calcul de CD(alpha) avec la couche limite 2D")
    print("3. Construction de la polar CL(CD) avec les xtr retenus par profil")
    print("4. Superposition des donnees Airfoil Tools pour comparaison")
    print()
    for result in results:
        profile = str(result["profile"])
        print(
            f"{profile}: "
            f"xtr retenu = {float(result['xtr']):.2f}, "
            f"points valides = {len(result['cl_valid'])}/{len(result['cl_curve'])}"
        )
    print(f"Figure sauvegardee : {OUTPUT_2D_CD_CL_FILE.name}")
    print()


def run_partie2_question1() -> None:
    cl_results = [
        compute_profile_hspm_cl_curve("NACA-0012"),
        compute_profile_hspm_cl_curve("NACA-4412"),
    ]
    cd_results = [
        compute_profile_cd_alpha_from_cache("NACA-0012"),
        compute_profile_cd_alpha_from_cache("NACA-4412"),
    ]
    polar_results = [
        compute_profile_cl_cd_from_cache("NACA-0012"),
        compute_profile_cl_cd_from_cache("NACA-4412"),
    ]

    plot_partie2_question1(cl_results)
    print_partie2_question1(cl_results)
    plot_cd_alpha_from_cache(cd_results)
    print_cd_alpha_from_cache(cd_results)
    plot_cl_cd_from_cache(polar_results)
    print_cl_cd_from_cache(polar_results)


def plot_ue_geometry_diagnostic(
    profiles: list[str] | None = None,
    angles: list[float] | None = None,
) -> None:
    if profiles is None:
        profiles = ["NACA-0012", "NACA-4412"]
    if angles is None:
        angles = PROFILE_DIAGNOSTIC_ANGLES

    fig, axes = plt.subplots(len(profiles), len(angles), figsize=(3.2 * len(angles), 3.0 * len(profiles)), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for i, profile in enumerate(profiles):
        ensure_hspm_cache(profile)
        for j, alpha in enumerate(angles):
            ax = axes[i, j]
            upper_points, _upper_ue = read_surface_cache(get_surface_cache_path(profile, alpha, "upper"))
            lower_points, _lower_ue = read_surface_cache(get_surface_cache_path(profile, alpha, "lower"))

            xu = [point[0] for point in upper_points]
            zu = [point[2] for point in upper_points]
            xl = [point[0] for point in lower_points]
            zl = [point[2] for point in lower_points]

            ax.plot(xu, zu, "-o", markersize=2.2, linewidth=1.0, label="upper")
            ax.plot(xl, zl, "-o", markersize=2.2, linewidth=1.0, label="lower")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True)
            ax.set_title(f"{profile}\nalpha={alpha:.1f} deg")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.savefig(OUTPUT_UE_GEOMETRY_FILE, dpi=200)


def print_ue_geometry_diagnostic() -> None:
    print()
    print("=== Diagnostic geometrie depuis les fichiers Ue ===")
    print("Methode utilisee :")
    print("1. Lecture des points x,z stockes dans Ue_upper et Ue_lower")
    print("2. Reconstruction du profil pour plusieurs angles")
    print("3. Verification visuelle de l'ordre des points et de la geometrie")
    print()
    print(f"Figure sauvegardee : {OUTPUT_UE_GEOMETRY_FILE.name}")
    print()


def score_cd_curve(alpha_range: np.ndarray, cd_alpha: np.ndarray, profile: str) -> float:
    mask = np.isfinite(cd_alpha)
    if np.count_nonzero(mask) < max(8, len(cd_alpha) // 2):
        return float("inf")

    alpha_valid = alpha_range[mask]
    cd_valid = cd_alpha[mask]

    if np.any(cd_valid < 0.0):
        return float("inf")

    first_diff = np.diff(cd_valid)
    second_diff = np.diff(cd_valid, n=2)

    roughness = float(np.mean(np.abs(second_diff))) if len(second_diff) > 0 else 0.0
    spike = float(np.max(np.abs(first_diff))) if len(first_diff) > 0 else 0.0
    spread = float(np.nanmax(cd_valid) - np.nanmin(cd_valid))
    invalid_penalty = 10.0 * (len(cd_alpha) - np.count_nonzero(mask))

    score = roughness + 2.0 * spike + 0.1 * spread + invalid_penalty

    if profile == "NACA-0012":
        mirrored = []
        for alpha in alpha_valid:
            target = -alpha
            idx = np.where(np.isclose(alpha_valid, target, atol=1.0e-9))[0]
            if len(idx) == 0:
                continue
            mirrored.append(abs(cd_valid[np.where(alpha_valid == alpha)[0][0]] - cd_valid[idx[0]]))
        if mirrored:
            score += float(np.mean(mirrored))

    return score


def compute_cd_alpha_sweep(profile: str, xtr_candidates: list[float] | None = None) -> dict[str, object]:
    if xtr_candidates is None:
        xtr_candidates = XTR_CANDIDATES

    curves = []
    alpha_range_reference = None
    for xtr in xtr_candidates:
        result = compute_profile_cd_alpha_from_cache(profile, xtr=xtr)
        alpha_range = np.asarray(result["alpha_range"], dtype=float)
        cd_alpha = np.asarray(result["cd_alpha"], dtype=float)
        score = score_cd_curve(alpha_range, cd_alpha, profile)
        curves.append(
            {
                "xtr": float(xtr),
                "alpha_range": alpha_range,
                "cd_alpha": cd_alpha,
                "score": score,
                "valid_count": int(np.count_nonzero(np.isfinite(cd_alpha))),
            }
        )
        if alpha_range_reference is None:
            alpha_range_reference = alpha_range

    curves_sorted = sorted(curves, key=lambda item: item["score"])
    best_curve = curves_sorted[0] if curves_sorted else None

    return {
        "profile": profile,
        "alpha_range": alpha_range_reference,
        "curves": curves,
        "best_curve": best_curve,
    }


def plot_cd_alpha_sweep(results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 10), constrained_layout=True)
    axis_map = {
        "NACA-0012": axes[0],
        "NACA-4412": axes[1],
    }

    for result in results:
        profile = str(result["profile"])
        ax = axis_map[profile]

        for curve in result["curves"]:
            alpha_range = curve["alpha_range"]
            cd_alpha = curve["cd_alpha"]
            xtr = curve["xtr"]
            score = curve["score"]
            label = f"xtr={xtr:.2f}"
            if result["best_curve"] is not None and abs(xtr - float(result["best_curve"]["xtr"])) < 1.0e-12:
                label += " (retenu)"
                ax.plot(alpha_range, cd_alpha, linewidth=2.4, marker="o", label=label)
            else:
                ax.plot(alpha_range, cd_alpha, linewidth=1.1, alpha=0.65, label=label)

        ax.set_title(f"{profile} - Balayage de CD en fonction de l'angle")
        ax.set_xlabel("Angle d'attaque [deg]")
        ax.set_ylabel("CD [-]")
        ax.grid(True)
        ax.legend(ncol=2, fontsize=8)

    fig.savefig(OUTPUT_2D_CD_ALPHA_SWEEP_FILE, dpi=200)


def print_cd_alpha_sweep(results: list[dict[str, object]]) -> None:
    print()
    print("=== Partie 2 - Balayage des points de transition ===")
    print("Methode utilisee :")
    print("1. Lecture des fichiers Ue deja caches")
    print("2. Calcul de CD(alpha) pour plusieurs points de transition xtr")
    print("3. Choix d'un xtr recommande a partir de la regularite de la courbe")
    print("4. Aux petits angles, l'algorithme peut aller jusqu'au bord de fuite sans separation")
    print()
    for result in results:
        profile = str(result["profile"])
        best_curve = result["best_curve"]
        print(f"Profil : {profile}")
        if best_curve is None or not np.isfinite(float(best_curve["score"])):
            print("  Aucun xtr robuste trouve.")
        else:
            print(f"  xtr recommande = {float(best_curve['xtr']):.2f}")
            print(f"  score          = {float(best_curve['score']):.5f}")
            print(f"  points valides = {int(best_curve['valid_count'])}/{len(best_curve['cd_alpha'])}")
        print()
    print(f"Figure sauvegardee : {OUTPUT_2D_CD_ALPHA_SWEEP_FILE.name}")
    print()


def plot_partie2_question1(results: list[dict[str, list[float] | str]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
    axis_map = {
        "NACA-0012": axes[0],
        "NACA-4412": axes[1],
    }

    for result in results:
        profile = str(result["profile"])
        alpha_range = result["alpha_range"]
        cl_hspm = result["cl_hspm"]
        ax_cl = axis_map[profile]

        ax_cl.plot(alpha_range, cl_hspm, marker="o", linewidth=1.8, label="HSPM")
        at_alpha, at_cl = read_xy_csv(AIRFOIL_TOOLS_FILES[profile]["cl_alpha"])
        if at_alpha and at_cl:
            ax_cl.plot(at_alpha, at_cl, linestyle="--", linewidth=1.8, label="Airfoil Tools")
        ax_cl.set_title(f"{profile} - CL en fonction de l'angle")
        ax_cl.set_xlabel("Angle d'attaque [deg]")
        ax_cl.set_ylabel("CL [-]")
        ax_cl.grid(True)
        ax_cl.legend()

    fig.savefig(OUTPUT_2D_Q1_FILE, dpi=200)


def print_partie2_question1(results: list[dict[str, list[float] | str]]) -> None:
    print()
    print("=== Partie 2 - Question 1 ===")
    print("Methode utilisee :")
    print("1. HSPM pour calculer la solution potentielle sur les profils NACA0012 et NACA4412")
    print("2. Export uniquement des fichiers CL et des fichiers Ue extrados/intrados")
    print("3. Suppression automatique des fichiers CPsol generes par HSPM")
    print("4. Superposition des donnees Airfoil Tools pour comparaison")
    print()
    for result in results:
        profile = str(result["profile"])
        cl_hspm = result["cl_hspm"]
        print(
            f"{profile}: "
            f"CL(0 deg) = {np.interp(0.0, ALPHA_RANGE, cl_hspm):.4f}"
        )
    print(f"Figure sauvegardee : {OUTPUT_2D_Q1_FILE.name}")
    print(f"Dossier de sortie HSPM : {CACHE_DIR}")
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

def compute_profile_separation_from_cache(profile: str, xtr: float | None = None) -> dict[str, object]:
    if xtr is None:
        xtr = get_recommended_xtr(profile)

    summary_rows = ensure_hspm_cache(profile)

    alpha_range = []
    cl_curve = []
    xsep_upper = []
    xsep_lower = []

    for row in summary_rows:
        alpha = float(row["alpha_deg"])
        cl = float(row["CL"])

        upper_path = get_surface_cache_path(profile, alpha, "upper")
        lower_path = get_surface_cache_path(profile, alpha, "lower")

        upper_bl = compute_surface_boundary_layer_from_ue_file(upper_path, xtr=xtr)
        lower_bl = compute_surface_boundary_layer_from_ue_file(lower_path, xtr=xtr)

        alpha_range.append(alpha)
        cl_curve.append(cl)
        xsep_upper.append(float(upper_bl["xsep"]))
        xsep_lower.append(float(lower_bl["xsep"]))

    alpha_arr = np.asarray(alpha_range, dtype=float)
    cl_arr = np.asarray(cl_curve, dtype=float)
    xsep_upper_arr = np.asarray(xsep_upper, dtype=float)
    xsep_lower_arr = np.asarray(xsep_lower, dtype=float)

    # Conversion en pourcentage de corde
    sep_upper_pct = 100.0 * np.clip(xsep_upper_arr, 0.0, 1.0)
    sep_lower_pct = 100.0 * np.clip(xsep_lower_arr, 0.0, 1.0)

    # Pour la courbe s_sep = f(CL), on trie selon CL pour éviter une courbe qui revient en arrière
    sort_idx = np.argsort(cl_arr)

    return {
        "profile": profile,
        "alpha_range": alpha_arr,
        "cl_curve": cl_arr,
        "sep_upper_pct": sep_upper_pct,
        "sep_lower_pct": sep_lower_pct,
        "cl_sorted": cl_arr[sort_idx],
        "sep_upper_pct_sorted": sep_upper_pct[sort_idx],
        "sep_lower_pct_sorted": sep_lower_pct[sort_idx],
        "xtr": float(xtr),
    }


def plot_partie2_question3(results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    axis_map = {
        "NACA-0012": (axes[0, 0], axes[0, 1]),
        "NACA-4412": (axes[1, 0], axes[1, 1]),
    }

    for result in results:
        profile = str(result["profile"])
        ax_alpha, ax_cl = axis_map[profile]

        alpha_range = np.asarray(result["alpha_range"], dtype=float)
        cl_curve = np.asarray(result["cl_curve"], dtype=float)
        sep_upper_pct = np.asarray(result["sep_upper_pct"], dtype=float)
        sep_lower_pct = np.asarray(result["sep_lower_pct"], dtype=float)

        cl_sorted = np.asarray(result["cl_sorted"], dtype=float)
        sep_upper_pct_sorted = np.asarray(result["sep_upper_pct_sorted"], dtype=float)
        sep_lower_pct_sorted = np.asarray(result["sep_lower_pct_sorted"], dtype=float)

        # s_sep = f(alpha)
        ax_alpha.plot(alpha_range, sep_upper_pct, marker="o", linewidth=1.8, label="Extrados")
        ax_alpha.plot(alpha_range, sep_lower_pct, marker="s", linewidth=1.8, label="Intrados")
        ax_alpha.set_title(f"{profile} - Position du point de séparation en fonction de l'angle")
        ax_alpha.set_xlabel("Angle d'attaque [deg]")
        ax_alpha.set_ylabel(r"$s_{sep}$ [% de la corde]")
        ax_alpha.set_ylim(0.0, 105.0)
        ax_alpha.grid(True)
        ax_alpha.legend()

        # s_sep = f(CL)
        ax_cl.plot(cl_sorted, sep_upper_pct_sorted, marker="o", linewidth=1.8, label="Extrados")
        ax_cl.plot(cl_sorted, sep_lower_pct_sorted, marker="s", linewidth=1.8, label="Intrados")
        ax_cl.set_title(f"{profile} - Position du point de séparation en fonction de $C_L$")
        ax_cl.set_xlabel(r"$C_L$ [-]")
        ax_cl.set_ylabel(r"$s_{sep}$ [% de la corde]")
        ax_cl.set_ylim(0.0, 105.0)
        ax_cl.grid(True)
        ax_cl.legend()

    fig.savefig(OUTPUT_2D_Q3_ALPHA_CL_FILE, dpi=200)


def print_partie2_question3(results: list[dict[str, object]]) -> None:
    print()
    print("=== Partie 2 - Question 3 ===")
    print("Methode utilisee :")
    print("1. Lecture des fichiers Ue extrados/intrados depuis le cache HSPM")
    print("2. Calcul de la couche limite sur chaque face")
    print("3. Extraction de la position de separation xsep")
    print("4. Conversion en pourcentage de corde avec 100% s'il n'y a pas de separation")
    print()

    for result in results:
        profile = str(result["profile"])
        alpha_range = np.asarray(result["alpha_range"], dtype=float)
        cl_curve = np.asarray(result["cl_curve"], dtype=float)
        sep_upper_pct = np.asarray(result["sep_upper_pct"], dtype=float)
        sep_lower_pct = np.asarray(result["sep_lower_pct"], dtype=float)

        no_sep_upper = int(np.count_nonzero(np.isclose(sep_upper_pct, 100.0)))
        no_sep_lower = int(np.count_nonzero(np.isclose(sep_lower_pct, 100.0)))

        i_min_upper = int(np.argmin(sep_upper_pct))
        i_min_lower = int(np.argmin(sep_lower_pct))

        print(f"Profil : {profile}")
        print(
            f"  Extrados : min(s_sep) = {sep_upper_pct[i_min_upper]:.2f}% "
            f"pour alpha = {alpha_range[i_min_upper]:.2f} deg, "
            f"CL = {cl_curve[i_min_upper]:.4f}"
        )
        print(
            f"  Intrados : min(s_sep) = {sep_lower_pct[i_min_lower]:.2f}% "
            f"pour alpha = {alpha_range[i_min_lower]:.2f} deg, "
            f"CL = {cl_curve[i_min_lower]:.4f}"
        )
        print(f"  Nombre de cas sans separation extrados : {no_sep_upper}/{len(alpha_range)}")
        print(f"  Nombre de cas sans separation intrados : {no_sep_lower}/{len(alpha_range)}")
        print()

    print(f"Figure sauvegardee : {OUTPUT_2D_Q3_ALPHA_CL_FILE.name}")
    print()


def run_partie2_question3() -> None:
    sep_results = [
        compute_profile_separation_from_cache("NACA-0012"),
        compute_profile_separation_from_cache("NACA-4412"),
    ]

    plot_partie2_question3(sep_results)
    print_partie2_question3(sep_results)

def plot_partie2_question4(results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
    axis_map = {
        "NACA-0012": axes[0],
        "NACA-4412": axes[1],
    }

    for result in results:
        profile = str(result["profile"])
        ax = axis_map[profile]

        alpha_range = np.asarray(result["alpha_range"], dtype=float)
        delta_cp = np.asarray(result["delta_cp_valarezo"], dtype=float)
        alpha_stall = float(result["alpha_stall_2d"])
        clmax = float(result["clmax_2d"])

        ax.plot(alpha_range, delta_cp, marker="o", linewidth=1.8, label=r"$\Delta C_p$")
        ax.axhline(VALAREZO_CRITERION, linestyle="--", linewidth=1.5, label=f"Critere Valarezo = {VALAREZO_CRITERION:.1f}")
        ax.axvline(alpha_stall, linestyle=":", linewidth=1.5, label=rf"$\alpha_{{stall}}={alpha_stall:.2f}^\circ$")
        ax.set_title(f"{profile} - Methode de Valarezo")
        ax.set_xlabel("Angle d'attaque [deg]")
        ax.set_ylabel(r"$\Delta C_p = |C_{p,TE} - C_{p,\min}|$ [-]")
        ax.grid(True)
        ax.legend()

        ax.text(
            0.02,
            0.05,
            rf"$\alpha_{{stall}}={alpha_stall:.2f}^\circ$" + "\n" + rf"$C_{{L,\max}}={clmax:.4f}$",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    fig.savefig(OUTPUT_2D_Q4_VALAREZO_FILE, dpi=200)


def print_partie2_question4(results: list[dict[str, object]]) -> None:
    print()
    print("=== Partie 2 - Question 4 ===")
    print("Methode utilisee :")
    print("1. Calcul HSPM sur toute la plage d'angles")
    print("2. Extraction de deltaCPvalarezo = |Cp_TE - Cp_min|")
    print("3. Interpolation de l'angle pour lequel deltaCP atteint le critere impose")
    print("4. Interpolation du CL correspondant pour obtenir CLmax")
    print()

    for result in results:
        profile = str(result["profile"])
        alpha_stall = float(result["alpha_stall_2d"])
        clmax = float(result["clmax_2d"])

        print(f"Profil : {profile}")
        print(f"  Critere de Valarezo = {VALAREZO_CRITERION:.2f}")
        print(f"  alpha_stall_2D      = {alpha_stall:.4f} deg")
        print(f"  CLmax_2D            = {clmax:.6f}")
        print()

    print(f"Figure sauvegardee : {OUTPUT_2D_Q4_VALAREZO_FILE.name}")
    print()


def run_partie2_question4() -> None:
    results = [
        compute_2d_valarezo("NACA-0012"),
        compute_2d_valarezo("NACA-4412"),
    ]

    plot_partie2_question4(results)
    print_partie2_question4(results)

def main() -> None:
    task = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "partie2question1"

    if task == "partie2question1" or task in ("partie2_q1", "question_2d_1", "q2d1"):
        run_partie2_question1()
        plt.show()
        return

    if task == "partie3question1" or task in ("partie3_q1", "question_1", "q1"):
        partie3_q1_results = solve_partie3_question1()
        print_partie3_question1(partie3_q1_results)
        return


    if task == "partie2question3" or task in ("partie2_q3", "question_2d_3", "q2d3"):
        run_partie2_question3()
        plt.show()
        return
    
    if task == "partie2question4" or task in ("partie2_q4", "question_2d_4", "q2d4"):
        run_partie2_question4()
        plt.show()
        return
    
    if task not in (
        "partie2question1",
        "partie2_q1",
        "question_2d_1",
        "q2d1",
        "partie3question1",
        "partie3_q1",
        "question_1",
        "q1",
        "partie2question3",
        "partie2_q3",
        "question_2d_3",
        "q2d3",
        "partie2question4",
        "partie2_q4",
        "question_2d_4",
        "q2d4",
    ):
        print("Argument non reconnu.")
        print("Utilise :")
        print("  python projet_aer8270.py partie2question1")
        print("  python projet_aer8270.py partie3question1")
        print("  python projet_aer8270.py partie2question3")
        print("  python projet_aer8270.py partie2question4")
        return


if __name__ == "__main__":
    main()
