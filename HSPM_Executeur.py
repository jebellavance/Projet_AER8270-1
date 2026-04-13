import numpy as np
import HSPM as hspm_module
import geometryGenerator
 
# ─────────────────────────────────────────────────────────────────
# CONDITIONS PHYSIQUES DE VOL
# (à modifier selon les conditions réelles de chaque test)
# ─────────────────────────────────────────────────────────────────
 
V_INF_MS   = 12.0        # Vitesse à l'infini [m/s]
CHORD_M    = 6.0*0.0254  # Corde (6 pouces = 0.1524 m)
T_CELSIUS  = 19.0        # Température ambiante [°C]
P_ATM_PA   = 97600       # Pression atmosphérique [Pa]
 
# ─────────────────────────────────────────────────────────────────
# PARAMÈTRES HSPM
# ─────────────────────────────────────────────────────────────────
 
POINTS_PER_SURFACE = 100
ALPHA_RANGE        = list(range(-20, 21, 1))   # -20° à 20°, pas de 1°
REFERENCE_POINT    = [0.25, 0.0, 0.0]          # Quart de corde
 
PROFILS = {
    'NACA0012': {
        'maxCamber'          : 0.0,
        'positionOfMaxCamber': 0.0,
        'thickness'          : 12.0,
    },
    'NACA4412': {
        'maxCamber'          : 4.0,
        'positionOfMaxCamber': 4.0,
        'thickness'          : 12.0,
    },
}
 
# ─────────────────────────────────────────────────────────────────
# GRAPHIQUE DE VALAREZO DIGITALISÉ
# Source : Valarezo & Chin (1992), Figure 2
# ΔCp_crit = f(Re_c [Millions], Mach)
# Courbes disponibles : Mach = 0.15, 0.20, 0.25
# Pour Re_c < 1e6 → extrapolation d'ordre 0 : ΔCp_crit = 7.0
# ─────────────────────────────────────────────────────────────────
 
VALAREZO_GRAPH = {
    # Mach : (array Re [millions], array ΔCp)
    0.15: (
        np.array([ 1.0,  2.0,  3.0,  4.0,  6.0,  8.0, 10.0, 14.0, 20.0]),
        np.array([ 8.0, 11.0, 13.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0])
    ),
    0.20: (
        np.array([ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  8.0, 10.0, 14.0, 20.0]),
        np.array([ 7.0, 10.0, 11.5, 12.5, 13.0, 13.2, 13.5, 13.7, 14.0, 14.0])
    ),
    0.25: (
        np.array([ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  8.0, 10.0, 14.0, 20.0]),
        np.array([ 7.0,  9.0, 10.5, 11.5, 12.0, 12.5, 13.0, 13.5, 13.5, 13.5])
    ),
}
 
MACH_CURVES     = np.array(sorted(VALAREZO_GRAPH.keys()))  # [0.15, 0.20, 0.25]
DELTACP_MIN     = 7.0   # Valeur minimale (extrapolation ordre 0 pour Re < 1e6)
RE_MIN_MILLIONS = 1.0   # Seuil en dessous duquel on applique l'extrapolation d'ordre 0
 
 
# ─────────────────────────────────────────────────────────────────
# FONCTIONS : CONDITIONS D'ÉCOULEMENT
# ─────────────────────────────────────────────────────────────────
 
def air_properties(T_C, P_Pa):
    """
    Calcule les propriétés de l'air à partir de T [°C] et P [Pa].
    Retourne (rho [kg/m³], nu [m²/s], a [m/s]).
    """
    R_air  = 287.058   # Constante gaz parfait air [J/(kg·K)]
    gamma  = 1.4
    T_K    = T_C + 273.15
 
    rho    = P_Pa / (R_air * T_K)
 
    # Viscosité dynamique : loi de Sutherland
    mu_ref = 1.716e-5   # [Pa·s] à T_ref = 273.15 K
    T_ref  = 273.15
    S      = 110.4      # Constante de Sutherland [K]
    mu     = mu_ref * (T_K / T_ref)**1.5 * (T_ref + S) / (T_K + S)
 
    nu     = mu / rho
    a      = np.sqrt(gamma * R_air * T_K)
 
    return rho, nu, a
 
 
def compute_flow_conditions(V_inf, chord, T_C, P_Pa):
    """
    Calcule Re_c et Mach à partir des conditions physiques.
    Retourne (Re_c, Mach, rho, nu, a).
    """
    rho, nu, a = air_properties(T_C, P_Pa)
    Re_c       = V_inf * chord / nu
    Mach       = V_inf / a
    return Re_c, Mach, rho, nu, a
 
 
# ─────────────────────────────────────────────────────────────────
# FONCTIONS : CRITÈRE DE VALAREZO
# ─────────────────────────────────────────────────────────────────
 
def get_deltaCP_crit_single_mach(Re_c, mach_key):
    """
    Interpole ΔCp_crit sur une courbe Mach donnée du graphique de Valarezo.
    Applique l'extrapolation d'ordre 0 si Re_c < Re_min.
    """
    Re_arr, dCP_arr = VALAREZO_GRAPH[mach_key]
    Re_millions     = Re_c / 1e6
 
    if Re_millions <= RE_MIN_MILLIONS:
        return DELTACP_MIN
 
    if Re_millions >= Re_arr[-1]:
        return float(dCP_arr[-1])
 
    return float(np.interp(Re_millions, Re_arr, dCP_arr))
 
 
def get_deltaCP_valarezo(Re_c, Mach):
    """
    Retourne ΔCp_crit depuis le graphique de Valarezo pour des conditions
    (Re_c, Mach) quelconques, par interpolation entre les courbes Mach disponibles.
 
    - Si Re_c < 1e6             → extrapolation d'ordre 0 : ΔCp_crit = 7.0
    - Si Mach <= Mach_min (0.15) → utilise la courbe Mach = 0.15
    - Si Mach >= Mach_max (0.25) → utilise la courbe Mach = 0.25
    - Sinon                      → interpolation linéaire entre courbes encadrantes
    """
    if Re_c < 1e6:
        return DELTACP_MIN
 
    if Mach <= MACH_CURVES[0]:
        return get_deltaCP_crit_single_mach(Re_c, MACH_CURVES[0])
 
    if Mach >= MACH_CURVES[-1]:
        return get_deltaCP_crit_single_mach(Re_c, MACH_CURVES[-1])
 
    # Trouver les deux courbes Mach encadrantes
    idx_upper = int(np.searchsorted(MACH_CURVES, Mach))
    idx_lower = idx_upper - 1
 
    mach_lo = MACH_CURVES[idx_lower]
    mach_hi = MACH_CURVES[idx_upper]
 
    dcp_lo  = get_deltaCP_crit_single_mach(Re_c, mach_lo)
    dcp_hi  = get_deltaCP_crit_single_mach(Re_c, mach_hi)
 
    t = (Mach - mach_lo) / (mach_hi - mach_lo)
    return dcp_lo + t * (dcp_hi - dcp_lo)
 
 
# ─────────────────────────────────────────────────────────────────
# FONCTIONS : HSPM
# ─────────────────────────────────────────────────────────────────
 
def vector3_list_to_x(points):
    """Convertit une liste de Vector3 en array numpy des coordonnées x (= x/c)."""
    return np.array([p[0] for p in points])
 
 
def run_single_alpha(panels, alpha_deg):
    """
    Lance HSPM pour un seul angle d'attaque.
    Retourne un dict avec CL, CD, CM, deltaCP, et Ue(x) extrados/intrados.
    """
    prob = hspm_module.HSPM(
        listOfPanels   = panels,
        alphaRange     = [alpha_deg],
        referencePoint = REFERENCE_POINT
    )
    prob.run()
 
    upper_coords, upper_Vtang = prob.getUpperVtangential()
    lower_coords, lower_Vtang = prob.getLowerVtangential()
 
    return {
        'CL'      : prob.CL[0],
        'CD'      : prob.CD[0],
        'CM'      : prob.CM[0],
        'deltaCP' : prob.deltaCPvalarezo[0],
        'x_upper' : vector3_list_to_x(upper_coords),
        'Ue_upper': np.array(upper_Vtang),
        'x_lower' : vector3_list_to_x(lower_coords),
        'Ue_lower': np.array(lower_Vtang),
    }
 
 
def find_clmax_valarezo(alpha_arr, deltaCP_arr, CL_arr, criterion):
    """
    Interpole CLmax et alphaMax au moment où deltaCP atteint le critère Valarezo.
    Ne considère que le côté positif (CL croissant) pour trouver CLmax.
    Retourne (alphaMax, CLmax) ou (None, None) si critère jamais atteint.
    """
    idx_zero    = int(np.argmin(np.abs(CL_arr)))
    alpha_pos   = alpha_arr[idx_zero:]
    deltaCP_pos = deltaCP_arr[idx_zero:]
    CL_pos      = CL_arr[idx_zero:]
 
    if criterion > np.max(deltaCP_pos):
        print("  AVERTISSEMENT : critère Valarezo jamais atteint sur la plage d'alpha.")
        return None, None
 
    mask      = deltaCP_pos >= criterion
    idx_cross = int(np.argmax(mask))
 
    if idx_cross == 0:
        return float(alpha_pos[0]), float(CL_pos[0])
 
    dp0, dp1 = deltaCP_pos[idx_cross - 1], deltaCP_pos[idx_cross]
    t        = (criterion - dp0) / (dp1 - dp0)
    alphaMax = float(alpha_pos[idx_cross-1] + t*(alpha_pos[idx_cross] - alpha_pos[idx_cross-1]))
    CLmax    = float(CL_pos[idx_cross-1]    + t*(CL_pos[idx_cross]    - CL_pos[idx_cross-1]))
 
    return alphaMax, CLmax
 
 
# ─────────────────────────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────────────────────────
 
def run_all_profiles(V_inf=V_INF_MS, chord=CHORD_M, T_C=T_CELSIUS,
                     P_Pa=P_ATM_PA, alpha_range=None, verbose=True):
    """
    Lance HSPM sur tous les profils définis.
    Calcule Re_c, Mach, ΔCp_crit (graphique Valarezo) avant de chercher CLmax.
 
    Retourne un dict :
        results[profil_name] = {
            'alpha', 'CL', 'CD', 'CM', 'deltaCP',
            'CLmax', 'alphaMax', 'deltaCP_crit',
            'Re_c', 'Mach', 'rho', 'nu',
            'Ue_upper', 'Ue_lower'
        }
    """
    if alpha_range is None:
        alpha_range = ALPHA_RANGE
 
    # ── Conditions d'écoulement ───────────────────────────────────
    Re_c, Mach, rho, nu, a = compute_flow_conditions(V_inf, chord, T_C, P_Pa)
    deltaCP_crit            = get_deltaCP_valarezo(Re_c, Mach)
    extrap_note             = "(extrapolation ordre 0, Re < 1e6)" if Re_c < 1e6 \
                              else "(interpolé sur graphique Valarezo)"
 
    if verbose:
        print(f"\n{'='*55}")
        print(f"  CONDITIONS D'ÉCOULEMENT")
        print(f"  V_inf    = {V_inf:.2f} m/s")  
        print(f"  corde    = {chord*100:.2f} cm")
        print(f"  T        = {T_C:.1f} °C")
        print(f"  P_atm    = {P_Pa:.0f} Pa")
        print(f"  rho      = {rho:.4f} kg/m³")
        print(f"  nu       = {nu:.2e} m²/s")
        print(f"  a        = {a:.2f} m/s")
        print(f"  Re_corde = {Re_c:.3e}")
        print(f"  Mach     = {Mach:.4f}")
        print(f"  ΔCp_crit = {deltaCP_crit:.2f}  {extrap_note}")
 
    all_results = {}
 
    for profil_name, params in PROFILS.items():
 
        if verbose:
            print(f"\n{'='*55}")
            print(f"  Profil : {profil_name}")
            print(f"{'='*55}")
            print(f"  {'':>4} | {'profil':>8} | {'alpha':>8} | {'CL':>9} | {'CM':>12} | {'ΔCP':>9}")
            print(f"  {'-'*65}")
 
        panels = geometryGenerator.GenerateNACA4digit(
            maxCamber           = params['maxCamber'],
            positionOfMaxCamber = params['positionOfMaxCamber'],
            thickness           = params['thickness'],
            pointsPerSurface    = POINTS_PER_SURFACE
        )
 
        alpha_list    = []
        CL_list       = []
        CD_list       = []
        CM_list       = []
        deltaCP_list  = []
        Ue_upper_dict = {}
        Ue_lower_dict = {}
 
        for alpha in alpha_range:
            res = run_single_alpha(panels, alpha)
 
            alpha_list.append(alpha)
            CL_list.append(res['CL'])
            CD_list.append(res['CD'])
            CM_list.append(res['CM'])
            deltaCP_list.append(res['deltaCP'])
            Ue_upper_dict[alpha] = (res['x_upper'], res['Ue_upper'])
            Ue_lower_dict[alpha] = (res['x_lower'], res['Ue_lower'])
 
            if verbose:
                print(f"  Q2.1 | {profil_name:<8} | "
                      f"{alpha:>+7.1f}° | "
                      f"{res['CL']:>+9.4f} | "
                      f"{res['CM']:>+12.6f} | "
                      f"{res['deltaCP']:>9.4f}")
 
        alpha_arr   = np.array(alpha_list)
        CL_arr      = np.array(CL_list)
        deltaCP_arr = np.array(deltaCP_list)
 
        alphaMax, CLmax = find_clmax_valarezo(
            alpha_arr, deltaCP_arr, CL_arr, deltaCP_crit
        )
 
        if verbose:
            print(f"\n  Q2.4 | {profil_name} | "
                  f"Re_c = {Re_c:.3e} | Mach = {Mach:.4f} | "
                  f"ΔCp_crit = {deltaCP_crit:.2f}")
            if CLmax is not None:
                print(f"  Q2.4 | {profil_name} | "
                      f"CLmax (Valarezo) = {CLmax:.4f}  "
                      f"alphaMax = {alphaMax:.2f}°")
 
        all_results[profil_name] = {
            'alpha'       : alpha_arr,
            'CL'          : CL_arr,
            'CD'          : np.array(CD_list),
            'CM'          : np.array(CM_list),
            'deltaCP'     : deltaCP_arr,
            'CLmax'       : CLmax,
            'alphaMax'    : alphaMax,
            'deltaCP_crit': deltaCP_crit,
            'Re_c'        : Re_c,
            'Mach'        : Mach,
            'rho'         : rho,
            'nu'          : nu,
            'Ue_upper'    : Ue_upper_dict,
            'Ue_lower'    : Ue_lower_dict,
        }
 
    return all_results
 
 
# ─────────────────────────────────────────────────────────────────
# SAUVEGARDE / CHARGEMENT .dat
# ─────────────────────────────────────────────────────────────────
 
def save_results_dat(all_results, prefix='hspm'):
    """
    Sauvegarde les résultats HSPM dans des fichiers .dat, un par profil.
 
    Fichiers générés par profil (ex. NACA0012) :
      hspm_NACA0012.dat          → alpha, CL, CD, CM, deltaCP  (+ métadonnées en en-tête)
      hspm_NACA0012_Ue_upper.dat → x/c et Ue extrados pour chaque alpha
      hspm_NACA0012_Ue_lower.dat → x/c et Ue intrados pour chaque alpha
 
    Format Ue_*.dat :
      Ligne 1 (en-tête) : nombre d'alphas et nombre de points par surface
      Blocs séparés par une ligne vide : un bloc = un alpha
      Chaque bloc : première colonne = x/c, deuxième = Ue/V_inf
    """
    for pname, res in all_results.items():
 
        # ── Fichier scalaires ─────────────────────────────────────
        fname = f"{prefix}_{pname}.dat"
        meta  = (f"Re_c={res['Re_c']:.6e}  Mach={res['Mach']:.6f}  "
                 f"rho={res['rho']:.6f}  nu={res['nu']:.6e}  "
                 f"CLmax={res['CLmax'] if res['CLmax'] is not None else float('nan'):.6f}  "
                 f"alphaMax={res['alphaMax'] if res['alphaMax'] is not None else float('nan'):.6f}  "
                 f"deltaCP_crit={res['deltaCP_crit']:.6f}")
        header = meta + "\nalpha          CL              CD              CM              deltaCP"
        data   = np.column_stack([res['alpha'], res['CL'],
                                  res['CD'],    res['CM'], res['deltaCP']])
        np.savetxt(fname, data, header=header, fmt='%+14.8f', comments='# ')
        print(f"  -> HSPM sauvegardé        : {fname}")
 
        # ── Fichiers Ue extrados et intrados ─────────────────────
        for surface in ('upper', 'lower'):
            ue_dict = res[f'Ue_{surface}']
            fname_ue = f"{prefix}_{pname}_Ue_{surface}.dat"
 
            alpha_list = res['alpha']
            n_alpha    = len(alpha_list)
            # Tous les vecteurs ont la même longueur (même géométrie)
            n_pts = len(ue_dict[alpha_list[0]][0])
 
            with open(fname_ue, 'w') as f:
                f.write(f"# n_alpha={n_alpha}  n_pts={n_pts}\n")
                f.write(f"# Format : blocs séparés par une ligne vide\n")
                f.write(f"# Chaque bloc : alpha / x_c / Ue\n")
                for alpha in alpha_list:
                    x_arr, Ue_arr = ue_dict[alpha]
                    f.write(f"# alpha={alpha:+.1f}\n")
                    for xi, ui in zip(x_arr, Ue_arr):
                        f.write(f"{xi:+14.8f}  {ui:+14.8f}\n")
                    f.write("\n")
 
            print(f"  -> Ue_{surface} sauvegardé       : {fname_ue}")
 
 
 
# ─────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    results = run_all_profiles(verbose=True)
    save_results_dat(results)