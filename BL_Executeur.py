import numpy as np
import os
 
# ─────────────────────────────────────────────────────────────────
# PARAMÈTRES DE LA COUCHE LIMITE
# ─────────────────────────────────────────────────────────────────
 
N_LAM_INIT     = 4       # Panneaux laminaires avant transition forcée
H_SEP_TURB     = 3.0     # Critère de séparation turbulente : H >= 3.0
LAMBDA_SEP_LAM = -0.09   # Critère de séparation laminaire : lambda <= -0.09
CF_SEP_TURB    = 0.0     # Critère de séparation turbulente : cf <= 0
H_TRANS        = 1.4     # Facteur de forme imposé à la transition lam -> turb
MICHEL_A       = 2.9     # Critère de Michel : Re_theta > A * Re_x^B
MICHEL_B       = 0.4
 
 
# ─────────────────────────────────────────────────────────────────
# SAUVEGARDE / CHARGEMENT bl_*.dat
# ─────────────────────────────────────────────────────────────────
 
def save_bl_dat(bl_results, prefix='bl'):
    """
    Sauvegarde les résultats de couche limite dans des fichiers .dat, un par profil.
    NaN dans x_sep / x_trans = pas de séparation / pas de transition sur cette surface.
 
    Colonnes : alpha, CD_visc, CD_upper, CD_lower,
               x_sep_upper, x_sep_lower,
               x_trans_upper, x_trans_lower,
               theta_te_upper, theta_te_lower, H_te_upper, H_te_lower
    """
    for pname, res in bl_results.items():
        fname  = f"{prefix}_{pname}.dat"
        header = ("alpha          CD_visc         CD_upper        CD_lower        "
                  "x_sep_upper     x_sep_lower     "
                  "x_trans_upper   x_trans_lower   "
                  "theta_te_up     theta_te_lo     H_te_up         H_te_lo")
        data = np.column_stack([
            res['alpha'],
            res['CD_visc'],        res['CD_upper'],      res['CD_lower'],
            res['x_sep_upper'],    res['x_sep_lower'],
            res['x_trans_upper'],  res['x_trans_lower'],
            res['theta_te_upper'], res['theta_te_lower'],
            res['H_te_upper'],     res['H_te_lower'],
        ])
        np.savetxt(fname, data, header=header, fmt='%+14.8f', comments='# ')
        print(f"  -> BL sauvegardé   : {fname}")
 
def load_hspm_dat(profil_names, prefix='hspm'):
    """
    Charge les résultats HSPM depuis les fichiers .dat (scalaires + Ue).
    Retourne un dict identique à celui de run_all_profiles()
    (incluant Ue_upper et Ue_lower).
    Retourne None si un fichier est manquant.
    """
    results = {}
 
    for pname in profil_names:
        fname    = f"{prefix}_{pname}.dat"
        fname_up = f"{prefix}_{pname}_Ue_upper.dat"
        fname_lo = f"{prefix}_{pname}_Ue_lower.dat"
 
        for f in (fname, fname_up, fname_lo):
            if not os.path.exists(f):
                print(f"  Fichier manquant : {f}")
                return None
 
        # ── Scalaires ─────────────────────────────────────────────
        with open(fname, 'r') as f:
            line1 = f.readline().strip().lstrip('#').strip()
 
        meta = {}
        for token in line1.split():
            if '=' in token:
                k, v = token.split('=')
                try:
                    meta[k] = float(v)
                except ValueError:
                    pass
 
        data     = np.loadtxt(fname, comments='#')
        CLmax    = meta.get('CLmax',    float('nan'))
        alphaMax = meta.get('alphaMax', float('nan'))
 
        res = {
            'alpha'       : data[:, 0],
            'CL'          : data[:, 1],
            'CD'          : data[:, 2],
            'CM'          : data[:, 3],
            'deltaCP'     : data[:, 4],
            'Re_c'        : meta.get('Re_c',  0.0),
            'Mach'        : meta.get('Mach',  0.0),
            'rho'         : meta.get('rho',   0.0),
            'nu'          : meta.get('nu',    0.0),
            'CLmax'       : None if np.isnan(CLmax)    else CLmax,
            'alphaMax'    : None if np.isnan(alphaMax) else alphaMax,
            'deltaCP_crit': meta.get('deltaCP_crit', 0.0),
        }
 
        # ── Ue extrados et intrados ───────────────────────────────
        for surface, fname_ue in (('upper', fname_up), ('lower', fname_lo)):
            ue_dict       = {}
            current_alpha = None
            x_buf, Ue_buf = [], []
 
            with open(fname_ue, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('# alpha='):
                        if current_alpha is not None and x_buf:
                            ue_dict[current_alpha] = (
                                np.array(x_buf), np.array(Ue_buf)
                            )
                        current_alpha = float(line.split('=')[1])
                        x_buf, Ue_buf = [], []
                    elif line.startswith('#') or line == '':
                        continue
                    else:
                        vals = line.split()
                        if len(vals) == 2:
                            x_buf.append(float(vals[0]))
                            Ue_buf.append(float(vals[1]))
 
            if current_alpha is not None and x_buf:
                ue_dict[current_alpha] = (np.array(x_buf), np.array(Ue_buf))
 
            res[f'Ue_{surface}'] = ue_dict
 
        results[pname] = res
 
    return results
 
 
# ─────────────────────────────────────────────────────────────────
# MÉTHODE DE THWAITES (LAMINAIRE)
# ─────────────────────────────────────────────────────────────────
 
def thwaites_l(lam):
    return 0.45 - 6.0 * lam
 
 
def thwaites_H(lam):
    if lam < 0.0:
        return 2.088 + 0.0731 / (lam + 0.05)
    else:
        return 2.61 - 3.75 * lam + 5.24 * lam**2
 
 
def solve_thwaites(x_arr, Ue_arr, nu):
    n         = len(x_arr)
    theta_arr = np.zeros(n)
    H_arr     = np.full(n, 2.591)
    sep_idx   = n - 1
 
    for i in range(1, n):
        dx  = x_arr[i] - x_arr[i-1]
        Ue0 = max(Ue_arr[i-1], 1e-12)
        Ue1 = max(Ue_arr[i],   1e-12)
 
        increment    = 0.45 * nu * 0.5 * (Ue0**5 + Ue1**5) * dx
        theta2_Ue6   = theta_arr[i-1]**2 * Ue0**6 + increment
        theta_arr[i] = np.sqrt(max(theta2_Ue6, 0.0) / Ue1**6)
 
        dUe_dx   = (Ue1 - Ue0) / max(dx, 1e-30)
        lam      = np.clip((theta_arr[i]**2 / nu) * dUe_dx, -0.5, 0.5)
        H_arr[i] = thwaites_H(lam)
 
        if lam <= LAMBDA_SEP_LAM:
            sep_idx = i
            break
 
    return theta_arr, H_arr, sep_idx
 
 
# ─────────────────────────────────────────────────────────────────
# MÉTHODE TURBULENTE INTÉGRALE
# ─────────────────────────────────────────────────────────────────
 
def solve_turbulent(x_arr, Ue_arr, nu, theta0, H0, i_start):
    n_turb       = len(x_arr) - i_start
    theta_arr    = np.zeros(n_turb)
    H_arr        = np.zeros(n_turb)
    cf_arr       = np.zeros(n_turb)
    theta_arr[0] = theta0
    H_arr[0]     = H0
    sep_local    = n_turb - 1
 
    for k in range(1, n_turb):
        i      = i_start + k
        dx     = x_arr[i] - x_arr[i-1]
        Ue     = max(Ue_arr[i-1], 1e-12)
        dUe_dx = (Ue_arr[i] - Ue_arr[i-1]) / max(dx, 1e-30)
        th     = theta_arr[k-1]
        H      = H_arr[k-1]
 
        Re_theta    = max(Ue * th / nu, 1.0)
        cf          = 0.246 * (10.0**(-0.678 * H)) * Re_theta**(-0.268)
        cf_arr[k-1] = cf
 
        dtheta_dx = cf/2.0 - (H+2.0)*(th/Ue)*dUe_dx
        A1 = -H*(H-1.0)*(3.0*H-1.0)*(th/Ue)*dUe_dx
        A2 =  H*(3.0*H-1.0)*cf/2.0
        A3 = -(3.0*H-1.0)**2 * 0.0056/2.0 * Re_theta**(-1.0/6.0)
        dH_dx = (A1 + A2 + A3) / max(th, 1e-30)
 
        theta_arr[k] = max(th + dtheta_dx*dx, 1e-10)
        H_arr[k]     = max(H  + dH_dx    *dx, 1.0001)
 
        Re_new    = max(Ue_arr[i]*theta_arr[k]/nu, 1.0)
        cf_arr[k] = 0.246*(10.0**(-0.678*H_arr[k]))*Re_new**(-0.268)
 
        if H_arr[k] >= H_SEP_TURB or cf_arr[k] <= CF_SEP_TURB:
            sep_local = k
            break
 
    return theta_arr, H_arr, cf_arr, i_start + sep_local
 
 
# ─────────────────────────────────────────────────────────────────
# CRITÈRE DE TRANSITION (MICHEL)
# ─────────────────────────────────────────────────────────────────
 
def michel_transition(x_arr, Ue_arr, theta_arr, nu):
    for i in range(1, len(x_arr)):
        Ue       = max(Ue_arr[i], 1e-12)
        Re_theta = Ue * theta_arr[i] / nu
        Re_x     = Ue * x_arr[i]    / nu
        if Re_x < 1.0:
            continue
        if Re_theta > MICHEL_A * Re_x**MICHEL_B:
            return i
    return -1
 
 
# ─────────────────────────────────────────────────────────────────
# SQUIRE-YOUNG (CD visqueux)
# ─────────────────────────────────────────────────────────────────
 
def squire_young(theta_te, H_te, Ue_te):
    return 2.0 * theta_te * max(Ue_te, 1e-12)**((max(H_te, 1.0001) + 5.0) / 2.0)
 
 
# ─────────────────────────────────────────────────────────────────
# RÉSOLUTION SUR UNE SURFACE
# ─────────────────────────────────────────────────────────────────
 
def solve_surface(x_arr, Ue_arr, nu):
    n = len(x_arr)
 
    theta_lam, H_lam, sep_lam = solve_thwaites(x_arr, Ue_arr, nu)
 
    # Séparation laminaire avant transition (x_trans = NaN car pas de transition)
    if sep_lam < n - 1:
        CD = squire_young(theta_lam[sep_lam], H_lam[sep_lam], Ue_arr[sep_lam])
        return {'CD_visc': CD, 'x_sep': float(x_arr[sep_lam]),
                'x_trans': np.nan,
                'theta_te': theta_lam[sep_lam], 'H_te': H_lam[sep_lam]}
 
    # Transition laminaire -> turbulente
    i_trans = michel_transition(x_arr, Ue_arr, theta_lam, nu)
    if i_trans < 0:
        i_trans = min(N_LAM_INIT, n - 2)
 
    theta_turb, H_turb, _, sep_global = solve_turbulent(
        x_arr, Ue_arr, nu, theta_lam[i_trans], H_TRANS, i_trans
    )
 
    k_te     = min(sep_global - i_trans, len(theta_turb) - 1)
    theta_te = theta_turb[k_te]
    H_te     = H_turb[k_te]
    Ue_te    = Ue_arr[sep_global]
    x_sep    = float(x_arr[sep_global]) if sep_global < n - 1 else None
    x_trans  = float(x_arr[i_trans])
 
    CD = squire_young(theta_te, H_te, Ue_te)
    return {'CD_visc': CD, 'x_sep': x_sep, 'x_trans': x_trans,
            'theta_te': theta_te, 'H_te': H_te}
 
 
# ─────────────────────────────────────────────────────────────────
# INTERFACE PRINCIPALE
# ─────────────────────────────────────────────────────────────────
 
def run_bl_all_alphas(hspm_results, V_inf, chord, nu, verbose=True):
    """
    Lance le calcul de couche limite pour tous les profils et tous les alphas
    issus de HSPM_Executeur.run_all_profiles() ou load_results_dat().
 
    Paramètres
    ----------
    hspm_results : dict retourné par run_all_profiles() ou load_results_dat()
    V_inf        : vitesse à l'infini [m/s]
    chord        : corde [m]
    nu           : viscosité cinématique réelle de l'air [m²/s]
    verbose      : affichage terminal
 
    Retourne
    --------
    bl_results[profil_name] = {
        'alpha', 'CD_visc', 'CD_upper', 'CD_lower',
        'x_sep_upper', 'x_sep_lower',
        'x_trans_upper', 'x_trans_lower',
        'theta_te_upper', 'theta_te_lower', 'H_te_upper', 'H_te_lower'
    }
    """
    nu_adim    = nu / (V_inf * chord)
    bl_results = {}
 
    for pname, res in hspm_results.items():
        alpha_arr = res['alpha']
 
        CD_visc_arr  = []
        CD_up_arr    = []
        CD_lo_arr    = []
        x_sep_up     = []
        x_sep_lo     = []
        x_trans_up   = []
        x_trans_lo   = []
        th_te_up     = []
        th_te_lo     = []
        H_te_up      = []
        H_te_lo      = []
 
        if verbose:
            print(f"\n{'='*55}")
            print(f"  Couche limite : {pname}")
            print(f"{'='*55}")
            print(f"  {'':>4} | {'profil':>8} | {'alpha':>8} | "
                  f"{'CD_visc':>10} | {'x_trans_up':>10} | {'x_trans_lo':>10} | "
                  f"{'x_sep_up':>10} | {'x_sep_lo':>10}")
            print(f"  {'-'*80}")
 
        for alpha in alpha_arr:
            x_up, Ue_up = res['Ue_upper'][alpha]
            x_lo, Ue_lo = res['Ue_lower'][alpha]
 
            r_up = solve_surface(np.array(x_up), np.array(Ue_up), nu_adim)
            r_lo = solve_surface(np.array(x_lo), np.array(Ue_lo), nu_adim)
 
            CD_tot = r_up['CD_visc'] + r_lo['CD_visc']
            xs_u   = r_up['x_sep'] if r_up['x_sep'] is not None else np.nan
            xs_l   = r_lo['x_sep'] if r_lo['x_sep'] is not None else np.nan
 
            CD_visc_arr.append(CD_tot)
            CD_up_arr.append(r_up['CD_visc'])
            CD_lo_arr.append(r_lo['CD_visc'])
            x_sep_up.append(xs_u)
            x_sep_lo.append(xs_l)
            x_trans_up.append(r_up['x_trans'])
            x_trans_lo.append(r_lo['x_trans'])
            th_te_up.append(r_up['theta_te'])
            th_te_lo.append(r_lo['theta_te'])
            H_te_up.append(r_up['H_te'])
            H_te_lo.append(r_lo['H_te'])
 
            if verbose:
                su = f"{xs_u:.4f}" if not np.isnan(xs_u) else "  —    "
                sl = f"{xs_l:.4f}" if not np.isnan(xs_l) else "  —    "
                tu = f"{r_up['x_trans']:.4f}" if not np.isnan(r_up['x_trans']) else "  —    "
                tl = f"{r_lo['x_trans']:.4f}" if not np.isnan(r_lo['x_trans']) else "  —    "
                print(f"  Q2.1 | {pname:>8} | {alpha:>+7.1f}° | "
                      f"{CD_tot:>10.5f} | {tu:>10} | {tl:>10} | "
                      f"{su:>10} | {sl:>10}")
 
        bl_results[pname] = {
            'alpha'         : alpha_arr,
            'CD_visc'       : np.array(CD_visc_arr),
            'CD_upper'      : np.array(CD_up_arr),
            'CD_lower'      : np.array(CD_lo_arr),
            'x_sep_upper'   : np.array(x_sep_up),
            'x_sep_lower'   : np.array(x_sep_lo),
            'x_trans_upper' : np.array(x_trans_up),
            'x_trans_lower' : np.array(x_trans_lo),
            'theta_te_upper': np.array(th_te_up),
            'theta_te_lower': np.array(th_te_lo),
            'H_te_upper'    : np.array(H_te_up),
            'H_te_lower'    : np.array(H_te_lo),
        }
 
    return bl_results
 
 
# ─────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE (exécution autonome)
# ─────────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import HSPM_Executeur as hspm_exec
 
    PROFIL_NAMES = list(hspm_exec.PROFILS.keys())
 
    # Vérification des fichiers d'entrée (hspm_*.dat avec Ue)
    hspm_dats_ok = all(
        os.path.exists(f'hspm_{p}.dat') and
        os.path.exists(f'hspm_{p}_Ue_upper.dat') and
        os.path.exists(f'hspm_{p}_Ue_lower.dat')
        for p in PROFIL_NAMES
    )
 
    if not hspm_dats_ok:
        print("\nERREUR : fichiers hspm_*.dat introuvables.")
        print("Veuillez d'abord exécuter HSPM_Executeur.py pour générer les données.")
        raise SystemExit(1)
 
    print("\nChargement des données HSPM depuis les fichiers .dat :")
    hspm_res = load_hspm_dat(PROFIL_NAMES)
 
    nu    = hspm_res[PROFIL_NAMES[0]]['nu']
    V_inf = hspm_exec.V_INF_MS
    chord = hspm_exec.CHORD_M
 
    print("\nCalcul de la couche limite :")
    bl_res = run_bl_all_alphas(hspm_res, V_inf, chord, nu, verbose=True)
    save_bl_dat(bl_res)