import numpy as np
import os
 
# ─────────────────────────────────────────────────────────────────
# semiempirique_Executeur.py
# Calcul de la traînée par formules semi-empiriques de plaque plane
# pour comparaison avec la méthode de couche limite (BL_Executeur)
#
# Formules utilisées (notes de cours AER8270) :
#   Laminaire  : CDF_lam  = 1.328 / sqrt(Re_x)
#   Turbulent  : CDF_turb = 0.455 / (log10(Re_x))^2.58
#   Transition : CDF = CDF_turb|c - (x_trans/c) * (CDF_turb|x_trans - CDF_lam|x_trans)
#   Facteur K  : CD0 = K * CDF  (K = 1.2 pour profil d'épaisseur moyenne)
#
# Le point de transition x_trans est lu depuis bl_*.dat (calculé par Michel dans BL_Executeur).
# Le calcul est fait séparément pour l'extrados et l'intrados, puis sommé.
# En 2D, le CD semi-empirique est directement K * (CDF_upper + CDF_lower).
#
# Fichiers .dat requis en entrée :
#   hspm_*.dat   (pour Re_c, alpha, CL)
#   bl_*.dat     (pour x_trans_upper, x_trans_lower par alpha)
#
# Fichiers .dat générés :
#   semi_NACA0012.dat, semi_NACA4412.dat
# ─────────────────────────────────────────────────────────────────
 
 
# ─────────────────────────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────────────────────────
 
K_FORME = 1.2    # Facteur de forme (Kw = 1.2 pour profil d'épaisseur moyenne)
 
 
# ─────────────────────────────────────────────────────────────────
# FORMULES SEMI-EMPIRIQUES
# ─────────────────────────────────────────────────────────────────
 
def CDF_lam(Re_x):
    """Traînée laminaire plaque plane (Blasius) : 1.328 / sqrt(Re_x)"""
    return 1.328 / np.sqrt(max(Re_x, 1.0))
 
 
def CDF_turb(Re_x):
    """Traînée turbulente plaque plane : 0.455 / (log10(Re_x))^2.58"""
    return 0.455 / (np.log10(max(Re_x, 1.0)))**2.58
 
 
def CDF_transition(Re_c, x_trans_sur_c):
    """
    Traînée de plaque plane avec transition laminaire -> turbulente,
    avec facteur de forme K appliqué.
 
    CD0 = K * [CDF_turb|c - (x_trans/c) * (CDF_turb|x_trans - CDF_lam|x_trans)]
 
    Paramètres
    ----------
    Re_c          : Reynolds basé sur la corde complète
    x_trans_sur_c : position de transition x_trans/c (entre 0 et 1)
                    NaN → écoulement entièrement turbulent (pas de transition trouvée)
 
    Retourne
    --------
    CD_surface : coefficient de traînée pour une surface (extrados OU intrados)
    """
    # Entièrement turbulent si pas de transition détectée
    if np.isnan(x_trans_sur_c):
        return K_FORME * CDF_turb(Re_c)
 
    # Reynolds au point de transition
    Re_trans = x_trans_sur_c * Re_c
 
    # Formule de transition avec facteur de forme
    CD = K_FORME * (CDF_turb(Re_c) - x_trans_sur_c * (CDF_turb(Re_trans) - CDF_lam(Re_trans)))
 
    return CD
 
 
# ─────────────────────────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────────────────────────
 
def save_semi_dat(semi_results, prefix='semi'):
    """
    Sauvegarde les résultats semi-empiriques dans des fichiers .dat, un par profil.
 
    Colonnes : alpha, CD_semi, CD_semi_upper, CD_semi_lower
    """
    for pname, res in semi_results.items():
        fname  = f"{prefix}_{pname}.dat"
        header = "alpha          CD_semi         CD_semi_upper   CD_semi_lower"
        data   = np.column_stack([
            res['alpha'],
            res['CD_semi'],
            res['CD_semi_upper'],
            res['CD_semi_lower'],
        ])
        np.savetxt(fname, data, header=header, fmt='%+14.8f', comments='# ')
        print(f"  -> Semi-empirique sauvegardé : {fname}")
 
 
# ─────────────────────────────────────────────────────────────────
# CHARGEMENT DES ENTRÉES
# ─────────────────────────────────────────────────────────────────
 
def load_hspm_scalaires(profil_names, prefix='hspm'):
    """
    Charge uniquement les scalaires HSPM (alpha, CL, Re_c, nu) depuis hspm_*.dat.
    Retourne None si un fichier est manquant.
    """
    results = {}
    for pname in profil_names:
        fname = f"{prefix}_{pname}.dat"
        if not os.path.exists(fname):
            print(f"  Fichier manquant : {fname}")
            return None
 
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
 
        data = np.loadtxt(fname, comments='#')
        results[pname] = {
            'alpha': data[:, 0],
            'CL'   : data[:, 1],
            'Re_c' : meta.get('Re_c', 0.0),
            'nu'   : meta.get('nu',   0.0),
        }
 
    return results
 
 
def load_bl_xtrans(profil_names, prefix='bl'):
    """
    Charge uniquement x_trans_upper et x_trans_lower depuis bl_*.dat.
    Retourne None si un fichier est manquant.
 
    Format attendu des colonnes bl_*.dat :
      0:alpha  1:CD_visc  2:CD_upper  3:CD_lower
      4:x_sep_upper  5:x_sep_lower
      6:x_trans_upper  7:x_trans_lower
      8:theta_te_up  9:theta_te_lo  10:H_te_up  11:H_te_lo
    """
    results = {}
    for pname in profil_names:
        fname = f"{prefix}_{pname}.dat"
        if not os.path.exists(fname):
            print(f"  Fichier manquant : {fname}")
            return None
 
        data = np.loadtxt(fname, comments='#')
        results[pname] = {
            'alpha'        : data[:, 0],
            'x_trans_upper': data[:, 6],
            'x_trans_lower': data[:, 7],
        }
 
    return results
 
 
# ─────────────────────────────────────────────────────────────────
# CALCUL PRINCIPAL
# ─────────────────────────────────────────────────────────────────
 
def run_semiempirique(hspm_scalaires, bl_xtrans, verbose=True):
    """
    Calcule le CD semi-empirique pour chaque profil et chaque alpha.
 
    Le CD total est la somme des contributions extrados et intrados
    (deux surfaces mouillées).
 
    Paramètres
    ----------
    hspm_scalaires : dict chargé par load_hspm_scalaires()
    bl_xtrans      : dict chargé par load_bl_xtrans()
    verbose        : affichage terminal
 
    Retourne
    --------
    semi_results[profil_name] = {
        'alpha', 'CD_semi', 'CD_semi_upper', 'CD_semi_lower'
    }
    """
    semi_results = {}
 
    for pname, hspm in hspm_scalaires.items():
        alpha_arr = hspm['alpha']
        Re_c      = hspm['Re_c']
        bl        = bl_xtrans[pname]
 
        CD_semi_arr    = []
        CD_semi_up_arr = []
        CD_semi_lo_arr = []
 
        if verbose:
            print(f"\n{'='*55}")
            print(f"  Semi-empirique : {pname}  (Re_c = {Re_c:.3e})")
            print(f"{'='*55}")
            print(f"  {'tag':<4} | {'profil':<8} | {'alpha':>8} | "
                  f"{'CD_semi':>9} | {'x_trans_up':>12} | {'x_trans_lo':>12}")
            print(f"  {'-'*70}")
 
        for i, alpha in enumerate(alpha_arr):
            x_tr_up = bl['x_trans_upper'][i]
            x_tr_lo = bl['x_trans_lower'][i]
 
            CD_up  = CDF_transition(Re_c, x_tr_up)
            CD_lo  = CDF_transition(Re_c, x_tr_lo)
            CD_tot = CD_up + CD_lo
 
            CD_semi_arr.append(CD_tot)
            CD_semi_up_arr.append(CD_up)
            CD_semi_lo_arr.append(CD_lo)
 
            if verbose:
                tu = f"{x_tr_up:.4f}" if not np.isnan(x_tr_up) else "  —    "
                tl = f"{x_tr_lo:.4f}" if not np.isnan(x_tr_lo) else "  —    "
                print(f"  Q2.2 | {pname:<8} | {alpha:>+7.1f}° | "
                      f"{CD_tot:>9.5f} | {tu:>12} | {tl:>12}")
 
        semi_results[pname] = {
            'alpha'         : alpha_arr,
            'CD_semi'       : np.array(CD_semi_arr),
            'CD_semi_upper' : np.array(CD_semi_up_arr),
            'CD_semi_lower' : np.array(CD_semi_lo_arr),
        }
 
    return semi_results
 
 
# ─────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE (exécution autonome)
# ─────────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import HSPM_Executeur as hspm_exec
 
    PROFIL_NAMES = list(hspm_exec.PROFILS.keys())
 
    # Vérification des fichiers d'entrée
    hspm_ok = all(os.path.exists(f'hspm_{p}.dat') for p in PROFIL_NAMES)
    bl_ok   = all(os.path.exists(f'bl_{p}.dat')   for p in PROFIL_NAMES)
 
    if not hspm_ok:
        print("\nERREUR : fichiers hspm_*.dat introuvables.")
        print("Veuillez d'abord exécuter HSPM_Executeur.py.")
        raise SystemExit(1)
 
    if not bl_ok:
        print("\nERREUR : fichiers bl_*.dat introuvables.")
        print("Veuillez d'abord exécuter BL_Executeur.py.")
        raise SystemExit(1)
 
    print("\nChargement des données...")
    hspm_scal = load_hspm_scalaires(PROFIL_NAMES)
    bl_xtr    = load_bl_xtrans(PROFIL_NAMES)
 
    print("\nCalcul semi-empirique :")
    semi_res = run_semiempirique(hspm_scal, bl_xtr, verbose=True)
    save_semi_dat(semi_res)