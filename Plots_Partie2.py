import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import os
 
matplotlib.rcParams['figure.dpi'] = 800
 
# ─────────────────────────────────────────────────────────────────
# plots_partie2.py
# Graphiques de la Partie 2 du projet AER8270
#
# Fichiers .dat requis en entrée :
#   hspm_*.dat   (HSPM_Executeur.py)
#   bl_*.dat     (BL_Executeur.py)
#   semi_*.dat   (semiempirique_Executeur.py)
#
# Graphiques générés :
#   Q2.1  | CL-alpha         NACA0012
#   Q2.1  | CL-CD            NACA0012
#   Q2.1  | CL-alpha         NACA4412
#   Q2.1  | CL-CD            NACA4412
#   Q2.2  | CD-alpha + semi  NACA0012
#   Q2.2  | CD-alpha + semi  NACA4412
#   Q2.3  | x_sep-alpha      NACA0012 (extrados + intrados)
#   Q2.3  | x_sep-CL         NACA0012 (extrados + intrados)
#   Q2.3  | x_sep-alpha      NACA4412 (extrados + intrados)
#   Q2.3  | x_sep-CL         NACA4412 (extrados + intrados)
#   Q2.4  | D-Vinf            NACA0012 (11 alphas)
#   Q2.4  | L-Vinf            NACA0012 (11 alphas)
#   Q2.4  | D-Vinf            NACA4412 (11 alphas)
#   Q2.4  | L-Vinf            NACA4412 (11 alphas)
# ─────────────────────────────────────────────────────────────────
 
sys.path.insert(0, os.path.dirname(__file__))
import HSPM_Executeur as hspm_exec
import BL_Executeur   as bl_exec
 
PROFIL_NAMES = list(hspm_exec.PROFILS.keys())
 
# ─────────────────────────────────────────────────────────────────
# PARAMÈTRES PHYSIQUES
# ─────────────────────────────────────────────────────────────────
 
CHORD_M  = hspm_exec.CHORD_M    # corde [m]
V_INF_MS = hspm_exec.V_INF_MS   # vitesse de référence [m/s]
V_RANGE  = np.linspace(0.01, 12.0, 200)   # plage de vitesse pour L/D vs Vinf
 
# Alphas affichés sur les graphiques L/D vs Vinf (11 courbes, pas de 4°)
ALPHAS_LD = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20]
 
 
# ─────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────
 
def load_bl_dat(profil_names, prefix='bl'):
    results = {}
    for pname in profil_names:
        fname = f"{prefix}_{pname}.dat"
        if not os.path.exists(fname):
            print(f"  Fichier manquant : {fname}")
            return None
        data = np.loadtxt(fname, comments='#')
        results[pname] = {
            'alpha'         : data[:, 0],
            'CD_visc'       : data[:, 1],
            'CD_upper'      : data[:, 2],
            'CD_lower'      : data[:, 3],
            'x_sep_upper'   : data[:, 4],
            'x_sep_lower'   : data[:, 5],
            'x_trans_upper' : data[:, 6],
            'x_trans_lower' : data[:, 7],
            'theta_te_upper': data[:, 8],
            'theta_te_lower': data[:, 9],
            'H_te_upper'    : data[:, 10],
            'H_te_lower'    : data[:, 11],
        }
    return results
 
 
def load_semi_dat(profil_names, prefix='semi'):
    results = {}
    for pname in profil_names:
        fname = f"{prefix}_{pname}.dat"
        if not os.path.exists(fname):
            print(f"  Fichier manquant : {fname}")
            return None
        data = np.loadtxt(fname, comments='#')
        results[pname] = {
            'alpha'        : data[:, 0],
            'CD_semi'      : data[:, 1],
            'CD_semi_upper': data[:, 2],
            'CD_semi_lower': data[:, 3],
        }
    return results
 
 
# Vérification des fichiers d'entrée
for label, pattern in [('hspm', 'hspm_{p}.dat'), ('bl', 'bl_{p}.dat'),
                        ('semi', 'semi_{p}.dat')]:
    if not all(os.path.exists(pattern.format(p=p)) for p in PROFIL_NAMES):
        executeur = {'hspm': 'HSPM_Executeur',
                     'bl':   'BL_Executeur',
                     'semi': 'semiempirique_Executeur'}[label]
        print(f"\nERREUR : fichiers {label}_*.dat introuvables.")
        print(f"Veuillez d'abord exécuter {executeur}.py.")
        raise SystemExit(1)
 
print("\nChargement des données depuis les fichiers .dat...")
hspm_res = bl_exec.load_hspm_dat(PROFIL_NAMES)
bl_res   = load_bl_dat(PROFIL_NAMES)
semi_res = load_semi_dat(PROFIL_NAMES)
 
rho = list(hspm_res.values())[0]['rho']
 
 
# ─────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────
 
COL_UPPER = '#1f77b4'   # bleu  → extrados
COL_LOWER = '#d62728'   # rouge → intrados
COL_SEMI  = '#2ca02c'   # vert  → semi-empirique
 
STYLE = {
    'NACA0012': {'color': '#1f77b4', 'ls': '-',  'marker': 'o', 'ms': 4},
    'NACA4412': {'color': '#d62728', 'ls': '--', 'marker': 's', 'ms': 4},
}
 
 
def apply_grid(ax):
    ax.minorticks_on()
    ax.grid(which='major', linewidth=0.8, alpha=0.7)
    ax.grid(which='minor', linewidth=0.4, alpha=0.4)
 
 
def add_minor_ticks(ax):
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
 
 
def save_and_show(fig, fname):
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
 
 
# ─────────────────────────────────────────────────────────────────
# Q2.1 — CL-alpha et CL-CD  (un graphique par profil)
# ─────────────────────────────────────────────────────────────────
 
print("\n" + "="*55)
print("  Q2.1 | CL-alpha et CL-CD par profil")
print("="*55)
 
for pname in PROFIL_NAMES:
    res = hspm_res[pname]
    bl  = bl_res[pname]
    st  = STYLE[pname]
 
    # CL vs alpha
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(res['alpha'], res['CL'],
            color=st['color'], ls=st['ls'],
            marker=st['marker'], ms=st['ms'], label=pname)
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.axvline(0, color='k', lw=0.6, ls=':')
    ax.set_xlabel(r'$\alpha$ [deg]', fontsize=12)
    ax.set_ylabel(r'$C_L$ [-]',      fontsize=12)
    ax.set_title(rf'$C_L$–$\alpha$ — {pname}', fontsize=11)
    ax.legend(fontsize=10)
    add_minor_ticks(ax)
    apply_grid(ax)
    save_and_show(fig, f'Q2_1_CL_alpha_{pname}.png')
 
    # CL vs CD
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot( bl['CD_visc'], res['CL'],
            color=st['color'], ls=st['ls'],
            marker=st['marker'], ms=st['ms'], label=pname)
    ax.set_xlabel(r'$C_D$ [-]', fontsize=12)
    ax.set_ylabel(r'$C_L$ [-]', fontsize=12)
    ax.set_title(rf'$C_L$–$C_D$ — {pname}', fontsize=11)
    ax.legend(fontsize=10)
    add_minor_ticks(ax)
    apply_grid(ax)
    save_and_show(fig, f'Q2_1_CL_CD_{pname}.png')
 
 
# ─────────────────────────────────────────────────────────────────
# Q2.2 — CD-alpha + semi-empirique  (un graphique par profil)
# ─────────────────────────────────────────────────────────────────
 
print("\n" + "="*55)
print("  Q2.2 | CD-alpha + semi-empirique par profil")
print("="*55)
 
for pname in PROFIL_NAMES:
    res  = hspm_res[pname]
    bl   = bl_res[pname]
    semi = semi_res[pname]
    st   = STYLE[pname]
 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(res['alpha'], bl['CD_visc'],
            color=st['color'], ls=st['ls'],
            marker=st['marker'], ms=st['ms'],
            label=f'{pname} (couche limite)')
    ax.plot(semi['alpha'], semi['CD_semi'],
            color=COL_SEMI, ls=':', marker='^', ms=4,
            label=f'{pname} (semi-empirique)')
    ax.axvline(0, color='k', lw=0.6, ls=':')
    ax.set_xlabel(r'$\alpha$ [deg]', fontsize=12)
    ax.set_ylabel(r'$C_D$ [-]',      fontsize=12)
    ax.set_title(rf'$C_D$–$\alpha$ — {pname}', fontsize=11)
    ax.legend(fontsize=10)
    add_minor_ticks(ax)
    apply_grid(ax)
    save_and_show(fig, f'Q2_2_CD_alpha_{pname}.png')
 
 
# ─────────────────────────────────────────────────────────────────
# Q2.3 — x_sep vs alpha et x_sep vs CL  (un graphique par profil)
# ─────────────────────────────────────────────────────────────────
 
print("\n" + "="*55)
print("  Q2.3 | x_sep-alpha et x_sep-CL par profil")
print("="*55)
 
for pname in PROFIL_NAMES:
    res = hspm_res[pname]
    bl  = bl_res[pname]
 
    alpha     = bl['alpha']
    CL        = res['CL']
    x_sep_up  = bl['x_sep_upper'].copy()
    x_sep_lo  = bl['x_sep_lower'].copy()
 
    # Pas de séparation → 100%
    x_sep_up_plot = np.where(np.isnan(x_sep_up), 1.0, x_sep_up) * 100
    x_sep_lo_plot = np.where(np.isnan(x_sep_lo), 1.0, x_sep_lo) * 100
 
    # x_sep vs alpha
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alpha, x_sep_up_plot,
            color=COL_UPPER, ls='-', marker='o', ms=4,
            label='Extrados', markevery=2)
    ax.plot(alpha, x_sep_lo_plot,
            color=COL_LOWER, ls='--', marker='s', ms=4,
            label='Intrados', markevery=2)
    ax.axvline(0, color='k', lw=0.6, ls=':')
    ax.set_xlabel(r'$\alpha$ [deg]',       fontsize=12)
    ax.set_ylabel(r'$x_{sep}/c$ [%]',      fontsize=12)
    ax.set_title(rf'$x_{{sep}}$–$\alpha$ — {pname}', fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    add_minor_ticks(ax)
    apply_grid(ax)
    save_and_show(fig, f'Q2_3_xsep_alpha_{pname}.png')
 
    # x_sep vs CL
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(CL, x_sep_up_plot,
            color=COL_UPPER, ls='-', marker='o', ms=4,
            label='Extrados', markevery=2)
    ax.plot(CL, x_sep_lo_plot,
            color=COL_LOWER, ls='--', marker='s', ms=4,
            label='Intrados', markevery=2)
    ax.set_xlabel(r'$C_L$ [-]',        fontsize=12)
    ax.set_ylabel(r'$x_{sep}/c$ [%]',  fontsize=12)
    ax.set_title(rf'$x_{{sep}}$–$C_L$ — {pname}', fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    add_minor_ticks(ax)
    apply_grid(ax)
    save_and_show(fig, f'Q2_3_xsep_CL_{pname}.png')
 
 
# ─────────────────────────────────────────────────────────────────
# Q2.4 — L et D vs V_inf  (11 alphas, un graphique par profil)
# ─────────────────────────────────────────────────────────────────
 
print("\n" + "="*55)
print("  Q2.4 | L et D vs V_inf par profil (11 alphas)")
print("="*55)
 
colors_ld = plt.cm.RdYlBu(np.linspace(0, 1, len(ALPHAS_LD)))
 
for pname in PROFIL_NAMES:
    res = hspm_res[pname]
    bl  = bl_res[pname]
 
    # Pour chaque alpha sélectionné, interpoler CL et CD
    CL_all    = res['CL']
    CD_all    = bl['CD_visc']
    alpha_all = res['alpha']
 
    fig_D, ax_D = plt.subplots(figsize=(7, 5))
    fig_L, ax_L = plt.subplots(figsize=(7, 5))
 
    for idx_a, alpha_target in enumerate(ALPHAS_LD):
        # Trouver l'indice exact (alpha est en entiers)
        idx = np.argmin(np.abs(alpha_all - alpha_target))
        CL_val = CL_all[idx]
        CD_val = CD_all[idx]
 
        # L et D en fonction de V_inf
        q      = 0.5 * rho * V_RANGE**2
        L_arr  = CL_val * q * CHORD_M   # [N/m d'envergure]
        D_arr  = CD_val * q * CHORD_M   # [N/m d'envergure]
 
        col   = colors_ld[idx_a]
        label = rf'$\alpha$ = {alpha_target:+d}°'
 
        ax_D.plot(V_RANGE, D_arr, color=col, label=label)
        ax_L.plot(V_RANGE, L_arr, color=col, label=label)
 
    for ax, ylabel, title_qty, fname_qty in [
        (ax_D, r'$D$ [N/m]', 'D', 'D'),
        (ax_L, r'$L$ [N/m]', 'L', 'L'),
    ]:
        ax.axhline(0, color='k', lw=0.6, ls=':')
        ax.set_xlabel(r'$V_\infty$ [m/s]', fontsize=12)
        ax.set_ylabel(ylabel,               fontsize=12)
        ax.set_title(rf'${title_qty}$–$V_\infty$ — {pname}', fontsize=11)
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        add_minor_ticks(ax)
        apply_grid(ax)
 
    save_and_show(fig_D, f'Q2_4_D_Vinf_{pname}.png')
    save_and_show(fig_L, f'Q2_4_L_Vinf_{pname}.png')
 
 
# ─────────────────────────────────────────────────────────────────
# IMPRESSION TERMINAL — résumé Q2.1
# ─────────────────────────────────────────────────────────────────
 
Re_c = list(hspm_res.values())[0]['Re_c']
 
print(f"\n{'='*65}")
print(f"  Q2.1 | RÉSULTATS NUMÉRIQUES — CL, CD (couche limite)")
print(f"  Re_c = {Re_c:.3e}   V_inf = {V_INF_MS:.2f} m/s   "
      f"chord = {CHORD_M*100:.2f} cm")
print(f"{'='*65}")
 
for pname in PROFIL_NAMES:
    res = hspm_res[pname]
    bl  = bl_res[pname]
    print(f"\n  Profil : {pname}")
    print(f"  {'alpha':>8} | {'CL':>9} | {'CD_visc':>10} | "
          f"{'x_sep_up':>10} | {'x_sep_lo':>10}")
    print(f"  {'-'*57}")
    for i, alpha in enumerate(res['alpha']):
        CL   = res['CL'][i]
        CD   = bl['CD_visc'][i]
        xs_u = bl['x_sep_upper'][i]
        xs_l = bl['x_sep_lower'][i]
        su   = f"{xs_u:.4f}" if not np.isnan(xs_u) else "  —    "
        sl   = f"{xs_l:.4f}" if not np.isnan(xs_l) else "  —    "
        print(f"  {alpha:>+8.1f} | {CL:>+9.4f} | {CD:>10.5f} | {su:>10} | {sl:>10}")