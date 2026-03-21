#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 12,
})

# --- Parameters ---
k = 4
dL_ref = 0.5
chi2_ref = 207
log10B_ref = 44
target_log10B = 2
prior_width = 0.5        # effective prior width per direction

ln10 = np.log(10)
C = log10B_ref - k * np.log10(dL_ref) - chi2_ref / (2 * ln10)

# Recover sigma_ref from the decomposition:
#   C = -k*log10(w) + k/2*log10(2pi) + k*log10(sigma_ref / dL_ref)
log10_sigma_over_dLref = (C + k * np.log10(prior_width)
                          - k / 2 * np.log10(2 * np.pi)) / k
sigma_ref = dL_ref * 10**log10_sigma_over_dLref
dL_cap_marginal = dL_ref * prior_width / sigma_ref

# --- Turning point (original curve minimum): chi2 = k ---
dL_min = dL_ref * np.sqrt(chi2_ref / k)

# Cap at whichever comes first: marginal hitting prior, or turning point
dL_cap = min(dL_cap_marginal, dL_min)
print(f'sigma_ref = {sigma_ref:.4f},  dL_cap_marginal = {dL_cap_marginal:.2f},  '
      f'dL_min = {dL_min:.2f},  dL_cap = {dL_cap:.2f}')

# --- dL grid (extend to cover both dL_cap and dL_min) ---
dL_hi = max(dL_min, dL_cap) * 5
dL_arr = np.geomspace(dL_ref * 0.5, dL_hi, 1000)

# --- chi2(dL) ---
chi2_arr = chi2_ref * (dL_ref / dL_arr)**2

# --- Original log10(B) ---
log10B_orig = C + k * np.log10(dL_arr) + chi2_arr / (2 * ln10)

# --- Capped log10(B): freeze log-det term at dL_cap ---
log10_dL_capped = np.minimum(np.log10(dL_arr), np.log10(dL_cap))
log10B_cap = C + k * log10_dL_capped + chi2_arr / (2 * ln10)

# --- Derivatives d(log10 B)/d(log10 dL) ---
deriv_orig = k - chi2_arr
deriv_cap = np.where(dL_arr <= dL_cap, k - chi2_arr, -chi2_arr)

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(dL_arr, log10B_orig, 'b-', lw=2.5, label='original SDDR', zorder=2)
ax1.plot(dL_arr, log10B_cap, color='mediumseagreen', lw=2.5,
         label='capped marginals', zorder=3)
ax1.axhline(target_log10B, color='k', ls=':', lw=1.2,
            label=f'target $\\log_{{10}} B = {target_log10B}$')
ax1.axvline(dL_min, color='blue', ls='--', lw=1.2, alpha=0.5,
            label=fr'turning point ($\chi^2 = k$, $d_L = {dL_min:.1f}$)')
ax1.axvline(dL_cap, color='mediumseagreen', ls='--', lw=1.2, alpha=0.5,
            label=fr'cap point ($\sigma = w$, $d_L = {dL_cap:.1f}$)')

# --- Two Newton steps on the original curve ---
def log10B_func(log10_dL):
    chi2 = chi2_ref * (dL_ref / 10**log10_dL)**2
    return C + k * log10_dL + chi2 / (2 * ln10)

def deriv_func(log10_dL):
    chi2 = chi2_ref * (dL_ref / 10**log10_dL)**2
    return k - chi2

log10_dL_arr = np.log10(dL_arr)
newton_colors = ['darkorange', 'darkorchid']
newton_x = [np.log10(dL_ref)]
newton_y = [log10B_ref]
for i in range(2):
    xi, yi = newton_x[-1], newton_y[-1]
    si = deriv_func(xi)
    tangent = yi + si * (log10_dL_arr - xi)
    ax1.plot(dL_arr, tangent, '--', color=newton_colors[i], lw=1.5, alpha=0.7)
    ax1.plot(10**xi, yi, 'o', color=newton_colors[i], ms=10, zorder=5,
             markeredgecolor='k', markeredgewidth=0.8)
    x_next = xi + (target_log10B - yi) / si
    ax1.plot(10**x_next, target_log10B, 's', color=newton_colors[i], ms=8,
             zorder=5, markeredgecolor='k', markeredgewidth=0.8)
    ax1.annotate('', xy=(10**x_next, log10B_func(x_next)),
                 xytext=(10**x_next, target_log10B),
                 arrowprops=dict(arrowstyle='->', color=newton_colors[i],
                                 lw=1.3))
    newton_x.append(x_next)
    newton_y.append(log10B_func(x_next))

ax1.plot(10**newton_x[-1], newton_y[-1], 'o', color=newton_colors[-1], ms=10,
         zorder=5, markeredgecolor='k', markeredgewidth=0.8)

ax1.plot([], [], '--', color='darkorange', lw=1.5, label='Newton tangent')
ax1.plot([], [], 'o', color='grey', ms=10, markeredgecolor='k',
         markeredgewidth=0.8, label='iterates on curve')

ax1.set_ylabel(r'$\log_{10}\, B$', fontsize=18)
y_lo = min(log10B_cap.min(), target_log10B - 2)
y_hi = log10B_ref + 3
ax1.set_ylim(y_lo, y_hi)
ax1.legend(loc='upper right', fontsize=16)
# ax1.set_title(fr'SDDR: $k={k}$, $\chi^2_{{\rm ref}}={chi2_ref:.0f}$, '
#               fr'$w={prior_width}$')

ax2.plot(dL_arr, deriv_orig, 'b-', lw=2.5, label='original')
ax2.plot(dL_arr, deriv_cap, color='mediumseagreen', lw=2.5, label='capped')
ax2.axhline(0, color='k', ls=':', lw=1.2)
ax2.axvline(dL_min, color='blue', ls='--', lw=1.2, alpha=0.5)
ax2.axvline(dL_cap, color='mediumseagreen', ls='--', lw=1.2, alpha=0.5)
ax2.set_ylabel(r'$k-\chi^2$', fontsize=18)
ax2.set_xlabel(r'$d_L$', fontsize=18)
ax2.legend(loc='lower right', fontsize=16)

for ax in (ax1, ax2):
    ax.set_xscale('log')
    ax.tick_params(direction='in', which='both')

plt.tight_layout()
label = f'capped_w{prior_width}_chi2{chi2_ref:.0f}'
plt.savefig(f'plots/newton_convergence_diagnostic_{label}.pdf')
print(f'Saved plots/newton_convergence_diagnostic_{label}.pdf')
