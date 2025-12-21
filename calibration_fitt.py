# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.optimize import curve_fit
# from testing_sample import *
# import numba as nb

# def pseudo_voigt(x, A, x0, sigma, gamma, eta):
#     G = np.exp(-(x - x0)**2 / (2 * sigma**2))
#     L = gamma**2 / ((x - x0)**2 + gamma**2)
#     return A * (eta * L + (1 - eta) * G)

# def rr2_pv(x, y, A, x0, sigma, gamma, eta):
#     y_fit = pseudo_voigt(x, A, x0, sigma, gamma, eta)
#     ss_res = np.sum((y - y_fit)**2)
#     ss_tot = np.sum((y - np.mean(y))**2)
#     return 1 - ss_res / ss_tot

# def plot_gauss_ap(peaks,count,x,segment_width = 15,two_theta = None):
#     x_min = np.min(x)
#     x_max = np.max(x)
#     # szerokość segmentu (np. 20° 2θ)
#     num_segments = int(np.ceil((x_max - x_min) / segment_width))
#     print(f"\nTworzę {num_segments} wykresów (co {segment_width}° 2θ)...")
#     for seg in range(num_segments):
#         seg_start = x_min + seg * segment_width
#         seg_end = seg_start + segment_width
#         mask = (x >= seg_start) & (x < seg_end)
#         if not np.any(mask):
#             continue
#         x_seg = x[mask]
#         y_seg = count[mask]
#         plt.figure(figsize=(8, 5))
#         plt.plot(x_seg, y_seg, label='Dane (po baseline)', alpha=0.6)
#         local_peaks = [pk for pk in peaks if seg_start <= x[pk] < seg_end]
#         plt.scatter(x[local_peaks], count[local_peaks], color='k', zorder=10, label='Piki')
#         plt.xlim(seg_start, seg_end)
#         # plt.ylim(y_seg.min() * 0.9, y_seg.max() * 1.1)
#         if two_theta is not None:
#             for i in range(len(two_theta)):
#                 plt.axvline(two_theta[i], color='red', linestyle='--', linewidth=0.5)
#         plt.ylim(0, y_seg.max() * 1.1)
#         plt.title(f'Dopasowanie Gaussów: {seg_start:.1f}–{seg_end:.1f}° 2θ')
#         plt.xlabel('2θ [°]')
#         plt.ylabel('Intensywność')
#         plt.legend(fontsize=8)
#         plt.tight_layout()
#         plt.show()


# def peak_detect_gauss_quality(counts, x, r2_min=0.9, fit_range=10, height=None, distance=None, prominence=None):
#     """
#     Peak detection + Gaussian fitting with R² quality filter.
#     Returns:
#         good_peaks : np.ndarray (indices)
#         x_peaks    : np.ndarray (x values of peaks)
#         y_peaks    : np.ndarray (y values of peaks)
#         gauss_fit  : list of fitted parameters [A, mu, sigma]
#         r2_list    : list of R² values
#     """
#     peaks, _ = find_peaks(
#         counts,
#         height=height,
#         distance=distance,
#         prominence=prominence
#     )
#     if len(peaks) == 0:
#         return np.array([]), np.array([]), np.array([]), [], []

#     good_peaks, x_peaks, y_peaks, gauss_fit, r2_list = [],[],[],[],[]
#     for pk in peaks:
#         left = max(0, pk - fit_range)
#         right = min(len(x), pk + fit_range)
#         x_fit = x[left:right]
#         y_fit = counts[left:right]
#         A0 = counts[pk]
#         mu0 = x[pk]
#         sigma0 = (x_fit[-1] - x_fit[0]) / 6 if len(x_fit) > 1 else 1.0
#         try:
#             popt, _ = curve_fit(
#                 gaussian,
#                 x_fit,
#                 y_fit,
#                 p0=[A0, mu0, sigma0],
#                 bounds=(
#                     [0.0, mu0 - fit_range, 1e-3],
#                     [np.inf, mu0 + fit_range, np.inf]
#                 )
#             )
#             r2 = rr2(x_fit, y_fit, *popt)
#             if r2 >= r2_min:
#                 good_peaks.append(pk)
#                 x_peaks.append(x[pk])
#                 y_peaks.append(counts[pk])
#                 gauss_fit.append(popt)
#                 r2_list.append(r2)
#         except RuntimeError:
#             continue
#     return np.array(y_peaks),gauss_fit,r2_list

# @nb.njit
# def finding_fwhm(peaks, gauss_fit):
#     """
#     Calculate FWHM and peak positions from Gaussian fit parameters.
#     gauss_fit: array-like of shape (N, 3)
#         Each row: [A, mu, sigma]
#     """
#     peaks = np.asarray(peaks)
#     peaks_max = np.max(peaks)
#     peaks_weight = np.zeros(len(peaks))
#     peaks_weight = peaks/peaks_max

#     gauss_fit = np.asarray(gauss_fit)
#     sigma = np.abs(gauss_fit[:, 2])
#     mu = gauss_fit[:, 1]

#     fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
#     t_fwhm = mu
#     return fwhm, t_fwhm, peaks_weight


# def fit_poly_lower(x, y, weight, x_min=60, q=0.2, deg=1, log_weight=False ):
#     x = np.asarray(x)
#     y = np.asarray(y)
#     weight = np.asarray(weight)
#     mask = x >= x_min
#     x0 = x[mask]
#     y0 = y[mask]
#     w0 = weight[mask]

#     if len(x0) < deg + 1:
#         return (
#             np.full_like(x, np.nan, dtype=float),
#             None,
#             np.full_like(x, np.nan, dtype=float)
#         )
#     coeffs = np.polyfit(x0, y0, deg, w=w0)
#     poly = np.poly1d(coeffs)
#     y_fit = poly(x)
#     y_fit[x < x_min] = np.nan

#     w = weight.astype(float)
#     if log_weight:
#         w = np.log1p(w)
#     w_min, w_max = np.nanmin(w), np.nanmax(w)
#     if w_max > w_min:
#         w_norm = (w - w_min) / (w_max - w_min)
#     else:
#         w_norm = np.zeros_like(w)
#     w_norm[x < x_min] = np.nan
#     return y_fit, poly, w_norm

# def set_aparature_fit(counts,x,r2_min = 0.6,plotting = False):
#     ap_good_peaks_counts,ap_gauss_fit,_ = peak_detect_gauss_quality(counts,x,r2_min)
#     ap_fwhm,ap_t_fwhm,ap_peaks_weight = finding_fwhm(ap_good_peaks_counts,ap_gauss_fit)
#     y_fit,poly, colorscale = fit_poly_lower(ap_t_fwhm, ap_fwhm,ap_peaks_weight, x_min=0, q=0.2, deg=2)

#     if plotting == True:
#         plt.figure(figsize=(8, 5))
#         plt.scatter(ap_t_fwhm, ap_fwhm, s=30, alpha=0.7, label="dane", c=colorscale,cmap="viridis")
#         plt.plot(ap_t_fwhm, y_fit, linewidth=2, label=f"polyfit deg={poly.order}")
#         plt.colorbar(label="Znaczenie punktu")
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#     return y_fit


    

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from testing_sample import *
import numpy as np
import numba as nb

def pseudo_voigt(x, A, x0, sigma, gamma, eta):
    G = np.exp(-(x - x0)**2 / (2 * sigma**2))
    L = gamma**2 / ((x - x0)**2 + gamma**2)
    return A * (eta * L + (1 - eta) * G)

def rr2_pv(x, y, A, x0, sigma, gamma, eta):
    y_fit = pseudo_voigt(x, A, x0, sigma, gamma, eta)
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

def plot_gauss_ap(peaks,count,x,segment_width = 15,two_theta = None):
    x_min = np.min(x)
    x_max = np.max(x)
    # szerokość segmentu (np. 20° 2θ)
    num_segments = int(np.ceil((x_max - x_min) / segment_width))
    print(f"\nTworzę {num_segments} wykresów (co {segment_width}° 2θ)...")
    for seg in range(num_segments):
        seg_start = x_min + seg * segment_width
        seg_end = seg_start + segment_width
        mask = (x >= seg_start) & (x < seg_end)
        if not np.any(mask):
            continue
        x_seg = x[mask]
        y_seg = count[mask]
        plt.figure(figsize=(8, 5))
        plt.plot(x_seg, y_seg, label='Dane (po baseline)', alpha=0.6)
        local_peaks = [pk for pk in peaks if seg_start <= x[pk] < seg_end]
        plt.scatter(x[local_peaks], count[local_peaks], color='k', zorder=10, label='Piki')
        plt.xlim(seg_start, seg_end)
        # plt.ylim(y_seg.min() * 0.9, y_seg.max() * 1.1)
        if two_theta is not None:
            for i in range(len(two_theta)):
                plt.axvline(two_theta[i], color='red', linestyle='--', linewidth=0.5)
        plt.ylim(0, y_seg.max() * 1.1)
        plt.title(f'Dopasowanie Gaussów: {seg_start:.1f}–{seg_end:.1f}° 2θ')
        plt.xlabel('2θ [°]')
        plt.ylabel('Intensywność')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()


def peak_detect_gauss_quality(counts, x, r2_min=0.9, fit_range=10, height=None, distance=None, prominence=None):
    """
    Peak detection + Gaussian fitting with R² quality filter.
    Returns:
        good_peaks : np.ndarray (indices)
        x_peaks    : np.ndarray (x values of peaks)
        y_peaks    : np.ndarray (y values of peaks)
        gauss_fit  : list of fitted parameters [A, mu, sigma]
        r2_list    : list of R² values
    """
    peaks, _ = find_peaks(
        counts,
        height=height,
        distance=distance,
        prominence=prominence
    )
    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([]), [], []

    good_peaks, x_peaks, y_peaks, gauss_fit, r2_list = [],[],[],[],[]
    for pk in peaks:
        left = max(0, pk - fit_range)
        right = min(len(x), pk + fit_range)
        x_fit = x[left:right]
        y_fit = counts[left:right]
        A0 = counts[pk]
        mu0 = x[pk]
        sigma0 = (x_fit[-1] - x_fit[0]) / 6 if len(x_fit) > 1 else 1.0
        try:
            popt, _ = curve_fit(
                gaussian,
                x_fit,
                y_fit,
                p0=[A0, mu0, sigma0],
                bounds=(
                    [0.0, mu0 - fit_range, 1e-3],
                    [np.inf, mu0 + fit_range, np.inf]
                )
            )
            r2 = rr2(x_fit, y_fit, *popt)
            if r2 >= r2_min:
                good_peaks.append(pk)
                x_peaks.append(x[pk])
                y_peaks.append(counts[pk])
                gauss_fit.append(popt)
                r2_list.append(r2)
        except RuntimeError:
            continue
    return np.array(y_peaks),gauss_fit,r2_list
    
# @nb.njit
def finding_fwhm(peaks, gauss_fit):
    """
    Calculate FWHM and peak positions from Gaussian fit parameters.
    gauss_fit: array-like of shape (N, 3)
        Each row: [A, mu, sigma]
    """
    peaks = np.asarray(peaks)
    peaks_max = np.max(peaks)
    peaks_weight = np.zeros(len(peaks))
    peaks_weight = peaks/peaks_max

    gauss_fit = np.asarray(gauss_fit)
    sigma = np.abs(gauss_fit[:, 2])
    mu = gauss_fit[:, 1]

    fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
    t_fwhm = mu
    return fwhm, t_fwhm, peaks_weight

def fit_poly_lower(x, y, weight, x_min=60, q=0.2, deg=1, log_weight=False ):
    x = np.asarray(x)
    y = np.asarray(y)
    weight = np.asarray(weight)
    mask = x >= x_min
    x0 = x[mask]
    y0 = y[mask]
    w0 = weight[mask]

    if len(x0) < deg + 1:
        return (
            np.full_like(x, np.nan, dtype=float),
            None,
            np.full_like(x, np.nan, dtype=float)
        )
    coeffs = np.polyfit(x0, y0, deg, w=w0)
    poly = np.poly1d(coeffs)
    y_fit = poly(x)
    y_fit[x < x_min] = np.nan

    w = weight.astype(float)
    if log_weight:
        w = np.log1p(w)
    w_min, w_max = np.nanmin(w), np.nanmax(w)
    if w_max > w_min:
        w_norm = (w - w_min) / (w_max - w_min)
    else:
        w_norm = np.zeros_like(w)
    w_norm[x < x_min] = np.nan
    return y_fit, poly, w_norm

def set_aparature_fit(counts,x,r2_min = 0.6,plotting = False,height=None,dist=None,prom=None):
    ap_good_peaks_counts,ap_gauss_fit,_ = peak_detect_gauss_quality(counts,x,r2_min,height=height, distance=dist, prominence=prom)
    ap_fwhm,ap_t_fwhm,ap_peaks_weight = finding_fwhm(ap_good_peaks_counts,ap_gauss_fit)
    y_fit,poly, colorscale = fit_poly_lower(ap_t_fwhm, ap_fwhm,ap_peaks_weight, x_min=0, q=0.2, deg=2)

    if plotting == True:
        plt.figure(figsize=(8, 5))
        plt.scatter(ap_t_fwhm, ap_fwhm, s=30, alpha=0.7, label="dane", c=colorscale,cmap="viridis")
        plt.plot(ap_t_fwhm, y_fit, linewidth=2, label=f"polyfit deg={poly.order}")
        plt.colorbar(label="Znaczenie punktu")
        plt.legend()
        plt.grid(True)
        plt.show()

    return y_fit,ap_t_fwhm


    















