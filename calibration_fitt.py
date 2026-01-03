import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from testing_sample import *
import numpy as np
import numba as nb

# def pseudo_voigt(x, A, x0, sigma, gamma, eta):
#     G = np.exp(-(x - x0)**2 / (2 * sigma**2))
#     L = gamma**2 / ((x - x0)**2 + gamma**2)
#     return A * (eta * L + (1 - eta) * G)

def pseudo_voigt(x, A, x0, fwhm, eta):
    """
    Pseudo-Voigt profile
    eta ∈ [0,1]  (0 = Gauss, 1 = Lorentz)
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gauss = np.exp(-(x - x0)**2 / (2 * sigma**2))
    lorentz = 1 / (1 + ((x - x0) / (fwhm / 2))**2)
    return A * (eta * lorentz + (1 - eta) * gauss)

def rr2_pv(x, y, A, x0, fwhm, eta):
    y_fit = pseudo_voigt(x, A, x0, fwhm, eta)
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

# def gaussian(x, A, x0, sigma):
#     return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

# def gaussian_bg(x, A, x0, sigma, c):
#     return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + c

# def gaussian_asym(x, A, x0, sigma_l, sigma_r):
#     return np.where(
#         x < x0,
#         A * np.exp(-(x - x0)**2 / (2 * sigma_l**2)),
#         A * np.exp(-(x - x0)**2 / (2 * sigma_r**2))
#     )

# def lorentzian(x, A, x0, gamma):
#     return A * (gamma**2 / ((x - x0)**2 + gamma**2))

# def gauss_lorentz(x, A, x0, sigma, gamma):
#     return (
#         A * np.exp(-(x - x0)**2 / (2 * sigma**2)) +
#         A * (gamma**2 / ((x - x0)**2 + gamma**2))
#     )

# def pseudo_voigt_bg(x, A, x0, sigma, gamma, eta, c):
#     return pseudo_voigt(x, A, x0, sigma, gamma, eta) + c

# def pseudo_voigt_bg_lin(x, A, x0, sigma, gamma, eta, a, b):
#     return pseudo_voigt(x, A, x0, sigma, gamma, eta) + a*x + b

# def multi_gaussian(x, *params):
#     y = np.zeros_like(x)
#     for i in range(0, len(params), 3):
#         A, x0, sigma = params[i:i+3]
#         y += gaussian(x, A, x0, sigma)
#     return y

# def rr2(x, y, A, x0, sigma):
#     y_fit = gaussian(x, A, x0, sigma)
#     ss_res = np.sum((y - y_fit)**2)
#     ss_tot = np.sum((y - np.mean(y))**2)
#     return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

# def rr2_weighted(x, y, A, x0, sigma):
#     y_fit = gaussian(x, A, x0, sigma)
#     w = 1 / np.maximum(y, 1)
#     ss_res = np.sum(w * (y - y_fit)**2)
#     ss_tot = np.sum(w * (y - np.mean(y))**2)
#     return 1 - ss_res / ss_tot



def plot_gauss_ap(peaks,count,x,segment_width = 15):
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
        plt.ylim(y_seg.min() * 0.9, y_seg.max() * 1.1)
        plt.title(f'Dopasowanie Gaussów: {seg_start:.1f}–{seg_end:.1f}° 2θ')
        plt.xlabel('2θ [°]')
        plt.ylabel('Intensywność')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

def finding_fwhm_gauss(peaks,gauss_fit):
    '''in order - A, mu,sigma '''
    N = len(peaks)
    fwhm = np.zeros(N)
    t_fwhm = np.zeros(N)
    for j in range(len(peaks)):
        fwhm[j] = 2 * gauss_fit[j][2]* math.sqrt(2 * math.log(2))
        t_fwhm[j] = gauss_fit[j][1]
    return fwhm,t_fwhm

def ask_user_peak_acceptance():
    while True:
        decision = input(
            "\nCzy zaakceptować ten pik?\n"
            "[a] akceptuj\n"
            "[r] dopasuj ponownie\n"
            "[s] pomiń pik\n"
            "[q] zakończ\n"
            ">>> "
        ).lower()
        if decision in ("a", "r", "s", "q"):
            return decision
        print("Nieprawidłowa opcja.")

def plot_peak_fit_with_fwhm_ap(x, y, popt,
                            peak_index=None,
                            window=20,
                            fwhm_theoretical=None):
    """
    Wizualizacja dopasowania pseudo-Voigt + FWHM (fit i teoria) + 2θ

    Parameters
    ----------
    x, y : array
        Dane pomiarowe
    popt : tuple
        (A, x0, fwhm, eta) – parametry pseudo-Voigt (dopasowanie)
    two_theta : array
        Teoretyczne pozycje pików 2θ
    peak_index : int or None
        Indeks piku w tablicy x
    window : float
        Połowa szerokości okna rysowania wokół x0
    fwhm_theoretical : float or None
        Teoretyczna wartość FWHM (np. instrumentalna)
    """
    A, x0, fwhm, eta = popt
    # zakres rysowania
    mask = (x >= x0 - window) & (x <= x0 + window)
    x_plot = x[mask]
    y_plot = y[mask]
    # dopasowana krzywa
    y_fit = pseudo_voigt(x_plot, A, x0, fwhm, eta)
    half_max = A / 2
    fwhm_left  = x0 - fwhm / 2
    fwhm_right = x0 + fwhm / 2
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot, 'o', ms=4, alpha=0.6, label='Dane')
    plt.plot(x_plot, y_plot, ms=4, alpha=0.6, label='Dane')
    plt.plot(x_plot, y_fit, 'r-', lw=2,alpha = 0.5, label='Pseudo-Voigt fit')
    plt.hlines(
        half_max,
        fwhm_left,
        fwhm_right,
        colors='blue',
        lw=2,
        label=f'FWHM (fit) = {fwhm:.3f}°'
    )
    plt.axvline(x0, color='green', linestyle=':', lw=1.5, label='x₀ (fit)')
    if peak_index is not None:
        plt.scatter(x[peak_index], y[peak_index],
                    color='black', zorder=10, label='Pik (dane)')
    plt.ylim(np.min(y),np.max(y_plot)*1.2)
    plt.xlabel('2θ [°]')
    plt.ylabel('Intensywność')
    plt.title(
        f'Pseudo-Voigt | η={eta:.2f} | '
    )
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    
def finding_fwhm_ap(gauss_fit):
    """
    Calculate FWHM, peak positions and peak weights from Gaussian fit.
    gauss_fit: array-like of shape (N, 3)
        Each row: [A, mu, sigma]
    """
    gauss_fit = np.asarray(gauss_fit)

    A = gauss_fit[:, 0]
    mu = gauss_fit[:, 1]
    sigma = np.abs(gauss_fit[:, 2])

    fwhm = 2 * sigma * np.sqrt(2 * np.log(2))

    # waga = pole pod pikiem
    area = A * sigma * np.sqrt(2*np.pi)
    weights = area / np.max(area)

    return fwhm, mu, weights



def peak_detect_pv_ap(counts, x, heigh, dist, prom,*,r2_min = 0.6, plotting=False,fitting = False,correction = False, fit_range = 10):
    ''' If fitting = False - function return values:
    len(peaks),len(correct_peaks)/len(two_theta),heigh,dist,prom ;
    If fitting = True - function return values:
    peaks,correct_peaks,index_corrected,gauss_fit,_  '''
    good_index = []
    peaks, props = find_peaks(counts, height=heigh,width=1.5,prominence=prom,distance=dist)
    if len(peaks) == 0:
        print(" Nie znaleziono pików - spróbuj obniżyć progu 'height' lub 'prominence'.")
        return False
    fitted_params,r2_list,gauss_fit  = [],[],[]  # (A, x0, sigma) dla każdego dopasowanego piku
    for pk in peaks:
        left = max(0, pk - fit_range)
        right = min(len(x), pk + fit_range)
    
        x_fit = x[left:right]
        y_fit = counts[left:right]
        A_guess = counts[pk]
        x0_guess = x[pk]
        try:
            popt, _ = curve_fit(
                pseudo_voigt, x_fit, y_fit,
                p0=[A_guess, x0_guess, 1.0, 0.5],   # fwhm, eta
                bounds=([0.0, x0_guess - 5.0, 0.01, 0.0],
                [np.max(counts) * 5, x0_guess + 5.0, 10.0, 1.0])
            )
            fitted_params.append(popt)
            A_fit, x0_fit, fwhm_fit, eta_fit = popt
            # oblicz R² lokalnie
            r2 = rr2_pv(x_fit, y_fit, A_fit, x0_fit, fwhm_fit, eta_fit)

            accepted = False
            if r2 >= r2_min:
                while not accepted:
                    plot_peak_fit_with_fwhm_ap(
                        x,
                        counts,
                        popt=[A_fit, x0_fit, fwhm_fit, eta_fit],
                        peak_index=pk,
                        window=1
                    )
                    print(f"R² = {r2:.3f}")
                    decision = ask_user_peak_acceptance()
                    if decision == "a":
                        gauss_fit.append([A_fit, x0_fit, fwhm_fit, eta_fit])
                        r2_list.append(r2)
                        good_index.append(pk)
                        accepted = True

                    elif decision == "s":
                        print("⏭ Pik pominięty")
                        break

                    elif decision == "r":
                        print(" Ponowne dopasowanie...")
                        A_guess *= 0.9
                        continue

                    elif decision == "q":
                        print(" Przerwano przez użytkownika")
                        return good_index, good_index, np.arange(len(good_index)), np.asarray(gauss_fit), props

        except RuntimeError:
            fitted_params.append((np.nan, np.nan, np.nan))
            r2_list.append(np.nan)
    correct_peaks,index_corrected = [],[]
    if correction ==  True:
        print("No correction available for aparature")
    else:
        correct_peaks = good_index           
        index_corrected = np.arange(len(good_index))   
    if plotting == True:
        plot_gauss_ap(good_index, counts, x)  
    if fitting == True:
        return good_index,correct_peaks,index_corrected,np.asarray(gauss_fit),props
    return good_index,correct_peaks,index_corrected,np.asarray(gauss_fit),props



def fit_poly_lower(
    x,
    y,
    weight,
    x_min=60,
    q=0.2,
    deg=1,
    log_weight=False
):
    x = np.asarray(x)
    y = np.asarray(y)
    weight = np.asarray(weight)

    # odetnij początek
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

    # --- dopasowanie z wagami ---
    coeffs = np.polyfit(x0, y0, deg, w=w0)
    poly = np.poly1d(coeffs)

    y_fit = poly(x)
    y_fit[x < x_min] = np.nan

    # --- COLORSCALE ---
    w = weight.astype(float)

    if log_weight:
        w = np.log1p(w)

    w_min, w_max = np.nanmin(w), np.nanmax(w)

    if w_max > w_min:
        w_norm = (w - w_min) / (w_max - w_min)
    else:
        w_norm = np.zeros_like(w)

    # punkty spoza zakresu dopasowania
    w_norm[x < x_min] = np.nan

    return y_fit, poly, w_norm






