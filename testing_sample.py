import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import math
import numba as nb
from numba import njit

def del_duplicates(variable,eps):
    temp = []
    for i in range(len(variable)-1):
        if abs(variable[i]-variable[i+1])<eps:
            continue
        else:
            temp.append(variable[i])
    if variable[-1] not in temp:
        temp.append(variable[-1]) 
    return np.asarray(temp, dtype=np.float64)

class Data_file:
    def __init__(self,index,filename="data_storage.json"):
        self.index = index
        ''' reading data function but without hkl and everything of hkl file'''
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            sample = data[self.index]
            self.name = sample["name"]
            self.path = sample["data"]["path"] 
            self.kalpha1 = np.asarray(sample["data"]["kalpha1"][0])
            self.kalpha2 = np.asarray(sample["data"]["kalpha2"][0])
            self.theta_start = np.asarray(sample["data"]["theta_start"][0])
            self.theta_stop = np.asarray(sample["data"]["theta_stop"][0])
            self.uvw_xy = np.asarray(sample["data"]["uvw_xy"] )
            self.start_step_end = np.asarray(sample["data"]["start_step_end"] )
            self.counts = np.asarray(json.loads(sample["data"]["counts"]), dtype=float )
            self.counts_bac = np.asarray(json.loads(sample["data"]["counts_bac"] ),dtype=float)
            self.xy_and_u_fitted = sample["data"]["x_y_fitted"]
        nr_of_step = len(self.counts)
        self.x = np.linspace(self.theta_start, self.theta_stop, nr_of_step)
        if nr_of_step > 1:
            self.step = (self.theta_stop - self.theta_start) / (nr_of_step - 1)
        else:
            self.step = 0.0

    def read_data_of_hkl(self,filename="data_storage.json"):
        '''function reading data hkl and everything of hkl file and reducing doubled two theta peaks '''
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            sample = data[self.index]
            self.hkl_plus = sample["data"]["hkl"][0]
        self.h = self.hkl_plus['h']
        self.k = self.hkl_plus['k']
        self.l = self.hkl_plus['l']
        self.mult = self.hkl_plus['mult']
        self.sinT_lamb = self.hkl_plus['sinT_lamb']
        self.two_theta = self.hkl_plus['tt']
        self.fwhm = self.hkl_plus['fwhm']
        self.f2 = self.hkl_plus['f2']
        self.sf2 = self.hkl_plus['sf2']
        self.fwhm = del_duplicates(self.fwhm,0.001)
        self.two_theta = del_duplicates(self.two_theta,0.001)

    def without_ap(self,without_ap_counts):
        self.without_aparature_counts =  np.asarray(without_ap_counts, dtype=float )
        
    def find_counts_of(self,peaks):
        counts_of_peak_in_index(self.counts,peaks)
        self.two_theta_in_counts = np.asarray(peaks,dtype=int )

    def del_bac_counts(self,only_counts):
        self.without_bac_counts = np.asarray(only_counts, dtype=float )

@njit
def delete_aparature(counts,counts_bac,x,ap_counts,ap_x,ap_step):
    """ function to subtract aparature """
    eps = 0.5*ap_step
    end_counts = np.zeros(len(counts))
    for i in range(len(ap_counts)):
        for j in range(len(counts)):
            if abs(ap_x[i] - x[j])<eps:
                a = int(counts[j]-counts_bac[j]-ap_counts[i])
                if a < 0 :
                    a = 0
                end_counts[j] = a
    return end_counts

@njit
def find_index_of_two_theta(two_theta,theta_start,step):
    '''function finding peak index in range of x where peak from two theta extsts'''
    peaks = np.zeros(len(two_theta))
    for i in range(len(two_theta)):
        peaks[i] = (two_theta[i]-theta_start)/step
    for i in range(len(peaks)):
        if peaks[i]-int(peaks[i])>0.5:
            peaks[i] = int(peaks[i]+1)
        else:
            peaks[i] = int(peaks[i]) 
    return peaks

def plot_sample__bac(counts,counts_bac,name,x,on):
    plt.figure(figsize=(10, 5))
    plt.plot(x, counts, label="Counts (sample)", color="blue")
    plt.xlabel("2Theta")
    plt.ylabel("Counts")
    plt.title(f"Sample of {name} ")
    if on:
        plt.plot(x, counts_bac, label="Counts (background)", color="orange")
    plt.legend()
    plt.show()

def plot_with_theta_sample(counts,name,two_theta,x):
    plt.plot(x, counts, label="Counts (sample)", color="blue")
    for i in range(len(two_theta)):
        plt.axvline(two_theta[i], color='red', linestyle='--', linewidth=0.5)
    plt.xlabel("2Theta")
    plt.ylabel("Counts")
    plt.title(f"Sample of {name} with marked existing peaks")
    plt.legend()
    plt.show()

def checking_part_of_plot(counts,counts_without_bac,counts_without_aparature, two_theta,x,a=150):
    range_of_part = [0]
    k = 1
    while range_of_part[-1]<len(x)-a:
        range_of_part.append(k*a)
        k+=1
    for j in range(len(range_of_part)-1):
        for i in range(len(two_theta)):
            if two_theta[i] < x[range_of_part[j+1]] and two_theta[i] > x[range_of_part[j]]:
                plt.axvline(x=two_theta[i], color='red', linestyle='--', linewidth=0.5)
        plt.plot(x[range_of_part[j]:range_of_part[j+1]],counts[range_of_part[j]:range_of_part[j+1]],color='green',label = 'Counts')
        plt.plot(x[range_of_part[j]:range_of_part[j+1]],counts_without_bac[range_of_part[j]:range_of_part[j+1]],color='red',label ='Counts without background')
        plt.plot(x[range_of_part[j]:range_of_part[j+1]],counts_without_aparature[range_of_part[j]:range_of_part[j+1]],color='blue',label ='Counts without aparature and background')
        # plt.plot(x[range_of_part[j]:range_of_part[j+1]:2],counts[range_of_part[j]:range_of_part[j+1]:2])
        plt.scatter(x[range_of_part[j]:range_of_part[j+1]],counts[range_of_part[j]:range_of_part[j+1]],color='orange', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

    for i in range(len(two_theta)):
            if two_theta[i] < x[range_of_part[-1]] and two_theta[i] > x[range_of_part[-2]]:
                plt.axvline(x=two_theta[i], color='red', linestyle='--', linewidth=0.5)
    plt.plot(x[range_of_part[-2]:],counts[range_of_part[-2]:],color= 'green',label ='Counts')
    plt.plot(x[range_of_part[-2]:],counts_without_bac[range_of_part[-2]:],color= 'red',label ='Counts without background')
    plt.plot(x[range_of_part[-2]:],counts_without_aparature[range_of_part[-2]:],color= 'blue',label ='Counts without aparature and background')
    plt.scatter(x[range_of_part[-2]:],counts[range_of_part[-2]:],color='orange', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

@nb.njit
def counts_of_peak_in_index(counts,peak_index):
    counts_in_peak = []
    for i in range(len(counts)):
        if i in peak_index:
            counts_in_peak.append(counts[i])
    return counts_in_peak  

@nb.njit
def delete_bac(counts,counts_bac):
    """ function to subtract background """
    only_counts=np.zeros(len(counts))
    for i in range(len(counts)):
        only_counts[i] = abs(counts[i]-counts_bac[i])
    return only_counts

# def gaussian_lorenzian(uvw_xy,kalpha_theta):
#     u = uvw_xy[0] 
#     v = uvw_xy[1]
#     w = uvw_xy[2]
#     x_ = uvw_xy[3]
#     y = uvw_xy[4]
#     theta = kalpha_theta[2]/2 #  tutaj coś jakby trzeba dopracować bo biorę tylko początkowy kąt, a co z końcowym ? 
#     fwhm2_gauss = u*math.tan(theta)**2+v*math.tan(theta)+w
#     fwhm_lorenz = x_*math.tan(theta)+(y/(math.cos(theta)))
#     print(fwhm2_gauss)
#     print(fwhm_lorenz)

     
@njit
def gaussian(x, A, x0, sigma):
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

# @njit
# def multi_gaussian(x, *params):
#     """Suma wielu funkcji Gaussa."""
#     y = np.zeros_like(x)
#     n = len(params) // 3
#     for i in range(n):
#         A = params[3*i]
#         x0 = params[3*i+1]
#         sigma = params[3*i+2]
#         y += gaussian(x, A, x0, sigma)
#     return y

# def plot_gauss(popt,peaks,count,fit_y,x,two_theta):
#     for i in range(len(peaks)):
#         A, x0, sigma = popt[3*i:3*i+3]
#         fwhm = 2.355 * sigma
#         print(f"Peak {i+1}: A={A:.3f}, x0={x0:.3f}, sigma={sigma:.3f}, FWHM={fwhm:.3f}")
#     # --- Podział wykresu co 20° 2θ ---
#     x_min = np.min(x)
#     x_max = np.max(x)
#     segment_width = 15  # szerokość segmentu (np. 20° 2θ)
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
#         fit_seg = fit_y[mask]
#         plt.figure(figsize=(8, 5))
#         plt.plot(x_seg, y_seg, label='Dane (po baseline)', alpha=0.6)
#         plt.plot(x_seg, fit_seg, lw=2, label='Suma Gaussów')
#         # rysuj tylko te Gaussy, które leżą w tym przedziale
#         for i in range(len(peaks)):
#             Ai, x0i, si = popt[3*i:3*i+3]
#             if seg_start <= x0i < seg_end:
#                 plt.plot(x_seg, gaussian(x_seg, Ai, x0i, si), '--', label=f'Gauss {i+1}')
#         # tylko piki w tym przedziale
#         local_peaks = [pk for pk in peaks if seg_start <= x[pk] < seg_end]
#         plt.scatter(x[local_peaks], count[local_peaks], color='k', zorder=10, label='Piki')
#         for i in range(len(two_theta)):
#             plt.axvline(two_theta[i], color='red', linestyle='--', linewidth=0.5)
#         plt.xlim(seg_start, seg_end)
#         plt.ylim(y_seg.min() * 0.9, y_seg.max() * 1.1)
#         plt.title(f'Dopasowanie Gaussów: {seg_start:.1f}–{seg_end:.1f}° 2θ')
#         plt.xlabel('2θ [°]')
#         plt.ylabel('Intensywność')
#         plt.legend(fontsize=8)
#         plt.tight_layout()
#         plt.show()

def plot_gauss(peaks,count,x,two_theta):
    # for i in range(len(peaks)):
    #     A, x0, sigma = popt[3*i:3*i+3]
    #     fwhm = 2.355 * sigma
    #     print(f"Peak {i+1}: A={A:.3f}, x0={x0:.3f}, sigma={sigma:.3f}, FWHM={fwhm:.3f}")
    # --- Podział wykresu co 20° 2θ ---
    x_min = np.min(x)
    x_max = np.max(x)
    segment_width = 15  # szerokość segmentu (np. 20° 2θ)
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
        # plt.plot(x_seg, fit_seg, lw=2, label='Suma Gaussów')
        # rysuj tylko te Gaussy, które leżą w tym przedziale
        # for i in range(len(peaks)):
        #     Ai, x0i, si = popt[3*i:3*i+3]
        #     if seg_start <= x0i < seg_end:
        #         plt.plot(x_seg, gaussian(x_seg, Ai, x0i, si), '--', label=f'Gauss {i+1}')
        # # tylko piki w tym przedziale
        local_peaks = [pk for pk in peaks if seg_start <= x[pk] < seg_end]
        plt.scatter(x[local_peaks], count[local_peaks], color='k', zorder=10, label='Piki')
        for i in range(len(two_theta)):
            plt.axvline(two_theta[i], color='red', linestyle='--', linewidth=0.5)
        plt.xlim(seg_start, seg_end)
        plt.ylim(y_seg.min() * 0.9, y_seg.max() * 1.1)
        plt.title(f'Dopasowanie Gaussów: {seg_start:.1f}–{seg_end:.1f}° 2θ')
        plt.xlabel('2θ [°]')
        plt.ylabel('Intensywność')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()


@njit
def rr2(x_fit,y_fit,A_guess, x0_guess, sigma_guess):
    fit_y = gaussian(x_fit, A_guess, x0_guess, sigma_guess)
    ss_res = np.sum((y_fit - fit_y) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    return  1 - ss_res / ss_tot

@njit
def gaussian(x, A, x0, sigma):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def all_find_peaks(peaks,two_theta,correct_peaks,x):
    plt.plot(x,peaks)
    plt.plot(x,two_theta)
    plt.plot(x,correct_peaks)

def peak_detect(count, x, heigh, dist, prom, two_theta,theta_start, step, eps = 4, plotting=False):
    peaks, _ = find_peaks(count, height=heigh, distance=dist, prominence=prom)
    if len(peaks) == 0:
        print(" Nie znaleziono pików - spróbuj obniżyć progu 'height' lub 'prominence'.")
        return False
    fitted_params,r2_list  = [], []  # (A, x0, sigma) dla każdego dopasowanego piku
    for pk in peaks:
        # ogranicz dane do małego okna wokół piku
        fit_range = 10
        left = max(0, pk - fit_range)
        right = min(len(x), pk + fit_range)
        x_fit = x[left:right]
        y_fit = count[left:right]
        A_guess = count[pk]
        x0_guess = x[pk]
        sigma_guess = 1.0
        try:
            popt, _ = curve_fit(
                gaussian, x_fit, y_fit,
                p0=[A_guess, x0_guess, sigma_guess],
                bounds=([0.0, x0_guess - 5.0, 0.01],
                        [np.max(count) * 5, x0_guess + 5.0, 10.0])
            )
            fitted_params.append(popt)
            A_fit, x0_fit, sigma_fit = popt
            # oblicz R² lokalnie
            r2_list.append(rr2(
                np.asarray(x_fit, dtype=np.float64),
                np.asarray(y_fit, dtype=np.float64),
                np.float64(A_fit),
                np.float64(x0_fit),
                np.float64(sigma_fit)
            ))
        except RuntimeError:
            # print(f" Nie udało się dopasować Gaussa do piku przy x = {x[pk]:.2f}")
            fitted_params.append((np.nan, np.nan, np.nan))
            r2_list.append(np.nan)

    # --- Analiza poprawności względem two_theta ---
    correct_peaks = []
    for i in range(len(two_theta)):
        for j in range(len(peaks)):
            if abs(peaks[j] - ((two_theta[i] - theta_start) / step)) < eps:
                correct_peaks.append(peaks[j])
    all_find_peaks(peaks,two_theta,correct_peaks,x)
    mean_r2 = np.nanmean(r2_list)
    # print(f"Znaleziono {len(peaks)} pików. Średnie R² dopasowań: {mean_r2:.4f},wydajność = {len(correct_peaks)/len(two_theta)}, eps = {eps}, h = {heigh}, d = {dist}, p = {prom}")
    if plotting == True:
        plot_gauss(correct_peaks, count, x, two_theta)
    return len(peaks),len(correct_peaks)/len(two_theta),heigh,dist,prom







# popt, pcov = curve_fit(gauss,f.x, f.without_aparature_counts, p0=[max(f.counts), f.x[np.argmax(f.counts)], 1])
# A, mu, sigma = popt

# # print("A =", A)
# # print("mu =", mu)
# # print("sigma =", sigma)

# # wykres
# plt.scatter(f.x, f.counts, s=10, label="dane")
# plt.plot(f.x, gauss(f.x, *popt), label="dopasowany Gauss", linewidth=2,color = 'red')
# plt.axvline(mu)
# plt.legend()
# plt.xlim(mu-sigma-1,mu+sigma+1)
# plt.axhline(A/2, color='r', linestyle='--', label='Pozioma linia')

# print(max(A * np.exp(-(f.x - mu)**2 / (2*sigma**2))))
# print(A/2)
# print(f.step)
# print((mu-f.theta_start)/f.step )

# print(f.x[int((mu-f.theta_start)/f.step)])

# for i in range(598,611):
#     # print(f.counts[i])
#     if abs(f.counts[i]-(A/2))<100:
#         print(f.counts[i]-(A/2))
#         print(i)
# for i in range(611,626):
#     # print(f.counts[i])
#     if abs(f.counts[i]-(A/2))<100:
#         print(f.counts[i]-(A/2))
#         print(i)
# print((616-606)*f.step)
# print(f.step)
# dd = [f.x[606],f.x[616]]
# kk = [f.counts[606],f.counts[616]]
# # # for i in range(len(f.counts)):
# # #     if f.counts[i]
# # for i in range(len(f.counts)):
# #     if (mu-f.theta_start)/f.step 
# plt.scatter(dd,kk,color ='green')
# plt.show()
