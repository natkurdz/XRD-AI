import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import math
import numba as nb


class Data_file:
    def __init__(self,index):
        self.index = index

    def read_data(self,filename="data_storage.json"):
        ''' reading data function but without hkl and everything of hkl file'''
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            sample = data[self.index]
            self.name = sample["name"]
            self.path = sample["data"]["path"] 
            self.kalpha1 = sample["data"]["kalpha1"][0]
            self.kalpha2 = sample["data"]["kalpha2"][0]
            self.theta_start = sample["data"]["theta_start"][0]
            self.theta_stop = sample["data"]["theta_stop"][0]
            self.uvw_xy = sample["data"]["uvw_xy"] 
            self.start_step_end = sample["data"]["start_step_end"] 
            self.counts = json.loads(sample["data"]["counts"] )
            self.counts_bac = json.loads(sample["data"]["counts_bac"])
        nr_of_step = len(self.counts)
        self.step = ((self.theta_stop - self.theta_start)/nr_of_step)
        self.x = np.arange(self.theta_start, self.theta_stop, self.step)        

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
        theta = []
        for i in range(len(self.two_theta)-1):
            if abs(self.two_theta[i]-self.two_theta[i+1])<0.001:
                continue
            else:
                theta.append(self.two_theta[i])
        if self.two_theta[-1] not in theta:
            theta.append(self.two_theta[-1]) 
        self.two_theta = theta

    def find_counts_of_peaks(self):
        '''function finding peak index in range of x where peak from two theta extsts'''
        peaks = np.zeros(len(self.two_theta))
        for i in range(len(self.two_theta)):
            peaks[i] = (self.two_theta[i]-self.theta_start)/self.step
        for i in range(len(peaks)):
            if peaks[i]-int(peaks[i])>0.5:
                peaks[i] = int(peaks[i]+1)
            else:
                peaks[i] = int(peaks[i]) 
        self.two_theta_x_index = peaks

    def delete_aparature(self,aparature_counts,x_ap,ap_step):
        """ function to subtract aparature """
        eps = 0.5*ap_step
        ap_counts = np.zeros(len(self.counts))
        for i in range(len(aparature_counts)):
            for j in range(len(self.counts)):
                if abs(x_ap[i] - self.x[j])<eps:
                    a = int(self.counts[j]-self.counts_bac[j]-aparature_counts[i])
                    if a < 0 :
                        a = 0
                    ap_counts[j] = a
        self.without_aparature_counts = ap_counts
    

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
    plt.title(f"Sample of {name} ")
    plt.legend()
    plt.show()

def checking_part_of_plot(counts,two_theta,x,a=150):
    range_of_part = [0]
    k = 1
    while range_of_part[-1]<len(x)-a:
        range_of_part.append(k*a)
        k+=1
    for j in range(len(range_of_part)-1):
        for i in range(len(two_theta)):
            if two_theta[i] < x[range_of_part[j+1]] and two_theta[i] > x[range_of_part[j]]:
                plt.axvline(x=two_theta[i], color='red', linestyle='--', linewidth=0.5)
        plt.plot(x[range_of_part[j]:range_of_part[j+1]],counts[range_of_part[j]:range_of_part[j+1]],color='green')
        # plt.plot(x[range_of_part[j]:range_of_part[j+1]:2],counts[range_of_part[j]:range_of_part[j+1]:2])
        plt.scatter(x[range_of_part[j]:range_of_part[j+1]],counts[range_of_part[j]:range_of_part[j+1]],color='orange', linestyle='--', linewidth=0.5)
        plt.show()
    for i in range(len(two_theta)):
            if two_theta[i] < x[-1] and two_theta[i] > x[3000]:
                plt.axvline(x=two_theta[i], color='red', linestyle='--', linewidth=0.5)
    plt.plot(x[3000:],counts[3000:])
    plt.scatter(x[3000:],counts[3000:],color='orange', linestyle='--', linewidth=0.5)
    plt.show()
   
def part_of_plot(counts,peak_index,x,a=150):
    range_of_part,peak_count = [0],[]
    k = 1
    while range_of_part[-1]<len(x)-a:
        range_of_part.append(k*a)
        k+=1
    for i in range(len(x)):
        if i in peak_index:
            peak_count.append(counts[i])
            plt.axvline(x[i],color='red', linestyle='--', linewidth=0.5)
    plt.plot(x, counts, label="Counts (sample)", color="blue")
    plt.show()
    return peak_count   

def delete_bac(counts,counts_bac):
    """ function to subtract background """
    only_counts=np.zeros(len(counts))
    for i in range(len(counts)):
        only_counts[i] = abs(counts[i]-counts_bac[i])
    return only_counts


  
def gaussian_lorenzian(uvw_xy,kalpha_theta):
    u = uvw_xy[0] 
    v = uvw_xy[1]
    w = uvw_xy[2]
    x_ = uvw_xy[3]
    y = uvw_xy[4]
    theta = kalpha_theta[2]/2 #  tutaj coś jakby trzeba dopracować bo biorę tylko początkowy kąt, a co z końcowym ? 
    fwhm2_gauss = u*math.tan(theta)**2+v*math.tan(theta)+w
    fwhm_lorenz = x_*math.tan(theta)+(y/(math.cos(theta)))
    print(fwhm2_gauss)
    print(fwhm_lorenz)

@nb.njit
def gaussian(x, A, x0, sigma):
    """Pojedyncza funkcja Gaussa."""
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

@nb.njit
def multi_gaussian(x, *params):
    """Suma wielu funkcji Gaussa."""
    y = np.zeros_like(x)
    n = len(params) // 3
    for i in range(n):
        A = params[3*i]
        x0 = params[3*i+1]
        sigma = params[3*i+2]
        y += gaussian(x, A, x0, sigma)
    return y

def plot_gauss(popt,peaks,count,fit_y,x,two_theta):
    for i in range(len(peaks)):
        A, x0, sigma = popt[3*i:3*i+3]
        fwhm = 2.355 * sigma
        print(f"Peak {i+1}: A={A:.3f}, x0={x0:.3f}, sigma={sigma:.3f}, FWHM={fwhm:.3f}")
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
        fit_seg = fit_y[mask]
        plt.figure(figsize=(8, 5))
        plt.plot(x_seg, y_seg, label='Dane (po baseline)', alpha=0.6)
        plt.plot(x_seg, fit_seg, lw=2, label='Suma Gaussów')
        # rysuj tylko te Gaussy, które leżą w tym przedziale
        for i in range(len(peaks)):
            Ai, x0i, si = popt[3*i:3*i+3]
            if seg_start <= x0i < seg_end:
                plt.plot(x_seg, gaussian(x_seg, Ai, x0i, si), '--', label=f'Gauss {i+1}')
        # tylko piki w tym przedziale
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

def peak_detect(count,x,heigh,dist,prom,two_theta,eps,theta_start,step,plotting = False):
    peaks, props = find_peaks(count, height=heigh, distance= dist , prominence= prom)
    # --- Przygotowanie zgadywań do dopasowania ---
    guesses = []
    bounds_lower = []
    bounds_upper = []
    for pk in peaks:
        A_guess = props['peak_heights'][np.where(peaks == pk)][0]
        x0_guess = x[pk]
        sigma_guess = 1.0
        guesses += [A_guess, x0_guess, sigma_guess]
        bounds_lower += [0.0, x0_guess - 5.0, 0.01]
        bounds_upper += [np.max(count) * 5, x0_guess + 5.0, 10.0]
    if len(guesses) == 0:
        return False
        # print("Nie znaleziono pików - spróbuj obniżyć progu 'height' lub 'prominence'.")
    else:
        # --- Dopasowanie sumy Gaussów ---
        popt, pcov = curve_fit(
            multi_gaussian, x, count,
            p0=guesses, bounds=(bounds_lower, bounds_upper), maxfev=20000
        )
        fit_y = multi_gaussian(x, *popt)
        residuals = count - fit_y
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((count - np.mean(count)) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"Znaleziono {len(peaks)} pików. R^2 dopasowania: {r2:.4f},wydajność = {len(peaks)/len(two_theta)}")
    correct_peaks = []
    for i in range(len(two_theta)):
        for j in range(len(peaks)):
            if abs(peaks[j] - ((two_theta[i]-theta_start)/step))<eps:
                correct_peaks.append(peaks[j])
    print(f"Znaleziono {len(peaks)} pików. R^2 dopasowania: {r2:.4f},wydajność = {len(correct_peaks)/len(two_theta)},eps = {eps}")
    if plotting == True:
        plot_gauss(popt,peaks,count,fit_y,x,two_theta)
     