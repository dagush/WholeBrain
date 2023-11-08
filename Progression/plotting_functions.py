# --------------------------------------------------------------------------------------
# Simulation of Alzheimer's disease progression
#
# By Christoffer Alexandersen
#
# [Alexandersen 2023] Alexandersen Christoffer G., de Haan Willem, Bick Christian and Goriely Alain (2023)
# A multi-scale model explains oscillatory slowing and neuronal hyperactivity in Alzheimer’s disease
# J. R. Soc. Interface
# https://doi.org/10.1098/rsif.2022.0607
#
# refactored by Gustavo Patow
# --------------------------------------------------------------------------------------
from math import log10
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


# --------------------------------------------------------------------------------------
# computes the power spectrum
# --------------------------------------------------------------------------------------
# def power_spectrum(y, t, plot=False):
#     y = np.array(y)
#     y = np.add(y, -np.mean(y))  # remove 0 frequency
#     ps = np.abs(np.fft.fft(y))**2
#     ps = ps
#     time_step = abs(t[1] - t[0])
#     freqs = np.fft.fftfreq(y.size, time_step)
#     idx = np.argsort(freqs)
#
#
#     if plot:
#         plt.plot(freqs[idx], ps[idx])
#         #plt.xscale('log')
#         #plt.yscale('log')
#         plt.xlim((-100, 100))
#         #plt.ylim((0, 0.1))
#         plt.show()
#
#     # find largest power
#     argmax = np.argmax(ps[idx])
#
#     return freqs[idx], ps[idx], ps[idx][argmax]


# --------------------------------------------------------------------------------------
# Compute the average power of the signal x in a specific frequency band
# taken from https://raphaelvallat.com/bandpower.html
# --------------------------------------------------------------------------------------
def bandpower(data, sf, band, window_sec=None, relative=False, modified=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch, periodogram
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Compute the (modified) periodogram
    if modified:
        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf
        freqs, psd = welch(data, sf, nperseg=nperseg)
    else:
        freqs, psd = periodogram(data, sf)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        glob_idx = np.logical_and(freqs >= 0, freqs <= 40)
        bp /= simps(psd[glob_idx], dx=freq_res)
    return bp


# --------------------------------------------------------------------------------------
# modified the above function (bandpower) to return spectrogram peaks
# --------------------------------------------------------------------------------------
def frequency_peaks(data, sf, band=None, window_sec=None, tol=10**-3, modified=False):
    """Compute the average power of the signal x in a specific frequency band. Modified to return spectrogram peaks

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    tol : float
        tolerance for ignoring maximum peak and set frequency to zero

    Return
    ------
    peak : float
        Largest PSD peak in frequency.
    """
    from scipy.signal import welch, periodogram
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Compute the (modified) periodogram
    if modified:
        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2/low) * sf
        freqs, psd = welch(data, sf, nperseg=nperseg)
    else:
        freqs, psd = periodogram(data, sf)

    # plot periodigram
    #plt.plot(freqs, psd)
    #plt.xlim([0, 14])
    #plt.show()

    # find peaks in psd
    if band.any():
        low, high = band
        filtered = np.array([i for i in range(len(freqs)) if (freqs[i] > low and freqs[i] < high)])
        psd = psd[filtered]
        freqs = freqs[filtered]

    max_peak = np.argmax(abs(psd))
    if max_peak is None or abs(psd[max_peak]) < tol:
        #freq_peak = 0
        freq_peak = float("NaN")
    else:
        freq_peak = freqs[max_peak]
    # we're done
    return freq_peak


# ---------------------------------------
# plot spreading
# Input
#   regions : list of lists, each list contains nodes to average over
# --------------------------------------
def plot_spreading(sol, colours, legends, xlimit=False, regions=[], averages=True, plot_c=False):
    # extract solution
    a = sol['a']
    b = sol['b']
    c = sol['c']
    qu = sol['qu']
    qv = sol['qv']
    u = sol['u']
    v = sol['v']
    up = sol['up']
    vp = sol['vp']
    w = sol['w']
    t = sol['t']

    # N of x-ticks
    nx = 5

    # find N
    N = a.shape[0]

    # if regions not given, plot all nodes
    if len(regions) == 0:
        regions = [[i] for i in range(N)]

    # plot 1-by-2 plot of all nodes'/regions a and b against time
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of all nodes'/regions tau and Abeta damage
    fig2, axs2 = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of all nodes'/regions toxic tau and Abeta concentration
    fig3, axs3 = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of average weight
    fig4, axs4 = plt.subplots(1, 1, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plt 1-by-2 plot of all nodes'/regions healthy tau and Abeta concentration
    fig5, axs5 = plt.subplots(1, 2, sharex=True, sharey=True)
    plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = nx) )
    sns.despine()

    # plot settings
    if xlimit:
        plt.xlim((0, xlimit))
    axs[0].set_xlabel('$t_{spread}$')
    axs[0].set_ylabel('Excitatory semiaxis, $a$')
    axs[1].set_xlabel('$t_{spread}$')
    axs[1].set_ylabel('Inhibitory semiaxis, $b$')

    axs2[0].set_xlabel('$t_{spread}$')
    axs2[0].set_ylabel('Amyloid-$\\beta$ damage, $q^{(\\beta)}$')
    axs2[1].set_xlabel('$t_{spread}$')
    axs2[1].set_ylabel('Tau damage, $q^{(\\tau)}$')
    axs2[0].set_ylim([-0.1, 1.1])
    axs2[1].set_ylim([-0.1, 1.1])

    axs3[0].set_xlabel('$t_{spread}$')
    axs3[0].set_ylabel('Toxic amyloid-$\\beta$ concentration, $\\tilde{u}$')
    axs3[1].set_xlabel('$t_{spread}$')
    axs3[1].set_ylabel('Toxic tau concentration, $\\tilde{v}$')

    axs5[0].set_xlabel('$t_{spread}$')
    axs5[0].set_ylabel('Healthy amyloid-$\\beta$ concentration, $u$')
    axs5[1].set_xlabel('$t_{spread}$')
    axs5[1].set_ylabel('Healthy tau concentration, $v$')

    # plot a, b, damage and concentrations against time
    for r in range(len(regions)):
        # initialize
        region = regions[r]
        avg_region_a = []
        avg_region_b = []
        avg_region_c = []
        avg_region_qu = []
        avg_region_qv = []
        avg_region_up = []
        avg_region_vp = []
        avg_region_u = []
        avg_region_v = []

        # compute averages over regions
        for node in region:
            avg_region_a.append(a[node,:])
            avg_region_b.append(b[node,:])
            avg_region_c.append(c[node,:])
            avg_region_qu.append(qu[node,:])
            avg_region_qv.append(qv[node,:])
            avg_region_up.append(up[node,:])
            avg_region_vp.append(vp[node,:])
            avg_region_u.append(u[node,:])
            avg_region_v.append(v[node,:])

        # convert lists to arrays
        avg_region_a = np.array(avg_region_a)
        avg_region_b = np.array(avg_region_b)
        avg_region_c = np.array(avg_region_c)
        avg_region_qu = np.array(avg_region_qu)
        avg_region_qv = np.array(avg_region_qv)
        avg_region_up = np.array(avg_region_up)
        avg_region_vp = np.array(avg_region_vp)
        avg_region_u = np.array(avg_region_u)
        avg_region_v = np.array(avg_region_v)

        # plot a, b
        axs[0].plot(t, np.mean(avg_region_a, axis=0), c=colours[r], label=legends[r])
        axs[1].plot(t, np.mean(avg_region_b, axis=0), c=colours[r], label=legends[r])
        if plot_c:
            axs[1].plot(t, np.mean(avg_region_c, axis=0), c=colours[r], label=legends[r])

        # plot damage
        axs2[0].plot(t, np.mean(avg_region_qu, axis=0), c=colours[r], label=legends[r])
        axs2[1].plot(t, np.mean(avg_region_qv, axis=0), c=colours[r], label=legends[r])

        # plot concentration
        axs3[0].plot(t, np.mean(avg_region_up, axis=0), c=colours[r], label=legends[r])
        axs3[1].plot(t, np.mean(avg_region_vp, axis=0), c=colours[r], label=legends[r])

        axs5[0].plot(t, np.mean(avg_region_u, axis=0), c=colours[r], label=legends[r])
        axs5[1].plot(t, np.mean(avg_region_v, axis=0), c=colours[r], label=legends[r])

    # plot averages over all nodes
    if averages:
        # a and b
        axs[0].plot(t, np.mean(a, axis=0), c='black', label='average')
        axs[1].plot(t, np.mean(b, axis=0), c='black', label='average')
        if plot_c:
            axs[1].plot(t, np.mean(c, axis=0), c='black', label='average')

        # damage
        axs2[0].plot(t, np.mean(qu, axis=0), c='black', label='average')
        axs2[1].plot(t, np.mean(qv, axis=0), c='black', label='average')

        # toxic concentratio
        axs3[0].plot(t, np.mean(up, axis=0), c='black', label='average')
        axs3[1].plot(t, np.mean(vp, axis=0), c='black', label='average')

        # healthy concentration
        axs5[0].plot(t, np.mean(u, axis=0), c='black', label='average')
        axs5[1].plot(t, np.mean(v, axis=0), c='black', label='average')


    # plot average weights over time
    axs4.plot(t, np.mean(w, axis=0), c='black')
    axs4.set_ylabel('Average link weight')
    axs4.set_xlabel('$t_{spread}$')

    # show
    axs[1].legend(loc='best')
    axs3[0].legend(loc='best')
    plt.tight_layout()

    # we're done
    figs = (fig, fig2, fig3, fig4, fig5)
    axss = (axs, axs2, axs3, axs4, axs5)
    return figs, axss


# -----------------------------------------------
# compute power-spectal properties
# -----------------------------------------------
def spectral_properties(solutions, bands, fourier_cutoff, modified=False,  # functional=False,
                        db=False, freq_tol=0,
                        relative=False, window_sec=None):
    # from mne.connectivity import spectral_connectivity

    # find size of rhythms
    _, x0, _ = solutions[0]
    L = x0.shape[0]
    N = x0.shape[1]
    len_rhythms = len(solutions)

    # find average power (and frequency peaks) in bands for each node over time stamps
    bandpowers = [[[[] for _ in range(len_rhythms)] for _ in range(N)] for _ in range(len(bands))]
    freq_peaks = [[[[] for _ in range(len_rhythms)] for _ in range(N)] for _ in range(len(bands))]

    # functional connectivity parameters and initializations
    # if functional:
    #     functional_methods = ['coh', 'pli', 'plv']
    #     average_strengths = [[[[] for _ in range(len_rhythms)] for _ in range(len(functional_methods))] for _ in
    #                          range(len(bands))]

    for b in range(len(bands)):
        for i in range((len_rhythms)):
            t, x, y = solutions[i]

            # iterate through trials
            for l in range(L):
                xl = x[l]

                # find last 10 seconds
                inds = [s for s in range(len(t)) if t[s] > fourier_cutoff]
                t = t[inds]
                x_cut = xl[:, inds]
                tot_t = t[-1] - t[0]
                sf = len(x_cut[0]) / tot_t

                # compute spectral connectivity
                # if functional:
                #     functional_connectivity = spectral_connectivity([x_cut], method=functional_methods, sfreq=sf,
                #                                                     fmin=bands[b][0], fmax=bands[b][1], mode='fourier',
                #                                                     faverage=True, verbose=False)
                #     # get average link strength
                #     for j in range(len(functional_methods)):
                #         functional_matrix = functional_connectivity[0][j]  # lower triangular
                #         n_rows, n_cols, _ = functional_matrix.shape
                #
                #         # compute average strength
                #         average_strength = 0
                #         for c in range(n_cols):
                #             for r in range(c + 1, n_cols):
                #                 average_strength += functional_matrix[r, c][0]
                #         average_strength /= N * (N - 1) / 2
                #
                #         # append
                #         average_strengths[b][j][i].append(average_strength)

                # find PSD and peak
                for j in range(N):
                    # PSD
                    bandpower_t = bandpower(x_cut[j], sf, bands[b], modified=modified, relative=relative,
                                            window_sec=window_sec)
                    if db:
                        bandpower_t = 10 * log10(bandpower_t)
                    bandpowers[b][j][i].append(bandpower_t)

                    # frequency peaks
                    freq_peak_t = frequency_peaks(x_cut[j, :], sf, band=bands[b], tol=freq_tol, modified=modified,
                                                  window_sec=window_sec)
                    freq_peaks[b][j][i].append(freq_peak_t)

    # package return value
    spectral_props = [bandpowers, freq_peaks]
    # if functional:
    #     spectral_props.append(average_strengths)

    return spectral_props


# ---------------------------------------------
# plot spectral properties
# -------------------------------------------
def plot_spectral_properties(t_stamps, bandpowers, freq_peaks, bands, wiggle, title, legends, colours,
                             bandpower_ylim=False, only_average=False, regions=[], n_ticks=5, relative=False):
    # find N and length of rhythms
    N = len(bandpowers[0])
    L = len(bandpowers[0][0][0])
    len_rhythms = len(bandpowers[0][0])

    # initialize
    figs_PSD = []
    figs_peaks = []
    axs_PSD = []
    axs_peaks = []
    for b in bands:
        fig_PSD = plt.figure()  # PSD
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
        figs_PSD.append(fig_PSD)

        fig_peaks = plt.figure()  # peaks
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
        # plt.xticks(np.arange(0, 30+1, 10))
        figs_peaks.append(fig_peaks)

    #  wiggle points in x-direction
    if regions:
        wiggle = (t_stamps[-1] - t_stamps[0]) * wiggle / (2 * len(regions))
        wiggled = [np.array(t_stamps) + (i - len(regions) / 2) * wiggle for i in range(len(regions) + 1)]
    else:
        wiggle = (t_stamps[-1] - t_stamps[0]) * wiggle / (2 * N)
        wiggled = [np.array(t_stamps) + (i - N / 2) * wiggle for i in range(N + 1)]

    # plot average power versus timestamps (one pipeline for regional input, one without)
    for b in range(len(bands)):
        if regions:
            # compute average over regions, and variance of region average over trials
            avg_powers = [[0 for _ in range(L)] for _ in range(len(t_stamps))]
            avg_peaks = [[0 for _ in range(L)] for _ in range(len(t_stamps))]

            for r in range(len(regions)):
                region = regions[r]
                powers = [[0 for _ in range(L)] for _ in range(len(t_stamps))]
                peaks = [[0 for _ in range(L)] for _ in range(len(t_stamps))]
                for ts in range(len(t_stamps)):
                    for l in range(L):
                        zero_peaks = 0
                        for node in region:
                            powers[ts][l] += bandpowers[b][node][ts][l]
                            node_peak = freq_peaks[b][node][ts][l]
                            if not np.isnan(node_peak):
                                peaks[ts][l] += freq_peaks[b][node][ts][l]
                            else:
                                zero_peaks += 1

                        powers[ts][l] = powers[ts][l] / len(region)
                        if zero_peaks < len(region) / 4:
                            peaks[ts][l] = peaks[ts][l] / (len(region) - zero_peaks)
                        else:
                            peaks[ts][l] = float("NaN")

                        # compute average over entire brain
                        if r == 0:
                            # average of trial
                            zero_peaks = 0
                            for n in range(N):
                                avg_powers[ts][l] += bandpowers[b][n][ts][l]

                                node_peak = freq_peaks[b][n][ts][l]
                                if not np.isnan(node_peak):
                                    avg_peaks[ts][l] += freq_peaks[b][n][ts][l]
                                else:
                                    zero_peaks += 1
                            avg_powers[ts][l] /= N
                            if zero_peaks < N / 4:
                                avg_peaks[ts][l] /= N - zero_peaks
                            else:
                                avg_peaks[ts][l] = float("NaN")

                # plot regions
                if not only_average:
                    # plot power
                    mean = np.mean(powers, axis=1)
                    std = np.std(powers, axis=1)
                    sns.despine()  # remove right and upper axis line

                    # plot peaks
                    mean = np.mean(peaks, axis=1)
                    std = np.std(peaks, axis=1)
                    sns.despine()  # remove right and upper axis line

                    figs_PSD[b].axes[0].spines['right'].set_visible(False)
                    figs_PSD[b].axes[0].spines['top'].set_visible(False)
                    figs_PSD[b].axes[0].errorbar(wiggled[r], np.mean(powers, axis=1), c=colours[r], \
                                                 label=legends[r], marker='o', linestyle='--', alpha=0.75,
                                                 yerr=np.std(powers, axis=1), capsize=6, capthick=2)
                    sns.despine()
                    figs_peaks[b].axes[0].errorbar(wiggled[r], np.nanmean(peaks, axis=1), c=colours[r], \
                                                   label=legends[r], marker='o', linestyle='--', alpha=0.75,
                                                   yerr=np.std(peaks, axis=1), capsize=6, capthick=2)
                    sns.despine()

            # plot average
            # global power
            mean = np.mean(avg_powers, axis=1)
            std = np.std(avg_powers, axis=1)
            # sns.despine()  # remove right and upper axis line

            # global peaks
            mean = np.mean(avg_peaks, axis=1)
            std = np.std(avg_peaks, axis=1)
            # sns.despine()  # remove right and upper axis line

            figs_PSD[b].axes[0].errorbar(wiggled[-1], np.mean(avg_powers, axis=1), c='black', \
                                         label='average', marker='o', linestyle='--', alpha=0.75,
                                         yerr=np.std(powers, axis=1), capsize=6, capthick=2)
            sns.despine()
            figs_peaks[b].axes[0].errorbar(wiggled[-1], np.nanmean(avg_peaks, axis=1), c='black', \
                                           label='average', marker='o', linestyle='--', alpha=0.75,
                                           yerr=np.std(peaks, axis=1), capsize=6, capthick=2)
            sns.despine()

        else:
            avg_power = np.array([[None for _ in range(N)] for _ in range(len_rhythms)])
            avg_peak = np.array([[None for _ in range(N)] for _ in range(len_rhythms)])
            for i in range(N):
                # power
                power = bandpowers[b][i]
                avg_power[:, i] = np.mean(power, axis=1)
                if not only_average:
                    figs_PSD[b].axes[0].errorbar(wiggled[i], np.mean(power, axis=1), c=colours[i], label=legends[i],
                                                 marker='o', linestyle='--', alpha=0.75, yerr=np.std(power, axis=1),
                                                 capsize=6, capthick=2)

                # peak
                peak = freq_peaks[b][i]
                avg_peak[:, i] = np.mean(peak, axis=1)
                if not only_average:
                    figs_peaks[b].axes[0].errorbar(wiggled[i], np.nanmean(peak, axis=1), c=colours[i], label=legends[i],
                                                   marker='o', linestyle='--', alpha=0.75, yerr=np.std(peak, axis=1),
                                                   capsize=6, capthick=2)

            # average power/peak over nodes
            avg_power = np.array(avg_power, dtype=np.float64)  # not included -> error due to sympy float values
            figs_PSD[b].axes[0].errorbar(t_stamps, np.mean(avg_power, axis=1), c='black', label='Node average',
                                         marker='o', linestyle='--', alpha=0.75, yerr=np.std(avg_power, axis=1),
                                         capsize=6, capthick=2)

            avg_peak = np.array(avg_peak, dtype=np.float64)  # not included -> error due to sympy float values
            figs_peaks[b].axes[0].errorbar(t_stamps, np.nanmean(avg_peak, axis=1), c='black', label='Node average',
                                           marker='o', linestyle='--', alpha=0.75, yerr=np.std(avg_peak, axis=1),
                                           capsize=6, capthick=2)

    # set labels
    for b in range(len(bands)):
        # power
        figs_PSD[b].axes[0].set_title(title)
        if relative:
            figs_PSD[b].axes[0].set_ylabel(f'Relative power (${bands[b][0]} - {bands[b][1]}$ Hz)')
        else:
            figs_PSD[b].axes[0].set_ylabel(f'Absolute power (${bands[b][0]} - {bands[b][1]}$ Hz)')
        # figs_PSD[b].axes[0].set_xlabel('Speading time (years)')
        figs_PSD[b].axes[0].set_xlabel(f'$t_{{spread}}$')
        figs_PSD[b].axes[0].set_xlim([np.amin(wiggled) - wiggle, np.amax(wiggled) + wiggle])

        if bandpower_ylim:
            figs_PSD[b].axes[0].set_ylim([-0.05, bandpower_ylim])

        # peak
        figs_peaks[b].axes[0].set_title(title)
        figs_peaks[b].axes[0].set_ylabel(f'Peak frequency (${bands[b][0]} - {bands[b][1]}$ Hz)')
        # figs_peaks[b].axes[0].set_xlabel('Spreading time (years)')
        figs_peaks[b].axes[0].set_xlabel(f'$t_{{spread}}$')
        figs_peaks[b].axes[0].set_xlim([np.amin(wiggled) - wiggle, np.amax(wiggled) + wiggle])
        figs_peaks[b].axes[0].set_ylim([bands[b][0] - 0.5, bands[b][-1] + 0.5])
        figs_peaks[b].axes[0].set_ylim([8, 11])

        # legends
        figs_PSD[b].axes[0].legend()
        plt.tight_layout()
        figs_peaks[b].axes[0].legend()
        plt.tight_layout()

    # we're done
    return figs_PSD, figs_peaks


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF