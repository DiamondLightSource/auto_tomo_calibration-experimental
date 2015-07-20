import pylab as pl
import numpy as np

import math
from math import pi, log

from scipy import optimize
from scipy.signal import argrelextrema
from scipy.integrate import simps
from scipy import fft, ifft
from scipy.optimize import curve_fit, leastsq


def gaussian(x, height, center, width, offset=0):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset
    
    
def two_gaussians(x, h1, c1, w1, h2, c2, w2, offset):
    return (gaussian(x, h1, c1, w1, offset) + 
            gaussian(x, h2, c2, w2, offset) + offset)


def modulation_transfer_function(lsf):
    """
    Takes in an array of gaussians from a slice
    of the sphere and calculates the fft of each line spread function.
    This is the MTF of that point. Adding the areas bellow the MTF,
    a number corresponding to relative image quality is obtained 
    """
    # normalize
    lsf_norm = []
    lsf_areas = []
    for item in lsf:
        try:
            lsf_norm.append(item / simps(item))
            lsf_areas.append(simps(item))
#             lsf_norm.append(item / max(item))
        except:
            pass
    
    lsf_avg_area = np.mean(lsf_areas)
    
#     # get the fft of the ideal system i.e.
#     # lsf is a delta function
#     centres = [int(len(centre) / 2.0) for centre in lsf_norm]
#     empty_lists = [np.zeros(len(item)) for item in lsf_norm]
#     
#     # get the ABSOLUTE fft value of each line spread function
#     measured_fft = [np.fft.fftshift(np.fft.fft(item[centres[i]-10:centres[i]+10])) for i, item in enumerate(lsf_norm)]
# 
#     
#     dirac_deltas = []
#     for i in range(len(empty_lists)):
#         ideal_lsf = empty_lists[i]
#         ideal_lsf[centres[i]] = lsf_norm[i][centres[i]]
#         dirac_deltas.append(ideal_lsf)
#      
#     ideal_fft = [np.fft.fftshift(np.fft.fft(item)) for item in dirac_deltas]
#      
#     # Object = Image * Impulse response
#     # F[obj] = F[Img] . F[Impulse] = F[img] . MTF
#     # MTF = F[obj] / F[img]
#     mtfs = []
#     for i in range(len(ideal_fft)):
# #         mtf = np.asarray(np.divide(measured_fft[i], ideal_fft[i]))
#         mtf = abs(measured_fft[i])
#         #normalize
# #         mtf = mtf / simps(mtf)
# #         mtf = mtf / max(mtf)
#         mtfs.append(mtf)
#         
# #     nyquist = int(len(mtf) / 4.)
# #     pl.plot(mtfs[0][centres[0]:centres[0] + nyquist])
#     pl.plot(mtfs[0])
#     pl.title("Mtf using division; centered")
#     pl.show()
# #     get the impulse response function
# #     impulse_response = [np.fft.ifft(np.fft.ifftshift(item)) for item in measured_fft]
# #     # center it at zero
# #     pl.plot(impulse_response[0][centres[0]:])
# #     pl.title("Impulse response; centered")
# #     pl.show()
#     
#     pl.plot(lsf_norm[0])
#     pl.title("Original normalized LSF")
#     pl.show()
# 
#     
#     # get a single average value used for comparison of images
#     #print "mean mtf area", np.mean(mtfs)
    
    return lsf_avg_area


def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise (ValueError, 
                'Input vectors y_axis and x_axis must have same length')
    
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis
    

def _peakdetect_parabole_fitter(raw_peaks, x_axis, y_axis, points):
    """
    Performs the actual parabole fitting for the peakdetect_parabole function.
    
    keyword arguments:
    raw_peaks -- A list of either the maximium or the minimum peaks, as given
        by the peakdetect_zero_crossing function, with index used as x-axis
    x_axis -- A numpy list of all the x values
    y_axis -- A numpy list of all the y values
    points -- How many points around the peak should be used during curve
        fitting, must be odd.
    
    return -- A list giving all the peaks and the fitted waveform, format:
        [[x, y, [fitted_x, fitted_y]]]
        
    """
    func = lambda x, k, tau, m: k * ((x - tau) ** 2) + m
    fitted_peaks = []
    for peak in raw_peaks:
        index = peak[0]
        x_data = x_axis[index - points // 2: index + points // 2 + 1]
        y_data = y_axis[index - points // 2: index + points // 2 + 1]
        # get a first approximation of tau (peak position in time)
        tau = x_axis[index]
        # get a first approximation of peak amplitude
        m = peak[1]
        
        # build list of approximations
        # k = -m as first approximation?
        p0 = (-m, tau, m)
        popt, pcov = curve_fit(func, x_data, y_data, p0)
        # retrieve tau and m i.e x and y value of peak
        x, y = popt[1:3]
        
        # create a high resolution data set for the fitted waveform
        x2 = np.linspace(x_data[0], x_data[-1], points * 10)
        y2 = func(x2, *popt)
        
        fitted_peaks.append([x, y, [x2, y2]])
        
    return fitted_peaks
    
    
def peakdetect(y_axis, x_axis = None, lookahead = 300, delta=0):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200) 
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    if lookahead < 1:
        raise ValueError, "Lookahead must be '1' or above in value"
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError, "delta must be a positive number"
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
        
    return [max_peaks, min_peaks]
    
    
def peakdetect_fft(y_axis, x_axis, pad_len = 5):
    """
    Performs a FFT calculation on the data and zero-pads the results to
    increase the time domain resolution after performing the inverse fft and
    send the data to the 'peakdetect' function for peak 
    detection.
    
    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as the index 50.234 or similar.
    
    Will find at least 1 less peak then the 'peakdetect_zero_crossing'
    function, but should result in a more precise value of the peak as
    resolution has been increased. Some peaks are lost in an attempt to
    minimize spectral leakage by calculating the fft between two zero
    crossings for n amount of signal periods.
    
    The biggest time eater in this function is the ifft and thereafter it's
    the 'peakdetect' function which takes only half the time of the ifft.
    Speed improvementd could include to check if 2**n points could be used for
    fft and ifft or change the 'peakdetect' to the 'peakdetect_zero_crossing',
    which is maybe 10 times faster than 'peakdetct'. The pro of 'peakdetect'
    is that it resutls in one less lost peak. It should also be noted that the
    time used by the ifft function can change greatly depending on the input.
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the postion of the peaks.
    pad_len -- (optional) By how many times the time resolution should be
        increased by, e.g. 1 doubles the resolution. The amount is rounded up
        to the nearest 2 ** n amount (default: 5)
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*tab)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    zero_indices = zero_crossings(y_axis, window = 11)
    #select a n amount of periods
    last_indice = - 1 - (1 - len(zero_indices) & 1)
    # Calculate the fft between the first and last zero crossing
    # this method could be ignored if the begining and the end of the signal
    # are discardable as any errors induced from not using whole periods
    # should mainly manifest in the beginning and the end of the signal, but
    # not in the rest of the signal
    fft_data = fft(y_axis[zero_indices[0]:zero_indices[last_indice]])
    padd = lambda x, c: x[:len(x) // 2] + [0] * c + x[len(x) // 2:]
    n = lambda x: int(log(x)/log(2)) + 1
    # padds to 2**n amount of samples
    fft_padded = padd(list(fft_data), 2 ** 
                n(len(fft_data) * pad_len) - len(fft_data))
    
    # There is amplitude decrease directly proportional to the sample increase
    sf = len(fft_padded) / float(len(fft_data))
    # There might be a leakage giving the result an imaginary component
    # Return only the real component
    y_axis_ifft = ifft(fft_padded).real * sf #(pad_len + 1)
    x_axis_ifft = np.linspace(
                x_axis[zero_indices[0]], x_axis[zero_indices[last_indice]],
                len(y_axis_ifft))
    # get the peaks to the interpolated waveform
    max_peaks, min_peaks = peakdetect(y_axis_ifft, x_axis_ifft, 500,
                                    delta = abs(np.diff(y_axis).max() * 2))
    #max_peaks, min_peaks = peakdetect_zero_crossing(y_axis_ifft, x_axis_ifft)
    
    # store one 20th of a period as waveform data
    data_len = int(np.diff(zero_indices).mean()) / 10
    data_len += 1 - data_len & 1
    
    
    fitted_wave = []
    for peaks in [max_peaks, min_peaks]:
        peak_fit_tmp = []
        index = 0
        for peak in peaks:
            index = np.where(x_axis_ifft[index:]==peak[0])[0][0] + index
            x_fit_lim = x_axis_ifft[index - data_len // 2:
                                    index + data_len // 2 + 1]
            y_fit_lim = y_axis_ifft[index - data_len // 2:
                                    index + data_len // 2 + 1]
            
            peak_fit_tmp.append([x_fit_lim, y_fit_lim])
        fitted_wave.append(peak_fit_tmp)
    
    #pylab.plot(range(len(fft_data)), fft_data)
    #pylab.show()
    
    pl.plot(x_axis, y_axis)
    pl.hold(True)
    pl.plot(x_axis_ifft, y_axis_ifft)
    #for max_p in max_peaks:
    #    pylab.plot(max_p[0], max_p[1], 'xr')
    pl.show()
    return [max_peaks, min_peaks]
    
    
def peakdetect_parabole(y_axis, x_axis, points = 9):
    """
    Function for detecting local maximas and minmias in a signal.
    Discovers peaks by fitting the model function: y = k (x - tau) ** 2 + m
    to the peaks. The amount of points used in the fitting is set by the
    points argument.
    
    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as index 50.234 or similar.
    
    will find the same amount of peaks as the 'peakdetect_zero_crossing'
    function, but might result in a more precise value of the peak.
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the postion of the peaks.
    points -- (optional) How many points around the peak should be used during
        curve fitting, must be odd (default: 9)
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a list
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # make the points argument odd
    points += 1 - points % 2
    #points += 1 - int(points) & 1 slower when int conversion needed
    
    # get raw peaks
    max_raw, min_raw = peakdetect_zero_crossing(y_axis)
    
    # define output variable
    max_peaks = []
    min_peaks = []
    
    max_ = _peakdetect_parabole_fitter(max_raw, x_axis, y_axis, points)
    min_ = _peakdetect_parabole_fitter(min_raw, x_axis, y_axis, points)
    
    max_peaks = map(lambda x: [x[0], x[1]], max_)
    max_fitted = map(lambda x: x[-1], max_)
    min_peaks = map(lambda x: [x[0], x[1]], min_)
    min_fitted = map(lambda x: x[-1], min_)
    
    
    #pylab.plot(x_axis, y_axis)
    #pylab.hold(True)
    #for max_p, max_f in zip(max_peaks, max_fitted):
    #    pylab.plot(max_p[0], max_p[1], 'x')
    #    pylab.plot(max_f[0], max_f[1], 'o', markersize = 2)
    #for min_p, min_f in zip(min_peaks, min_fitted):
    #    pylab.plot(min_p[0], min_p[1], 'x')
    #    pylab.plot(min_f[0], min_f[1], 'o', markersize = 2)
    #pylab.show()
    
    return [max_peaks, min_peaks]
    
 
def peakdetect_sine(y_axis, x_axis, points = 9, lock_frequency = False):
    """
    Function for detecting local maximas and minmias in a signal.
    Discovers peaks by fitting the model function:
    y = A * sin(2 * pi * f * x - tau) to the peaks. The amount of points used
    in the fitting is set by the points argument.
    
    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as index 50.234 or similar.
    
    will find the same amount of peaks as the 'peakdetect_zero_crossing'
    function, but might result in a more precise value of the peak.
    
    The function might have some problems if the sine wave has a
    non-negligible total angle i.e. a k*x component, as this messes with the
    internal offset calculation of the peaks, might be fixed by fitting a 
    k * x + m function to the peaks for offset calculation.
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the postion of the peaks.
    points -- (optional) How many points around the peak should be used during
        curve fitting, must be odd (default: 9)
    lock_frequency -- (optional) Specifies if the frequency argument of the
        model function should be locked to the value calculated from the raw
        peaks or if optimization process may tinker with it. (default: False)
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*tab)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # make the points argument odd
    points += 1 - points % 2
    #points += 1 - int(points) & 1 slower when int conversion needed
    
    # get raw peaks
    max_raw, min_raw = peakdetect_zero_crossing(y_axis)
    
    # define output variable
    max_peaks = []
    min_peaks = []
    
    # get global offset
    offset = np.mean([np.mean(max_raw, 0)[1], np.mean(min_raw, 0)[1]])
    # fitting a k * x + m function to the peaks might be better
    #offset_func = lambda x, k, m: k * x + m
    
    # calculate an approximate frequenzy of the signal
    Hz = []
    for raw in [max_raw, min_raw]:
        if len(raw) > 1:
            peak_pos = [x_axis[index] for index in zip(*raw)[0]]
            Hz.append(np.mean(np.diff(peak_pos)))
    Hz = 1 / np.mean(Hz)
    
    # model function
    # if cosine is used then tau could equal the x position of the peak
    # if sine were to be used then tau would be the first zero crossing
    if lock_frequency:
        func = lambda x, A, tau: A * np.sin(2 * pi * Hz * (x - tau) + pi / 2)
    else:
        func = lambda x, A, Hz, tau: A * np.sin(2 * pi * Hz * (x - tau) + 
                                                pi / 2)
    #func = lambda x, A, Hz, tau: A * np.cos(2 * pi * Hz * (x - tau))
    
    
    #get peaks
    fitted_peaks = []
    for raw_peaks in [max_raw, min_raw]:
        peak_data = []
        for peak in raw_peaks:
            index = peak[0]
            x_data = x_axis[index - points // 2: index + points // 2 + 1]
            y_data = y_axis[index - points // 2: index + points // 2 + 1]
            # get a first approximation of tau (peak position in time)
            tau = x_axis[index]
            # get a first approximation of peak amplitude
            A = peak[1]
            
            # build list of approximations
            if lock_frequency:
                p0 = (A, tau)
            else:
                p0 = (A, Hz, tau)
            
            # subtract offset from waveshape
            y_data -= offset
            popt, pcov = curve_fit(func, x_data, y_data, p0)
            # retrieve tau and A i.e x and y value of peak
            x = popt[-1]
            y = popt[0]
            
            # create a high resolution data set for the fitted waveform
            x2 = np.linspace(x_data[0], x_data[-1], points * 10)
            y2 = func(x2, *popt)
            
            # add the offset to the results
            y += offset
            y2 += offset
            y_data += offset
            
            peak_data.append([x, y, [x2, y2]])
       
        fitted_peaks.append(peak_data)
    
    # structure date for output
    max_peaks = map(lambda x: [x[0], x[1]], fitted_peaks[0])
    max_fitted = map(lambda x: x[-1], fitted_peaks[0])
    min_peaks = map(lambda x: [x[0], x[1]], fitted_peaks[1])
    min_fitted = map(lambda x: x[-1], fitted_peaks[1])
    
    
    #pylab.plot(x_axis, y_axis)
    #pylab.hold(True)
    #for max_p, max_f in zip(max_peaks, max_fitted):
    #    pylab.plot(max_p[0], max_p[1], 'x')
    #    pylab.plot(max_f[0], max_f[1], 'o', markersize = 2)
    #for min_p, min_f in zip(min_peaks, min_fitted):
    #    pylab.plot(min_p[0], min_p[1], 'x')
    #    pylab.plot(min_f[0], min_f[1], 'o', markersize = 2)
    #pylab.show()
    
    return [max_peaks, min_peaks]
 
    
def peakdetect_sine_locked(y_axis, x_axis, points = 9):
    """
    Convinience function for calling the 'peakdetect_sine' function with
    the lock_frequency argument as True.
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the postion of the peaks.
    points -- (optional) How many points around the peak should be used during
        curve fitting, must be odd (default: 9)
        
    return -- see 'peakdetect_sine'
    """
    return peakdetect_sine(y_axis, x_axis, points, True)
    
    
def peakdetect_zero_crossing(y_axis, x_axis = None, window = 11):
    """
    Function for detecting local maximas and minmias in a signal.
    Discovers peaks by dividing the signal into bins and retrieving the
    maximum and minimum value of each the even and odd bins respectively.
    Division into bins is performed by smoothing the curve and finding the
    zero crossings.
    
    Suitable for repeatable signals, where some noise is tolerated. Excecutes
    faster than 'peakdetect', although this function will break if the offset
    of the signal is too large. It should also be noted that the first and
    last peak will probably not be found, as this function only can find peaks
    between the first and last zero crossing.
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    window -- the dimension of the smoothing window; should be an odd integer
        (default: 11)
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*tab)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    
    zero_indices = zero_crossings(y_axis, window = window)
    period_lengths = np.diff(zero_indices)
            
    bins_y = [y_axis[index:index + diff] for index, diff in 
        zip(zero_indices, period_lengths)]
    bins_x = [x_axis[index:index + diff] for index, diff in 
        zip(zero_indices, period_lengths)]
        
    even_bins_y = bins_y[::2]
    odd_bins_y = bins_y[1::2]
    even_bins_x = bins_x[::2]
    odd_bins_x = bins_x[1::2]
    hi_peaks_x = []
    lo_peaks_x = []
    
    #check if even bin contains maxima
    if abs(even_bins_y[0].max()) > abs(even_bins_y[0].min()):
        hi_peaks = [bin.max() for bin in even_bins_y]
        lo_peaks = [bin.min() for bin in odd_bins_y]
        # get x values for peak
        for bin_x, bin_y, peak in zip(even_bins_x, even_bins_y, hi_peaks):
            hi_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])
        for bin_x, bin_y, peak in zip(odd_bins_x, odd_bins_y, lo_peaks):
            lo_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])
    else:
        hi_peaks = [bin.max() for bin in odd_bins_y]
        lo_peaks = [bin.min() for bin in even_bins_y]
        # get x values for peak
        for bin_x, bin_y, peak in zip(odd_bins_x, odd_bins_y, hi_peaks):
            hi_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])
        for bin_x, bin_y, peak in zip(even_bins_x, even_bins_y, lo_peaks):
            lo_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])
    
    max_peaks = [[x, y] for x,y in zip(hi_peaks_x, hi_peaks)]
    min_peaks = [[x, y] for x,y in zip(lo_peaks_x, lo_peaks)]
    
    return [max_peaks, min_peaks]
        
    
def _smooth(x, window_len=11, window='hanning'):
    """
    smooth the data using a window of the requested size.
    
    This method is based on the convolution of a scaled window on the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd 
            integer
        window: the type of window from 'flat', 'hanning', 'hamming', 
            'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t = linspace(-2,2,0.1)
    x = sin(t)+randn(len(t))*0.1
    y = _smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, 
    numpy.convolve, scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if a list instead of
    a string   
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
 
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    
    if window_len<3:
        return x
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError,
            "Window is not one of '{0}', '{1}', '{2}', '{3}', '{4}'".format(
            *('flat', 'hanning', 'hamming', 'bartlett', 'blackman')))
    
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.' + window + '(window_len)')
 
    y = np.convolve(w / w.sum(), s, mode = 'valid')
    return y
    
    
def zero_crossings(y_axis, window = 11):
    """
    Algorithm to find zero crossings. Smoothens the curve and finds the
    zero-crossings by looking for a sign change.
    
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find zero-crossings
    window -- the dimension of the smoothing window; should be an odd integer
        (default: 11)
    
    return -- the index for each zero-crossing
    """
    # smooth the curve
    length = len(y_axis)
    x_axis = np.asarray(range(length), int)
    
    # discard tail of smoothed signal
    y_axis = _smooth(y_axis, window)[:length]
    zero_crossings = np.where(np.diff(np.sign(y_axis)))[0]
    indices = [x_axis[index] for index in zero_crossings]
    
    # check if zero-crossings are valid
    diff = np.diff(indices)
    if diff.std() / diff.mean() > 0.2:
        print diff.std() / diff.mean()
        print np.diff(indices)
        raise(ValueError, 
            "False zero-crossings found, indicates problem {0} or {1}".format(
            "with smoothing window", "problem with offset"))
    # check if any zero crossings were found
    if len(zero_crossings) < 1:
        raise(ValueError, "No zero crossings found")
    
    return indices 


def rms_noise_image(image):
    """
    Find the rms of noise
    
    TODO: use non local means for the noise
    """
    
    Xc = int(image.shape[0] / 2.)
    Yc = int(image.shape[1] / 2.)
    
    # Denoise the image
    
    # Get values of pixels according to an increasing radius for a same angle
    # Get the radius to be at most half the image since the circle can't be bigger
    R = image.shape[0] / 4. 
    
    
    
    # Simple trig identities
    # R is the max value that we can reach
    delta_x = R * math.sin(0)
    delta_y = R * math.cos(0)
    points = []
    # Go from 0 to 1.001 in steps of 0.001
    for alpha in np.arange(0, 1.001, 0.001):
            # Xc and Yc are the positions from the center
            # points stores all the points from the center going along the radius
            points.append(image[int(Xc + alpha * delta_x), int(Yc + alpha * delta_y)])

    # peak to peak value
    ptp = np.max(points) - np.min(points)
    
    rms_sq = [item**2 for item in points]
    rms_noise = np.sqrt(np.sum(rms_sq) / len(rms_sq))

    return rms_noise


def rms_noise_signal(signal):
    """
    Find the rms of noise
    
    TODO: use non local means for the noise
    """
    rms_sq = [item**2 for item in signal]
    rms_noise = np.sqrt(np.sum(rms_sq) / len(rms_sq))

    return rms_noise


def find_contact(points, theta, noiz):
    """
    Finds the max intensity along the line
    Compares it with the background intensity
    If it is within the RMS of noise, then
    that point indicates an edge
    """
    max_int = np.max(points)
    
    angle = None
    if max_int <= noiz:
        angle = theta  * 180. / math.pi
        
    return max_int

def get_radius(image, theta, phi, centre, rad_min, sigma):

    pl.close('all')
    
    # Plot value of pixel as a function of radius
    Xc = centre[0]
    Yc = centre[1]
    Zc = centre[2]
    
    R = min(image.shape[0] / 2, image.shape[1] / 2, image.shape[2] / 2) - 1
    
    delta_x = R * np.sin(phi) * np.cos(theta)
    delta_y = R * np.sin(phi) * np.sin(theta)
    delta_z = R * np.cos(phi)
    
    points = []
    
    step = 0.001
    
    for alpha in np.arange(0,1 + step, step):
        if np.sqrt((alpha*delta_x)**2 + (alpha*delta_y)**2 + (alpha*delta_z)**2) > rad_min:
            points.append( image[int(Xc + alpha * delta_x), int(Yc + alpha * delta_y), int(Zc + alpha * delta_z)] )
    
    # calculate the rms of the noise and use it
    # to filter the signal
    
    # normalize
    points = points / np.max(points)
    
    rms_noise5 = rms_noise_signal(points)
    peaks = peakdetect(points / np.max(points),
                       lookahead = 15 * sigma, delta = rms_noise5)
    
    contact_int = find_contact(points, theta, rms_noise5 / 5)
    ################################################################
    #                 FAIL PROOF DETECOTR (?)
    # if the margins in the selector are changed then
    # change the 1.3 factor
    ################################################################
    try:
        centre_guess1 = peaks[0][0][0]
        print centre_guess1
    except:
        centre_guess1 = image.shape[1] / 1.2 - rad_min
    
    
    try:
        centre_guess2 = peaks[1][0][0]
    except:
        try:
            # select the values
            stdev2 = np.std(points) * 2
            mean_pts = np.mean(points)
            bin_pts = points >= mean_pts + stdev2
            dif = np.diff(points[int(centre_guess1):])
            sep = np.argwhere(bin_pts == 1)
            
            centre_guess2 = sep[-1][0]
            #print "guess2 from first exception"
        except:
            centre_guess2 = centre_guess1
            #print "guess2 from second exception"
    
    # distance between peaks
    offset_guess = centre_guess2 - centre_guess1
    # width of the gaussian
    width_guess = 10.
    
    # the initial guesses for the Gaussians
    # for 2: height, centre, width, height, centre, width, offset
    # for 1: height, centre, width, offset
    guess2 = [0., centre_guess1, width_guess, 0., centre_guess2, width_guess, offset_guess]
    guess1 = [0., centre_guess1, width_guess, 0]
    
    # make the array into the correct format
    data = np.array([range(len(points)), points]).T
    
    # the functions to be minimized
    errfunc2 = lambda p, xdata, ydata: (two_gaussians(xdata, *p) - ydata)
    errfunc1 = lambda p, xdata, ydata: (gaussian(xdata, *p) - ydata)
 
    # perform the least square minimization
    optim2, success1 = optimize.leastsq(errfunc2, guess2[:], args=(data[:,0], data[:,1]))
    optim1, success2 = optimize.leastsq(errfunc1, guess1[:], args=(data[:,0], data[:,1]))
    
    # sometimes values are negative for some weird reason
    optim2 = abs(optim2)
    optim1 = abs(optim1)
    
#     if theta >= 290 / 180. * 3.14 :
#         pl.plot(data[:,0], data[:,1], lw=5, c='g', label='measurement')
#         pl.plot(data[:,0], gaussian(data[:,0], *optim1),
#             lw=3, c='b', label='fit of 1 Gaussian')
#         pl.plot(data[:,0], two_gaussians(data[:,0], *optim2),
#             lw=1, c='r', ls='--', label='fit of 2 Gaussians')
#         pl.legend(loc='best')
#         pl.show()
#         pl.plot(data[optim2[1] - first_width:optim2[1] + first_width, 0],
#                 data[optim2[1] - first_width:optim2[1] + first_width, 1], lw=5, c='g', label='measurement')
#         pl.plot(data[optim2[1] - first_width:optim2[1] + first_width, 0],
#                 gaussian(data[optim2[1] - first_width:optim2[1] + first_width, 0], *optim1), lw=3, c='b', label='fit of 1 Gaussian')
#         pl.plot(data[optim2[1] - first_width:optim2[1] + first_width, 0],
#                 two_gaussians(data[optim2[1] - first_width:optim2[1] + first_width, 0], *optim2),lw=1, c='r', ls='--', label='fit of 2 Gaussians')
#         pl.legend(loc='best')
#         pl.show()


    # Parameters from 2 Gaussian plots for the first plot 
    reshaped_optim2 = [optim2[0], optim2[1], optim2[2], optim2[-1]]
    
    # smallest error will give the best result and
    # two Gaussians always satisfy this condition,
    # but for edge location I need a perfect fit only on the first peak.
    # Two Gaussians are always better since there are small bumps in the region,
    # which, even though negligible, still contribute
    #
    # Optim2 gives the first peak always correctly so I can select a region
    # around the peak within 3 deviations
    # This will test only the fit on the first peak
    # which will give the first edge perfectly
    # and, in turn, the radius
    #first_width = optim1[2] * 3
    
    err2 = np.sqrt((errfunc1(reshaped_optim2, data[:, 0],
                                     data[:, 1] )) ** 2).sum()
    err1 = np.sqrt((errfunc1(optim1, data[:, 0],
                                     data[:, 1] )) ** 2).sum()
    
    # check which error is smaller and select that fit 
    if err1 > err2:
        # 2nd error is smaller take the
        # centre for the first peakfrom the second 
        index_edge = optim2[1]
#         line_spread_function = gaussian(data[optim2[1] - optim2[2] * 3:optim2[1] + optim2[2] * 3, 0], *reshaped_optim2)
    else:
        # 1st error is smaller take the centre 
        # for the first peak from the first fit
        index_edge = optim1[1]
#         line_spread_function = gaussian(data[optim1[1] - optim1[2] * 3:optim1[1] + optim1[2] * 3, 0], *optim1)


    # Calculate the radius
    radius_sphere = index_edge * 0.001 * R + rad_min
    # the intensity of the edge
#     edge_intensity = points[int(index_edge)]
    
#     return round(radius_sphere, 4), edge_intensity, line_spread_function, ayy
    return round(radius_sphere, 4), contact_int


def plot_radii(image_area, centre, start, stop, sigma = 1):
    # Calculate radii for every angle 
    
    theta_bord = np.arange(start, stop)
    phi_bord = np.arange(0, 180)
    
    
    # initial guess for the radius to make calculations faster
    rad_min = image_area.shape[2] / 3
    step = 1
    
    radii_sphere = np.zeros((len(theta_bord), len(phi_bord)))
    intensity_sphere = np.zeros((len(theta_bord), len(phi_bord)))
    lsf_sphere = np.zeros((len(theta_bord), len(phi_bord)))
    contact_points = np.zeros((len(theta_bord), len(phi_bord)))
    
    for theta in theta_bord:
 
        for phi in phi_bord:
            theta_rad = np.radians(theta)
            phi_rad = np.radians(phi)
             
            rad, contact = get_radius(image_area, theta_rad, phi_rad, centre, rad_min, sigma)
             
            radii_sphere[(theta-start)/step, phi/step] = rad
            #intensity_sphere[(theta-start)/step, phi/step] = pixel
            #lsf_sphere[(theta-start)/step, phi/step] = gauss
            contact_points[(theta-start)/step, phi/step] = contact
#     
#     pl.plot(phi_bord, radii_sphere.T, '*')
#     pl.xlabel('angle')
#     pl.ylabel('radius')
#     pl.show()
    
    return radii_sphere, contact_points

# import mhd_utils_3d as md
#   
# image_area, meta_header = md.load_raw_data_with_mhd("/dls/tmp/jjl36382/complicated_data/spheres/sphere_hessian1/gradientgauss.mhd")
# # pl.imshow(image_area)
# # pl.show()
#   
#   
# print image_area.shape[0]
# print image_area.shape[1]
# print image_area.shape[2]
#         
# centre = (int(128 * 1.2), int(128 * 1.2), int(128 * 1.2))
# #centre = (380*1.2,380*1.2,380*1.2)
# start = 290
# stop = 291
# step = 1
#        
# print plot_radii(image_area, centre, start, stop, 1)