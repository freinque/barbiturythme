# Slightly modified partial version of John Glover's modal module
# https://github.com/johnglover/modal

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Spectral processing


def toPolar(x, y):
    return (np.sqrt((x * x) + (y * y)), np.arctan2(y, x))


def toRectangular(mag, phase):
    return np.complex(mag * np.cos(phase),
                      mag * np.sin(phase))

# -----------------------------------------------------------------------------
# Moving average


def moving_average(signal, num_points):
    '''Smooth signal by returning a num_points moving average.
    The first and last num_points/2 are zeros.
    See: http://en.wikipedia.org/wiki/Moving_average'''
    ma = np.zeros(signal.size)
    # make sure num_points is odd
    if num_points % 2 == 0:
        num_points += 1
    n = int(num_points / 2)
    centre = n
    # for each num_points window in the signal, calculate the average
    while centre < signal.size - n:
        avg = 0.0
        for i in np.arange(centre - n, centre + n + 1):
            avg += signal[i]
        avg /= num_points
        ma[centre] = avg
        centre += 1
    return ma


# -----------------------------------------------------------------------------
# Savitzky-Golay


def savitzky_golay(signal, num_points):
    '''Smooth a signal using the Savitzky-Golay algorithm.
    The first and last num_points/2 are zeros.
    See: http://www.statistics4u.com/fundstat_eng/cc_filter_savgolay.html'''
    sg = np.zeros(signal.size)

    # make sure num_points is valid. If not, use defaults
    if not num_points in [5, 7, 9, 11]:
        print 'Invalid number of points to Savitzky-Golay algorithm, ',
        print 'using default (5).'
        num_points = 5
    n = int(num_points / 2)
    centre = n

    # set up savitzky golay coefficients
    if num_points == 5:
        coefs = np.array([-3, 12, 17, 12, -3])
    elif num_points == 7:
        coefs = np.array([-2, 3, 6, 7, 6, 3, -2])
    elif num_points == 9:
        coefs = np.array([-21, 14, 39, 54, 59, 54, 39, 14, -21])
    elif num_points == 11:
        coefs = np.array([-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36])

    # calculate denominator
    denom = np.sum(coefs)

    # for each num_points window in the signal, calculate the average
    while centre < signal.size - n:
        avg = 0.0
        c = 0
        for i in np.arange(centre - n, centre + n + 1):
            # calculate weighted average
            avg += signal[i] * coefs[c]
            c += 1
        avg /= denom
        sg[centre] = avg
        centre += 1
    return sg


# -----------------------------------------------------------------------------
# Low-pass filter


def lpf(signal, order, cutoff):
    'Low-pass FIR filter'
    filter = scipy.signal.firwin(order, cutoff)
    return np.convolve(signal, filter, 'same')



# -----------------------------------------------------------------------------
# Normalise

#FC changed
def normalise(values):
    if np.max(np.abs(values)):
        norm_values = np.zeros(values.size)
        norm_values = values / float(np.max(np.abs(values)))
        return norm_values

# -----------------------------------------------------------------------------
# Onset Detection Functions


class OnsetDetectionFunction(object):
    SMOOTH_NONE = 0
    SMOOTH_MOVING_AVERAGE = 1  # Moving average filter
    SMOOTH_SAVITZKY_GOLAY = 2  # Savitzky-Golay algorithm
    SMOOTH_LPF = 3  # low-pass filter

    def __init__(self):
        self.det_func = np.array([])
        self._sampling_rate = 44100
        self._frame_size = 512
        self._hop_size = 256
        self.smooth_types = [self.SMOOTH_NONE]
        self.smooth_windows = [5]
        self.lpf_cutoff = 0.15
        self.lpf_order = 101
    
    sampling_rate = property(lambda self: self.get_sampling_rate(),
                             lambda self, x: self.set_sampling_rate(x))

    frame_size = property(lambda self: self.get_frame_size(),
                          lambda self, x: self.set_frame_size(x))

    hop_size = property(lambda self: self.get_hop_size(),
                        lambda self, x: self.set_hop_size(x))

    def get_sampling_rate(self):
        return self._sampling_rate

    def set_sampling_rate(self, sampling_rate):
        self._sampling_rate = sampling_rate

    def get_frame_size(self):
        return self._frame_size

    def set_frame_size(self, frame_size):
        self._frame_size = frame_size

    def get_hop_size(self):
        return self._hop_size

    def set_hop_size(self, hop_size):
        self._hop_size = hop_size

    #FC changed
    def smooth(self):
        for i in range(len(self.det_func)):
            if self.smooth_types[i] == self.SMOOTH_MOVING_AVERAGE:
                self.det_func[i] = moving_average(self.det_func[i], self.smooth_windows[i])
            elif self.smooth_types[i] == self.SMOOTH_SAVITZKY_GOLAY:
                self.det_func[i] = savitzky_golay(self.det_func[i], self.smooth_windows[i])
            elif self.smooth_types[i] == self.SMOOTH_LPF:
                self.det_func[i] = filter.lpf(self.det_func[i], self.lpf_order, self.lpf_cutoff)
        
    #FC added    
    def normalise(self):
        for i in range(len(self.det_func)):
            self.det_func[i] = normalise(self.det_func[i])

    def process_frame(self, frame):
        return 0.0

    def process(self, signal): #FC , detection_function):
        # give a warning if the hop size does not divide evenly into the
        # signal size
        if len(signal) % self.hop_size != 0:
            print 'Warning: hop size (%d) is not a factor of signal size (%d)'\
                % (self.hop_size, len(signal))

        # get a list of values for each frame
        detection_function = np.zeros( (self.n_freq_int, \
            len(signal)/self.hop_size), dtype=float )
        
        sample_offset = 0
        i = 0
        while sample_offset + self.frame_size <= len(signal):
            frame = signal[sample_offset:sample_offset + self.frame_size]
            detection_function[:,i] = self.process_frame(frame)
            sample_offset += self.hop_size
            i += 1

        self.det_func = detection_function
        
        #FC added
        self.smooth()  
        # perform any post-processing on the ODF
        self.normalise()
      
        detection_function = self.det_func


class EnergyODF(OnsetDetectionFunction):
    #FC changed the energy ODF to calulate energy on on the freq spectrum
    #(see Parseval Thm)
    def __init__(self):
        OnsetDetectionFunction.__init__(self)
        self.window = np.hanning(self.frame_size)
        self.num_bins = (self.frame_size / 2) + 1
        self.prev_amps = np.zeros(self.num_bins)
        #FC added
        self.n_freq_int = 1
        #FC added
        self.freq_bounds = [[0,self.num_bins]]
        
    def set_frame_size(self, frame_size):
        self._frame_size = frame_size
        self.window = np.hanning(frame_size)
        self.num_bins = (frame_size / 2) + 1
        self.prev_amps = np.zeros(self.num_bins)
         
    #FC added
    def set_freq_bounds(self, bounds):
        for bound in bounds:
            self.freq_bounds = bounds
            self.n_freq_int = len(self.freq_bounds)
            self.prev_amps = np.zeros(self.num_bins)
            
    def process_frame(self, frame):
        # fft
        spectrum = np.fft.rfft(frame * self.window)
        # calculate spectral difference for each bin
        sum_ = np.zeros(self.n_freq_int)
        frame_size_over_sampling_rate = self.frame_size/float(self.sampling_rate)
        for freq_interv in range(self.n_freq_int):
            freq_interv_inf_bin = int(self.freq_bounds[freq_interv][0]*frame_size_over_sampling_rate)
            freq_interv_sup_bin = int(self.freq_bounds[freq_interv][1]*frame_size_over_sampling_rate)

            #FC for bin in range(self.num_bins):
            for bin in range(freq_interv_inf_bin,freq_interv_sup_bin):
                real = spectrum[bin].real
                imag = spectrum[bin].imag
                amp = np.sqrt((real * real) + (imag * imag))
                
                sum_[freq_interv] += (amp - self.prev_amps[bin])*(amp - self.prev_amps[bin])
                self.prev_amps[bin] = amp
        return sum_

class SpectralFlowODF(OnsetDetectionFunction):
    def __init__(self):
        OnsetDetectionFunction.__init__(self)
        self.window = np.hanning(self.frame_size)
        self.num_bins = (self.frame_size / 2) + 1
        self.prev_amps = np.zeros(self.num_bins)
        #FC added
        self.n_freq_int = 1
        #FC added
        self.freq_bounds = [[0,self.num_bins]]
        
    def set_frame_size(self, frame_size):
        self._frame_size = frame_size
        self.window = np.hanning(frame_size)
        self.num_bins = (frame_size / 2) + 1
        self.prev_amps = np.zeros(self.num_bins)
         
    #FC added
    def set_freq_bounds(self, bounds):
        for bound in bounds:
            self.freq_bounds = bounds
            self.n_freq_int = len(self.freq_bounds)
            self.prev_amps = np.zeros(self.num_bins)
            
    def process_frame(self, frame):
        # fft
        spectrum = np.fft.rfft(frame * self.window)
        # calculate spectral difference for each bin
        sum_ = np.zeros(self.n_freq_int)
        frame_size_over_sampling_rate = self.frame_size/float(self.sampling_rate)
        for freq_interv in range(self.n_freq_int):
            freq_interv_inf_bin = int(self.freq_bounds[freq_interv][0]*frame_size_over_sampling_rate)
            freq_interv_sup_bin = int(self.freq_bounds[freq_interv][1]*frame_size_over_sampling_rate)

            #FC for bin in range(self.num_bins):
            for bin in range(freq_interv_inf_bin,freq_interv_sup_bin):
                real = spectrum[bin].real
                imag = spectrum[bin].imag
                amp = np.sqrt((real * real) + (imag * imag))
                #FC L^2 norm
                #sum_[freq_interv] += (amp - self.prev_amps[bin])*(amp - self.prev_amps[bin])
                #FC Flow
                sum_[freq_interv] += ( amp - self.prev_amps[bin] ) 
                #FC Positive flow
                #min(0.0, amp - self.prev_amps[bin]) 
                #FC L^1 norm
                #sum_[freq_interv] += abs(amp - self.prev_amps[bin])
                self.prev_amps[bin] = amp
        #FC L^2 norm
        #return np.sqrt(sum_)
        #FC L^1 norm
        return sum_


class SpectralPhaseODF(OnsetDetectionFunction):
    def __init__(self):
        OnsetDetectionFunction.__init__(self)
        self.window = np.hanning(self.frame_size)
        self.num_bins = (self.frame_size / 2) + 1
        #FC added
        self.n_freq_int = 1
        #FC added
        self.freq_bounds = [[0,self.num_bins]]
        self.prev_mags = np.zeros((self.n_freq_int, self.num_bins))
        self.prev_phases = np.zeros((self.n_freq_int, self.num_bins))
        self.prev_phases2 = np.zeros((self.n_freq_int, self.num_bins))
        self.prediction = np.zeros((self.n_freq_int, self.num_bins), dtype=np.complex)
                
    def set_frame_size(self, frame_size):
        self._frame_size = frame_size
        self.window = np.hanning(frame_size)
        self.num_bins = (frame_size / 2) + 1
        self.prev_mags = np.zeros((self.n_freq_int, self.num_bins))
        self.prev_phases = np.zeros((self.n_freq_int, self.num_bins))
        self.prev_phases2 = np.zeros((self.n_freq_int, self.num_bins))
        self.prediction = np.zeros((self.n_freq_int, self.num_bins), dtype=np.complex)

    #FC added
    def set_freq_bounds(self, bounds):
        self.freq_bounds = bounds
        self.n_freq_int = len(self.freq_bounds)
        self.prev_mags = np.zeros((self.n_freq_int, self.num_bins))
        self.prev_phases = np.zeros((self.n_freq_int, self.num_bins))
        self.prev_phases2 = np.zeros((self.n_freq_int, self.num_bins))
        self.prediction = np.zeros((self.n_freq_int, self.num_bins), dtype=np.complex)

    def process_frame(self, frame):
        # fft
        spectrum = np.fft.rfft(frame * self.window)
        # calculate complex difference for each bin
        cd = np.zeros(self.n_freq_int)
        frame_size_over_sampling_rate = self.frame_size/float(self.sampling_rate)
        
        for freq_interv in range(self.n_freq_int):
            freq_interv_inf_bin = int(self.freq_bounds[freq_interv][0]*frame_size_over_sampling_rate)
            freq_interv_sup_bin = int(self.freq_bounds[freq_interv][1]*frame_size_over_sampling_rate)

            for bin in range(freq_interv_inf_bin,freq_interv_sup_bin):
                # magnitude prediction is just the previous magnitude
                # phase prediction is the previous phase plus the difference
                # between the previous two frames
                predicted_phase = (2 * self.prev_phases[freq_interv][bin]) - \
                    self.prev_phases2[freq_interv][bin]
                # bring it into the range +- pi
                predicted_phase -= 2 * np.pi * \
                    np.round(predicted_phase / (2 * np.pi))
                # convert back into the complex domain to calculate stationarities
                self.prediction[freq_interv][bin] = toRectangular(self.prev_mags[freq_interv][bin],
                                                 predicted_phase)
                # get stationarity measures in the complex domain
                real = (self.prediction[freq_interv][bin].real - spectrum[bin].real)
                real = real * real
                imag = (self.prediction[freq_interv][bin].imag - spectrum[bin].imag)
                imag = imag * imag
                cd[freq_interv] += np.sqrt(real + imag)
                # update previous phase info for the next frame
                self.prev_phases2[freq_interv][bin] = self.prev_phases[freq_interv][bin]
                self.prev_mags[freq_interv][bin], self.prev_phases[freq_interv][bin] = \
                    toPolar(spectrum[bin].real, spectrum[bin].imag)
        return cd
           
##################################################################################################

class ODFPeak(object):
    def __init__(self):
        self.location = 0
        self.value = 0
        self.threshold_at_peak = 0
        self.size = 0


class OnsetDetection(object):
    # threshold types
    THRESHOLD_NONE = 0
    THRESHOLD_FIXED = 1
    THRESHOLD_MEDIAN = 2
    # onset location in relation to peak
    ONSET_AT_PEAK = 0       # on the peak
    ONSET_AT_PEAK_DIFF = 1  # largest point in diff(det_func) behind peak
    ONSET_AT_MINIMA = 2     # at the previous local minima
    ONSET_AT_THRESHOLD = 3  # last point before the peak where det_func >= threshold

    def __init__(self):
        self.onsets = []
        self.onset_offsets = []
        self.det_func = np.array([])
        self.threshold_types = self.THRESHOLD_MEDIAN
        #FC changed
        self.min_thresholds = [0.6] 
        self.median_b = 1.0
        self.median_windows = [9]
        self.onset_location = self.ONSET_AT_PEAK
        # number of neighbouring samples on each side that a peak
        # must be larger than
        self.peak_sizes = 1
        self.peaks = []
        #FC added
        self.n_freq_int = 1
        self.hop_size = 256

    def _calculate_threshold(self):
        if not len(self.det_func):
            return

        self.threshold = np.zeros(self.det_func.shape)
        for freq_interv in range(self.n_freq_int):
            if (self.threshold_types[freq_interv] == self.THRESHOLD_NONE):
                self.threshold[freq_interv] = np.zeros(len(self.threshold[freq_interv]))
            if (self.threshold_types[freq_interv] == self.THRESHOLD_FIXED):
                self.threshold[freq_interv] = np.ones(len(self.threshold[freq_interv]))*self.min_thresholds[freq_interv]
            if (self.threshold_types[freq_interv] == self.THRESHOLD_MEDIAN):
                for i in range(len(self.det_func[freq_interv])):
                    # make sure we have enough signal either side of i to calculate the
                    # median threshold
                    start_sample = 0
                    end_sample = len(self.det_func[freq_interv])
                    if i > (self.median_windows[freq_interv] / 2):
                        start_sample = i - (self.median_windows[freq_interv] / 2)
                    if i < len(self.det_func[freq_interv]) - (self.median_windows[freq_interv] / 2):
                        end_sample = i + (self.median_windows[freq_interv] / 2) + 1
                    median_samples = self.det_func[freq_interv,start_sample:end_sample]
                    self.threshold[freq_interv,i] = \
                    max(self.min_thresholds[freq_interv],
                            self.median_b*np.mean(median_samples)) #?FC

    def find_peaks(self):
        self.peaks = []
        for freq_interv in range(self.n_freq_int):
            self.peaks.append([])
            for i in range(len(self.det_func[freq_interv])):
                # check that the current value is above the threshold
                if self.det_func[freq_interv,i] < self.threshold[freq_interv,i]:
                    continue
                # find local maxima
                # peaks only need to be larger than the nearest self.peak_sizes[freq_interv]
                # neighbours at boundaries
                forward_neighbours = min(self.peak_sizes[freq_interv], len(self.det_func[freq_interv]) - (i + 1))
                backward_neighbours = min(self.peak_sizes[freq_interv], i)
                maxima = True
                # search all forward neighbours (up to a max of self.peak_sizes[freq_interv]),
                # testing to see if the current sample is bigger than all of them
                for p in range(forward_neighbours):
                    if self.det_func[freq_interv,i] < self.det_func[freq_interv,i + p + 1]:
                        maxima = False
                        break
                # if it is less than 1 of the forward neighbours,
                # no need to check backwards
                if not maxima:
                    continue
                # now test the backwards neighbours
                for p in range(backward_neighbours):
                    if self.det_func[freq_interv,i] < self.det_func[freq_interv,i - (p + 1)]:
                        maxima = False
                        break
                if maxima:
                    peak = ODFPeak()
                    peak.location = i
                    peak.value = self.det_func[freq_interv,i]
                    peak.threshold_at_peak = self.threshold[freq_interv,i]
                    peak.size = self.peak_sizes[freq_interv]
                    self.peaks[freq_interv].append(peak)
        #return self.peaks

    def get_peak(self, location):
        """Return the ODFPeak object with a given peak.location value.
        Returns None if no such peak exists."""
        if self.peaks:
            for p in self.peaks:
                if p.location == location:
                    return p
        return None

    def find_onsets(self):
        self.onset_offsets = []

        #if (len(self.det_func)<2):
        #    msg = 'must initialize self.det_func to a nontrivial array'
        #    raise Exception(msg) 
        
        #FC added
        self.n_freq_int = len(self.det_func)
       
        self._calculate_threshold()
        self.find_peaks()

        for freq_interv in range(self.n_freq_int):
            prev_peak_location = 0
            self.onset_offsets.append([])
            for peak in self.peaks[freq_interv]:
                onset_location = peak.location
                # if onset locations are peak locations, we're done

                if self.onset_location == self.ONSET_AT_PEAK_DIFF:
                    # get previous peak_size/2 samples, including peak.location
                    start = (peak.location + 1) - (self.peak_sizes[freq_interv] / 2)
                    if start < 0:
                        start = 0
                    end = peak.location + 1
                    if end >= len(self.det_func[freq_interv]):
                        end = len(self.det_func[freq_interv]) - 1
                    samples = self.det_func[freq_interv,start:end]
                    # get the point of biggest change in the samples
                    samples_diff = np.diff(samples)
                    max_diff = abs(samples_diff[-1])
                    max_diff_pos = 0
                    for i in range(1, len(samples_diff)):
                        if samples_diff[i] >= max_diff:
                            max_diff = samples_diff[i]
                            max_diff_pos = i
                    onset_location = peak.location - (len(samples_diff) -
                                                  max_diff_pos)

                elif self.onset_location == self.ONSET_AT_MINIMA:
                    if peak.location > 1:
                        i = peak.location - 1
                        # find the nearest local minima behind the peak
                        while i > 1:
                            if (self.det_func[freq_interv,i] <= self.det_func[freq_interv,i + 1] and
                                self.det_func[freq_interv,i] <= self.det_func[freq_interv,i - 1]):
                                break
                            if (i - 1) <= prev_peak_location:
                                break
                            i -= 1
                        onset_location = i

                elif self.onset_location == self.ONSET_AT_THRESHOLD:
                    if peak.location > 1:
                        i = peak.location - 1
                        # find the last point before the peak where the
                        # det_func is above the threshold
                        while i > 1:
                            if self.det_func[freq_interv,i - 1] < peak.threshold_at_peak:
                                break
                            if (i - 1) <= prev_peak_location:
                                break
                            i -= 1
                        onset_location = i

                self.onset_offsets[freq_interv].append(onset_location)
                prev_peak_location = peak.location
        
        self.onset_offsets = np.array(self.onset_offsets)

        self.onsets =np.array( [ [self.onset_offsets[i][j]*self.hop_size \
                        for j in range(len(self.onset_offsets[i]))] \
                        for i in range(self.n_freq_int)] )
 

#plots
##################################################################################

def plot_onsets(onsets, max_height=1.0, min_height=-1.0):
    y = [max_height, min_height]
    for onset in onsets:
        x = [onset, onset]
        plt.plot(x, y, 'r--')


def plot_odf(det_func, hop_size=1, colour='0.4',
                            linestyle='-'):
    df = np.zeros(len(det_func) * hop_size)
    sample_offset = 0
    for frame in det_func:
        df[sample_offset:sample_offset + hop_size] = frame
        sample_offset += hop_size
    plt.plot(df, color=colour, linestyle=linestyle)


