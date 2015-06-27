'''
@mainpage BarBitURythme documentation: Overview

\b Bar: gather your best beat bars  \n
\b Bit: encode them into computationally sound data  \n
\b U: interact \n
\b Rythme: get new rhythms, with perhaps some exotic touch  \n
\n
\b BarBitURythme is a
Pyhton module for reading .wav audio data files, extracting their rhythm
sequences, 'learning' from both their short and long term 'patterns' and
generating new sequences of similar 'style'.\n
\n
@author Francois Charest freinque.prof@gmail.com and Arst D.H. Neio arstdashneio@gmail.com

\b Examples \n
 \n
http://freinque.github.io/barbiturythme/html/bbur_reader_test_0.0.4.html \n
http://freinque.github.io/barbiturythme/html/bbur_gener_test_0.0.4.html \n
\n
\b Documentation \n
http://freinque.github.io/barbiturythme/docs/html/index.html \n
\n
\b Using: \n
\n
- A (long term) statistical model and optimization algorithm adapted from 
J.-F. Paiement, Y. Grandvalet, S. Bengio, D. Eck  \n
'A Distance Model for Rythms', Proc. 25th Int. Conf. on Machine Learning, 2008.
\n
- M. Hamilton's hmm.py module for the EM training/optimization of Hidden Markov 
Models for short term dependencies. See \n
http://www.cs.colostate.edu/~hamiltom/code.html and \n
L.R. Rabiner, 'A Tutorial on Hidden Markov Models and Selected Applications
in Speech Recognition', Proc. of the IEEE, Vol. 77, 2, 1989.
\n
- modal_modif.py an adaptation of J. Glover's modal module for onset (e.g. 
sudden change in the signal) detection implementating spectral difference 
models considered by Masri (and perhaps some other people before). 
See \n 
https://github.com/johnglover/modal and \n
J.P. Bello, L. Daudet, S. Abdallah, C. Duxbury, M. Davies, M.B. Sandler \n
'A Tutorial on Onset Detection in Music Signals', IEEE Trans. Speech and 
Audio Processing, Vol. 13, 5, 2005.
\n
- The scikit-learn package for standard machine learning algorithms (kmeans 
clustering, etc.) 
\n
- csound for .wav generation

'''

__version_info__ = ('0','0','4')

__version__ = '.'.join(__version_info__)


import numpy as np
import scipy.io.wavfile
import scipy.signal
import scipy.fftpack
import sklearn.cluster
import matplotlib.pyplot as plt
import hmm          #is written clearly, and seems right
import modal_modif
import bbur_gene

class BarBit:
    '''
    reads, preprocesses and processes an audio signal to extract its
    main rhythmic features. Onset detection using 2 algorithms 
    (spectral flow and phase spectra) written
    in the modal python module. Actually better at detecting percussive 
    instruments, although fancier onset detection techniques could
    be used.
    
    '''
    ODF_ENERGY = 0
    ##@var ODF_ENERGY
    #@brief amount of energy in each freq_int
    ODF_SPECTRAL_FLOW = 1
    ##@var ODF_SPECTRAL_FLOW
    #@brief flow of spectrum in each freq_int
    ODF_SPECTRAL_PHASE = 2
    ##@var ODF_SPECTRAL_PHASE
    #@brief second derivative of spectrum phase in each freq_int

    ONSET_AT_PEAK = 0 
    ##@var ONSET_AT_PEAK
    #@brief onsets at peaks
    ONSET_AT_PEAK_DERIV = 1 
    ##@var ONSET_AT_PEAK_DERIV
    #@brief onsets at peak values of the derivative behind peaks
    ONSET_AT_MINIMUM = 2
    ##@var ONSET_AT_MINIMUM
    #@brief onsets at minima behind peaks
    ONSET_AT_THRESHOLD = 3 
    ##@var ONSET_AT_THRESHOLD
    #@brief onsets at points where odf=threshold behind peaks

    def __init__( self ):
        self.odf_type = self.ODF_SPECTRAL_FLOW
        ##@var odf_type 
        #@brief list of constants ODF_ENERGY, ODF_SPECTRAL_FLOW
        #and ODF_SPECTRAL_PHASE
        self.onset_locations = [self.ONSET_AT_PEAK_DERIV]
        ##@var onset_locations
        #@brief list of constants ONSET_AT_PEAK, ONSET_AT_PEAK_DERIV, 
        #ONSET_AT_MINIMUM and ONSET_AT_THRESHOLD 
        self.det_func = np.array([])
        ##@var det_func
        #@brief list of onset detection function of length n_freq_int
        self._threshold = np.array([])
        self.onsets = np.array([]) #sample location of the onsets
        ##@var onsets
        #@brief list of onset locations (in samples) of length n_freq_int
        self._onset_offsets = np.array([]) 
        #location of the onsets in hops
        
        self.window_size = 512
        ##@var window_size
        #@brief size of the (Hanning) window used for the spectral analysis 
        #(=512)
        self.hop_size = 245
        ##@var hop_size
        #@brief size of the elementary translation of the window used for the 
        #spectral analysis (=256)
   
        self.signal = np.array([])
        ##@var signal
        #@brief signal to be analysed (shape (n channels,total n samples))
        self.sampling_rate = 44100
        ##@var sampling_rate
        #@brief of the signal to be analysed (=44100)
        self.start_time = 0.0
        ##@var start_time
        #@brief time at which the analysis process starts (in s) (=0.0)
        self.end_time = 0.0
        ##@var end_time
        #@brief time at which the analysis process ends (in s) (=0.0)
        self.mean_signal = np.array([])
        ##@var mean_signal
        #@brief average of signal over its channels
        
        self.n_freq_int = 1
        ##@var n_freq_int
        #@brief number of frequency intervals over which the analysis is done 
        #(independently) (=1)
        self.freq_bounds = [[20.0,22000.0]]
        ##@var freq_bounds
        #@brief list of bounds of the frequency intervals 
        #(form [low bound, hi bound]) (=[[20,22000]])
        self.smooth_windows = [5]
        ##@var smooth_windows
        #@brief list of widths for the moving average smoothing of det_func 
        #(in hops) (=[5])
        self.peak_sizes = [0.2] 
        ##@var peak_sizes
        #@brief list of widths of the peaks of det_func (in s) (=[0.2])
        self.min_thresholds = [0.5]
        ##@var min_thresholds
        #@brief list of min values for the peaks of det_func 
        #([-1,1] normalized) (=[0.5])
        self.median_windows = [9]
        ##@var median_windows
        #@brief list of widths for the moving median threshold calc over 
        #det_func (in hops) (=[9])


        self.bpm = 0.0
        ##@var bpm
        #@brief tempo (in bpm) of signal used to encode bars (=0.0)
        self.n = 4
        ##@var n
        #@brief number of bars per observed sequence (=4)
        self.beats_per_bar = 4
        ##@var beats_per_bar
        #@brief number of beats in analysed bars (=4)
        self.subs_per_beat = 8
        ##@var subs_per_beat
        #@brief number of subdivisions per bar on which the onsets are projected (=8)
        
        self.n_bars_offsets = []
        ##@var n_bars_offsets
        #@brief list of onset offsets (in hops) for each n_bar. \n
        #Indices [hop shift][n_bar number][freq_interv number][onset_number]
        self.n_bars_sub = []
        ##@var n_bars_sub
        #@brief list of onset offsets (in integer number of subdivs) for each n_bar. \n
        #Indices [hop shift][n_bar number][freq_interv number][onset_number]
        self.n_bars_bin = np.array([])
        ##@var n_bars_bin
        #@brief list of binary repr of the onsets for each n_bar (a [0,1]**self.n_freq_int valued time series). \n
        #Indices [hop shift][n_bar number][subdiv number][freq_interv number]
        self.n_bars_int = np.array([])
        ##@var n_bars_int
        #@brief list of integer repr of the onsets for each n_bar (a {0,...,2**self.n_freq_int-1} valued time series). \n
        #Indices [hop shift][n_bar number][subdiv number][freq_interv number]
        self.bars_offsets = []
        ##@var bars_offsets
        #@brief list of onset offsets (in hops) for each bar. \n
        #Indices [hop shift][n_bar number][bar number][freq_interv number][onset_number]
        self.bars_sub = []
        ##@var bars_sub
        #@brief list of onset offsets (in integer number of subdivs) for each bar. \n
        #Indices [hop shift][n_bar number][bar number][freq_interv number][onset_number]
        self.bars_bin = np.array([])
        ##@var bars_bin
        #@brief list of binary repr of the onsets for each bar (a [0,1]**self.n_freq_int valued time series). \n
        #Indices [hop shift][n_bar number][bar number][subdiv number][freq_interv number]
        self.bars_int = np.array([])
        ##@var bars_int
        #@brief list of integer repr of the onsets for each bar (a {0,...,2**self.n_freq_int-1} valued time series). \n
        #Indices [hop shift][n_bar number][bar number][subdiv number][freq_interv number]

    def read( self, filestr ):
        '''
        reads the .wav file at filestr, scales its values to [-1,1].
        Sets sampling_rate and signal
        
        @param filestr string encoding the location of a .wav file
        '''
        
        if (filestr[-4:] != '.wav'):
            msg = 'must input string of .wav file location'
            raise ValueError(msg)
        
        print 'WarningWarning: Here, depending on which program+option generated \
your .wav file, you may get the \'chunk not understood\' error. It doesn\'t seem \
to alter the signal reading, though.'

        self.sampling_rate, self.signal = scipy.io.wavfile.read( filestr )
        
        self.signal = self.signal / 32768.0 #normalize signal to [-1,1]
    
    def set_freq_bounds( self, listofintervals ):
        '''
        sets freq_bounds, a list of intervals. A interval: is a list of length
        2 where the 0th element is smaller than the 1st. Bounds are in Hz.
        
        @param listofintervals List of lists of type [lower bound, upper bound]
        '''
        
        self.freq_bounds = listofintervals
        self.n_freq_int = len(listofintervals)
        if (len(self.smooth_windows) != self.n_freq_int):
            self.smooth_windows = [5]*self.n_freq_int
        if (len(self.peak_sizes) != self.n_freq_int):
            self.peak_sizes = [0.2]*self.n_freq_int 
        if (len(self.min_thresholds) != self.n_freq_int):
            self.min_thresholds = [0.5]*self.n_freq_int
        if (len(self.median_windows) != self.n_freq_int):
            self.median_windows = [9]*self.n_freq_int 
        if (len(self.onset_locations) != self.n_freq_int):
            self.onset_locations = [self.ONSET_AT_PEAK_DERIV]*self.n_freq_int 


    def calc_odf( self, starttime=0.0, endtime=1.0e20, ):
        '''
        calculates the onset detection function on the freq_bound intervals
        using the signal from starttime to endtime (in s).
        
        @param starttime time at which the analysis starts
        @param endtime time at which the analysis ends 
        '''
        if (len(self.signal)==0):
            msg = 'must read (set signal) first'
            raise ValueError(msg)

        self.start_time = max(min(starttime,endtime), 0.0)
        self.end_time = min(max(starttime,endtime), \
                            len(self.signal)/float(self.sampling_rate))
        
        #uses the average of the signal on the channels, but 
        #TODO would eventually process the channels separately
        self.mean_signal = \
    self.signal.sum(1)[ self.start_time*self.sampling_rate:\
    self.end_time*self.sampling_rate ] / float(self.signal.shape[1])
        
        #EnergyODF, SpectralFlowODF or SpectralPhaseODF() are chosen since 
        #they seem more simple and natural, given our needs
        if (self.odf_type == self.ODF_ENERGY):
            odf = modal_modif.EnergyODF() #spectral energy
        if (self.odf_type == self.ODF_SPECTRAL_FLOW):
            odf = modal_modif.SpectralFlowODF() #spectral flow
        if (self.odf_type == self.ODF_SPECTRAL_PHASE):
            odf = modal_modif.SpectralPhaseODF() #phase spectra
        
        odf.set_sampling_rate(self.sampling_rate)
        odf.set_hop_size(self.hop_size)
        odf.set_frame_size(self.window_size)
        odf.smooth_types = [odf.SMOOTH_MOVING_AVERAGE]*self.n_freq_int
        odf.smooth_windows = self.smooth_windows
        odf.set_freq_bounds(self.freq_bounds)

        #calculation of the onset detection function
        odf.process(self.mean_signal)
        self.det_func = np.array(odf.det_func)

    def plot_odf( self ):
        '''
        plots the onset detection functions using matplotlib.pyplot
        '''
        if (len(self.det_func)==0):
            msg = 'must calculate det_func first'
            raise ValueError(msg)

        fig = plt.figure(1, figsize=(10, 10))
        for f_i in range(self.n_freq_int):
            plt.subplot(self.n_freq_int, 1, f_i+1)
            plt.title('Onset detection in range ' + \
                    str(self.freq_bounds[f_i][0]) +' to '+ \
                    str(self.freq_bounds[f_i][1]) + ' Hz' )
            modal_modif.plot_odf(self.det_func[f_i], self.hop_size)
        plt.show()

    def calc_onsets( self ):
        '''
        calculates the onsets from the onset detection function on the 
        freq_bound intervals using the signal from starttime to endtime 
        (in s).
        '''
        if (len(self.det_func)==0):
            msg = 'must calculate det_func first'
            raise ValueError(msg)

        onset_finder = modal_modif.OnsetDetection()
        onset_finder.det_func = self.det_func
        onset_finder.hop_size = self.hop_size
        onset_finder.peak_sizes = [ int(x*(self.sampling_rate/self.hop_size)) \
                for x in self.peak_sizes] #now in seconds
        onset_finder.min_thresholds = self.min_thresholds
        onset_finder.threshold_types = \
        [onset_finder.THRESHOLD_MEDIAN]*self.n_freq_int
        onset_finder.median_windows = self.median_windows
        onset_finder.onset_locations = self.onset_locations

        onset_finder.find_onsets()
        self.onsets = np.array(onset_finder.onsets)
        self._threshold = np.array(onset_finder.threshold) #class level just 
                                                           #to be plotted
        
        self._onset_offsets = np.array( [ [ \
                self.onsets[i][j]/self.hop_size \
                for j in range(len(self.onsets[i])) ] \
                for i in range(len(self.onsets)) ] )

    
    def plot_onsets( self ):
        '''
        plots the onset detection functions using matplotlib.pyplot
        '''
        if (len(self.onsets)==0):
            msg = 'must calculate onsets first'
            raise ValueError(msg)

        fig = plt.figure(1, figsize=(10, 10))
        for f_i in range(self.n_freq_int):
            plt.subplot(self.n_freq_int, 1, f_i+1)
            plt.title('Onset detection in range ' + \
                    str(self.freq_bounds[f_i][0]) +' to '+ \
                    str(self.freq_bounds[f_i][1]) + ' Hz' )
            modal_modif.plot_onsets(self.onsets[f_i], 1.0, 0.0)
            modal_modif.plot_odf(self.det_func[f_i], self.hop_size)
            modal_modif.plot_odf(self._threshold[f_i], self.hop_size, "green")
        plt.show()
            
    def encode_onsets( self ):
        '''
        encodes an onset list to lists of n_bars and bars representations
        of various types:
        offsets: list of onsets of the bar or n_bar in hops
        sub: list onsets projected on the bar or n_bar subdivisions
        bin: vector with components corresp to bar or n_bar subdivisions 
        having ones at onset subdivision locations
        '''
        if (self.bpm==0.0):
            msg = 'must initialize bpm to non-trivial value'
            raise ValueError(msg)
        if (len(self.onsets)==0):
            msg = 'must calculate onsets first'
            raise ValueError(msg)

        hops_per_beat = self.sampling_rate*60.0/(self.bpm*self.hop_size)
        hops_per_bar = self.beats_per_bar*hops_per_beat
        hops_per_n_bar = hops_per_bar*self.n
        hops_per_sub = hops_per_beat/self.subs_per_beat

        n_n_bars = int( \
                len(self.mean_signal)/float(self.hop_size*hops_per_n_bar) + 2 )
        
        #self.n_bars_offsets is a partition of _onset_offsets w.r. to n_bars
        self.n_bars_offsets = [ [ [ []
                for f_i in range(self.n_freq_int) ] \
                for n_b in range(n_n_bars) ] \
                for s_s in range(int(hops_per_n_bar)) ]

        for f_i in range(self.n_freq_int):
            for s_s in range(int(hops_per_n_bar)):
                for i in range(len(self._onset_offsets[f_i])):
                    n_bar_n = \
                        int((self._onset_offsets[f_i][i] + s_s) \
                        / hops_per_n_bar )
            
                    rest = \
                        int((self._onset_offsets[f_i][i] + s_s) \
                        % hops_per_n_bar )

                    self.n_bars_offsets[s_s][n_bar_n][f_i].append( rest )
        
        self.n_bars_offsets = np.array(self.n_bars_offsets)

        #self.n_bars_sub is self.n_bars_offsets projected on the subdivs
        self.n_bars_sub = [ [ [ [ int( x/hops_per_sub )
                for x in self.n_bars_offsets[s_s][n_b][f_i] ] \
                for f_i in range(self.n_freq_int) ] \
                for n_b in range(n_n_bars) ] \
                for s_s in range(int(hops_per_n_bar)) ]

        self.n_bars_sub = np.array(self.n_bars_sub)

        #self.n_bars_bin is a [0,1]**self.n_freq_int valued time series 
        self.n_bars_bin = np.zeros( (hops_per_n_bar, n_n_bars, \
                self.subs_per_beat*self.beats_per_bar*self.n, self.n_freq_int), \
                dtype=int)
        for s_s in range(int(hops_per_n_bar)):
            for n_bar in range(len(self.n_bars_sub[s_s])):
                for f_i in range(self.n_freq_int):
                    for i in self.n_bars_sub[s_s][n_bar][f_i]:
                        self.n_bars_bin[s_s][n_bar][ i ][f_i] = 1
        
        self.n_bars_int = np.zeros( (hops_per_n_bar, n_n_bars, \
                self.subs_per_beat*self.beats_per_bar*self.n), \
                dtype=int)
        for s_s in range(int(hops_per_n_bar)):
            for n_bar in range(len(self.n_bars_bin[s_s])):
                for i in range(len(self.n_bars_bin[s_s][n_bar])):
                    self.n_bars_int[s_s][n_bar][i] = \
                        bbur_gene.bin_to_int( self.n_bars_bin[s_s][n_bar][i] )


        #a bar is an n^{th} of an n_bar. the index of bar comes right after 
        #that of the n_bar

        self.bars_offsets = [ [ [ bbur_gene.split( \
                self.n_bars_offsets[s_s][n_b][f_i], self.n, hops_per_n_bar ) \
                for f_i in range(self.n_freq_int) ] \
                for n_b in range(n_n_bars) ] \
                for s_s in range(int(hops_per_n_bar)) ]
        self.bars_offsets = np.array( self.bars_offsets )
        self.bars_offsets = np.transpose( self.bars_offsets, (0,1,3,2) )

        #self.bars_sub is self.bars_offsets projected on the subs
        self.bars_sub = [ [ [ [ [ int( x/hops_per_sub )
                for x in self.bars_offsets[s_s][n_b][b][f_i] ] \
                for f_i in range(self.n_freq_int) ] \
                for b in range(self.n) ] \
                for n_b in range(n_n_bars) ] \
                for s_s in range(int(hops_per_bar)) ]

        self.bars_sub = np.array(self.bars_sub)


        self.bars_bin = [ [ np.split( self.n_bars_bin[s_s][n_b], self.n ) \
                for n_b in range(n_n_bars) ] \
                for s_s in range(int(hops_per_n_bar)) ]

        self.bars_bin = np.array(self.bars_bin)
        
        self.bars_int = [ [ np.split( self.n_bars_int[s_s][n_b], self.n ) \
                for n_b in range(n_n_bars) ] \
                for s_s in range(int(hops_per_n_bar)) ]

        self.bars_int = np.array(self.bars_int)


class discrete_hmm:
    '''
    standard HMM parameter optimization and prediction (modeling short term
    patterns)
    '''
    def __init__( self, nobsstates=2, nmarkovstates=20 ):
        '''
        @param nobsstates Number of observable states
        @param nmarkovstates Number of hidden states
        '''
        self.epochs = 20 
        ##@var epochs
        #@brief number of EM optimization iterations
        self.n_obs_states = nobsstates
        ##@var n_obs_states
        #@brief number of observable states
        self.n_markov_states = nmarkovstates 
        ##@var n_markov_states
        #@brief number of hidden states
        
        #uniform initial Markov state prob vector
        self.ini_markov_state = np.ones(self.n_markov_states) \
                /float(self.n_markov_states)
        ##@var ini_markov_state
        #@brief initial hidden state (=uniform)

        #randomly chosen initial Markov trans matrix
        self.ini_trans_matrix = []
        ##@var ini_trans_matrix
        #@brief initial transition matrix (=uniform normalized coeffs)
        for i in range(self.n_markov_states):
            r = np.random.rand(self.n_markov_states)
            self.ini_trans_matrix.append(r/float(np.sum(r)))
        self.ini_trans_matrix = np.array(self.ini_trans_matrix).transpose()
        
        #uniform markov to observed matrix
        self.ini_b = np.ones((self.n_markov_states,self.n_obs_states)) \
                /float(self.n_obs_states)
        ##@var ini_b
        #@brief initial hidden to obs matrix (=uniform)
                
        self.init_model()
    
    def init_model( self ):
        '''
        initializes self.model with parameters self.n_obs_states,
        self.n_markov_states, self.ini_markov_state, self.ini_trans_matrix
        and self.ini_b
        '''
        self.model = hmm.HMM(n_states=self.n_markov_states, \
                Pi=self.ini_markov_state, V=np.arange(self.n_obs_states), \
                A=self.ini_trans_matrix, B=self.ini_b )
        ##@var model
        #@brief instance of hmm.HMM

    def fit_model( self, observations ):
        '''
        fits (MLE) the parameters to the sequences (with values in
        np.arange(self.n_obs_states)) of obsvervations using EM iterations

        @param observations List of observations ({0,...,n_obs_states} valued list)
        '''
        
        hmm.baum_welch(self.model, np.array(observations), \
                epochs = self.epochs, graph = False)

        print 'trans matrix (self.model.A) is ', self.model.A
        print 'markov to observed matrix (self.model.B) is ', self.model.B
        
    def proba( self, observation ):
        '''
        calculates the probability of observing the sequence observation
        given the self.model parameters
        
        @param observation {0,...,n_obs_states} valued list
        '''
        return hmm.forward(self.model, \
                np.array(observation),scaling=False)[0]
    
    def log_proba( self, observation ):
        '''
        calculates the log of the probability of observing the sequence 
        observation given the self.model parameters

        @param observation {0,...,n_obs_states} valued list
        '''
        return hmm.forward(self.model, \
                np.array(observation),scaling=True)[0]
    
    def cond_proba( self, x_future, x_past ):
        '''
        calculates the log of the probability of observing the sequence 
        x_future assuming that we've just observed x_past given the 
        self.model parameters

        @param x_future {0,...,n_obs_states} valued list
        @param x_past {0,...,n_obs_states} valued list
        '''
        return hmm.forward(
            self.model,np.hstack((x_past,x_future)), scaling=False )[0] / \
            hmm.forward( self.model,np.array(x_past),scaling=False )[0]

    def log_cond_proba( self, x_future, x_past ):
        '''
        calculates the log of the probability of observing the sequence 
        x_future assuming that we've just observed x_past given the 
        self.model parameters
        
        @param x_future {0,...,n_obs_states} valued list
        @param x_past {0,...,n_obs_states} valued list
        '''
        return hmm.forward(
            self.model,np.hstack((x_past,x_future)), scaling=True )[0] - \
            hmm.forward( self.model,np.array(x_past),scaling=True )[0]


class PGBE:
    '''
    contains the long term model and optimiztion algorithm of 
    Paiement-Grandvalet-Bengio-Eck, A Distance Model for Rythms
    '''
    def __init__( self, rho_in ):
        '''
        @param rho_in number into which the observations are to be split 
        '''
        self.rho = rho_in
        ##@var rho
        #@brief number into which the observations are to be split 
        #(e.g. BarBit.n) 
        #into equal parts
        self.c = 5 
        ##@var c
        #@brief 'mixing' parameter of the mixed binomial model (=5)
        
        self._m = 0 
        self._m_over_rho = self._m/self.rho
        self._n_bin_states = 1
        
        self.d = []
        ##@var d
        #@brief (rho,rho) array of distances between sub-observations
        self.alpha = []         
        ##@var alpha
        #@brief (rho,rho) array of lower bounds on distances 
        #between sub-observations
        self.beta = []
        ##@var beta
        #@brief (rho,rho) array of upper bounds on distances 
        #between sub-observations

        self.p = []
        ##@var p
        #@brief (rho,rho,c) array of proba parameters of mixed binom
        self.w = []
        ##@var w
        #@brief (rho,rho,c) array of weight parameters of mixed binom

    def _hamming_d( self, v1, v2 ):
        '''
        returns the Hamming distance of two vectors
        
        @param v1 List of numbers
        @param v2 List of numbers
        @param Number. Hamming distance
        '''
        if (len(v1) != len(v2) ):
            msg = 'hamming dist of vect of unequal length '
            raise ValueError(msg)
        
        dist=0
        for i in range(len(v1)):
            if (v1[i] != v2[i]):
                dist = dist+1
        return dist
    
    def _ret_d( self, observation ):
        '''
        returns the shape (rho,rho) distance matrix of observation in
        binary form
        
        @param observation List of binary valued lists (e.g. BarBit.n_bars_bin)
        @returns (rho,rho) array
        '''
        d = np.zeros((self.rho,self.rho))
        for i in range(self.rho):
            for j in range(i+1,self.rho):
                s = 0.0
                for t in range(self._m_over_rho):
                    s = s + self._hamming_d( \
            observation[i*self._m_over_rho+t], \
            observation[j*self._m_over_rho+t] )
                d[i][j] = s
        return d

    def _ret_alpha( self, matrix ):
        '''
        returns the alpha (see PGBE paper) matrix of matrix

        @param matrix (rho,rho) array
        @returns (rho,rho) array
        '''
        al = np.ones_like(matrix)*self._m_over_rho*self._n_bin_states
        for i in range(self.rho):
            for j in range(i+1,self.rho):
                list_sum = [self._m_over_rho*self._n_bin_states]
                for k in range(i):
                    list_sum.append(matrix[k][i] + matrix[k][j])
                al[i][j] = min(list_sum)
        return al

    def _ret_beta( self, matrix ):
        '''
        returns the beta (see PGBE paper) matrix of matrix
        
        @param matrix (rho,rho) array
        @returns (rho,rho) array
        '''
        be = np.zeros_like(matrix)
        for i in range(self.rho):
            for j in range(i+1,self.rho):
                list_sum = [0]
                for k in range(i):
                    list_sum.append(abs(matrix[k][i] - matrix[k][j]))
                be[i][j] = max(list_sum)
        return be


    def calc_d( self, observations ):
        '''
        calculates the distance matrices of observations 
        
        @param observations List of observations in binary form (a binary obs being a list of binary valued lists (e.g. BarBit.n_bars_bin))
        '''
        self.d = []
        
        lengths = []
        for l in range(len(observations)):
            lengths.append( len(observations[l]) )
            
            bin_lengths = []
            for t in range(len(observations[l])):
                bin_lengths.append( len(observations[l][t]) )
            if (len(set(lengths)) != 1):
                msg = 'bin observations must have the same length'
                raise ValueError(msg)
        if (len(set(lengths)) != 1):
            msg = 'observations must have the same length'
            raise ValueError(msg)
        
        self._m = len(observations[0])
        self._m_over_rho = self._m/self.rho
        self._n_bin_states = len(observations[0][0])

        for l in range(len(observations)):
            self.d.append( self._ret_d(observations[l]) )

        self.d = np.array( self.d )
    
    def calc_alpha( self ):
        '''
        calculates the alpha matrices of self.d
        '''
        self.alpha = []
        for l in range(len(self.d)):
            self.alpha.append( self._ret_alpha(self.d[l]) )
        self.alpha = np.array( self.alpha )
    
    def calc_beta( self ):
        '''
        calculates the beta matrices of self.d
        '''
        self.beta = []
        for l in range(len(self.d)):
            self.beta.append( self._ret_beta(self.d[l]) )
        self.beta = np.array( self.beta )


    def _phi( self, w_in, p_in ):
        '''
        returns the values of phi as defined in PGBE given the binomial
        mixture parameters w_in and p_in. ret_p_zlij_equal_k will depend on it.
        '''
        phi_temp = np.zeros((len(self.d),self.rho,self.rho,self.c))
        for l in range(len(self.d)):
            for i in range(self.rho):
                for j in range(i+1,self.rho):
                    for k in range(self.c):
                        phi_temp[l][i][j][k] = w_in[i][j][k] * \
                bbur_gene.binom(self.d[l][i][j]- self.beta[l][i][j], \
                self.alpha[l][i][j] - self.beta[l][i][j], p_in[i][j][k])
        return phi_temp

    def _p_zlij_equal_k( self, phi_in ):
        '''
        returns the values of p_zlij_equal_k as defined in PGBE given phi
        '''
        p_zlij_equal_k_temp = np.zeros((len(self.d),self.rho,self.rho,self.c))
        for l in range(len(self.d)):
            for i in range(self.rho):
                for j in range(i+1,self.rho):
                    s = np.sum( phi_in[l][i][j] )
                    for k in range(self.c):
                        p_zlij_equal_k_temp[l][i][j][k] = \
                            phi_in[l][i][j][k]/float(s)
        return p_zlij_equal_k_temp

    def _update_p( self, p_zlij_equal_k_in ):
        '''
        updates the values of the binomial mixture parameters p. Part of the 
        M step of the EM optimization process.
        '''
        for i in range(self.rho):
            for j in range(i+1,self.rho):
                for k in range(self.c):
                    denom = 0.0
                    num = 0.0
                    for l in range(len(self.d)):
                        num = num + \
                            (self.d[l][i][j] - self.beta[l][i][j])* \
                            p_zlij_equal_k_in[l][i][j][k]
                        denom = denom + \
                            (self.alpha[l][i][j] - self.beta[l][i][j])* \
                            p_zlij_equal_k_in[l][i][j][k]
                    if (denom == 0.0):
                        #print 'calc_p would have /0 ' 
                        self.p[i][j][k] = 0.5 # irrelevant
                    else:
                        self.p[i][j][k] = num/float(denom)

    def _update_w( self, p_zlij_equal_k_in ):
        '''
        updates the values of the binomial mixture parameters p. part of the 
        M step of the EM optimization process
        '''
        for i in range(self.rho):
            for j in range(i+1,self.rho):
                for k in range(self.c):
                    self.w[i][j][k] = \
                            np.sum( p_zlij_equal_k_in[:,i,j,k] )/ \
                            float(len(p_zlij_equal_k_in) )

    def _mix_bino_values( self ):
        '''
        returns values of (d-beta)/(alpha-beta) which are assumed to be mixed
        bino
        
        @returns List of values of (d-beta)/(alpha-beta) 
        '''
        values = []
        for i in range(self.rho):
            values.append([])
            for j in range(self.rho): 
                values[i].append([])
                for l in range(len(self.d)):
                    if (self.alpha[l][i][j] != self.beta[l][i][j]):
                        #values[i][j].append(0.0) #doesn't matter so much 
                    #else:
                        values[i][j].append( \
                        (self.d[l][i][j] - self.beta[l][i][j]) \
                        /float(self.alpha[l][i][j]-self.beta[l][i][j]) )
        values = np.array(values)
        return values

    def km_init_p_w( self ):
        '''
        calculates the initial values of p and w using a sklearn.cluster.KMeans
        model kmeans on the m_b_v matrices of data
        '''
        m_b_v = self._mix_bino_values()
        
        max_iter = 300
        tol = 0.001
        n_init_cond = 10 
        init='k-means++'
        km = sklearn.cluster.KMeans(self.c, init, n_init_cond, max_iter, tol) 
        
        self.p = []
        self.w = []
        for i in range(self.rho):
            self.p.append([])
            self.w.append([])
            for j in range(self.rho):
                self.p[i].append([])
                self.w[i].append([])
                if (len(m_b_v[i][j]) < self.c):
                    if (len(m_b_v[i][j]) == 0):
                        m_b_v[i][j] = [0]
                    m_b_v[i][j] = np.array(list(m_b_v[i][j])*self.c)
                km.fit( np.reshape(m_b_v[i][j],(len(m_b_v[i][j]),1)) )
                labels = list(km.labels_)
                for k in range(self.c):
                    self.p[i][j].append( max(km.cluster_centers_[k][0], 0.0) )
                    self.w[i][j].append( labels.count(k)/float(len(m_b_v[i][j])) )
        self.p = np.array(self.p)
        self.w = np.array(self.w)
        print 'kmeans initial p = ', self.p
        print 'kmeans initial w = ', self.w


    def em_p_w( self, epsilon=1e-4 ):
        '''
        EM loop on p,w as params and latent z in {1, ldots, c}
        
        @param epsilon Step under which we stop iterating (=1e-4)
        '''
        #epsilon = 1e-3
        step = 1.0
        old_p = np.zeros((self.rho,self.rho,self.c))
        old_w = np.zeros((self.rho,self.rho,self.c))
        iter_ = 0
        print 'em iteration ', iter_
        while ( step > epsilon):
            iter_ = iter_ + 1
            phi = self._phi(self.w, self.p)
            p_zlij_equal_k = self._p_zlij_equal_k(phi)
    
            old_p = [[[ self.p[i][j][k] \
                    for k in range(self.c) ] \
                    for j in range(self.rho) ] \
                    for i in range(self.rho) ] 

            old_w = [[[ self.w[i][j][k] \
                    for k in range(self.c) ] \
                    for j in range(self.rho) ] \
                    for i in range(self.rho) ] 
    
            self._update_p(p_zlij_equal_k)
            self._update_w(p_zlij_equal_k)

            step = 0.0
            for i in range(self.rho):
                for j in range(self.rho):
                    for k in range(self.c):
                        step = step + \
                        abs(self.p[i][j][k] - old_p[i][j][k]) + \
                        abs(self.w[i][j][k] - old_w[i][j][k])
            print 'em iteration ', iter_ , ', step = ', step

        print 'optimal p = ', self.p, '\n optimal w = ', self.w

    def fit_model( self, observations, epsilon=1e-4, kmeans_init=False ):
        '''
        composition that calculates the relavant quantities from 
        observations and optimizes self.p and self.wself.calc_d(observations)
        
        i.e. \n 
        self.calc_alpha() \n
        self.calc_beta() \n

        self.km_init_p_w( ) \n
        self.em_p_w() \n

        @param observations List of observations in binary form (a binary obs being a list of binary valued lists (e.g. BarBit.n_bars_bin))
        @param epsilon Parameter of em_p_w (=1e-4)
        @param kmeans_init Bool on km_init_p_w() (=False)
        '''
        self.calc_d( observations )
        self.calc_alpha()
        self.calc_beta()

        if kmeans_init:
            self.km_init_p_w()
        self.em_p_w( epsilon )
    
    def _proba_dij_eq_betaij_plus_deltaij( self, deltaij, \
                                alphaij, betaij, pij, wij ): 
        '''
        returns the probability of finding a bar that has d[i][j] equal to
        beta_ij plus delta_ij, given parameters pij and wij and the
        mixed binomial model on the distances
        '''
        sum_k = 0.0
        for k in range(self.c):
            sum_k = sum_k + \
            wij[k]*bbur_gene.binom(deltaij,alphaij-betaij,pij[k])
        return sum_k
    
    def proba( self, observation ):
        '''
        returns the probability of observing observation
        given the parameters self self.p and self.w and the
        mixed binomial model on the distances
        
        @param observation list of binary valued lists (e.g. BarBit.n_bars_bin)
        @returns A probability
        '''
        distmatrix = self._ret_d( observation )
        al = self._ret_alpha( distmatrix )
        be = self._ret_beta( distmatrix )
        delta = distmatrix - be
    
        prod = 1.0
        for i in range(self.rho):
            for j in range(i+1,self.rho):
                prod = prod * \
                        self._proba_dij_eq_betaij_plus_deltaij( \
                        delta[i][j], al[i][j], be[i][j], \
                        self.p[i][j], self.w[i][j])
    
        return prod

    def log_proba( self, observation ):
        '''
        returns the log probability of observing observation
        given the parameters self self.p and self.w and the
        mixed binomial model on the distances
        
        @param observation list of binary valued lists (e.g. BarBit.n_bars_bin)
        @returns A log-probability
        '''
        proba = self.proba(observation) 
        if (proba == 0.0):
            return -100.0
        else:
            return np.log(proba)



class PGBEhmm:
    '''
    contains the main model (short and long term) and optimiztion 
    algorithm of Paiement-Grandvalet-Bengio-Eck, A Distance Model for Rythms

    '''
    def __init__( self, pgbe_in, hmm_in ):
        '''
        @param pgbe_in Trained PGBE
        @param hmm_in Trained discrete_hmm
        '''
        self.pgbe = pgbe_in
        ##@var pgbe
        #@brief Trained PGBE
        self.hmm = hmm_in
        ##@var hmm
        #@brief Trained discrete_hmm 
        
        self.x_past = []
        ##@var x_past 
        #@brief {0,...,n_obs_states} valued list (=empty)
        self.x_future = np.random.randint( 0, \
                self.hmm.n_obs_states,size=self.pgbe._m-len(self.x_past) )
        ##@var x_future 
        #@brief {0,...,n_obs_states} valued list (=uniform random)
 
        self.lambda_ = 10
        ##@var lambda_
        #@brief Lagrange multiplier that balances hmm and pgbe probas
    
    def set_x_past( self, xpast ):
        '''
        initializes the values of the sequence on which we will condition
        as the previous observed values
        
        @param xpast {0,...,n_obs_states} valued list (=empty)
        '''
        self.x_past = np.array( xpast )
        self.x_future = np.random.randint( 0, \
                self.hmm.n_obs_states,size=self.pgbe._m-len(self.x_past) )
 
    def init_x_future( self ):
        '''
        varies x_future so that it maximizes 
        self.hmm.log_cond_proba( x_future, x_past ) only\n
        \n
        Makes sense from predictive point of view, not so from generative,
        especially if it converges towards the zero sequence.
        '''

        print 'optimization of hmm cond proba '
        print 'started at ', self.x_future 
        print 'where self.hmm.log_cond_proba( self.x_future, self.x_past ) = ',\
                self.hmm.log_cond_proba( self.x_future, self.x_past )

        x_prov = np.zeros(len(self.x_future), dtype=int)
        end = False

        while (end==False):
            for j in range(len(self.x_future)):
                p_hmm_tests = np.zeros( self.hmm.n_obs_states )

                for i in range( self.hmm.n_obs_states ):
                    x_test = [ x for x in self.x_future]
                    x_test[j] = i
                    p_hmm_tests[i] = \
                        self.hmm.log_cond_proba( x_test, self.x_past )
        
                if (len(set(p_hmm_tests)) == 1):
                    amax = np.random.randint( 0, self.hmm.n_obs_states)
                else:
                    amax = p_hmm_tests.argmax()

                self.x_future[j] = amax
    
            print 'now at ', self.x_future
            print 'where hmm log cond proba is ', \
                    self.hmm.log_cond_proba( self.x_future, self.x_past )

            if np.array_equal( x_prov, self.x_future ):
                end = True
            else:
                x_prov = [ x for x in self.x_future ]

        print 'hmm cond proba optimization ended at ', self.x_future
        print 'where hmm log cond proba =', \
                self.hmm.log_cond_proba( self.x_future, self.x_past )


    def fit_x_future( self, with_init_x_future=False ):
        '''
        varies x_future so that it maximizes 
        self.hmm.log_cond_proba( self.x_future, self.x_past ) + 
            self.lambda_ * self.pgbe.log_proba( self.x_future, self.x_past )  
        
        @param with_init_x_future Bool on adding init_x_future or not (=False)
        '''
        if with_init_x_future:
            self.init_x_future() 
        
        end = False
        
        x_past_bin = [ bbur_gene.int_to_bin(x,self.pgbe._n_bin_states) \
                            for x in self.x_past ]
        x_prov = np.zeros(len(self.x_future), dtype=int)

        while (end==False):
            for j in range(len(self.x_future)):
                func_tests = np.zeros( self.hmm.n_obs_states )

                for i in range( self.hmm.n_obs_states ):
                    x_test = [ x for x in self.x_future ]
                    x_test_bin = [ \
                            bbur_gene.int_to_bin(x,self.pgbe._n_bin_states) \
                            for x in self.x_future ]
                    x_test[j] = i
                    x_test_bin[j] = bbur_gene.int_to_bin(i,self.pgbe._n_bin_states)
                    
                    func_tests[i] = \
            self.hmm.log_cond_proba( x_test, self.x_past ) + \
            self.lambda_ * self.pgbe.log_proba( x_past_bin+x_test_bin ) 
        
                if (len(set(func_tests)) == 1):
                    amax = 0
                else:
                    amax = func_tests.argmax()

                self.x_future[j] = amax
        
            print 'now at ', self.x_future
            print 'where pgbehmm log cond proba is ', func_tests[amax]

            if np.array_equal( x_prov, self.x_future ):
                end = True
            else:
                x_prov = [ x for x in self.x_future ]

        print 'hmm + distance optimization '
        print 'ended at ', self.x_future
        print 'where pgbehmm log cond proba = ', func_tests[amax]



