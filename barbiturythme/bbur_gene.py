'''
bbur_gene.py contains generic functions used by barbiturythme

By Francois Charest, freinque.prof@gmail.com
'''

import os
import numpy as np
import scipy
from scipy.stats import binom
import scipy.fftpack
import matplotlib.pyplot as plt
import barbiturythme

def binom( k, n, p ):
    '''
    A scipy.stats.binom.pmf that can handle p=0.0 and 1.0

    @param  k Number of successes
    @param  n Number of trials
    @param  p Probability of succes in each trial
    @returns   Binomial coefficient
    '''
    if (p==0.0):
        if (k==0):
            return 1.0
        else:
            return 0.0
    elif (p==1.0):
        if (k==n): 
            return 1.0
        else:
            return 0.0
    else:
        return scipy.stats.binom.pmf(k, n, p)

def int_to_bin( integer, n_bin_dig ):
    '''
    returns the first  n_bin_dig binary digits (coefficients of 2**0 
    to 2**n_bin_dig) of the integer integer 
   
    @param integer Integer
    @param n_bin_dig Number of binary digits returned
    @returns List of binary values
    '''
    if (integer/(2**n_bin_dig) >= 1):
        print 'warning: int_to_bin will ignore some digits of integer'

    bin_dig = np.zeros(n_bin_dig, dtype=int)
    for i in range(n_bin_dig):
        bin_dig[i] = integer/(2**i) % 2
    
    return bin_dig

def int_to_bin_obs( observations, n_bin_dig ):
    '''
    int_to_bin on observations

    @param observations List of list of integers
    @param n_bin_dig Number of binary digits returned for each int
    @returns A list of n_bin_dig binary values for each int
    '''
    observations_bin = [ [ \
        int_to_bin( observations[obs][t], n_bin_dig) \
            for t in range(len(observations[obs]))] \
            for obs in range(len(observations))]
    return np.array(observations_bin)

def bin_to_int( v ):
    '''
    returns the integer corresponding to a binary digit list v
    
    @param v List of binary digits (coefficients of 2**0 to 2**len(v))
    @returns  Integer
    '''
    i = 0
    for l in range(len(v)):
        i = i + v[l]*(2**l)
    
    return int(i)

def int_vec_to_bin_vec( int_vec, len_bin_vec ):
    '''
    returns a binary vector of length len_bin_vec having 1s at the indices
    contained in int_vec
    
    @param int_vec List of integers
    @param len_bin_vec Maximal index of the binary vector returned
    @returns Array of binary values
    '''
    bin_vec = np.zeros(len_bin_vec, dtype=int)
    for integer in int_vec:
        bin_vec[integer] = 1
    
    return bin_vec


def correlation( v1, v2, minshift, maxshift ):
    '''
    takes two vectors v1 and v2 of equal lengths and returns
    maxshift-minshift dot profucts shifted under periodic conditions.\n
    Ex. correlation(v,v,0,len(v)) is the autocorrelation of v
    
    @param v1 Array of numbers
    @param v2 Array of numbers of length len(v1)
    @param minshift Integer. Minimal (positive) shift of v2 relative to v1
    @param maxshift Integer. Minimal (positive) shift of v2 relative to v1
    @returns Array of numbers of length maxshift - minshift
    '''
    if ( len(v1) != len(v2) ):
        print 'correlation of vectors of different lengths'
    else:
        a = np.zeros(maxshift-minshift)
        npv1 = np.array(v1)
        rolled = np.roll(v2, minshift)
        for i in range(maxshift-minshift):
            a[i] = np.dot(npv1, rolled)
            rolled =  np.roll(rolled,1)
        return a

def moving_average( v, s ):
    '''
    moving average smoothing of v using a window of size s
    
    @param v Array of numbers
    @param s Integer. Window size used for averaging the values of v
    @returns  Array of numbers of length len(v)
    '''
    npv = np.array(v)
    smooth_window = np.ones(int(s))*1.0/float(s)
    v_smooth = np.convolve(npv, smooth_window, 'same')
    
    return v_smooth

def split( v, n, m ):
    '''
    splits the target [0,m[ of a list v values into n equal 
    intervals isomorphic to [0,m/n[
    @param v List of numbers
    @param n Integer. Number of lists returned
    @param m Upper bound of the target interval to split 
    @returns  List of n list
    '''
    split_v = []
    for i in range(n):
        split_v.append( [] )

    m_over_n = m/n
    for x in v:
        index = int(x / m_over_n)
        rest = x % m_over_n
        
        split_v[index].append( rest )
    
    return split_v

def plot_hist( np_hist ):
    '''
    plots the output of numpy.histogram, rescaling so that it is the
    plot of a probability distribution
    
    @param np_hist (hist, bin_edges) where hist is a list of values of 
    the histogram and bin_edges list of bin edges (of length len(hist)+1)
    '''
    plt.bar(np_hist[1][:-1], np_hist[0]/float(np.sum(np_hist[0])), \
                                        width=np_hist[1][1]-np_hist[1][0] )
    plt.show()

def plot_hist_nn( np_hist ):
    '''
    plots the output of numpy.histogram
     
    @param np_hist (hist, bin_edges) where hist is a list of values of 
    the histogram and bin_edges list of bin edges (of length len(hist)+1)
    '''
    plt.bar(np_hist[1][:-1],np_hist[0], \
                width=np_hist[1][1]-np_hist[1][0] )
    plt.show()

class bpm_finder:
    '''
    contains functions that calculate tempo (in bpm) of a signal 
    based on a list (with values in samles, multiples of gcd, ideally) 
    of onsets. \n
    \n
    The algorithm here was thought from scratch in a few minutes and doesn't 
    pretend to be top-of-the-art, but rather very simple and understandable:
    - it takes the positions of the onsets in a signal
    - it calculates the distance between pairs of nearby onsets
    - it makes a distribution (normalized histogram) out of them
    - it calculates the autocorrelation of that distribution (over a window
    corresponding to musically possible tempos)
    - it identifies the peaks of the autocorrelation function
    - it lets the user choose among the different peaks the one that
    corresponds to the tempo\n
    \n
    Depending on the type of onsets entered (e.g. onsets of a snare,
    kick drum, hihat, etc.) the user might also have to divide or multiply
    the bpm values returned (e.g. by 2, 1/2 etc.)
    '''

    def __init__( self, onsets=np.array([]), gcd=256.0, \
                samplingrate=44100 ):
        self.sampling_rate = samplingrate 
        ##@var sampling_rate
        #@brief of the underlying signal (=44100)
        self.bpms = [0.0] 
        ##@var bpms
        #@brief stores the most likely tempos (in bpm) 
        self.gcd = gcd  
        ##@var gcd
        #@brief gcd of the onsets locations (e.g. BarBit.hop_size) (= 256.0) 
        self.max_distance = 9.0
        ##@var max_distance
        #@brief in seconds (= 9.0)
        self.min_bpm = 40.0
        ##@var min_bpm
        #@brief minimal tempo considered (=40.0)
        self.max_bpm = 250.0
        ##@var max_bpm
        #@brief maximal tempo considered (=250.0)
        
        self.onsets = onsets
        ##@var onsets
        #@brief list of onset locations, in samples
        self.onset_distances = [ \
                (self.onsets[j]-self.onsets[i])/float(self.gcd) \
            for i in range(len(self.onsets)) \
            for j in range(i+1,len(self.onsets)) \
            if ((self.onsets[j]-self.onsets[i]) < \
            (self.sampling_rate*self.max_distance)) ]
        ##@var onset_distances
        #@brief list of dist between onset locations (in samples) up to max_distance

        self.dista_distrib = np.histogram([1],[1]) 
        ##@var dista_distrib
        #@brief dista_distrib of onset_distances (in np.histogram format)

        self.dista_distrib_autocor = []
        ##@var dista_distrib_autocor
        #@brief autocorrelation function of dista_distrib
        
        self.autocor_peaks = [[]]
        ##@var autocor_peaks 
        #@brief top values of the autocorrelation function (in [value, location] form)

        self._min_hop_shift = int(self.sampling_rate*60.0/ \
                float(self.max_bpm*self.gcd))
        self._max_hop_shift = int(self.sampling_rate*60.0/ \
                float(self.min_bpm*self.gcd))


    def set_onsets( self, onsets ):
        '''
        sets the list of onsets and calculates pairwise distances 
        up to self.max_distance
        
        @param onsets Array of onset locations (in samples)
        '''
        self.onsets = np.array(onsets) 
        self.onset_distances = [ \
                (self.onsets[j]-self.onsets[i])/float(self.gcd) \
            for i in range(len(self.onsets)) \
            for j in range(i+1,len(self.onsets)) \
            if ((self.onsets[j]-self.onsets[i]) < \
            (self.sampling_rate*self.max_distance)) ]


    def calc_dista_distrib( self ):
        '''
        lists the distances between pairs of onsets less than max_distance 
        seconds apart and calculates their distribution
        '''
        N = int( self.sampling_rate*self.max_distance/self.gcd )
        self.dista_distrib = np.histogram( self.onset_distances, \
            bins=np.linspace(0, N, num=N) )


    def plot_dista_distrib( self ):
        '''
        plots the distances distribution (in seconds, up to max_distance) 
        '''
        dist_display = ( self.dista_distrib[0], \
                self.dista_distrib[1]*self.gcd/float(self.sampling_rate) ) 
        plot_hist( dist_display )
 
    
    def calc_dista_distrib_autocor( self ):
        '''
        calculates autocorrelation of the onset distances 
        '''
        
        self.dista_distrib_autocor = correlation( self.dista_distrib[0], \
                self.dista_distrib[0], self._min_hop_shift, self._max_hop_shift) 
        self.dista_distrib_autocor = \
                moving_average(self.dista_distrib_autocor, 10)
    
    def plot_dista_distrib_autocor( self ):
        '''
        plots dista_distrib_autocor
        '''
        
        plt.plot(np.arange(self._min_hop_shift,self._max_hop_shift) \
                *self.gcd/float(self.sampling_rate), \
                self.dista_distrib_autocor)
        plt.title('autocorrelation of distribution of onset distances')
        plt.xlabel('lags (s)')
        plt.ylabel('autocor')
        plt.show()


    def calc_bpms( self ):
        '''
        aggregate function that calculates likely tempos as the strongest 
        autocorrelation lag of the onset distance distribution
        '''
        self.calc_dista_distrib()
        self.calc_dista_distrib_autocor()
        
        self.plot_dista_distrib_autocor()
       
        dista_distrib_autocor_enum = \
                [ [self.dista_distrib_autocor[i], 
                    (self._min_hop_shift+ i) *self.gcd/float(self.sampling_rate)] \
                    for i in range(0, self._max_hop_shift - self._min_hop_shift) ]
  
        dista_distrib_autocor_enum.sort(reverse=True)
        self.autocor_peaks = [ dista_distrib_autocor_enum[i] for i in range(20)]
        self.bpms = [ 60.0/float(self.autocor_peaks[i][1]) for i in range(20)]
        
        print 'Highest values at lags: ', self.autocor_peaks
        print 'corresponding to bpm values of: ', self.bpms

class bbur_io:
    '''
    class used for writing and reading from data files
    '''
    
    def __init__( self ):
        self.data_folder_loc = '/'.join(os.getcwd().split('/')[:-1]) + '/data'
        ##@var data_folder_loc
        #@brief string representing the path where the 'data' folder is
        self.data_folder_name = ''
        ##@var data_folder_name
        #@brief string representing the name of a folder in the 'data' 
        self.data_file_name = ''
        ##@var data_file_name
        #@brief string representing the name of our data files
        self.n_freq_int = 1
        ##@var n_freq_int
        #@brief number of binary states at a given subdivision (e.g. BarBit.n_freq_int)
        

    def load_data( self ):
        '''
        reads the .dat file and returns its lines in obs, a list of
        observetions
        
        @returns A list of observations in integer form 
        @returns A list of observations in binary form
        '''
        obs = np.loadtxt( '/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '.dat', dtype=int, delimiter=',')
        obs_bin = int_to_bin_obs( obs, self.n_freq_int )

        return obs, obs_bin

    def append_data( self, observations ):
        '''
        writes a list of observations to the .dat data file

        @param observations List of observation in integer form
        '''
        datafile = open( '/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '.dat', 'a')
        for i in range(len(observations)):
            for s in range(len(observations[i])-1):
                datafile.write( str(observations[i][s]) + ',' )
            datafile.write( str(observations[i][-1]) + '\n' )
        datafile.close()

    def load_hmm( self, n_markov_states=20 ):
        '''
        returns an initialized barbiturythme.discrete_hmm using, if possible, 
        the parameters saved to the .npy data files

        @param n_markov_states Number of hidden states, if none in files (=20)
        @returns A barbiturythme.discrete_hmm
        '''
        hmm = barbiturythme.discrete_hmm( 2**self.n_freq_int, n_markov_states ) 
        
        try:
            A = np.load('/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_A.npy')
            if (A.shape[0] == A.shape[1]):
                hmm.n_markov_states = A.shape[0]
                hmm.ini_trans_matrix = A
            else:
                1./0.
        except:
            print 'no valid ' + '/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_A.npy'
            print 'new file will be created'
        
        try:
            B = np.load('/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_B.npy')
            if (B.shape[0] == A.shape[0]):
                hmm.ini_b = B
        except:
            print 'no valid ' + '/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_B.npy'
            print 'new file will be created'
        
        try:
            Pi = np.load('/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_Pi.npy')
            if (len(Pi) == A.shape[0]):
                hmm.ini_markov_state = Pi
        except:
            print 'no valid ' + '/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_Pi.npy'
            print 'new file will be created'
        
        return hmm
    
    def load_pgbe( self, rho ):
        '''
        returns an initialized barbiturythme.pgbe using, if possible,
        the parameters saved to the .npy data files

        @param rho barbiturythme.PGBE.rho
        @returns A barbiturythme.discrete_hmm
        '''
        pgbe = barbiturythme.PGBE( rho )
        
        try:
            p_temp = np.load('/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_p.npy')
            if (p_temp.shape[0] == p_temp.shape[1]):
                pgbe.rho = p_temp.shape[0]
                pgbe.c = p_temp.shape[2]
                pgbe.p = p_temp
            else:
                1./0.
        except:
            print 'no valid ' + '/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_p.npy'
            print 'new file will be created'
        
        try:
            w = np.load('/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_w.npy')
            if (w.shape == p_temp.shape):
                pgbe.w = w
            else:
                1./0.
        except:
            print 'no valid ' + '/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_w.npy'
            print 'new file will be created'

        return pgbe
        

    def save_hmm( self, hmm ):
        '''
        writes the relevant parameters of hmm to .npy files

        @param hmm barbiturythme.hmm
        '''
        np.save('/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_A', hmm.model.A)
        np.save('/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_B', hmm.model.B)
        np.save('/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_Pi', hmm.model.Pi)
    
    def save_pgbe( self, pgbe ):
        '''
        writes the relevant parameters of pgbe to .npy files

        @param pgbe barbiturythme.PGBE
        '''
        np.save('/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_p', pgbe.p)
        np.save('/'.join([self.data_folder_loc,self.data_folder_name,self.data_file_name]) + '_w', pgbe.w)

    def print_score_3bands( self, x_bin ):
        '''
        writes the observation x_bin into a csound score with three
        instruments kick drum, snare and hihat

        @param x_bin An observation (sequence of ints in 
        {0,...,2**io.n_freq_int-1}
        '''
        scorefile = open( '/'.join(os.getcwd().split('/')) + self.data_file_name + '.sco', 'w')
        scorefile.write('f1 0 0 1 "kick.wav" 0 0 0\n')
        scorefile.write('f2 0 0 1 "snare.wav" 0 0 0\n')
        scorefile.write('f3 0 0 1 "hihat.wav" 0 0 0\n\n')

        bpm = 72.0
        one_beat = 60.0/bpm
        subdiv_per_beat = 4
        one_subdiv = one_beat/float(subdiv_per_beat)
        scale_fact = one_subdiv
        
        d = [0.72, 0.46, 0.084]
        f = 60.0
        decib = [95,95,95]
        for t in range(len(x_bin)):
            for freq_interv in range(self.n_freq_int):
                if (x_bin[t][freq_interv]==1):
                    time =  2. + t*scale_fact
                    instr = freq_interv+1
                    string = 'i%1d %.4f %.3f %.2d %2d\n' % (instr, time, d[freq_interv], f, decib[freq_interv])
                    scorefile.write(string)


