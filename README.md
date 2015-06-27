 **Bar**: gather your best beat bars  
 **Bit**: encode them into computationally sound data  
 **U**: interact 
 **Rythme**: get new rhythms, with perhaps some exotic touch 


**BarBitURythme** is a Pyhton module for reading .wav audio data files, 
extracting their rhythm sequences, 'learning' from both their short 
and long term 'patterns' and generating new sequences of similar 'style'.


By Francois Charest <freinque.prof@gmail.com> 
and Arst D.H. Neio <arstdashneio@gmail.com>


## Examples

[Encoding/transcription example](http://freinque.github.io/barbiturythme/html/bbur_reader_test_0.0.4.html)

[Learning/generation example](http://freinque.github.io/barbiturythme/html/bbur_gener_test_0.0.4.html)


## Documentation
[freinque.github.io/barbiturythme/docs/html/index.html](http://freinque.github.io/barbiturythme/docs/html/index.html)

## Using: 

- A (long term) statistical model and optimization algorithm adapted from 
J.-F. Paiement, Y. Grandvalet, S. Bengio, D. Eck 
'A Distance Model for Rythms', Proc. 25th Int. Conf. on Machine Learning, 2008.

- **M. Hamilton's hmm.py module** for the EM training/optimization of Hidden Markov 
Models for short term dependencies.
See
[http://www.cs.colostate.edu/~hamiltom/code.html](http://www.cs.colostate.edu/~hamiltom/code.html) and 
L.R. Rabiner, 'A Tutorial on Hidden Markov Models and Selected Applications
in Speech Recognition', Proc. of the IEEE, Vol. 77, 2, 1989.

- **modal_modif.py an adaptation of J. Glover's modal module** for onset (e.g. 
sudden change in the signal) detection implementating spectral difference 
models considered by Masri (and perhaps some other people before). 
See  
[https://github.com/johnglover/modal](https://github.com/johnglover/modal) and 
J.P. Bello, L. Daudet, S. Abdallah, C. Duxbury, M. Davies, M.B. Sandler
'A Tutorial on Onset Detection in Music Signals', IEEE Trans. Speech and 
Audio Processing, Vol. 13, 5, 2005.

- The **scikit-learn** package for standard machine learning algorithms (kmeans 
clustering, etc.) 

- **csound** for .wav generation

## (non-)Installation:

-make a local copy of the barbiturythme repository

    git clone https://github.com/freinque/barbiturythme

-recommended for better visualisation/interaction: reach the 
barbiturythme/notebooks/ folder and
    
    ipython notebook --pylab=inline

## License:
GNU GPL2
