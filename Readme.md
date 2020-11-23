## ART(Adaptive Resonance Theory) py(thon) :

This just as fast(relatively) implementation on python of several ART algorithm for my own purposes.


I used this brilliant [article](https://arxiv.org/abs/1905.11437) as reference. A many thanks to its authors.

### List of implemented ARTs:
 - FuzzyART - unsupervised
 - HyperShpereArt - unsupervised

## Implementation details:
1. I used PyTorch as backend(but you can easy replace it e.g. with NumPy) for speed up vector computations.
2. I add the ```restrict_nodes``` param to restrict ART to grow in memory. This help to fit the algorithm when there is memory restrictions.






