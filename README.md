# Study and optimization of PSF processing for the ZTF experiment

## Description

As part of my M2 internship at the Centre de Calcul [CC-IN2P3](https://cc.in2p3.fr/), I optimized the PSF (Point Spread Function) code for the [ZTF](https://www.ztf.caltech.edu/) (Zwicky Transient Facility) collaboration.
Located atop Palomar Mountain in California (USA), the ZTF telescope is an astronomy observatory. 
It was inaugurated in March 2018 and is supported jointly by an international partnership of universities and institutes of Europe and Asia and the US National Science Foundation. 
Its purpose is to scan the night sky for transient and variable astronomical phenomena such as SNeIa. 
To study supernovae, ZTF team uses an image processing pipeline that consists in cleaning the data (bias, flat, non-linearity effects, etc.) and then analyzing the images, in particular with the PSF.

The official PSF code is written in C++ to run on CPUs. Despite the fact that it is optimized, the question is: can it be further optimized?
For my part, I used Python with [Google JAX](https://github.com/google/jax) on CPU and GPU to compare performance on different frameworks. 
I also did some profiling thanks to [Perfetto](https://perfetto.dev/docs/) to understand how my different codes behaved. 

## Installation

```bash
git clone https://github.com/SybilleVoisin/Stage_M2_ztf.git
```

**Dependencies**

You will need the following packages [ztfimg](https://github.com/MickaelRigault/ztfimg), [ztfin2p3](https://github.com/MickaelRigault/ztfin2p3) and [ztfquery](https://github.com/MickaelRigault/ztfquery) to get the data and the ztf pipeline.

I used a Python environment (3.11.5) overloading that of ztf environment to work as close as possible to the official one.

**Requirements**
- [NumPy](https://numpy.org/) (1.23.5)
- [SciPy](https://scipy.org/) (1.11.4)
- [Astropy](https://www.astropy.org/) (5.3.4)
- [matplotlib](https://matplotlib.org/) (3.8.2)
- [pandas](https://pandas.pydata.org/) (2.1.4)
- [IPython](https://ipython.org/) (8.15.0)
- [JAX](https://jax.readthedocs.io/en/latest/) (0.4.26)

## Licence

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](LICENCE.txt)
