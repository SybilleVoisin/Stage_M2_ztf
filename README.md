# Study and optimization of PSF processing for the ZTF experiment

## Description

As part of my M2 internship at the Centre de Calcul [CC-IN2P3](https://cc.in2p3.fr/), I optimized the PSF (Point Spread Function) code for the [ZTF](https://www.ztf.caltech.edu/) (Zwicky Transient Facility) collaboration.
Located atop Palomar Mountain in California (USA), the ZTF telescope is an astronomy observatory. 
It was inaugurated in March 2018 and is supported jointly by an international partnership of universities and institutes of Europe and Asia and the US National Science Foundation. 
Its purpose is to scan the night sky for transient and variable astronomical phenomena such as SNeIa. 
To study supernovae, ZTF team uses an image processing pipeline that consists in cleaning the data (bias, flat, non-linearity effects, etc.) and then analyzing the images, in particular with the PSF.

The official PSF code is written in C++ to run on CPUs. Although it is already optimized, we wonder if it can be further optimized. 
The PSF code consists mainly of image processing. That is why the idea of parallelization on GPUs seems a good optimization solution.
For my part, I used [Python](https://www.python.org/) with [Google JAX](https://github.com/google/jax) on CPU and GPU to compare performance on different frameworks. 
I also did some profiling thanks to [Perfetto](https://perfetto.dev/docs/) to help to understand how my code behaved. 

My work involved using two different distributions to fit the PSF model (the [Gaussian](https://en.wikipedia.org/wiki/Gaussian_function) function and the [Moffat](https://en.wikipedia.org/wiki/Moffat_distribution) function), and testing different optimizers (in my case: [minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) from SciPy, [Adam](https://optax.readthedocs.io/en/latest/api/optimizers.html) from Optax and [TN-CG](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf) from the ztf collaboration). For more details, please read my [internship report](presentation/internship_report_Voisin_Sybille_01_06_2024.pdf).




## Installation

```bash
git clone https://github.com/SybilleVoisin/Stage_M2_ztf.git
```

**Dependencies**

You will need the following packages [ztfimg](https://github.com/MickaelRigault/ztfimg), [ztfin2p3](https://github.com/MickaelRigault/ztfin2p3) and [ztfquery](https://github.com/MickaelRigault/ztfquery) to get the data and the ztf pipeline.

I used a [Python (3.11.5)](https://www.python.org/downloads/release/python-3115/) environment overloading the ztf environment to work as close as possible to the official one.
I worked on NVIDIA V100 GPUs, at the time of testing, the CUDA version was 12.2, which limited the version of JAX compatible.

**Requirements**

Download requirements.json file.

```json
{
  "dependencies": {
    "numpy": "1.23.5",
    "scipy": "1.11.4",
    "astropy": "5.3.4",
    "matplotlib": "3.8.2",
    "pandas": "2.1.4",
    "ipython": "8.15.0",
    "jax": "0.4.26"
  }
}
```


- [NumPy](https://numpy.org/) (1.23.5)
- [SciPy](https://scipy.org/) (1.11.4)
- [Astropy](https://www.astropy.org/) (5.3.4)
- [matplotlib](https://matplotlib.org/) (3.8.2)
- [pandas](https://pandas.pydata.org/) (2.1.4)
- [IPython](https://ipython.org/) (8.15.0)
- [JAX](https://jax.readthedocs.io/en/latest/) (0.4.26)

## Licence

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](LICENCE.txt)

This license lets others remix, tweak, and build upon this work non-commercially, as long as they credit the authors and license their new creations under the identical terms.
