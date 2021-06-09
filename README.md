# CMB decomposition using StyleGAN2
**Author**: Manuel Blanco Valentin ([manuel.blanco.valentin@gmail.com](mailto://manuel.blanco.valentin@gmail.com))
<br/> **Based on**: [StyleGAN2-Tensorflow-2.0](https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0) <br/>
**Collaborators**: João Caldeira, Brian Nord, Kimmy Wu, Clécio R. Bom
<br/>**Affiliation**: Computer & Electrical Eng. Department - Northwestern University
---

## Table of Contents
1. [Introduction to CMB](#Introduction-to-CMB)
2. [CMB decomposition](#CMB-decomposition)
3. [Previous Work](#Previous-work)
4. [Proposed Methodology](#Proposed-methodology)
5. [Requirements](#Requirements)
6. [astroDGAN building blocks](#astroDGAN-building-blocks)
7. [How to use astroDGAN](#How-to-use-astroDGAN)
8. [Experiments](#Experiments)
9. [Further tests and discussion](#Further-tests-and-discussion)
10. [Future work](#Future-work)


## Introduction to CMB
[[go back to the top]](#Table-of-Contents)
The evidences that Hubble found about the apparent redshift of galaxies relative to each other (and us), as well as the 
recently proposed (at the time) theory of general relativity seem to indicate
that the fabric of space-time in our universe is constantly expanding at an accelerated rate.

These discoveries made Lemaître conclude that in order for the universe to be expanding 
its fabric, it must had begun in an extremely small and dense state. In fact, stretching this contraction back
until the limit, our current mathematical formulations and physical laws guide us to the contradiction of the singularity:
 the whole matter of the universe, condensed in an infinitessimal amount of space, with an extreme local density.

There is a point at which applying such equations and laws does not make any real sense, and most physicists even have trouble 
accepting the concept of the singularity itself. Instead they prefer to conclude that our current understanding of the 
universe is somehow limited, and thus the singularity itself is simply a consequence of our flawed science. 

Despite this, our current understanding of most part of the history of the universe is pretty extense. We might not be able
to apply our techniques to understand how the universe was before a certain point in time (when the universe was only nanoseconds
old), but we still can apply them to know how it was after that moment; in the part of the history of our universe that where
our science still holds. 

Right after the big bang, in a period known as **inflation**, the universe was so hot and condensed that atoms could not form.
Subatomic particles existed in a sort of quantic soup in form of plasma, due to the immense pressure and temperature. 
At this stage, the photons that were emitted during the interaction of the particles were immediately reabsorbed by other
particles, thus preventing them (and thus, light) to escape and to propagate outside of the plasma object. This phenomenon occurs, e.g.,
inside of our sun, where light might take up to 100,000 years to escape (see [here](https://www.abc.net.au/science/articles/2012/04/24/3483573.htm#:~:text=A%20photon%20of%20light%20takes,at%20the%20speed%20of%20light.)).

Once the universe was able to expand enough and cool down, the pressure was reduced, which allowed atoms to be formed. The
reduced density of the universe allowed the photons to escape and propagate along with the expanding universe for the very first time.
This last emission of light precisely describes how the structure of the universe was right after the big bang, how matter
was organized and distributed, which holds the key to understanding, for example, why there exist clusters of matter,
why there is an imbalance in the proportion of matter/antimatter, or even what is the origin of dark matter. 

This __snapshot__ of the early universe is precisely what we call **Cosmic microwave background radiation** or CMB. This radiation
can be seen absolutely everywhere in the universe, no matter where we point our telescopes to, and it always has the same
structure. 

![CMB_bigbang](res/cmb2.jpeg)

**Figure 1.** Depiction of the evolution of the early universe right after the big bang.


![CMBplank](res/cmbplank.jpeg)

**Figure 2.** Real image of the CMB captured by the satellite Planck.

## CMB decomposition
Although CMB contains a great amount of information regarding a great range of physical phenomena, the data in the raw maps as 
shown in **Figure 2** (in µK) is not directly useful. In order to extract useful information from these maps, it is necessary to
decompose them into different submaps, each one of them related to a certain property or phenomenon. 

![eqs](res/eqs.png)

**Figure 3.** Analytical equations used to decompose maps.

<br/>

| map | img |
|:---:|:---:|
| q | ![q](res/q.png) | 
| u | ![u](res/u.png) | 
| k | ![k](res/k.png) | 
| e | ![e](res/e.png) | 
| b | ![b](res/b.png) |

**Figure 4.** Example of decomposition of CMB.


The workflow that we are required to implement consists on:
```bash
- Convert maps q,u -> k,e,b
- Image2Image transformation
```


## Previous work
[[go back to the top]](#Table-of-Contents)

Previous work on lensing reconstruction of CMB with machine learning showed competitive results even for different levels of noise (see [DeepCMB][1]). On J. Caldeira's et al. work a UResNet is used in a image transformation problem and trained so that **e** and **k** maps can be obtained from sourced **q** and **u** lensed maps.  

The following image shows the schematic of the methodology proposed:
![Previous work methodology URESNET](images/previous_work_qu_ek_uresnet.png)
The following images display the average spectra for e and kappa modes for both original and predicted samples.
![Previous work e map recovery](images/e_previous.png)
![Previous work kappa map recovery](images/k_previous.png)

[1]: <https://www.sciencedirect.com/science/article/pii/S221313371830132X> "DeepCMB: Lensing reconstruction of the cosmic microwave background with deep neural networks"


## Proposed methodology
[[go back to the top]](#Table-of-Contents)

We propose to use adversarial networks (GANs) to improve the map retrieving accuracy. Adversarial networks are based on the combination of two different networks that compete with each other during the training procedure: a generator network (designed to generate fake images from true sources) and a discriminator network (designed to be able to distinguish between true and fake/generated images). 

It was found that each one of the maps to be predicted behaves differently to the type of generator used for optimization. Two different generators were tested: João Caldeira's Uresnet used in the astroencoder project and a Unet based on google's implementation of Pix2Pix algorithm (see [Pix2Pix][2]). Results are shown in the following sections.

```
Disclaimer: Due to confidentiality agreements between 
Fermilab and NU, the actual data cannot be posted on 
github. If you want, you can use this code for a 
different application/dataset.
```

[2]: <https://www.tensorflow.org/tutorials/generative/pix2pix> "Pix2Pix"


## Requirements
[[go back to the top]](#Table-of-Contents)

Take a look at the requirements to run this code:

```python
numpy~=1.19.5
matplotlib~=3.4.2
tensorflow~=2.5.0
Pillow~=8.2.0
DynamicTable~=0.0.4
```

You can install [DynamicTable](https://github.com/manuelblancovalentin/DynamicTable) via pip:

`pip install DynamicTable`


## Results

![quekb_spectra](res/quekb_spectra.gif)


