# CMB decomposition using StyleGAN2
**Author**: Manuel Blanco Valentin ([manuel.blanco.valentin@gmail.com](mailto://manuel.blanco.valentin@gmail.com))
<br/> **Based on**: :octocat: [StyleGAN2-Tensorflow-2.0](https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0) <br/>
**Collaborators**: João Caldeira, Brian Nord, Kimmy Wu, Clécio R. Bom
**Affiliation**: Computer & Electrical Eng. Department - Northwestern University
---

## Table of Contents

1. [Previous Work](#Previous-work)
2. [Proposed Methodology](#Proposed-methodology)
3. [Requirements](#Requirements)
4. [astroDGAN building blocks](#astroDGAN-building-blocks)
5. [How to use astroDGAN](#How-to-use-astroDGAN)
6. [Experiments](#Experiments)
7. [Further tests and discussion](#Further-tests-and-discussion)
8. [Future work](#Future-work)





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

[2]: <https://www.tensorflow.org/tutorials/generative/pix2pix> "Pix2Pix"


## Requirements
[[go back to the top]](#Table-of-Contents)

Take a look at the requirements to run astroDGAN:

```python
python >= 3.
tensorflow-gpu >= 2.1.0
DynamicTable >= 0.0.4
numpy ~= 1.18.1
matplotlib ~= 3.1.3
pickle ~= 4.0
```

You can install [DynamicTable](https://github.com/manuelblancovalentin/DynamicTable) via pip:

`pip install DynamicTable`


## astroDGAN building blocks
[[go back to the top]](#Table-of-Contents)

A COMPLETE DETAILED EXPLANATION ON HOW THIS MODULE WORKS AND ON EACH SINGLE FUNCTION, CLASS AND METHOD THAT BUILDS THE MODULE AND HOW IT IS ALL INTEGRATED CAN BE FOUND

 ===================> [here](./Docs/module_blocks.md) <===================

The following tree shows the overall structure of the module blocks (which are explained in detail in the link above).

```sh
astroDGAN
├── experiments.sh
├── main.py
└── utils
    ├── analytics.py
    ├── blocks.py
    ├── common.py
    ├── cs_nets.py
    ├── data_loader.py
    ├── history.py
    ├── losses.py
    └── models.py
```

## How to use astroDGAN
[[go back to the top]](#Table-of-Contents)

Using astroDGAN is as easy as it gets. Make sure you meet the package requirements as shown in [here](#Requirements). Then, all you need is to call the `main.py` file with the desired parameters and let the process running. 

You can get the message help with all available input arguments and flags using :

```console
python main.py -h
```

This will generate an output such as the one in the following image:

![Help console](images/help_console.png)

All the available flags and their meaning are shown below:


| Short flag | Long.                         |  Required| Default  | Comments           | e.g. |
|------------|:------------------------------|:--------:|:--------:|:-------------------|:-----|
| -h         | --help                        |          |          | Show help message  |      |
| -i         | --inputs                      |    ✅    |          | Input channels to model                | qu  |
| -o         | --outputs                     |    ✅    |          | Output channels to model                  | e   |
|            | --gpus                        |          |  None    | Gpus to be used for computation | 0123 |
|            | --seed                        |          |  483     | Random seed for reproducibility | 
|            | --data-dir                    |          |  ./data/ | Directory where data is located | ./DATA_DIR/... |
|            | --noise                       |          |  0.0     | Level of noise to be applied to maps | 0.1 |
|            | --noise-channels              |          |  qu      | Channels to which noise will be applied  | quek |
|            | --ndeg                        |          |  5.0     | Angular resolution of maps | 3.2 |
|            | --no-normalize-maps           |          |  False   | If present, maps are not normalized     | |
|            | --no-apodize-maps             |          |  False   | If present, maps are not apodized    | |
| -vs        | --validation-split            |          |  0.2     | Ratio of data to be used for validation | 0.1 |
| -a         | --architecture                |          |  pix2pix | Either pix2pix or astroencoder. Architecture to be used for the generator | |
|            | --no-adversarial              |          |  False   | If present, model is not adversarial (discriminator is ignored) | |
|            | --no-skip-connections         |          |  False   | If present, no skip connections are used | |
|            | --no-residuals                |          |  False   | If present, no residual connections are used | |
|            | --output-dir                  |          | ./results| Directory where results will be stored | ./OUTPUT_DIR|
|-gdrop      | --generator-dropout-ratio     |          | 0.5      | Dropout ratio for dropout layers in the generator | 0.7 |
|-ddrop      | --discriminator-dropout-ratio |          | 0.5      | Dropout ratio for dropout layers in the discriminator | 0.7 |
|            | --no-generator-use-bias       |          |  False   | If present, biases are not used in the generator | |
|            | --no-discriminator-use-bias   |          |  False   | If present, biases are not used in the discriminator | |
|            | --high-res-len                |          |  2       | Integer specifying the length of the high resolution output stem | 3 |
| -e         | --epochs                      |          |  100     | Integer specifying the number of epochs used to train the model |
|            | --lambda                      |          |  100.0   | Float specifying the weight of the l1-divergence loss when computing the global generator loss (see explanation on lambda in [here](./Docs/module_blocks.md#Generator)) | 10.0 |
|            | --discriminator-interval      |          |  20      | Integer specifying the number of steps between each consecutive time the discriminator is trained (generator is trained every step). If this index is 1 the generator is trained the same number of times as the generator. | 1 |
|            | --spectra-weight              |          |  0.0     | Float specifying how much the spectra must weight in the calculation of the final generator loss. If 0.0, the spectra is not computed. | 10.0 |


So as an example, if we wanted to generate `k` maps from `q,u` maps, using a pix2pix GAN with no residual connections, applying a noise of `0.2` to maps `u,k`,  without normalizing nor apodizing the maps, using a validation\_split of `0.35` and a dropout ratio of `0.3` for the generator and `0.75` for the discriminator, training the model for `200` epochs, using only GPU devices 1 and 2, and storing the result into folder `$HOME/RESULTS/astroDGAN_TEST/` the command issued should be:

```console
OUTPUT_DIR=$HOME/RESULTS/astroDGAN_TEST
python main.py -i qu -o k -a pix2pix --gpus 12 --noise 0.2 --noise-channels uk -vs 0.35 -gdrop 0.3 -ddrop 0.75 -e 200 --output-dir $OUTPUT_DIR
```

When main.py is called, and the model folder is created in `output-dir`, a file called `command` is created. Inside this file, the command to be used to reproduce the exact results is stored. 

For further examples on commands for different experiments and configurations, check the links in the [experiments section](#Experiments) below.

## Experiments
[[go back to the top]](#Table-of-Contents)

| Test                    | inputs   |  outputs | architecture (gen) | epochs | lambda | adversarial | gen_bias| residuals | skip | noise|
|-------------------------|:--------:|:--------:|:------------------:|:------:|:------:|:-----------:|:-------:|:---------:|:----:|:-----|
| [Test0](./Docs/Test0.md)|  q,u     | e        | astroencoder       |  150   | 100    |✅           |❌      |✅         |✅    | 0.0  |
| [Test1](./Docs/Test1.md)|  q,u     | k        | astroencoder       |  100   | 100    | ✅          |❌      |✅         |✅    | 0.0  |
| [Test2](./Docs/Test2.md)|  q,u,e,k | b        | pix2pix            | 250    | 10     | ✅          |❌      |❌         |✅    | 0.0  |


## Further tests and discussion
[[go back to the top]](#Table-of-Contents)

| Test | inputs |  outputs | Comments  |
|------|:------:|:--------:|:--------:|
| Test3|  q,u   | k        | Test1 showed overfitting in the discriminator, it would be interesting to check what happens when the discriminator is better trained, so that it truly identifies the fake samples. Combining this with, maybe, different values for lambda and more epochs for training we could achieve better retrieval of the high-freq signal.|
| Test4|  q,u   | e        | |
| Test5|  q,u   | k        | Minimize spectrum diff in loss too |
| Test6|  q,u   | e        | Minimize spectrum diff in loss too |
| Test7|  q,u,e,k | b        | Try different values for lambda, kernel size, dropout_ratio (to get rid of overfitting) |
| Test8|  q,u,e,k | b        | Two step process: First train the system using more or less the same configuration used in Test2 (so the generator can learn how to create B modes) and then train the generator+discriminator with a low discriminator interval value |
| Test9|  q,u,e,k,E,B | b    | Use analytic predicted E,B maps |
| Test10|  q,u,e,k,E,B | b    | Minimize spectrum diff in loss too |




## Future work
[[go back to the top]](#Table-of-Contents)

### Use E,B approximations
[[go back to section header]](#Future-work)

Two main issues are expected to be addressed with this work: 
* Improve the accuracy for retrieving e and k maps from q, u maps.
* Retrieve b maps from q, u maps.

There exists and analytical relation between **unlensed** Q, U and E, B maps which is given by (1),

![Analytical Equation](images/equation.gif)

. We know that this equation, in reality, yields untrusty results when applied to discrete size maps (images), corrupting the obtained signals usually with an effect known as __bleeding__. 

As the inputs to our model are **lensed** (q, u), if one wanted to use the analytical equation (1) to estimate the E and B modes, it would be necessary to, first, obtain the unlensed maps Q, U, which are inputs to (1). This can be achieved by using a deep network (GAN0/Delensing GAN), which can also be used at the same time to estimate the lensing map (k). Afterwards, the estimated delensed maps ~Q, ~U can be processed by the analytical equation (1) to estimate ~E and ~B, which will be corrupted due to the effects explained before. Later, the ~E and ~B maps can be inputed to second deep network (GAN1/Decomposition GAN) to correct the undesired effects introduced by the analytical equation (1) and thus obtain the final maps e, b. 

Discriminative models have been observed to perform much better in image reconstruction/generation tasks such as the one presented here. This is why both neural networks used in our proposed methodology are GANs. Apart from that, both of them are based on João Caldeira's UResNet. 

The workflow proposed for this problem can be seen in the following image, 

![GAN](images/gan.jpeg)


### Use different losses 
[[go back to section header]](#Future-work)

Take a look at this [article](https://towardsdatascience.com/metric-learning-loss-functions-5b67b3da99a5) 






