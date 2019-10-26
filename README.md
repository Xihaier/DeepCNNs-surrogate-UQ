# DeepCNNs-surrogate-UQ
Code for CACAIE paper [Deep convolutional neural networks for random field uncertainty propagation](https://xihaier.github.io/)

[Xihaier Luo](https://xihaier.github.io/), [Ahsan Kareem](https://engineering.nd.edu/profiles/akareem)

## Description
The development of a reliable and robust surrogate model is often constrained by the dimensionality of the problem. For a system with high-dimensional inputs/outputs (I/O), conventional approaches usually use a low-dimensional manifold to describe the high-dimensional system, where the I/O data is first reduced to more manageable dimensions and then the condensed representation is used for surrogate modeling. In this study, a new solution scheme for this type of problems based on a deep learning approach is presented. The proposed surrogate is based on a particular network architecture, i.e. convolutional neural networks. The surrogate architecture is designed in a hierarchical style containing three different levels of model structures, advancing the efficiency and effectiveness of the model in the aspect of training. To assess the model performance, uncertainty quantification is carried out in a continuum mechanics benchmark problem. Numerical results suggest the proposed model is capable of directly inferring a wide variety of I/O mapping relationships. Uncertainty analysis results obtained via the proposed surrogate have successfully characterized the statistical properties of the output fields compared to the Monte Carlo estimates.

## Contents

## Dependency
* Python 3.0 and above
* TensorFlow 1.0 and above
* Matplotlib 3.0 and above
* Scipy 1.3 and above
* Numpy 1.10 and above

## Citation
