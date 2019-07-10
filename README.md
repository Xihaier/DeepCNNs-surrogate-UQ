# DeepCNNs-surrogate-UQ
Deep CNNs for UQ problems

Deep convolutional neural networks for random field uncertainty propagation
[Xihaier Luo](https://xihaier.github.io/), [Ahsan Kareem](https://engineering.nd.edu/profiles/akareem)

> The development of a reliable and robust surrogate model is often constrained by the dimensionality of the problem. For a system with high-dimensional inputs/outputs (I/O), conventional approaches usually use a low-dimensional manifold to describe the high-dimensional system, where the I/O data is first reduced to more manageable dimensions and then the condensed representation is used for surrogate modeling. In this study, we present a new solution scheme for this type of problems based on a deep learning approach. The proposed surrogate is based on a particular network architecture, i.e. the convolutional neural networks. The surrogate architecture is designed in a hierarchical style containing three different levels of model structures, advancing the efficiency and effectiveness of the model in the aspect of training and deploying. To assess the model performance, we carry out uncertainty quantification on a continuum mechanics benchmark problem. Numerical results suggest the proposed model is capable of directly inferring a wide variety of I/O mapping relationships. Uncertainty analysis results obtained via the proposed surrogate have successfully characterized the statistical properties of the output fields compared to the Monte Carlo estimates.