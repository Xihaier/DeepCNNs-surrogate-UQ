# DeepCNNs-surrogate-UQ
Code for CACAIE paper [Deep convolutional neural networks for random field uncertainty propagation](https://xihaier.github.io/)

## Summary
A machine learning approach is proposed for quantifying the effect of spatial variabilities in coupled elliptic systems. The learning model takes a hierarchical form where deep convolutional neural networks are used as the underlying components.

<p><img src="Images/truth.png" width="260"> <img src="Images/prediction.gif" width="270"> <img src="Images/error.gif" width="260"><p>

The learning process of this field-to-field mapping is efficient as distant connections among nonadjacent layers are established to improve the model efficiency in terms of training and deploying.

<p><img src="Images/optimization.gif" width="700"><p>

## Contents
* Manuscript: revised paper.
* Responses: peer reviews from 9 reviewers. 
* Examples: computer codes of the case study.

## Dependency
* Python 3.0
* TensorFlow 1.3
* Matplotlib 3.0
* Scipy 1.3
* Numpy 1.10

## Citation
