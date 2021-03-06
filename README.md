# DeepCNNs-surrogate-UQ
The repository contains all files of the *CACAIE paper* entitled [Deep convolutional neural networks for uncertainty propagation in random fields](https://onlinelibrary.wiley.com/doi/full/10.1111/mice.12510). Browse each of the folders for more information.
* The Examples folder provides the computer codes.
* The Responses folder provides peer reviews from 9 reviewers.
* The Manuscript folder provides the revised paper.

## Summary
A machine learning approach is proposed for quantifying the effect of spatial variabilities in coupled elliptic systems. The learning model takes a hierarchical form where deep convolutional neural networks are used as the underlying components.

<p><img src="Images/truth.png" title="ground truth" width="260"> <img src="Images/prediction.gif" title="prediction" width="270"> <img src="Images/error.gif" title="error" width="260"><p>

The learning process of this field-to-field mapping is efficient as distant connections among nonadjacent layers are established to improve the model efficiency in terms of training and deploying.

<p><img src="Images/optimization.gif" width="700"><p>

## Citation
The paper has been accepted by the Journal of Computer-Aided Civil and Infrastructure Engineering. The below information contains the references of Journal. 

```latex
@article{luo2019deep,
  title={Deep convolutional neural networks for uncertainty propagation in random fields},
  author={Luo, Xihaier and Kareem, Ahsan},
  journal={Computer-Aided Civil and Infrastructure Engineering},
  year={2019},
  publisher={Wiley Online Library}
}
```


## Dependency
* Python 3.0
* TensorFlow 1.3
* Matplotlib 3.0
* Scipy 1.3
* Numpy 1.10

## Dataset
We provide the data to help researchers reproduce the results. Please place the data in a proper directory.

[Link to the datasets](https://drive.google.com/drive/folders/1uyrN4RGuNNU_ya4nlvD40XrOpwtFSC2n?usp=sharing)
