# Experiments

Two sets of experiments were conducted using GANs after they had converged. The first set involved using GANs for a classification task. Here, each data sample was labeled, and the GANs were tasked with detecting anomalies. The objective was for each generator to label data samples from the class it was trained on as normal and the rest as anomalous. A threshold was established: data samples with an anomaly score below the threshold were considered normal, while those with a score above were considered anomalous. The expectation was that anomalous class data samples would have higher anomaly scores than those from the normal class. This experiment was repeated six times, with a generator trained using GAN and WGAN for each class.

The second set of experiments involved unsupervised anomaly scoring. Data samples were fed into the generator without providing any ground truth. The generator assigned an anomaly score to  each sample, which was then ranked from lowest to highest. Similar to the first set of experiments, it was expected that data samples belonging to the same class as the data the generator was trained on would receive lower anomaly scores.

| Label Category | Model | Maximum Accuracy (%) | Optimal Accuracy (%) | AUC | Average Loss for Rec  | Average Loss for Cyl  | Average Loss for Sph  |
|----------------|-------|----------------------|----------------------|-----|------|------|------|
| Rectangle      | GAN   | 80                   | 64                   | 0.71| **0.09** | 0.11 | 0.13 |
|                | WGAN  | 78.7                 | 62.0                 | 0.68| **0.05** | 0.05 | 0.09 |
| Cylinder       | GAN   | 78                   | 68                   | 0.76| 0.14 | **0.08** | 0.20 |
|                | WGAN  | 86.7                 | 78                   | 0.87| 0.08 | **0.03** | 0.12 |
| Spheres        | GAN   | 82.0                 | 76.0                 | 0.79| 0.17 | 0.13 | **0.10** |
|                | WGAN  | 87.3                 | 72.0                 | 0.81| 0.11 | 0.07 | **0.04** |

*Table 1: Accuracies, average loss, and AUC for each category using both GANs and WGANs. The average loss shows the rectangular, cylindrical, and spherical loss for each generator's class training data. The lowest average loss for each class is written in bold.*



## Experiment 1: Supervised Classification

To classify data as anomalous or normal, a threshold is set to determine the generator's accuracy. A data sample with an anomaly score below the threshold is considered normal, while one above is deemed abnormal. Thus, the threshold value is crucial for the model's accuracy.

<p align="center">
  <img src="https://github.com/user-attachments/assets/45dcb6a1-ba31-4adf-bab4-6698f1b23062" alt="sph_wgan_per_metrics_vs_th" width="50%" />
  <br>
  <i>Figure 1: Overall accuracy, true positive, and true negative rate for data samples of a DCGAN trained with WGAN on the spherical dataset as a function of threshold</i>
</p>

Figure 1 illustrates the different accuracies of a generator trained using WGAN on a dataset of spherical 3D models at various threshold values. When the threshold is set to 0, all normal and anomalous data points are classified as abnormal, resulting in a true positive value of 0 and a true negative value of 100. Conversely, if the threshold is set very high, all data points are classified as normal, leading to true positive and true negative values of 100 and 0, respectively. 

Given that the test set contains an equal number of data points for each of the three classes, the overall accuracy is 66% when the threshold is 0, as all abnormal data points are correctly classified, and 33% when the threshold is high, as all normal data points are correctly classified. Due to this trade off, two types of accuracies are determined for each label's WGAN and GAN generator: Maximum and Optimal accuracy. Maximum accuracy is the accuracy of the model at its highest, while optimal accuracy is the accuracy of the model when the trade off between true positive and negative is at a minimum. Effectively, optimal accuracy is found to be the accuracy of when the true positive and true negative converge to the same value.

It is important to note that the anomaly score (loss) of each data sample determines the model's overall accuracy. If the average loss of the generator's respective label is lower than that of other labels, the generator's overall accuracy should be higher. As shown in Table 1, the average loss for each type of generator is indeed lowest for its respective label, confirming this relationship.

<p align="center">
  <img src="https://github.com/user-attachments/assets/728d8b53-00cf-420b-8426-a22887d44606" alt="sph_wgan_per_metrics_vs_th" width="110%" />
  <br>
  <i>Figure 2: ROC and AUC values for WGAN and GAN generators for every class</i>
</p>

A better method of exploring the performance of the generators could be to look at their respective ROC curves and AUC scores. ROC curves show a trade off between the true positive rate as a function of the false positive rate. Figure 2 shows the ROC curve for each model type with their respective class of training data. On average, generators trained using WGAN recieved a slightly higher or significantly higher AUC score than GAN, with the exception of Rectangle generators where WGAN scored slightly below GAN. Rectangular Generators scored the highest AUC score of **0.71**, Cylindrical **0.87**, and Spheres **0.81**. 

## Experiment 2: Unsupervised Anomaly Scoring
Experiment 2 is to rank the data samples based on their anomaly score. Samples are ranked based on what the generator deemed from least anomalous to most anomalous. Figure 3 shows the ranking for generator trained on cylindrical data using wgan. The generator ranked their own respective label to be the least anomalous. The purpose of this experiment is to show another practical usage of the anomalous scoring algorithm.

<p align="center">
  <img src="https://github.com/user-attachments/assets/86db159a-bcf6-4d02-8c27-525c85f2049d" alt="sph_wgan_per_metrics_vs_th" width="80%" />
  <br>
  <i>Figure 3: Generated samples through multiple iterations of latent search with low and high generator learning rate for the rectangular GAN. Samples with low generator learning rates remain in the same realm of rectangular boxes, while samples with high generator learning rates changes the generator's $p_{\theta}$ to no longer belonging within the realm of rectangular boxes.</i>
</p>
