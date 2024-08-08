
![Blank diagram(1)](https://github.com/user-attachments/assets/1238b477-6651-4c25-8f51-3549290ad56d)

# Anomaly Detection in 3D Reconstruction Using GANs
Research conducted at Osaka University during a 2024 ESROP Global summer research position

## Introduction
Anomaly detection is a critical task in machine learning, where the goal is to identify instances in data that deviate significantly from the norm. This problem is particularly challenging in high dimensional spaces, such as those encountered in 3D reconstruction, where traditional methods often struggle due to the complexity and volume of data. Generative Adversarial Networks (GANs) have emerged as a powerful tool for anomaly detection, particularly in scenarios where abnormal data is scarce or hard to define. GANs are capable of learning the distribution of normal data and can be used to detect anomalies by assessing how well new data fits within this learned distribution. A GAN consists of two neural networks: a generator, which creates data samples, and a discriminator, which distinguishes between real and generated samples. The generator aims to produce samples indistinguishable from the real data, while the discriminator learns to identify which samples are real and which are generated. This adversarial process encourages the generator to create increasingly realistic samples, effectively learning the underlying data distribution. This paper builds on the method introduced by Deecke et al. [1], who demonstrated the effectiveness of GANs for anomaly detection in 2D images. Their approach is extended in this paper to the domain of 3D reconstruction, focusing on the identification of anomalies in 3D datasets. By leveraging the power of GANs, the paper’s aim is to develop a robust system for detecting anomalies in complex, high-dimensional 3D data, which could have significant implications for various applications such as quality control in manufacturing, medical imaging, and autonomous systems. Our approach adapts the GAN framework to handle the unique challenges posed by 3D data, providing a novel contribution to the field of anomaly detection.

## Repository Overview

- **Anomaly Detection Scripts:** This folder contains the script of the algorithm used to get the generator to assign an anomaly score to data samples, and the read me file has a detailed explaination of the algorithm, and the results.
- **Data:** Data folder contains a sample of 500 data points per label for each label. It will also have a breakdown of the dataset used in this experiment and a definition for anomalous data.
- **GAN:** GAN folder contains the model's architecture and training code. The read me file shows generated samples that the GAN and WGAN generators have created and the theory behind GAN and WGAN.
- **Generated Samples:** Generated samples contain examples of generators of each label for both GAN and WGAN.
- **Graphs and Plots** Contains all figures, graphs and plots from experiments. Pretty self-explanatory... LOL
- **Obtain Data Scripts:** Shows the scripts needed to use the RGBD Camera and obtain the datapoints. The process of data processing will also be explained in the read me file.

## References

[1] L. Deecke, R. Vandermeulen, L. Ruff, S. Mandt, and M. Kloft, “Image anomaly detection with generative
adversarial networks,” in Machine Learning and Knowledge Discovery in Databases: European Conference,
ECML PKDD 2018, Dublin, Ireland, September 10–14, 2018, Proceedings, Part I 18, pp. 3–17, Springer, 2019.

[2] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio,
“Generative adversarial nets,” in Advances in Neural Information Processing Systems (Z. Ghahramani,
M. Welling, C. Cortes, N. Lawrence, and K. Weinberger, eds.), vol. 27, Curran Associates, Inc., 2014.

[3] A. Radford, L. Metz, and S. Chintala, “Unsupervised representation learning with deep convolutional generative
adversarial networks,” arXiv preprint arXiv:1511.06434, 2015.

[4] M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein generative adversarial networks,” in International conference
on machine learning, pp. 214–223, PMLR, 2017.

[5] M. F. Augusteijn and B. A. Folkert, “Neural network classification and novelty detection,” International Journal
of Remote Sensing, vol. 23, no. 14, pp. 2891–2902, 2002.

[6] J. An and S. Cho, “Variational autoencoder based anomaly detection using reconstruction probability,” Special
lecture on IE, vol. 2, no. 1, pp. 1–18, 2015.

[7] X. Xia, X. Pan, N. Li, X. He, L. Ma, X. Zhang, and N. Ding, “Gan-based anomaly detection: A review,”
Neurocomputing, vol. 493, pp. 497–535, 2022.

[8] P. Bergmann, X. Jin, D. Sattlegger, and C. Steger, “The mvtec 3d-ad dataset for unsupervised 3d anomaly
detection and localization,” arXiv preprint arXiv:2112.09045, 2021.

___
### Acknowledgements
A special thanks to to Professor Daisuke Iwai and Professor Sato for their constant stewardship and supervision through out this project
