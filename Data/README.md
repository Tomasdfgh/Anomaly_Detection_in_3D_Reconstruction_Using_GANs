![Blank diagram(1)](https://github.com/user-attachments/assets/eada789a-0b44-42a3-b856-5ef6035a435d)

# Data
This directory contains a 500 data sample for each label. This read me file will go over the description of the dataset used within this experiment, and how anomalous data is defined.

## Description of the Dataset

Due to a lack of available 3D Modelling Data, a dataset is created as a part of this research. To test for anomalous data, there has to be multiple labels where one will be deemed as normal and the others anomalous. As a result, 3 types of data is created corresponding to 3 different labels: Rectangular, Cylindrical, and Spherical objects. Each label will contain 2000 different models of their respective object type at different angles and object sizes. 1500 scans will be allocated to the training set, while 500 will be allocated to the test set.

| Category    | # Train | # Test | Image Size    |
|-------------|---------|--------|---------------|
| Rectangular | 1500    | 500    | 1080 x 1080   |
| Cylindrical | 1500    | 500    | 1080 x 1080   |
| Spherical   | 1500    | 500    | 1080 x 1080   |

**Table 1:** Statistical breakdown of the dataset used for the anomaly detection task


## Defining Anomalous Data

While the definition of anomalous data can vary widely depending on its application, this paper defines anomalous data as those samples that lie outside the distribution $p$ of the primary dataset. For the dataset used in this study, each class: rectangle, cylinder, and spheres has its own distinct data distribution: $p_{rectangle}$, $p_{cylinder}$, and $p_{spheres}$ respectively. Given any classes, any data points from a different class would fall outside of its data distribution. This classification ensures that the identification of anomalies is precise and contextual to the specific distributions of each class within the dataset.
