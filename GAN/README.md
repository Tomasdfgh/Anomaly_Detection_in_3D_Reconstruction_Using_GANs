# GAN

This directory contains the scripts for the architecutre of the DCGAN model, GAN training, and WGAN training. The torch trace of every generators trained for this project can be found in the Model directory. This read me file will breakdown the theory behind GANs and WGANs.

## Generative Adversarial Networks (GANs)

GANs represent a significant advancement in the field of generative modeling. GANs consist of two neural networks: the generator $G_{\theta}$ and the discriminator $D_{\omega}$. These two networks are trained simultaneously through a competitive game process, where the generator creates data and the discriminator evaluates its authenticity. This section delves into the underlying theory of GANs, explores the roles and interactions of both the generator and the discriminator in detail, and discusses the stability improvements introduced by Wasserstein GANs (WGANs). Additionally, it highlights the significant contributions of WGANs to more robust and effective generative modeling, emphasizing their impact on reducing training instability and improving the quality of generated data.

## Theory Behind GANs

The core concept of GANs revolves around a min-max optimization problem where the generator $G_{\theta}$ and the Discriminator $D_{\omega}$ engage in a two-person game. The objective of the Generator is to generate data samples that are indistinguishable from real data samples, while the discriminator is tasked with distinguishing between real and generated samples. This interaction can be formalized using the equation

<div align="center">
  
$\min_G \max_D E_{x \sim p}{[\log D_{\omega}(x)]} + E_{z \sim p_z}{[\log (1 - D_{\omega}(G_{\theta}(z)))]}$
</div>

Where:
- $p$ represents the distribution of real data samples.
- $p_z$ is the prior distribution of the latent space. This paper uses normal distribution
- $G_{\theta}(z)$ denotes the generator's output given a latent vector $z$
- $D_{\omega}(x)$ is the probability that a given sample $x$ is real

With this GAN objective function setup:

- The generator $G_{\theta}$ aims to minimize $\log(1 - D_{\omega}(G_{\theta}(z)))$, which is the log probability of the discriminator classifying a generated sample as fake.
- The discriminator $D_{\omega}$ aims to maximize $\log(D_{\omega}(x)) + \log(1 - D_{\omega}(G_{\theta}(z)))$, which is the log probability of correctly classifying real and fake samples.

## Mapping Latent Space to Data Space

Generative Adversarial Networks (GANs) map latent vectors $z$ from the latent space, following a normal distribution in this experiment, to the data space, where the generator $G_{\theta}$ approximates the data distribution $p$. The generator transforms these noise vectors $z$ into samples mimicking the data distribution $p$. Ideally, if well-trained, the generated data distribution $p_{\theta}$ should closely approximate the real data distribution, such that $p_{\theta} \sim p$. This can be effectively used for anomaly detection. Since $p_{\theta} \sim p$, if a data sample falls outside of $p$, it likely falls outside of $p_{\theta}$. Consequently, the generator's ability to produce data similar to the real distribution helps it flag non-conforming data, thus detecting anomalies.

<p align="center">
  <img src="https://github.com/user-attachments/assets/531aa082-5b26-4b24-b283-ceea82945bfc" alt="GAN Reconstruction" width="85%" />
  <br/>
  <i>Figure 1: Generated examples for generators trained on rectangular, cylindrical, and spherical dataset using the normal GAN training process. Each example shown in this figure contain an RGB and Depth Image</i>
</p>

## Wasserstein Generative Adversarial Networks (WGANs)

### Challenges in Training GANs

Generative Adversarial Networks (GANs) have revolutionized data generation tasks but are notorious for being challenging to train. The primary difficulties include mode collapse, where the generator produces limited varieties of samples, and training instability, often due to the discriminator overwhelming the generator. These issues are typically exacerbated by the use of Jensen-Shannon divergence as the metric for training, which can cause gradients to vanish, leading to poor updates for the generator.

### Wasserstein GAN: A Solution}
The Wasserstein GAN (WGAN) addresses these challenges by replacing the Jensen-Shannon divergence with the Wasserstein distance (or Earth Mover's distance) to measure similarity between real and generated data distributions. The Wasserstein distance provides a smoother gradient landscape, which helps prevent the vanishing gradient problem and allows for more stable generator updates.

WGANs use a critic network instead of a discriminator to estimate the Wasserstein distance between data distributions. The critic is constrained to be Lipschitz continuous, typically enforced by weight clipping or gradient penalty methods.

### WGAN Objective Function
The objective function for WGAN is:

<div align="center">
  
$\min_G \max_{C \in \mathcal{C}} E_{x \sim p}[C(x)] - E_{z \sim p_z}[C(G(z))]$
</div>

Where:
- $G_{\theta}$ is the generator
- $C$ is the critic network
- $\mathcal{C}$ is the set of 1-Lipschitz functions
- $p$ is the real data distribution
- $p_z$ is the latent space distribution, normal distribution for this paper

The critic maximizes the difference between its output expectations for real and generated samples, while the generator minimizes this difference, aiming to produce indistinguishable samples from real data.

### Impact and Results

WGANs have significantly improved the generation of diverse and realistic samples compared to traditional GANs. The stable training process leads to better convergence and reduces issues like mode collapse. The success of WGANs has led to widespread adoption and innovations, including WGAN-GP (WGAN with Gradient Penalty), which addresses the limitations of weight clipping. In this paper, generators are trained using both the traditional GAN training technique as well as WGAN training technique, and both employed the same DCGAN architecture. Visually, both training methods successfully captured the data distribution of their respective training datasets. However, the models trained with WGANs produced results that were clearer and more accurately representative of the dataset.

<p align="center">
  <img src="https://github.com/user-attachments/assets/22325b1d-7bd0-4f4c-9795-6ae8ce508023" alt="GAN Reconstruction" width="85%" />
  <br/>
  <i>Figure 2: Generated examples for generators trained on rectangular, cylindrical, and spherical datasets using the WGAN training process. Each example shown in this figure contains an RGB and Depth Image.</i>
</p>
