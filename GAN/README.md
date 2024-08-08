# GAN

This directory contains the scripts for the architecutre of the DCGAN model, GAN training, and WGAN training. This read me file will breakdown the theory behind GANs and WGANs.

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
