![Blank diagram(1)](https://github.com/user-attachments/assets/1238b477-6651-4c25-8f51-3549290ad56d)

# Anomaly Detection Scripts

## Algorithm

The algorithm employed in this experiment is based on the approach used by Deecke et al., but it has been adapted to accommodate a 4-channel data frame (incorporating both RGB and Depth information, instead of just RGB). The core principle behind the algorithm is that if the generator has accurately captured the distribution of real data through training, then for any given normal data sample $x \sim p$, there exists a latent vector $z$ that enables the generator to produce a 3D model closely resembling the sample $x$. Conversely, if a data sample $x$ does not belong to the data distribution $p$, there will be no corresponding latent vector $z$ that can accurately reproduce the sample. In such cases, the model will struggle to find a suitable $z$, and the data sample will be classified as anomalous.

### Latent Space and Reconstruction Loss Propagation
The algorithm is performed by propagating through the latent space to find a latent vector that would allow $G_{\theta}(z)$ to most closely resemble $x$. $z_{0}$ is initialized from $p_z$, where $p_z$ is the distribution of the latent space. A reconstruction loss $l$ is obtained from $G_{\theta}(z_0)$ and $x$, defined as $l(G_{\theta}(z_0), x)$. This loss is propagated back to the latent space to find latent vector $z_1$. This process is repeated for $k$ steps, following through different generations of generated samples of $G_{\theta} \(z_0)$, $G_{\theta}(z_1)$ $, ...,$ $G_{\theta}(z_k)$, the anomaly score can be calculated from $l(G_{\theta}(z_k), x)$.

Since optimizing in the latent space is a non-convex problem, it is helpful to search through multiple latent vectors initialized at different locations to avoid seeding at an unsuitable region. As a result, the final anomaly score can be found by averaging the reconstruction loss of all the generated samples at different seeds after the $k^{th}$ search. Additionally, after every iteration of k, the recreation loss $l$ is also propagated to the generator's weights $\theta$. Even though these changes do not affect the weight of the generator permanently and is reset back to its original weight for each new testing point. Changing the weights of the generator would allow the generator itself to improve on its representative capacities. Mathematically, the overall anomaly score of a given data sample can be defined as the equation below

<div align="center">
  
$loss = \frac{1}{n_{seed}}\{ \sum_{n} l(G_{\theta_{n,k}}(z_{n,k}), x)}$
</div>

Where $z_k$ and $G_{\theta_k}$ is the latent vector and generator after $k^{th}$ iteration of reconstruction loss propagation. If the final loss value is low for that data sample, then the data is deemed as less anomalous, and vice versa.

### Anomaly Detection Algorithm using GANs for 4 Channels (RGBD) Data Frame

**Input:**
- Learning rate for latent vector and generator ($γ_z$, $γ_θ$)
- Number of seeds ($n_{seed}$)
- Data sample ($x$)
- Latent space distribution ($p_z$)
- Generator ($G_θ$)
- Reconstruction loss ($l$)

**Output:**
- Anomaly Score

**Algorithm:**

1. Initialize { $\{ z_{j,0} \mid z_{j,0} \sim p_z, \, j = 1, \ldots, n_{seed} \}$ } and { ${ G_{θ_{j, 0}} | G_{θ_{j, 0}} ≡ G_{θ}, j = 1, ..., n_{seed} }$ }

2. For each $j$ from $1$ to $n_{seed}$:
    1. For each $t$ from $1$ to $k$:
        - Update latent vector: $z_{j,t} ← z_{j,t-1} - γ_z * ∇_{z_{j,t-1}} l(G_{θ_{j, t-1}}(z_{j,t-1}), x)$
        - Update generator parameters: $θ_{j,t} ← θ_{j,t-1} - γ_θ * ∇_{θ_{j,t-1}} l(G_{θ_{j, t-1}}(z_{j,t-1}), x)$

3. Return: $(1/n_{seed}) * Σ_{j=1}^{n_{seed}} l(G_{θ_{j,k}}(z_{j,k}), x)$


### Learning Rate for Latent Space ($\gamma_z$) and Generator ($\gamma_{\theta}$)

While searching through the latent space to detect anomalous data is a common technique using GANs, it is uncommon to also change the generator's weights to enhance its representational capabilities. While both improve accuracy in assigning an anomaly score to the data sample, there is a difference between propagating through the latent space and adjusting the generator's weights. The learning rates for each have different implications that can affect various applications of anomaly detection.

Changing the latent space learning rate affects the rate of difference in the generated data with each iteration of $k$; however, the generated data will still closely resemble the data distribution $p$ on which the generator was trained. In this experiment, if the Rectangle GAN is used to predict anomalous data with a high latent space learning rate, the generated data will change significantly after each step but remain a rectangle since the generated data distribution $p_{\theta}$ has remained unchanged and still resemblance of $p_{rectangle}$.

Conversely, the generator's learning rate affects the generated data distribution $p_{\theta}$ itself. This parameter is important because $p_{\theta}$ will never exactly match $p$. This discrepancy could result from various factors, including a lack of diversity in the dataset. Consequently, when given a data sample that belong to the actual data distribution $p$ but not $p_{\theta}$, adjusting the model's weights allow $p_{\theta}$ to approach closer to $p$. However, a high generator's learning rate could result in $p_{\theta}$ changing completely outside of $p$ if the data is anomalous. Therefore, the generator's learning rate is typically set low. See Figure 1 for a visual comparison of the effects of setting the generator learning rate to high and low values. Both examples use the same generator, trained on rectangular 3D model data, and evaluated with the same spherical ball data sample. Each example shows the generated data over iterations of the search process. In both cases, the generated data adjusts in size and shape to more closely resemble the data sample. However, with a low learning rate, the generated data retains its rectangular form, while with a high learning rate, the shapes become non-rectangular. This indicates that $p_{\theta}$ has deviated significantly from $p_{rectangle}$.

<div align="center">
  <img src="https://github.com/user-attachments/assets/ae55d3a5-c655-4b1f-8a77-f7c856fd460a" alt="Low and High Generator Learning Rate Demonstration" width="85%">

  <p><b>Figure 1:</b> Generated samples through multiple iterations of latent search with low and high generator learning rate for the rectangular GAN. Samples with low generator learning rates remain in the same realm of rectangular boxes, while samples with high generator learning rates changes the generator's $p_{\theta}$ to no longer belonging within the realm of rectangular boxes.</p>
</div>

### Hyperparameters
The same set hyper-parameter values were employed through out the different experiments conducted. Latent space ($\gamma_z$) and Generator ($\gamma_{\theta}$) Learning Rate were set to be $5 \cdot 10 ^{-2}$ and $6 \cdot 10 ^{-9}$ respectively, $n_{seed}$ as $3$, and $k$ as $19$. Adam optimizer used for the generator weights loss propagation and latent space loss propagation. MSE loss is used as the reconstruction loss.


