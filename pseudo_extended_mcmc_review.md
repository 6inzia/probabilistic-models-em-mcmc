# Research review: Pseudo-Extended Markov Chain Monte Carlo

Review of *C. Nemeth, F. Lindsten, M. Filippone, J. Hensman, "Pseudo-Extended Markov Chain Monte Carlo"*
(NeurIPS 2019), written for the "Generative models" class, M2 Statistics, Sorbonne University (April 2023, team of 4).

## 1. Context

Markov chain Monte Carlo (MCMC) methods sample from a probability distribution by running a Markov chain
whose stationary distribution is the target. They are the standard tool for high-dimensional sampling,
in particular in Bayesian statistics where the posterior density is only known up to a constant.
Asymptotically, and under mild conditions, they produce samples whose distribution converges to the posterior.

Their main weakness appears with **multimodal posteriors** (mixture models, deep neural networks, …):
the sampler can stay trapped in a local mode for a very large number of iterations, and reaching
stationarity becomes extremely expensive.

The paper proposes the **pseudo-extended** method: the state space is enlarged with *pseudo-samples* used as
auxiliary variables, so that the sampler moves easily between connected modes. Two advantages over
existing approaches:

- the number and location of the modes do not need to be known in advance;
- no sequence of intermediate (tempered) distributions has to be specified.

A reweighting step turns the pseudo-samples into samples from the true posterior. The method can wrap many
MCMC algorithms; the paper applies it to **Hamiltonian Monte Carlo (HMC)** and shows better posterior
exploration than standard HMC, in particular for Boltzmann machines and models with sparsity-inducing priors.

## 2. Algorithm

Let $\pi$ be the target density on $\mathcal X = \mathbb R^d$:

$$\pi(x) = \frac{\gamma(x)}{Z} = \frac{\exp\{-\phi(x)\}}{Z}$$

with $\phi$ continuously differentiable and $Z$ an unknown normalizing constant.
The state space is extended with $N$ pseudo-samples $x_{1:N}$, and the **extended target** on $\mathcal X^N$ is

$$\pi^N(x_{1:N}) = \frac{1}{N}\sum_{i=1}^{N} \pi(x_i)\prod_{j\neq i} q(x_j)
= \frac{1}{Z}\left\{\frac{1}{N}\sum_{i=1}^{N}\frac{\gamma(x_i)}{q(x_i)}\right\}\prod_{j=1}^{N} q(x_j).$$

This has the same form as in pseudo-marginal MCMC, but $q$ plays a different role: in pseudo-marginal MCMC
it produces unbiased estimates of the normalizing constant, whereas here it is used to **improve the mixing**
of the chain. As a consequence, $q$ **does not need to be sampled from**: it only has to be evaluated
pointwise (up to a constant), which leaves a lot of freedom in its choice.

### Pseudo-extended HMC

HMC is used for its fast mixing. For $t = 1, \dots, T$:

1. set $y^{t-1} = x^{t-1}_{1:N}$ and draw a momentum $p \sim \mathcal N(0, M)$;
2. run $L$ leapfrog steps of size $\epsilon$:
   - $p_{l+1/2} = p_l + \frac{\epsilon}{2}\nabla \log \pi^N(y_l)$
   - $y_{l+1} = y_l + \epsilon M^{-1} p_{l+1/2}$
   - $p_{l+1} = p_{l+1/2} + \frac{\epsilon}{2}\nabla \log \pi^N(y_{l+1})$
3. accept $y_{L+1}$ with probability $\min\{1, \exp[H^N(y^{t-1}, p^{t-1}) - H^N(y_{L+1}, p_{L+1})]\}$.

$L$ and $\epsilon$ are tuned automatically with the No-U-Turn Sampler (NUTS).
The extra potential term lets the dynamics jump from one mode to another instead of staying in a local mode.

## 3. Theoretical contribution

The paper combines two lines of work:

- **tempered MCMC**: sampling from $\pi_\beta \propto e^{-\beta\phi}$ instead of $\pi \propto e^{-\phi}$ raises the
  density of low-probability regions, which makes mode switching more likely. The pseudo-extended method avoids
  running several temperatures $\beta$ (expensive) and effectively estimates a suitable one;
- **auxiliary-variable methods**, with the key difference that $q$ only has to be evaluated, not sampled.

The main result is an **unbiasedness** property: for any integrable $f$,

$$\mathbb E_{\pi^N}\left[\frac{\sum_{i=1}^N f(x_i)\,\gamma(x_i)/q(x_i)}{\sum_{i=1}^N \gamma(x_i)/q(x_i)}\right] = \mathbb E_\pi[f(x)],$$

so self-normalized importance weights on the pseudo-samples give unbiased estimates under the true target.

## 4. Limitations

- **Sensitivity to hyperparameters**: the temperature $\beta$, the step size and the number of iterations must be
  chosen carefully; a poorly chosen $\beta$ gives worse results.
- **Computational cost**: each iteration evaluates the target density and its gradient for $N$ pseudo-samples and
  integrates Hamilton's equations; the cost grows with $L$, $T$ and $N$.
- For **simple, unimodal targets**, standard HMC remains preferable.

Pseudo-extended HMC is therefore a sound choice for complex multimodal distributions, as long as these
limitations are taken into account.
