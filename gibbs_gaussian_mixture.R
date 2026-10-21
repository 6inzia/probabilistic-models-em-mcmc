# Bayesian inference for a 2-component Gaussian mixture with a Gibbs sampler (data augmentation),
# and Gibbs sampling of a correlated bivariate Gaussian.
#
# Conjugate priors:  pi ~ Dirichlet(gamma),  sigma_k^2 ~ InvGamma(lambda_k/2, beta_k/2),
#                    mu_k | sigma_k^2 ~ N(alpha_k, sigma_k^2 / lambda_k)
# The sampler alternates  z | x, theta  and  theta | z, x.
# Adapted from the course notes of the "Latent structure models" class (Sorbonne University).

library(gtools)     # rdirichlet, ddirichlet
library(mvtnorm)    # rmvnorm

set.seed(0)

# ---- Simulation helpers ---------------------------------------------------------------------
simulate_theta <- function(psi) {
  K <- length(psi$gamma)
  pi <- as.vector(rdirichlet(1, psi$gamma))
  sig2 <- 1 / rgamma(K, psi$lambda / 2, psi$beta / 2)
  mu <- rnorm(K, psi$alpha, sqrt(sig2 / psi$lambda))
  list(pi = pi, mu = mu, sig = sqrt(sig2))
}

simulate_mixture <- function(n, theta) {
  z <- sample(seq_along(theta$pi), n, replace = TRUE, prob = theta$pi)
  rnorm(n, theta$mu[z], theta$sig[z])
}

dnormmix <- function(x, theta) {
  rowSums(sapply(seq_along(theta$pi), function(k) theta$pi[k] * dnorm(x, theta$mu[k], theta$sig[k])))
}

# ---- Full conditionals ----------------------------------------------------------------------
# Latent labels: P(z_i = k | x_i, theta) proportional to pi_k * phi(x_i; mu_k, sigma_k)
sample_z <- function(theta, x) {
  K <- length(theta$pi)
  P <- sapply(1:K, function(k) theta$pi[k] * dnorm(x, theta$mu[k], theta$sig[k]))
  apply(P, 1, function(row) if (sum(row) > 0) sample(1:K, 1, prob = row) else sample(1:K, 1))
}

# Parameters given the labels (conjugate updates)
sample_theta <- function(z, x, psi) {
  K <- length(psi$gamma)
  n_k  <- sapply(1:K, function(k) sum(z == k))
  s_k  <- sapply(1:K, function(k) sum(x[z == k]))
  s2_k <- sapply(1:K, function(k) sum(x[z == k]^2))

  pi <- as.vector(rdirichlet(1, psi$gamma + n_k))
  a <- (n_k + psi$lambda) / 2
  b <- (s2_k + psi$lambda * psi$alpha^2 + psi$beta - (s_k + psi$alpha * psi$lambda)^2 / (n_k + psi$lambda)) / 2
  sig2 <- 1 / rgamma(K, a, b)
  m <- (s_k + psi$lambda * psi$alpha) / (n_k + psi$lambda)
  mu <- rnorm(K, m, sqrt(sig2 / (n_k + psi$lambda)))
  list(pi = pi, mu = mu, sig = sqrt(sig2))
}

# Gibbs sampler: R kept iterations after B burn-in iterations
gibbs <- function(x, theta_init, psi, R = 1000, B = 500) {
  theta <- vector("list", B + R + 1)
  theta[[1]] <- theta_init
  for (r in 1:(B + R)) {
    z <- sample_z(theta[[r]], x)
    theta[[r + 1]] <- sample_theta(z, x, psi)
  }
  theta[(B + 2):(B + R + 1)]
}

# ---- Log-posterior of (mu_1, mu_2), other parameters fixed --------------------------------
log_posterior <- function(theta, x, psi) {
  sum(log(dnormmix(x, theta))) +
    log(ddirichlet(theta$pi, psi$gamma)) +
    sum(dnorm(theta$mu, psi$alpha, theta$sig / sqrt(psi$lambda), log = TRUE))
}


# ---- Example ------------------------------------------------------------------------------
psi <- list(gamma = c(10, 20), alpha = rep(1, 2), lambda = rep(1, 2), beta = rep(1, 2))
theta_true <- simulate_theta(psi)
x <- simulate_mixture(2000, theta_true)
print(theta_true)

hist(x, breaks = 50, freq = FALSE, main = "Simulated 2-component mixture", xlab = "x")
curve(dnormmix(x, theta_true), add = TRUE, col = "blue", lwd = 2)

draws <- gibbs(x, theta_true, psi)
mu <- sapply(draws, function(t) t$mu)
cat("Posterior means of mu:", round(rowMeans(mu), 3), " (true:", round(theta_true$mu, 3), ")\n")

# Posterior samples on top of the log-posterior surface
grid <- seq(-5, 3, by = 0.1)
L <- outer(grid, grid, Vectorize(function(m1, m2) {
  t <- theta_true; t$mu <- c(m1, m2); log_posterior(t, x, psi)
}))
image(grid, grid, L, col = grey(0:255 / 255), xlab = "mu_1", ylab = "mu_2",
      main = "Log-posterior and Gibbs samples")
contour(grid, grid, L, add = TRUE, nlevels = 20, col = "red")
points(mu[1, ], mu[2, ], col = "orange", pch = 20)


# ---- Gibbs sampling of a bivariate Gaussian with correlation rho ---------------------------
# Full conditionals: x | y ~ N(rho * y, 1 - rho^2),  y | x ~ N(rho * x, 1 - rho^2)
gibbs_bivariate_gaussian <- function(n, rho, burn_in = 100) {
  samples <- matrix(NA, n, 2)
  x <- 0; y <- 0
  for (i in 1:(n + burn_in)) {
    x <- rnorm(1, rho * y, sqrt(1 - rho^2))
    y <- rnorm(1, rho * x, sqrt(1 - rho^2))
    if (i > burn_in) samples[i - burn_in, ] <- c(x, y)
  }
  samples
}

rho <- 0.5
par(mfrow = c(1, 2))
plot(gibbs_bivariate_gaussian(2000, rho), pch = 19, cex = 0.4, xlim = c(-4, 4), ylim = c(-4, 4),
     main = "Gibbs sampler", xlab = "x", ylab = "y")
plot(rmvnorm(2000, c(0, 0), matrix(c(1, rho, rho, 1), 2)), pch = 19, cex = 0.4,
     xlim = c(-4, 4), ylim = c(-4, 4), main = "Direct sampling (mvtnorm)", xlab = "x", ylab = "y")
par(mfrow = c(1, 1))
