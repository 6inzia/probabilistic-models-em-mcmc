# EM algorithm for a univariate Gaussian mixture with K components, implemented from scratch,
# with model selection by BIC.

set.seed(42)

# Simulate n observations from the mixture sum_k p_k N(mu_k, sigma_k^2)
rgmm <- function(n, p, mu, sigma) {
  z <- sample(seq_along(p), n, replace = TRUE, prob = p)
  list(x = rnorm(n, mu[z], sigma[z]), z = z)
}

# Density of the mixture at x
dgmm <- function(x, p, mu, sigma) {
  rowSums(sapply(seq_along(p), function(k) p[k] * dnorm(x, mu[k], sigma[k])))
}

# EM algorithm
#   E-step: t_ik = P(z_i = k | x_i, theta)
#   M-step: closed-form updates of the proportions, means and standard deviations
em_gmm <- function(x, K, maxit = 500, tol = 1e-6) {
  n <- length(x)
  # Initialization: means spread over the data quantiles, common variance, equal weights
  mu <- as.numeric(quantile(x, probs = (1:K) / (K + 1)))
  sigma <- rep(sd(x), K)
  p <- rep(1 / K, K)
  loglik <- numeric(0)

  for (iter in 1:maxit) {
    # E-step
    dens <- sapply(1:K, function(k) p[k] * dnorm(x, mu[k], sigma[k]))   # n x K
    tik <- dens / rowSums(dens)
    loglik[iter] <- sum(log(rowSums(dens)))

    # M-step
    nk <- colSums(tik)
    p <- nk / n
    mu <- colSums(tik * x) / nk
    sigma <- sqrt(colSums(tik * outer(x, mu, "-")^2) / nk)

    # Stop when the log-likelihood no longer increases
    if (iter > 1 && abs(loglik[iter] - loglik[iter - 1]) < tol) break
  }

  list(p = p, mu = mu, sigma = sigma, loglik = loglik,
       cluster = max.col(tik),              # MAP classification
       bic = -2 * loglik[length(loglik)] + (3 * K - 1) * log(n))
}


# ---- Example: 3-component mixture --------------------------------------------------------
true_p     <- c(0.3, 0.3, 0.4)
true_mu    <- c(-20, 0, 20)
true_sigma <- c(1, 2, 2)
sim <- rgmm(1000, true_p, true_mu, true_sigma)

fit <- em_gmm(sim$x, K = 3)
o <- order(fit$mu)                          # components are identifiable up to a permutation
print(data.frame(true_p, est_p = round(fit$p[o], 3),
                 true_mu, est_mu = round(fit$mu[o], 3),
                 true_sigma, est_sigma = round(fit$sigma[o], 3)))
cat("Iterations:", length(fit$loglik), "\n")
cat("MAP classification accuracy:", mean(match(fit$cluster, o) == sim$z), "\n")

# Log-likelihood along the iterations (EM never decreases it)
plot(fit$loglik, type = "b", xlab = "iteration", ylab = "log-likelihood",
     main = "EM: log-likelihood along the iterations")

# Fitted vs true density
hist(sim$x, breaks = 60, freq = FALSE, col = "grey90", border = "white",
     main = "Fitted mixture (EM) vs true density", xlab = "x")
curve(dgmm(x, true_p, true_mu, true_sigma), add = TRUE, lwd = 2, col = "navy")
curve(dgmm(x, fit$p, fit$mu, fit$sigma), add = TRUE, lwd = 2, lty = 2, col = "violet")
legend("topright", c("true", "EM"), col = c("navy", "violet"), lty = 1:2, lwd = 2)

# ---- Choosing the number of components with BIC --------------------------------------------
bic <- sapply(1:6, function(K) em_gmm(sim$x, K)$bic)
plot(1:6, bic, type = "b", xlab = "number of components K", ylab = "BIC",
     main = "Model selection by BIC (lower is better)")
cat("Selected K:", which.min(bic), "\n")
