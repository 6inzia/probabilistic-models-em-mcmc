# Random-walk Metropolis-Hastings: tuning of the proposal scale, posterior sampling for the means
# of a Gaussian mixture, and tempering to escape a local mode.
# The tempering experiment follows Marin & Robert, "Bayesian Essentials with R" (2014).

library(bayess)     # plotmix: mixture data set and log-posterior surface from Marin & Robert (2014)

set.seed(100)

# ---- 1. Sampling a bimodal density: effect of the proposal scale ---------------------------
target <- function(x) 0.3 * dnorm(x, -2, 1) + 0.7 * dnorm(x, 5, 1)

rw_metropolis <- function(target, n_iter, sd, start = 0) {
  chain <- numeric(n_iter)
  chain[1] <- start
  accepted <- 0
  for (i in 2:n_iter) {
    candidate <- rnorm(1, chain[i - 1], sd)
    if (runif(1) < target(candidate) / target(chain[i - 1])) {
      chain[i] <- candidate
      accepted <- accepted + 1
    } else {
      chain[i] <- chain[i - 1]
    }
  }
  list(chain = chain, acceptance_rate = accepted / (n_iter - 1))
}

par(mfrow = c(2, 2))
for (sd in c(0.1, 1, 10, 100)) {
  run <- rw_metropolis(target, 10000, sd, start = -2)
  curve(target(x), -10, 15, lwd = 2, col = "navy", ylab = "density",
        main = sprintf("proposal sd = %g, acceptance = %.2f", sd, run$acceptance_rate))
  lines(density(run$chain), col = "turquoise3", lwd = 2)
}
par(mfrow = c(1, 1))
# Too small a step: almost everything is accepted but the chain stays in one mode.
# Too large a step: almost everything is rejected. A moderate scale explores both modes.


# ---- 2. Posterior of (mu_1, mu_2) in the mixture 0.7 N(mu_1, 1) + 0.3 N(mu_2, 1) -------------
# Prior: mu_k ~ N(delta, 1/lambda), independent
log_post <- function(x, mu, p = 0.7, delta = 0, lambda = 1) {
  sum(log(p * dnorm(x, mu[1]) + (1 - p) * dnorm(x, mu[2]))) +
    sum(dnorm(mu, delta, 1 / sqrt(lambda), log = TRUE))
}

# Tempered random-walk MH: targets pi(mu | x)^alpha (alpha = 1 is the true posterior)
mh_mixture_means <- function(x, n_iter = 10000, var = 0.1, alpha = 1, start = c(1, 3)) {
  mu <- matrix(0, n_iter, 2)
  mu[1, ] <- start
  for (i in 2:n_iter) {
    proposal <- rnorm(2, mu[i - 1, ], sqrt(var))
    log_ratio <- log_post(x, proposal) - log_post(x, mu[i - 1, ])
    mu[i, ] <- if (runif(1) <= exp(alpha * log_ratio)) proposal else mu[i - 1, ]
  }
  mu
}

# ---- 3. Tempering: flattening the posterior to escape the spurious mode ---------------------
# Chains start next to the spurious mode (1, 3). With alpha = 1, 10^4 iterations do not leave it;
# with alpha = 0.01 the chain moves between modes after a few iterations.
for (alpha in c(1, 0.1, 0.01)) {
  x <- plotmix()$sample                    # draws the log-posterior surface and returns the data
  points(mh_mixture_means(x, alpha = alpha), pch = ".")
  title(sub = paste("alpha =", alpha))
}
