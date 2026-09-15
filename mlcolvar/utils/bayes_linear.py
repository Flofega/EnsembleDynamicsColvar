"""Closed-form (conjugate Normal-Normal) hierarchical Bayesian linear
regression used to fit per-residue-instance ``CG_ENERGY`` weights (Phase 3
of the coarse-grained residue-energy CV pipeline).

Two-pass scheme (no MCMC/sampling, everything is a plain closed-form linear
algebra update):

1. **Type-level prior** (:func:`fit_type_prior`): pool all (features, label)
   rows across every residue *instance* that shares a residue *type* (e.g.
   all TYR residues) and fit an empirical-Bayes ridge regression. This gives
   a prior mean ``mu_type`` and covariance ``Sigma_type`` (interpreted as the
   Bayesian-ridge posterior for the *pooled* type-level dataset) plus a
   pooled residual (noise) variance ``sigma2_type``.
2. **Per-instance posterior** (:func:`fit_instance_posterior`): for each
   individual residue instance, do a standard closed-form conjugate
   Normal-Normal Bayesian linear regression update using the type-level fit
   as the prior and the type-level ``sigma2`` as the (shared) noise
   variance. Instances with little data shrink strongly toward the
   type-level mean ("partial pooling"); instances with lots of data are
   dominated by their own likelihood.

Residue types with no training data at all in the current dataset fall back
to a weakly-informative default prior (:func:`default_type_prior`).
"""
import numpy as np
from dataclasses import dataclass

__all__ = [
    "TypePrior",
    "InstancePosterior",
    "fit_type_prior",
    "default_type_prior",
    "decouple_bias_prior",
    "fit_instance_posterior",
]


@dataclass
class TypePrior:
    mu: np.ndarray  # (n_features,) prior/pooled-posterior mean
    Sigma: np.ndarray  # (n_features, n_features) prior/pooled-posterior covariance
    sigma2: float  # pooled residual (noise) variance, shared by all instances of this type
    n_samples: int  # number of pooled (residue-instance, frame) rows used to fit this
    has_data: bool  # False if this type had no training data (generic fallback prior)


@dataclass
class InstancePosterior:
    mu: np.ndarray  # (n_features,) posterior mean weights for this residue instance
    Sigma: np.ndarray  # (n_features, n_features) posterior covariance
    n_samples: int  # number of (frame) rows used for this instance


def fit_type_prior(
    X: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float = 1e-3,
    has_bias: bool = True,
    bias_ridge_lambda: float = 1e-8,
) -> TypePrior:
    """Empirical-Bayes ridge fit of a type-level prior from pooled
    (features, label) rows of all residue instances of one residue type.

    Equivalent to Bayesian linear regression with prior ``w ~ N(0, (sigma2 /
    ridge_lambda) I)`` and likelihood ``y = X w + eps``, ``eps ~ N(0, sigma2
    I)``: returns the resulting posterior ``(mu, Sigma)``, which is used
    downstream as the *prior* for the per-instance conjugate update (see
    :func:`fit_instance_posterior`).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    ridge_lambda : float
        Ridge regularization strength: encodes a weakly-informative outer
        prior on the type-level fit itself, and keeps ``X^T X`` invertible
        when ``n_samples < n_features`` or features are collinear.
    has_bias : bool
        If True (matching ``CGResidueEnergy(use_bias=True)``), the LAST
        feature column is treated as a constant intercept term and is
        regularized with ``bias_ridge_lambda`` instead of ``ridge_lambda``.
        Penalizing the intercept the same as the other weights shrinks it
        toward zero; since real energy labels typically have a large
        near-constant baseline (e.g. mean ~-800 kJ/mol) with small
        fluctuations, shrinking the intercept makes predictions collapse
        toward zero, which is far worse than predicting the mean and can
        produce catastrophically negative R^2. The intercept column should
        therefore be left effectively unpenalized.
    bias_ridge_lambda : float
        Regularization strength applied to the intercept column when
        ``has_bias`` is True. Kept tiny (not exactly zero) only to guard
        against numerical singularity.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_samples, n_features = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    ridge_diag = np.full(n_features, ridge_lambda)
    if has_bias:
        ridge_diag[-1] = bias_ridge_lambda
    A = XtX + np.diag(ridge_diag)
    mu = np.linalg.solve(A, Xty)
    resid = y - X @ mu
    dof = max(n_samples - n_features, 1)
    sigma2 = float(resid @ resid) / dof
    sigma2 = max(sigma2, 1e-8)  # guard against a degenerate all-zero residual
    Sigma = sigma2 * np.linalg.inv(A)
    return TypePrior(mu=mu, Sigma=Sigma, sigma2=sigma2, n_samples=n_samples, has_data=True)


def default_type_prior(
    n_features: int,
    sigma2: float,
    ridge_lambda: float = 1e-3,
    has_bias: bool = True,
    bias_prior_variance: float = 1e8,
) -> TypePrior:
    """Weakly-informative fallback prior for residue types with no training
    data in the current dataset: ``mu = 0``, ``Sigma = (sigma2 /
    ridge_lambda) I``, using a caller-supplied ``sigma2`` (e.g. the median
    ``sigma2`` across residue types that *do* have data).

    If ``has_bias`` is True, the LAST feature column (the constant
    intercept) instead gets prior variance ``bias_prior_variance`` (very
    large / weakly-informative), consistent with :func:`fit_type_prior`
    leaving the intercept effectively unpenalized.
    """
    diag = np.full(n_features, sigma2 / ridge_lambda)
    if has_bias:
        diag[-1] = bias_prior_variance
    Sigma = np.diag(diag)
    return TypePrior(
        mu=np.zeros(n_features), Sigma=Sigma, sigma2=sigma2, n_samples=0, has_data=False
    )


def decouple_bias_prior(prior: TypePrior, bias_prior_variance: float = 1e8) -> TypePrior:
    """Return a copy of ``prior`` with the intercept (last feature column)
    prior made independent of, and uninformative relative to, every other
    feature: its covariance row/column are zeroed except for a large
    marginal variance ``bias_prior_variance``.

    Rationale: :func:`fit_type_prior` pools rows across every residue
    *instance* of a given residue *type* to build a single shared prior.
    Different instances of the same chemical type can have very different
    baseline energies (e.g. depending on their position/local environment),
    so a *pooled* intercept is not a good prior for any individual
    instance's intercept -- it should instead be estimated essentially from
    that instance's own data during :func:`fit_instance_posterior`. Applying
    this function to a ``TypePrior`` before it is used as an instance prior
    keeps the pooled shrinkage benefit for the (still shared) feature
    weights while letting each instance's own likelihood term freely
    determine its intercept.
    """
    Sigma = prior.Sigma.copy()
    Sigma[-1, :] = 0.0
    Sigma[:, -1] = 0.0
    Sigma[-1, -1] = bias_prior_variance
    return TypePrior(
        mu=prior.mu.copy(),
        Sigma=Sigma,
        sigma2=prior.sigma2,
        n_samples=prior.n_samples,
        has_data=prior.has_data,
    )


def fit_instance_posterior(
    X: np.ndarray, y: np.ndarray, prior: TypePrior, pooling_strength: float = 1.0
) -> InstancePosterior:
    """Closed-form conjugate Normal-Normal posterior update for a single
    residue instance, using ``prior`` (its residue type's pooled fit) as the
    prior and ``prior.sigma2`` as the (shared, type-level) noise variance::

        Lambda_post = pooling_strength * Sigma_prior^-1 + X^T X / sigma2
        Sigma_post = Lambda_post^-1
        mu_post = Sigma_post @ (pooling_strength * Sigma_prior^-1 @ mu_prior + X^T y / sigma2)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    prior : TypePrior
        The residue-type-level prior (see :func:`fit_type_prior` /
        :func:`default_type_prior`).
    pooling_strength : float
        Scales how strongly the pooled type-level prior constrains this
        instance's posterior, from 1.0 (full trust in the pooled prior, the
        original behavior) down toward 0.0 (the prior's influence vanishes
        and the fit is driven almost entirely by this instance's own data).
        Lower values are appropriate when instances of the same residue type
        can have substantially different behavior (e.g. very different local
        structural/energetic context despite sharing a chemical identity),
        which pooling would otherwise force them to share. Values very close
        to 0 can make ``Lambda_post`` ill-conditioned/singular when an
        instance's own data alone is insufficient to constrain all features
        (``n_samples < n_features``); some nonzero pooling is still needed
        in that regime.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_samples, n_features = X.shape
    prior_precision = pooling_strength * np.linalg.inv(prior.Sigma)
    XtX = X.T @ X
    Xty = X.T @ y
    Lambda_post = prior_precision + XtX / prior.sigma2
    Sigma_post = np.linalg.inv(Lambda_post)
    mu_post = Sigma_post @ (prior_precision @ prior.mu + Xty / prior.sigma2)
    return InstancePosterior(mu=mu_post, Sigma=Sigma_post, n_samples=n_samples)
