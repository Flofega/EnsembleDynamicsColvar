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
    "fit_type_prior_from_stats",
    "default_type_prior",
    "decouple_bias_prior",
    "fit_instance_posterior",
    "fit_instance_posterior_from_stats",
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
    I)``, fit on the *mean* (per-row) Gram statistics rather than their raw
    sums (see :func:`fit_type_prior_from_stats` for why): returns the
    resulting posterior ``(mu, Sigma)``, which is used downstream as the
    *prior* for the per-instance conjugate update (see
    :func:`fit_instance_posterior`).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    ridge_lambda : float
        Ridge regularization strength, relative to the *per-row* feature
        scale/variance (NOT the total pooled sample count -- see
        :func:`fit_type_prior_from_stats`). Encodes a weakly-informative
        outer prior on the type-level fit itself, and keeps ``X^T X``
        invertible when ``n_samples < n_features`` or features are
        collinear.
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
    yty = float(y @ y)
    return fit_type_prior_from_stats(
        XtX, Xty, yty, n_samples,
        ridge_lambda=ridge_lambda, has_bias=has_bias, bias_ridge_lambda=bias_ridge_lambda,
    )


def fit_type_prior_from_stats(
    XtX: np.ndarray,
    Xty: np.ndarray,
    yty: float,
    n_samples: int,
    ridge_lambda: float = 1e-3,
    has_bias: bool = True,
    bias_ridge_lambda: float = 1e-8,
) -> TypePrior:
    """Same fit as :func:`fit_type_prior`, but takes precomputed sufficient
    statistics (``X^T X``, ``X^T y``, ``y^T y``, ``n_samples``) instead of
    raw (X, y) rows.

    This is what makes the fit possible without ever holding the full pooled
    design matrix (all training frames x all instances of a type) in memory:
    ``XtX``/``Xty``/``yty`` are additive over rows, so a caller can
    accumulate them incrementally -- frame-chunk by frame-chunk, trajectory
    by trajectory, or residue-instance by residue-instance -- and only pass
    in the final running totals. See ``train_cg_energy.py``'s streaming
    accumulation (``ResidueStats``) for the intended usage.

    Numerical note: the fit is done on the *mean* (per-row) Gram statistics
    ``XtX/n_samples``, ``Xty/n_samples``, ``yty/n_samples`` rather than their
    raw sums. Two reasons:

    1. ``ridge_lambda`` is added directly to this matrix's diagonal, so
       working with the mean keeps ``ridge_lambda``'s meaning (a
       regularization strength relative to the per-row feature
       scale/variance) independent of how many rows happen to be pooled for
       a given type. Adding a fixed ``ridge_lambda`` straight to the raw
       (unnormalized) sum ``XtX`` -- whose scale grows linearly with
       ``n_samples`` -- makes the same ``ridge_lambda`` an ever-weaker
       relative regularizer as more frames/instances are pooled (as happens
       for larger systems or longer trajectories), which previously required
       re-guessing an appropriate ``ridge_lambda`` for every new system size.
    2. It keeps the residual-sum-of-squares computation (see below) well
       conditioned: with raw sums, ``yty``/``mu@Xty``/``mu@(XtX@mu)`` all
       scale with ``n_samples`` (easily 1e5-1e6+ for large systems), so their
       differences -- which should equal the (much smaller) true residual --
       are dominated by floating-point cancellation error, producing
       wildly-inflated, physically meaningless ``sigma2`` values (not a
       literal float64 overflow, just catastrophic loss of precision well
       before the ~1.8e308 overflow threshold). Computing the equivalent
       *mean* residual first (all terms O(per-row scale), no ``n_samples``
       blow-up) and rescaling by ``n_samples`` only at the very end (a benign
       multiplication, not a subtraction of near-equal numbers) avoids this.

    Mathematically this is exactly equivalent to solving the raw (unnormalized)
    system with ``ridge_lambda`` scaled up by ``n_samples`` -- it only changes
    floating-point conditioning/precision and the user-facing meaning of
    ``ridge_lambda``, not the underlying model.

    Parameters
    ----------
    XtX : np.ndarray, shape (n_features, n_features)
        Pooled ``X^T X``.
    Xty : np.ndarray, shape (n_features,)
        Pooled ``X^T y``.
    yty : float
        Pooled ``y^T y`` (i.e. ``sum(y**2)``), needed to recover the
        residual sum of squares without the raw rows: ``resid @ resid ==
        yty - 2*mu@Xty + mu@XtX@mu``.
    n_samples : int
        Total number of pooled (residue-instance, frame) rows.
    """
    n_features = XtX.shape[0]
    ridge_diag = np.full(n_features, ridge_lambda)
    if has_bias:
        ridge_diag[-1] = bias_ridge_lambda
    # Mean (per-row) Gram statistics -- see the numerical note above for why
    # this (rather than the raw sums) is what ridge_lambda is applied to.
    mean_XtX = XtX / n_samples
    mean_Xty = Xty / n_samples
    mean_yty = yty / n_samples
    A = mean_XtX + np.diag(ridge_diag)
    mu = np.linalg.solve(A, mean_Xty)
    mean_resid_ss = float(mean_yty - 2.0 * (mu @ mean_Xty) + mu @ (mean_XtX @ mu))
    mean_resid_ss = max(mean_resid_ss, 0.0)  # guard against float cancellation for a near-zero residual
    resid_ss = mean_resid_ss * n_samples  # rescale back to the true (unnormalized) sum of squares
    dof = max(n_samples - n_features, 1)
    sigma2 = resid_ss / dof
    sigma2 = max(sigma2, 1e-8)  # guard against a degenerate all-zero residual
    # A is the mean-Gram matrix's ridge system, i.e. A == (XtX + n_samples*ridge_diag) / n_samples;
    # the true (unnormalized) posterior covariance is sigma2 * inv(XtX + n_samples*ridge_diag)
    # == (sigma2 / n_samples) * inv(A).
    Sigma = (sigma2 / n_samples) * np.linalg.inv(A)
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
    XtX = X.T @ X
    Xty = X.T @ y
    return fit_instance_posterior_from_stats(XtX, Xty, n_samples, prior, pooling_strength=pooling_strength)


def fit_instance_posterior_from_stats(
    XtX: np.ndarray, Xty: np.ndarray, n_samples: int, prior: TypePrior, pooling_strength: float = 1.0
) -> InstancePosterior:
    """Same update as :func:`fit_instance_posterior`, but takes precomputed
    sufficient statistics (``X^T X``, ``X^T y``, ``n_samples``) instead of
    raw (X, y) rows -- see :func:`fit_type_prior_from_stats` for why this
    matters (streaming accumulation instead of holding the full per-instance
    design matrix in memory).
    """
    prior_precision = pooling_strength * np.linalg.inv(prior.Sigma)
    Lambda_post = prior_precision + XtX / prior.sigma2
    Sigma_post = np.linalg.inv(Lambda_post)
    mu_post = Sigma_post @ (prior_precision @ prior.mu + Xty / prior.sigma2)
    return InstancePosterior(mu=mu_post, Sigma=Sigma_post, n_samples=n_samples)
