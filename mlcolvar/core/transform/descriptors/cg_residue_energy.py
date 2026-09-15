import torch
from typing import List, Optional, Tuple, Dict, Union
from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.reduced_ff_energy import (
    _ensure_tensor,
    _min_image,
    _pairwise_displacements,
    _angle,
)

__all__ = ["CGResidueEnergy", "CGEnergy", "AMINO_ACID_VOCAB", "resname_to_type_index"]

# Canonical 20-amino-acid vocabulary (index 0..19), plus a catch-all "unknown"
# type at index 20. Residue names are normalized (common CHARMM/AMBER
# protonation-state and histidine-tautomer variants folded onto the parent
# amino acid) so the type index is stable regardless of force-field naming.
AMINO_ACID_VOCAB = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
UNKNOWN_TYPE_NAME = "UNK"
DEFAULT_N_TYPES = len(AMINO_ACID_VOCAB) + 1  # + UNK

_RESNAME_ALIASES = {
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "CYX": "CYS", "CYM": "CYS",
    "ASH": "ASP", "ASPP": "ASP",
    "GLH": "GLU", "GLUP": "GLU",
    "LYN": "LYS",
}


def resname_to_type_index(resname: str, vocab: Optional[List[str]] = None) -> int:
    """Map a (possibly force-field-specific) residue name to a canonical
    amino-acid type index. Unrecognized names map to the last ("UNK") index.
    """
    vocab = vocab if vocab is not None else AMINO_ACID_VOCAB
    name = resname.strip().upper()
    name = _RESNAME_ALIASES.get(name, name)
    if name in vocab:
        return vocab.index(name)
    return len(vocab)  # UNK


def _cosine_cutoff(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Standard Behler-Parrinello smooth cutoff function: 1 at r=0, 0 at r>=cutoff."""
    inside = (r < cutoff).to(r.dtype)
    fc = 0.5 * (torch.cos(torch.pi * r / cutoff) + 1.0)
    return fc * inside


def _diag_safe(rij: torch.Tensor, N: int, D: int, device, dtype) -> torch.Tensor:
    """Perturbs the (i == i) diagonal of a pairwise-displacement tensor
    ``rij`` (shape (Btot, N, N, D), with rij[:,i,i] == 0 exactly) with a
    fixed nonzero vector.

    Self-pair contributions are always masked to zero *value* further
    downstream (via a cosine-cutoff mask on the diagonal), but
    ``torch.linalg.norm``'s (and, transitively, ``_angle``'s) backward at an
    exact zero-norm vector is 0/0 = NaN. Multiplying that NaN by the
    downstream zero mask does *not* recover a zero gradient (NaN * 0 = NaN in
    IEEE arithmetic), so the diagonal must never be exactly zero before any
    norm/angle is computed, even though its forward value is discarded.
    """
    unit = torch.zeros(D, device=device, dtype=dtype)
    unit[0] = 1.0
    diag = torch.eye(N, dtype=dtype, device=device).unsqueeze(0).unsqueeze(-1)  # (1,N,N,1)
    return rij + diag * unit


class CGResidueEnergy(Transform):
    """Coarse-grained, residue-level descriptor: Behler-Parrinello-style radial
    Gaussian symmetry functions (over C-alpha - C-alpha distances) plus
    low-order angular (C-alpha - C-alpha - C-alpha) symmetry functions,
    indexed by neighbor residue-type (SNAP-style "elements" = amino acid
    types). Mirrors the shape conventions and helper functions of
    ``ReducedFFEnergy`` so it slots into the same training/validation
    pipeline, but operates on C-alpha pseudo-atom positions only (one
    position per residue instance) instead of all-atom positions.

    Accepted input shapes (with n_replicas = R, n_residues = N, ndim = D):
      1. (B, N*D*R)
      2. (B, R, N*D)
      3. (B, R, N, D)
      4. (B, N*D) (only if R == 1)
      5. (B, R*N, D)

    Output of ``forward``: full per-residue feature tensor (B, R, N, n_features),
    where n_features = n_types*K_rad (+ n_type_pairs*K_ang if angular terms are
    used). This is analogous to ``.components()`` in ``ReducedFFEnergy``: Phase 3
    (Bayesian regression) needs the raw features, and the PLUMED ``CG_ENERGY``
    action needs to reproduce the same per-residue feature -> energy contraction
    for validation.

    Note that, unlike ``ReducedFFEnergy``, ``forward`` here does NOT return a
    total energy -- it can't, since (unlike a fixed-parameter force field) the
    per-residue weights are exactly what Phase 3 is fitting, so they don't
    exist yet when this class is used for training. Once a set of weights has
    been fit (``weights.npz``), wrap this class in :class:`CGEnergy` to get a
    ``forward() -> total energy`` model with the same contract as
    ``ReducedFFEnergy``, for use downstream (e.g. as a physical feature in a
    larger CV pipeline) or for validation against the PLUMED ``CG_ENERGY``
    action.
    """

    def __init__(
        self,
        n_residues: int,
        residue_types: List[int],
        radial_centers: List[float],
        radial_widths: List[float],
        radial_cutoff: float,
        angular_centers: Optional[List[float]] = None,
        angular_widths: Optional[List[float]] = None,
        angular_cutoff: Optional[float] = None,
        n_types: int = DEFAULT_N_TYPES,
        ndim: int = 3,
        pbc: bool = True,
        box: Optional[Union[float, List[float]]] = None,
        n_replicas: int = 1,
        angular_chunk_size: Optional[int] = None,
        use_bias: bool = True,
    ):
        """Initialises a CGResidueEnergy descriptor. We recommend initialising
        it from the Phase 1 ``residues.tsv`` schema and basis-function files
        via the ``from_files`` static method.

        Parameters
        ----------
        n_residues : int
            Number of residues (= number of C-alpha pseudo-atoms).
        residue_types : List[int]
            Residue-type index (0..n_types-1) per residue, in GROUP order.
        radial_centers : List[float]
            Gaussian centers mu_k for the radial (G2-style) basis.
        radial_widths : List[float]
            Gaussian widths sigma_k for the radial basis (same length as radial_centers).
        radial_cutoff : float
            Cutoff distance for radial neighbor terms (smooth cosine cutoff applied).
        angular_centers : Optional[List[float]], optional
            Angle centers theta0_m (radians) for the angular basis, by default None (no angular terms).
        angular_widths : Optional[List[float]], optional
            Angle widths w_m (radians) for the angular basis, by default None.
        angular_cutoff : Optional[float], optional
            Cutoff distance applied to both neighbor legs of the angular term, by default None.
        n_types : int, optional
            Size of the residue-type vocabulary, by default 21 (20 amino acids + UNK).
        ndim : int, optional
            Number of spatial dimensions, by default 3.
        pbc : bool, optional
            Whether to apply minimum-image periodic boundary conditions, by default True.
        box : Optional[float or List[float]], optional
            Box dimensions, by default None.
        n_replicas : int, optional
            Number of replicas, by default 1.
        angular_chunk_size : Optional[int], optional
            Process the angular (O(N^3)) term in chunks of this many "central"
            residues at a time to bound peak memory for large proteins, by
            default None (process all residues in a single chunk).
        use_bias : bool, optional
            Append a constant (always-1.0) feature per residue, by default
            True. Real per-residue energy labels have a large, near-constant
            baseline (bonded + background nonbonded/solvent energy) on top of
            comparatively small conformational fluctuations; a purely
            linear-in-features model with no constant term cannot represent
            that baseline well (it can only scale bounded, zero-at-cutoff
            basis functions). This feature acts as a per-residue-instance
            intercept once fitted (Phase 3).
        """
        self.n_residues = n_residues
        self.ndim = ndim
        self.n_replicas = int(n_replicas) if n_replicas is not None else 1
        in_features = n_residues * ndim * self.n_replicas
        self.n_types = n_types

        if len(residue_types) != n_residues:
            raise ValueError("residue_types must have length n_residues")
        if any((t < 0 or t >= n_types) for t in residue_types):
            raise ValueError(f"residue_types entries must be in [0, {n_types})")
        if len(radial_centers) != len(radial_widths):
            raise ValueError("radial_centers and radial_widths must have the same length")

        self.radial_centers = list(radial_centers)
        self.radial_widths = list(radial_widths)
        self.radial_cutoff = float(radial_cutoff)
        self.n_radial_basis = len(radial_centers)

        self.use_angular = angular_centers is not None and len(angular_centers) > 0
        if self.use_angular:
            if angular_widths is None or len(angular_widths) != len(angular_centers):
                raise ValueError("angular_widths must be provided with the same length as angular_centers")
            if angular_cutoff is None or angular_cutoff <= 0:
                raise ValueError("angular_cutoff must be > 0 when angular terms are used")
        self.angular_centers = list(angular_centers) if self.use_angular else []
        self.angular_widths = list(angular_widths) if self.use_angular else []
        self.angular_cutoff = float(angular_cutoff) if self.use_angular else None
        self.n_angular_basis = len(self.angular_centers)
        self.n_type_pairs = n_types * (n_types + 1) // 2

        n_features = n_types * self.n_radial_basis
        if self.use_angular:
            n_features += self.n_type_pairs * self.n_angular_basis
        self.use_bias = use_bias
        if self.use_bias:
            n_features += 1
        self.n_features = n_features
        out_features = n_residues * n_features

        super().__init__(in_features=in_features, out_features=out_features)

        self.residue_types = list(residue_types)
        self.pbc = pbc
        self.box = None if (box is None or not pbc) else _ensure_tensor(box)
        self.angular_chunk_size = angular_chunk_size if angular_chunk_size else n_residues

        # (n_residues, n_types) one-hot neighbor-type indicator, used to bin
        # radial/angular contributions by neighbor type via einsum.
        type_idx = torch.as_tensor(self.residue_types, dtype=torch.long)
        self._type_onehot = torch.nn.functional.one_hot(type_idx, num_classes=n_types).to(
            torch.get_default_dtype()
        )

        if self.use_angular:
            # Symmetric (unordered) type-pair -> linear index lookup table.
            pair_index = torch.zeros((n_types, n_types), dtype=torch.long)
            p = 0
            for a in range(n_types):
                for b in range(a, n_types):
                    pair_index[a, b] = p
                    pair_index[b, a] = p
                    p += 1
            self._type_pair_index = pair_index
            # Per-residue-pair (not per-type-pair!) bin id: residue_pair_id[j,k]
            # = pair_index[type(j), type(k)], then one-hot -> (N, N, n_type_pairs)
            # for vectorized binning of neighbor-pair (j,k) contributions.
            residue_pair_id = pair_index[type_idx][:, type_idx]  # (N, N)
            self._type_pair_onehot = torch.nn.functional.one_hot(
                residue_pair_id, num_classes=self.n_type_pairs
            ).to(torch.get_default_dtype())

    @staticmethod
    def from_files(
        residues_tsv: str,
        radial_basis_file: str,
        radial_cutoff: float,
        angular_basis_file: Optional[str] = None,
        angular_cutoff: Optional[float] = None,
        resname_to_type: Optional[Dict[str, int]] = None,
        n_types: int = DEFAULT_N_TYPES,
        n_replicas: int = 1,
        **kwargs,
    ):
        """Initialises a CGResidueEnergy descriptor from the Phase 1
        ``residues.tsv`` schema (see ``gen_residue_energy_groups.py``) and
        plain-table basis-function files (mirrors ``ReducedFFEnergy.from_files``'s
        file-parsing conventions).

        Parameters
        ----------
        residues_tsv : str
            Path to residues.tsv (columns: instance_id, moltype,
            moltype_copy_index, resnr, resname, ca_atom_index, n_atoms,
            first_atom_index, last_atom_index, batch_id, group_name), in
            GROUP (residue) order.
        radial_basis_file : str
            Plain whitespace table, one "mu sigma" pair per line (# comments skipped).
        radial_cutoff : float
            Cutoff distance for radial neighbor terms.
        angular_basis_file : Optional[str], optional
            Plain whitespace table, one "theta0_deg width_deg" pair per line, by default None.
        angular_cutoff : Optional[float], optional
            Cutoff distance for angular neighbor terms, by default None.
        resname_to_type : Optional[Dict[str, int]], optional
            Explicit residue-name -> type-index override map; falls back to
            ``resname_to_type_index`` (canonical 20-AA vocabulary) for any
            residue not present in this map, by default None.
        n_types : int, optional
            Size of the residue-type vocabulary, by default 21 (20 amino acids + UNK).
        n_replicas : int, optional
            Number of replicas, by default 1.
        """

        def read_tokens(path):
            out = []
            with open(path, "r") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    out.append(s.split())
            return out

        resnames = []
        with open(residues_tsv, "r") as f:
            header = f.readline().rstrip("\n").split("\t")
            resname_col = header.index("resname")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                resnames.append(parts[resname_col])
        n_residues = len(resnames)

        residue_types = []
        for resname in resnames:
            if resname_to_type is not None and resname in resname_to_type:
                residue_types.append(resname_to_type[resname])
            else:
                residue_types.append(resname_to_type_index(resname))

        radial_centers = []
        radial_widths = []
        for toks in read_tokens(radial_basis_file):
            if len(toks) < 2:
                continue
            radial_centers.append(float(toks[0]))
            radial_widths.append(float(toks[1]))

        angular_centers = None
        angular_widths = None
        if angular_basis_file:
            angular_centers = []
            angular_widths = []
            for toks in read_tokens(angular_basis_file):
                if len(toks) < 2:
                    continue
                angular_centers.append(float(toks[0]) * (torch.pi / 180.0))
                angular_widths.append(float(toks[1]) * (torch.pi / 180.0))

        return CGResidueEnergy(
            n_residues=n_residues,
            residue_types=residue_types,
            radial_centers=radial_centers,
            radial_widths=radial_widths,
            radial_cutoff=radial_cutoff,
            angular_centers=angular_centers,
            angular_widths=angular_widths,
            angular_cutoff=angular_cutoff,
            n_types=n_types,
            n_replicas=n_replicas,
            **kwargs,
        )

    def _parse_input(self, x: torch.Tensor) -> torch.Tensor:
        N = self.n_residues
        D = self.ndim
        R = self.n_replicas
        if x.dim() == 1:
            x = x.view(1, -1)
        if x.dim() == 2:
            if x.shape[1] == N * D * R:
                pos = x.view(x.shape[0], R, N, D)
            elif x.shape[1] == N * D and R == 1:
                pos = x.view(x.shape[0], 1, N, D)
            else:
                raise ValueError(f"Unexpected flattened shape {tuple(x.shape)} for n_replicas={R}")
        elif x.dim() == 3:
            if x.shape[1] == R * N and x.shape[2] == D:
                pos = x.view(x.shape[0], R, N, D)
            elif x.shape[1] == R and x.shape[2] == N * D:
                pos = x.view(x.shape[0], R, N, D)
            else:
                raise ValueError("3D input must be (B, R, N*D) or (B, R*N, D)")
        elif x.dim() == 4:
            if x.shape[1] == R and x.shape[2] == N and x.shape[3] == D:
                pos = x
            else:
                raise ValueError("4D input must be (B, R, N, D)")
        else:
            raise ValueError("Input must have dim 1..4")
        return pos

    def _radial_features(self, pos: torch.Tensor, box: Optional[torch.Tensor]) -> torch.Tensor:
        """pos: (Btot, N, D) -> (Btot, N, n_types*K_rad)."""
        Btot, N, D = pos.shape
        device, dtype = pos.device, pos.dtype
        rij = _pairwise_displacements(pos, box)  # (Btot, N, N, D), rij[:,i,j] = pos_i - pos_j
        rij = _diag_safe(rij, N, D, device, dtype)
        r = torch.linalg.norm(rij, dim=-1)  # (Btot, N, N)
        eye = torch.eye(N, dtype=torch.bool, device=device)
        r_safe = r.clamp_min(1e-12)

        fc = _cosine_cutoff(r_safe, self.radial_cutoff)
        fc = fc.masked_fill(eye, 0.0)  # exclude self-interaction (j == i)

        mu = torch.as_tensor(self.radial_centers, device=device, dtype=dtype).view(1, 1, 1, -1)
        sigma = torch.as_tensor(self.radial_widths, device=device, dtype=dtype).view(1, 1, 1, -1)
        gauss = torch.exp(-((r_safe.unsqueeze(-1) - mu) ** 2) / (2.0 * sigma ** 2))  # (Btot,N,N,K)
        gauss = gauss * fc.unsqueeze(-1)

        type_onehot = self._type_onehot.to(device=device, dtype=dtype)  # (N, n_types)
        # (Btot,N,N,K) x (N,n_types) -> (Btot,N,n_types,K), summed over neighbor index j
        radial = torch.einsum("bijk,jt->bitk", gauss, type_onehot)
        radial = radial.reshape(Btot, N, self.n_types * self.n_radial_basis)
        return radial

    def _angular_features(self, pos: torch.Tensor, box: Optional[torch.Tensor]) -> torch.Tensor:
        """pos: (Btot, N, D) -> (Btot, N, n_type_pairs*K_ang), chunked over the
        central-residue axis to bound peak memory for large N."""
        Btot, N, D = pos.shape
        device, dtype = pos.device, pos.dtype
        rij = _pairwise_displacements(pos, box)  # rij[:,i,j] = pos_i - pos_j
        # rij[:,i,i] is exactly the zero vector. Its forward contribution is
        # always masked out below (fc is zeroed on the diagonal), but computing
        # torch.linalg.norm()/_angle() on an exact zero vector still yields a
        # NaN gradient (the backward of ||v|| at v=0 is v/||v|| = 0/0), which
        # then poisons the whole loss via autograd even though the
        # corresponding *value* is multiplied by zero downstream. Perturb only
        # the diagonal with a fixed nonzero vector to avoid ever
        # differentiating through a zero-norm vector; this has no effect on
        # the (already zero-weighted) forward output.
        rij = _diag_safe(rij, N, D, device, dtype)
        dvec = -rij  # dvec[:,i,j] = pos_j - pos_i  (ray from i to j)
        r = torch.linalg.norm(rij, dim=-1)  # (Btot, N, N)
        r_safe = r.clamp_min(1e-12)
        eye = torch.eye(N, dtype=torch.bool, device=device)

        theta0 = torch.as_tensor(self.angular_centers, device=device, dtype=dtype).view(1, 1, 1, -1)
        w = torch.as_tensor(self.angular_widths, device=device, dtype=dtype).view(1, 1, 1, -1)
        type_pair_onehot = self._type_pair_onehot.to(device=device, dtype=dtype)  # (N,N,P)

        fc = _cosine_cutoff(r_safe, self.angular_cutoff).masked_fill(eye, 0.0)  # (Btot,N,N)

        # Unordered neighbor-pair indices (j < k) over the FULL residue set.
        # Gathering explicit j<k pairs (instead of computing a dense (N,N)
        # grid and masking the j==k diagonal afterward) is essential, not
        # just an optimization: on the j==k diagonal, v1 and v2 are the exact
        # same vector, so cos(theta) == 1 exactly and acos'(1) == -inf; that
        # -inf, multiplied by the (correct) zero mask, still yields NaN in
        # autograd (-inf * 0 = NaN). Never evaluating _angle() on j==k avoids
        # this at the source.
        ju, ku = torch.triu_indices(N, N, offset=1, device=device)  # each (P,), P=N*(N-1)/2
        pair_onehot = type_pair_onehot[ju, ku]  # (P, n_type_pairs)

        chunk = max(1, self.angular_chunk_size)
        out_chunks = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            idx_i = slice(start, end)
            n_i = end - start

            dvec_i = dvec[:, idx_i, :, :]  # (Btot, n_i, N, D)
            fc_i = fc[:, idx_i, :]  # (Btot, n_i, N)

            v1 = dvec_i[:, :, ju, :]  # (Btot, n_i, P, D) = dvec[i,j]
            v2 = dvec_i[:, :, ku, :]  # (Btot, n_i, P, D) = dvec[i,k]
            theta = _angle(v1, v2)  # (Btot, n_i, P)

            fc_ij = fc_i[:, :, ju]  # (Btot, n_i, P)
            fc_ik = fc_i[:, :, ku]  # (Btot, n_i, P)
            weight = fc_ij * fc_ik  # (Btot, n_i, P)

            gauss = torch.exp(-((theta.unsqueeze(-1) - theta0) ** 2) / (2.0 * w ** 2))  # (Btot,n_i,P,M)
            gauss = gauss * weight.unsqueeze(-1)

            # Bin by (type(j),type(k)) unordered pair -> (Btot, n_i, n_type_pairs, M)
            binned = torch.einsum("bipm,pq->biqm", gauss, pair_onehot)
            out_chunks.append(binned.reshape(Btot, n_i, self.n_type_pairs * self.n_angular_basis))

        return torch.cat(out_chunks, dim=1)


    def _compute_features_batch(self, pos: torch.Tensor) -> torch.Tensor:
        """pos: (Btot, N, D) -> (Btot, N, n_features)."""
        Btot, N, D = pos.shape
        device, dtype = pos.device, pos.dtype
        box = None if (self.box is None or not self.pbc) else self.box.to(device=device, dtype=dtype)
        radial = self._radial_features(pos, box)
        parts = [radial]
        if self.use_angular:
            parts.append(self._angular_features(pos, box))
        if self.use_bias:
            parts.append(torch.ones((Btot, N, 1), device=device, dtype=dtype))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_all = self._parse_input(x)
        B, R, N, D = pos_all.shape
        pos_flat = pos_all.reshape(B * R, N, D)
        features = self._compute_features_batch(pos_flat)
        return features.view(B, R, N, self.n_features)

    def energy(
        self, x: torch.Tensor, weights: Union[torch.Tensor, "list", "tuple"]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Contract features with fitted linear weights (Phase 3 output) to
        produce the coarse-grained energy CV, for validation against the
        PLUMED ``CG_ENERGY`` action.

        Parameters
        ----------
        x : torch.Tensor
            Input positions (see accepted shapes in the class docstring).
        weights : array-like
            Either per-residue-instance weights, shape (n_residues, n_features)
            (the deployed Phase 3 artifact), or per-residue-type weights,
            shape (n_types, n_features) (e.g. the Phase 3 pooled prior, for a
            quick sanity check before per-instance partial pooling).

        Returns
        -------
        total_energy : torch.Tensor
            Shape (B, R): total CG energy (sum over residues).
        per_residue_energy : torch.Tensor
            Shape (B, R, n_residues): per-residue energy contributions.
        """
        features = self.forward(x)  # (B, R, N, n_features)
        w = _ensure_tensor(weights, device=features.device, dtype=features.dtype)
        if w.shape[0] == self.n_residues:
            w_per_residue = w
        elif w.shape[0] == self.n_types:
            type_idx = torch.as_tensor(self.residue_types, dtype=torch.long, device=w.device)
            w_per_residue = w[type_idx]
        else:
            raise ValueError(
                f"weights first dimension ({w.shape[0]}) must match either "
                f"n_residues ({self.n_residues}) or n_types ({self.n_types})"
            )
        per_residue_energy = (features * w_per_residue.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        total_energy = per_residue_energy.sum(dim=-1)
        return total_energy, per_residue_energy


class CGEnergy(Transform):
    """Deployment-time wrapper around a fitted :class:`CGResidueEnergy`:
    ``forward()`` returns the total (scalar per-replica) coarse-grained
    energy, matching ``ReducedFFEnergy``'s ``forward() -> total energy``
    contract so a fitted CG energy model can be plugged into the same
    downstream code (e.g. as a physical feature inside a larger CV pipeline,
    or for validating the PLUMED ``CG_ENERGY`` action) without that code
    needing to know about ``CGResidueEnergy``'s raw per-residue-feature
    output or do the weight contraction itself.

    ``CGResidueEnergy`` itself is deliberately left unchanged: Phase 3
    (Bayesian regression) fits the per-residue weights, so it needs
    ``forward()`` to return raw features, not a fixed energy -- there are no
    weights yet at that point. This class is only meaningful *after* fitting,
    once a ``weights.npz`` exists.

    Construct with :meth:`from_weights_npz` in the common case (loads
    everything -- basis centers/widths/cutoffs, residue types, weights --
    straight from Phase 3's ``weights.npz``, so there is no separate
    radial/angular basis file to keep in sync with the fitted weights). The
    plain constructor is also available for composing an already-built
    ``CGResidueEnergy`` with an arbitrary weight tensor (e.g. the pooled
    per-type prior, for a quick sanity check before per-instance fitting).
    """

    def __init__(self, featurizer: "CGResidueEnergy", weights: Union[torch.Tensor, "list", "tuple"]):
        """
        Parameters
        ----------
        featurizer : CGResidueEnergy
            A configured descriptor (residue types + radial/angular basis).
        weights : array-like
            Per-residue-instance weights, shape (n_residues, n_features)
            (the deployed Phase 3 artifact), or per-residue-type weights,
            shape (n_types, n_features) -- see ``CGResidueEnergy.energy``.
        """
        out_features = featurizer.n_replicas if featurizer.n_replicas > 1 else 1
        super().__init__(in_features=featurizer.in_features, out_features=out_features)
        self.featurizer = featurizer
        self.n_replicas = featurizer.n_replicas
        self.register_buffer("weights", _ensure_tensor(weights))

    @staticmethod
    def from_weights_npz(
        weights_npz: str,
        n_replicas: int = 1,
        pbc: bool = True,
        box: Optional[Union[float, List[float]]] = None,
        angular_chunk_size: Optional[int] = None,
    ) -> "CGEnergy":
        """Build a ready-to-use, fixed-weight CG energy model directly from
        Phase 3's ``weights.npz`` (see ``train_cg_energy.py``). No
        ``residues.tsv``/basis-file paths are needed: ``weights.npz`` already
        stores the exact residue types and radial/angular basis
        centers/widths/cutoffs used to fit the weights, so re-deriving them
        from separate files would only risk a silent mismatch between the
        basis actually fit and the one used at inference time.

        Parameters
        ----------
        weights_npz : str
            Path to Phase 3's ``weights.npz`` output.
        n_replicas : int, optional
            Number of replicas, by default 1.
        pbc : bool, optional
            Whether to apply the minimum-image convention, by default True.
        box : Optional[float or List[float]], optional
            Box size(s) for the minimum-image convention, by default None.
        angular_chunk_size : Optional[int], optional
            Chunk size for the angular-feature loop (memory/speed trade-off
            only; does not affect the result), by default None.
        """
        import numpy as np

        data = np.load(weights_npz, allow_pickle=True)
        weights = data["weights"]
        type_index = data["type_index"]
        n_types = int(data["n_types"])
        radial_centers = data["radial_centers"]
        radial_widths = data["radial_widths"]
        radial_cutoff = float(data["radial_cutoff"])
        angular_centers = data["angular_centers"]
        angular_widths = data["angular_widths"]
        n_angular = int(angular_centers.size)

        # weights.npz doesn't store use_bias explicitly (train_cg_energy.py
        # always fits with the default use_bias=True) -- recover it from the
        # weight-row width, same check as gen_cg_energy_files.py performs.
        n_radial = int(radial_centers.size)
        n_type_pairs = n_types * (n_types + 1) // 2
        n_features_no_bias = n_types * n_radial + (n_type_pairs * n_angular if n_angular > 0 else 0)
        n_features = weights.shape[1]
        if n_features == n_features_no_bias + 1:
            use_bias = True
        elif n_features == n_features_no_bias:
            use_bias = False
        else:
            raise ValueError(
                f"weights.npz n_features={n_features} does not match n_types*K_rad"
                f"{'+n_type_pairs*K_ang' if n_angular > 0 else ''} (+1 for bias) = "
                f"{n_features_no_bias} or {n_features_no_bias + 1}"
            )

        featurizer = CGResidueEnergy(
            n_residues=len(type_index),
            residue_types=list(int(t) for t in type_index),
            radial_centers=radial_centers.tolist(),
            radial_widths=radial_widths.tolist(),
            radial_cutoff=radial_cutoff,
            angular_centers=angular_centers.tolist() if n_angular > 0 else None,
            angular_widths=angular_widths.tolist() if n_angular > 0 else None,
            angular_cutoff=float(data["angular_cutoff"]) if n_angular > 0 else None,
            n_types=n_types,
            pbc=pbc,
            box=box,
            n_replicas=n_replicas,
            angular_chunk_size=angular_chunk_size,
            use_bias=use_bias,
        )
        return CGEnergy(featurizer, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_energy, _ = self.featurizer.energy(x, self.weights)
        if self.n_replicas == 1:
            return total_energy.view(-1, 1)
        return total_energy

    def components(self, x: torch.Tensor) -> torch.Tensor:
        """Per-residue energy breakdown, shape (B, R, n_residues) --
        analogous to ``ReducedFFEnergy.components()``'s per-term breakdown.
        """
        _, per_residue_energy = self.featurizer.energy(x, self.weights)
        return per_residue_energy
