import torch
from typing import List, Tuple, Optional, Dict, Union
from mlcolvar.core.transform import Transform

__all__ = ["ReducedFFEnergy"]

def _ensure_tensor(x, device=None, dtype=None):
    if x is None:
        return None
    t = torch.as_tensor(x, device=device, dtype=dtype if dtype is not None else torch.get_default_dtype())
    return t

def _min_image(d, box: Optional[torch.Tensor]):
    if box is None:
        return d
    if box.ndim == 0:
        L = box
        return d - torch.round(d / L) * L
    elif box.ndim == 1:
        return d - torch.round(d / box) * box
    else:
        return d

def _pairwise_displacements(pos: torch.Tensor, box: Optional[torch.Tensor]):
    rij = pos.unsqueeze(2) - pos.unsqueeze(1)
    if box is not None:
        rij = _min_image(rij, box)
    return rij

def _angle(v1: torch.Tensor, v2: torch.Tensor, eps=1e-12):
    n1 = torch.linalg.norm(v1, dim=-1).clamp_min(eps)
    n2 = torch.linalg.norm(v2, dim=-1).clamp_min(eps)
    cos_th = (v1 * v2).sum(dim=-1) / (n1 * n2)
    cos_th = cos_th.clamp(-1.0, 1.0)
    return torch.acos(cos_th)

def _dihedral_pbc(pi, pj, pk, pl, box: Optional[torch.Tensor], eps=1e-12):
    b1 = pj - pi
    b2 = pk - pj
    b3 = pl - pk
    if box is not None:
        b1 = _min_image(b1, box)
        b2 = _min_image(b2, box)
        b3 = _min_image(b3, box)
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    n1u = n1 / torch.linalg.norm(n1, dim=-1).unsqueeze(-1).clamp_min(eps)
    n2u = n2 / torch.linalg.norm(n2, dim=-1).unsqueeze(-1).clamp_min(eps)
    b2u = b2 / torch.linalg.norm(b2, dim=-1).unsqueeze(-1).clamp_min(eps)
    m1 = torch.cross(n1u, b2u, dim=-1)
    x = (n1u * n2u).sum(dim=-1)
    y = (m1 * n2u).sum(dim=-1)
    return torch.atan2(y, x)

class ReducedFFEnergy(Transform):
    """This descriptor mirrors the corresponding Plumed CV for calculating partial force field potential energies.
    It requires the user to provide the same input files as the Plumed CV, which contain bond information, angle information,
    dihedral information, and optionally charges and per-atom LJ parameters. A helper script for generating these files is
    provided in the Plumed fork https://github.com/Flofega/EnsembleDynamics.
    Supports multiple replicas.

    Accepted input shapes (with n_replicas = R, n_atoms = N, ndim = D):
      1. (B, N*D*R)                           flattened all replicas
      2. (B, R, N*D)
      3. (B, R, N, D)
      4. (B, N*D) (only if R == 1)
      5. (B, R*N, D)  NEW: replicas concatenated along atom dimension

    Output:
      (B, R) if R > 1 else (B, 1)
    """
    def __init__(
        self,
        n_atoms: int,
        bonds: Optional[List[Tuple[int,int,float,float]]] = None,
        angles: Optional[List[Tuple[int,int,int,float,float]]] = None,
        dihedrals: Optional[List[Tuple[int,int,int,int,float,int,float]]] = None,
        ndim: int = 3,
        use_lj: bool = False,
        lj_epsilon: Optional[float] = None,
        lj_sigma: Optional[float] = None,
        lj_cutoff: Optional[float] = None,
        lj_sigma_i: Optional[List[float]] = None,
        lj_epsilon_i: Optional[List[float]] = None,
        lj_comb: str = "LB",
        use_coulomb: bool = False,
        charges: Optional[List[float]] = None,
        epsilon_r: float = 1.0,
        coulomb_cutoff: float = 0.0,
        exclude12: bool = True,
        exclude13: bool = True,
        scale14: float = 1.0,
        pbc: bool = True,
        box: Optional[Union[float, List[float]]] = None,
        force_atoms: Optional[List[int]] = None,
        n_replicas: int = 1,
    ):
        """Initialises a ReducedFFEnergy object, however we reccomend initialising it from the plumed input files using the provided static method.

        Parameters
        ----------
        n_atoms : int
            number of atoms
        bonds : Optional[List[Tuple[int,int,float,float]]], optional
            list of bonds, by default None
        angles : Optional[List[Tuple[int,int,int,float,float]]], optional
            list of angles, by default None
        dihedrals : Optional[List[Tuple[int,int,int,int,float,int,float]]], optional
            list of dihedrals, by default None
        ndim : int, optional
            number of dimensions, by default 3
        use_lj : bool, optional
            whether to use Lennard-Jones interactions, by default False
        lj_epsilon : Optional[float], optional
            Lennard-Jones epsilon parameter, by default None
        lj_sigma : Optional[float], optional
            Lennard-Jones sigma parameter, by default None
        lj_cutoff : Optional[float], optional
            Lennard-Jones cutoff distance, by default None
        lj_sigma_i : Optional[List[float]], optional
            list of per-atom Lennard-Jones sigma parameters, by default None
        lj_epsilon_i : Optional[List[float]], optional
            list of per-atom Lennard-Jones epsilon parameters, by default None
        lj_comb : str, optional
            Lennard-Jones combination rule, by default "LB"
        use_coulomb : bool, optional
            whether to use Coulomb interactions, by default False
        charges : Optional[List[float]], optional
            list of atomic charges, by default None
        epsilon_r : float, optional
            relative permittivity, by default 1.0
        coulomb_cutoff : float, optional
            Coulomb cutoff distance, by default 0.0
        exclude12 : bool, optional
            whether to exclude 1-2 interactions, by default True
        exclude13 : bool, optional
            whether to exclude 1-3 interactions, by default True
        scale14 : float, optional
            scaling factor for 1-4 interactions, by default 1.0
        pbc : bool, optional
            whether to use periodic boundary conditions, by default True
        box : Optional[float or List[float]], optional
            box dimensions, by default None
        force_atoms : Optional[List[int]], optional
            list of atoms to apply forces to, by default None
        n_replicas : int, optional
            number of replicas, by default 1
        """
        self.n_atoms = n_atoms
        self.ndim = ndim
        self.n_replicas = int(n_replicas) if n_replicas is not None else 1
        in_features = n_atoms*ndim*self.n_replicas
        out_features = self.n_replicas if self.n_replicas>1 else 1
        super().__init__(in_features=in_features, out_features=out_features)
        self.bonds = bonds or []
        self.angles = angles or []
        self.dihedrals = dihedrals or []
        self.use_lj = use_lj
        self.lj_epsilon = lj_epsilon
        self.lj_sigma = lj_sigma
        self.lj_cutoff = lj_cutoff if lj_cutoff is not None else 0.0
        self.lj_peratom = (lj_sigma_i is not None) and (lj_epsilon_i is not None)
        self.lj_sigma_i = lj_sigma_i
        self.lj_epsilon_i = lj_epsilon_i
        self.lj_comb = lj_comb.upper()
        self.use_coulomb = use_coulomb
        self.charges = charges
        self.epsilon_r = float(epsilon_r)
        self.coulomb_cutoff = float(coulomb_cutoff)
        self.exclude12 = exclude12
        self.exclude13 = exclude13
        self.scale14 = float(scale14)
        self.pbc = pbc
        self.box = None if (box is None or not pbc) else _ensure_tensor(box)
        self._build_topology_maps()
        self.ke = torch.tensor(138.935458111, dtype=torch.get_default_dtype())
        if force_atoms is None or len(force_atoms) == 0:
            self.force_mask = None
        else:
            mask = torch.zeros(self.n_atoms, dtype=torch.get_default_dtype())
            mask[torch.as_tensor(force_atoms, dtype=torch.long)] = 1.0
            self.force_mask = mask
        if self.use_lj and not self.lj_peratom:
            if not (self.lj_epsilon and self.lj_sigma and self.lj_cutoff and self.lj_epsilon>0 and self.lj_sigma>0 and self.lj_cutoff>0):
                raise ValueError("LJ requires LJ_EPSILON>0, LJ_SIGMA>0 and LJ_CUTOFF>0 (or per-atom params + cutoff).")
        if self.use_lj and self.lj_peratom and (self.lj_cutoff is None or self.lj_cutoff<=0):
            raise ValueError("When using per-atom LJ, LJ_CUTOFF must be > 0.")
        if self.use_coulomb:
            if self.charges is None or len(self.charges)!=self.n_atoms:
                raise ValueError("Coulomb requires charges list of length n_atoms.")
            if self.epsilon_r <= 0:
                raise ValueError("EPSILON_R must be > 0.")

    def _build_topology_maps(self):
        N = self.n_atoms
        bonded12 = [set() for _ in range(N)]
        bonded13 = [set() for _ in range(N)]
        for (i,j,_,_) in self.bonds:
            bonded12[i].add(j); bonded12[j].add(i)
        for (i,j,k,_,_) in self.angles:
            bonded13[i].add(k); bonded13[k].add(i)
        self._excl12 = bonded12
        self._excl13 = bonded13
        pairs14 = set()
        scale14 = {}
        if len(self.dihedrals) > 0:
            for (i,j,k,l,_,_,_) in self.dihedrals:
                a, b = (i, l) if i<l else (l, i)
                pairs14.add((a,b))
                if self.scale14 != 1.0:
                    scale14[(a,b)] = self.scale14
        self._pairs14 = pairs14
        self._scale14 = scale14

    @staticmethod
    def from_files(
        n_atoms: int,
        bonds_file: str,
        angles_file: Optional[str] = None,
        dihedrals_file: Optional[str] = None,
        charges_file: Optional[str] = None,
        lj_peratom_file: Optional[str] = None,
        n_replicas: int = 1,
        **kwargs
    ):
        """Initialises the energy calculation object in a way that mirrors the Plumed implementation.

        Parameters
        ----------
        n_atoms : int
            Number of atoms
        bonds_file : str
            File containing bonding information in the Plumed format
        angles_file : Optional[str], optional
            File containing the angle information in the Plumed format, by default None
        dihedrals_file : Optional[str], optional
            File containing the dihedral information in the Plumed format, by default None
        charges_file : Optional[str], optional
            File containing the per atom charges in the Plumed format, by default None
        lj_peratom_file : Optional[str], optional
            File containing the per atom lj parameters in the Plumed format, by default None
        n_replicas : int, optional
            Number of replicas (Only adjust this when training models for Ensemble simulations), by default 1
        """
        def read_tokens(path):
            out = []
            with open(path, "r") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    toks = s.split()
                    out.append(toks)
            return out
        bonds = []
        for toks in read_tokens(bonds_file):
            if len(toks) < 4:
                continue
            i,j,k,r0 = int(toks[0])-1, int(toks[1])-1, float(toks[2]), float(toks[3])
            if i<0 or j<0:
                raise ValueError("BONDS_FILE indices must be 1-based within GROUP")
            bonds.append((i,j,k,r0))
        angles = []
        if angles_file:
            for toks in read_tokens(angles_file):
                if len(toks) < 5:
                    continue
                i,j,k,kk,t0deg = int(toks[0])-1, int(toks[1])-1, int(toks[2])-1, float(toks[3]), float(toks[4])
                if i<0 or j<0 or k<0:
                    raise ValueError("ANGLES_FILE indices must be 1-based within GROUP")
                angles.append((i,j,k,kk, t0deg))
        dihedrals = []
        if dihedrals_file:
            for toks in read_tokens(dihedrals_file):
                if len(toks) < 7:
                    continue
                i,j,k,l,kk,n,p0deg = int(toks[0])-1, int(toks[1])-1, int(toks[2])-1, int(toks[3])-1, float(toks[4]), int(toks[5]), float(toks[6])
                if i<0 or j<0 or k<0 or l<0:
                    raise ValueError("DIHEDRALS_FILE indices must be 1-based within GROUP")
                dihedrals.append((i,j,k,l,kk,n,p0deg))
        charges = None
        if charges_file:
            seq = []
            idx_map: Dict[int,float] = {}
            mixed_indexed = False
            mixed_seq = False
            for toks in read_tokens(charges_file):
                if len(toks)==1:
                    if mixed_indexed:
                        raise ValueError("CHARGES_FILE mixes formats")
                    mixed_seq = True
                    seq.append(float(toks[0]))
                elif len(toks)==2:
                    if mixed_seq:
                        raise ValueError("CHARGES_FILE mixes formats")
                    mixed_indexed = True
                    i,q = int(toks[0])-1, float(toks[1])
                    if i<0 or i>=n_atoms:
                        raise ValueError("CHARGES_FILE index out of range")
                    idx_map[i] = q
                else:
                    raise ValueError("CHARGES_FILE: invalid line")
            if mixed_indexed:
                if len(idx_map)!=n_atoms:
                    raise ValueError("CHARGES_FILE indexed mode: one entry per atom required")
                charges = [idx_map[i] for i in range(n_atoms)]
            else:
                if len(seq)!=n_atoms:
                    raise ValueError("CHARGES_FILE sequential mode: one charge per atom required")
                charges = seq
        lj_sigma_i = None
        lj_epsilon_i = None
        if lj_peratom_file:
            lj_sigma_i = [0.0] * n_atoms
            lj_epsilon_i = [0.0] * n_atoms
            filled = [False] * n_atoms
            mode_seq = False
            mode_indexed = False
            with open(lj_peratom_file, "r") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    toks = s.split()
                    if len(toks) == 2:
                        if mode_indexed:
                            raise ValueError("LJ_PERATOM_FILE mixes formats")
                        mode_seq = True
                        idx = sum(filled)
                        if idx >= n_atoms:
                            raise ValueError("LJ_PERATOM_FILE has more lines than atoms (sequential mode)")
                        sig = float(toks[0]); eps = float(toks[1])
                        lj_sigma_i[idx] = sig
                        lj_epsilon_i[idx] = eps
                        filled[idx] = True
                    elif len(toks) == 3:
                        if mode_seq:
                            raise ValueError("LJ_PERATOM_FILE mixes formats")
                        mode_indexed = True
                        idx = int(toks[0]) - 1
                        if idx < 0 or idx >= n_atoms:
                            raise ValueError("LJ_PERATOM_FILE index out of range (1..n_atoms)")
                        sig = float(toks[1]); eps = float(toks[2])
                        lj_sigma_i[idx] = sig
                        lj_epsilon_i[idx] = eps
                        filled[idx] = True
                    else:
                        raise ValueError("LJ_PERATOM_FILE invalid line; expected 2 or 3 tokens")
            if mode_indexed:
                if not all(filled):
                    missing = [i+1 for i,v in enumerate(filled) if not v]
                    raise ValueError(f"LJ_PERATOM_FILE missing entries for atoms: {missing}")
            else:
                if sum(filled) != n_atoms:
                    raise ValueError("LJ_PERATOM_FILE sequential mode must have exactly one line per atom")
        kdict = dict(kwargs)
        if lj_sigma_i is not None:
            kdict.update(lj_sigma_i=lj_sigma_i, lj_epsilon_i=lj_epsilon_i)
        return ReducedFFEnergy(
            n_atoms=n_atoms, bonds=bonds, angles=angles, dihedrals=dihedrals,
            charges=charges, n_replicas=n_replicas, **kdict
        )

    def _apply_force_mask(self, pos: torch.Tensor):
        if self.force_mask is None:
            return pos
        mask = self.force_mask.to(pos.device, dtype=pos.dtype).view(1, 1, self.n_atoms, 1)
        return pos * mask + pos.detach() * (1 - mask)

    def _pairwise_terms(self, pos: torch.Tensor, box: Optional[torch.Tensor]):
        B, N, D = pos.shape
        rij = _pairwise_displacements(pos, box)
        r = torch.linalg.norm(rij, dim=-1).clamp_min(1e-12)
        iu, ju = torch.triu_indices(N, N, offset=1, device=pos.device)
        r_ij = r[:, iu, ju]
        excl = torch.zeros((N, N), dtype=torch.bool, device=pos.device)
        if self.exclude12:
            for i in range(N):
                for j in self._excl12[i]:
                    excl[i, j] = True; excl[j, i] = True
        if self.exclude13:
            for i in range(N):
                for j in self._excl13[i]:
                    excl[i, j] = True; excl[j, i] = True
        scale = torch.ones((N, N), dtype=pos.dtype, device=pos.device)
        if len(self._scale14) > 0:
            for (i,j), s in self._scale14.items():
                scale[i,j] = s; scale[j,i] = s
        excl_u = excl[iu, ju]
        scale_u = scale[iu, ju].view(1, -1)
        return iu, ju, r_ij, excl_u, scale_u

    def _compute_total_energy_batch(self, pos: torch.Tensor):
        device = pos.device
        dtype = pos.dtype
        box = None if (self.box is None or not self.pbc) else self.box.to(device=device, dtype=dtype)
        if self.force_mask is not None:
            fm = self.force_mask.to(device=device, dtype=dtype).view(1, self.n_atoms, 1)
            pos = pos * fm + pos.detach() * (1 - fm)
        Btot = pos.shape[0]
        total = torch.zeros(Btot, dtype=dtype, device=device)
        if len(self.bonds)>0:
            idx_i = torch.tensor([b[0] for b in self.bonds], device=device)
            idx_j = torch.tensor([b[1] for b in self.bonds], device=device)
            k = torch.tensor([b[2] for b in self.bonds], device=device, dtype=dtype).view(1,-1)
            r0 = torch.tensor([b[3] for b in self.bonds], device=device, dtype=dtype).view(1,-1)
            ri = pos[:, idx_i, :]; rj = pos[:, idx_j, :]
            rij = ri - rj
            if box is not None: rij = _min_image(rij, box)
            r = torch.linalg.norm(rij, dim=-1).clamp_min(1e-12)
            Eb = 0.5 * k * (r - r0)**2
            total += Eb.sum(dim=1)
        if len(self.angles)>0:
            idx_i = torch.tensor([a[0] for a in self.angles], device=device)
            idx_j = torch.tensor([a[1] for a in self.angles], device=device)
            idx_k = torch.tensor([a[2] for a in self.angles], device=device)
            kk = torch.tensor([a[3] for a in self.angles], device=device, dtype=dtype).view(1,-1)
            theta0 = torch.tensor([a[4] for a in self.angles], device=device, dtype=dtype) * (torch.pi/180.0)
            theta0 = theta0.view(1,-1)
            ri = pos[:, idx_i, :]; rj = pos[:, idx_j, :]; rk = pos[:, idx_k, :]
            vji = ri - rj; vjk = rk - rj
            if box is not None:
                vji = _min_image(vji, box); vjk = _min_image(vjk, box)
            th = _angle(vji, vjk)
            Ea = 0.5 * kk * (th - theta0)**2
            total += Ea.sum(dim=1)
        if len(self.dihedrals)>0:
            idx_i = torch.tensor([d[0] for d in self.dihedrals], device=device)
            idx_j = torch.tensor([d[1] for d in self.dihedrals], device=device)
            idx_k = torch.tensor([d[2] for d in self.dihedrals], device=device)
            idx_l = torch.tensor([d[3] for d in self.dihedrals], device=device)
            k_t = torch.tensor([d[4] for d in self.dihedrals], device=device, dtype=dtype).view(1,-1)
            n   = torch.tensor([d[5] for d in self.dihedrals], device=device, dtype=dtype).view(1,-1)
            phi0 = torch.tensor([d[6] for d in self.dihedrals], device=device, dtype=dtype) * (torch.pi/180.0)
            phi0 = phi0.view(1,-1)
            pi = pos[:, idx_i, :]; pj = pos[:, idx_j, :]; pk = pos[:, idx_k, :]; pl = pos[:, idx_l, :]
            phi = _dihedral_pbc(pi, pj, pk, pl, box)
            Ed = k_t * (1.0 - torch.cos(n * (phi - phi0)))
            total += Ed.sum(dim=1)
        if self.use_lj or self.use_coulomb:
            iu, ju, r_ij, excl_u, scale_u = self._pairwise_terms(pos, box)
            if self.use_lj:
                if self.lj_peratom:
                    sig_i = _ensure_tensor(self.lj_sigma_i, device=pos.device, dtype=pos.dtype)
                    eps_i = _ensure_tensor(self.lj_epsilon_i, device=pos.device, dtype=pos.dtype)
                    if self.lj_comb == "LB":
                        sij = 0.5 * (sig_i[iu] + sig_i[ju])
                        eij = torch.sqrt((eps_i[iu] * eps_i[ju]).clamp_min(0))
                    else:
                        sij = torch.sqrt((sig_i[iu] * sig_i[ju]).clamp_min(0))
                        eij = torch.sqrt((eps_i[iu] * eps_i[ju]).clamp_min(0))
                else:
                    m = iu.numel()
                    sij = torch.full((m,), float(self.lj_sigma), device=pos.device, dtype=pos.dtype)
                    eij = torch.full((m,), float(self.lj_epsilon), device=pos.device, dtype=pos.dtype)
                cutoff_mask = (r_ij < self.lj_cutoff).to(pos.dtype)
                invr = 1.0 / r_ij
                sr = sij.view(1,-1) * invr
                sr2 = sr*sr
                sr6 = sr2*sr2*sr2
                sr12 = sr6*sr6
                v_lj = 4.0 * eij.view(1,-1) * (sr12 - sr6)
                v_lj = v_lj * scale_u * cutoff_mask * (~excl_u).to(pos.dtype)
                total += v_lj.sum(dim=1)
            if self.use_coulomb:
                q = _ensure_tensor(self.charges, device=pos.device, dtype=pos.dtype)
                qq = q[iu] * q[ju]
                if self.coulomb_cutoff > 0.0:
                    c_mask = (r_ij < self.coulomb_cutoff).to(pos.dtype)
                else:
                    c_mask = torch.ones_like(r_ij, dtype=pos.dtype)
                invr = 1.0 / r_ij
                v_c = (self.ke.to(pos.dtype) * qq.view(1,-1) * invr) / self.epsilon_r
                v_c = v_c * scale_u * c_mask * (~excl_u).to(pos.dtype)
                total += v_c.sum(dim=1)
        return total

    def _parse_input(self, x: torch.Tensor) -> torch.Tensor:
        N = self.n_atoms; D = self.ndim; R = self.n_replicas
        if x.dim()==1:
            x = x.view(1, -1)
        if x.dim()==2:
            if x.shape[1] == N*D*R:
                pos = x.view(x.shape[0], R, N, D)
            elif x.shape[1] == N*D and R==1:
                pos = x.view(x.shape[0], 1, N, D)
            else:
                raise ValueError(f"Unexpected flattened shape {tuple(x.shape)} for n_replicas={R}")
        elif x.dim()==3:
            # New case: (B, R*N, D)
            if x.shape[1] == R*N and x.shape[2] == D:
                pos = x.view(x.shape[0], R, N, D)
            # Existing case: (B, R, N*D)
            elif x.shape[1]==R and x.shape[2]==N*D:
                pos = x.view(x.shape[0], R, N, D)
            else:
                raise ValueError("3D input must be (B, R, N*D) or (B, R*N, D)")
        elif x.dim()==4:
            if x.shape[1]==R and x.shape[2]==N and x.shape[3]==D:
                pos = x
            else:
                raise ValueError("4D input must be (B, R, N, D)")
        else:
            raise ValueError("Input must have dim 1..4")
        return pos

    def _total_energy(self, x: torch.Tensor):
        pos_all = self._parse_input(x)
        B, R, N, D = pos_all.shape
        pos_flat = pos_all.view(B*R, N, D)
        energies = self._compute_total_energy_batch(pos_flat)
        energies = energies.view(B, R)
        return energies

    def forward(self, x: torch.Tensor):
        e = self._total_energy(x)
        if self.n_replicas == 1:
            return e.view(-1,1)
        return e

    def components(self, x: torch.Tensor):
        pos_all = self._parse_input(x)
        B, R, N, D = pos_all.shape
        pos_flat = pos_all.view(B*R, N, D)
        device = pos_flat.device
        dtype = pos_flat.dtype
        box = None if (self.box is None or not self.pbc) else self.box.to(device=device, dtype=dtype)
        if self.force_mask is not None:
            fm = self.force_mask.to(device=device, dtype=dtype).view(1, N, 1)
            pos_flat = pos_flat * fm + pos_flat.detach() * (1 - fm)
        out = {"bonds": torch.zeros(pos_flat.shape[0], dtype=dtype, device=device),
               "angles": torch.zeros(pos_flat.shape[0], dtype=dtype, device=device),
               "dihedrals": torch.zeros(pos_flat.shape[0], dtype=dtype, device=device),
               "lj": torch.zeros(pos_flat.shape[0], dtype=dtype, device=device),
               "coul": torch.zeros(pos_flat.shape[0], dtype=dtype, device=device)}
        if len(self.bonds)>0:
            idx_i = torch.tensor([b[0] for b in self.bonds], device=device)
            idx_j = torch.tensor([b[1] for b in self.bonds], device=device)
            k = torch.tensor([b[2] for b in self.bonds], device=device, dtype=dtype).view(1,-1)
            r0 = torch.tensor([b[3] for b in self.bonds], device=device, dtype=dtype).view(1,-1)
            ri = pos_flat[:, idx_i, :]; rj = pos_flat[:, idx_j, :]
            rij = ri - rj
            if box is not None: rij = _min_image(rij, box)
            r = torch.linalg.norm(rij, dim=-1).clamp_min(1e-12)
            Eb = 0.5 * k * (r - r0)**2
            out["bonds"] += Eb.sum(dim=1)
        if len(self.angles)>0:
            idx_i = torch.tensor([a[0] for a in self.angles], device=device)
            idx_j = torch.tensor([a[1] for a in self.angles], device=device)
            idx_k = torch.tensor([a[2] for a in self.angles], device=device)
            kk = torch.tensor([a[3] for a in self.angles], device=device, dtype=dtype).view(1,-1)
            theta0 = torch.tensor([a[4] for a in self.angles], device=device, dtype=dtype) * (torch.pi/180.0)
            theta0 = theta0.view(1,-1)
            ri = pos_flat[:, idx_i, :]; rj = pos_flat[:, idx_j, :]; rk = pos_flat[:, idx_k, :]
            vji = ri - rj; vjk = rk - rj
            if box is not None:
                vji = _min_image(vji, box); vjk = _min_image(vjk, box)
            th = _angle(vji, vjk)
            Ea = 0.5 * kk * (th - theta0)**2
            out["angles"] += Ea.sum(dim=1)
        if len(self.dihedrals)>0:
            idx_i = torch.tensor([d[0] for d in self.dihedrals], device=device)
            idx_j = torch.tensor([d[1] for d in self.dihedrals], device=device)
            idx_k = torch.tensor([d[2] for d in self.dihedrals], device=device)
            idx_l = torch.tensor([d[3] for d in self.dihedrals], device=device)
            k_t = torch.tensor([d[4] for d in self.dihedrals], device=device, dtype=dtype).view(1,-1)
            n   = torch.tensor([d[5] for d in self.dihedrals], device=device, dtype=dtype).view(1,-1)
            phi0 = torch.tensor([d[6] for d in self.dihedrals], device=device, dtype=dtype) * (torch.pi/180.0)
            phi0 = phi0.view(1,-1)
            pi = pos_flat[:, idx_i, :]; pj = pos_flat[:, idx_j, :]; pk = pos_flat[:, idx_k, :]; pl = pos_flat[:, idx_l, :]
            phi = _dihedral_pbc(pi, pj, pk, pl, box)
            Ed = k_t * (1.0 - torch.cos(n * (phi - phi0)))
            out["dihedrals"] += Ed.sum(dim=1)
        if self.use_lj or self.use_coulomb:
            iu, ju, r_ij, excl_u, scale_u = self._pairwise_terms(pos_flat, box)
            if self.use_lj:
                if self.lj_peratom:
                    sig_i = _ensure_tensor(self.lj_sigma_i, device=pos_flat.device, dtype=pos_flat.dtype)
                    eps_i = _ensure_tensor(self.lj_epsilon_i, device=pos_flat.device, dtype=pos_flat.dtype)
                    if self.lj_comb == "LB":
                        sij = 0.5 * (sig_i[iu] + sig_i[ju])
                        eij = torch.sqrt((eps_i[iu] * eps_i[ju]).clamp_min(0))
                    else:
                        sij = torch.sqrt((sig_i[iu] * sig_i[ju]).clamp_min(0))
                        eij = torch.sqrt((eps_i[iu] * eps_i[ju]).clamp_min(0))
                else:
                    m = iu.numel()
                    sij = torch.full((m,), float(self.lj_sigma), device=pos_flat.device, dtype=pos_flat.dtype)
                    eij = torch.full((m,), float(self.lj_epsilon), device=pos_flat.device, dtype=pos_flat.dtype)
                cutoff_mask = (r_ij < self.lj_cutoff).to(pos_flat.dtype)
                invr = 1.0 / r_ij
                sr = sij.view(1,-1) * invr
                sr2 = sr*sr
                sr6 = sr2*sr2*sr2
                sr12 = sr6*sr6
                v_lj = 4.0 * eij.view(1,-1) * (sr12 - sr6)
                v_lj = v_lj * scale_u * cutoff_mask * (~excl_u).to(pos_flat.dtype)
                out["lj"] += v_lj.sum(dim=1)
            if self.use_coulomb:
                q = _ensure_tensor(self.charges, device=pos_flat.device, dtype=pos_flat.dtype)
                qq = q[iu] * q[ju]
                if self.coulomb_cutoff > 0.0:
                    c_mask = (r_ij < self.coulomb_cutoff).to(pos_flat.dtype)
                else:
                    c_mask = torch.ones_like(r_ij, dtype=pos_flat.dtype)
                invr = 1.0 / r_ij
                v_c = (self.ke.to(pos_flat.dtype) * qq.view(1,-1) * invr) / self.epsilon_r
                v_c = v_c * scale_u * c_mask * (~excl_u).to(pos_flat.dtype)
                out["coul"] += v_c.sum(dim=1)
        for k in list(out.keys()):
            out[k] = out[k].view(B, R)
        out["total"] = out["bonds"] + out["angles"] + out["dihedrals"] + out["lj"] + out["coul"]