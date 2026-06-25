import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from gnn_utils import soft_weighted_unsorted_segment_sum, unsorted_segment_sum, _sanitize_cell_local, compute_distances_matrix_safe, CurvatureRegularizationLoss
from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import sanitize_positions_shape
import lightning


__all__ = ["GNNTransformerDescriptor", "LightningGNNTransformer"]
# Graph Transformer Convolutional layer
class TransGCL(nn.Module):
    def __init__(self, hidden_nf: int, n_heads: int, act_fn=nn.ReLU()):
        """Defines the Graph convolutional layer for graph-based models including attention heads. Do not instantiate directly.

        Parameters
        ----------
        hidden_nf : int
            Hidden dimensionality of the latent node representation.
        n_heads : int
            Number of Attention heads, 
        act_fn : torch.nn.modules.activation, optional
            PyTorch activation function to be used in the multi-layer perceptrons, by default nn.ReLU()
        """
        super(TransGCL, self).__init__()
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.dropout = nn.Dropout(p=0.1)
        for i in range(n_heads):
            # Incorporate source-target relation in message (target, target - source)
            self.add_module(
                f"edge_{i}",
                nn.Sequential(
                    nn.Linear(hidden_nf * 2, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf, hidden_nf),
                ),
            )
        for i in range(n_heads):
            att_block = nn.Sequential(
                nn.Linear(hidden_nf * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 1),
            )
            # Small init to keep logits tame
            nn.init.uniform_(att_block[-1].weight, a=-1e-3, b=1e-3)
            nn.init.zeros_(att_block[-1].bias)
            self.add_module(f"attention_{i}", att_block)

        concat_dim = hidden_nf * (n_heads + 1)
        self.pre_norm = nn.LayerNorm(concat_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )
        self.residual = True
        
        # Store modules in ModuleLists for TorchScript compatibility
        self.edge_modules = nn.ModuleList([self._modules[f"edge_{i}"] for i in range(n_heads)])
        self.attention_modules = nn.ModuleList([self._modules[f"attention_{i}"] for i in range(n_heads)])

    def edge_model(self, source: torch.Tensor, target: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        outs: List[torch.Tensor] = []
        att_logits: List[torch.Tensor] = []
        rel = target - source
        cat_src_tgt = torch.cat([source, target], dim=1)
        cat_tgt_rel = torch.cat([target, rel], dim=1)
        for i in range(self.n_heads):
            outs.append(self.edge_modules[i](cat_tgt_rel))
            att_logits.append(self.attention_modules[i](cat_src_tgt))
        return outs, att_logits  # unnormalized logits

    def node_model(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: List[torch.Tensor], att_logits: List[torch.Tensor]) -> torch.Tensor:
        row, _ = edge_index[0], edge_index[1]
        agg_parts: List[torch.Tensor] = [x]
        for i in range(self.n_heads):
            weighted = soft_weighted_unsorted_segment_sum(
                edge_attr[i], row, att_logits[i], num_segments=x.size(0)
            )
            # Scale to control variance across heads
            weighted = weighted / (self.n_heads ** 0.5)
            agg_parts.append(weighted)
        agg = torch.cat(agg_parts, dim=1)
        agg = self.pre_norm(agg)
        out = self.node_mlp(self.dropout(agg))
        if self.residual and out.shape[0] == x.shape[0] and out.shape[1] == x.shape[1]:
            out = out + x
        return out

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]
        if row.numel() == 0:
            # No edges, just return input unchanged
            return h
        edge_feat, att_logits = self.edge_model(h[row], h[col])
        h_out = self.node_model(h, edge_index, edge_feat, att_logits)
        return h_out

# Graph model
class GCL(nn.Module):
    """Defines the Graph convolutional layer for graph-based models. Do not instantiate directly.

    Parameters
    ----------
    hidden_nf : int
        Hidden dimensionality of the latent node representation.
    act_fn : torch.nn.modules.activation, optional
        PyTorch activation function to be used in the multi-layer perceptrons, by default nn.ReLU()
    """
    def __init__(self, hidden_nf: int, act_fn=nn.ReLU()):
        super(GCL, self).__init__()

        self.edge_mlp = nn.Sequential(
            # Only takes the neighbourhood node
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            # Maps to the same dimension
            nn.Linear(hidden_nf, hidden_nf),
        )

        self.node_mlp = nn.Sequential(
            # Node MLP just takes the current vector and the resulting neighbourhood vector
            nn.Linear(hidden_nf * 2, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

    def edge_model(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        out = torch.cat([source - target], dim=1)
        out = self.edge_mlp(out)
        return out

    def node_model(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        row = edge_index[0]
        # Get the summed edge vectors for each node
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)

        return out

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]

        edge_feat = self.edge_model(h[row], h[col])
        h = self.node_model(h, edge_index, edge_feat)
        return h

class GNNTransformerDescriptor(Transform):
    """Graph Transformer descriptor usable as a preprocessing step. This requires a lightning module for training.

    Two modes:
    - mode="graph": outputs a graph-level vector of size out_features per frame.
    - mode="node": outputs node-level predictions for every atom, flattened to (B, N*out_features).

    Note that training node level outputs becomes significantly more complex for large system than graph-level outputs even if the graph level output represents a simple function of the node level outputs.
    When using this architecture please cite:

    /////////////////////////////////////////////////////
        J. Chem. Theory Comput. 2024, 20, 4, 1600-1611
        https://doi.org/10.1021/acs.jctc.3c00722
    /////////////////////////////////////////////////////
    
    This is trainable with Lightning (it is a torch.nn.Module). After training, set to eval() and
    use it as a descriptor feeding a downstream CV model.
    """

    def __init__(
        self,
        n_atoms: int,
        out_features: int = 1,
        in_node_nf: int = 3,
        hidden_nf: int = 64,
        n_layers: int = 1,
        n_heads: int = 1,
        PBC: bool = True,
        cell: Union[float, List[float]] = 1.0,
        cutoff: float = 1.0,
        pool: str = "sum",
        mode: str = "graph",  # "graph" or "node"
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize the GNN Transformer descriptor.

        Parameters
        ----------
        n_atoms : int
            Number of atoms in the system.
        out_features : int
            Output features per graph (graph mode) or per node (node mode; flattened in the descriptor).
        in_node_nf : int
            Input node feature dimension (defaults to 3: xyz coordinates).
        hidden_nf : int
            Hidden/latent dimensionality: Main parameter for tuning the flexibility of the model.
        n_layers : int
            Number of transformer convolutional layers. Practical reccomendation is to keep this as small as possible to improve biasing performance and model reproducibility. Start with 1 and increase as necessary.
        n_heads : int
            Number of attention heads per layer. Setting to 0 deactivates attention and aggregates node features with a constant edge weight of 1.
        PBC : bool
            Whether to use periodic boundary conditions in distance/edge building.
        cell : float | List[float]
            Cell dimensions for PBC handling (orthorhombic).
        cutoff : float
            Distance cutoff to connect edges (in real units; applied on PBC-aware distances).
        pool : str
            Pooling on nodes for graph output: one of {"sum", "mean", "max"}.
        mode : str
            "graph" or "node". Graph returns (B, out_features); node returns (B, n_atoms*out_features).
        device : Optional device
            torch device to place parameters and computations.
        """
        self.n_atoms = int(n_atoms)
        self.mode = str(mode).lower()
        if self.mode not in ("graph", "node"):
            raise ValueError("mode must be either 'graph' or 'node'")

        # Determine descriptor out_features shape for base Transform
        desc_out = int(out_features) if self.mode == "graph" else int(n_atoms * out_features)
        super().__init__(in_features=int(n_atoms * 3), out_features=desc_out)

        self.out_features = int(out_features)
        self.in_node_nf = int(in_node_nf)
        self.hidden_nf = int(hidden_nf)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.PBC = bool(PBC)
        # Register cell as buffer so it moves with .to(device)
        if isinstance(cell, torch.Tensor):
            self.register_buffer('cell', cell.clone())
        elif cell is not None:
            self.register_buffer('cell', torch.tensor(cell, dtype=torch.float32))
        else:
            self.register_buffer('cell', None)
        self.cutoff = float(cutoff)
        self.device = torch.device(device) if device is not None else None

        # Pooling function
        pool = pool.lower()
        if pool == "sum":
            self._pool_fn = torch.sum
        elif pool == "mean":
            self._pool_fn = torch.mean
        elif pool == "max":
            self._pool_fn = torch.amax
        else:
            raise ValueError("pool must be one of {'sum','mean','max'}")

        # Encoder from node input features to hidden
        self.embedding = torch.nn.Linear(self.in_node_nf, self.hidden_nf)

        # Transformer conv layers
        if self.n_heads == 0:
            self.layers = torch.nn.ModuleList([GCL(self.hidden_nf, act_fn=torch.nn.ReLU()) for _ in range(self.n_layers)])
        else:
            self.layers = torch.nn.ModuleList([TransGCL(self.hidden_nf, self.n_heads, act_fn=torch.nn.ReLU()) for _ in range(self.n_layers)])

        # Node-level head (pre-pooling)
        self.node_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_nf, self.hidden_nf),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_nf, self.hidden_nf),
        )

        # Graph-level head (post-pooling)
        self.graph_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_nf, self.hidden_nf),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_nf, self.out_features),
        )

        # Node-level prediction head (if mode=node)
        if self.mode == "node":
            self.node_pred = torch.nn.Linear(self.hidden_nf, self.out_features)
        else:
            self.node_pred = None

    # ---------------- helpers ----------------
    def _ensure_device_dtype(
        self, 
        pos: torch.Tensor, 
        cell_override: Optional[torch.Tensor] = None
    ) -> Tuple[torch.device, torch.dtype, torch.Tensor]:
        """Ensure device/dtype consistency and prepare cell matrix.
        
        Parameters
        ----------
        pos : torch.Tensor
            Input positions tensor (determines device and dtype)
        cell_override : torch.Tensor, optional
            Runtime cell for NPT simulations. If provided, uses this instead of self.cell.
            Can be:
            - (3,) or (1, 3): [Lx, Ly, Lz] orthorhombic box lengths
            - (6,) or (1, 6): [xlo, xhi, ylo, yhi, zlo, zhi] LAMMPS bounds
            - (9,) or (1, 9): Flattened 3x3 cell matrix
            - (3, 3) or (1, 3, 3): Full cell matrix
            - (B, 3), (B, 6), (B, 9), (B, 3, 3): Per-batch cells for NPT
            
        Returns
        -------
        Tuple[device, dtype, box]
            Device, dtype, and properly shaped cell tensor as [Lx, Ly, Lz]
        """
        # Use model's device (from embedding layer weights), not input device
        # This ensures inputs are moved to match the model, not vice versa
        try:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
        except StopIteration:
            # Fallback to input device if no parameters
            device = pos.device
            dtype = pos.dtype
        B = pos.shape[0]
        
        if cell_override is not None:
            # Use runtime cell (NPT mode)
            if isinstance(cell_override, torch.Tensor):
                box = cell_override.to(device=device, dtype=dtype)
            else:
                box = torch.tensor(cell_override, device=device, dtype=dtype)
            
            # Convert various formats to [Lx, Ly, Lz] shape (3,) or (B, 3)
            box = self._convert_cell_to_lengths(box, B)
        else:
            # Use stored cell (NVT mode) - use local sanitization for device safety
            box = _sanitize_cell_local(self.cell, device, dtype)
        
        return device, dtype, box
    
    def _convert_cell_to_lengths(
        self, 
        cell: torch.Tensor, 
        B: int,
    ) -> torch.Tensor:
        """Convert various cell formats to [Lx, Ly, Lz] box lengths.
        
        Parameters
        ----------
        cell : torch.Tensor
            Cell tensor in various formats
        B : int
            Batch size (for validation)
            
        Returns
        -------
        torch.Tensor
            Cell as [Lx, Ly, Lz] shape (3,) 
        """
        # Handle different input shapes
        if cell.dim() == 0:
            # Scalar -> cubic box
            return cell.expand(3)
        
        elif cell.dim() == 1:
            if cell.shape[0] == 1:
                # Single value -> cubic box
                return cell.expand(3)
            elif cell.shape[0] == 3:
                # Already [Lx, Ly, Lz]
                return cell
            elif cell.shape[0] == 6:
                # LAMMPS bounds: [xlo, xhi, ylo, yhi, zlo, zhi]
                Lx = cell[1] - cell[0]
                Ly = cell[3] - cell[2]
                Lz = cell[5] - cell[4]
                return torch.stack([Lx, Ly, Lz])
            elif cell.shape[0] == 9:
                # Flattened 3x3 -> extract diagonal
                mat = cell.view(3, 3)
                return torch.diag(mat)
            else:
                raise ValueError(f"Unsupported 1D cell shape: {cell.shape}")
        
        elif cell.dim() == 2:
            # Batched cells (B, ...)
            if cell.shape[0] == B:
                if cell.shape[1] == 3:
                    # (B, 3) - per-batch box lengths, use first sample for edge building
                    # Note: edge_index is shared across batch, so use representative box
                    return cell[0]
                elif cell.shape[1] == 6:
                    # (B, 6) - per-batch LAMMPS bounds, use first sample
                    Lx = cell[0, 1] - cell[0, 0]
                    Ly = cell[0, 3] - cell[0, 2]
                    Lz = cell[0, 5] - cell[0, 4]
                    return torch.stack([Lx, Ly, Lz])
                elif cell.shape[1] == 9:
                    # (B, 9) - per-batch flattened matrix, use first sample
                    mat = cell[0].view(3, 3)
                    return torch.diag(mat)
            elif cell.shape == (3, 3):
                # Full 3x3 cell matrix
                return torch.diag(cell)
            elif cell.shape == (1, 3):
                return cell.squeeze(0)
            elif cell.shape == (1, 6):
                cell_1d = cell.squeeze(0)
                Lx = cell_1d[1] - cell_1d[0]
                Ly = cell_1d[3] - cell_1d[2]
                Lz = cell_1d[5] - cell_1d[4]
                return torch.stack([Lx, Ly, Lz])
            else:
                raise ValueError(f"Unsupported 2D cell shape: {cell.shape}")
        
        elif cell.dim() == 3 and cell.shape[-2:] == (3, 3):
            # (B, 3, 3) - per-batch full matrix, use first sample
            return torch.diag(cell[0])
        
        else:
            raise ValueError(f"Unsupported cell tensor shape: {cell.shape}")

    def _build_edges(self, pos: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Build a batched edge index for a cutoff graph.

        Returns a 2 x E tensor of integer indices over the flattened nodes (batchwise offset by b*N).
        
        This implementation is fully tensorized to be TorchScript-compatible (no Python loops/lists).
        """
        B, N, _ = pos.shape
        device = pos.device
        # distances: (B, N, N) - use safe version with epsilon for 2nd derivative stability
        D = compute_distances_matrix_safe(pos=pos, n_atoms=N, PBC=self.PBC, cell=box, scaled_coords=False)
        # mask edges below cutoff, exclude self
        self_mask = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1)
        cut = (D > 0) & (D <= self.cutoff) & self_mask  # (B, N, N)
        
        # Fully tensorized edge building (TorchScript compatible)
        # Get all (batch, row, col) indices where cut is True
        indices = cut.nonzero(as_tuple=False)  # (E, 3) where columns are [batch, row, col]
        
        if indices.shape[0] == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        
        batch_idx = indices[:, 0]
        row_idx = indices[:, 1]
        col_idx = indices[:, 2]
        
        # Add batch offset: node i in batch b becomes i + b*N
        offset = batch_idx * N
        row = row_idx + offset
        col = col_idx + offset
        
        edge_index = torch.stack([row, col], dim=0).to(torch.long)
        return edge_index

    def _prepare_node_features(self, pos: torch.Tensor) -> torch.Tensor:
        """Default node features: raw xyz coordinates per node."""
        # pos: (B, N, 3) -> (B*N, 3)
        B, N, _ = pos.shape
        x = pos.reshape(B * N, 3)
        if self.in_node_nf != 3:
            # project to requested input size with a linear layer if needed
            proj = getattr(self, "_proj_in", None)
            if proj is None:
                self._proj_in = torch.nn.Linear(3, self.in_node_nf)
                proj = self._proj_in
            x = proj(x)
        return x

    # --------------- forward ----------------
    def forward(
        self, 
        X: torch.Tensor, 
        edge_index: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional runtime cell for NPT simulations.
        
        Parameters
        ----------
        X : torch.Tensor
            Input positions (B, N*3) or (B, N, 3)
        edge_index : torch.Tensor, optional
            Pre-computed edge indices (2, E) for Verlet list caching
        cell : torch.Tensor, optional
            Runtime cell for NPT. If None, uses self.cell.
            Shapes: (3,), (6,), (9,), (3,3), or batched (B, ...) for per-frame cells.
            
        Returns
        -------
        torch.Tensor
            Descriptor output (B, out_features) for graph mode
        """
        # X is positions: (B, N*3) or (B, N, 3)
        pos, _ = sanitize_positions_shape(pos=X, n_atoms=self.n_atoms)
        B, N, D = pos.shape
        if N != self.n_atoms or D != 3:
            raise ValueError(f"Expected positions of shape (B, {self.n_atoms}, 3), got {tuple(pos.shape)}")

        device, dtype, box = self._ensure_device_dtype(pos, cell_override=cell)
        pos = pos.to(device=device, dtype=dtype)

        # Build graph
        if edge_index is None:
            edge_index = self._build_edges(pos, box)
        else:
            edge_index = edge_index.to(device)

        # Node features and encoder
        x = self._prepare_node_features(pos)  # (B*N, in_node_nf)
        h = self.embedding(x)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, edge_index)

        # Node head
        h = self.node_head(h)  # (B*N, hidden)

        if self.mode == "graph":
            # reshape and pool: (B, N, hidden) -> (B, hidden)
            h_b = h.view(B, N, self.hidden_nf)
            if self._pool_fn is torch.amax:
                pooled = self._pool_fn(h_b, dim=1)
            else:
                pooled = self._pool_fn(h_b, dim=1)
            out = self.graph_head(pooled)  # (B, out_features)
            return out

        # node mode: per-node predictions, then flatten to (B, N*out_features)
        assert self.node_pred is not None
        node_out = self.node_pred(h)  # (B*N, out_features)
        node_out = node_out.view(B, N * self.out_features)
        return node_out
    

class LightningGNNTransformer(lightning.LightningModule):
    """LightningModule wrapper around GNNTransformerDescriptor for training the preprocessing layer (enables Trainer.fit()).

    Usage:
      - Pretraining: create the wrapper with a descriptor (or descriptor args), fit with a datamodule
        producing batches with keys 'data' (positions) and 'labels' (targets).
      - After training: take `module.descriptor.eval()` and use it as a preprocessing Transform.
      
    Curvature Regularization (CR):
      - Optionally adds a physics-informed regularization that correlates model gradient space
        with potential energy space for more physically meaningful learned representations.
      - Enable by setting lambda_cr > 0 and providing cr_energy_calculator and cr_reference_state.
      - For further details see: J. Chem. Phys. 14 March 2026; 164 (10): 104115. https://doi.org/10.1063/5.0311722
      - Please cite the publication if you use this feature in your work.
    """

    def __init__(
        self,
        descriptor: Optional[GNNTransformerDescriptor] = None,
        # If descriptor is None, the following are used to build one
        n_atoms: Optional[int] = None,
        out_features: int = 1,
        in_node_nf: int = 3,
        hidden_nf: int = 64,
        n_layers: int = 1,
        n_heads: int = 1,
        PBC: bool = True,
        cell: Union[float, List[float]] = 1.0,
        cutoff: float = 1.0,
        pool: str = "sum",
        mode: str = "graph",
        device: Optional[Union[str, torch.device]] = None,
        # Optimization
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss: str = "mse",  # 'mse'
        options: dict = {},
        # Curvature Regularization (CR) parameters
        lambda_cr: float = 0.0,
        cr_energy_calculator: Optional[nn.Module] = None,
        cr_reference_state: Optional[torch.Tensor] = None,
        cr_step_scale: float = 0.05,
        cr_n_steps: int = 5,
        cr_decay_alpha: float = 0.9,
        cr_beta: float = 1.0,
        cr_max_loss: float = 10.0,
        cr_batch_fraction: float = 1.0,
    ):
        """Initialize the Lightning GNN Transformer.
        
        Parameters
        ----------
        descriptor : GNNTransformerDescriptor, optional
            Pre-built descriptor. If None, one is created from other params.
        n_atoms : int, optional
            Number of atoms (required if descriptor is None)
        out_features : int
            Number of output features
        in_node_nf : int
            Input node feature dimension
        hidden_nf : int
            Hidden dimension
        n_layers : int
            Number of GNN layers
        n_heads : int
            Number of attention heads
        PBC : bool
            Use periodic boundary conditions
        cell : float or list
            Cell dimensions
        cutoff : float
            Distance cutoff for edges
        pool : str
            Pooling method ('sum', 'mean', 'max')
        mode : str
            'graph' or 'node'
        device : str or torch.device, optional
            Device
        lr : float
            Learning rate
        weight_decay : float
            Weight decay
        loss : str
            Loss function ('mse')
        options : dict
            Additional options for optimizer/scheduler
        lambda_cr : float
            Weight for CR loss (0 = disabled)
        cr_energy_calculator : nn.Module, optional
            Energy calculator (e.g., EAM_FS) for CR loss
        cr_reference_state : torch.Tensor, optional
            Average representation of the reference state B to define the target for CR loss, shape (out_features,)
        cr_step_scale : float
            Position step size for CR path (in Angstroms)
        cr_n_steps : int
            Number of steps along CR path
        cr_decay_alpha : float
            Decay factor for distant steps
        cr_beta : float
            Energy scaling factor
        cr_max_loss : float
            Maximum CR loss per step
        cr_batch_fraction : float
            Fraction of batches to apply CR loss (0.0 to 1.0). This can be used to reduce the computational overhead.
        """
        super().__init__()
        if descriptor is None:
            if n_atoms is None:
                raise ValueError("n_atoms is required when descriptor is not provided")
            descriptor = GNNTransformerDescriptor(
                n_atoms=n_atoms,
                out_features=out_features,
                in_node_nf=in_node_nf,
                hidden_nf=hidden_nf,
                n_layers=n_layers,
                n_heads=n_heads,
                PBC=PBC,
                cell=cell,
                cutoff=cutoff,
                pool=pool,
                mode=mode,
                device=device,
            )
        self.descriptor = descriptor
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        # OPTIM
        self._optimizer_name = "Adam"
        self.optimizer_kwargs = {}
        self.lr_scheduler_kwargs = {}
        for o in options.keys():
            if o == "optimizer":
                self.optimizer_kwargs.update(options[o])
            elif o == "lr_scheduler":
                self.lr_scheduler_kwargs.update(options[o])

        loss = loss.lower()
        if loss == "mse":
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError("Unsupported loss: choose from {'mse'}")
        
        # Curvature Regularization setup
        self.lambda_cr = float(lambda_cr)
        self.cr_batch_fraction = float(cr_batch_fraction)
        self.cr_loss_fn: Optional[CurvatureRegularizationLoss] = None
        
        if self.lambda_cr > 0:
            if cr_energy_calculator is None:
                raise ValueError("cr_energy_calculator required when lambda_cr > 0")
            if cr_reference_state is None:
                raise ValueError("cr_reference_state required when lambda_cr > 0")
            
            self.cr_loss_fn = CurvatureRegularizationLoss(
                energy_calculator=cr_energy_calculator,
                reference_state=cr_reference_state,
                n_atoms=descriptor.n_atoms,
                step_scale=cr_step_scale,
                n_steps=cr_n_steps,
                decay_alpha=cr_decay_alpha,
                beta=cr_beta,
                max_loss=cr_max_loss,
            )
            # Store energy calculator as submodule for device management
            self.cr_energy_calculator = cr_energy_calculator

    def forward(
        self, 
        X: torch.Tensor, 
        edge_index: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional pre-computed edge_index and runtime cell.
        
        Parameters
        ----------
        X : torch.Tensor
            Input positions
        edge_index : torch.Tensor, optional
            Pre-computed edge indices for Verlet list support
        cell : torch.Tensor, optional
            Runtime cell for NPT simulations
            
        Returns
        -------
        torch.Tensor
            Descriptor output
        """
        return self.descriptor(X, edge_index=edge_index, cell=cell)

    def _prepare_targets(self, y: torch.Tensor, B: int) -> torch.Tensor:
        # Reshape targets to match descriptor outputs depending on mode
        mode = getattr(self.descriptor, "mode", "graph")
        n_atoms = getattr(self.descriptor, "n_atoms", None)
        out_features = getattr(self.descriptor, "out_features", 1)

        if mode == "graph":
            # Expect (B, out_features) or (B,) -> (B, out_features)
            y = y.reshape(B, -1)
            if y.shape[1] == 1 and out_features > 1:
                y = y.expand(B, out_features)
            return y
        else:
            # Node mode: expect (B, n_atoms*out_features) or (B, n_atoms, out_features)
            if y.dim() == 3 and y.shape[1] == n_atoms and y.shape[2] == out_features:
                y = y.reshape(B, n_atoms * out_features)
            elif y.dim() == 2 and y.shape[1] == n_atoms:
                # Single scalar per node -> tile to out_features if >1
                if out_features > 1:
                    y = y.unsqueeze(-1).expand(B, n_atoms, out_features).reshape(B, n_atoms * out_features)
            return y

    def _bounds_to_cell(self, bounds: torch.Tensor) -> torch.Tensor:
        """Convert LAMMPS-style bounds to cell dimensions.
        
        Parameters
        ----------
        bounds : torch.Tensor
            Shape (B, 6) for LAMMPS format [xlo, xhi, ylo, yhi, zlo, zhi]
            or (B, 3) already as cell dimensions [Lx, Ly, Lz]
            
        Returns
        -------
        torch.Tensor
            Cell dimensions shape (B, 3) as [Lx, Ly, Lz]
        """
        if bounds.dim() == 1:
            bounds = bounds.unsqueeze(0)
        
        if bounds.shape[-1] == 6:
            # LAMMPS format: [xlo, xhi, ylo, yhi, zlo, zhi]
            Lx = bounds[:, 1] - bounds[:, 0]
            Ly = bounds[:, 3] - bounds[:, 2]
            Lz = bounds[:, 5] - bounds[:, 4]
            return torch.stack([Lx, Ly, Lz], dim=-1)  # (B, 3)
        elif bounds.shape[-1] == 3:
            # Already cell dimensions
            return bounds
        else:
            raise ValueError(f"Unsupported bounds shape: {bounds.shape}. Expected (B, 6) or (B, 3)")

    def training_step(self, batch, batch_idx):
        # Expect a Dict-like with 'data' (positions) and 'labels' (targets)
        X = batch["data"]
        edge_index = None
        if "graph" in batch:
            g = batch["graph"]
            # Support list of per-sample graphs or pre-batched tensor
            if isinstance(g, (list, tuple)):
                # Concatenate with batch offsets
                B = X.shape[0]
                N = self.descriptor.n_atoms
                parts = []
                offset = 0
                for b in range(B):
                    gi = g[b]
                    if gi is None:
                        gi = torch.zeros((2, 0), dtype=torch.long)
                    parts.append(gi + offset)
                    offset += N
                edge_index = torch.cat(parts, dim=1) if len(parts) > 0 else None
            elif isinstance(g, torch.Tensor):
                # Accept either a single batched edge_index (2, E_tot) or padded per-sample (B, 2, E_max)
                if g.dim() == 2 and g.shape[0] == 2:
                    edge_index = g
                elif g.dim() == 3 and g.shape[1] == 2:
                    B = X.shape[0]
                    N = self.descriptor.n_atoms
                    parts = []
                    offset = 0
                    # Optional lengths per-sample
                    glen = batch.get("graph_len", None)
                    for b in range(B):
                        gb = g[b]
                        if glen is not None:
                            L = int(glen[b])
                            gb = gb[:, :L]
                        else:
                            # filter -1 padding if present
                            if gb.numel() == 0:
                                parts.append(torch.zeros((2, 0), dtype=torch.long))
                                offset += N
                                continue
                            mask = (gb[0] >= 0) & (gb[1] >= 0)
                            gb = gb[:, mask]
                        parts.append(gb.to(torch.long) + offset)
                        offset += N
                    edge_index = torch.cat(parts, dim=1) if len(parts) > 0 else None
                else:
                    # Unsupported tensor shape
                    edge_index = None
        y = batch["labels"].to(dtype=X.dtype, device=X.device)
        B = X.shape[0]
        
        # Extract bounds for NPT support (if present in batch)
        bounds = batch.get("bounds", None)
        if bounds is not None:
            bounds = bounds.to(device=X.device, dtype=X.dtype)
        
        preds = self.descriptor(X, edge_index=edge_index, cell=bounds)
        y = self._prepare_targets(y, B)
        loss_mse = self.criterion(preds, y)
        
        # Curvature Regularization loss
        loss_cr = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        if self.lambda_cr > 0 and self.cr_loss_fn is not None:
            # Apply CR loss to a fraction of batches (controlled by cr_batch_fraction)
            apply_cr = (self.cr_batch_fraction >= 1.0) or (torch.rand(1).item() < self.cr_batch_fraction)
            if apply_cr:
                loss_cr = self.cr_loss_fn(
                    descriptor=self.descriptor,
                    positions=X,
                    current_predictions=preds,
                    bounds=bounds,  # Pass bounds for NPT support
                )
                self.log("train_loss_cr", loss_cr, prog_bar=True, on_step=True, on_epoch=True)
        
        # Total loss
        loss = loss_mse + self.lambda_cr * loss_cr
        
        self.log("train_loss_mse", loss_mse, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        X = batch["data"]
        edge_index = None
        if "graph" in batch:
            g = batch["graph"]
            if isinstance(g, (list, tuple)):
                B = X.shape[0]
                N = self.descriptor.n_atoms
                parts = []
                offset = 0
                for b in range(B):
                    gi = g[b]
                    if gi is None:
                        gi = torch.zeros((2, 0), dtype=torch.long)
                    parts.append(gi + offset)
                    offset += N
                edge_index = torch.cat(parts, dim=1) if len(parts) > 0 else None
            elif isinstance(g, torch.Tensor):
                if g.dim() == 2 and g.shape[0] == 2:
                    edge_index = g
                elif g.dim() == 3 and g.shape[1] == 2:
                    B = X.shape[0]
                    N = self.descriptor.n_atoms
                    parts = []
                    offset = 0
                    glen = batch.get("graph_len", None)
                    for b in range(B):
                        gb = g[b]
                        if glen is not None:
                            L = int(glen[b])
                            gb = gb[:, :L]
                        else:
                            if gb.numel() == 0:
                                parts.append(torch.zeros((2, 0), dtype=torch.long))
                                offset += N
                                continue
                            mask = (gb[0] >= 0) & (gb[1] >= 0)
                            gb = gb[:, mask]
                        parts.append(gb.to(torch.long) + offset)
                        offset += N
                    edge_index = torch.cat(parts, dim=1) if len(parts) > 0 else None
                else:
                    edge_index = None
        y = batch["labels"].to(dtype=X.dtype, device=X.device)
        B = X.shape[0]
        
        # Extract bounds for NPT support (if present in batch)
        bounds = batch.get("bounds", None)
        if bounds is not None:
            bounds = bounds.to(device=X.device, dtype=X.dtype)
        
        preds = self.descriptor(X, edge_index=edge_index, cell=bounds)
        y = self._prepare_targets(y, B)
        loss = self.criterion(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @property
    def optimizer_name(self) -> str:
        """Optimizer name. Options can be set using optimizer_kwargs. Actual optimizer will be return during training from configure_optimizer function."""
        return self._optimizer_name

    @optimizer_name.setter
    def optimizer_name(self, optimizer_name: str):
        if not hasattr(torch.optim, optimizer_name):
            raise AttributeError(
                f"torch.optim does not have a {optimizer_name} optimizer."
            )
        self._optimizer_name = optimizer_name

    def configure_optimizers(self):
        """
        Initialize the optimizer based on self._optimizer_name and self.optimizer_kwargs.

        Returns
        -------
        torch.optim
            Torch optimizer
        """

        optimizer = getattr(torch.optim, self._optimizer_name)(
            self.parameters(), **self.optimizer_kwargs
        )

        if self.lr_scheduler_kwargs:
            scheduler_cls = self.lr_scheduler_kwargs['scheduler']
            scheduler_kwargs = {k: v for k, v in self.lr_scheduler_kwargs.items() if k != 'scheduler'}
            lr_scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
            return [optimizer] , [lr_scheduler]
        else: 
            return optimizer