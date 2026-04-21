"""
Heterogeneous Graph Neural Network for Job Shop Scheduling.

This module implements a heterogeneous GNN that explicitly models:
1. Task nodes and Machine nodes as different node types
2. Multiple edge types:
   - Machine -> Task (assignment relationship)
   - Task -> Task (precedence constraint)
   - Task -> Task (solution order on same machine)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_add
from typing import Dict, Tuple, Optional


class EdgeMessagePassing(nn.Module):
    """
    Message passing layer with edge features.
    Supports different aggregation strategies.
    """
    def __init__(self, in_dim, out_dim, edge_dim=0, aggr='mean', use_edge_features=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.aggr = aggr
        self.use_edge_features = use_edge_features and edge_dim > 0
        
        # Message transformation
        if self.use_edge_features:
            self.msg_mlp = nn.Sequential(
                nn.Linear(in_dim * 2 + edge_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
        else:
            self.msg_mlp = nn.Sequential(
                nn.Linear(in_dim * 2, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
        
        # Update transformation
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x_src, x_dst, edge_index, edge_attr=None):
        """
        Args:
            x_src: Source node features (N_src, in_dim)
            x_dst: Destination node features (N_dst, in_dim)
            edge_index: (2, E) edge indices [src_idx, dst_idx]
            edge_attr: (E, edge_dim) edge features
            
        Returns:
            Updated destination node features (N_dst, out_dim)
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # Gather node features for edges
        x_src_gathered = x_src[src_idx]  # (E, in_dim)
        x_dst_gathered = x_dst[dst_idx]  # (E, in_dim)
        
        # Compute messages
        if self.use_edge_features and edge_attr is not None:
            msg_input = torch.cat([x_src_gathered, x_dst_gathered, edge_attr], dim=-1)
        else:
            msg_input = torch.cat([x_src_gathered, x_dst_gathered], dim=-1)
        
        messages = self.msg_mlp(msg_input)  # (E, out_dim)
        
        # Aggregate messages to destination nodes
        if self.aggr == 'mean':
            aggr_msg = scatter_mean(messages, dst_idx, dim=0, dim_size=x_dst.shape[0])
        elif self.aggr == 'max':
            aggr_msg, _ = scatter_max(messages, dst_idx, dim=0, dim_size=x_dst.shape[0])
        elif self.aggr == 'sum':
            aggr_msg = scatter_add(messages, dst_idx, dim=0, dim_size=x_dst.shape[0])
        else:
            aggr_msg = scatter_mean(messages, dst_idx, dim=0, dim_size=x_dst.shape[0])
        
        # Update destination nodes
        update_input = torch.cat([x_dst, aggr_msg], dim=-1)
        x_dst_updated = self.update_mlp(update_input)
        
        return x_dst_updated


class HeteroGATLayer(nn.Module):
    """
    Graph Attention Network layer for heterogeneous graphs.
    Uses attention mechanism for edge-specific message aggregation.
    """
    def __init__(self, in_dim, out_dim, edge_dim=0, num_heads=4, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        # Query, Key, Value transformations
        self.q_linear = nn.Linear(in_dim, out_dim)
        self.k_linear = nn.Linear(in_dim, out_dim)
        self.v_linear = nn.Linear(in_dim, out_dim)
        
        # Edge feature incorporation
        if edge_dim > 0:
            self.edge_linear = nn.Linear(edge_dim, num_heads)
        else:
            self.edge_linear = None
        
        # Output projection
        self.out_linear = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, x_src, x_dst, edge_index, edge_attr=None):
        """
        Args:
            x_src: Source node features (N_src, in_dim)
            x_dst: Destination node features (N_dst, in_dim)
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features
            
        Returns:
            Updated destination features (N_dst, out_dim)
        """
        N_dst = x_dst.shape[0]
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # Transform features
        Q = self.q_linear(x_dst)  # (N_dst, out_dim)
        K = self.k_linear(x_src)  # (N_src, out_dim)
        V = self.v_linear(x_src)  # (N_src, out_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(-1, self.num_heads, self.head_dim)  # (N_dst, H, D)
        K = K.view(-1, self.num_heads, self.head_dim)  # (N_src, H, D)
        V = V.view(-1, self.num_heads, self.head_dim)  # (N_src, H, D)
        
        # Gather for edges
        Q_gathered = Q[dst_idx]  # (E, H, D)
        K_gathered = K[src_idx]  # (E, H, D)
        V_gathered = V[src_idx]  # (E, H, D)
        
        # Attention scores
        attn_scores = (Q_gathered * K_gathered).sum(dim=-1) / (self.head_dim ** 0.5)  # (E, H)
        
        # Add edge features to attention
        if self.edge_linear is not None and edge_attr is not None:
            edge_bias = self.edge_linear(edge_attr)  # (E, H)
            attn_scores = attn_scores + edge_bias
        
        # Softmax over incoming edges for each destination node
        # Use scatter_softmax
        attn_max, _ = scatter_max(attn_scores, dst_idx, dim=0, dim_size=N_dst)
        attn_scores = attn_scores - attn_max[dst_idx]
        attn_exp = torch.exp(attn_scores)
        attn_sum = scatter_add(attn_exp, dst_idx, dim=0, dim_size=N_dst)
        attn_weights = attn_exp / (attn_sum[dst_idx] + 1e-8)  # (E, H)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted message aggregation
        weighted_v = V_gathered * attn_weights.unsqueeze(-1)  # (E, H, D)
        aggr_v = scatter_add(weighted_v, dst_idx, dim=0, dim_size=N_dst)  # (N_dst, H, D)
        
        # Reshape and project
        aggr_v = aggr_v.view(N_dst, -1)  # (N_dst, out_dim)
        out = self.out_linear(aggr_v)
        
        # Residual connection if dimensions match
        if x_dst.shape[-1] == out.shape[-1]:
            out = self.layer_norm(out + x_dst)
        else:
            out = self.layer_norm(out)
        
        return out


class HeteroGNNLayer(nn.Module):
    """
    A single heterogeneous GNN layer that processes:
    1. Machine -> Task edges (assignment)
    2. Task -> Task edges (precedence)
    3. Task -> Task edges (solution order)
    
    Each edge type has its own message passing network.
    """
    def __init__(self, hidden_dim, machine_task_edge_dim=2, 
                 task_precedence_edge_dim=1, task_solution_edge_dim=1,
                 gnn_type='gat', num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        
        if gnn_type == 'gat':
            # Machine -> Task attention
            self.machine_to_task = HeteroGATLayer(
                hidden_dim, hidden_dim, machine_task_edge_dim, num_heads, dropout
            )
            # Task -> Task (precedence) attention
            self.task_precedence = HeteroGATLayer(
                hidden_dim, hidden_dim, task_precedence_edge_dim, num_heads, dropout
            )
            # Task -> Task (solution) attention
            self.task_solution = HeteroGATLayer(
                hidden_dim, hidden_dim, task_solution_edge_dim, num_heads, dropout
            )
            # Task -> Machine (reverse assignment)
            self.task_to_machine = HeteroGATLayer(
                hidden_dim, hidden_dim, machine_task_edge_dim, num_heads, dropout
            )
        else:
            # Simple message passing
            self.machine_to_task = EdgeMessagePassing(
                hidden_dim, hidden_dim, machine_task_edge_dim, aggr='mean'
            )
            self.task_precedence = EdgeMessagePassing(
                hidden_dim, hidden_dim, task_precedence_edge_dim, aggr='mean'
            )
            self.task_solution = EdgeMessagePassing(
                hidden_dim, hidden_dim, task_solution_edge_dim, aggr='mean'
            )
            self.task_to_machine = EdgeMessagePassing(
                hidden_dim, hidden_dim, machine_task_edge_dim, aggr='mean'
            )
        
        # Combine multi-source messages for tasks
        self.task_combine = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.task_norm = nn.LayerNorm(hidden_dim)
        
        # Update machine embeddings
        self.machine_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.machine_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, task_emb, machine_emb, 
                machine_task_edge_idx, machine_task_edge_val,
                other_machine_task_edge_idx, other_machine_task_edge_val,
                task_precedence_edge_idx, task_precedence_edge_val,
                task_solution_edge_idx, task_solution_edge_val):
        """
        Forward pass of heterogeneous GNN layer.
        
        Args:
            task_emb: (N_t, hidden_dim) task embeddings
            machine_emb: (N_m, hidden_dim) machine embeddings
            machine_task_edge_idx: (2, E1) machine-task edge indices
            machine_task_edge_val: (E1, 2) machine-task edge features
            other_machine_task_edge_idx: (2, E2) alternative machine-task edges
            other_machine_task_edge_val: (E2, 2) alternative edge features
            task_precedence_edge_idx: (2, E3) task precedence edges
            task_precedence_edge_val: (E3, 1) precedence edge features
            task_solution_edge_idx: (2, E4) solution order edges
            task_solution_edge_val: (E4, 1) solution edge features
            
        Returns:
            Updated task_emb, machine_emb
        """
        # 1. Machine -> Task messages (current assignment)
        if machine_task_edge_idx.shape[1] > 0:
            msg_m2t = self.machine_to_task(
                machine_emb, task_emb,
                machine_task_edge_idx, machine_task_edge_val
            )
        else:
            msg_m2t = task_emb
        
        # 2. Other Machine -> Task messages (alternative assignments)
        if other_machine_task_edge_idx.shape[1] > 0:
            msg_other_m2t = self.machine_to_task(
                machine_emb, task_emb,
                other_machine_task_edge_idx, other_machine_task_edge_val
            )
        else:
            msg_other_m2t = task_emb
        
        # 3. Task -> Task messages (precedence)
        if task_precedence_edge_idx.shape[1] > 0:
            msg_prec = self.task_precedence(
                task_emb, task_emb,
                task_precedence_edge_idx, task_precedence_edge_val
            )
        else:
            msg_prec = task_emb
        
        # 4. Task -> Task messages (solution order)
        if task_solution_edge_idx.shape[1] > 0:
            msg_sol = self.task_solution(
                task_emb, task_emb,
                task_solution_edge_idx, task_solution_edge_val
            )
        else:
            msg_sol = task_emb
        
        # Combine all messages for tasks
        task_combined = torch.cat([msg_m2t, msg_other_m2t, msg_prec, msg_sol], dim=-1)
        task_emb_new = self.task_combine(task_combined)
        task_emb_new = self.task_norm(task_emb_new + task_emb)
        
        # 5. Task -> Machine messages (for machine update)
        # Reverse the edge direction
        if machine_task_edge_idx.shape[1] > 0:
            reverse_edge_idx = torch.stack([machine_task_edge_idx[1], machine_task_edge_idx[0]], dim=0)
            msg_t2m = self.task_to_machine(
                task_emb_new, machine_emb,
                reverse_edge_idx, machine_task_edge_val
            )
        else:
            msg_t2m = machine_emb
        
        # Update machine embeddings
        machine_combined = torch.cat([machine_emb, msg_t2m], dim=-1)
        machine_emb_new = self.machine_update(machine_combined)
        machine_emb_new = self.machine_norm(machine_emb_new + machine_emb)
        
        return task_emb_new, machine_emb_new


class HeteroGNNEncoder(nn.Module):
    """
    Heterogeneous GNN Encoder for Job Shop Scheduling.
    Encodes task and machine nodes using multiple HeteroGNN layers.
    """
    def __init__(self, input_task_dim=15, input_machine_dim=11, hidden_dim=64,
                 num_layers=3, gnn_type='gat', num_heads=4, dropout=0.1,
                 machine_task_edge_dim=2, task_precedence_edge_dim=1,
                 task_solution_edge_dim=1, use_global_aggr=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_global_aggr = use_global_aggr
        
        # Input embeddings
        self.task_embed = nn.Sequential(
            nn.LayerNorm(input_task_dim),
            nn.Linear(input_task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.machine_embed = nn.Sequential(
            nn.LayerNorm(input_machine_dim),
            nn.Linear(input_machine_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            HeteroGNNLayer(
                hidden_dim=hidden_dim,
                machine_task_edge_dim=machine_task_edge_dim,
                task_precedence_edge_dim=task_precedence_edge_dim,
                task_solution_edge_dim=task_solution_edge_dim,
                gnn_type=gnn_type,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Global aggregation
        if use_global_aggr:
            self.global_aggr = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.task_global_combine = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
    def forward(self, x_tasks, x_machines, 
                machine_task_edge_idx, machine_task_edge_val,
                other_machine_task_edge_idx, other_machine_task_edge_val,
                task_precedence_edge_idx, task_precedence_edge_val,
                task_solution_edge_idx, task_solution_edge_val,
                task_batch=None, machine_batch=None, assigned_machine_idx=None):
        """
        Forward pass of the encoder.
        
        Returns:
            task_emb: (N_t, hidden_dim) task embeddings
            machine_emb: (N_m, hidden_dim) machine embeddings
        """
        # Initial embeddings
        task_emb = self.task_embed(x_tasks)
        machine_emb = self.machine_embed(x_machines)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            task_emb, machine_emb = layer(
                task_emb, machine_emb,
                machine_task_edge_idx, machine_task_edge_val,
                other_machine_task_edge_idx, other_machine_task_edge_val,
                task_precedence_edge_idx, task_precedence_edge_val,
                task_solution_edge_idx, task_solution_edge_val
            )
        
        # Global aggregation
        if self.use_global_aggr and task_batch is not None:
            # Aggregate global task and machine info
            task_global = scatter_mean(task_emb, task_batch, dim=0)  # (B, hidden)
            machine_global = scatter_mean(machine_emb, machine_batch, dim=0)  # (B, hidden)
            
            global_emb = self.global_aggr(
                torch.cat([task_global, machine_global], dim=-1)
            )  # (B, hidden)
            
            # Expand global to task level
            global_expanded = global_emb[task_batch]  # (N_t, hidden)
            
            # Get assigned machine embedding for each task
            if assigned_machine_idx is not None:
                # Handle -1 indices (unassigned)
                valid_mask = assigned_machine_idx >= 0
                machine_emb_padded = torch.cat([
                    machine_emb, 
                    torch.zeros(1, self.hidden_dim, device=machine_emb.device)
                ], dim=0)
                assigned_idx_safe = assigned_machine_idx.clone()
                assigned_idx_safe[~valid_mask] = machine_emb.shape[0]  # Point to padding
                assigned_machine_emb = machine_emb_padded[assigned_idx_safe]
            else:
                assigned_machine_emb = torch.zeros_like(task_emb)
            
            # Combine local and global information
            task_emb = self.task_global_combine(
                torch.cat([task_emb, global_expanded, assigned_machine_emb], dim=-1)
            )
        
        return task_emb, machine_emb


class HeteroGNNModel(nn.Module):
    """
    Complete Heterogeneous GNN model for task classification.

    Predicts which tasks should be "fixed" in the current RHO round.
    Supports multi-task learning with critical path prediction as auxiliary task.
    """
    def __init__(self, input_task_dim=15, input_machine_dim=11, hidden_dim=64,
                 num_layers=3, gnn_type='gat', num_heads=4, dropout=0.1,
                 machine_task_edge_dim=2, task_precedence_edge_dim=1,
                 task_solution_edge_dim=1, use_global_aggr=True,
                 x_tasks_norm=None, x_machines_norm=None,
                 use_critical_path_head=True):
        super().__init__()

        self.input_task_dim = input_task_dim
        self.input_machine_dim = input_machine_dim
        self.hidden_dim = hidden_dim
        self.use_critical_path_head = use_critical_path_head

        # Normalizers
        self.x_tasks_norm = x_tasks_norm
        self.x_machines_norm = x_machines_norm

        # Encoder
        self.encoder = HeteroGNNEncoder(
            input_task_dim=input_task_dim,
            input_machine_dim=input_machine_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            num_heads=num_heads,
            dropout=dropout,
            machine_task_edge_dim=machine_task_edge_dim,
            task_precedence_edge_dim=task_precedence_edge_dim,
            task_solution_edge_dim=task_solution_edge_dim,
            use_global_aggr=use_global_aggr
        )

        # Main classification head (Fix/Not-Fix prediction)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Auxiliary head: Critical path prediction (for multi-task learning)
        if use_critical_path_head:
            self.critical_path_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
    
    def normalize_data(self, data):
        """Normalize input features."""
        if self.x_tasks_norm is not None:
            data.x_tasks = self.x_tasks_norm.normalize(data.x_tasks)
        if self.x_machines_norm is not None:
            data.x_machines = self.x_machines_norm.normalize(data.x_machines)
        return data
    
    def get_assigned_machine_idx(self, data):
        """Get the assigned machine index for each task."""
        TASK_FEAT_IDX = 8  # Machine assignment index in task features
        
        task_m = data.x_tasks[:, TASK_FEAT_IDX].clone()
        
        # Adjust for batched graphs
        if hasattr(data, 'x_machines_batch'):
            num_machines_per_graph = torch.bincount(data.x_machines_batch)
            cumulative_machines = torch.cumsum(num_machines_per_graph, dim=0) - num_machines_per_graph
            task_m += cumulative_machines[data.x_tasks_batch]
            task_m[data.x_tasks[:, TASK_FEAT_IDX] == -1] = -1
        
        return task_m.long()
    
    def forward(self, data, return_critical=False):
        """
        Forward pass.

        Args:
            data: PyG Data/Batch object with:
                - x_tasks: (N_t, input_task_dim)
                - x_machines: (N_m, input_machine_dim)
                - overlap_machine_task_edge_idx: (2, E1)
                - overlap_machine_task_edge_val: (E1, 2)
                - other_machine_task_edge_idx: (2, E2)
                - other_machine_task_edge_val: (E2, 2)
                - task_precedence_edge_idx: (2, E3)
                - task_precedence_edge_val: (E3,)
                - task_solution_edge_idx: (2, E4)
                - task_solution_edge_val: (E4,)
                - task_label_idx: indices of tasks to classify
                - task_critical_idx: indices for critical path prediction (optional)
            return_critical: Whether to return critical path predictions.
                            Default is False (inference mode, only returns fix predictions).
                            Set to True during training for multi-task learning.

        Returns:
            If return_critical is False or no critical_path_head:
                logits_fix: (L, 1) classification logits for Fix/Not-Fix
            If return_critical is True:
                (logits_fix, logits_critical): Tuple of both predictions
        """

        # Get batch info
        task_batch = getattr(data, 'x_tasks_batch', None)
        machine_batch = getattr(data, 'x_machines_batch', None)

        # Get assigned machine indices
        assigned_machine_idx = self.get_assigned_machine_idx(data)

        # Normalize data
        data = self.normalize_data(data)

        # Ensure edge values have correct shape
        def ensure_2d(val, default_dim=1):
            if val.dim() == 1:
                return val.unsqueeze(-1)
            return val

        machine_task_edge_val = ensure_2d(data.overlap_machine_task_edge_val, 2)
        other_machine_task_edge_val = ensure_2d(data.other_machine_task_edge_val, 2)
        task_precedence_edge_val = ensure_2d(data.task_precedence_edge_val, 1)
        task_solution_edge_val = ensure_2d(data.task_solution_edge_val, 1)

        # Encode
        task_emb, machine_emb = self.encoder(
            data.x_tasks, data.x_machines,
            data.overlap_machine_task_edge_idx, machine_task_edge_val,
            data.other_machine_task_edge_idx, other_machine_task_edge_val,
            data.task_precedence_edge_idx, task_precedence_edge_val,
            data.task_solution_edge_idx, task_solution_edge_val,
            task_batch, machine_batch, assigned_machine_idx
        )

        # Main task: Fix/Not-Fix classification
        task_emb_labeled = task_emb[data.task_label_idx]
        logits_fix = self.classifier(task_emb_labeled)

        # Auxiliary task: Critical path prediction
        if return_critical and self.use_critical_path_head:
            # Use task_critical_idx if available, otherwise use task_label_idx
            critical_idx = getattr(data, 'task_critical_idx', None)
            if critical_idx is not None and len(critical_idx) > 0:
                task_emb_critical = task_emb[critical_idx]
            else:
                task_emb_critical = task_emb_labeled

            logits_critical = self.critical_path_head(task_emb_critical)
            return logits_fix, logits_critical

        return logits_fix

