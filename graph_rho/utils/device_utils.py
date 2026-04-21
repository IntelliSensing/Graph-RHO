"""Device utilities for graph_rho."""
import torch


def get_device(prefer='auto'):
    """
    Get the best available device.
    
    Args:
        prefer: 'auto', 'cuda', 'mps', or 'cpu'
        
    Returns:
        torch.device
    """
    if prefer == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif prefer == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif prefer == 'cpu':
        return torch.device('cpu')
    elif prefer == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')

