import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torch.utils.data import Dataset, DataLoader

def compute_sharpness(
    model: nn.Module,
    dataset: Dataset,
    device: str | torch.device,
    criterion: nn.Module | None = None,
    batch_size: int = 1024,
    epsilon: float = 5e-4,
    maxiter: int = 10,
    subspace_dim: int = 100,
    seed: int = 42,
    workers: int = 0,
    optimize: bool = False
) -> float:
    """
    Вычисляет Sharpness.
    Поддерживает распределение по потокам (workers) и смешанную точность (optimize).
    """
    device_obj = torch.device(device)
    model.eval()
    torch.manual_seed(seed)

    use_cuda = device_obj.type == "cuda"
    use_amp = optimize and use_cuda

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=(workers > 0),
        drop_last=False
    )
    dataset_size = len(dataset)
    
    with torch.no_grad():
        theta0 = torch.cat([p.detach().flatten() for p in model.parameters()])
    n_params = theta0.numel()

    if subspace_dim is None or subspace_dim >= n_params:
        A_t = None
        p_dim = n_params
    else:
        A_t = torch.randn((n_params, subspace_dim), device=device_obj, dtype=torch.float32)
        p_dim = subspace_dim

    z = torch.zeros(p_dim, device=device_obj, dtype=torch.float32, requires_grad=True)
    lbfgs = torch.optim.LBFGS([z], max_iter=maxiter, line_search_fn="strong_wolfe")

    max_loss_val = [-float('inf')]
    buffers_dict = dict(model.named_buffers())

    def closure():
        lbfgs.zero_grad()
        with torch.no_grad():
            y_val = epsilon * torch.tanh(z)
            delta_val = y_val if A_t is None else A_t @ y_val
            theta_perturbed_leaf = (theta0 + delta_val).detach().requires_grad_(True)

        perturbed_params = {}
        idx = 0
        for name, prm in model.named_parameters():
            n = prm.numel()
            perturbed_params[name] = theta_perturbed_leaf[idx:idx + n].view_as(prm)
            idx += n

        total_loss_val = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device_obj, non_blocking=use_cuda)
            y_batch = y_batch.to(device_obj, non_blocking=use_cuda)
            
            with torch.autocast(device_type=device_obj.type, dtype=torch.float16, enabled=use_amp):
                out = functional_call(model, (perturbed_params, buffers_dict), x_batch)
                if criterion is not None:
                    loss_sum = criterion(out, y_batch) * y_batch.size(0)
                else:
                    loss_sum = F.cross_entropy(out, y_batch, reduction='sum')
            
            batch_obj = -loss_sum / dataset_size
            batch_obj.backward()
            total_loss_val += loss_sum.item()

        y_graph = epsilon * torch.tanh(z)
        delta_graph = y_graph if A_t is None else A_t @ y_graph
        theta_perturbed_graph = theta0 + delta_graph
        
        if theta_perturbed_leaf.grad is not None:
            theta_perturbed_graph.backward(gradient=theta_perturbed_leaf.grad)

        loss_mean = total_loss_val / max(dataset_size, 1)
        if loss_mean > max_loss_val[0]:
            max_loss_val[0] = loss_mean

        return torch.tensor(-loss_mean, device=device_obj)

    lbfgs.step(closure)

    with torch.no_grad():
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device_obj, non_blocking=use_cuda)
            y_batch = y_batch.to(device_obj, non_blocking=use_cuda)
            
            with torch.autocast(device_type=device_obj.type, dtype=torch.float16, enabled=use_amp):
                out = model(x_batch)
                if criterion is not None:
                    total_loss += (criterion(out, y_batch) * y_batch.size(0)).item()
                else:
                    total_loss += F.cross_entropy(out, y_batch, reduction='sum').item()
                
        loss_min = total_loss / max(dataset_size, 1)

    sharpness_raw = max_loss_val[0] - loss_min
    denom = 1.0 + loss_min
    
    return float('nan') if abs(denom) < 1e-8 else float(sharpness_raw / denom * 100.0)
