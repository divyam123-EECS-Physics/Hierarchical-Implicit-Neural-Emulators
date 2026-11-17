"""Float64 physical-coordinate L=2 step, JVP, and VJP.

Blocks are named A = dF_H/du, B = dF_H/dz, C = dF_Z/du, K = dF_Z/dz.
This module applies A and B only. D is average pooling.

Network tensors are Omega / norm. Returned fields are physical vorticity
in float64. ModelStepPhysical._out is not used: it casts to float32.

The staggered convention is
  input  (u_n, z_{n+1} = D u_{n+1})
  output (u_{n+1}, z_{n+2} = D u_{n+2}).
Free rollouts start from (u_0, D u_1).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from energy_metric import EnergyMetric


def lock_precision() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    if torch.is_autocast_enabled():
        raise RuntimeError("autocast is enabled; energy diagnostics require it off")


def downsample(field: torch.Tensor, ds: int) -> torch.Tensor:
    """D: nonoverlapping average pool. The only use of the name D."""
    return F.avg_pool2d(field, kernel_size=int(ds), stride=int(ds))


def _image(field: torch.Tensor | np.ndarray, device, dtype) -> torch.Tensor:
    if isinstance(field, torch.Tensor):
        tensor = field.to(device=device, dtype=dtype)
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(field, dtype=np.float64))).to(
            device=device, dtype=dtype
        )
    if tensor.ndim == 2:
        tensor = tensor.view(1, 1, tensor.shape[0], tensor.shape[1])
    if tensor.ndim != 4:
        raise ValueError(f"expected NCHW or HW, got {tuple(tensor.shape)}")
    return tensor


class EnergyL2Step:
    """One direct step of an L=2 module with signature ema(u, [z], t) -> (high, [coarse])."""

    def __init__(self, ema: torch.nn.Module, norm: float, ds: int, device: torch.device):
        self.ema = ema
        self.norm = float(norm)
        self.ds = int(ds)
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.ema.eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def _nets(self, u_phys: torch.Tensor, z_phys: torch.Tensor):
        u_net = (u_phys / self.norm).detach()
        z_net = (z_phys / self.norm).detach()
        t = torch.ones(u_net.shape[0], device=u_net.device, dtype=u_net.dtype)
        return u_net, z_net, t

    def step(self, u_phys: torch.Tensor, z_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_net, z_net, t = self._nets(u_phys, z_phys)
        with torch.no_grad():
            high, coarses = self.ema(u_net, [z_net], t)
        return (high * self.norm).detach(), (coarses[0] * self.norm).detach()

    def jvp_A(self, u_phys: torch.Tensor, z_phys: torch.Tensor, v_phys: torch.Tensor) -> torch.Tensor:
        """A v in physical vorticity. Latent input is held fixed."""
        u_net, z_net, t = self._nets(u_phys, z_phys)
        v_net = (v_phys / self.norm).detach()

        def forward(u_in):
            high, _coarses = self.ema(u_in, [z_net], t)
            return high * self.norm

        _primal, tangent = torch.autograd.functional.jvp(forward, (u_net,), (v_net,))
        if tangent.dtype != torch.float64:
            raise RuntimeError(f"A JVP dtype {tangent.dtype}")
        return tangent.detach()

    def jvp_B(self, u_phys: torch.Tensor, z_phys: torch.Tensor, w_phys: torch.Tensor) -> torch.Tensor:
        """B w in physical vorticity. Fine input is held fixed."""
        u_net, z_net, t = self._nets(u_phys, z_phys)
        w_net = (w_phys / self.norm).detach()

        def forward(z_in):
            high, _coarses = self.ema(u_net, [z_in], t)
            return high * self.norm

        _primal, tangent = torch.autograd.functional.jvp(forward, (z_net,), (w_net,))
        if tangent.dtype != torch.float64:
            raise RuntimeError(f"B JVP dtype {tangent.dtype}")
        return tangent.detach()

    def vjp_A(self, u_phys: torch.Tensor, z_phys: torch.Tensor, cot_phys: torch.Tensor) -> torch.Tensor:
        """Euclidean (sum) adjoint A* cot, physical fine coordinates."""
        u_net, z_net, t = self._nets(u_phys, z_phys)
        u_net = u_net.detach().requires_grad_(True)

        def forward(u_in):
            high, _coarses = self.ema(u_in, [z_net], t)
            return high * self.norm

        _primal, vjp = torch.autograd.functional.vjp(forward, u_net, cot_phys.detach())
        out = (vjp / self.norm).detach()
        if out.dtype != torch.float64:
            raise RuntimeError(f"A VJP dtype {out.dtype}")
        return out

    def vjp_B(self, u_phys: torch.Tensor, z_phys: torch.Tensor, cot_phys: torch.Tensor) -> torch.Tensor:
        """Euclidean (sum) adjoint B* cot, physical latent coordinates."""
        u_net, z_net, t = self._nets(u_phys, z_phys)
        z_net = z_net.detach().requires_grad_(True)

        def forward(z_in):
            high, _coarses = self.ema(u_net, [z_in], t)
            return high * self.norm

        _primal, vjp = torch.autograd.functional.vjp(forward, z_net, cot_phys.detach())
        out = (vjp / self.norm).detach()
        if out.dtype != torch.float64:
            raise RuntimeError(f"B VJP dtype {out.dtype}")
        return out

    def finite_difference_A(
        self, u_phys: torch.Tensor, z_phys: torch.Tensor, v_phys: torch.Tensor, eps_rel: float
    ) -> torch.Tensor:
        return _central_A(self, u_phys, z_phys, v_phys, eps_rel)

    def finite_difference_B(
        self, u_phys: torch.Tensor, z_phys: torch.Tensor, w_phys: torch.Tensor, eps_rel: float
    ) -> torch.Tensor:
        return _central_B(self, u_phys, z_phys, w_phys, eps_rel)


def _rms(field: torch.Tensor) -> float:
    flat = field.detach().reshape(-1).double()
    return float(torch.sqrt(torch.mean(flat * flat)))


def _central_A(step: EnergyL2Step, u, z, v, eps_rel: float) -> torch.Tensor:
    scale = _rms(v)
    if scale == 0.0:
        return torch.zeros_like(u)
    direction = v / scale
    height = float(eps_rel) * max(_rms(u), 1.0)
    plus, _ = step.step(u + height * direction, z)
    minus, _ = step.step(u - height * direction, z)
    return ((plus - minus) / (2.0 * height)) * scale


def _central_B(step: EnergyL2Step, u, z, w, eps_rel: float) -> torch.Tensor:
    scale = _rms(w)
    if scale == 0.0:
        return torch.zeros_like(u)
    direction = w / scale
    height = float(eps_rel) * max(_rms(w), 1e-8)
    plus, _ = step.step(u, z + height * direction)
    minus, _ = step.step(u, z - height * direction)
    return ((plus - minus) / (2.0 * height)) * scale


def relative_error(left: torch.Tensor, right: torch.Tensor, floor: float = 1e-30) -> float:
    numer = float((left - right).detach().reshape(-1).double().norm())
    denom = float(right.detach().reshape(-1).double().norm())
    return numer / max(denom, floor)


def load_l2_checkpoint(checkpoint: str | Path, device: torch.device, norm: float = 10.0) -> tuple[object, EnergyL2Step, dict]:
    """Load the EMA in float64. Does not modify the checkpoint file or the loader."""
    lock_precision()
    from eval_jacobian_manual_mse_plus import load_model

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    model, _ema = load_model(str(path), torch.device(device))
    if int(getattr(model, "num_coarse", -1)) != 1:
        raise RuntimeError(f"L=2 requires num_coarse=1, got {getattr(model, 'num_coarse', None)}")
    if list(model.latent_res) != [32]:
        raise RuntimeError(f"L=2 latent_res must be [32], got {list(model.latent_res)}")
    if int(model.img_size) != 256:
        raise RuntimeError(f"img_size must be 256, got {model.img_size}")
    if int(model.ds_factors[0]) != 8:
        raise RuntimeError(f"ds_factors[0] must be 8, got {model.ds_factors}")
    model = model.double()
    ema = model.ema_vit
    if next(ema.parameters()).dtype != torch.float64:
        raise RuntimeError("EMA did not cast to float64")
    step = EnergyL2Step(ema, norm=norm, ds=int(model.ds_factors[0]), device=device)
    info = {
        "num_coarse": int(model.num_coarse),
        "latent_res": [int(r) for r in model.latent_res],
        "img_size": int(model.img_size),
        "ds_factors": [int(v) for v in model.ds_factors],
        "norm": float(norm),
        "checkpoint_sha256": sha256_file(path),
        "ema_sha256": sha256_module(ema),
    }
    return model, step, info


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for tensor in module.state_dict().values():
        array = tensor.detach().to(dtype=torch.float64).cpu().contiguous().numpy()
        digest.update(array.tobytes())
    return digest.hexdigest()


def _field_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().reshape(tensor.shape[-2], tensor.shape[-1]).cpu().numpy().astype(np.float64)


class WeightedBlocks:
    """Energy-weighted A and B with RMS input coordinates and the KE output metric.

    v_phys = sqrt(n_in) v_coord, so a Euclidean unit coordinate has physical RMS 1.
    y_coord = pack(y_phys), so ||y_coord||^2 = <y, G y>.
    """

    def __init__(self, step: EnergyL2Step, metric: EnergyMetric, u_phys: torch.Tensor, z_phys: torch.Tensor):
        self.step = step
        self.metric = metric
        self.u = u_phys.detach()
        self.z = z_phys.detach()
        self.n_fine = int(metric.nx * metric.ny)
        self.n_latent = int(self.z.shape[-2] * self.z.shape[-1])
        self.n_out = int(metric.n_coord)
        self.fine_shape = (metric.ny, metric.nx)
        self.latent_shape = (int(self.z.shape[-2]), int(self.z.shape[-1]))
        if self.n_fine != metric.n:
            raise RuntimeError("metric grid does not match the fine field")

    def _as_fine(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(
            device=self.u.device, dtype=torch.float64
        ).view(1, 1, *self.fine_shape)

    def _as_latent(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(
            device=self.u.device, dtype=torch.float64
        ).view(1, 1, *self.latent_shape)

    def apply_A(self, v_coord: np.ndarray) -> np.ndarray:
        physical = self.metric.physical_from_rms_coord(v_coord, self.fine_shape)
        tangent = self.step.jvp_A(self.u, self.z, self._as_fine(physical))
        return self.metric.pack(_field_np(tangent))

    def apply_A_adjoint(self, y_coord: np.ndarray) -> np.ndarray:
        cot = self.metric.pack_adjoint(y_coord)
        back = self.step.vjp_A(self.u, self.z, self._as_fine(cot))
        return _field_np(back).reshape(-1) * np.sqrt(float(self.n_fine))

    def apply_B(self, w_coord: np.ndarray) -> np.ndarray:
        physical = self.metric.physical_from_rms_coord(w_coord, self.latent_shape)
        tangent = self.step.jvp_B(self.u, self.z, self._as_latent(physical))
        return self.metric.pack(_field_np(tangent))

    def apply_B_adjoint(self, y_coord: np.ndarray) -> np.ndarray:
        cot = self.metric.pack_adjoint(y_coord)
        back = self.step.vjp_B(self.u, self.z, self._as_fine(cot))
        return _field_np(back).reshape(-1) * np.sqrt(float(self.n_latent))

    def apply_M_HZ(self, w_coord: np.ndarray) -> np.ndarray:
        """Ã* B̃. Potential interaction, not the signed realized cross term."""
        return self.apply_A_adjoint(self.apply_B(w_coord))

    def apply_M_HZ_adjoint(self, v_coord: np.ndarray) -> np.ndarray:
        return self.apply_B_adjoint(self.apply_A(v_coord))

    def apply_Q_E(self, product_coord: np.ndarray) -> np.ndarray:
        """[A B]* G [A B] on the RMS-scaled product coordinate (fine, latent)."""
        vec = np.asarray(product_coord, dtype=np.float64).reshape(-1)
        if vec.size != self.n_fine + self.n_latent:
            raise ValueError(f"Q_E input {vec.size} != {self.n_fine + self.n_latent}")
        y = self.apply_A(vec[: self.n_fine]) + self.apply_B(vec[self.n_fine :])
        return np.concatenate([self.apply_A_adjoint(y), self.apply_B_adjoint(y)])


def prepare_image(field: np.ndarray, device, dtype=torch.float64) -> torch.Tensor:
    return _image(field, device, dtype)
