"""ddmtorch: differentiable drift-diffusion model estimation in PyTorch."""

from .device import resolve_device, resolve_dtype
from .density import wfpt_logpdf, wfpt_pdf, choice_probability
from .simulate import sample_wfpt, simulate_wfpt_euler
from .fit import fit_mle, FitResult, loglik

__all__ = [
    "resolve_device",
    "resolve_dtype",
    "wfpt_logpdf",
    "wfpt_pdf",
    "choice_probability",
    "sample_wfpt",
    "simulate_wfpt_euler",
    "fit_mle",
    "FitResult",
    "loglik",
]

__version__ = "0.1.0"
