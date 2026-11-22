"""Standalone runtime patch for YuE2 on Windows (app-local, NOT upstream).

Root cause (verified 2026-11-10): the official yue2_infer wheel keeps AR /
VAE weights in DTensor sharding even at quantization="none". Moving those
modules with nn.Module.to() crashes torch 2.10+cu130 on Windows inside
torch.nn.modules.module.convert (SIGSEGV) — under both offload_ar=True
(nar._offload_ar) and VAE decode (pipeline.decode's model.to(device)).

ComfyUI never hits this because comfy.model_management.load_models_gpu()
materializes meta/DTensor parameters (to_empty + copy_) before anything
calls .to(). The standalone app must do the same: after the pipeline
constructs its models, materialize every non-material parameter in-place,
then everything behaves like a normal eager model.

Call install() once after bootstrap, before the first YuE2 load.
"""
from __future__ import annotations


def _materialize_module(module, device) -> None:
    import torch
    for name, param in list(module.named_parameters(recurse=True)):
        if param.is_meta or isinstance(param, torch.distributed.tensor.DTensor):
            local = getattr(param, "to_local", lambda: param)()
            fresh = torch.nn.Parameter(
                torch.empty(tuple(local.shape), dtype=local.dtype, device=device),
                requires_grad=False,
            )
            fresh.data.copy_(local.detach().to(device=device))
            _setattr(module, name, fresh)
    for name, buf in list(module.named_buffers(recurse=True)):
        if buf is not None and (buf.is_meta
                                or isinstance(buf, torch.distributed.tensor.DTensor)):
            local = getattr(buf, "to_local", lambda: buf)()
            fresh = torch.empty(tuple(local.shape), dtype=local.dtype, device=device)
            fresh.copy_(local.detach().to(device=device))
            _setattr(module, name, fresh)


def _setattr(root, dotted, value):
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if hasattr(obj, "_parameters") and parts[-1] in obj._parameters:
        obj._parameters[parts[-1]] = value
    elif hasattr(obj, "_buffers") and parts[-1] in obj._buffers:
        obj._buffers[parts[-1]] = value
    else:
        setattr(obj, parts[-1], value)


_patched = False


def install() -> None:
    """Patch yue2.YuE2Pipeline.from_pretrained to materialize DTensor/meta
    weights before returning (and reset the per-process memory fraction the
    pipeline constructor sets)."""
    global _patched
    if _patched:
        return
    import torch
    from yue2_repo.compat.yue2_runtime import import_yue2  # ensures sys.path wired lazily

    try:
        import yue2.pipeline as yp
    except ImportError:
        return  # not imported yet; install again after first import_yue2

    cls = yp.YuE2Pipeline
    if getattr(cls, "_studio_materialize", False):
        _patched = True
        return

    orig_fpp = cls.from_pretrained.__func__ if hasattr(cls.from_pretrained, "__func__") else cls.from_pretrained

    @classmethod
    def from_pretrained_patched(klass, *args, **kwargs):
        pipe = orig_fpp(klass, *args, **kwargs)
        dev = pipe.device
        if getattr(pipe, "_model", None) is not None:
            _materialize_module(pipe._model, dev)
        if getattr(pipe, "_vae", None) is not None:
            _materialize_module(pipe._vae, dev)
        torch.cuda.set_per_process_memory_fraction(1.0, dev)
        print("[studio] materialized YuE2 DTensor weights", flush=True)
        return pipe

    cls.from_pretrained = from_pretrained_patched
    cls._studio_materialize = True
    _patched = True
