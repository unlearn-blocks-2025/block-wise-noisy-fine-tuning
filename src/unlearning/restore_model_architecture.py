import copy
import torch.nn as nn
import warnings


def _reconstruct_vector(decomp_vec: "DecomposedVector"):
    """Return 1D tensor reconstructed from DecomposedVector."""
    return decomp_vec().detach().clone()


def _copy_batchnorm_from_decomposed(
    dst_bn: nn.modules.batchnorm._BatchNorm, src_decomp_bn: "DecomposedBatchNorm"
):
    """Copy affine params and running stats from DecomposedBatchNorm to nn.BatchNorm."""
    # Affine parameters (gamma/beta) may be decomposed vectors
    if dst_bn.affine:
        if src_decomp_bn.weight is not None:
            dst_bn.weight.data.copy_(_reconstruct_vector(src_decomp_bn.weight))
        if src_decomp_bn.bias is not None:
            dst_bn.bias.data.copy_(_reconstruct_vector(src_decomp_bn.bias))

    # Running stats
    if dst_bn.track_running_stats and src_decomp_bn.track_running_stats:
        dst_bn.running_mean.data.copy_(src_decomp_bn.running_mean.detach())
        dst_bn.running_var.data.copy_(src_decomp_bn.running_var.detach())
        dst_bn.num_batches_tracked.data.copy_(
            src_decomp_bn.num_batches_tracked.detach()
        )


def _get_by_path(root: nn.Module, path: str):
    """Return submodule by dotted path (like named_modules keys)."""
    if path == "":
        return root
    curr = root
    for part in path.split("."):
        curr = getattr(curr, part)
    return curr


def _matchable_state_pairs(dst_state, src_state):
    """Yield (k, v) from src_state that exist in dst_state with same shape."""
    for k, v in src_state.items():
        if k in dst_state and dst_state[k].shape == v.shape:
            yield k, v


def antiwrap_model(wrapped_model: nn.Module, original_template: nn.Module) -> nn.Module:
    """
    Build a new model with the architecture of `original_template` and copy *reconstructed*
    weights from `wrapped_model` (which may contain Decomposed* modules).
    """
    new_model = copy.deepcopy(original_template)

    # 1) Walk over destination modules; for each known leaf, pull from wrapped counterpart.
    for name, dst_mod in new_model.named_modules():
        # find source module at the same path
        try:
            src_mod = _get_by_path(wrapped_model, name) if name else wrapped_model
        except AttributeError:
            # Path does not exist in wrapped model -> skip (may be newly added head, etc.)
            continue

        # Linear and Conv2d
        if isinstance(dst_mod, nn.Linear) or isinstance(dst_mod, nn.Conv2d):
            if (
                src_mod.__class__.__name__ == "DecomposedLinear"
                or src_mod.__class__.__name__ == "DecomposedConv2d"
            ):
                W, b = src_mod.reconstruct_weight()
                W = W.detach().clone()
                if b is not None:
                    b = b.detach().clone()
                if dst_mod.weight.shape != W.shape:
                    raise RuntimeError(
                        f"[antiwrap] Shape mismatch at '{name}': "
                        f"Linear weight {W.shape} -> {dst_mod.weight.shape}"
                    )
                dst_mod.weight.data.copy_(W)
                if dst_mod.bias is not None:
                    if b is None:
                        dst_mod.bias.data.zero_()
                        warnings.warn(
                            f"[antiwrap] '{name}' had no bias in wrapped; zero-initialized bias."
                        )
                    else:
                        if dst_mod.bias.shape != b.shape:
                            raise RuntimeError(
                                f"[antiwrap] Bias shape mismatch at '{name}'"
                            )
                        dst_mod.bias.data.copy_(b)
                continue

        # BatchNorm (1d/2d)
        if isinstance(dst_mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if src_mod.__class__.__name__ == "DecomposedBatchNorm":
                _copy_batchnorm_from_decomposed(dst_mod, src_mod)
                continue

    # copy of any remaining parameters/buffers with matching names & shapes.
    # This covers submodules that were not wrapped (e.g., embeddings, layernorm, heads)
    dst_sd = new_model.state_dict()
    src_sd = wrapped_model.state_dict()
    updated = {}
    for k, v in _matchable_state_pairs(dst_sd, src_sd):
        # Skip decomposed internals (heuristic: A_*, B_blocks.*) to avoid overwriting dense weights
        if (".A_" in k) or ("B_blocks" in k):
            continue
        updated[k] = v
    dst_sd.update(updated)
    new_model.load_state_dict(dst_sd, strict=False)

    return new_model
