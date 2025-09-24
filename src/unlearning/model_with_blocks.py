import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from src.unlearning.decompose_matrix_on_blocks import (
    solve_B_blocks,
    build_A_blocks,
)


# TODO: make it work for arbitraty model
def get_r_list(A_dim, blocks_split_type, name, n_blocks):

    if blocks_split_type == "equal":
        r_list = equal_split(A_dim, n_blocks)

    elif blocks_split_type == "layers":
        try:
            n_in_name = int(name.split(".")[0].split("layer")[1])
        except:
            n_in_name = 0

        r_list = [0 for _ in range(n_blocks)]
        layer_group = n_in_name % n_blocks
        r_list[layer_group] = A_dim

    elif blocks_split_type == "head":
        assert n_blocks == 2
        r_list = [0 for _ in range(n_blocks)]
        if name == "fc":
            print(f"head {name} is wrapped")
            r_list[0] = A_dim
        else:
            r_list[1] = A_dim
    return r_list


class DecomposedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_blocks,
        W_init,
        bias_init,
        blocks_mode="qr",
        device="cpu",
        blocks_split_type="equal",
        name=None,
    ):
        super().__init__()

        self.shape = W_init.shape

        r_list = get_r_list(in_features, blocks_split_type, name, n_blocks)

        # build orthogonal subspaces
        self.A_blocks = build_A_blocks(
            in_features, r_list, blocks_mode=blocks_mode, device=device
        )
        # in x r_i

        # compute B_blocks from W_init
        B_blocks = solve_B_blocks(self.A_blocks, W_init.T)  # (r_i, out)

        # register A as buffers (fixed), B as parameters
        self.B_blocks = nn.ParameterList(
            [nn.Parameter(B.clone().T) for B in B_blocks]  # store as (out, r_i)
        )
        self.A_blocks_buf = [
            self.register_buffer(f"A_{i}", A.clone(), persistent=False)
            for i, A in enumerate(self.A_blocks)
        ]

        # bias
        if bias_init is not None:
            self.bias = DecomposedVector(
                bias_init,
                n_blocks=n_blocks,
                blocks_mode=blocks_mode,
                device=device,
                blocks_split_type=blocks_split_type,
                name=name,
            )
        else:
            self.bias = None

    def forward(self, x):
        # reconstruct weight
        W, bias = self.reconstruct_weight()
        return F.linear(x, W, bias)

    def reconstruct_weight(self):
        """Reconstruct weight tensor of shape (out, in, kH, kW)."""
        W = sum(B @ getattr(self, f"A_{i}").T for i, B in enumerate(self.B_blocks))
        if self.bias is not None:
            bias = self.bias.reconstruct_weight()
        else:
            bias = None

        assert (
            W.shape == self.shape
        ), f"Shape of reconstructed W {W.shape} != initial shape {self.shape}"
        return W, bias


class DecomposedConv2d(nn.Module):
    """
    Conv2d with weight parameterized as:
        W_mat = sum_i B_i A_i^T,  where W_mat ∈ R^{out x (in*kH*kW)}.
    A_i are fixed (buffers), B_i are trainable Parameters.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        n_blocks,
        W_init=None,
        bias_init=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        blocks_mode="qr",
        device="cpu",
        blocks_split_type="equal",
        name=None,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size

        # ResNet basic blocks use groups=1; grouped convs are not handled here.
        assert groups == 1, "This DecomposedConv2d currently supports groups=1 only."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH, self.kW = kH, kW
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        d = in_channels * kH * kW
        dev = device if W_init is None else W_init.device

        # Build orthonormal A blocks on the same device as W_init
        r_list = get_r_list(out_channels, blocks_split_type, name, n_blocks)

        A_blocks = build_A_blocks(
            out_channels, r_list, blocks_mode=blocks_mode, device=dev
        )
        for i, A in enumerate(A_blocks):
            self.register_buffer(f"A_{i}", A, persistent=False)
        self.num_blocks = len(r_list)

        # Compute initial B blocks
        # W_init is (out, in, kH, kW) -> flatten to (out, d)
        W_mat = W_init.reshape(out_channels, d).to(dev)

        B_blocks = solve_B_blocks(A_blocks, W_mat)  # list of (out x r_i)

        self.B_blocks = nn.ParameterList([nn.Parameter(B.clone()) for B in B_blocks])

        # Bias
        if bias_init is None:
            self.bias = None
        else:
            self.bias = DecomposedVector(
                bias_init,
                n_blocks=n_blocks,
                blocks_mode=blocks_mode,
                device=device,
                blocks_split_type=blocks_split_type,
                name=name,
            )

    def reconstruct_weight(self):
        """Reconstruct weight tensor of shape (out, in, kH, kW)."""
        # W_mat = sum_i (A_i @ B_i)  -> (out x d)
        W_mat = 0
        for i, B in enumerate(self.B_blocks):
            A = getattr(self, f"A_{i}")  # (out x r_i)

            # We don't NEED to reconstruct it every time, but here for easy implementation we do it
            W_mat = W_mat + (A @ B)  # (out x d)

        W = W_mat.reshape(self.out_channels, self.in_channels, self.kH, self.kW)

        if self.bias is not None:
            bias = self.bias.reconstruct_weight()
        else:
            bias = None
        return W, bias

    def forward(self, x):
        W, bias = self.reconstruct_weight()
        return F.conv2d(
            x,
            W,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class DecomposedVector(nn.Module):
    def __init__(
        self,
        w_init,
        n_blocks,
        blocks_mode="qr",
        device="cpu",
        blocks_split_type="equal",
        name=None,
    ):
        super().__init__()

        length = w_init.shape[0]

        r_list = get_r_list(length, blocks_split_type, name, n_blocks)

        # build orthogonal subspaces
        self.A_blocks = build_A_blocks(
            length, r_list, blocks_mode=blocks_mode, device=device
        )

        # compute B_blocks from w_init (as m x 1)
        w_init = w_init.view(-1, 1)  # ensure shape (m,1)
        B_blocks = solve_B_blocks(self.A_blocks, w_init)  # each (r_i, 1)

        # register A as buffers (fixed), B as parameters
        self.B_blocks = nn.ParameterList(
            [nn.Parameter(B.clone()) for B in B_blocks]  # store as (r_i, 1)
        )
        self.A_blocks_buf = [
            self.register_buffer(f"A_{i}", A.clone(), persistent=False)
            for i, A in enumerate(self.A_blocks)
        ]

    def forward(self):
        return self.reconstruct_weight()

    def reconstruct_weight(self):
        w = sum(getattr(self, f"A_{i}") @ B for i, B in enumerate(self.B_blocks))
        return w.view(-1)


class DecomposedBatchNorm(nn.Module):
    def __init__(
        self,
        num_features,
        n_blocks,
        weight_init=None,
        bias_init=None,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device="cpu",
        blocks_mode="qr",
        blocks_split_type="equal",
        name=None,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Running stats (same as BatchNorm)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, device=device)
            )
            self.register_buffer("running_var", torch.ones(num_features, device=device))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long, device=device)
            )

        if self.affine:
            # Decompose gamma (weight)
            if weight_init is None:
                weight_init = torch.ones(num_features, device=device)
            self.weight = DecomposedVector(
                weight_init,
                n_blocks=n_blocks,
                blocks_mode=blocks_mode,
                device=device,
                blocks_split_type=blocks_split_type,
                name=name,
            )

            # Decompose beta (bias)
            if bias_init is None:
                bias_init = torch.zeros(num_features, device=device)
            self.bias = DecomposedVector(
                bias_init,
                n_blocks=n_blocks,
                blocks_mode=blocks_mode,
                device=device,
                blocks_split_type=blocks_split_type,
                name=name,
            )

        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        # Reconstruct gamma and beta
        weight = self.weight() if self.weight is not None else None
        bias = self.bias() if self.bias is not None else None

        return F.batch_norm(
            x,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            weight,
            bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


def equal_split(d: int, n_blocks: int):
    """Split integer d into n_blocks near-equal positive parts (sums to d)."""

    # if d < n_blocks:
    #     warnings.warn(
    #         f"n_blocks={n_blocks} > d={d}; using {d} blocks of size 1 and dropping the rest."
    #     )
    #     return [1] * d

    base = d // n_blocks
    rem = d % n_blocks
    return [base + 1] * rem + [base] * (n_blocks - rem)


def _wrap_one(
    parent: nn.Module,
    name: str,
    child: nn.Module,
    full_name="",
    n_blocks=2,
    device=None,
    blocks_mode="qr",
    blocks_split_type="equal",
):
    """Replace a single child module in parent if Linear/Conv2d; return (replaced_module or None)."""

    # Linear -> DecomposedLinear
    if isinstance(child, nn.Linear):

        W = child.weight.data
        b = child.bias.data if child.bias is not None else None
        wrapped = DecomposedLinear(
            child.in_features,
            child.out_features,
            n_blocks=n_blocks,
            W_init=W,
            bias_init=b,
            blocks_mode=blocks_mode,
            device=device,
            name=full_name,
            blocks_split_type=blocks_split_type,
        )
        setattr(parent, name, wrapped)
        return wrapped

    # Conv2d -> DecomposedConv2d
    if isinstance(child, nn.Conv2d):
        if child.groups != 1:
            warnings.warn(
                f"Skipping wrap for Conv2d '{name}' with groups={child.groups} (only groups=1 supported)."
            )
            return None
        W = child.weight.data
        b = child.bias.data if child.bias is not None else None
        k = child.kernel_size
        wrapped = DecomposedConv2d(
            in_channels=child.in_channels,
            out_channels=child.out_channels,
            kernel_size=k,
            n_blocks=n_blocks,
            W_init=W,
            bias_init=b,
            stride=child.stride,
            padding=child.padding,
            dilation=child.dilation,
            groups=child.groups,
            blocks_mode=blocks_mode,
            device=device,
            blocks_split_type=blocks_split_type,
            name=full_name,
        )
        setattr(parent, name, wrapped)
        return wrapped

    if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
        wrapped = DecomposedBatchNorm(
            num_features=child.num_features,
            n_blocks=n_blocks,
            weight_init=child.weight.data if child.affine else None,
            bias_init=child.bias.data if child.affine else None,
            eps=child.eps,
            momentum=child.momentum,
            affine=child.affine,
            track_running_stats=child.track_running_stats,
            device=device,
            blocks_mode=blocks_mode,
            blocks_split_type=blocks_split_type,
            name=full_name,
        )

        if child.track_running_stats:
            wrapped.running_mean.data.copy_(child.running_mean.data)
            wrapped.running_var.data.copy_(child.running_var.data)
            wrapped.num_batches_tracked.data.copy_(child.num_batches_tracked.data)

        setattr(parent, name, wrapped)

        return wrapped

    # Handle direct 1D parameters (e.g. biases, LayerNorm weights, etc.)
    if isinstance(child, nn.Parameter) and child.dim() == 1:
        print("One Dimention Module detected:", child)
        vec_data = child.data
        wrapped = DecomposedVector(
            w_init=vec_data,
            n_blocks=n_blocks,
            blocks_mode=blocks_mode,
            device=device,
            blocks_split_type=blocks_split_type,
            name=name,
        )
        setattr(parent, name, wrapped)
        return wrapped

    return None


def wrap_model_modules(
    model: nn.Module,
    n_blocks=2,
    device=None,
    blocks_mode="qr",
    blocks_split_type="equal",
) -> nn.Module:
    def dfs(parent: nn.Module, prefix: str = ""):
        for name, child in list(parent.named_children()):

            full_name = f"{prefix}.{name}" if prefix else name
            replaced = _wrap_one(
                parent,
                name,
                child,
                full_name=full_name,
                n_blocks=n_blocks,
                device=device,
                blocks_mode=blocks_mode,
                blocks_split_type=blocks_split_type,
            )
            if replaced is None:
                dfs(child, full_name)

    dfs(model)

    # collect all params that belong to wrapped layers (B-blocks, bias, etc.)
    wrapped_param_ids = set()
    for m in model.modules():
        if isinstance(
            m,
            (DecomposedLinear, DecomposedConv2d, DecomposedVector, DecomposedBatchNorm),
        ):
            for p in m.parameters(recurse=True):  # <-- recurse=True
                wrapped_param_ids.add(id(p))

    # warn about any other trainable params
    unfamiliar = []
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in wrapped_param_ids:
            unfamiliar.append((name, tuple(p.shape), p.__class__.__name__))

    if unfamiliar:
        msg_lines = [
            "Found trainable parameters outside decomposed Linear/Conv2d wrappers:"
        ]
        for name, shape, cls in unfamiliar:
            msg_lines.append(f" - {name}: shape={shape} ({cls})")
        warnings.warn("\n".join(msg_lines))

    return model


def wrap_model(
    model, n_blocks=2, device="cpu", blocks_mode="qr", blocks_split_type="equal"
):
    wrapped_model = wrap_model_modules(
        model,
        n_blocks=n_blocks,
        device=device,
        blocks_mode=blocks_mode,
        blocks_split_type=blocks_split_type,
    )

    return wrapped_model


def freeze_all_except_Bi(model: nn.Module, i: int, train_rest: bool = False):
    """
    Freeze all parameters except the i-th B block in each decomposed layer.
    If train_rest=True, also unfreeze parameters that do not belong to B_blocks
    (e.g., biases, BatchNorm gamma/beta, etc.).
    """
    # 1) Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) Unfreeze only the i-th B block in each decomposed layer
    for m in model.modules():
        if hasattr(m, "B_blocks"):  # works for DecomposedLinear and DecomposedConv2d
            if i < len(m.B_blocks):
                m.B_blocks[i].requires_grad = True

    # 3) Optionally unfreeze all "non-B" parameters
    if train_rest:
        for m in model.modules():
            if hasattr(m, "B_blocks"):
                # unfreeze bias if present
                if getattr(m, "bias", None) is not None:
                    m.bias.requires_grad = True
