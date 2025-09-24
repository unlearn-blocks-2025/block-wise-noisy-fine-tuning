import matplotlib.pyplot as plt
from src.utils.utils import load_accuracy_history
from src.utils.compute_parameters import load_config_yaml, compute_dp_budget
import numpy as np
import math
import pandas as pd
import random
import matplotlib.colors as mcolors
from pathlib import Path


def plot_accuracy_runs_common_grid(
    x: np.ndarray,
    y_runs: np.ndarray,
    label: str = "mean",
    show_individual: bool = False,
    alpha_ribbon: float = 0.2,
):
    """
    Plot mean accuracy with a shaded interval for multiple runs on the same x-grid.

    Parameters
    ----------
    x : array of shape [T]
        Common step grid shared by all runs.
    y_runs : array of shape [n_runs, T]
        Accuracy curves for each run on the same grid.
    mode : 'ci' | 'std' | 'pct'
        Interval type: t-based confidence interval, mean±std, or percentile ribbon.
    """
    x = np.asarray(x)
    Y = np.asarray(y_runs)
    assert Y.ndim == 2 and Y.shape[1] == x.shape[0], "Shape mismatch: x vs y_runs"
    n, T = Y.shape

    mean = Y.mean(axis=0)

    std = Y.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    low, up = mean - std, mean + std
    ribbon_label = "±1 std"

    if show_individual:
        for i in range(n):
            plt.plot(x, Y[i], alpha=0.35)

    plt.plot(x, mean, label=label)
    plt.fill_between(x, low, up, alpha=alpha_ribbon, label=ribbon_label)
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)  # accuracy in [0,1]; adjust if needed
    plt.legend()
    plt.tight_layout()


def visualize(
    graphs,
    tag=None,
    figsize=(8, 5),
    alpha_ribbon=0.2,
    T_max=1000,
    one_color=True,
    fix_budget=None,
):
    groups = {}
    labels = {}
    colors = {}
    linestyles = {}
    grids = {}

    for accuracy_filename in graphs:
        if tag is not None:
            if tag not in accuracy_filename:
                continue

        if "_EXP" in accuracy_filename:
            filename = accuracy_filename.split("_EXP")[0] + ".pt"
        else:
            filename = accuracy_filename
        config_path = filename.replace(
            "experiments_results/accuracy", "experiments"
        ).replace(".pt", ".yaml")

        # # fix budget
        # if fix_budget is not None:
        #     config = load_config_yaml(config_path)
        # if "budget" in config:
        #     if config["budget"] != fix_budget:
        #         continue
        # elif "budget" in config[0]:
        #     if config[0]["budget"] != fix_budget:
        #         continue

        graph = load_accuracy_history(accuracy_filename)
        iters, accs = zip(*graph)

        if "retrain" not in filename:
            name = filename.split("/")[-1]
        else:
            name = "retrain"

        if name not in labels:
            style, dp_budget = _label_and_style_from_filename(
                name, config_path, one_color=one_color
            )

            print(dp_budget)
            if (
                fix_budget is not None
                and dp_budget is not None
                and abs(dp_budget - fix_budget) >= 1e-3
            ):
                continue
            grids[name] = iters
            labels[name] = style["label"]
            colors[name] = _pick_deterministic_color(
                name, style["palette"], style["color_seed_kind"], one_color=one_color
            )
            linestyles[name] = style["linestyle"]
            groups[name] = []

        groups[name].append(accs)

        if name == "retrain":
            gold_standart = "best"
            best_acc = max(accs)
            groups[gold_standart] = [[best_acc] * max(iters)]
            grids[gold_standart] = np.arange(max(iters))
            labels[gold_standart] = "final retrain quality"
            colors[gold_standart] = "red"
            linestyles[gold_standart] = "--"

    plt.figure(figsize=figsize)

    for name in labels:
        iters = grids[name]
        accs = np.array(groups[name])  # n_runs x acc_calculation_steps

        mean, low, up = _compute_band(accs)

        color = colors[name]
        ls = linestyles[name]
        label = labels[name]

        plt.plot(iters, mean, label=label, linestyle=ls, color=color, linewidth=1.0)
        plt.fill_between(iters, low, up, color=color, alpha=alpha_ribbon, label=None)

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.grid(True, alpha=0.3)
    if T_max is not None:
        plt.xlim(0, T_max)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()


def coefficients_a_b_c(x, eta, lam, C0, C1, eps, q):
    """
    Compute quadratic coefficients a(x), b(x), c(x) for s = 1/sigma^2.
    We use the simplified forms with z = 1 - lam*C0/C1 and K1 = 2 q C1^2 / (eps lam^2).
    """
    # Guard x \in (0,1); clip slightly away from 1 to avoid c=0
    x = float(x)
    if x >= 1.0:
        x = np.nextafter(1.0, 0.0)  # closest float < 1

    A = eta * lam * (2.0 - eta * lam)  # positive for typical eta*lam in (0,2)
    z = 1.0 - lam * C0 / C1
    K1 = (2.0 * q * (C1**2)) / (eps * (lam**2))

    a = (K1**2) * (z**2) * (1.0 - x * z) ** 2
    b = (K1 / A) * (1.0 - 2.0 * x * z + (2.0 * x * x - 1.0) * (z**2))
    c = (x * x - 1.0) / (A**2)  # <= 0 for x <= 1

    return a, b, c


def sigma2_from_x(x, eta, lam, C0, C1, eps, q):
    """
    Numerically stable solution for sigma^2(x) using the quadratic in s = 1/sigma^2:
      a s^2 + b s + c = 0.
    We return sigma^2 = (-b - sqrt(b^2 - 4ac)) / (2c), which avoids subtractive cancellation
    because c <= 0 for x in (0,1].
    """
    a, b, c = coefficients_a_b_c(x, eta, lam, C0, C1, eps, q)

    # Discriminant with safety clamp against tiny negative due to rounding
    disc = b * b - 4.0 * a * c
    if disc < 0 and disc > -1e-18:
        disc = 0.0
    if disc < 0:
        raise ValueError(f"Negative discriminant for x={x}: {disc}")

    sqrt_disc = math.sqrt(disc)

    # If c is ~0 (x ~ 1), handle gracefully by nudging x away from 1 in caller; still guard here
    if abs(c) < 1e-18:
        # Fallback to the alternative formula sigma^2 = 2a / (-b + sqrt_disc)
        denom = -b + sqrt_disc
        if abs(denom) < 1e-18:
            # As a last resort, return large value
            return float("inf")
        return (2.0 * a) / denom

    # Stable branch (preferred)
    sigma2 = (-b - sqrt_disc) / (2.0 * c)
    if sigma2 <= 0:
        # Fallback to the alternative expression if numerical issues appear
        denom = -b + sqrt_disc
        if abs(denom) < 1e-18:
            return float("inf")
        sigma2_alt = (2.0 * a) / denom
        sigma2 = sigma2_alt

    return float(sigma2)


def sigma2_series_x_root_k(base_x, K, eta, lam, C0, C1, eps, q):
    """
    Compute sigma^2(x^(1/k)) for k=1..K.
    Returns a DataFrame with columns: k, x_k, sigma2_k.
    """
    rows = []
    for k in range(1, K + 1):
        x_k = base_x ** (1.0 / k)
        # Clip x_k just below 1 to stabilize c(x)
        x_k = min(x_k, np.nextafter(1.0, 0.0))
        sig2 = sigma2_from_x(x_k, eta, lam, C0, C1, eps, q)
        rows.append((k, x_k, sig2))
    df = pd.DataFrame(rows, columns=["k", "x_k", "sigma2_k"])
    return df


def visualize_sigma_by_xk(
    eta=1e-4, lam=10.0, C0=0.01, C1=1.0, eps=0.025, q=5.0, K=12  # C_0  # C_1  # epsilon
):

    z = 1.0 - lam * C0 / C1
    base_x = z

    df = sigma2_series_x_root_k(
        base_x, K, eta=1e-4, lam=10.0, C0=0.01, C1=1.0, eps=0.025, q=5.0, K=12
    )

    plt.figure()
    plt.plot(df["k"].values, df["sigma2_k"].values, marker="o")
    plt.xlabel("k")
    plt.ylabel(r"$\sigma^2(x^{1/k})$")
    plt.title(r"$\sigma^2$ as a function of $k$ for $x^{1/k}$")
    plt.grid(True)
    plt.show()


def _label_and_style_from_filename(base_filename, config_path, one_color=True):

    if one_color:
        block_cmap = plt.cm.Greens
        noisy_ft_cmap = plt.cm.Blues
    else:
        random_colors = np.random.rand(256, 3)
        block_cmap = mcolors.ListedColormap(random_colors, name="random_block")

        noisy_ft_cmap = mcolors.ListedColormap(random_colors, name="random_block")

    if "retrain" in base_filename:
        return {
            "label": "retrain-from-scratch",
            "palette": "red",
            "color_seed_kind": "fixed",
            "linestyle": "-",
        }, None
    if "basemodel" in base_filename:
        return {
            "label": "model before unlearning",
            "palette": "black",
            "color_seed_kind": "fixed",
            "linestyle": "-",
        }, None

    # Non-baseline: relies on YAML config
    config = load_config_yaml(config_path)

    # Non-block structure (has 'eta_t' key)
    if "eta_t" in config:
        if "budget" in config:
            dp_budget = config["budget"]
        else:
            dp_budget = compute_dp_budget(
                config["epsilon"], config["q"], config["delta"]
            )
        return {
            "label": "Noisy Fine-Tuning",
            "palette": "blue",
            "color_seed_kind": "fixed",
            "linestyle": "-",
        }, dp_budget

    # Non-block structure (has 'eta_t' key)
    # if "eta_t" in config:
    #     if "budget" in config:
    #         dp_budget = config["budget"]
    #     else:
    #         dp_budget = compute_dp_budget(
    #             config["epsilon"], config["q"], config["delta"]
    #         )
    #     label = f"BASIC:{base_filename.split('/')[-1]}_dp{dp_budget:.2f}"
    #     return {
    #         "label": label,
    #         "palette": noisy_ft_cmap,
    #         "color_seed_kind": "cmap",
    #         "linestyle": "-",
    #     }, dp_budget

    # Block structure (aggregate epsilons; use first block's q, delta as in your code)
    if "budget" in config[0]:
        dp_budget = config[0]["budget"]
    else:
        epsilon_sum = sum([block["epsilon"] for i, block in config.items()])
        dp_budget = compute_dp_budget(epsilon_sum, config[0]["q"], config[0]["delta"])
    # label = (
    #     f"BLOCKs:{base_filename.split('/')[-1]}_Blocks:{len(config)}_dp:{dp_budget:.2f}"
    # )

    if "num_blocks" in config[0]:
        num_blocks = config[0]["num_blocks"]
    else:
        num_blocks = len(config)
    label = f"Block-wise Noisy Fine-Tuning (Ours)"  # f"Block-wise NFT, k={num_blocks}"

    numblocks2color = {2: "#9DFF00CF", 4: "#36BD36", 10: "#046B04"}

    if num_blocks in numblocks2color:
        palette = numblocks2color[num_blocks]
    else:
        palette = "black"

    type2color = {
        "permutation matrix": "#E6144FCF",
        "cyclic layers": "#2D3DD0CF",
        "random": "#046D00FF",
        "head": "#079BB8FF",
    }

    if "type" in config[0]:
        matrix_a_type = config[0]["type"]
        label = f"Block-wise NFT, A:{matrix_a_type}"

        palette = type2color[matrix_a_type]

    return {
        "label": label,
        "palette": palette,
        "color_seed_kind": "fixed",
        "linestyle": "-",
    }, dp_budget


def _pick_deterministic_color(key: str, palette, kind: str, one_color=False):
    if kind == "fixed":
        return palette
    rnd = random.Random(hash(key) & 0xFFFFFFFF)
    if one_color:
        u = rnd.uniform(0.3, 0.9)
    else:
        u = rnd.uniform(0, 1)
    return palette(u)


def _compute_band(Y):
    """
    Compute (mean, low, up) across runs axis=0 for a given mode.
    """
    n = Y.shape[0]
    mean = Y.mean(axis=0)
    std = Y.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    low, up = mean - std, mean + std
    return mean, low, up


def _collect_runs(graph_paths, dataset_tag, budget, one_color=True, max_iters=None):
    """
    Return dict: {method_key: {"iters": ..., "accs": [runs...], "label": ..., "color": ..., "ls": ...}}
    Filters by dataset_tag (substring) and dp budget ~= budget.
    """
    groups, labels, colors, linestyles, grids = {}, {}, {}, {}, {}
    for acc_path in graph_paths:
        # normalize filename to find its config
        filename = acc_path.split("_EXP")[0] + ".pt" if "_EXP" in acc_path else acc_path
        config_path = filename.replace(
            "experiments_results/accuracy", "experiments"
        ).replace(".pt", ".yaml")

        # parse style + budget using your helper
        name = "retrain" if "retrain" in filename else filename.split("/")[-1]

        if "basemodel" in name:
            # continue
            name = "basemodel"

        try:
            style, dp_budget = _label_and_style_from_filename(
                name, config_path, one_color=one_color
            )
        except FileNotFoundError:
            continue
        print(dp_budget)

        # print("ACC_PATH", acc_path, f"BUDGET {budget} ---- ", dp_budget)

        # budget filter (allow tiny tolerance)
        if dp_budget is not None and abs(dp_budget - budget) > 1e-3:
            continue

        # load curve
        curve = load_accuracy_history(acc_path)
        iters, accs = zip(*curve)

        # register style
        if name not in labels:
            grids[name] = np.array(iters)
            labels[name] = style["label"]
            colors[name] = _pick_deterministic_color(
                name, style["palette"], style["color_seed_kind"], one_color=one_color
            )
            linestyles[name] = style["linestyle"]
            groups[name] = []

        groups[name].append(np.array(accs))

        # add "final retrain quality" as a flat reference if retrain present
        if name == "retrain":
            best = float(np.max(accs))
            ref_key = "retrain_ref"
            groups[ref_key] = [[best] * max(iters)]
            grids[ref_key] = np.arange(max(iters))
            labels[ref_key] = "retrain acc after 182 epochs"
            colors[ref_key] = "red"
            linestyles[ref_key] = "--"
    # pack
    packed = {}
    for key in labels:
        packed[key] = {
            "iters": grids[key],
            "accs": groups[key],
            "label": labels[key],
            "color": colors[key],
            "ls": linestyles[key],
        }
    return packed


def _mean_ribbon(acc_runs):
    """Compute mean and a simple CI band."""
    arr = np.vstack(acc_runs)  # shape: n_runs x T
    mean = arr.mean(axis=0)
    # 1 standard error band
    se = (
        arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    low, up = mean - se, mean + se
    return mean, low, up


def visualize_multipanel_by_budget_and_dataset(
    experiment_types,
    budgets=[(3.0, 1.0, 0.5), (7.0, 5.0, 3.0)],
    datasets=("MNIST", "CIFAR10"),
    one_color=True,
    T_max=1000,
    alpha_ribbon=0.20,
    figsize=(20, 10),
    plotfilename="plot",
    priority_order=None,
    sharey=False,
    # [
    #     "retrain (final)",
    #     "retrain",
    #     "NFT",
    #     "Block-wise NFT, k=2",
    #     "Block-wise NFT, k=4",
    #     "Block-wise NFT, k=10",
    # ],
):
    """
    Create a 2x3 grid: rows=datasets, cols=budgets.
    Legends are shared (one for the entire figure).
    """

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    n_rows, n_cols = len(datasets), len(budgets[0])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=sharey)
    axes = np.atleast_2d(axes)

    # remember handles/labels once for a shared legend
    legend_handles = []
    legend_labels = []

    for r, ds in enumerate(datasets):
        folder = Path(f"experiments_results/accuracy/{experiment_types[r]}")
        graph_paths = [str(path) for path in list(folder.glob("*.pt"))]
        for c, eps in enumerate(budgets[r]):

            ax = axes[r, c]
            packed = _collect_runs(
                graph_paths, dataset_tag=ds, budget=eps, one_color=one_color
            )

            print(ds, eps, packed.keys())

            # pick a small canonical subset to avoid clutter
            # keys = _select_methods(packed)

            for key, v in packed.items():

                if (
                    # "full" in v["label"]
                    # or
                    "k=5" in v["label"]
                    or "k=7" in v["label"]
                    or "k=13" in v["label"]
                    or "k=3" in v["label"]
                ):
                    continue
                iters = v["iters"]
                mean, low, up = _mean_ribbon(v["accs"])
                (line,) = ax.plot(
                    iters,
                    mean,
                    v["ls"],
                    color=v["color"],
                    linewidth=1,
                    alpha=0.9,
                    label=v["label"],
                )
                ax.fill_between(iters, low, up, color=v["color"], alpha=alpha_ribbon)
                # capture one handle per unique label for the shared legend
                if v["label"] not in legend_labels:
                    legend_labels.append(v["label"])
                    legend_handles.append(line)

            ax.set_xlim(-300, T_max)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 10))
            if r == n_rows - 1:
                ax.set_xlabel("Iteration")
            ax.set_ylabel("Test Accuracy (%)")

            # if c == 0:
            #     ax.set_xlim(0, 200)
            # elif c == 1:
            #     ax.set_xlim(0, 500)

            # if c == 0 and r == 1:
            #     ax.set_ylim(70, 100)

            # if c == 0:
            #     ax.set_ylim(40, 100)
            #     ax.set_yticks(range(40, 101, 10))
            # else:
            #     ax.set_ylim(80, 100)
            #     ax.set_yticks(range(80, 101, 5))

            ax.set_title(f"{ds}: $\\varepsilon={eps}, \\delta={10**(-5)}$")

    if priority_order is not None:
        assert len(legend_labels) == len(priority_order)

        legend_dict = dict(zip(legend_labels, legend_handles))
        legend_labels = priority_order
        legend_handles = [legend_dict[label] for label in priority_order]

    plt.axvline(x=0, linestyle="--", color="k", alpha=0.4, linewidth=1)

    # Draw the vertical arrow and label
    ax.annotate(
        None,  # label
        xy=(-20, 10),  # arrow head (down)
        xytext=(-20, 85),  # arrow tail (up)
        arrowprops=dict(
            arrowstyle="->",  # filled arrow head
            linestyle="--",
            alpha=0.7,  # a bit dim
            color="blue",
            linewidth=1,
        ),
    )

    # Draw the vertical arrow and label
    ax.annotate(
        "Accuracy drop",
        xy=(-180, 50),
        xytext=(-180, 50),  # arrow tail (up)
        # arrowprops=dict(
        #     arrowstyle="->",  # filled arrow head
        #     linestyle="-",
        #     alpha=0.7,  # a bit dim
        #     color="black",
        #     linewidth=1,
        # ),
    )

    ax.annotate(
        "Our method",  # label
        xy=(100, 85),  # arrow head (down)
        xytext=(200, 80),  # arrow tail (up)
        arrowprops=dict(
            arrowstyle="->",  # filled arrow head
            linestyle="-",
            alpha=0.7,  # a bit dim
            color="black",
            linewidth=1,
        ),
    )

    # Draw the vertical arrow and label
    ax.annotate(
        "Noisy Fine-Tuning",  # label
        xy=(700, 22),  # arrow head (down)
        xytext=(720, 30),  # arrow tail (up)
        arrowprops=dict(
            arrowstyle="->",  # filled arrow head
            linestyle="-",
            alpha=0.7,  # a bit dim
            color="black",
            linewidth=1,
        ),
    )

    # Draw the vertical arrow and label
    ax.annotate(
        "Model training\n before \nunlearning request",  # label
        xy=(-150, 92.5),  # arrow head (down)
        xytext=(-150, 80),  # arrow tail (up)
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",  # filled arrow head
            linestyle="-",
            alpha=0.7,  # a bit dim
            color="black",
            linewidth=1,
        ),
    )

    # Draw the vertical arrow and label
    ax.annotate(
        "Retrained model\nafter 182 epochs",  # label
        xy=(400, 94.5),  # arrow head (down)
        xytext=(600, 80),  # arrow tail (up)
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",  # filled arrow head
            linestyle="-",
            alpha=0.7,  # a bit dim
            color="black",
            linewidth=1,
        ),
    )

    # Draw the vertical arrow and label
    ax.annotate(
        "Retrain-from-scratch",  # label
        xy=(630, 45),  # arrow head (down)
        xytext=(700, 60),  # arrow tail (up)
        arrowprops=dict(
            arrowstyle="->",  # filled arrow head
            linestyle="-",
            alpha=0.7,  # a bit dim
            color="black",
            linewidth=1,
        ),
    )

    ax.axvspan(
        -300,
        0,
        ymin=0,
        ymax=1,  # span full height of the axes
        facecolor="0.5",
        alpha=0.2,  # gray + transparency
        zorder=0,
        linewidth=0,
    )  # draw behind lines, no border

    # shared legend below all panels
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=2,  # min(5, len(legend_labels)),
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])  # leave space for the legend
    plt.savefig(f"{plotfilename}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
