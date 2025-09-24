import yaml
import numpy as np
import torch
import numpy as np
from scipy.optimize import minimize_scalar


def load_config_yaml(filepath: str):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return config


def compute_dp_budget(epsilon, q, delta):
    return epsilon + np.log(1 / delta) / (q - 1)


def compute_sigma(c_0, c_1, eta_t, lambda_t, epslion, q, compute_error=1e-3):
    B = (2 * epslion / q) ** (0.5)  # WAS FIXED
    t = lambda_t * c_0 / c_1
    sigma = 1 / B * (eta_t * (2 - eta_t * lambda_t) * c_0 * c_1 * (2 - t)) ** 0.5

    # sigma_prev = (
    #     (2 * c_1) / B * ((eta_t / lambda_t) ** 0.5) * (2 - eta_t * lambda_t) ** 0.5
    # )
    return sigma


def find_optimal_budget_parameters(delta, C):

    a = np.log(1 / delta)

    def f(q):
        if q <= 1 + 1e-9:  # q > 1
            return np.inf
        eps = C - a / (q - 1)
        if eps <= 0:  # eps > 0
            return np.inf
        return q / eps

    res = minimize_scalar(f, bounds=(1 + 1e-6, 1e6), method="bounded")
    q_opt = res.x
    eps_opt = C - a / (q_opt - 1)
    return q_opt, eps_opt


def check_unlearning_condition(beta_0, beta_1, gamma):
    # Without this requirement we cannot achive unlearning
    return gamma + beta_1**2 - beta_0**2 >= 0


def stable_roots_from_params(beta0, beta1, gamma):
    """
    Compute roots of (beta1^2 + gamma) x^2 + 2 beta0 beta1 x + (beta0^2 - gamma) = 0
    in a numerically stable way.
    All computations in float64, with adaptive scaling.
    """
    # --- promote to float64 ---
    beta0 = torch.as_tensor(beta0, dtype=torch.float64)
    beta1 = torch.as_tensor(beta1, dtype=torch.float64)
    gamma = torch.as_tensor(gamma, dtype=torch.float64)

    # --- adaptive scaling to O(1) ---
    s = torch.max(
        torch.tensor(1.0, dtype=torch.float64),
        torch.max(torch.abs(beta0), torch.max(torch.abs(beta1), torch.sqrt(gamma))),
    )
    b0 = beta0 / s
    b1 = beta1 / s
    g = gamma / (s * s)

    # Quadratic coefficients a x^2 + b x + c = 0
    a = b1 * b1 + g
    b = 2.0 * b0 * b1
    c = b0 * b0 - g

    # Discriminant in a stable form: Δ = 4 γ (β1^2 - β0^2 + γ) → with scaled vars
    # Note Δ_scaled = 4 g (b1^2 - b0^2 + g) equals (b*b - 4*a*c)
    disc = 4.0 * g * (b1 * b1 - b0 * b0 + g)

    # Clip tiny negative due to roundoff
    eps = torch.finfo(torch.float64).eps
    disc = torch.clamp(disc, min=-10 * eps)  # allow tiny negative
    if disc < 1e-10:
        disc = torch.tensor(0.0, dtype=torch.float64)

    assert disc >= 0, f"discriminant equals to {disc} < 0"

    sqrt_disc = torch.sqrt(disc)

    # Kahan's stable quadratic formula
    sign_b = torch.where(
        b >= 0,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(-1.0, dtype=torch.float64),
    )
    q = -0.5 * (b + sign_b * sqrt_disc)

    # Handle a == 0 (degenerate) or q == 0 cases robustly
    # a > 0 always here because g > 0, but we guard anyway
    if torch.isclose(a, torch.tensor(0.0, dtype=torch.float64)):
        # Linear fallback: b x + c = 0
        x1 = -c / b
        x2 = x1
    else:
        x1 = q / a
        if torch.isclose(q, torch.tensor(0.0, dtype=torch.float64)):
            # Then use the symmetric formula for the "other" root
            x2 = -b / a - x1
        else:
            x2 = c / q

    # Order roots: x_min <= x_max
    x_max = torch.maximum(x1, x2)
    return x_max.item()


def compute_number_epochs(
    c_0, c_1, eta_t, lambda_t, epsilon, q, sigma, M, custom_sigma=False
):
    # computes number of epoch we needed if we have noise level sigma (even if sigma is not C_0/C_1)

    print(f"custom_sigma: {custom_sigma}")
    if custom_sigma is True:
        print(custom_sigma)
        B = (2 * epsilon / q) ** (0.5)  # WAS FIXED
        c_b = B * (sigma)

        beta_0 = 2 * c_1 / (c_b * lambda_t)
        beta_1 = (2 / (lambda_t * c_b)) * ((c_0 * lambda_t) - c_1)
        gamma = 1 / (lambda_t * eta_t * (2 - lambda_t * eta_t))

        # assert check_unlearning_condition(beta_0, beta_1, gamma), f"in 0 more than 0"
        x_optimal = stable_roots_from_params(beta_0, beta_1, gamma)

    else:
        # minimal sigma in the case where grad decent is dominating
        x_optimal = 1 - lambda_t * c_0 / c_1

    T_ft_optimal = np.log(x_optimal) / np.log(1 - eta_t * lambda_t)
    T_ft_optimal = (int(T_ft_optimal + M - 2) // (M - 1)) * (M - 1)
    T_optimal = T_ft_optimal * M // (M - 1)
    T_optimal = max(M, T_optimal)

    print(f"Optimal number of steps is {T_optimal} with noise {sigma}")
    return T_optimal


def calculate_noise_coefficient(eta_t, lambda_t, M):
    noise_coefficient = (
        (1 - (1 - eta_t * lambda_t) ** M) / (1 - (1 - eta_t * lambda_t) ** 2)
    ) ** 0.5
    return noise_coefficient


def calculate_parameters(config_path, block_unlearning=False):

    base_config = load_config_yaml(config_path)

    if not block_unlearning:
        configs = {}
        configs["base_block"] = base_config
    else:
        if "num_blocks" not in base_config[0]:
            configs = base_config
        else:
            configs = {}
            num_blocks = base_config[0]["num_blocks"]
            for i in range(num_blocks):
                configs[i] = base_config[0].copy()

    n_blocks = len(configs)

    q, epsilon = None, None

    for block in configs:

        config = configs[block]

        # calculate best q, eps for the fixed privacy budget
        if "budget" in config:
            if epsilon is None:
                q, epsilon = find_optimal_budget_parameters(
                    config["delta"], config["budget"]
                )

                epsilon /= n_blocks

            config["epsilon"] = epsilon
            config["q"] = q

            budget = config["budget"]
            print(f"eps: {epsilon} and q: {q} are calculated by budget {budget}")

        # you can ignore it
        if "M" not in config:
            # one step of noise + one step of finetuning
            config["M"] = 2

        # calculate sigma based on params
        if "sigma" not in config:
            config["sigma"] = compute_sigma(
                config["c_0"],
                config["c_1"],
                config["eta_t"],
                config["lambda_t"],
                config["epsilon"],
                config["q"],
            )
            custom_sigma = False
        else:
            custom_sigma = True

        # if M = 2, sigma_new = sigma
        config["sigma_new"] = (
            calculate_noise_coefficient(
                config["eta_t"], config["lambda_t"], config["M"]
            )
            * config["sigma"]
        )

        # calculate number of steps for unlearning
        config["T"] = compute_number_epochs(
            config["c_0"],
            config["c_1"],
            config["eta_t"],
            config["lambda_t"],
            config["epsilon"],
            config["q"],
            config["sigma"],
            config["M"],
            custom_sigma=custom_sigma,
        )

    if not block_unlearning:
        return configs["base_block"]
    else:
        return configs
