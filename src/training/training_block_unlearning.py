import torch
import torch.nn as nn
from itertools import cycle
from src.unlearning.compute_parameters import calculate_parameters
from src.training.training_base import choose_device
from src.utils.calculate_accuracy import evaluate
from src.utils.utils import save_accuracy_history, save_model
from src.training.training_unlearning import clip_gradients_l2
from src.unlearning.model_with_blocks import (
    freeze_all_except_Bi,
    wrap_model,
)
from src.unlearning.restore_model_architecture import antiwrap_model
from src.training.training_unlearning import finetune_model


def unlearn_one_block(
    model,
    block_idx,
    config,
    infinite_loader,
    device,
    criterion,
    test_loader,
    log_every,
    test_acc_per_iteration,
    current_step,
):

    print(f"---- Unlearning Block {block_idx} ------")

    # freeze everything exept one group
    freeze_all_except_Bi(model, block_idx)

    for step in range(config["T"]):
        model.zero_grad()

        images, labels = next(infinite_loader)
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Calculte params and grads
        params = [p for p in model.parameters() if p.requires_grad]
        grads = [p.grad for p in params]

        with torch.no_grad():
            # Every M-th step is noise step
            if (step + 1) % config["M"] == 0:
                print("noise step")
                for p in params:  # trainable
                    p.add_(torch.randn_like(p) * config["sigma_new"])
            # All other steps are finetuning steps
            else:
                # Clip gradients
                clipped_grads = clip_gradients_l2(grads, config["c_1"])

                for p, g in zip(params, clipped_grads):  # trainable
                    # Weight decay
                    p.mul_(1 - config["eta_t"] * config["lambda_t"])
                    # Gradient step
                    p.add_(g, alpha=-config["eta_t"])

        if step < config["T"] or step % log_every == 0:
            acc = evaluate(model, test_loader, device=device)
            test_acc_per_iteration.append((step + current_step, acc))

            T = config["T"]
            print(
                f"[Inner Step {step}/{T}] Loss: {loss.item():.4f} - Test Acc: {acc:.2f}%"
            )

    return model, test_acc_per_iteration


def train_block_unlearning_model(
    trained_full_state_path,
    retain_loader,
    test_loader,
    config_path,
    modelname,
    log_every=500,
    max_T=5000,
    model_type=LinearNet,
    save_model_weights=True,
    blocks_mode="qr",
    blocks_split_type="equal",
):

    configs = calculate_parameters(config_path, block_unlearning=True)

    n_blocks = len(configs)

    # Define the model again
    device = choose_device()

    # load base model
    trained_full_state = torch.load(trained_full_state_path, map_location=device)
    old_model = model_type(num_classes=10).to(device)
    old_model.load_state_dict(trained_full_state)

    acc = evaluate(old_model, test_loader, device=device)
    print(f"before wrapping: {acc}")

    model = wrap_model(
        old_model,
        n_blocks=n_blocks,
        device=device,
        blocks_mode=blocks_mode,
        blocks_split_type=blocks_split_type,
    )

    # DataLoader over retain dataset
    infinite_loader = cycle(retain_loader)  # Create infinite stream of batches

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Training loop: T iterations
    model.train()
    test_acc_per_iteration = []
    step = 0

    for block_idx in configs:

        model, test_acc_per_iteration = unlearn_one_block(
            model,
            block_idx,
            configs[block_idx],
            infinite_loader,
            device,
            criterion,
            test_loader,
            log_every,
            test_acc_per_iteration,
            current_step=step,
        )

        step += configs[block_idx]["T"]

    model = finetune_model(
        model,
        infinite_loader,
        criterion,
        device,
        max_steps=max_T,
        step=step,
        test_acc_per_iteration=test_acc_per_iteration,
        log_every=log_every,
        test_loader=test_loader,
        lr=float(configs[0]["finetuning_lr"]),
        weight_decay=float(configs[0]["finetuning_weight_decay"]),
        momentum=float(configs[0]["finetuning_momentum"]),
    )

    accuracy_filepath = f"experiments_results/accuracy/{modelname}.pt"
    save_accuracy_history(test_acc_per_iteration, filepath=accuracy_filepath)

    template_model = model_type(num_classes=10).to(device)
    new_model_in_old_format = antiwrap_model(model, template_model)

    if save_model_weights:
        save_model(
            model,
            filepath=f"experiments_results/models/{modelname}.pt",
        )
        save_model(
            new_model_in_old_format,
            filepath=f"experiments_results/models_restored/{modelname}.pt",
        )

    return new_model_in_old_format, test_acc_per_iteration
