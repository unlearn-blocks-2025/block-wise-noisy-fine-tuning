from torch.utils.data import DataLoader
from pathlib import Path
from src.training.training_base import train_base_model
from src.training.training_unlearning import train_base_unlearning_model
from src.training.training_block_unlearning import train_block_unlearning_model
from src.data.dataset import create_test_dataloader, create_train_dataset
from src.utils.visualize_plots import visualize
from src.models.get_model import get_resnet
from src.utils.calculate_accuracy import evaluate_and_save_metrics
from src.utils.args import get_args
from src.models.get_model import BigMNISTMLP, get_resnet

args = get_args()

model_type = get_resnet if args.dataset == "cifar10" else BigMNISTMLP

forget_dataset, retain_dataset, train_dataset = create_train_dataset(
    dataset=args.dataset, forget_percent=10
)

test_loader, test_dataset = create_test_dataloader(dataset=args.dataset, batchsize=64)

retain_loader = DataLoader(retain_dataset, batch_size=60, shuffle=True)
forget_loader = DataLoader(forget_dataset, batch_size=6, shuffle=True)
full_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print(f"---- datasets are ready -----")

if args.training:
    for modelname in args.modelnames:
        print(f"======= Experiment for {modelname} =======")

        for experiment_tag in args.exps:
            print(f"-------experiment {experiment_tag} out of {len(args.exps)}--------")
            if args.mode == "train_base":
                model, _ = train_base_model(
                    full_loader,
                    test_loader,
                    lr=0.01,
                    momentum=0.9,
                    weight_decay=5e-4,
                    num_epochs=10,
                    log_every=100,
                    modelname=f"{args.experiment_type}/{modelname}",
                    model_type=model_type,
                )
            elif args.mode == "retrain":
                model, _ = train_base_model(
                    retain_loader,  # full or retain loader
                    test_loader,
                    lr=0.01,
                    momentum=0.9,
                    weight_decay=1e-5,
                    num_epochs=10,
                    log_every=100,
                    modelname=f"{args.experiment_type}/{modelname}",
                    model_type=model_type,
                )
            elif args.mode == "unlearning_base":
                model, _ = train_base_unlearning_model(
                    trained_full_state_path=args.trained_full_state_path,
                    retain_loader=retain_loader,
                    test_loader=test_loader,
                    config_path=f"experiments/{args.experiment_type}/{modelname}.yaml",
                    modelname=f"{args.experiment_type}/{modelname}_EXP{experiment_tag}",
                    log_every=100,
                    max_T=1000,
                    save_model_weights=True,
                    model_class=model_type,
                )
            elif args.mode == "blocks":
                model, _ = train_block_unlearning_model(
                    trained_full_state_path=args.trained_full_state_path,
                    retain_loader=retain_loader,
                    test_loader=test_loader,
                    config_path=f"experiments/{args.experiment_type}/{modelname}.yaml",
                    modelname=f"{args.experiment_type}/{modelname}_EXP{experiment_tag}",
                    log_every=20,
                    max_T=300,
                    model_type=model_type,
                    blocks_mode=args.blocks_mode,
                    save_model_weights=True,
                    blocks_split_type=args.blocks_split_type,
                )

            print("---- processs is finished -------")

            if args.evaluation:
                evaluate_and_save_metrics(
                    model,
                    retain_loader,
                    forget_loader,
                    test_loader,
                    f"{args.experiment_type}/{modelname}_EXP{experiment_tag}",
                )

print("---- evaluation is finished -------")

if args.do_visualization:
    folder = Path(
        f"experiments_results/accuracy/{args.experiment_type}"
    )  # {experiment_type}")
    paths = [str(path) for path in list(folder.glob("*.pt"))]
    print(paths)

    visualize(
        paths,
        tag=args.tag,
        T_max=args.T_max,
        figsize=(10, 10),
        one_color=args.one_color,
        fix_budget=args.fix_budget,
    )
