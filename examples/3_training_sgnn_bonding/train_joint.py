#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from phyneo.utils import resolve_default_sgnn_specs


DEFAULT_STANDARD_MONOMERS = ["conf_03_DMC", "conf_09_EC"]
DEFAULT_ABN_MONOMERS = ["conf_18_DFP", "conf_20_PF6", "conf_15_BF4"]


class MolDataSet:
    def __init__(self, data):
        self.data = data
        self.n_data = len(self.data["positions"])

    def __getitem__(self, i):
        return [self.data["positions"][i], self.data["energies"][i]]

    def __len__(self):
        return self.n_data


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Jointly train standard and ABn sGNN models with separate parameters."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=[],
        help="Input dataset pickle. Can be passed multiple times.",
    )
    parser.add_argument(
        "--standard-monomer",
        action="append",
        dest="standard_monomers",
        default=[],
        help="Standard (nn=1, max_valence=4) training monomer key. Can be passed multiple times.",
    )
    parser.add_argument(
        "--abn-monomer",
        action="append",
        dest="abn_monomers",
        default=[],
        help="ABn (nn=0, max_valence=6) training monomer key. Can be passed multiple times.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--box-size", type=float, default=50.0)
    parser.add_argument(
        "--pdb-bank",
        type=Path,
        default=script_dir.parent.parent / "data" / "pdb_bank",
        help="Directory containing monomer PDB templates.",
    )
    parser.add_argument(
        "--standard-params-out",
        type=Path,
        default=script_dir / "params_sgnn_joint_standard.pickle",
        help="Path to save the best standard model parameters.",
    )
    parser.add_argument(
        "--abn-params-out",
        type=Path,
        default=script_dir / "params_sgnn_joint_abn.pickle",
        help="Path to save the best ABn model parameters.",
    )
    parser.add_argument(
        "--standard-restart",
        type=Path,
        default=None,
        help="Optional checkpoint to resume the standard model from.",
    )
    parser.add_argument(
        "--abn-restart",
        type=Path,
        default=None,
        help="Optional checkpoint to resume the ABn model from.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=script_dir / "nn_joint.err",
        help="Training log output.",
    )
    parser.add_argument(
        "--test-data-out",
        type=Path,
        default=script_dir / "test_data_joint.xvg",
        help="Validation scatter output.",
    )
    args = parser.parse_args()
    if not args.datasets:
        args.datasets = [
            script_dir.parent.parent / "data" / "data_sgnn_300k_remove_nb.pickle",
            script_dir / "dataset" / "data_300k_remove_nb.pickle",
            script_dir / "dataset" / "data_1000k_remove_nb.pickle",
            script_dir / "dataset" / "dataset300k_combine_remove_nb.pickle",
            script_dir / "dataset" / "dataset1000k_combine_remove_nb.pickle",
        ]
    else:
        args.datasets = [Path(path) for path in args.datasets]
    if not args.standard_monomers:
        args.standard_monomers = list(DEFAULT_STANDARD_MONOMERS)
    if not args.abn_monomers:
        args.abn_monomers = list(DEFAULT_ABN_MONOMERS)
    return args


def load_combined_datasets(dataset_paths: list[Path]) -> dict:
    datasets = {}
    loaded_paths = []
    for path in dataset_paths:
        if not path.exists():
            continue
        with open(path, "rb") as ifile:
            data = pickle.load(ifile)
        datasets.update(data)
        loaded_paths.append(path)
    if not loaded_paths:
        joined = ", ".join(str(path) for path in dataset_paths)
        raise FileNotFoundError(f"No dataset pickle found. Checked: {joined}")
    print("Loaded datasets:")
    for path in loaded_paths:
        print(f"  - {path}")
    return datasets


def split_dataset(data: dict, train_fraction: float) -> tuple[dict, dict]:
    train = {}
    test = {}
    for comp in ["positions", "energies"]:
        split_idx = int(train_fraction * len(data[comp]))
        train[comp] = data[comp][:split_idx]
        test[comp] = data[comp][split_idx:]
    return train, test


def make_train_batches(data: dict, batch_size: int):
    import jax.numpy as jnp

    dataset = MolDataSet(data)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    batches = []
    for pos, e in train_loader:
        batches.append((jnp.array(pos.numpy()), jnp.array(e.numpy())))
    return batches


def evaluate_dataset(cal_energy, data: dict, box, params):
    import jax.numpy as jnp

    ene_pred = cal_energy(data["positions"], box, params)
    ene_ref = data["energies"]
    centered_ref = ene_ref - jnp.average(ene_ref)
    centered_pred = ene_pred - jnp.average(ene_pred)
    loss = jnp.sqrt(jnp.average((centered_ref - centered_pred) ** 2))
    return centered_pred, centered_ref, loss


def main() -> int:
    args = parse_args()

    import jax
    import jax.numpy as jnp
    import optax
    from dmff.sgnn.gnn import MolGNNForce
    from dmff.sgnn.graph import from_pdb
    from jax import jit, value_and_grad
    from jax.lib import xla_bridge

    specs = resolve_default_sgnn_specs()

    print(jax.devices()[0])
    print(xla_bridge.get_backend().platform)

    box = jnp.eye(3) * args.box_size
    tot_data = load_combined_datasets(args.datasets)

    family_to_monomers = {
        "standard": args.standard_monomers,
        "abn": args.abn_monomers,
    }
    output_paths = {
        "standard": args.standard_params_out,
        "abn": args.abn_params_out,
    }
    restart_paths = {
        "standard": args.standard_restart,
        "abn": args.abn_restart,
    }

    trunk_train = []
    trunk_test = {}
    cal_energy = {}
    mse_loss_grad = {}
    params_by_family = {}
    optimizers = {}
    opt_states = {}

    for family, monomers in family_to_monomers.items():
        if not monomers:
            continue
        spec = specs[family]
        model = None
        trunk_test[family] = {}
        cal_energy[family] = {}
        mse_loss_grad[family] = {}

        for key in monomers:
            if key not in tot_data:
                raise KeyError(f"Dataset key not found: {key}")
            print(f"{family}: {key}")
            data_train, data_test = split_dataset(tot_data[key], args.train_fraction)
            trunk_test[family][key] = {
                "positions": jnp.array(data_test["positions"]),
                "energies": jnp.array(data_test["energies"]),
            }
            for pos, ene_ref in make_train_batches(data_train, args.batch_size):
                trunk_train.append((family, key, pos, ene_ref))

            pdb = args.pdb_bank / f"{key.split('_')[-1]}.pdb"
            if not pdb.exists():
                raise FileNotFoundError(f"PDB not found for {key}: {pdb}")

            G = from_pdb(str(pdb))
            model = MolGNNForce(
                G,
                nn=spec.nn,
                max_valence=spec.max_valence,
                n_layers=spec.n_layers,
                sizes=[tuple(layer_sizes) for layer_sizes in spec.sizes],
                sigma=spec.sigma,
                mu=spec.mu,
            )
            cal_energy[family][key] = jax.vmap(model.forward, in_axes=(0, None, None), out_axes=(0))

            def mse_loss(params, positions, box, ene_ref, family_=family, key_=key):
                ene = cal_energy[family_][key_](positions, box, params)
                err = ene - ene_ref
                err -= jnp.average(err)
                return jnp.average(err ** 2)

            mse_loss_grad[family][key] = jit(value_and_grad(mse_loss, argnums=0))

        if model is None:
            continue
        restart_path = restart_paths[family]
        if restart_path is not None:
            with open(restart_path, "rb") as f:
                params_by_family[family] = pickle.load(f)
        else:
            params_by_family[family] = model.params
        optimizers[family] = optax.adam(args.lr)
        opt_states[family] = optimizers[family].init(params_by_family[family])

    if not params_by_family:
        raise RuntimeError("No training families were initialized.")

    best_loss = jnp.array(1e30)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.test_data_out.parent.mkdir(parents=True, exist_ok=True)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.log_file, "w") as fout:
        fout.write("Joint sGNN training for standard and ABn intramolecular energy\n")
        fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
        fout.flush()

        for i_epoch in range(args.epochs):
            np.random.shuffle(trunk_train)
            lossprop = 0.0
            family_losses = {family: 0.0 for family in params_by_family}
            for family, key, pos, ene_ref in trunk_train:
                loss, gradients = mse_loss_grad[family][key](params_by_family[family], pos, box, ene_ref)
                family_losses[family] += loss
                lossprop += loss
                updates, opt_states[family] = optimizers[family].update(
                    gradients,
                    opt_states[family],
                )
                params_by_family[family] = optax.apply_updates(params_by_family[family], updates)

            lossprop = jnp.sqrt(lossprop)
            family_rmse = {
                family: jnp.sqrt(loss) if loss > 0 else jnp.array(0.0)
                for family, loss in family_losses.items()
            }
            print(f"epoch={i_epoch} total={lossprop} standard={family_rmse.get('standard')} abn={family_rmse.get('abn')}")

            if lossprop < best_loss:
                for family, path in output_paths.items():
                    if family in params_by_family:
                        with open(path, "wb") as f:
                            pickle.dump(params_by_family[family], f)
                best_loss = lossprop

                ene_refs = []
                ene_preds = []
                test_losses = {}
                for family, family_tests in trunk_test.items():
                    for key, data in family_tests.items():
                        ene_pred, ene_ref, loss_mol = evaluate_dataset(
                            cal_energy[family][key],
                            data,
                            box,
                            params_by_family[family],
                        )
                        print(f"{family}:{key} {loss_mol}")
                        test_losses[f"{family}:{key}"] = loss_mol
                        ene_preds.append(ene_pred)
                        ene_refs.append(ene_ref)

                ene_ref_tot = jnp.concatenate(ene_refs)
                ene_pred_tot = jnp.concatenate(ene_preds)
                test_loss = jnp.sqrt(jnp.average((ene_pred_tot - ene_ref_tot) ** 2))

                fout.write(
                    "{:5} {:4} {:15} {:5e} train {:10.5f} test {:10.5f} standard {:10.5f} abn {:10.5f}\n".format(
                        "Epoch=",
                        i_epoch,
                        "learning rate",
                        args.lr,
                        float(lossprop),
                        float(test_loss),
                        float(family_rmse.get("standard", 0.0)),
                        float(family_rmse.get("abn", 0.0)),
                    )
                )
                fout.flush()

                with open(args.test_data_out, "w") as f:
                    print(f"# RMSE = {float(test_loss):10.5f}", file=f)
                    for e_pred, e_ref in zip(np.array(ene_pred_tot), np.array(ene_ref_tot)):
                        print(e_ref, e_ref, e_pred, file=f)

        fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
        fout.write("terminated normal\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
