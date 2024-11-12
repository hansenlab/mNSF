import pickle
import random
from pathlib import Path
import click
import pandas as pd
from mNSF import process_multiSample, training_multiSample

def load_data(pth: Path, n_sample: int):
    X = [pd.read_csv(pth / f"X_sample{k}.csv") for k in range(1, n_sample + 1)]
    Y = [pd.read_csv(pth / f"Y_sample{k}.csv") for k in range(1, n_sample + 1)]
    D = [process_multiSample.get_D(x, y) for x, y in zip(X, Y)]
    X = [D[k]["X"] for k in range(0, n_sample)]
    return D, X

def run(
    data_dir: str | Path,
    output_dir: str | Path,
    n_loadings: int = 3,
    n_sample: int = 2,
    epochs: int = 10,
    legacy: bool = False,
):
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    # step 0  Data loading
    D, X = load_data(data_dir, n_sample)
    list_nchunk = [2, 2]
    listDtrain = process_multiSample.get_listDtrain(D, list_nchunk=list_nchunk)
    print("len(listDtrain)")
    print(len(listDtrain))
    
    list_D_chunked = process_multiSample.get_listD_chunked(D, list_nchunk=list_nchunk)
    for ksample in range(0, len(list_D_chunked)):
        random.seed(10)
        ninduced = round(list_D_chunked[ksample]["X"].shape[0] * 0.35)
        D_tmp = list_D_chunked[ksample]
        list_D_chunked[ksample]["Z"] = D_tmp["X"][random.sample(range(0, D_tmp["X"].shape[0] - 1), ninduced), :]
        
    print("len(list_D_chunked)")
    print(len(list_D_chunked))
    
    # step 1 initialize model
    fit = process_multiSample.ini_multiSample(list_D_chunked, n_loadings, "nb", chol=False)
    
    # step 2 fit model
    (pp := (output_dir / "models" / "pp")).mkdir(parents=True, exist_ok=True)
    fit = training_multiSample.train_model_mNSF(fit, pp, listDtrain, list_D_chunked, legacy=legacy, num_epochs=epochs)
    (output_dir / "list_fit_smallData.pkl").write_bytes(pickle.dumps(fit))
    
    # step 3 save results
    inpf12 = process_multiSample.interpret_npf_v3(fit, X, list_nchunk, S=2, lda_mode=False)
    (
        pd.DataFrame(
            inpf12["loadings"] * inpf12["totalsW"][:, None],
            columns=range(1, n_loadings + 1),
        ).to_csv(output_dir / "loadings_spde_smallData.csv")
    )
    factors = inpf12["factors"][:, :n_loadings]
    D, X = load_data(data_dir, n_sample)
    for k in range(n_sample):
        indices = process_multiSample.get_listSampleID(D)[k].astype(int)
        pd.DataFrame(factors[indices, :]).to_csv(output_dir / f"factors_sample{k + 1:02d}_smallData.csv")
    print("Done!")

@click.command()
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--n_loadings", "-L", type=int, default=1)
@click.option("--n_sample", "-n", type=int, default=1)
@click.option("--epochs", "-e", type=int, default=10)
@click.option("--legacy", "-l", is_flag=True)
def run_cli(
    data_dir: str | Path,
    output_dir: str | Path,
    n_loadings: int = 1,
    n_sample: int = 2,
    epochs: int = 10,
    legacy: bool = True,
):
    return run(data_dir, output_dir, n_loadings, n_sample, epochs, legacy)

def test_small_run():
    run("tests/data", ".", n_loadings=1, n_sample=2, legacy=True)

if __name__ == "__main__":
    run_cli()
