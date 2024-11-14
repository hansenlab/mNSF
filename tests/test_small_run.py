import pickle
import random
from pathlib import Path
import click
import pandas as pd
from mNSF import process_multiSample, training_multiSample

def load_data(pth: Path, n_sample: int):
    print(f"Loading data from {pth}")
    print(f"Number of samples: {n_sample}")
    
    # Check if files exist
    for k in range(1, n_sample + 1):
        x_file = pth / f"X_sample{k}.csv"
        y_file = pth / f"Y_sample{k}.csv"
    
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
    """
    Run the mNSF analysis pipeline
    """
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    print("Loading data from", data_dir)
    print("Number of samples:", n_sample)
    
    # Step 0: Data loading
    D, X = load_data(data_dir, n_sample)
    
    # Step 1: Prepare chunks
    list_nchunk = [2, 2]  # Two chunks per sample
    listDtrain = process_multiSample.get_listDtrain(D, list_nchunk=list_nchunk)
    list_D_chunked = process_multiSample.get_listD_chunked(D, list_nchunk=list_nchunk)
    
    # Step 2: Initialize model
    fit = process_multiSample.ini_multiSample(list_D_chunked, n_loadings, "nb", chol=False)
    
    # Step 3: Fit model
    pp_dir = output_dir / "models" / "pp"
    pp_dir.mkdir(parents=True, exist_ok=True)
    fit = training_multiSample.train_model_mNSF(fit, pp_dir, listDtrain, list_D_chunked, legacy=legacy, num_epochs=epochs)
    
    # Save fitted model
    (output_dir / "list_fit_smallData.pkl").write_bytes(pickle.dumps(fit))
    
    # Step 4: Process and save results
    inpf12 = process_multiSample.interpret_npf_v3(fit, X, list_nchunk, S=2, lda_mode=False)
    
    # Save loadings
    loadings_df = pd.DataFrame(
        inpf12["loadings"] * inpf12["totalsW"][:, None],
        columns=range(1, n_loadings + 1)
    )
    loadings_df.to_csv(output_dir / "loadings_spde_smallData.csv")
    
    # Save factors - with fix for index bounds
    factors = inpf12["factors"][:, :n_loadings]
    
    print("factors.shape")
    print(factors.shape)
    
    list_sampleID = process_multiSample.get_listSampleID(D)
    
    for k in range(n_sample):
        indices = list_sampleID[k]
        # Verify indices are within bounds
        if indices.max() >= factors.shape[0]:
            print(f"Warning: Sample {k+1} has indices exceeding factor matrix dimensions")
            print(f"Max index: {indices.max()}, Factors shape: {factors.shape}")
            indices = indices[indices < factors.shape[0]]
        
        if len(indices) > 0:
            pd.DataFrame(factors[indices, :]).to_csv(
                output_dir / f"factors_sample{k + 1:02d}_smallData.csv"
            )
    
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
    """Test the small run workflow"""
    run("tests/data", ".", n_loadings=1, n_sample=2, legacy=True)

if __name__ == "__main__":
    run_cli()
