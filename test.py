import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from sentence_transformers import SentenceTransformer
from train import get_embeddings, normalize


def rotate_data(arr: np.ndarray) -> np.ndarray:
    """Rotate 2D array by 45 degrees around Z-axis"""
    r = R.from_euler('z', 45, degrees=True)
    # Extend to 3D, rotate, then drop Z
    arr3d = np.hstack([arr, np.zeros((arr.shape[0], 1))])
    rotated = r.apply(arr3d)
    return rotated[:, :2]


def load_embeddings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Embeddings'] = df['Embeddings'].apply(json.loads)
    return df


def ensure_embeddings(df: pd.DataFrame, model: SentenceTransformer, out_path: str) -> pd.DataFrame:
    if not os.path.isfile(out_path):
        df = get_embeddings(df, model)
        df.to_csv(out_path, index=False)
    else:
        print(f"Embeddings file exists: {out_path}")
    return load_embeddings(out_path)


def compute_warmth_competence(
    df: pd.DataFrame,
    model_name: str,
    polar_model: str = 'original',
    use_pls: bool = False,
    use_pca: bool = False
) -> pd.DataFrame:
    # Load appropriate dimensionality reduction
    if use_pls:
        pls = pickle.load(open(f'pls_model_{model_name}.sav', 'rb'))
        reduced = pls.transform(np.vstack(df['Embeddings'].values))
        embeddings = [normalize(vec) for vec in reduced]
        rot_mat = np.load(f'rotation_{polar_model}_PLS_{model_name}.npy')
    elif use_pca:
        pca = pickle.load(open(f'pca_model_{model_name}.sav', 'rb'))
        reduced = pca.transform(np.vstack(df['Embeddings'].values))
        embeddings = [normalize(vec) for vec in reduced]
        rot_mat = np.load(f'rotation_{polar_model}_PCA_{model_name}.npy')
    else:
        embeddings = df['Embeddings'].tolist()
        rot_mat = np.load(f'rotation_{polar_model}_none_{model_name}.npy')

    print("Computing warmth and competence...")
    proj = np.dot(np.vstack(embeddings), rot_mat)
    if polar_model != 'original':
        proj = rotate_data(proj)

    df['Competence'] = proj[:, 0]
    df['Warmth'] = proj[:, 1]
    return df


def compute_accuracy(df: pd.DataFrame) -> float:
    if 'Target' not in df.columns:
        print("No target labels, skipping accuracy.")
        return 0.0

    use_1d = all('_' not in lbl for lbl in df['Target'].unique())
    correct = []
    for _, row in df.iterrows():
        c, w = row['Competence'], row['Warmth']
        label = row['Target']
        if use_1d:
            cond = (
                (label == 'warm' and w >= 0) or
                (label == 'cold' and w < 0) or
                (label == 'comp' and c >= 0) or
                (label == 'incomp' and c < 0)
            )
        else:
            cond = (
                (label == 'warm_comp' and w >= 0 and c >= 0) or
                (label == 'cold_comp' and w < 0 and c >= 0) or
                (label == 'warm_incomp' and w >= 0 and c < 0) or
                (label == 'cold_incomp' and w < 0 and c < 0)
            )
        correct.append(int(cond))
    return float(np.mean(correct))


def main():
    parser = argparse.ArgumentParser(
        description="Compute warmth and competence from sentence embeddings"
    )
    parser.add_argument("--input-dir", default="data", help="Directory for input CSV")
    parser.add_argument("--input-file", required=True, help="Base filename (without extension)")
    parser.add_argument("--embedding-dir", default="embeddings", help="Directory for embeddings CSV")
    parser.add_argument("--output-dir", default="output", help="Directory for output CSV")
    parser.add_argument(
        "--model-name", 
        default="roberta-large-nli-mean-tokens",
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--polar-model", 
        choices=["original", "axis_rotated"], 
        default="original",
        help="Rotation scheme for warmth-competence axes"
    )
    parser.add_argument("--pls", action="store_true", help="Use PLS reduction")
    parser.add_argument("--pca", action="store_true", help="Use PCA reduction")
    parser.add_argument("--compute-acc", action="store_true", help="Compute accuracy if targets provided")

    args = parser.parse_args()

    os.makedirs(args.embedding_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = SentenceTransformer(args.model_name)

    input_csv = os.path.join(args.input_dir, args.input_file + ".csv")
    embed_csv = os.path.join(
        args.embedding_dir,
        f"{args.input_file}_{args.model_name}.csv"
    )

    df_raw = pd.read_csv(input_csv)
    df = ensure_embeddings(df_raw, model, embed_csv)

    df = compute_warmth_competence(
        df,
        args.model_name,
        polar_model=args.polar_model,
        use_pls=args.pls,
        use_pca=args.pca
    )

    output_csv = os.path.join(
        args.output_dir,
        f"{args.input_file}_{args.model_name}.csv"
    )
    df.to_csv(output_csv, index=False)
    print(f"Result saved to {output_csv}")

    if args.compute_acc:
        acc = compute_accuracy(df)
        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
