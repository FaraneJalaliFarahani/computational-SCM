import argparse
import os
import json
import random
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

from train import normalize

def get_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    """Compute and append sentence embeddings to DataFrame."""
    sentences = df['Sentence'].tolist()
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)
    df['Embeddings'] = [vec.tolist() for vec in embeddings]
    return df


def reduce_embeddings(
    df: pd.DataFrame,
    method: str,
    model_name: str,
    n_components: int = 10
) -> pd.DataFrame:
    """Apply PLS or PCA reduction, save model, and append reduced embeddings."""
    arr = np.vstack(df['Embeddings'].tolist())
    if method == 'PLS':
        labels_map = {
            'warm': [0,1], 'comp': [1,0], 'cold':[0,-1], 'incomp':[-1,0],
            'warm_comp':[1,1], 'warm_incomp':[-1,1], 'cold_comp':[1,-1], 'cold_incomp':[-1,-1]
        }
        labels = np.array([labels_map[label] for label in df['Target']])
        model = PLSRegression(n_components=n_components, scale=True)
    else:  # PCA
        labels = None
        model = PCA(n_components=n_components)

    model.fit(arr, labels) if labels is not None else model.fit(arr)
    reduced = model.transform(arr)
    df[f'{method}_embeddings'] = [vec.tolist() for vec in reduced]

    with open(f'{method.lower()}_model_{model_name}.sav', 'wb') as f:
        pickle.dump(model, f)

    return df


def compute_centroid(
    df: pd.DataFrame,
    label: str,
    method: str
) -> np.ndarray:
    """Return centroid of embeddings for a given label and method."""
    col = f'{method}_embeddings' if method in ('PLS','PCA') else 'Embeddings'
    subset = df[df['Target'] == label]
    if subset.empty:
        raise ValueError(f"No samples for label '{label}'")
    arr = np.vstack(subset[col].tolist())
    return np.mean(arr, axis=0)


def compute_rotation_matrix(
    df: pd.DataFrame,
    model_name: str,
    polar_model: str,
    method: str
) -> None:
    """Compute and save rotation matrix for warmth-competence axes."""
    use_method = method in ('PLS','PCA')
    if polar_model == 'original':
        comp_cent = normalize(compute_centroid(df, 'comp', method if use_method else 'orig'))
        incomp_cent = normalize(compute_centroid(df, 'incomp', method if use_method else 'orig'))
        warm_cent = normalize(compute_centroid(df, 'warm', method if use_method else 'orig'))
        cold_cent = normalize(compute_centroid(df, 'cold', method if use_method else 'orig'))
        dirs = np.vstack([comp_cent - incomp_cent, warm_cent - cold_cent])
    elif polar_model == 'axis_rotated':
        wc = normalize(compute_centroid(df, 'warm_comp', method if use_method else 'orig'))
        cc = normalize(compute_centroid(df, 'cold_comp', method if use_method else 'orig'))
        wi = normalize(compute_centroid(df, 'warm_incomp', method if use_method else 'orig'))
        ci = normalize(compute_centroid(df, 'cold_incomp', method if use_method else 'orig'))
        dirs = np.vstack([wc - ci, wi - cc])
    else:
        raise ValueError("polar_model must be 'original' or 'axis_rotated'")

    rot = np.linalg.pinv(dirs)
    suffix = method if use_method else 'none'
    np.save(f'rotation_{polar_model}_{suffix}_{model_name}.npy', rot)


def main():
    parser = argparse.ArgumentParser(
        description="Train and save rotation matrices from sentence embeddings"
    )
    parser.add_argument('--input-dir', default='data', help='Folder with CSVs')
    parser.add_argument('--files', nargs='+', required=True,
                        help='Base filenames (without .csv) to process')
    parser.add_argument('--model-name', default='roberta-large-nli-mean-tokens',
                        help='SentenceTransformer model name')
    parser.add_argument('--method', choices=['orig','PLS','PCA'], default='PLS',
                        help='Dimension reduction method')
    parser.add_argument('--polar-model', choices=['original','axis_rotated'], default='axis_rotated',
                        help='Rotation scheme')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--n-components', type=int, default=10,
                        help='Number of components for PLS/PCA')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs('embeddings', exist_ok=True)

    model = SentenceTransformer(args.model_name)
    all_df = []
    for name in args.files:
        path = os.path.join(args.input_dir, f'{name}.csv')
        df = pd.read_csv(path)
        emb_csv = os.path.join('embeddings', f'{name}_{args.model_name}.csv')
        if not os.path.isfile(emb_csv):
            df = get_embeddings(df, model)
            df.to_csv(emb_csv, index=False)
        else:
            df = pd.read_csv(emb_csv)
            df['Embeddings'] = df['Embeddings'].apply(json.loads)
        all_df.append(df)

    combined = pd.concat(all_df, ignore_index=True)

    if args.method in ('PLS','PCA'):
        combined = reduce_embeddings(combined, args.method, args.model_name, args.n_components)

    compute_rotation_matrix(
        combined,
        args.model_name,
        args.polar_model,
        args.method
    )
    print("Training complete.")

if __name__ == '__main__':
    main()
