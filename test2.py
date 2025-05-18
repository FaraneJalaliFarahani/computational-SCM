import pandas as pd
import numpy as np
import random
import os
import os.path
import json
from sentence_transformers import SentenceTransformer
from sklearn.cross_decomposition import PLSRegression
import pickle
from scipy.spatial.transform import Rotation as R

from train import get_embeddings, normalize
from test import compute_warmth_competence, compute_1d_accuracy, compute_2d_accuracy, compute_accuracy

def rotate_data(arr):
    ''' Rotate array by 45 degrees  '''
    r = R.from_euler('z', 45, degrees=True)
    arr = [list(a) + [0] for a in arr]
    arr = r.apply(np.array(arr))
    return np.delete(arr, 2, 1)

if __name__ == "__main__":

    # sentence embedding model
    model_name = 'roberta-large-nli-mean-tokens'
    model = SentenceTransformer(model_name)

    # test file should contain sentences and, optionally, labels
    test_dir      = 'data'
    test_filename = 'training_all_two_adjectives2'  # e.g. 'mydata'

    # ensure output folders exist
    os.makedirs('embeddings', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # 1) Embed if needed
    emb_path = f'embeddings/{test_filename}_{model_name}.csv'
    if not os.path.isfile(emb_path):
        df = pd.read_csv(f'{test_dir}/{test_filename}.csv')
        df = get_embeddings(df, model)
        df.to_csv(emb_path, index=False)
    else:
        df = pd.read_csv(emb_path)
        df['Embeddings'] = df['Embeddings'].apply(json.loads)

    # 2) Compute warmth & competence
    df = compute_warmth_competence(
        df,
        model_name,
        polar_model='axis_rotated',
        PLS=True,
        PCA=False
    )

    # 3) Derive Predicted_Label per example
    def derive_label(row):
        if 'Target' in row and '_' in str(row['Target']):
            for lbl in ['warm_comp','cold_comp','warm_incomp','cold_incomp']:
                if compute_2d_accuracy(lbl, row['Competence'], row['Warmth']) == 1:
                    return lbl
        else:
            for lbl in ['warm','cold','comp','incomp']:
                if compute_1d_accuracy(lbl, row['Competence'], row['Warmth']) == 1:
                    return lbl
        return None

    df['Predicted_Label'] = df.apply(derive_label, axis=1)

    # 4) Save out: input sentence, true label, predicted label, scores
    # Detect text column automatically
    text_cols = [c for c in df.columns
                 if df[c].dtype == object
                 and c not in ('Embeddings','Target','Predicted_Label')]
    out_cols = text_cols + ['Target', 'Predicted_Label', 'Competence', 'Warmth']

    out_path = f'output/{test_filename}_{model_name}_predictions.csv'
    df[out_cols].to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    # 5) Optional: compute & print accuracy
    acc = compute_accuracy(df)
    print('ACCURACY:', acc)
