import pandas as pd
import numpy as np
import random
import os.path
import json
from sentence_transformers import SentenceTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
import pickle


def get_embeddings(df, model):
    ''' get list of Sentences from df; return df containing list of embeddings according to model '''
    sentences = df['Sentence'].tolist()
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)
    df['Embeddings'] = [emb.tolist() for emb in embeddings]
    return df


def do_PLS(df, m):
    ''' Given a dataframe with high-d embeddings from model m, add a column with PLS-reduced embeddings '''
    embeddings = np.array(df['Embeddings'].tolist())

    # if continuous labels present, use them; otherwise map discrete 'Target' to vectors
    if {'competence', 'warmth'}.issubset(df.columns):
        labels = df[['competence', 'warmth']].values.tolist()
    else:
        PLS_labels = {
            'warm':        [0,  1],
            'comp':        [1,  0],
            'cold':        [0, -1],
            'incomp':     [-1,  0],
            'warm_comp':   [1,  1],
            'warm_incomp':[-1,  1],
            'cold_comp':   [1, -1],
            'cold_incomp':[-1, -1]
        }
        labels = [PLS_labels[row['Target']] for _, row in df.iterrows()]

    # fit PLS model
    pls = PLSRegression(n_components=10, scale=True)
    pls.fit(embeddings, labels)

    # transform data
    PLS_embeddings = pls.transform(embeddings)
    df['PLS_embeddings'] = PLS_embeddings.tolist()

    # save PLS model
    filename = f'pls_model_{m}.sav'
    pickle.dump(pls, open(filename, 'wb'))
    return df


def do_PCA(df, m):
    ''' Given a dataframe with high-d embeddings from model m, add a column with PCA-reduced embeddings '''
    embeddings = np.array(df['Embeddings'].tolist())

    # fit PCA model
    pca = PCA(n_components=10)
    pca.fit(embeddings)

    # transform data
    pca_embeddings = pca.transform(embeddings)
    df['PCA_embeddings'] = pca_embeddings.tolist()

    # save PCA model
    filename = f'pca_model_{m}.sav'
    pickle.dump(pca, open(filename, 'wb'))
    return df


def get_centroid(df, label, use_PLS=True, use_PCA=False):
    ''' Given df and direction label (warm, cold, etc.), return centroid of vectors for that direction '''
    # handle continuous labels
    if {'competence', 'warmth'}.issubset(df.columns):
        if label == 'comp':
            mask = df['competence'] >= 0
        elif label == 'incomp':
            mask = df['competence'] < 0
        elif label == 'warm':
            mask = df['warmth'] >= 0
        elif label == 'cold':
            mask = df['warmth'] < 0
        else:
            # quadrant labels e.g. 'warm_comp'
            w, c = label.split('_')
            w_mask = df['warmth'] >= 0 if w == 'warm' else df['warmth'] < 0
            # for competence side of quadrant: treat 'comp' and 'warm_comp' and 'cold_comp' as >=0
            c_mask = df['competence'] >= 0 if c in ('comp', 'warm_comp', 'cold_comp') else df['competence'] < 0
            mask = w_mask & c_mask
        temp_df = df[mask]
    else:
        temp_df = df[df['Target'] == label]

    if temp_df.empty:
        raise ValueError(f"No data for label '{label}'")

    # select embeddings type
    if use_PLS and 'PLS_embeddings' in temp_df:
        vecs = np.array(temp_df['PLS_embeddings'].tolist())
    elif use_PCA and 'PCA_embeddings' in temp_df:
        vecs = np.array(temp_df['PCA_embeddings'].tolist())
    else:
        vecs = np.array(temp_df['Embeddings'].tolist())

    # return centroid
    return vecs.mean(axis=0)


def normalize(v):
    ''' Return normalized vector v  '''
    return v / np.linalg.norm(v)


def compute_rotation_matrix(df, model_name, polar_model='original', PLS=False, PCA=False):
    ''' Compute the rotation matrix. '''
    if polar_model == 'original':
        comp_ctr = normalize(get_centroid(df, 'comp', use_PLS=PLS, use_PCA=PCA))
        incomp_ctr = normalize(get_centroid(df, 'incomp', use_PLS=PLS, use_PCA=PCA))
        warm_ctr = normalize(get_centroid(df, 'warm', use_PLS=PLS, use_PCA=PCA))
        cold_ctr = normalize(get_centroid(df, 'cold', use_PLS=PLS, use_PCA=PCA))
        dir_comp = normalize(np.array([comp_ctr - incomp_ctr]))
        dir_warm = normalize(np.array([warm_ctr - cold_ctr]))
        dir = np.vstack((dir_comp, dir_warm))
        dir_T_inv = np.linalg.pinv(dir)
    elif polar_model == 'axis_rotated':
        wc = normalize(get_centroid(df, 'warm_comp',   use_PLS=PLS, use_PCA=PCA))
        cc = normalize(get_centroid(df, 'cold_comp',   use_PLS=PLS, use_PCA=PCA))
        wi = normalize(get_centroid(df, 'warm_incomp', use_PLS=PLS, use_PCA=PCA))
        ci = normalize(get_centroid(df, 'cold_incomp', use_PLS=PLS, use_PCA=PCA))
        dir1 = normalize(np.array([wc - ci]))
        dir2 = normalize(np.array([wi - cc]))
        dir = np.vstack((dir1, dir2))
        dir_T_inv = np.linalg.pinv(dir)
    else:
        raise ValueError("polar_model must be 'original' or 'axis_rotated'")

    # save matrix
    suffix = 'none'
    if PLS:
        suffix = 'PLS'
    elif PCA:
        suffix = 'PCA'
    np.save(f'rotation_{polar_model}_{suffix}_{model_name}.npy', dir_T_inv)


if __name__ == "__main__":
    # reproducibility
    rseed = 1
    np.random.seed(rseed)
    random.seed(rseed)

    # model
    model_name = 'roberta-large-nli-mean-tokens'
    model = SentenceTransformer(model_name)

    # load training data (modify path as needed)
    #train_df = pd.read_csv('/content/computational-SCM/data/BWS_annotations_modified1.csv')
    train_df = pd.read_csv('/content/computational-SCM/data/cross_validation/training_two_adjectives2.csv')
    # embeddings
    train_df = get_embeddings(train_df, model)

    # PLS (uses continuous or discrete labels)
    train_df = do_PLS(train_df, model_name)

    # optional PCA
    # train_df = do_PCA(train_df, model_name)

    # compute rotation matrix
    compute_rotation_matrix(train_df, model_name, polar_model='axis_rotated', PLS=True, PCA=False)
