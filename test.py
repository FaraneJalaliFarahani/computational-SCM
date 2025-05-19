import pandas as pd
import numpy as np
import random
import os.path
import json
from sentence_transformers import SentenceTransformer
from sklearn.cross_decomposition import PLSRegression
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.stats import spearmanr 
from train import get_embeddings, normalize

    
def rotate_data(arr):

    ''' Rotate array by 45 degrees  '''

    r = R.from_euler('z', 45, degrees=True)
    arr = list(arr)

    for i in range(len(arr)):
        arr[i] = list(arr[i])
        arr[i].append(0) # add z-coordinate
        
    arr = np.array(arr)
    arr = r.apply(arr)
    
    arr = np.delete(arr, 2, 1) #remove z-coordinate

    return arr
    


def save_predictions(df, filename_prefix, model_name, output_dir='output'):
    ''' Save full results and prediction scores (inputs, true labels, warmth & competence) '''
    os.makedirs(output_dir, exist_ok=True)
    # full output
    full_path = os.path.join(output_dir, f'{filename_prefix}_{model_name}.csv')
    print('Saving full output to', full_path)
    df.to_csv(full_path, index=False)

    # just the predictions
    if 'Target' in df.columns:
        preds = df[['Sentence', 'Target', 'Competence', 'Warmth']]
        preds_path = os.path.join(output_dir, f'predictions_{filename_prefix}_{model_name}.csv')
        print('Saving prediction scores with true labels to', preds_path)
        preds.to_csv(preds_path, index=False)


def compute_warmth_competence(df, model_name, polar_model='original', PLS=False, PCA=False):
    ''' Given df and arguments, compute the warmth and competence values '''

    if PLS:

        # get saved PLS model
        print('Loading PLS model ...')
        pls = pickle.load(open('pls_model_' + model_name + '.sav', 'rb'))

        # do PLS dimensionality reduction 
        print('Doing PLS dimensionality reduction ...')
        PLS_embeddings = pls.transform(np.array(df['Embeddings'].tolist())) 
        embeddings = [normalize(s) for s in PLS_embeddings]
        
        dir_T_inv = np.load('rotation_' + polar_model + '_PLS_' + model_name + '.npy')

        
    elif PCA:
    
        # get saved PCA model
        print('Loading PCA model ...')
        pca = pickle.load(open('pca_model_' + model_name + '.sav', 'rb'))

        # do PCA dimensionality reduction 
        print('Doing PCA dimensionality reduction ...')
        PCA_embeddings = pca.transform(np.array(df['Embeddings'].tolist())) 
        embeddings = [normalize(s) for s in PCA_embeddings]
        
        dir_T_inv = np.load('rotation_' + polar_model + '_PCA_' + model_name + '.npy')
        
    else:
    
        embeddings = df['Embeddings'].tolist()
        dir_T_inv = np.load('rotation_' + polar_model + '_none_' + model_name + '.npy')



    # project to 2D warmth-competence plane (with rotation for axis-rotated POLAR)
    print('Computing warmth and competence ...')
    if polar_model == 'original':
        SCM_embeddings = np.array(np.matmul(embeddings, dir_T_inv))
    else:
        SCM_embeddings = rotate_data(np.array(np.matmul(embeddings, dir_T_inv)))

    # make warmth and competence columns 
    df['Competence'] = SCM_embeddings[:,0].tolist()
    df['Warmth'] = SCM_embeddings[:,1].tolist()
    
    return df


def compute_1d_accuracy(label, c, w):
    ''' Given gold label, competence value c, and warmth value w, return 1 if hypothesis matches gold and 0 otherwise. '''

    result = 0

    if label == 'warm' and w >= 0:
        result = 1
    elif label == 'cold' and w < 0:
        result = 1
    elif label ==  'comp' and c >= 0:
        result = 1
    elif label == 'incomp' and c < 0:
        result = 1
        
    return result


def compute_2d_accuracy(label, c, w):
    ''' Given gold label, competence value c, and warmth value w, return 1 if hypothesis matches gold and 0 otherwise. '''

    result = 0
    
    if label == 'warm_comp' and w >= 0 and c >= 0:
        result = 1
    elif label == 'cold_comp' and w < 0 and c >= 0:
        result = 1
    elif label ==  'warm_incomp' and w >= 0 and c < 0:
        result = 1
    elif label == 'cold_incomp' and w < 0 and c < 0:
        result = 1        
    
    return result       


def compute_accuracy(df):
    ''' Given df containing columns 'Target' (gold labels), 'Warmth', and 'Competence', return accuracy. '''

    if 'Target' in df:
        acc = []
    
        # heuristic: if target labels contain an underscore, use 2-D accuracy 
        # (i.e. projected point should lie in correct quadrant)
        # otherwise use 1D accuracy (correct along the relevant axis only)
    
        labels = list(set(df['Target'].tolist()))  
        use_1d_accuracy = True
    
        for label in labels:
            if '_' in label:
                 use_1d_accuracy = False

        # compute correctness for each point 
        for index, row in df.iterrows():
            if use_1d_accuracy:
                acc.append(compute_1d_accuracy(row['Target'], row['Competence'], row['Warmth']))
            else:
                acc.append(compute_2d_accuracy(row['Target'], row['Competence'], row['Warmth']))
                
        return np.mean(acc)
    
    else:
        print('No target label information available')
        return 0
        


def compute_spearman_correlation(df):
    """
    Given a DataFrame with true continuous labels in columns
    'competence' and 'warmth' (lower‐case) and predicted values
    in 'Competence' and 'Warmth' (upper‐case), compute Spearman ρ.
    """
    # drop any rows with NaNs
    sub = df.dropna(subset=['competence','warmth','Competence','Warmth'])
    comp_r, _ = spearmanr(sub['competence'], sub['Competence'])
    warm_r, _ = spearmanr(sub['warmth'],    sub['Warmth'])
    print(f"Spearman ρ – Competence: {comp_r:.4f}, Warmth: {warm_r:.4f}")
    return comp_r, warm_r


if __name__ == "__main__":

    # sentence embedding model
    model_name = 'roberta-large-nli-mean-tokens'
    model = SentenceTransformer(model_name)

    # load test file (could have either discrete “Target” labels or continuous)
    #test_dir      = 'data'
    test_dir      = 'data/cross_validation'
    #test_filename = 'BWS_annotations_modified2'  # e.g. testing_continuous.csv
    test_filename = 'training_two_adjectives1' 
    test_df = pd.read_csv(f'{test_dir}/{test_filename}.csv')

    # always compute embeddings & polar projection
    test_df = get_embeddings(test_df, model)
    test_df = compute_warmth_competence(
        test_df, model_name,
        polar_model='axis_rotated',
        PLS=True, PCA=False
    )
    save_predictions(test_df, test_filename, model_name)

    # now choose evaluation based on columns present:
    if {'competence','warmth'}.issubset(test_df.columns):
        # continuous true labels → Spearman
        compute_spearman_correlation(test_df)
    elif 'Target' in test_df.columns:
        # discrete labels → accuracy
        accuracy = compute_accuracy(test_df)
        print('ACCURACY: ', accuracy)
    else:
        print("No suitable true‐label columns found (need 'Target' or 'competence'+'warmth').")
        
