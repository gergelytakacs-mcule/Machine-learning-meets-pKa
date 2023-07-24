import csv
import gzip
import pickle as pkl
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools, AllChem as Chem

from scripts.gen_clean_mono_dataset import run_oe_tautomers, cleaning, filtering, run_molvs_tautomers

__author__ = 'Marcel Baltruschat'
__copyright__ = 'Copyright © 2020-2023'
__license__ = 'MIT'
__version__ = '1.1.0'


def valid_file(path: str) -> str:
    with open(path):
        pass
    return path


def predict_for_df(df, model, pka_prediciton_colname, no_openeye=False):
    """Run the prediction for all entries in the dataframe"""
    print('Start preparing dataset...')
    df = cleaning(df, list(df.columns[df.columns != 'ROMol']))
    print(f'After cleaning: {len(df)}')

    df = filtering(df)
    print(f'After filtering: {len(df)}')

    if not no_openeye:
        print('Using OpenEye QuacPac for tautomer and charge standardization...')
        df = run_oe_tautomers(df)
        print(f'After QuacPac tautomers: {len(df)}')
    else:
        print('Using RDKit MolVS for tautomer and charge standardization...')
        df = run_molvs_tautomers(df)
        print(f'After MolVS: {len(df)}')

    fmorgan3 = []
    for mol in df.ROMol:
        fmorgan3.append(Chem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096, useFeatures=True))
    fmorgan3 = np.array(fmorgan3)
    df[pka_prediciton_colname] = model.predict(fmorgan3)
    return df


def load_pka_cache(filename):
    """Reads tha chached pKa values"""
    pka_cache = {}
    if filename:
        open_func = gzip.open if filename.endswith('.gz') else open
        reader = csv.DictReader(open_func(filename, 'rt').readlines())
        entry_id, prediction = reader.fieldnames
        pka_cache = {
            ele[entry_id]: (float(ele[prediction]) if ele[prediction] != '' else np.NaN)
            for ele in reader
        }
    return pka_cache


def load_model(filename):
    print('Loading model...')
    model_open_func = gzip.open if filename.endswith('.gz') else open
    with model_open_func(filename, 'rb') as f:
        model = pkl.load(f)
    return model


def load_sdf(filename):
    print('Loading SDF...')
    df = PandasTools.LoadSDF(filename)
    try:
        df.set_index('ID', inplace=True, verify_integrity=True)
    except ValueError:
        print('Warning: Molblock names are not unique (or missing), adding an unique index')
        df.ID = list(range(len(df)))
        df.ID = df.ID.astype(str)
        df.set_index('ID', inplace=True)
        for ix, mol in df.ROMol.items():
            mol.SetProp('_Name', ix)
    print(f'{len(df)} molecules loaded')
    return df


def get_args():
    parser = ArgumentParser()
    parser.add_argument('sdf', metavar='INPUT_SDF', type=valid_file)
    parser.add_argument('out', metavar='OUTPUT_PATH')
    parser.add_argument('--no-openeye', '-noe', action='store_true')
    parser.add_argument('--trained-model', type=str, help='Trained AI model as .pkl file.')
    parser.add_argument('--pka-cache', default=None, help='CSV containting MCULE IDs and pKa values.')
    parser.add_argument(
        '--pka-prediction-colname', default='pKa_prediction',
        help='Column name to be used for the pKa prediction.'
    )
    parser.add_argument('--mol-id', default='Molecule name', help='Mcule ID field')
    return parser.parse_args()


def run(args):
    model = load_model(args.trained_model)
    df = load_sdf(args.sdf)
    pka_cache = load_pka_cache(args.pka_cache)
    pka_prediciton_colname = args.pka_prediction_colname

    # insert pka from cache
    mol_id = args.mol_id
    df[pka_prediciton_colname] = df[mol_id].map(pka_cache)

    # drop rows which would be filtered out anyway (in pka_cache, but pka=None)
    drop_indices = df[(df[mol_id].isin(pka_cache) == True) & df[pka_prediciton_colname].isna()].index
    df.drop(drop_indices, inplace=True)

    # split dataframe into cached and predicted values
    df_cache = df[df[pka_prediciton_colname].isna() == False]
    df_predict = df[df[pka_prediciton_colname].isna()]

    if not df_predict.empty:
        df_predict = predict_for_df(df_predict, model, pka_prediciton_colname, args.no_openeye)

    print(f'pKa values inserted from cache for {df_cache.shape[0]} structures')
    print(f'pKa values predicted for {df_predict.shape[0]} structures')

    # join the two dfs
    result_df = pd.concat([df_cache, df_predict]).sort_index()

    print('Writing result file...')
    PandasTools.WriteSDF(result_df, args.out, properties=result_df.columns, idName='RowID')


# TODO Since the initial SDF entries are modified, the scripts should probably output a CSV instead of
# an SDF, or keep the original SDF molblocks and only append the prediction
if __name__ == '__main__':
    args = get_args()
    run(args)
