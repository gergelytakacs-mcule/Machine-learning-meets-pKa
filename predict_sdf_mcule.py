import sys
import click
import logging
import pickle
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
from rdkit.Chem.MolStandardize.standardize import TautomerCanonicalizer, Uncharger
from mchem.sdf import SDFParserWrapper, SDFStruct

from scripts.gen_clean_mono_dataset import (
    ADDITIONAL_SALTS,
    ADDITIONAL_FILTER_RULES,
    LIPINSKI_RULES,
    check_sanitization,
    check_on_remaining_salts,
)


# NOTE setting loglevel to DEBUG prints a lot of unnecessary RDKit messages
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(ch)



def clean_mol(mol):
    """
    Single molecule implementation of the cleaning method in
    scripts.gen_clean_mono_dataset
    """
    if not mol:
        return

    salt_rm = SaltRemover.SaltRemover()
    salt_rm.salts.extend(ADDITIONAL_SALTS)
    mol = salt_rm.StripMol(mol)

    if not mol.GetNumAtoms():
    # the whole mol was salt
        return

    if not check_on_remaining_salts(mol):
    # mol still is multicomponent
        return

    # try to sanitize the mol
    return check_sanitization(mol)


# XXX get properties from SDF? more efficient
def filter_mol(mol):
    """
    Single molecule implementation of the filtering method in
    scripts.gen_clean_mono_dataset
    """
    if not mol:
        return

    # allow max 1 RO5 violation
    violations = 0
    for func, thres in LIPINSKI_RULES:
        if func(mol) > thres:
            violations += 1
        if violations > 1:
            return

    # additional filtering
    for func, thres in ADDITIONAL_FILTER_RULES:
        if func(mol) > thres:
            return

    return mol


def run_molvs_tautomers(mol, max_tautomers=1000):
    """Tautomer generation with
    Single molecule implementation of the run_molvs_tautomers method
    in scripts.gen_clean_mono_dataset
    """
    if not mol:
        return
    uc = Uncharger()
    tc = TautomerCanonicalizer(max_tautomers=max_tautomers)

    uncharded_mol = uc.uncharge(mol)
    canonical_mol = tc.canonicalize(uncharded_mol)
    return check_sanitization(canonical_mol)


@click.command()
@click.option('--model', type=str, help='Trained model.')
def run(
    model,
    pka_colname='pKa_prediction'
):
    with open(model, 'rb') as model_file:
        model = pickle.load(model_file)

    for idx, sdf in enumerate(SDFParserWrapper(file_name=None, factory=SDFStruct), 1):
        mol = Chem.MolFromMolBlock(sdf.write(mol=True))
        mol = clean_mol(mol)
        mol = filter_mol(mol)
        mol = run_molvs_tautomers(mol)

        if not mol:
            continue

        start = time.time()
        fmorgan3 = np.array([
            AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=3,
                nBits=4096,
                useFeatures=True
            )
        ])

        pka_prediction = model.predict(fmorgan3)
        end = time.time()

        # assert len(pka_prediction) == 1

        # correct for pandas default precision
        pka_prediction = round(pka_prediction[0], 6)

        sdf_fields = {
            **sdf.fields,
            pka_colname: pka_prediction
        }
        sdf.set_fields(sdf_fields)
        sys.stdout.write(f'{sdf.write()}\n')

        logger.info('Progress: %s | FP and prediction time %s', idx, end - start)

if __name__ == '__main__':
    run()