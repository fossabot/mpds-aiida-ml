#!/usr/bin/env python
"""
An example MPDS ab initio data generation scenario
Authors: E. Blokhin*12, A. Sobolev*1, and P. Villars*3
(1.) some (random) elements -> MPDS structures
(2.) MPDS structures -> experimental properties
(3.) MPDS structures -> machine-learning properties
(4.) MPDS structures -> abinitio properties
"""
import os
import sys
import time
import random
from copy import deepcopy
import json
from pprint import pprint

import httplib2
import numpy as np

from mpds_client import MPDSDataRetrieval, APIError

from mpds_ml_labs.prediction import prop_models, periodic_elements
from mpds_ml_labs.common import make_request
from mpds_ml_labs.struct_utils import get_formula, json_to_ase
from mpds_ml_labs.cif_utils import ase_to_eq_cif


supported_arities = {1: 'unary', 2: 'binary', 3: 'ternary', 4: 'quaternary', 5: 'quinary'}

supported_aninitio_props = {
    'heat_capacity': {
        'name': 'heat capacity at constant pressure',
        'rounding': 1,
        'units': 'J K-1 mol-1',
        'conditions': [
            {'name': 'Temperature', 'scalar': 298.15, 'units': 'K'},
            {'name': 'Pressure', 'scalar': 0.101325, 'units': 'MPa'}
        ],
        'symbol': 'C<sub>p</sub>'
    },
    'bulk_modulus': {
        'name': 'isothermal bulk modulus',
        'rounding': 0,
        'units': 'GPa',
        'conditions': [{'name': 'Temperature', 'scalar': 0, 'units': 'K'}],
        'symbol': 'B<sub>T</sub>'
    },
    'young_modulus': {
        'name': 'Young modulus',
        'rounding': 1,
        'units': 'GPa',
        'conditions': [{'name': 'Temperature', 'scalar': 0, 'units': 'K'}],
        'symbol': 'E'
    },
    'shear_modulus': {
        'name': 'shear modulus',
        'rounding': 1,
        'units': 'GPa',
        'conditions': [{'name': 'Temperature', 'scalar': 0, 'units': 'K'}],
        'symbol': 'G'
    },
    'poisson_ratio': {
        'name': 'poisson ratio',
        'rounding': 3,
        'units': None,
        'conditions': [{'name': 'Temperature', 'scalar': 0, 'units': 'K'}],
        'symbol': '&mu;'
    },
    'direct_band_gap': {
        'name': 'band gap for direct transition',
        'rounding': 1,
        'units': 'eV',
        'conditions': [{'name': 'Temperature', 'scalar': 0, 'units': 'K'}],
        'symbol': 'E<sub>g</sub>'
    }
}

mapping_ml_abinitio = {
    'z': 'bulk_modulus',
    'x': 'heat_capacity',
    'w': 'direct_band_gap'
}

CACHE_FILE = os.path.dirname(
    os.path.realpath(os.path.abspath(__file__))
) + os.sep + 'example_aiida_cache.json'
if not os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'w') as f:
        f.write('{}')

LABS_SERVER_ADDR = 'https://labs.mpds.io/predict' # http://127.0.0.1:5000/predict
raw_req = httplib2.Http()
mpds_api = MPDSDataRetrieval()


def get_structures(elements):
    """
    Given some arbitrary chemical elements,
    get their possible crystalline structures
    """
    assert sorted(list(set(elements))) == sorted(elements) and \
    len(elements) <= len(supported_arities)

    structures = []
    for item in mpds_api.get_data(
        {
            "props": "atomic structure",
            "elements": '-'.join(elements),
            "classes": supported_arities[len(elements)]
        },
        fields={
            'S':[
                'phase',
                'phase_id',
                'entry',
                'occs_noneq',
                'cell_abc',
                'sg_n',
                'basis_noneq',
                'els_noneq'
            ]
        }
    ):
        ase_obj, error = json_to_ase(item[3:])
        if error:
            print("Structure compilation error: %s" % error)
            continue
        ase_obj.info['phase'] = item[0]
        ase_obj.info['phase_id'] = item[1]
        ase_obj.info['entry'] = item[2]
        structures.append(ase_obj)

    return structures


def get_ab_initio_props(ase_obj):
    """
    This is the mock-up, showing memoization
    of the MPDS phases inside AiiDA
    """
    # Option I. So far used
    formulae = get_formula(ase_obj)
    sgs = ase_obj.info['spacegroup'].no
    fingerprint = "%s-%s" % (formulae, sgs)

    # Option II. More robust
    #assert 'phase' in ase_obj.info
    #fingerprint = ase_obj.info['phase']

    f = open(CACHE_FILE)
    available = json.loads(f.read()) or {}
    f.close()

    if fingerprint in available:
        print("Getting the result from AiiDA cache")
        return available[fingerprint]

    tpl = deepcopy(supported_aninitio_props)
    for prop in tpl:
        tpl[prop]['factual'] = 42

    print("Generated the result anew")
    available[fingerprint] = tpl

    f = open(CACHE_FILE, "w")
    f.write(json.dumps(available, indent=4))
    f.close()

    return tpl


def get_machine_learning_props(ase_obj):
    time.sleep(3) # to decrease request rate
    output = make_request(raw_req, LABS_SERVER_ADDR, {'structure': ase_to_eq_cif(ase_obj)})
    if 'error' in output:
        print("Error while getting the results: %s" % output['error'])
        return None

    tpl = {}
    for prop_id in output['prediction']:
        if not mapping_ml_abinitio.get(prop_id):
            continue
        tpl.update({mapping_ml_abinitio[prop_id]: supported_aninitio_props[mapping_ml_abinitio[prop_id]]})
        tpl[mapping_ml_abinitio[prop_id]]['factual'] = output['prediction'][prop_id]['value']

    return tpl


def get_peer_reviewed_props(ase_obj=None, phase_id=None):
    assert (ase_obj and not phase_id) or (not ase_obj and phase_id)

    tpl = {}
    query = {}
    if ase_obj:
        query = dict(formulae=get_formula(ase_obj), sgs=ase_obj.info['spacegroup'].no)

    for prop in supported_aninitio_props:
        query['props'] = supported_aninitio_props[prop]['name']
        try:
            outdf = mpds_api.get_dataframe(
                query,
                fields={'P': [
                    'sample.material.chemical_formula',
                    'sample.material.phase_id',
                    'sample.measurement[0].property.scalar',
                    'sample.measurement[0].property.units',
                    'sample.measurement[0].condition[0].units',
                    'sample.measurement[0].condition[0].name',
                    'sample.measurement[0].condition[0].scalar'
                ]},
                columns=['Compound', 'Phase', 'Value', 'Units', 'Cunits', 'Cname', 'Cvalue'],
                phases=[phase_id] if phase_id else None
            )
        except APIError as e:
            if e.code != 204: # NB empty result
                print("While checking against the MPDS an error %s occured" % e.code)
            continue

        time.sleep(3) # to decrease request rate

        if supported_aninitio_props[prop]['units']:
            outdf = outdf[outdf['Units'] == supported_aninitio_props[prop]['units']]
        if outdf.empty:
            continue
        outdf['Value'] = outdf['Value'].astype('float64') # NB to treat values out of JSON bounds given as str
        tpl.update({prop: supported_aninitio_props[prop]})
        tpl[prop]['factual'] = np.median(outdf['Value'])

    return tpl


if __name__ == "__main__":
    """
    Main procedure
    """
    if len(sys.argv) > 1:
        elements = sys.argv[1:]
    else:
        elements = [random.choice(periodic_elements[1:]) for _ in range(random.randint(1, 5))]
    print("Elements: %s" % ', '.join(elements))

    structures = get_structures(elements)
    structures.sort(key=lambda x: x.info['spacegroup'].no)

    structures_by_sg = []
    last_sgn = None
    for s in structures:
        if s.info['spacegroup'].no != last_sgn:
            structures_by_sg.append([])
        last_sgn = s.info['spacegroup'].no
        structures_by_sg[-1].append(s)

    for sg_cls in structures_by_sg:
        print("%s (SG%s)" % (get_formula(sg_cls[0]), sg_cls[0].info['spacegroup'].no))
        minimal_struct = min([len(s) for s in sg_cls])

        # get structures with the minimal number of atoms and find the one with median cell vectors
        # proposed by @as
        cells = np.array([s.get_cell().reshape(9) for s in sg_cls if len(s) == minimal_struct])
        median_cell = np.median(cells, axis=0)
        median_idx = int(np.argmin(np.sum((cells - median_cell)**2, axis=1)**0.5))

        target_obj = sg_cls[median_idx]

        results = {}
        results['ab_initio'] = get_ab_initio_props(target_obj)
        results['machine_learning'] = get_machine_learning_props(target_obj)
        time.sleep(1) # to decrease request rate
        results['peer_reviewed'] = get_peer_reviewed_props(phase_id=target_obj.info['phase_id'])
        #results['peer_reviewed'] = get_peer_reviewed_props(target_obj)
        pprint(results)