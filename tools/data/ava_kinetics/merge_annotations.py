# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import pickle


def check_file(path):
    if os.path.isfile(path):
        return
    else:
        path = path.split('/')
        folder = '/'.join(path[:-1])
        filename = path[-1]
        info = '%s not found at %s' % (filename, folder)
        raise FileNotFoundError(info)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--avakinetics_root',
        type=str,
        default='../../../data/ava_kinetics',
        help='the path to save ava-kinetics dataset')
    root = p.parse_args().avakinetics_root

    kinetics_annot = root + '/kinetics_train.csv'
    ava_annot = root + '/annotations/ava_train_v2.2.csv'

    check_file(kinetics_annot)
    check_file(ava_annot)

    with open(kinetics_annot) as f:
        record = f.readlines()

    with open(ava_annot) as f:
        record += f.readlines()

    with open(ava_annot, 'w') as f:
        for line in record:
            f.write(line)

    kinetics_proposal = root + '/kinetics_proposal.pkl'
    ava_proposal = root + '/annotations/' \
                          'ava_dense_proposals_train.FAIR.recall_93.9.pkl'

    check_file(kinetics_proposal)
    check_file(ava_proposal)

    lookup = pickle.load(open(kinetics_proposal, 'rb'))
    lookup.update(pickle.load(open(ava_proposal, 'rb')))

    with open(ava_proposal, 'wb') as f:
        pickle.dump(lookup, f)
