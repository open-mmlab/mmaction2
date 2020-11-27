#!/usr/bin/env bash

cd ../../../
PYTHONPATH=. python tools/data/denormalize_proposal_file.py thumos14 --norm-proposal-file data/thumos14/proposals/thumos14_tag_val_normalized_proposal_list.txt --data-prefix data/thumos14/rawframes/val/
echo "Proposal file denormalized for val set"

PYTHONPATH=. python tools/data/denormalize_proposal_file.py thumos14 --norm-proposal-file data/thumos14/proposals/thumos14_tag_test_normalized_proposal_list.txt --data-prefix data/thumos14/rawframes/test/
echo "Proposal file denormalized for test set"

cd tools/data/thumos14/
