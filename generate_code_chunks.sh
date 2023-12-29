#!/bin/sh
python extraction_top_n.py --random_seed=42 --batch_size=48 --N=5000 --outfile=code_chunks_and_scores.csv
