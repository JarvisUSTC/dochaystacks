#!/bin/bash

DATASET_FOLDER="/path/to/dataset/folder"
DATASET_FILE="$DATASET_FOLDER/test_docVQA.json"
IMAGE_ROOT="$DATASET_FOLDER/Test"
IMAGE_DIR="DocHaystack_1000"

OUTPUT_DIR="./output/docvqa_1000"


python model/VRAG_retrieval.py --dataset_file $DATASET_FILE --image_root $IMAGE_ROOT --image_dir $IMAGE_DIR --output_dir $OUTPUT_DIR