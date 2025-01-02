import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os
import numpy as np
import json
import argparse
import time
import logging
from typing import Annotated, Dict, List, Optional, cast
from tqdm import tqdm
import math

from vidore_benchmark.utils.iter_utils import batched
from vidore_benchmark.retrievers.registry_utils import load_vision_retriever_from_registry

class CoPaliSolver:
    def __init__(self, vision_retriever, image_root='data/Test', image_dir="DocHaystack_1000",use_question_query=False, batch_size=32):
        
        self.image_root = image_root
        self.image_dir = os.path.join(self.image_root, image_dir)
        
        self.vision_retriever = vision_retriever

        self.use_question_query = use_question_query
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.construct_vision_embeddings(batch_size) # construct vision embeddings
    
    def construct_vision_embeddings(self, batch_size=32):
        start_time = time.time()
        image_list = os.listdir(self.image_dir)
        image_list = [os.path.join(self.image_dir, x) for x in image_list]
        self.image_files = image_list

        # obtain List[Image.Image] from List[str]
        images = [Image.open(x) for x in image_list]

        # NOTE: To prevent overloading the RAM for large datasets, we will load the passages (images)
        # that will be fed to the model in batches (this should be fine for queries as their memory footprint
        # is negligible. This optimization is about efficient data loading, and is not related to the model's
        # forward pass which is also batched.
        self.emb_images: List[torch.Tensor] = []

        dataloader_prebatch_size = 10 * batch_size

        for passage_batch in tqdm(
            batched(images, n=dataloader_prebatch_size),
            desc="Dataloader pre-batching",
            total=math.ceil(len(images) / (dataloader_prebatch_size)),
        ):
            batch_emb_passages = self.vision_retriever.forward_passages(passage_batch, batch_size=batch_size)
            if isinstance(batch_emb_passages, torch.Tensor):
                batch_emb_passages = list(torch.unbind(batch_emb_passages))
                self.emb_images.extend(batch_emb_passages)
            else:
                self.emb_images.extend(batch_emb_passages)
        
        elapsed_time = time.time() - start_time
        print(f"Constructed vision embeddings in {elapsed_time:.4f} seconds")
        self.construct_time = elapsed_time
        return

    def get_combined_top_k_images(self, needle_word, pos_image, k=10):
        start_time = time.time()
        emb_queries = self.vision_retriever.forward_queries([needle_word], batch_size=1)
        scores = self.vision_retriever.get_scores(emb_queries, self.emb_images, batch_size=1)[0] # [1000]
        score_dict = {}
        for file, score in zip(self.image_files, scores):
            score_dict[file] = score

        top_k_images = sorted(score_dict.keys(), key=lambda x: score_dict[x], reverse=True)[:k]
        top_k_includes_pos = [int(pos_image in top_k_images[:i + 1]) for i in range(k)]

        elapsed_time = time.time() - start_time
        return top_k_images, top_k_includes_pos, elapsed_time
    
    def process_dataset(self, dataset_file, output_dir, top_k_filter=10, use_filter=False):


        with open(dataset_file, 'r') as f:
            data = json.load(f)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        total_entries = len(data)
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        top10_correct = 0

        total_combined_time = 0
        total_llava_filtering_time = 0

        for idx, entry in enumerate(data):
            question = entry["conversations"][0]["value"]

            id = entry["id"]


            if self.use_question_query:
                needle_word = question
            else:
                needle_word = entry["needle"]


            pos_image = os.path.join(self.image_dir, entry["pos_image"][0])

            top_k_images, top_k_includes_pos, combined_time = self.get_combined_top_k_images(needle_word, pos_image, k=top_k_filter)
            total_combined_time += combined_time

            if use_filter:
                filtered_images, llava_filtering_time = self.filter_with_llava(top_k_images, question)
                total_llava_filtering_time += llava_filtering_time
            else:
                filtered_images = top_k_images
                total_llava_filtering_time += 0

            filtered_images = [img.split("/")[-1] for img in filtered_images]

            pos_image = pos_image.split("/")[-1]

            top_k_includes_pos_filtered = [int(pos_image in filtered_images[:i + 1]) for i in range(len(filtered_images))]

            if top_k_includes_pos_filtered[0]:
                top1_correct += 1
            if any(top_k_includes_pos_filtered[:3]):
                top3_correct += 1
            if any(top_k_includes_pos_filtered[:5]):
                top5_correct += 1
            if any(top_k_includes_pos_filtered[:10]):
                top10_correct += 1


            file_name = id+".json"
            output_file = os.path.join(output_dir, f"{file_name}")

            pos_image = pos_image.split(".")[0]
            with open(output_file, "w") as f_out:
                json.dump({
                    "question": question,
                    "needle_word": needle_word,
                    "top_10_images": filtered_images[:10],
                    "top_k_includes_pos": top_k_includes_pos_filtered,
                    "real_positive_image": pos_image
                }, f_out, indent=4)

            print(f"Processed entry {idx + 1}/{total_entries}: Saved top 10 images to {output_file}")

        top1_accuracy = top1_correct / total_entries
        top3_accuracy = top3_correct / total_entries
        top5_accuracy = top5_correct / total_entries
        top10_accuracy = top10_correct / total_entries

        avg_combined_time = (total_combined_time + self.construct_time) / total_entries
        avg_llava_filtering_time = total_llava_filtering_time / total_entries

        log_file = os.path.join(output_dir, 'accuracy.log')

    
        with open(log_file, 'w') as log:
            log.write(f"Total Entries: {total_entries}\n")
            log.write(f"Top-1 Accuracy: {top1_accuracy:.2%}\n")
            log.write(f"Top-3 Accuracy: {top3_accuracy:.2%}\n")
            log.write(f"Top-5 Accuracy: {top5_accuracy:.2%}\n")
            log.write(f"Top-10 Accuracy: {top10_accuracy:.2%}\n")
            log.write(f"Average Combined Encoder Inference Time: {avg_combined_time:.4f} seconds\n")
            log.write(f"Average LLaVA Filtering Time: {avg_llava_filtering_time:.4f} seconds\n")

        print(f"Total Entries: {total_entries}")
        print(f"Top-1 Accuracy: {top1_accuracy:.2%}")
        print(f"Top-3 Accuracy: {top3_accuracy:.2%}")
        print(f"Top-5 Accuracy: {top5_accuracy:.2%}")
        print(f"Top-10 Accuracy: {top10_accuracy:.2%}")
        print(f"Average Combined Encoder Inference Time: {avg_combined_time:.4f} seconds")
        print(f"Average LLaVA Filtering Time: {avg_llava_filtering_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CoPali Solver with adjustable dataset and image directory.")
    parser.add_argument('--dataset_file', type=str, default="data/test_docVQA.json", help="Path to the dataset JSON file")
    parser.add_argument('--image_root', type=str, default="data/Test", help="Root directory for images")
    parser.add_argument('--image_dir', type=str, default="DocHaystack_1000", help="Directory for specific images")
    parser.add_argument('--top_k_filter', type=int, default=10, help="Top K filter for image retrieval") # used in v-rag
    parser.add_argument('--use_filter', action='store_true', help="Use LLaVA to filter irrelevant images")
    parser.add_argument('--output_dir', type=str, default="output/copali_results", help="Output directory for results")
    parser.add_argument('--use_question_query', action='store_true', help="Use question as the query instead of needle word")
    parser.add_argument('--model_class', type=str, default="colqwen2", help="Model class for the retriever")
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='Path to the pretrained model')
    args = parser.parse_args()

    retriever = load_vision_retriever_from_registry(
        args.model_class,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
    )

    solver = CoPaliSolver(vision_retriever=retriever, image_root=args.image_root, image_dir=args.image_dir, use_question_query=args.use_question_query)
    solver.process_dataset(args.dataset_file, args.output_dir, top_k_filter=args.top_k_filter, use_filter=args.use_filter)