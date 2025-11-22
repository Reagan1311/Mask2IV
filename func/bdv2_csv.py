import torch
import numpy as np
import torch.nn.functional as F
import os
import glob
import csv
import re
import spacy

nlp = spacy.load("en_core_web_sm")
def is_noun(word):
    doc = nlp(word)
    return doc[0].pos_ in ["NOUN", "PROPN"]


path = '/train_data/BridgeDataV2/raw/bridge_data_v2/'

raw_files = glob.glob(os.path.join(path, '**/lang.txt'), recursive=True)

head = ['path', 'duration', 'caption', 'object', 'confidence']

with open("/workspace/exp_outputs/train_bdv2.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write each row of data
    for i, row in enumerate(raw_files):
        caption_file = open(row, 'r')
        caption = caption_file.readline().strip()
        caption = ' '.join(caption.split())
        pattern_no_article = r"^\w+\s+([\w\s]+?)\s+(from|to|on|in|at|with|by|for|of|out|off|and|right|put|left)\b"
        pattern_article = r"(?:the|a)\s+([\w\s]+?)\s+(from|to|on|in|at|with|by|for|of|out|off|and|right|put|left)\b"
        caption_split = caption.split()
        if len(caption_split) > 3:
            if caption_split[1] not in ['the', 'a']:
                pattern = pattern_no_article
            else:
                pattern = pattern_article
            match = re.search(pattern, caption)
            object_name = match.group(1).lower() if match else 'n/a'

        elif len(caption_split) == 3 or len(caption_split) == 2:
            if '.' in caption_split[-1]:
                object_name = caption_split[-1].replace('.', '').lower()
            else:
                object_name = caption_split[-1].lower()
            if not is_noun(object_name):
                continue
        
        confidence = caption_file.readline()
        
        if 'confidence' in confidence:
            confidence = float(confidence.strip().replace('confidence: ', ''))
        else:
            confidence = 'n/a'
        
        dir_path = os.path.dirname(row)
        duration = len(os.listdir(os.path.join(dir_path, 'images0')))
        
        if i == 0:
            writer.writerow(head)
        writer.writerow([dir_path, duration, caption, object_name, confidence])
