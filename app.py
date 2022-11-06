from sentence_transformers import SentenceTransformer, util
import json
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from os.path import exists
import numpy as np
import math
import torch
import requests


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1', device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    downloadURL = model_inputs['downloadURL']
    path = model_inputs['fileName']
    first_chapter = model_inputs['firstChapter'] if 'firstChapter' in model_inputs else None
    last_chapter = model_inputs['lastChapter'] if 'lastChapter' in model_inputs else None
    preview_mode = False
    query = model_inputs['query'] if 'query' in model_inputs else None

    chapters, embeddings = process_file(path, downloadURL)
    results = search(query, chapters, embeddings) if 'query' else []

    # Run the model

    # Return the results as a dictionary
    return {'results': results}


def part_to_chapter(part):
    soup = BeautifulSoup(part.get_body_content(), 'html.parser')
    paragraphs = [para.get_text().strip() for para in soup.find_all('p')]
    paragraphs = [para for para in paragraphs if len(para) > 0]
    if len(paragraphs) == 0:
        return None
    title = ' '.join([heading.get_text() for heading in soup.find_all('h1')])
    return {'title': title, 'paras': paragraphs}

min_words_per_para = 150
max_words_per_para = 500

def format_paras(chapters):
    for i in range(len(chapters)):
        for j in range(len(chapters[i]['paras'])):
            split_para = chapters[i]['paras'][j].split()
            if len(split_para) > max_words_per_para:
                chapters[i]['paras'].insert(j + 1, ' '.join(split_para[max_words_per_para:]))
                chapters[i]['paras'][j] = ' '.join(split_para[:max_words_per_para])
            k = j
            while len(chapters[i]['paras'][j].split()) < min_words_per_para and k < len(chapters[i]['paras']) - 1:
                chapters[i]['paras'][j] += '\n' + chapters[i]['paras'][k + 1]
                chapters[i]['paras'][k + 1] = ''
                k += 1            

        chapters[i]['paras'] = [para.strip() for para in chapters[i]['paras'] if len(para.strip()) > 0]
        if len(chapters[i]['title']) == 0:
            chapters[i]['title'] = '(Unnamed) Chapter {no}'.format(no=i + 1)

def print_previews(chapters):
    for (i, chapter) in enumerate(chapters):
        title = chapter['title']
        wc = len(' '.join(chapter['paras']).split(' '))
        paras = len(chapter['paras'])
        initial = chapter['paras'][0][:20]
        preview = '{}: {} | wc: {} | paras: {}\n"{}..."\n'.format(i, title, wc, paras, initial)
        print(preview)

def get_chapters(book_path, print_chapter_previews, first_chapter, last_chapter):
    book = epub.read_epub(book_path)
    parts = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    chapters = [part_to_chapter(part) for part in parts if part_to_chapter(part) is not None]
    last_chapter = min(last_chapter, len(chapters) - 1)
    chapters = chapters[first_chapter:last_chapter + 1]
    format_paras(chapters)
    if print_chapter_previews:
        print_previews(chapters)
    return chapters

def get_embeddings(texts) -> np.array:
    if type(texts) == str:
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]
    return model.encode(texts)

def read_json(json_path):
    print('Loading embeddings from "{}"'.format(json_path))
    with open(json_path, 'r') as f:
        values = json.load(f)
    return (values['chapters'], np.array(values['embeddings']))

    
def read_epub(book_path, json_path, preview_mode, first_chapter, last_chapter):
    chapters = get_chapters(book_path, preview_mode, first_chapter, last_chapter)
    if preview_mode:
        return (chapters, None)
    print('Generating embeddings for chapters {}-{} in "{}"\n'.format(first_chapter, last_chapter, book_path))
    paras = [para for chapter in chapters for para in chapter['paras']]
    embeddings = get_embeddings(paras)
    try:
        with open(json_path, 'w') as f:
            json.dump({'chapters': chapters, 'embeddings': embeddings.tolist()}, f)
    except:
        print('Failed to save embeddings to "{}"'.format(json_path))
    return (chapters, embeddings)

def process_file(path, downloadURL, preview_mode=False, first_chapter=0, last_chapter=math.inf):
    chapters, embeddings = None, None
    if path[-4:] == 'json':
        chapters, embeddings = read_json(path)
    elif path[-4:] == 'epub':
        json_path = 'embeddings-{}-{}-{}.json'.format(first_chapter, last_chapter, path)
        if exists(json_path):
            chapters, embeddings = read_json(json_path)
        else:
            if downloadURL == None:
                return {'message': "No url provided"}
            content = requests.get(downloadURL).content
            open(path, 'wb').write(content)
            chapters, embeddings = read_epub(path, json_path, preview_mode, first_chapter, last_chapter) 
    else:
        print('Invalid file format. Either upload an epub or a json of book embeddings.')  
    return chapters, embeddings      


def index_to_para_chapter_index(index, chapters):
    for chapter in chapters:
        paras_len = len(chapter['paras'])
        if index < paras_len:
            return chapter['paras'][index], chapter['title'], index
        index -= paras_len
    return None

def search(query, chapters, embeddings, n=3):
    query_embedding = get_embeddings(query)[0]
    scores = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
    sorted_indexes = sorted([i for i in range(len(embeddings))], key=lambda i: scores[i], reverse=True)[:n]

    results = []

    # header_msg ='Results for query "{}" in "The Wizard and the Prophet.epub"'.format(query)
    # results.append(header_msg)

    for index in sorted_indexes:
        para, title, para_no = index_to_para_chapter_index(index, chapters)
        result = {'chapter': title, 'para': para, 'para_no': para_no, 'score': scores[index]}
        results.append(result)
        # result_msg = '\nChapter: "{}", Passage number: {}, Score: {:.2f}\n"{}"'.format(title, para_no, scores[index], para)
    #     print_and_write(result_msg, f)
    # print_and_write('\n', f)
    return results