#!/usr/bin/env python
# coding: utf-8
import sys
import csv
import re
import spacy
import pandas as pd
import networkx as nx
sys.settrace
nlp = spacy.load('en_core_web_sm')


data = pd.read_csv('../dataset/GAD_Y_N_wPubmedID_preprocessed.csv')


def get_shortest_path(graph, pairs):
    """ Gets the shortest dependency tree paths for each pair"""
    path_lens = []
    for p in pairs:
        try:
            path_lens.append(nx.shortest_path_length(graph, p[0], p[1]))
        except:
            continue
    if len(path_lens) == 0:
        return [-1]
    return path_lens


def tree_distance(gene, disease, parsed):
    """ Get the minimum, maxium, and average minimal dep tree distance for the
        terms in a sentence"""
    edges = []
    gene_mentions = []
    disease_mentions = []
    for token in parsed:
        token_format = '{0}-{1}'.format(token.text, token.i)
        if gene in token.text:
            gene_mentions.append(token_format)
        if disease in token.text:
            disease_mentions.append(token_format)
        for child in token.children:
            edges.append((token_format, '{0}-{1}'.format(child.text, child.i)))
    graph = nx.Graph(edges)
    pairs = [(g, d) for g in gene_mentions for d in disease_mentions]
    min_dists = get_shortest_path(graph, pairs)
    if len(min_dists) == 0:
        min_dists = [-1]
    word_dists = [abs(int(p[0].rsplit('-', 1)[1]) - int(p[1].rsplit('-', 1)[1])) for p in pairs]
    try:
        return (max(min_dists), min(min_dists), sum(min_dists) / len(min_dists),
                min(word_dists), max(word_dists), sum(word_dists) / len(word_dists))
    except:
        print(gene, disease, [t.text for t in parsed])


def join_bad_splits(parsed):
    """ Default tokenizer splits over some or all '-'s do this as adding rules wasn't working"""
    for token in parsed:
        if re.fullmatch(r'[A-Z]', token.text) is not None:
            i = token.i
            if i == 0:
                continue
            with parsed.retokenize() as retokenizer:
                retokenizer.merge(parsed[i-1:i+1])
            return join_bad_splits(parsed)
        if token.text == '-':
            i = token.i
            with parsed.retokenize() as retokenizer:
                retokenizer.merge(parsed[i-1:i+2])
            # Merging removes a token, so iterating over the list goes out of index
            return join_bad_splits(parsed)
    return parsed


dists = []
for index, entry in data.iterrows():
    print(index)
    dists.append(tree_distance(entry.gene, entry.disease, join_bad_splits(nlp(entry.sentence))))

with open('dist_features.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for t in dists:
        writer.writerow(t)
