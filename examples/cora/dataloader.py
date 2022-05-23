import os
import pandas as pd
import networkx as nx

def load_cora(path, categorial_class_labels=True):
    
    cites_csv = os.path.join(path, "cites.csv")
    papers_csv = os.path.join(path, "paper.csv")
    contents_csv = os.path.join(path, "content.csv")

    cites = pd.read_csv(cites_csv)
    papers = pd.read_csv(papers_csv)
    contents = pd.read_csv(contents_csv)

    # Make One-Hot Word Vectors
    contents.word_cited_id = pd.to_numeric(contents.word_cited_id.apply(lambda x: x[4:]))
    contents = pd.crosstab(contents.paper_id, contents.word_cited_id)
    # Join word vectors with class_labels
    papers = papers.join(contents, on='paper_id')
    
    class_label_mapping = {'Genetic_Algorithms': 0,
                    'Reinforcement_Learning': 1,
                    'Theory': 2,
                    'Rule_Learning': 3,
                    'Case_Based': 4,
                    'Probabilistic_Methods': 5,
                    'Neural_Networks': 6}

    G = nx.DiGraph()
    # Add nodes (papers) with features (word vectors) and labels to graph G
    for index, row in papers.iterrows():
        wordvector = row[papers.columns[2:]].to_list()
        paper_id = row[papers.columns[0]]
        paper_label = row[papers.columns[1]]
        
        if categorial_class_labels:
            # convert class label to categorial one-hot vector
            one_hot_class_label = [0] * len(class_label_mapping)
            one_hot_class_label[class_label_mapping[paper_label]] = 1
            G.add_node(paper_id, features=wordvector, label=one_hot_class_label)
        else:
            G.add_node(paper_id, features=wordvector, label=paper_label)
    
    # Add edges (cites) to graph G
    for index, row in cites.iterrows():
        G.add_edge(row[cites.columns[0]], row[cites.columns[1]])

    return G
