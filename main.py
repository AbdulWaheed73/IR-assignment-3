import pandas as pd
import numpy as np
import re
import pickle
import os
import json
import time
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize, bigrams, trigrams
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# global vars cuz easier this way
all_docs_original = []
all_docs_processed = []
doc_metadata = {}
bm25_model = None
bigrams_dict = {}
ps = PorterStemmer()  # porter stemmer

# load stopwords
try:
    stop_words = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # basic cleaning
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenize
    tokens = text.split()

    # remove stopwords and short words
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # stemming
    stemmed = [ps.stem(token) for token in tokens]

    return stemmed


def extract_bigrams(tokens):
    # get bigrams from token list
    bg = list(bigrams(tokens))
    return bg


def load_and_process_data():
    global all_docs_original, all_docs_processed, doc_metadata, bm25_model, bigrams_dict

    print("Loading data...")
    # try different encodings cuz csv might have weird chars
    try:
        df = pd.read_csv('Articles.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('Articles.csv', encoding='latin-1')

    # drop na rows
    df = df.dropna()
    print(f"Loaded {len(df)} articles")

    # print(df.head())  # debug

    # process each document
    for idx, row in df.iterrows():
        # combine article text and heading
        full_text = str(row['Article']) + ' ' + str(row['Heading'])

        all_docs_original.append(full_text)

        # preprocess
        processed = preprocess_text(full_text)
        all_docs_processed.append(processed)

        # extract bigrams
        bg = extract_bigrams(processed)
        bigrams_dict[idx] = bg

        # store metadata
        doc_metadata[idx] = {
            'heading': row['Heading'],
            'date': row['Date'],
            'type': row['NewsType'],
            'text': full_text,
            'word_count': len(full_text.split())
        }

    # initialize BM25
    print("Building BM25 index...")
    bm25_model = BM25Okapi(all_docs_processed)

    print("Done processing!")


def save_processed_data():
    # save everything so we don't have to reprocess
    if not os.path.exists('saved_data'):
        os.makedirs('saved_data')

    with open('saved_data/processed_docs.pkl', 'wb') as f:
        pickle.dump(all_docs_processed, f)

    with open('saved_data/original_docs.pkl', 'wb') as f:
        pickle.dump(all_docs_original, f)

    with open('saved_data/doc_metadata.pkl', 'wb') as f:
        pickle.dump(doc_metadata, f)

    with open('saved_data/bigrams.pkl', 'wb') as f:
        pickle.dump(bigrams_dict, f)

    with open('saved_data/bm25.pkl', 'wb') as f:
        pickle.dump(bm25_model, f)

    print("Saved processed data to saved_data/")


def load_processed_data():
    global all_docs_original, all_docs_processed, doc_metadata, bm25_model, bigrams_dict

    print("Loading preprocessed data...")

    with open('saved_data/processed_docs.pkl', 'rb') as f:
        all_docs_processed = pickle.load(f)

    with open('saved_data/original_docs.pkl', 'rb') as f:
        all_docs_original = pickle.load(f)

    with open('saved_data/doc_metadata.pkl', 'rb') as f:
        doc_metadata = pickle.load(f)

    with open('saved_data/bigrams.pkl', 'rb') as f:
        bigrams_dict = pickle.load(f)

    with open('saved_data/bm25.pkl', 'rb') as f:
        bm25_model = pickle.load(f)

    print("Loaded!")


def search(query, top_k=5):
    # preprocess query
    q_processed = preprocess_text(query)

    # get bm25 scores
    bm25_scores = bm25_model.get_scores(q_processed)

    # tfidf stuff
    tfidf_vec = TfidfVectorizer()
    tfidf_mat = tfidf_vec.fit_transform(all_docs_original + [query])

    # last row is the query
    query_vector = tfidf_mat[-1]
    doc_vectors = tfidf_mat[:-1]
    cos_sims = cosine_similarity(query_vector, doc_vectors)[0]

    # combine both scores
    # tried 0.5/0.5, 0.7/0.3 but this works better
    # also tried just bm25 alone but hybrid is def better
    final_scores = []
    for i in range(len(all_docs_original)):
        # normalize bm25 first
        bm25_normalized = bm25_scores[i] / (max(bm25_scores) + 0.0001)  # avoid division by zero

        score = 0.6 * bm25_normalized + 0.4 * cos_sims[i]
        # score = bm25_normalized  # old approach

        # reranking stuff

        # boost if query words are in the title
        title = doc_metadata[i]['heading'].lower()
        q_words = query.lower().split()
        if any(word in title for word in q_words):
            score *= 1.1  # 10% boost

        # exact phrase matching
        if query.lower() in all_docs_original[i].lower():
            score *= 1.15

        # idk but short docs seem less relevant
        if doc_metadata[i]['word_count'] < 50:
            score *= 0.95

        final_scores.append((i, score))

    # sort descending
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # get top k
    results = []
    for doc_id, score in final_scores[:top_k]:
        results.append({
            'doc_id': doc_id,
            'score': score,
            'heading': doc_metadata[doc_id]['heading'],
            'date': doc_metadata[doc_id]['date'],
            'type': doc_metadata[doc_id]['type'],
            'text': doc_metadata[doc_id]['text']
        })

    return results


def display_results(results, query):
    print(f"\nResults for '{query}':\n")
    print("=" * 80)

    for i, res in enumerate(results):
        print(f"\nRank {i+1} (Score: {res['score']:.4f})")
        print(f"Title: {res['heading']}")
        print(f"Date: {res['date']}")
        print(f"Type: {res['type']}")

        # show snippet
        snippet = res['text'][:200] + "..." if len(res['text']) > 200 else res['text']
        print(f"Snippet: {snippet}")
        print("-" * 80)


def show_stats():
    print("\n=== Index Statistics ===")
    print(f"Total documents: {len(all_docs_original)}")

    avg_len = np.mean([doc_metadata[i]['word_count'] for i in doc_metadata])
    print(f"Average document length: {avg_len:.2f} words")

    # count news types
    types = [doc_metadata[i]['type'] for i in doc_metadata]
    type_counts = Counter(types)
    print(f"\nDocument types:")
    for t, count in type_counts.most_common():
        print(f"  {t}: {count}")

    # vocab size
    all_tokens = []
    for doc in all_docs_processed:
        all_tokens.extend(doc)
    vocab_size = len(set(all_tokens))
    print(f"\nVocabulary size: {vocab_size} unique terms")


def run_evaluation():
    # load test queries
    if not os.path.exists('test_queries.json'):
        print("No test queries found. Create test_queries.json first.")
        return

    with open('test_queries.json', 'r') as f:
        test_set = json.load(f)

    print("\n=== Running Evaluation ===\n")

    precision_at_5 = []
    precision_at_10 = []
    recall_at_10 = []
    query_times = []

    for item in test_set:
        query = item['query']
        relevant_docs = set(item['relevant_docs'])

        # time the query
        start = time.time()
        results = search(query, top_k=10)
        query_time = time.time() - start
        query_times.append(query_time)

        retrieved_ids = [r['doc_id'] for r in results]

        # precision at 5
        top5 = retrieved_ids[:5]
        p5 = len([d for d in top5 if d in relevant_docs]) / 5.0
        precision_at_5.append(p5)

        # precision at 10
        p10 = len([d for d in retrieved_ids if d in relevant_docs]) / 10.0
        precision_at_10.append(p10)

        # recall at 10
        if len(relevant_docs) > 0:
            r10 = len([d for d in retrieved_ids if d in relevant_docs]) / len(relevant_docs)
            recall_at_10.append(r10)

        print(f"Query: '{query}'")
        print(f"  P@5: {p5:.3f}, P@10: {p10:.3f}, R@10: {r10:.3f}, Time: {query_time:.4f}s")

    print("\n=== Overall Results ===")
    print(f"Mean Precision@5: {np.mean(precision_at_5):.3f}")
    print(f"Mean Precision@10: {np.mean(precision_at_10):.3f}")
    print(f"Mean Recall@10: {np.mean(recall_at_10):.3f}")
    print(f"Average query time: {np.mean(query_times):.4f}s")

    # save results
    with open('evaluation_results.txt', 'w') as f:
        f.write("=== Evaluation Results ===\n\n")
        f.write(f"Mean Precision@5: {np.mean(precision_at_5):.3f}\n")
        f.write(f"Mean Precision@10: {np.mean(precision_at_10):.3f}\n")
        f.write(f"Mean Recall@10: {np.mean(recall_at_10):.3f}\n")
        f.write(f"Average query time: {np.mean(query_times):.4f}s\n")

    print("\nResults saved to evaluation_results.txt")


def main():
    print("=" * 80)
    print("        Information Retrieval System")
    print("=" * 80)

    # check if processed data exists
    if os.path.exists('saved_data/processed_docs.pkl'):
        load_processed_data()
    else:
        load_and_process_data()
        save_processed_data()

    print("\nCommands:")
    print("  search <query>  - Search for articles")
    print("  stats           - Show index statistics")
    print("  eval            - Run evaluation on test set")
    print("  quit            - Exit")
    print()

    while True:
        cmd = input("Enter command: ").strip()

        if cmd == 'quit' or cmd == 'exit' or cmd == 'q':
            print("Bye!")
            break
        elif cmd == 'stats':
            show_stats()
        elif cmd == 'eval':
            run_evaluation()
        elif cmd.startswith('search '):
            query = cmd[7:]
            if query:
                results = search(query)
                display_results(results, query)
            else:
                print("Please provide a query.")
        elif cmd == 'help':
            print("\nCommands:")
            print("  search <query>  - Search for articles")
            print("  stats           - Show index statistics")
            print("  eval            - Run evaluation on test set")
            print("  quit            - Exit")
        else:
            print("Unknown command. Type 'help' for available commands.")


if __name__ == '__main__':
    main()
