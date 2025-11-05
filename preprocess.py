# preprocess.py
"""
Preprocessing pipeline:
- reads CSV with columns 'text' and 'label'
- cleans text
- computes sentence embeddings using SentenceTransformer(all-mpnet-base-v2)
- computes auxiliary lexical/behavioral features
- writes numeric CSV at data/sample_data_embeddings.csv with columns:
    f0,f1,...,f{D-1}, aux_len, aux_avg_wordlen, aux_lexdiv, aux_punct_count, aux_exclaim, aux_question, label
"""
import os, re, sys, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# text cleaning helpers (based on your earlier code)
import nltk
from nltk.corpus import stopwords

try:
    _ = stopwords.words("english")
except Exception:
    nltk.download("stopwords")
    _ = stopwords.words("english")

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def clean_stopwords_shortwords(body):
    stopwords_list = stopwords.words('english')
    words = body.split()
    clean_words = [word for word in words if (word not in stopwords_list)]
    return " ".join(clean_words)

def removing_unwanted_data(body):
    if not isinstance(body, str):
        body = str(body)
    body = body.strip()
    body = re.sub(r'https*\S+', '', body)   # urls
    body = re.sub(r'#\S+', ' ', body)       # hashtags
    body = re.sub(r'&amp;', '', body)
    body = re.sub(r'@\w+', '', body)        # mentions
    body = re.sub(r"[^a-zA-Z?.!,¿'\s]+", " ", body)  # keep punctuation that helps
    body = re.sub(r'\s+', ' ', body).strip()
    body = body.lower()
    body = clean_stopwords_shortwords(body)
    return body

def stem_words(body):
    return " ".join([stemmer.stem(word) for word in body.split()])

# Auxiliary features
def aux_features_from_text(text):
    # text assumed pre-cleaned (string)
    words = text.split()
    n_words = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if n_words>0 else 0.0
    lex_div = len(set(words))/n_words if n_words>0 else 0.0
    punct_count = sum([1 for c in text if c in ".,;:'\"-()[]{}"])
    exclaim = text.count('!')
    question = text.count('?')
    char_len = len(text)
    return {
        "aux_len": char_len,
        "aux_avg_wordlen": float(avg_word_len),
        "aux_lexdiv": float(lex_div),
        "aux_punct_count": int(punct_count),
        "aux_exclaim": int(exclaim),
        "aux_question": int(question)
    }

def main(input_csv="data/sample_data.csv", output_csv="data/sample_data_embeddings.csv", model_name="sentence-transformers/all-mpnet-base-v2", batch_size=64):
    # lazy imports for heavy packages
    from sentence_transformers import SentenceTransformer

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain at least 'text' and 'label' columns")

    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].astype(int).values

    # Clean + stem + compute aux features
    cleaned_texts = []
    aux_features_list = []
    for t in tqdm(texts, desc="Cleaning texts"):
        cl = removing_unwanted_data(t)
        st = stem_words(cl)
        cleaned_texts.append(st)
        aux = aux_features_from_text(st)
        aux_features_list.append(aux)

    # compute sentence embeddings
    print("Loading embedding model:", model_name)
    model = SentenceTransformer(model_name)
    print("Computing embeddings (batch_size=%d)..." % batch_size)
    embeddings = model.encode(cleaned_texts, batch_size=batch_size, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    print("Embeddings shape:", embeddings.shape)

    # Prepare output DataFrame: numeric columns f0..f{D-1}, aux columns, label
    D = embeddings.shape[1]
    emb_cols = [f"f{i}" for i in range(D)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    aux_df = pd.DataFrame(aux_features_list)
    out_df = pd.concat([emb_df, aux_df.reset_index(drop=True)], axis=1)
    out_df["label"] = labels

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print("Saved embeddings+aux features to:", output_csv)
    print("Output columns:", out_df.columns.tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sample_data.csv")
    parser.add_argument("--output", type=str, default="data/sample_data_embeddings.csv")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(input_csv=args.input, output_csv=args.output, model_name=args.model, batch_size=args.batch_size)
