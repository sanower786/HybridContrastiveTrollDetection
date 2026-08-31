# preprocess.py
"""
Preprocessing pipeline for the Hybrid Contrastive–Classification framework.

Input CSV:
    data/sample_data.csv

Required columns:
    text
    label

Processing:
    1. Basic text cleaning
    2. MPNet sentence embeddings
    3. Four lightweight comment-level linguistic features:
         - comment length
         - punctuation count
         - uppercase ratio
         - sentiment polarity
    4. Save embeddings + auxiliary features + labels

Output:
    data/sample_data_embeddings.csv

Feature dimensionality:
    MPNet embedding : 768
    Auxiliary       :   4
    ----------------------
    Hybrid vector   : 772
"""

import os
import re
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Performs basic comment-level text normalization.

    The cleaning removes URLs, mentions, hashtags and HTML
    artifacts while preserving alphabetic characters and
    basic punctuation required for linguistic features.
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove hashtags while retaining the surrounding text
    text = re.sub(r"#\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove HTML artifact
    text = re.sub(r"&amp;", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# AUXILIARY LINGUISTIC FEATURES
# ============================================================

def auxiliary_features(text):
    """
    Extracts the four lightweight comment-level linguistic
    features used in the proposed framework.

    Features:
        1. Comment length
        2. Punctuation count
        3. Uppercase ratio
        4. Sentiment polarity
    """

    # --------------------------------------------------------
    # 1. Comment length
    # --------------------------------------------------------

    comment_length = len(text)

    # --------------------------------------------------------
    # 2. Punctuation count
    # --------------------------------------------------------

    punctuation_count = sum(
        1 for char in text
        if char in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    )

    # --------------------------------------------------------
    # 3. Uppercase ratio
    # --------------------------------------------------------

    alphabetic_chars = [
        char for char in text
        if char.isalpha()
    ]

    uppercase_chars = [
        char for char in alphabetic_chars
        if char.isupper()
    ]

    if len(alphabetic_chars) > 0:
        uppercase_ratio = (
            len(uppercase_chars) /
            len(alphabetic_chars)
        )
    else:
        uppercase_ratio = 0.0

    # --------------------------------------------------------
    # 4. Sentiment polarity
    # --------------------------------------------------------

    try:
        from textblob import TextBlob

        sentiment_polarity = TextBlob(
            text
        ).sentiment.polarity

    except ImportError:
        raise ImportError(
            "TextBlob is required for sentiment polarity. "
            "Install it using: pip install textblob"
        )

    return {
        "aux_len": float(comment_length),
        "aux_punct_count": float(punctuation_count),
        "aux_uppercase_ratio": float(uppercase_ratio),
        "aux_sentiment": float(sentiment_polarity)
    }


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def main(
    input_csv="data/sample_data.csv",
    output_csv="data/sample_data_embeddings.csv",
    model_name="sentence-transformers/all-mpnet-base-v2",
    batch_size=64
):

    # Lazy import because SentenceTransformer is relatively heavy
    from sentence_transformers import SentenceTransformer

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}"
        )

    df = pd.read_csv(input_csv)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "Input CSV must contain 'text' and 'label' columns."
        )

    print("=" * 70)
    print("HYBRID FRAMEWORK PREPROCESSING")
    print("=" * 70)

    print(f"Input file : {input_csv}")
    print(f"Number of samples: {len(df)}")

    # --------------------------------------------------------
    # Load text and labels
    # --------------------------------------------------------

    texts = (
        df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    labels = (
        df["label"]
        .astype(int)
        .values
    )

    # --------------------------------------------------------
    # Clean text + extract auxiliary features
    # --------------------------------------------------------

    cleaned_texts = []
    auxiliary_features_list = []

    print("\nExtracting comment-level linguistic features...")

    for text in tqdm(
        texts,
        desc="Processing comments"
    ):

        cleaned = clean_text(text)

        cleaned_texts.append(cleaned)

        features = auxiliary_features(cleaned)

        auxiliary_features_list.append(features)

    # --------------------------------------------------------
    # Load MPNet
    # --------------------------------------------------------

    print("\nLoading embedding model:")
    print(model_name)

    model = SentenceTransformer(model_name)

    # --------------------------------------------------------
    # Generate MPNet embeddings
    # --------------------------------------------------------

    print(
        "\nGenerating MPNet embeddings "
        f"(batch size = {batch_size})..."
    )

    embeddings = model.encode(
        cleaned_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(
        "MPNet embedding shape:",
        embeddings.shape
    )

    # --------------------------------------------------------
    # Verify MPNet dimensionality
    # --------------------------------------------------------

    if embeddings.shape[1] != 768:
        raise ValueError(
            f"Expected MPNet embedding dimension 768, "
            f"but received {embeddings.shape[1]}."
        )

    # --------------------------------------------------------
    # Construct embedding dataframe
    # --------------------------------------------------------

    embedding_columns = [
        f"f{i}"
        for i in range(embeddings.shape[1])
    ]

    embedding_df = pd.DataFrame(
        embeddings,
        columns=embedding_columns
    )

    # --------------------------------------------------------
    # Construct auxiliary feature dataframe
    # --------------------------------------------------------

    auxiliary_df = pd.DataFrame(
        auxiliary_features_list
    )

    # --------------------------------------------------------
    # Verify exactly four auxiliary features
    # --------------------------------------------------------

    expected_auxiliary_columns = [
        "aux_len",
        "aux_punct_count",
        "aux_uppercase_ratio",
        "aux_sentiment"
    ]

    if list(auxiliary_df.columns) != expected_auxiliary_columns:
        raise ValueError(
            "Auxiliary feature configuration does not match "
            "the expected four-feature representation."
        )

    if auxiliary_df.shape[1] != 4:
        raise ValueError(
            f"Expected 4 auxiliary features, "
            f"but found {auxiliary_df.shape[1]}."
        )

    # --------------------------------------------------------
    # Combine MPNet + auxiliary features
    # --------------------------------------------------------

    output_df = pd.concat(
        [
            embedding_df,
            auxiliary_df.reset_index(drop=True)
        ],
        axis=1
    )

    output_df["label"] = labels

    # --------------------------------------------------------
    # Verify final dimensionality
    # --------------------------------------------------------

    feature_columns = (
        embedding_columns +
        expected_auxiliary_columns
    )

    if len(feature_columns) != 772:
        raise ValueError(
            f"Expected 772 feature dimensions, "
            f"but obtained {len(feature_columns)}."
        )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    output_directory = os.path.dirname(output_csv)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    output_df.to_csv(
        output_csv,
        index=False
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)

    print(
        f"MPNet dimensions       : {len(embedding_columns)}"
    )

    print(
        f"Auxiliary dimensions   : {len(expected_auxiliary_columns)}"
    )

    print(
        f"Hybrid dimensions      : {len(feature_columns)}"
    )

    print(
        f"Number of samples      : {len(output_df)}"
    )

    print(
        f"Output file            : {output_csv}"
    )

    print("\nAuxiliary features:")

    for feature in expected_auxiliary_columns:
        print(f"  - {feature}")

    print("=" * 70)


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess text and construct the "
            "772-dimensional hybrid representation."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_data.csv",
        help="Input CSV containing text and label columns."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_data_embeddings.csv",
        help="Output CSV containing embeddings and features."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding generation batch size."
    )

    args = parser.parse_args()

    main(
        input_csv=args.input,
        output_csv=args.output,
        model_name=args.model,
        batch_size=args.batch_size
    )