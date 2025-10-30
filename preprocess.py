import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df['body'] = df['body'].astype(str)

    print("Generating text embeddings...")
    model = SentenceTransformer('all-mpnet-base-v2')
    text_embeddings = model.encode(df['body'], show_progress_bar=True)

    df['text_embedding'] = list(text_embeddings)
    X = np.stack(df['text_embedding'])
    y = df['is_troll'].values

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    np.savez(output_path, X=X_scaled, y=y)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data("data/sample_data.csv", "data/processed_data.npz")
