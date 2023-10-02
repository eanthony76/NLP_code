from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model = SentenceTransformer('ms-marco-MiniLM-L-12-v2')

def embed(text):
    encoded_data = model.encode(text, show_progress_bar=True)
    encoded_data = np.asarray(encoded_data.astype('float32'))

    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    ids = np.array(range(0, len(data)), dtype='int64')
    index.add_with_ids(encoded_data, ids)

    faiss.write_index(index, 'text.index')
    return encoded_data


def embed_langchain(text, model):
    model_name = model
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vector_store = FAISS.from_texts(text, hf)
    return vector_store