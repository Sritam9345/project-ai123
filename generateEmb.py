from sentence_transformers import SentenceTransformer

embedmodel = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def embeddings(input):
    embedds = embedmodel.encode(input)
    
    return embedds