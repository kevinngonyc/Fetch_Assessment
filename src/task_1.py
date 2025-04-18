import sys
from config import *
from sentence_transformers import SentenceTransformer


def get_sentence_transformer():
    """
    Returns a sentence transformer instance.

    The model consists of a BERT transformer, a pooling layer, and a normalization
    layer. The pooling layer is used to aggregate the token vectors produced by
    BERT to produce overall sentence vectors. Mean pooling is used in this case. 
    Normalization is used to ensure all outputs have the same distribution, improving
    training performance. 
    """
    return SentenceTransformer(sentence_transformer_model)


def get_embeddings(model, sentences):
    """
    Returns embeddings of ``sentences`` according to ``model``
    """
    return model.encode(sentences)


def get_similarity(model, embeddings):
    """
    Returns similarity of ``embeddings`` according to ``model``
    """
    return model.similarity(embeddings, embeddings)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentences = sys.argv[1:]
    else:
        sentences = sample_sentences
    
    model = get_sentence_transformer()
    embeddings = get_embeddings(model, sentences)
    similarities = get_similarity(model, embeddings)

    print("\n----Sentences----")
    print(sentences)
    print("\n----Embeddings----")
    print(embeddings)
    print("\n----Similarities----")
    print(similarities)