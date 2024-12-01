from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

documents = ["Today I Learned", "Coffee Jelly", "Coffee Break"]
queries = ["test", "warnings"]

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)
query_vectors = vectorizer.transform(queries)

consine_sim = np.dot(query_vectors, doc_vectors.T).toarray()
print("Cosine Similarity: \n", consine_sim)

