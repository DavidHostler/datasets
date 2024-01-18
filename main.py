import numpy as np 
from sentence_transformers import SentenceTransformer 
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

doc_a = 'What is the meaning of life?'
doc_b = 'What do you think the meaning of life is?'


embeddings_a = model.encode(doc_a)
embeddings_b = model.encode(doc_b)

sum_of_embeddings = embeddings_a + embeddings_b
magnitude = np.sqrt(sum_of_embeddings.dot(sum_of_embeddings) )
sum_of_embeddings = sum_of_embeddings/magnitude

# print('MAGNITUDE: {}'.format(sum_of_embeddings))

combined_doc = doc_a + doc_b 
embed_combined_doc = model.encode(combined_doc)
embed_combined_doc /= np.sqrt(embed_combined_doc.dot(embed_combined_doc))

print('COSINE SIMILARITY: {}{}'.format(embed_combined_doc.dot(sum_of_embeddings) * 100, '%'))