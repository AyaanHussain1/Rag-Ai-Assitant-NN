import numpy as np
def cosine_similarity(vec1,vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1,v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product/(norm_v1*norm_v2)
a = [1,2,3]
b = [4,5,6]
print("NumPy Similarity:", cosine_similarity(a, b))