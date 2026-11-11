
import numpy as np


def cosine_similarity(a, b):
    # 1. Producto punto
    producto_punto = np.dot(a, b)

    # 2. Calcular magnitudes
    magnitud_a = np.linalg.norm(a)
    magnitud_b = np.linalg.norm(b)

    # 3. Evitar división entre cero
    if magnitud_a == 0 or magnitud_b == 0:
        raise ValueError("Los vectores no pueden ser cero")

    # 4. Calcular similitud
    return producto_punto / (magnitud_a * magnitud_b)


vector_a = np.array([1, 2])
vector_b = np.array([2, 4])

resultado = cosine_similarity(vector_a, vector_b)

print(f"Similitud: {resultado:.4f}")
