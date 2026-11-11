
from sentence_transformers import SentenceTransformer

# 1. Cargar el modelo local
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. Nuestra base de conocimientos

documentos = [
    (
        "JWT es un estándar utilizado para autenticar usuarios, "
        "verificar su identidad y gestionar tokens de acceso."
    ),
    (
        "PostgreSQL es un sistema de bases de datos relacionales "
        "que permite guardar información permanentemente "
        "y realizar consultas SQL."
    ),
    (
        "Docker es una plataforma para crear y ejecutar "
        "aplicaciones dentro de contenedores."
    ),
    (
        "Redis es una base de datos en memoria utilizada "
        "principalmente para caché, sesiones y almacenamiento "
        "temporal de información."
    ),
    (
        "FastAPI es un framework de Python utilizado "
        "para desarrollar APIs REST, crear endpoints HTTP "
        "y construir servicios backend."
    ),
    (
        "AWS es una plataforma de computación en la nube "
        "que permite desplegar aplicaciones, servidores "
        "y servicios mediante infraestructura cloud."
    )
]


# 3. Generar los embeddings una sola vez
embeddings = model.encode_document(documentos)


def buscar(pregunta, top_k=3):
    # Generamos el embedding de la consulta
    query_embedding = model.encode_query(pregunta)

    # Comparamos contra los documentos
    scores = model.similarity(
        query_embedding,
        embeddings
    )[0]

    # Ordenamos de mayor a menor similitud
    indices = scores.argsort(descending=True)[:top_k]

    resultados = []

    for indice in indices:
        resultados.append({
            "documento": documentos[indice],
            "similitud": round(scores[indice].item(), 4)
        })

    return resultados


# 4. Probar nuestro motor
while True:
    pregunta = input("\nEscribe tu pregunta (salir para terminar): ")

    if pregunta.strip().lower() == "salir":
        break

    resultados = buscar(pregunta)

    for posicion, resultado in enumerate(resultados, start=1):
        print(f"\n{posicion}. {resultado['documento']}")
        print(f"   Similitud: {resultado['similitud']}")
