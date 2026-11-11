print("Importando Sentence Transformers...", flush=True)
from sentence_transformers import SentenceTransformer

#este es el modelo preentrenado multilingue 
print("Cargando modelo (la primera vez se descarga)...", flush=True)
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

documentos = [
    "JWT permite autenticar ususarios mediante tokens",
    "PostgreSQL permite almacenae informacion",
    "Docker ejecuta aplicaciones en contenedores"
]

pregunta = "¿como puedo verificar la identidad de un usuario?"

#generar embeddings de los documentos
print("Generando embeddings...", flush=True)
vectores = model.encode_document(documentos)

#generar embedding de la pregunta
vector_pregunta = model.encode_query(pregunta)

#calcular similitud 
similitudes = model.similarity(
    vector_pregunta,
    vectores
)[0]

#ordenar resultados de mayor a menor similitud
indices = similitudes.argsort(descending=True)

for indice in indices:
    print(
        f"{similitudes[indice]:.4f} - "
        f"{documentos[indice]}"
    )

    
