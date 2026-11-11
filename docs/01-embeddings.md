# Clase 01 — Embeddings

**Proyecto:** Semantic Search API  
**Fecha:** 11 de noviembre de 2026

## Objetivo
Comprender qué es un embedding y cómo utilizar un modelo de IA local para buscar documentos por significado.

## Conceptos aprendidos
- Un embedding es una representación numérica de un texto.
- Los embeddings permiten comparar relaciones semánticas.
- La similitud coseno permite medir qué tan alineados están dos vectores.
- Podemos ejecutar modelos preentrenados localmente mediante Sentence Transformers.

## Tecnologías utilizadas
- Python 3.13
- PyTorch
- Sentence Transformers
- Modelo: paraphrase-multilingual-MiniLM-L12-v2

## Experimento
Realicé una búsqueda semántica con tres documentos sobre JWT, Docker y PostgreSQL.

**Pregunta:** ¿Cómo puedo verificar la identidad de un usuario?

**Resultados:**
- JWT: 0.4230
- Docker: 0.0461
- PostgreSQL: 0.0408

## Conclusión
El modelo identificó el documento de JWT como el más relacionado con la pregunta, aunque no utilizaba exactamente las mismas palabras.

La similitud es una puntuación matemática, no un porcentaje de confianza.

## Próximos pasos
- Experimentar con nuevas preguntas.
- Estudiar la similitud coseno.
- Crear una API con FastAPI.
- Investigar el almacenamiento de vectores con pgvector.
