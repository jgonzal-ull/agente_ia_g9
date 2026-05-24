# Agente IA G9 — Sistema RAG con LlamaIndex

Sistema de preguntas y respuestas sobre documentos internos basado en RAG (Retrieval-Augmented Generation). Procesa PDFs página a página, los indexa en un vector store en memoria y responde preguntas usando un LLM, citando las fuentes consultadas.

---

## Estructura del proyecto

```
agente_ia_g9/
├── Aplicaciones/
│   ├── Dockerfile              # Imagen Docker ligera (python:3.11-slim)
│   ├── docker-compose.yml      # Configuración del contenedor
│   ├── main_lmstudio.py        # Pipeline con servidor LM Studio externo
│   └── main_openai.py          # Pipeline con API de OpenAI
├── Datos/
│   └── indice.xlsx             # Registro de documentos (metadatos)
├── Nuevos documentos/          # Drop zone: PDFs pendientes de indexar
└── README.md
```

---

## Scripts disponibles

### `main_lmstudio.py`

Conecta con un servidor LM Studio externo que expone una API compatible con OpenAI. No requiere GPU local ni clave de pago: toda la inferencia ocurre en el servidor remoto.

| Parámetro | Valor por defecto |
|---|---|
| Servidor | `http://openai.ull.es:8080/v1` |
| LLM | `qwen/qwen3.5-9b` |
| Embeddings | `text-embedding-qwen3-embedding-0.6b` |
| Token API | `LMSTUDIO_APITOKEN` en el script |
| Tamaño de chunk | 3000 palabras / 50 solapamiento |

### `main_openai.py`

Usa directamente la API oficial de OpenAI para embeddings y LLM. Requiere una clave de API válida (`OPENAI_API_KEY`).

| Parámetro | Valor por defecto |
|---|---|
| LLM | `gpt-4o-mini` |
| Embeddings | `text-embedding-3-small` |
| Clave API | `OPENAI_API_KEY` en el script |
| Tamaño de chunk | 300 palabras / 50 solapamiento |

### Funcionamiento común

Ambos scripts siguen el mismo pipeline:

1. Leen los PDFs de `Nuevos documentos/`
2. Cruzan cada fichero con `Datos/indice.xlsx` para obtener sus metadatos
3. Extraen el texto página a página con `pypdf`
4. Dividen el texto en trozos con solapamiento
5. Crean un índice vectorial en memoria insertando nodo a nodo (con log por pantalla)
6. Abren un chat engine y responden las preguntas definidas en `main()`

---

## Requisito previo: `indice.xlsx`

Antes de colocar un PDF en `Nuevos documentos/`, debe existir una fila para ese fichero en `Datos/indice.xlsx` (hoja `Hoja1`) con al menos estas columnas:

| Columna | Descripción |
|---|---|
| `Nombre Archivo` | Nombre exacto del fichero (p.ej. `informe.pdf`) |
| `Identificador` | ID único del documento, usado como prefijo de los chunk IDs |
| `Estado` | Estado del documento (p.ej. `Vigente`) |

Los documentos no registrados en el índice se omiten durante la indexación.

---

## Instrucciones Docker

### 1. Clonar el repositorio

```bash
git clone https://github.com/jgonzal-ull/agente_ia_g9.git
cd agente_ia_g9
```

### 2. Ajustar la ruta del volumen

Editar `Aplicaciones/docker-compose.yml` y cambiar la ruta del volumen a la ruta local del repositorio:

```yaml
volumes:
  - /ruta/local/agente_ia_g9:/app   # <-- cambiar por tu ruta
```

En Windows usar formato con barra invertida o comillas:

```yaml
volumes:
  - C:\Users\usuario\agente_ia_g9\:/app
```

### 3. Construir la imagen

```bash
docker compose -f Aplicaciones/docker-compose.yml up -d --build
```

Esto construye la imagen `agente_ia_g9` a partir del `Dockerfile` e inicia el contenedor.

### 4. Ejecutar `main_lmstudio.py`

Verificar que el servidor LM Studio en `http://openai.ull.es:8080/v1` está activo y tiene cargados los modelos indicados. Ajustar `LMSTUDIO_APITOKEN` en el script si es necesario.

```bash
docker exec -it agente_ia_g9-app-1 python3 /app/Aplicaciones/main_lmstudio.py
```

### 5. Ejecutar `main_openai.py`

Poner la clave de API de OpenAI en la constante `OPENAI_API_KEY` del script (o usar la variable de entorno).

```bash
docker exec -it agente_ia_g9-app-1 python3 /app/Aplicaciones/main_openai.py
```

### Obtener el nombre exacto del contenedor

Si el nombre del contenedor es diferente:

```bash
docker ps
```

---

## Personalizar las preguntas

Las preguntas que se lanzan al chat engine están definidas al final de `main()` en cada script:

```python
chatear(index, [
    "¿Quién es el responsable funcional de cada GLPI?",
    "¿Cómo es la estructura de GLPI?",
    "¿Cuáles son los días entre festivos del 2026?",
])
```

Editar la lista para adaptarla a los documentos indexados.

---

## Dependencias principales

| Paquete | Uso |
|---|---|
| `llama-index` | Framework RAG (índice, chat engine) |
| `llama-index-llms-openai-like` | LLM para LM Studio |
| `llama-index-llms-openai` | LLM para OpenAI |
| `llama-index-embeddings-openai` | Embeddings para OpenAI y LM Studio |
| `pypdf` | Extracción de texto de PDFs |
| `pandas` / `openpyxl` | Lectura del índice de documentos |
| `requests` | Llamadas HTTP al servidor LM Studio |
