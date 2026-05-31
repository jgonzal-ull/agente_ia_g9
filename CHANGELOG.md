# Changelog

Registro de cambios del proyecto **Agente IA G9** — Sistema RAG con LlamaIndex.

---

## [3.0] — main_lmstudio_v3.0.py / main_openai_v3.0.py

### Añadido
- **Soporte Markdown (.md)**: los ficheros `.md` se convierten a texto plano con `pypandoc` y se trocean con el mismo mecanismo que el resto de formatos.
- **Soporte DokuWiki (.dokuwiki)**: los ficheros DokuWiki se convierten a texto plano con `pypandoc` antes de dividirlos en chunks.
- **Validación de extensión**: ficheros con extensión no soportada muestran el mensaje `Extensión no soportada: '<ext>'` y se omiten sin error.
- **Despachador de formatos** (`digitaliza_un_documento`): arquitectura basada en dispatcher que delega en `_digitaliza_pdf`, `_digitaliza_dokuwiki` o `_digitaliza_markdown` según la extensión detectada.
- **Núcleo compartido pandoc** (`_digitaliza_con_pandoc`): función interna común para todos los formatos de texto procesados con pandoc; añadir un nuevo formato requiere solo una función de una línea y un `elif` en el despachador.
- **Variante OpenAI** (`main_openai_v3.0.py`): versión paralela del pipeline que usa la API oficial de OpenAI (`gpt-4o-mini` + `text-embedding-3-small`) en lugar del servidor LM Studio.
- **Documentación de funciones**: todas las funciones incluyen docstring de una línea.
- **Sección de configuración estructurada**: variables agrupadas por bloques (Rutas / Servidor / Chunking) con comentario inline en cada una.

### Cambiado
- **Lectura de PDFs**: sustituido `SimpleDirectoryReader` por `pypdf.PdfReader` directo, que extrae texto página a página de forma explícita y evita la lectura en binario que producía el dispatcher genérico de LlamaIndex.
- **Índice vectorial en memoria**: eliminada la persistencia en disco (`StorageContext`, `load_index_from_storage`). El índice se crea en memoria en cada ejecución.
- **Inserción nodo a nodo con log**: `construir_indice` inserta cada `TextNode` individualmente mostrando su ID, número de palabras y tokens estimados.
- **Estimación de tokens**: añadida función `_estimar_tokens` (~4 caracteres/token) para mostrar el coste aproximado de cada chunk durante la indexación.
- **Credenciales por variable de entorno**: `LMSTUDIO_APITOKEN` y `OPENAI_API_KEY` se leen con `os.getenv` en los ficheros versionados para no exponer claves en el repositorio.

### Infraestructura
- **Imagen Docker**: cambiada la base de `nvidia/cuda:12.8.1-runtime-ubuntu22.04` (~3 GB) a `python:3.11-slim` (~200 MB). Toda la inferencia es externa, por lo que no se necesita GPU en el contenedor.
- **Dependencias reducidas**: eliminados `torch`, `torchvision`, `torchaudio`, `transformers`, `sentence-transformers`, `scikit-learn`, `shap`, `matplotlib`, `datasets`, `evaluate`, `einops`, `accelerate`. Añadidos `llama-index-llms-openai` y `llama-index-llms-openai-like`.
- **`container_name` fijo**: el contenedor siempre se llama `agente_ia_g9` independientemente de la carpeta desde la que se ejecute `docker compose`.
- **Volumen adaptado**: la ruta del volumen en `docker-compose.yml` apunta al repositorio real en lugar de `c:\G9-IA\`.
- **`build` integrado en compose**: el `docker-compose.yml` incluye la directiva `build` para poder hacer `up --build` sin ejecutar `docker build` por separado.

---

## [4.0] — main_v4.0.py

### Añadido
- **Bot de Telegram**: integración con `python-telegram-bot`; el bot responde mensajes de texto usando el chat engine RAG.
- **Lista blanca de usuarios**: solo los `chat_id` incluidos en `lista_blanca` reciben respuestas del agente.
- **Comando `/start`**: mensaje de bienvenida al iniciar conversación con el bot.

### Cambiado
- El bot está desactivado por defecto (bloque `Application` comentado en `main()`); se activa descomentando las líneas correspondientes.

---

## [3.0] — main_v3.0.py

### Añadido
- **Persistencia del índice**: el `VectorStoreIndex` se guarda en disco (`Datos/data_storage/`) y se reutiliza entre ejecuciones.
- **Soporte DokuWiki**: los ficheros `.dokuwiki` se convierten con `pypandoc` y se trocean con `SemanticSplitterNodeParser`.
- **Filtros de metadatos**: posibilidad de filtrar por campo `Estado` del índice documental.
- **Función `actualizar_metadato_en_store`**: permite modificar metadatos de nodos ya indexados sin reindexar.
- **Refactorización en funciones**: el código se organiza en funciones reutilizables (`load_llm`, `load_index`, `create_or_add_index`, `chatear_simple`).

### Cambiado
- **Chunking por palabras con solapamiento** (`split_text_with_overlap`): reemplaza la división por páginas de la v1.
- **Embedding personalizado** (`LMStudioEmbedding`): clase propia que llama al endpoint REST `/embeddings` del servidor LM Studio mediante `requests`.
- **LLM vía `OpenAILike`**: en lugar del SDK nativo de LM Studio se usa el adaptador compatible con la API OpenAI de LlamaIndex.

---

## [1.0] — main.py

### Añadido
- Pipeline inicial: lectura de PDFs con `SimpleDirectoryReader`, indexación con `VectorStoreIndex` y chat con `as_chat_engine`.
- Integración con servidor LM Studio mediante el SDK nativo (`lmstudio`) sobre WebSocket (`ws://`).
- Clases `LMStudioEmbedding` y `LMStudioLLM` usando el SDK oficial de LM Studio.
- Ejecución en contenedor Docker con imagen CUDA (`nvidia/cuda:12.8.1`) y acceso GPU.
