# Changelog

Registro de cambios del proyecto **Agente IA G9** — Sistema RAG con LlamaIndex.

---

## [4.0] — main_lmstudio_v4.0.py / main_openai_v4.0.py

### Añadido
- **Sincronización de `Estado` con el índice documental** (`sincronizar_estado_con_indice`): recorre `indice.xlsx`, construye un mapa `Identificador → Estado` y aplica ese valor a todos los chunks del store con el mismo `Identificador`, dejando la persistencia coherente con el índice (Vigente / No vigente, en ambos sentidos). Sustituye al ejemplo inicial de un único documento con valores fijos. Solo persiste si hay cambios y avisa de nodos sin `Identificador` o con identificadores ausentes en la hoja. Helper `_acceder_nodos_docstore` para acceder a los nodos tolerando el atributo protegido `_docs`.
- **Bot de Telegram** (`python-telegram-bot`): front-end conversacional sobre el agente RAG.
  - `iniciar_bot`: crea la `Application`, registra los manejadores y arranca por `run_polling()`.
  - `comando_start`: mensaje de bienvenida con `reply_html` + `mention_html()`.
  - `manejar_mensaje`: atiende el texto que no es comando, comprueba la lista blanca y responde; a los usuarios no autorizados les indica que contacten con el administrador.
  - `error_handler`: registra en el log los errores de los `Update`.
  - `obtener_respuesta_ia`: atajos para saludos ("hola", "cómo estás", "adiós") y, en otro caso, delega en `chatear_simple`.
  - `chatear_simple`: chat engine con filtro `Estado=Vigente`, `top_k` y score mínimo; devuelve la respuesta + sección **Referencias** (URLs de las fuentes) y vuelca el detalle verboso (score, metadatos, texto parcial) al log para no exceder el límite de mensaje de Telegram.
- **Lista blanca por variable de entorno** (`TELEGRAM_WHITELIST`): `chat_id` autorizados con formato `id:nombre,id:nombre`, parseados por `_parse_lista_blanca`. Los identificadores reales viven solo en `.env`; `.env.example` incluye un ejemplo ficticio para no publicar datos personales.
- **Token del bot por entorno** (`TELEGRAM_TOKEN`): leído de `.env`, nunca versionado.
- **Logging estructurado** (`logging.basicConfig` + `logger`): trazas con timestamp para depurar el bot y las consultas.
- **Variante OpenAI** (`main_openai_v4.0.py`): versión paralela con todas las funcionalidades de v4.0 (sincronización de `Estado`, bot de Telegram, lista blanca, logging) usando la API oficial de OpenAI (`gpt-4o-mini` + `text-embedding-3-small`) en lugar del servidor LM Studio. Solo difiere en la integración del modelo; el resto del código es idéntico.
- **Presentación HTML** (`Presentaciones/presentacion_v4.html`): 19 diapositivas con tema claro centradas en las novedades de la v4.0 respecto a la v3.0 (paso de consola a bot de Telegram, handlers, lista blanca, `obtener_respuesta_ia`/`chatear_simple`, sincronización de `Estado`, logging, nuevo `main()` y variante OpenAI); navegable con teclado.

### Cambiado
- **`main()` arranca el bot**: mantiene el flujo diario (ingesta de nuevos documentos → construir/actualizar índice → sincronizar `Estado` → reporte) y, en lugar del chat por consola (`chatear`), lanza el bot de Telegram (`iniciar_bot`). El índice pasa a ser variable global para que lo consulten los manejadores.

### Corregido
- **Índice persistido separado por backend**: `CARPETA_DATOS_SALVADOS` pasa a `Datos/data_storage_lmstudio/` (variantes LM Studio) y `Datos/data_storage_openai/` (variantes OpenAI) en los cuatro scripts v3.0 y v4.0. Los embeddings de ambos backends tienen dimensiones distintas y no son intercambiables: con una carpeta común, un índice creado por un backend fallaba (o daba resultados incorrectos) al consultarse con el otro. `.gitignore` pasa a ignorar `Datos/data_storage*/`.

---

## [3.0] — main_lmstudio_v3.0.py / main_openai_v3.0.py

### Añadido (iteración 2)
- **Troceado semántico** (`MODO_TROCEADO = "semantico"`): usa `SemanticSplitterNodeParser` de LlamaIndex, que llama al modelo de embeddings para detectar cambios de tema y producir trozos más coherentes. La variable `MODO_TROCEADO` permite alternar entre `"fijo"` y `"semantico"` sin tocar el resto del código.
- **Parámetros del splitter semántico**: `BUFFER_SIZE_SEMANTICO` y `BREAKPOINT_PERCENTILE_SEMANTICO` expuestos en la sección de configuración.
- **Filtrado de trozos cortos** (`_filtrar_trozos_cortos`): trozos por debajo de `UMBRAL_VISUALIZACION_PALABRAS` (30 palabras) se eliminan del índice y se muestran por pantalla. El último trozo de cada documento siempre se conserva.
- **Persistencia del índice** (`CARPETA_DATOS_SALVADOS`): el índice vectorial se guarda en `Datos/data_storage/` usando `StorageContext`. En ejecuciones posteriores se carga desde disco sin recalcular embeddings.
- **Función `_indice_existe()`**: comprueba si `docstore.json` está presente para decidir si cargar o construir.
- **Función `actualizar_indice()`**: añade nuevos nodos a un índice existente y repersiste, permitiendo un flujo incremental.
- **Carpeta `Documentos/`** y constante `CARPETA_DOCUMENTOS_PROCESADOS`: cada documento procesado se mueve automáticamente de `Nuevos documentos/` a `Documentos/` con `shutil.move`. Los documentos sin registro en `indice.xlsx` permanecen en la carpeta de entrada como aviso.
- **Flujo diario completo en `main()`**: cubre 5 escenarios (construir / cargar+actualizar / cargar existente / salir) según el estado de la carpeta de entrada y del índice persistido.
- **Reporte del índice** (`mostrar_reporte_indice`): tabla impresa antes de cada chat con columnas Documento / Trozos / Palabras / ~Tokens y fila de totales.
- **Filtro por estado** (`FILTRO_ESTADO = "Vigente"`): el retriever usa `MetadataFilters` + `ExactMatchFilter` para excluir documentos marcados como no vigentes.
- **Control del retriever**: `RETRIEVER_TOP_K = 4` limita el número máximo de nodos recuperados; `SimilarityPostprocessor(similarity_cutoff=0.40)` descarta nodos poco relevantes.
- **Timeout configurable** (`LMSTUDIO_TIMEOUT = 600.0`): aplicado tanto a las llamadas HTTP de embeddings como al cliente `OpenAILike`.
- **Presentación HTML** (`Presentaciones/presentacion_v3.html`): 16 diapositivas con tema claro que explican el pipeline, la configuración y ejemplos de ejecución reales; navegable con teclado.
- **Gestión de credenciales con `.env`**: `Aplicaciones/.env` (ignorado por git) y `Aplicaciones/.env.example` (plantilla versionada); `docker-compose.yml` carga las claves automáticamente con `env_file`.

### Añadido (iteración 1)
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
- **Índice vectorial en memoria → persistente**: el índice se construye en memoria la primera vez y se persiste en disco; las ejecuciones siguientes lo cargan directamente.
- **Inserción nodo a nodo con log**: `construir_indice` y `actualizar_indice` insertan cada `TextNode` individualmente mostrando su ID, número de palabras y tokens estimados.
- **Estimación de tokens**: función `_estimar_tokens` (~4 caracteres/token) para mostrar el coste aproximado de cada chunk durante la indexación.
- **Credenciales por variable de entorno**: `LMSTUDIO_APITOKEN` y `OPENAI_API_KEY` se leen con `os.getenv`; en local se usa el fichero `.env`.

### Infraestructura
- **Imagen Docker**: cambiada la base de `nvidia/cuda:12.8.1-runtime-ubuntu22.04` (~3 GB) a `python:3.11-slim` (~200 MB). Toda la inferencia es externa, por lo que no se necesita GPU en el contenedor.
- **Dependencias reducidas**: eliminados `torch`, `torchvision`, `torchaudio`, `transformers`, `sentence-transformers` y otros paquetes ML. Añadidos `llama-index-llms-openai`, `llama-index-llms-openai-like`.
- **`container_name` fijo**: el contenedor siempre se llama `agente_ia_g9` independientemente de la carpeta desde la que se ejecute `docker compose`.
- **Volumen adaptado**: la ruta del volumen en `docker-compose.yml` apunta al repositorio real en lugar de `c:\G9-IA\`.
- **`build` integrado en compose**: el `docker-compose.yml` incluye la directiva `build` para poder hacer `up --build` sin ejecutar `docker build` por separado.

---

## [1.0] — main.py

### Añadido
- Pipeline inicial: lectura de PDFs con `SimpleDirectoryReader`, indexación con `VectorStoreIndex` y chat con `as_chat_engine`.
- Integración con servidor LM Studio mediante el SDK nativo (`lmstudio`) sobre WebSocket (`ws://`).
- Clases `LMStudioEmbedding` y `LMStudioLLM` usando el SDK oficial de LM Studio.
- Ejecución en contenedor Docker con imagen CUDA (`nvidia/cuda:12.8.1`) y acceso GPU.
