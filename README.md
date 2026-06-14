# Agente IA G9 — Sistema RAG con LlamaIndex

Sistema de preguntas y respuestas sobre documentación interna basado en **RAG** (Retrieval-Augmented Generation). Ingiere documentos (PDF, DokuWiki y Markdown), los trocea y los indexa en un **vector store persistente** con LlamaIndex, y responde preguntas mediante un LLM citando las fuentes consultadas. La versión actual (**v4.0**) expone el agente a través de un **bot de Telegram** con lista blanca de usuarios.

Toda la inferencia (embeddings + LLM) es **externa**: se delega en un servidor LM Studio compatible con OpenAI o en la API oficial de OpenAI. El contenedor no necesita GPU.

---

## Tabla de contenidos

- [Estructura del proyecto](#estructura-del-proyecto)
- [Versiones de los scripts](#versiones-de-los-scripts)
- [Funcionalidades por fichero](#funcionalidades-por-fichero)
- [Arquitectura y flujo de datos](#arquitectura-y-flujo-de-datos)
- [El índice documental `indice.xlsx`](#el-índice-documental-indicexlsx)
- [Carpetas de datos](#carpetas-de-datos)
- [Configuración de credenciales (`.env`)](#configuración-de-credenciales-env)
- [Bot de Telegram](#bot-de-telegram)
- [Instrucciones Docker](#instrucciones-docker)
- [Detener y eliminar el contenedor](#detener-y-eliminar-el-contenedor)
- [Dependencias principales](#dependencias-principales)

---

## Estructura del proyecto

```
agente_ia_g9/
├── Aplicaciones/
│   ├── Dockerfile                 # Imagen Docker ligera (python:3.11-slim, sin GPU)
│   ├── docker-compose.yml         # Servicio del contenedor; monta el repo en /app y carga .env
│   ├── .env.example               # Plantilla de credenciales (versionada)
│   ├── .env                       # Credenciales reales (IGNORADO por git)
│   ├── main_lmstudio.py           # v1 — pipeline básico, índice en memoria (LM Studio)
│   ├── main_openai.py             # v1 — pipeline básico, índice en memoria (OpenAI)
│   ├── main_lmstudio_v3.0.py      # v3 — persistencia + multiformato + semántico + filtros (LM Studio)
│   ├── main_openai_v3.0.py        # v3 — idéntico con la API de OpenAI
│   ├── main_lmstudio_v4.0.py      # v4 — bot de Telegram + sincronización de Estado (LM Studio)
│   └── main_openai_v4.0.py        # v4 — idéntico con la API de OpenAI
├── Datos/
│   ├── indice.xlsx                # Registro y metadatos de los documentos (hoja «Hoja1»)
│   └── data_storage/              # Índice vectorial persistido (IGNORADO; se crea al ejecutar)
├── Documentos/                    # Archivo de documentos ya procesados (IGNORADO salvo .gitkeep)
├── Nuevos documentos/             # Drop zone: documentos pendientes de indexar
├── Presentaciones/
│   └── presentacion_v3.html       # Presentación didáctica del código v3.0 (16 diapositivas)
├── CHANGELOG.md                   # Historial de cambios por versión
├── CLAUDE.md                      # Guía para Claude Code (IGNORADO por git)
└── README.md
```

> Los ficheros marcados como **IGNORADO** no se publican en GitHub (ver `.gitignore`). Al clonar el repositorio no aparecerán: `.env`, `CLAUDE.md`, el contenido de `Datos/data_storage/` y de `Documentos/`, y los `__pycache__/`.

---

## Versiones de los scripts

El proyecto evoluciona en tres generaciones. Cada una existe en dos variantes que comparten **exactamente** el mismo código salvo la integración del modelo: `main_lmstudio_*` (servidor LM Studio) y `main_openai_*` (API de OpenAI).

| Versión | Fichero | Novedades principales |
|---|---|---|
| **v1** | `main_lmstudio.py` / `main_openai.py` | Pipeline mínimo: lee PDFs, los trocea por palabras con solapamiento, construye un índice **en memoria** y responde preguntas por consola. Sin persistencia. |
| **v3.0** | `main_lmstudio_v3.0.py` / `main_openai_v3.0.py` | Refactor en funciones; **persistencia** del índice en disco; ingesta **multiformato** (PDF + DokuWiki + Markdown); **troceado semántico** opcional; **filtrado de trozos cortos**; **reporte** del índice; **filtro por `Estado`** y **score mínimo** en el retriever; flujo diario con archivado de documentos. |
| **v4.0** | `main_lmstudio_v4.0.py` / `main_openai_v4.0.py` | **Bot de Telegram** como front-end con **lista blanca** de usuarios; **sincronización del metadato `Estado`** de los chunks con `indice.xlsx`; logging estructurado. Es la versión recomendada. |

### LM Studio vs OpenAI

| | `main_lmstudio_v4.0.py` | `main_openai_v4.0.py` |
|---|---|---|
| Embeddings | clase propia `LMStudioEmbedding` (REST) | `OpenAIEmbedding` |
| LLM | `OpenAILike` | `OpenAI` |
| Credencial | `LMSTUDIO_APITOKEN` | `OPENAI_API_KEY` |
| Servidor | `http://openai.ull.es:8080/v1` | API de OpenAI |
| LLM por defecto | `qwen/qwen3.5-9b` | `gpt-4o-mini` |
| Embeddings por defecto | `text-embedding-qwen3-embedding-0.6b` | `text-embedding-3-small` |

---

## Funcionalidades por fichero

### `main_lmstudio_v4.0.py` / `main_openai_v4.0.py` (versión actual)

Estructurados en bloques claros. Funciones destacadas:

**Configuración e integración del modelo**
- `LMStudioEmbedding` *(solo LM Studio)* — clase `BaseEmbedding` que llama al endpoint REST `/embeddings` del servidor con `requests`, reintentos y timeout configurable.
- `configurar_modelos()` — registra el modelo de embeddings y el LLM en `Settings` de LlamaIndex. La variante OpenAI valida que `OPENAI_API_KEY` exista.
- `_parse_lista_blanca()` — convierte la cadena `TELEGRAM_WHITELIST` (`id:nombre,id:nombre`) en el diccionario de usuarios autorizados.

**Ingesta y troceado**
- `leer_archivos_carpeta()` — lista los ficheros de `Nuevos documentos/` con sus metadatos.
- `digitaliza_un_documento()` — **despachador** que delega según la extensión en:
  - `_digitaliza_pdf()` — extrae texto con `pypdf` página a página.
  - `_digitaliza_dokuwiki()` / `_digitaliza_markdown()` → `_digitaliza_con_pandoc()` — convierten a texto plano con `pypandoc`.
- `dividir_con_solapamiento()` — troceado **fijo** por número de palabras con solapamiento.
- `_get_semantic_splitter()` — troceado **semántico** con `SemanticSplitterNodeParser` (usa el modelo de embeddings para detectar cambios de tema). La variable `MODO_TROCEADO` alterna `"fijo"` / `"semantico"`.
- `_filtrar_trozos_cortos()` — elimina los trozos por debajo de `UMBRAL_VISUALIZACION_PALABRAS` (salvo el último de cada documento) y los muestra por pantalla.
- `digitaliza_documentos()` — orquesta todo, genera los `TextNode` y **mueve** cada documento procesado a `Documentos/`.

**Índice vectorial (persistencia)**
- `construir_indice()` / `actualizar_indice()` — crean o amplían el índice insertando nodo a nodo y lo persisten en `Datos/data_storage/`.
- `cargar_indice()` / `_indice_existe()` — recargan el índice desde disco sin recalcular embeddings.
- `mostrar_reporte_indice()` — tabla por documento con trozos, palabras y tokens estimados.

**Sincronización de metadatos**
- `sincronizar_estado_con_indice()` — recorre `indice.xlsx`, construye el mapa `Identificador → Estado` y actualiza el metadato `Estado` de **todos** los chunks para que el store quede coherente con el índice documental (Vigente / No vigente, en ambos sentidos). Persiste solo si hay cambios.
- `_acceder_nodos_docstore()` — helper que accede a los nodos tolerando el atributo protegido `_docs`.

**Consulta RAG**
- `chatear_simple()` — crea el chat engine con filtro `Estado=Vigente`, `RETRIEVER_TOP_K` y `SimilarityPostprocessor` (score mínimo); devuelve la respuesta + sección **Referencias** (URLs) y vuelca el detalle (score, metadatos) al log.
- `obtener_respuesta_ia()` — atajos para saludos y, en otro caso, delega en `chatear_simple()`.

**Bot de Telegram**
- `comando_start()` — bienvenida al `/start`.
- `manejar_mensaje()` — atiende mensajes de texto; comprueba la lista blanca; responde a autorizados y rechaza al resto.
- `error_handler()` — registra errores en el log.
- `iniciar_bot()` — monta la `Application`, registra los manejadores y arranca por *polling*.

**Punto de entrada**
- `main()` — ejecuta el **flujo diario** (ingesta → construir/actualizar índice → sincronizar `Estado` → reporte) y después **arranca el bot**.

### `main_lmstudio_v3.0.py` / `main_openai_v3.0.py`

Misma base de ingesta, persistencia y retriever que v4.0, **pero sin bot ni sincronización de `Estado`**: la consulta se hace por consola con la función `chatear()` sobre una lista de preguntas fija en `main()`.

### `main_lmstudio.py` / `main_openai.py` (v1)

Versión inicial y didáctica: solo PDF, troceado por palabras, índice **en memoria** (no persiste) y chat por consola. Útil para entender el flujo mínimo de un RAG.

---

## Arquitectura y flujo de datos

1. **Registro** — se añade una fila al documento en `Datos/indice.xlsx` con sus metadatos.
2. **Drop** — se coloca el fichero (`.pdf`, `.dokuwiki`, `.md`) en `Nuevos documentos/`.
3. **Ingesta** — `main()` cruza cada fichero con el índice, extrae el texto, lo trocea (fijo o semántico) y lo enriquece con metadatos; los documentos sin fila en el índice se omiten.
4. **Indexado** — cada trozo se convierte en un `TextNode` con un `IdTrozoTexto` único, se embebe y se inserta en el `VectorStoreIndex`, que se persiste en `Datos/data_storage/`.
5. **Archivado** — los documentos procesados se mueven a `Documentos/`.
6. **Sincronización** *(v4.0)* — se alinea el metadato `Estado` de los chunks con el índice documental.
7. **Consulta** — el retriever recupera los trozos `Vigente` más relevantes (top-k + score mínimo) y el LLM responde citando fuentes. En v4.0 esto ocurre vía el bot de Telegram.

---

## El índice documental `indice.xlsx`

Antes de colocar un documento en `Nuevos documentos/` debe existir una fila para él en `Datos/indice.xlsx` (hoja **`Hoja1`**). Columnas:

| Columna | Descripción |
|---|---|
| `Identificador` | ID único del documento; se usa como prefijo de los `IdTrozoTexto` y para la sincronización de `Estado`. |
| `Descripción` | Descripción libre del documento. |
| `Nombre Archivo` | Nombre **exacto** del fichero, con extensión (p. ej. `informe.pdf`). |
| `URL` | Enlace a la fuente original; se muestra en las **Referencias** de las respuestas. |
| `Fecha de publicación` | Fecha de publicación del documento. |
| `Estado` | `Vigente` / `No vigente`. El retriever solo usa los chunks `Vigente`. |
| `Fecha de fin de vigencia` | Fecha hasta la que el documento es válido. |

> Los documentos no registrados en el índice se omiten durante la indexación y permanecen en `Nuevos documentos/` como aviso.

---

## Carpetas de datos

| Carpeta | Contenido | Versionada |
|---|---|---|
| `Nuevos documentos/` | Documentos pendientes de indexar (drop zone). | Sí |
| `Documentos/` | Documentos ya procesados (movidos automáticamente). | No (solo `.gitkeep`) |
| `Datos/data_storage/` | Índice vectorial persistido (`docstore.json`, `default__vector_store.json`, etc.). | No (se crea al ejecutar) |

---

## Configuración de credenciales (`.env`)

Las claves y tokens se gestionan con un fichero `.env` que **nunca se sube al repositorio**. Copiar la plantilla y rellenar los valores reales:

```bash
cp Aplicaciones/.env.example Aplicaciones/.env
```

Variables disponibles en `Aplicaciones/.env`:

```env
LMSTUDIO_APITOKEN=tu-token-lmstudio-aqui
OPENAI_API_KEY=sk-proj-tu-clave-openai-aqui
TELEGRAM_TOKEN=tu-token-telegram-aqui
TELEGRAM_WHITELIST=123456789:Nombre Ejemplo
```

| Variable | Para qué | La usa |
|---|---|---|
| `LMSTUDIO_APITOKEN` | Token del servidor LM Studio (`http://openai.ull.es:8080/v1`). | Variantes `main_lmstudio_*` |
| `OPENAI_API_KEY` | Clave de [platform.openai.com/api-keys](https://platform.openai.com/api-keys). | Variantes `main_openai_*` |
| `TELEGRAM_TOKEN` | Token del bot devuelto por BotFather. | Bot (v4.0) |
| `TELEGRAM_WHITELIST` | Usuarios autorizados, formato `id:nombre,id:nombre`. | Bot (v4.0) |

El `docker-compose.yml` carga este fichero automáticamente (`env_file: .env`), por lo que no hay que tocar el código ni exportar variables a mano.

---

## Bot de Telegram

La versión v4.0 atiende las consultas a través de un bot de Telegram.

1. **Crear el bot**: hablar con [`@BotFather`](https://t.me/BotFather), comando `/newbot`, elegir un nombre y un *username* terminado en `bot`. BotFather devuelve el **token** → ponerlo en `TELEGRAM_TOKEN`.
2. **Autorizar usuarios**: añadir cada `chat_id` autorizado a `TELEGRAM_WHITELIST`. Para conocer un `chat_id`, escribe al bot y míralo en el log (`Mensaje recibido de <chat_id>`).
3. **Uso**: `/start` saluda; cualquier otro mensaje de texto se responde con el agente RAG (solo a usuarios de la lista blanca). Cada respuesta incluye un bloque **Referencias** con las URLs de las fuentes.

---

## Instrucciones Docker

La aplicación está pensada para ejecutarse en un contenedor Docker que monta el repositorio en `/app`.

### 1. Clonar el repositorio

```bash
git clone https://github.com/jgonzal-ull/agente_ia_g9.git
cd agente_ia_g9
```

### 2. Ajustar la ruta del volumen

Editar `Aplicaciones/docker-compose.yml` y apuntar el volumen a la ruta local del repositorio:

```yaml
volumes:
  - /ruta/local/agente_ia_g9:/app   # <-- cambiar por tu ruta
```

En Windows:

```yaml
volumes:
  - C:\Users\usuario\agente_ia_g9\:/app
```

### 3. Configurar las credenciales

Crear `Aplicaciones/.env` a partir de la plantilla (ver [Configuración de credenciales](#configuración-de-credenciales-env)).

### 4. Construir la imagen e iniciar el contenedor

```bash
docker compose -f Aplicaciones/docker-compose.yml up -d --build
```

Construye la imagen `agente_ia_g9` a partir del `Dockerfile` e inicia el contenedor (que queda a la espera con un shell).

### 5. Ejecutar el agente

Verificar que el backend elegido está disponible (servidor LM Studio activo con los modelos cargados, o clave de OpenAI válida) y lanzar la versión deseada:

```bash
# Versión actual con bot de Telegram (LM Studio)
docker exec -it agente_ia_g9 python3 /app/Aplicaciones/main_lmstudio_v4.0.py

# Versión actual con bot de Telegram (OpenAI)
docker exec -it agente_ia_g9 python3 /app/Aplicaciones/main_openai_v4.0.py
```

> Las versiones v3.0 (`main_*_v3.0.py`) ejecutan el mismo pipeline pero responden por consola a la lista de preguntas definida al final de `main()`, útil para pruebas sin Telegram.

### Obtener el nombre exacto del contenedor

```bash
docker ps
```

---

## Detener y eliminar el contenedor

```bash
# Parar y eliminar el contenedor (conserva la imagen)
docker compose -f Aplicaciones/docker-compose.yml down

# Eliminar también la imagen
docker rmi agente_ia_g9

# Eliminar todo (contenedor, imagen y volúmenes anónimos)
docker compose -f Aplicaciones/docker-compose.yml down --rmi all
```

---

## Dependencias principales

| Paquete | Uso |
|---|---|
| `llama-index` | Framework RAG (índice vectorial, chat engine, splitters). |
| `llama-index-llms-openai-like` | LLM para LM Studio (`OpenAILike`). |
| `llama-index-llms-openai` | LLM para OpenAI. |
| `llama-index-embeddings-openai` | Embeddings para OpenAI. |
| `python-telegram-bot` | Bot de Telegram (v4.0). |
| `pypdf` | Extracción de texto de PDFs. |
| `pypandoc` / `pypandoc_binary` | Conversión de DokuWiki y Markdown a texto plano. |
| `pandas` / `openpyxl` | Lectura del índice documental `indice.xlsx`. |
| `requests` | Llamadas HTTP al servidor LM Studio. |

Todas se instalan en la imagen mediante el `Dockerfile`.

---

## Historial de cambios

El detalle de cada versión está en [`CHANGELOG.md`](CHANGELOG.md). Material didáctico del código v3.0 en [`Presentaciones/presentacion_v3.html`](Presentaciones/presentacion_v3.html).
