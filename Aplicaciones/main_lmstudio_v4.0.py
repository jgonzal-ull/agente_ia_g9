"""
Agente RAG con LlamaIndex + LM Studio (servidor externo OpenAI-compatible).
Pipeline: leer documentos → digitalizar → indexar → chatear.
v4.0: añade actualización de metadatos en el store persistido.
"""

import asyncio
import datetime
import logging
import os
import shutil
import time
from collections import defaultdict
from typing import Any, List

import pandas as pd
import pypandoc
import requests

from pypdf import PdfReader

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.llms.openai_like import OpenAILike

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# ── Configuración ──────────────────────────────────────────────────────────────

# Rutas
FICHERO_INDICE                = "/app/Datos/indice.xlsx"       # registro de documentos
CARPETA_ENTRADA_DOCUMENTOS    = "/app/Nuevos documentos/"      # documentos pendientes de indexar
CARPETA_DOCUMENTOS_PROCESADOS = "/app/Documentos/"             # archivo de documentos ya procesados
CARPETA_DATOS_SALVADOS        = "/app/Datos/data_storage_lmstudio/"  # índice vectorial (propio del backend LM Studio)

# Servidor LM Studio
SERVIDOR_LMSTUDIO = "http://openai.ull.es:8080/v1"
LMSTUDIO_APITOKEN = os.getenv("LMSTUDIO_APITOKEN", "")  # o sustituir por el token directamente
LMSTUDIO_TIMEOUT  = 600.0                               # segundos de espera máxima por llamada
MODELO_LLM        = "qwen/qwen3.5-9b"
MODELO_EMBEDDINGS = "text-embedding-qwen3-embedding-0.6b"

# Bot de Telegram (token obtenido de BotFather; se lee del entorno, nunca se sube a git)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")


def _parse_lista_blanca(raw: str) -> dict:
    """Convierte la cadena 'id:nombre,id:nombre' en el diccionario {int(id): nombre}."""
    autorizados: dict = {}
    for entrada in raw.split(","):
        entrada = entrada.strip()
        if not entrada:
            continue
        id_str, _, nombre = entrada.partition(":")
        try:
            autorizados[int(id_str.strip())] = nombre.strip()
        except ValueError:
            print(f"Aviso: entrada de TELEGRAM_WHITELIST ignorada (id no numérico): '{entrada}'")
    return autorizados


# Usuarios autorizados del bot. Los chat_id reales se leen de TELEGRAM_WHITELIST (.env),
# con formato "id:nombre,id:nombre", para no publicarlos en el repositorio.
# Para conocer un chat_id, escribe al bot y míralo en el log "Mensaje recibido de <chat_id>".
lista_blanca = _parse_lista_blanca(os.getenv("TELEGRAM_WHITELIST", ""))

# Retriever y filtrado en el chat
FILTRO_ESTADO          = "Vigente"  # solo se recuperan trozos cuyo campo "Estado" tenga este valor
RETRIEVER_TOP_K        = 4          # número máximo de nodos recuperados por consulta
RETRIEVER_SCORE_MINIMO = 0.40       # score mínimo de similitud; nodos por debajo se descartan

# Chunking de documentos
MODO_TROCEADO                 = "semantico"  # "fijo" | "semantico"
UMBRAL_MAXIMO_PALABRAS        = 3000    # (fijo) máx. palabras por trozo
PALABRAS_SOLAPAMIENTO         = 50      # (fijo) palabras compartidas entre trozos consecutivos
NUMERO_MINIMO_PALABRAS        = 0       # (fijo) trozos con menos palabras se descartan
UMBRAL_VISUALIZACION_PALABRAS = 30      # trozos con menos palabras se eliminan (salvo el último)

# Parámetros del troceado semántico (solo cuando MODO_TROCEADO = "semantico")
BUFFER_SIZE_SEMANTICO           = 1   # frases de contexto a cada lado del corte
BREAKPOINT_PERCENTILE_SEMANTICO = 95  # percentil para detectar ruptura semántica (0-100)


# ── Logging ──────────────────────────────────────────────────────────────────
# Configuramos el logging para ver qué está pasando (opcional pero útil para el bot).
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Índice global: lo construye/carga main() y lo consultan los manejadores del bot.
index: Any = None


# ── Integración LM Studio ↔ LlamaIndex ────────────────────────────────────────

class LMStudioEmbedding(BaseEmbedding):
    """Embedding personalizado que llama al endpoint REST de LM Studio."""

    _api_base:   str               = PrivateAttr()
    _model_name: str               = PrivateAttr()
    _session:    requests.Session  = PrivateAttr()

    def __init__(self, model_name: str, api_base: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name
        self._api_base   = api_base.rstrip("/")
        self._session    = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LMSTUDIO_APITOKEN}",
        })

    def _embed(self, text: str) -> List[float]:
        if not text.strip():
            return []
        payload  = {"model": self._model_name, "input": text.strip()}
        response = self._session.post(
            f"{self._api_base}/embeddings", json=payload, timeout=LMSTUDIO_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        results = []
        for t in texts:
            results.append(self._embed(t))
            time.sleep(0.05)
        return results

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.get_event_loop().run_in_executor(None, self._embed, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.get_event_loop().run_in_executor(None, self._embed, text)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def configurar_modelos() -> None:
    """Registra el modelo de embeddings y el LLM en la configuración global de LlamaIndex."""
    Settings.embed_model = LMStudioEmbedding(
        model_name=MODELO_EMBEDDINGS, api_base=SERVIDOR_LMSTUDIO
    )
    Settings.llm = OpenAILike(
        model=MODELO_LLM,
        api_base=SERVIDOR_LMSTUDIO,
        api_key=LMSTUDIO_APITOKEN,
        is_chat_model=True,
        timeout=LMSTUDIO_TIMEOUT,
    )
    print(f"Servidor : {SERVIDOR_LMSTUDIO}")
    print(f"LLM      : {MODELO_LLM}")
    print(f"Embedding: {MODELO_EMBEDDINGS}\n")


def leer_archivos_carpeta(ruta: str) -> list:
    """Devuelve una lista de dicts con metadatos de cada archivo encontrado en ruta."""
    try:
        nombres = os.listdir(ruta)
    except FileNotFoundError:
        print(f"Carpeta no encontrada: '{ruta}'")
        return []

    archivos = []
    for nombre in nombres:
        ruta_completa = os.path.abspath(os.path.join(ruta, nombre))
        if not os.path.isfile(ruta_completa):
            continue
        nombre_base, ext = os.path.splitext(nombre)
        archivos.append({
            "carpeta":               os.path.dirname(ruta_completa),
            "nombre":                nombre_base,
            "extension":             ext.lstrip("."),
            "nombre_con_extension":  nombre,
            "ruta_completa":         ruta_completa,
        })

    print(f"Archivos encontrados en '{ruta}': {len(archivos)}")
    return archivos


def dividir_con_solapamiento(texto: str, max_palabras: int, solapamiento: int) -> List[str]:
    """Divide texto en trozos de max_palabras palabras con solapamiento entre ellos."""
    palabras = texto.split()
    if not palabras:
        return []
    paso = max_palabras - solapamiento
    return [
        " ".join(palabras[i : i + max_palabras])
        for i in range(0, len(palabras), paso)
    ]


def _get_semantic_splitter() -> SemanticSplitterNodeParser:
    """Crea un SemanticSplitterNodeParser usando el modelo de embeddings activo."""
    return SemanticSplitterNodeParser(
        embed_model=Settings.embed_model,
        buffer_size=BUFFER_SIZE_SEMANTICO,
        breakpoint_percentile_threshold=BREAKPOINT_PERCENTILE_SEMANTICO,
    )


def _filtrar_trozos_cortos(resultado: list) -> list:
    """Elimina trozos por debajo del umbral de palabras, conservando siempre el último del documento."""
    if len(resultado) <= 1:
        return resultado
    filtrados = []
    for i, item in enumerate(resultado):
        n_palabras = len(item["texto"].split())
        es_ultimo = (i == len(resultado) - 1)
        if n_palabras < UMBRAL_VISUALIZACION_PALABRAS and not es_ultimo:
            print(f"  [TROZO CORTO ELIMINADO] {item['metadatos']['IdTrozoTexto']} "
                  f"({n_palabras} palabras): {item['texto']!r}")
            continue
        filtrados.append(item)
    return filtrados


def _digitaliza_pdf(documento: dict, fila_indice: Any) -> list:
    """Extrae texto del PDF y genera trozos con metadatos (modo fijo o semántico)."""
    resultado = []
    try:
        pdf = PdfReader(documento["ruta_completa"])

        if MODO_TROCEADO == "semantico":
            texto_completo = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
            if not texto_completo:
                return []
            nodos = _get_semantic_splitter().get_nodes_from_documents(
                [Document(text=texto_completo)]
            )
            for indice_trozo, nodo in enumerate(nodos):
                trozo = nodo.text
                metadatos = fila_indice.to_dict()
                metadatos.update({
                    "file_name":              documento["nombre_con_extension"],
                    "file_path":              documento["ruta_completa"],
                    "LongitudTexto":          len(trozo),
                    "NumeroPalabras":         len(trozo.split()),
                    "NumeroTrozoEnDocumento": indice_trozo,
                    "IdTrozoTexto":           f"{fila_indice['Identificador']}-{indice_trozo}",
                })
                resultado.append({"texto": trozo, "metadatos": metadatos})
        else:
            indice_trozo = 0
            for num_pagina, page in enumerate(pdf.pages):
                texto_pagina = page.extract_text() or ""
                if not texto_pagina.strip():
                    continue
                if len(texto_pagina.split()) < NUMERO_MINIMO_PALABRAS:
                    continue
                for trozo in dividir_con_solapamiento(
                    texto_pagina, UMBRAL_MAXIMO_PALABRAS, PALABRAS_SOLAPAMIENTO
                ):
                    metadatos = fila_indice.to_dict()
                    metadatos.update({
                        "file_name":              documento["nombre_con_extension"],
                        "file_path":              documento["ruta_completa"],
                        "page_label":             str(num_pagina + 1),
                        "LongitudTexto":          len(trozo),
                        "NumeroPalabras":         len(trozo.split()),
                        "NumeroTrozoEnDocumento": indice_trozo,
                        "IdTrozoTexto":           f"{fila_indice['Identificador']}-{indice_trozo}",
                    })
                    resultado.append({"texto": trozo, "metadatos": metadatos})
                    indice_trozo += 1
    except Exception as e:
        print(f"  Error procesando {documento['nombre_con_extension']}: {e}")
    return _filtrar_trozos_cortos(resultado)


def _digitaliza_con_pandoc(documento: dict, fila_indice: Any, formato: str) -> list:
    """Convierte un fichero de texto con pandoc a texto plano y genera trozos con metadatos (modo fijo o semántico)."""
    resultado = []
    try:
        with open(documento["ruta_completa"], "r", encoding="utf-8") as f:
            contenido = f.read()
        texto_plano = pypandoc.convert_text(
            contenido, "plain", format=formato, extra_args=["--wrap=none"]
        )
        if not texto_plano.strip():
            return []

        if MODO_TROCEADO == "semantico":
            nodos = _get_semantic_splitter().get_nodes_from_documents(
                [Document(text=texto_plano)]
            )
            for indice_trozo, nodo in enumerate(nodos):
                trozo = nodo.text
                metadatos = fila_indice.to_dict()
                metadatos.update({
                    "file_name":              documento["nombre_con_extension"],
                    "file_path":              documento["ruta_completa"],
                    "LongitudTexto":          len(trozo),
                    "NumeroPalabras":         len(trozo.split()),
                    "NumeroTrozoEnDocumento": indice_trozo,
                    "IdTrozoTexto":           f"{fila_indice['Identificador']}-{indice_trozo}",
                })
                resultado.append({"texto": trozo, "metadatos": metadatos})
        else:
            indice_trozo = 0
            for trozo in dividir_con_solapamiento(
                texto_plano, UMBRAL_MAXIMO_PALABRAS, PALABRAS_SOLAPAMIENTO
            ):
                if len(trozo.split()) < NUMERO_MINIMO_PALABRAS:
                    continue
                metadatos = fila_indice.to_dict()
                metadatos.update({
                    "file_name":              documento["nombre_con_extension"],
                    "file_path":              documento["ruta_completa"],
                    "LongitudTexto":          len(trozo),
                    "NumeroPalabras":         len(trozo.split()),
                    "NumeroTrozoEnDocumento": indice_trozo,
                    "IdTrozoTexto":           f"{fila_indice['Identificador']}-{indice_trozo}",
                })
                resultado.append({"texto": trozo, "metadatos": metadatos})
                indice_trozo += 1
    except Exception as e:
        print(f"  Error procesando {documento['nombre_con_extension']}: {e}")
    return _filtrar_trozos_cortos(resultado)


def _digitaliza_dokuwiki(documento: dict, fila_indice: Any) -> list:
    """Procesa un fichero DokuWiki usando el parser pandoc correspondiente."""
    return _digitaliza_con_pandoc(documento, fila_indice, "dokuwiki")


def _digitaliza_markdown(documento: dict, fila_indice: Any) -> list:
    """Procesa un fichero Markdown usando el parser pandoc correspondiente."""
    return _digitaliza_con_pandoc(documento, fila_indice, "markdown")


def digitaliza_un_documento(documento: dict, fila_indice: Any) -> list:
    """Despacha el procesamiento al parser adecuado según la extensión del fichero."""
    ext = documento["extension"].lower()
    if ext == "pdf":
        print(f"  Digitalizando (PDF)      : {documento['nombre_con_extension']}")
        return _digitaliza_pdf(documento, fila_indice)
    elif ext == "dokuwiki":
        print(f"  Digitalizando (DokuWiki) : {documento['nombre_con_extension']}")
        return _digitaliza_dokuwiki(documento, fila_indice)
    elif ext == "md":
        print(f"  Digitalizando (Markdown) : {documento['nombre_con_extension']}")
        return _digitaliza_markdown(documento, fila_indice)
    else:
        print(f"  Extensión no soportada   : '{ext}' ({documento['nombre_con_extension']})")
        return []


def digitaliza_documentos(listado: list, df_indice: Any) -> List[TextNode]:
    """Procesa la lista de documentos, mueve cada uno a Documentos/ y devuelve los TextNodes."""
    os.makedirs(CARPETA_DOCUMENTOS_PROCESADOS, exist_ok=True)
    nodes: List[TextNode] = []

    for doc in listado:
        nombre = doc["nombre_con_extension"]
        filas = df_indice[df_indice["Nombre Archivo"] == nombre]
        if filas.empty:
            print(f"  '{nombre}' no aparece en el índice — se omite (permanece en Nuevos documentos).")
            continue

        for i, data in enumerate(digitaliza_un_documento(doc, filas.iloc[0])):
            metadatos = {
                k: v.isoformat() if isinstance(v, (datetime.datetime, pd.Timestamp)) else v
                for k, v in data["metadatos"].items()
            }
            nodes.append(
                TextNode(
                    text=data["texto"],
                    metadata=metadatos,
                    id_=metadatos.get("IdTrozoTexto", f"node_{i}"),
                )
            )

        destino = os.path.join(CARPETA_DOCUMENTOS_PROCESADOS, nombre)
        try:
            shutil.move(doc["ruta_completa"], destino)
            print(f"  Movido → Documentos/: {nombre}")
        except Exception as e:
            print(f"  Error al mover {nombre}: {e}")

    print(f"\nNodos generados: {len(nodes)}\n")
    return nodes


def _estimar_tokens(texto: str) -> int:
    """Estima el número de tokens de un texto (~4 caracteres por token)."""
    return len(texto) // 4


def _insertar_nodos(index: VectorStoreIndex, nodes: List[TextNode]) -> None:
    """Inserta nodos en el índice uno a uno mostrando progreso por pantalla."""
    total = len(nodes)
    for i, node in enumerate(nodes, 1):
        palabras = node.metadata.get("NumeroPalabras", len(node.text.split()))
        tokens   = _estimar_tokens(node.text)
        id_trozo = node.metadata.get("IdTrozoTexto", node.node_id)
        print(f"  [{i:>4}/{total}] {id_trozo} — {palabras} palabras, ~{tokens} tokens")
        index.insert_nodes([node])


def construir_indice(nodes: List[TextNode]) -> VectorStoreIndex:
    """Crea un VectorStoreIndex vacío, inserta los nodos y lo persiste en disco."""
    print(f"Creando índice ({len(nodes)} nodos)...")
    index = VectorStoreIndex(nodes=[])
    _insertar_nodos(index, nodes)
    os.makedirs(CARPETA_DATOS_SALVADOS, exist_ok=True)
    index.storage_context.persist(persist_dir=CARPETA_DATOS_SALVADOS)
    print(f"Índice creado y persistido en '{CARPETA_DATOS_SALVADOS}'.\n")
    return index


def actualizar_indice(index: VectorStoreIndex, nodes: List[TextNode]) -> VectorStoreIndex:
    """Añade nuevos nodos al índice existente y actualiza la persistencia en disco."""
    print(f"Añadiendo {len(nodes)} nodo(s) al índice existente...")
    _insertar_nodos(index, nodes)
    index.storage_context.persist(persist_dir=CARPETA_DATOS_SALVADOS)
    print(f"Índice actualizado en '{CARPETA_DATOS_SALVADOS}'.\n")
    return index


def cargar_indice() -> VectorStoreIndex:
    """Carga el índice vectorial desde el almacén persistente en disco."""
    print(f"Cargando índice de la carpeta {CARPETA_DATOS_SALVADOS}...")
    storage_context = StorageContext.from_defaults(persist_dir=CARPETA_DATOS_SALVADOS)
    index = load_index_from_storage(storage_context)
    print("Índice cargado con éxito.\n")
    return index


def _indice_existe() -> bool:
    """Devuelve True si hay un índice persistido en CARPETA_DATOS_SALVADOS."""
    return os.path.isfile(os.path.join(CARPETA_DATOS_SALVADOS, "docstore.json"))


def mostrar_reporte_indice(index: VectorStoreIndex) -> None:
    """Muestra un resumen tabular del contenido del índice agrupado por documento."""
    nodos = list(index.docstore.docs.values())
    if not nodos:
        print("El índice está vacío.\n")
        return

    por_doc: dict = defaultdict(list)
    for nodo in nodos:
        por_doc[nodo.metadata.get("file_name", "desconocido")].append(nodo)

    W = 48
    sep = f"  {'─' * W} {'─':>7} {'─':>9} {'─':>8}"
    print(f"\n{'═' * (W + 30)}")
    print(f"  ÍNDICE — {len(nodos)} trozos en {len(por_doc)} documento(s)")
    print(f"  {'':<{W}} {'Trozos':>7} {'Palabras':>9} {'~Tokens':>8}")
    print(sep)

    tot_t = tot_p = tot_k = 0
    for nombre, lista in sorted(por_doc.items()):
        t = len(lista)
        p = sum(n.metadata.get("NumeroPalabras", len(n.text.split())) for n in lista)
        k = sum(_estimar_tokens(n.text) for n in lista)
        nom = nombre if len(nombre) <= W else nombre[:W - 1] + "…"
        print(f"  {nom:<{W}} {t:>7} {p:>9} {k:>8}")
        tot_t += t; tot_p += p; tot_k += k

    print(sep)
    print(f"  {'TOTAL':<{W}} {tot_t:>7} {tot_p:>9} {tot_k:>8}")
    print(f"{'═' * (W + 30)}\n")


def _acceder_nodos_docstore(docstore: Any) -> list:
    """Devuelve la lista de nodos del docstore, tolerando el atributo protegido '_docs'."""
    if hasattr(docstore, "docs") and isinstance(docstore.docs, dict):
        return list(docstore.docs.values())
    if hasattr(docstore, "_docs") and isinstance(docstore._docs, dict):
        print("Advertencia: Usando el atributo protegido '_docs'.")
        return list(docstore._docs.values())
    print("Error: No se pudo encontrar el diccionario de nodos ('docs' o '_docs') en el docstore.")
    print(f"Tipo de docstore: {type(docstore)}")
    return []


def sincronizar_estado_con_indice(index: VectorStoreIndex) -> None:
    """Sincroniza el metadato 'Estado' de cada chunk con el valor del índice documental.

    Recorre indice.xlsx construyendo un mapa Identificador -> Estado y aplica ese valor a
    todos los chunks cuyo Identificador coincida, dejando el store coherente con el índice
    (si un documento está 'Vigente', todos sus chunks quedan 'Vigente', y viceversa).
    """
    CLAVE_IDENTIFICADOR = "Identificador"  # campo que enlaza chunk ↔ fila del índice documental
    CLAVE_ESTADO        = "Estado"         # campo de vigencia que se mantiene coherente

    # ── 1. Construir mapa Identificador -> Estado desde el índice documental ──────
    try:
        df_indice = pd.read_excel(FICHERO_INDICE, sheet_name="Hoja1")
    except Exception as e:
        print(f"Error leyendo el índice documental '{FICHERO_INDICE}': {e}")
        return

    mapa_estado: dict = {}
    for _, fila in df_indice.iterrows():
        identificador = fila.get(CLAVE_IDENTIFICADOR)
        estado        = fila.get(CLAVE_ESTADO)
        if pd.isna(identificador) or pd.isna(estado):
            continue
        mapa_estado[str(identificador)] = str(estado)

    if not mapa_estado:
        print("El índice documental no contiene pares Identificador/Estado válidos.")
        return

    print(f"Índice documental: {len(mapa_estado)} identificadores con Estado definido.")

    # ── 2. Acceder a los nodos del store persistido ──────────────────────────────
    if not hasattr(index, "docstore"):
        print("Error: El objeto 'index' no tiene un atributo 'docstore'.")
        return

    docstore = index.docstore
    all_nodes = _acceder_nodos_docstore(docstore)
    if not all_nodes:
        print("No se encontraron nodos en el store.")
        return

    print(f"Número total de nodos en el store: {len(all_nodes)}")

    # ── 3. Recorrer chunks y sincronizar 'Estado' con el índice documental ───────
    nodes_a_actualizar = []
    sin_identificador  = 0
    sin_referencia     = 0

    for node in all_nodes:
        if not isinstance(node, BaseNode) or not hasattr(node, "metadata"):
            continue

        identificador = node.metadata.get(CLAVE_IDENTIFICADOR)
        if identificador is None:
            sin_identificador += 1
            continue
        identificador = str(identificador)

        if identificador not in mapa_estado:
            sin_referencia += 1
            continue

        estado_correcto = mapa_estado[identificador]
        estado_actual   = node.metadata.get(CLAVE_ESTADO)

        if estado_actual != estado_correcto:
            print(
                f"  [{identificador}] {node.node_id}: "
                f"'{estado_actual}' → '{estado_correcto}'"
            )
            node.metadata[CLAVE_ESTADO] = estado_correcto
            nodes_a_actualizar.append(node)

    # ── 4. Persistir solo si hubo cambios ────────────────────────────────────────
    if nodes_a_actualizar:
        print(f"\nSincronizando {len(nodes_a_actualizar)} nodo(s) con el índice documental...")
        docstore.add_documents(nodes_a_actualizar)
        index.storage_context.persist(persist_dir=CARPETA_DATOS_SALVADOS)
        print("Metadatos 'Estado' sincronizados y store persistido.")
    else:
        print("Todos los chunks ya son coherentes con el índice documental. Nada que actualizar.")

    if sin_identificador:
        print(f"Aviso: {sin_identificador} nodo(s) sin metadato 'Identificador'.")
    if sin_referencia:
        print(f"Aviso: {sin_referencia} nodo(s) con Identificador ausente en el índice documental.")


def chatear_simple(indice: VectorStoreIndex, llm: Any, pregunta: str) -> str:
    """Crea un chat engine sobre el índice y devuelve la respuesta junto con sus referencias."""
    logger.info("Creando el Chat Engine...")
    filtros = MetadataFilters(
        filters=[ExactMatchFilter(key="Estado", value=FILTRO_ESTADO)]
    )
    chat_engine = indice.as_chat_engine(
        chat_mode="context",
        similarity_top_k=RETRIEVER_TOP_K,
        filters=filtros,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=RETRIEVER_SCORE_MINIMO)],
        system_prompt=(
            "Eres un asistente experto que responde preguntas basándose únicamente "
            "en la información proporcionada en los documentos. "
            "Al final de la respuesta, en un párrafo separado cita tus fuentes si es "
            "posible usando los metadatos (ej: nombre_archivo, pagina y URL)."
        ),
        verbose=True,
    )
    logger.info("Chat Engine listo.")

    logger.info("Iniciando conversación con el agente (usando LLM local):")
    responses: List[str] = []
    referencias: List[str] = ["\nReferencias: "]
    try:
        obj_response = chat_engine.chat(pregunta)
        responses.append(str(obj_response.response))
        logger.info(f"Respuesta del Agente: {obj_response.response}")
        if obj_response.source_nodes:
            logger.info("Fuentes consultadas:")
            for i, source_node in enumerate(obj_response.source_nodes, 1):
                logger.info(f"  Fuente {i}: Score={source_node.score:.4f}")
                logger.info(f"    Metadatos: {source_node.node.metadata}")
                url = source_node.node.metadata.get("URL")
                referencia = f"- {url} \n"
                if url and referencia not in referencias:
                    referencias.append(referencia)
        if len(referencias) > 1:
            responses += referencias
    except Exception as e:
        responses.append("\nError durante la conversación. ¿Está el servidor LM Studio corriendo?")
        responses.append(f"¿Está el modelo LLM '{getattr(llm, 'model', '?')}' cargado y seleccionado?")
        responses.append(f"Error detallado: {e}")

    chat_engine.reset()  # limpia el historial para la siguiente pregunta
    return "\n".join(responses)


def obtener_respuesta_ia(mensaje_usuario: str) -> str:
    """Devuelve la respuesta del agente IA para el mensaje recibido por el bot."""
    logger.info(f"Mensaje para la IA: '{mensaje_usuario}'")

    # Atajos para saludos básicos antes de consultar al LLM.
    if "hola" in mensaje_usuario.lower():
        respuesta = "¡Hola! Soy tu asistente IA. ¿En qué puedo ayudarte hoy?"
    elif "cómo estás" in mensaje_usuario.lower():
        respuesta = "Estoy funcionando perfectamente, ¡listo para asistirte!"
    elif "adiós" in mensaje_usuario.lower():
        respuesta = "¡Hasta pronto! Que tengas un buen día."
    else:
        respuesta = chatear_simple(index, Settings.llm, mensaje_usuario)

    logger.info(f"Respuesta de la IA: '{respuesta}'")
    return respuesta


# ── Manejadores de comandos y mensajes de Telegram ───────────────────────────

async def comando_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Envía un mensaje de bienvenida cuando se emite el comando /start."""
    user = update.effective_user
    await update.message.reply_html(
        rf"¡Hola {user.mention_html()}! Soy tu bot IA. Envíame un mensaje y te responderé.",
    )


async def manejar_mensaje(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Procesa cada mensaje de texto (no comando) recibido por el bot."""
    mensaje_recibido = update.message.text
    chat_id = update.message.chat_id
    if chat_id in lista_blanca:
        logger.info(f"Mensaje recibido de {chat_id}: '{mensaje_recibido}'")
        respuesta_ia = obtener_respuesta_ia(mensaje_recibido)
    else:
        respuesta_ia = (
            f"{update.effective_user.first_name} con id {update.effective_user.id} "
            "no estás autorizado para el uso de este agente. Ponte en contacto con "
            "el administrador para que te autoricen el acceso."
        )
    # Enviar la respuesta de la IA de vuelta al usuario.
    await update.message.reply_text(respuesta_ia)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Loguea los errores causados por los Updates."""
    logger.error(f"Update {update} causó error {context.error}")


def iniciar_bot() -> None:
    """Configura los manejadores y arranca el bot de Telegram por polling."""
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_TOKEN no definido: rellena la variable en .env para arrancar el bot.")
        return

    # Crea la Application y pásale el token del bot.
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Manejador para el comando /start
    application.add_handler(CommandHandler("start", comando_start))

    # Manejador para todos los mensajes de texto que no son comandos.
    # filters.TEXT procesa solo texto; ~filters.COMMAND excluye los comandos.
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manejar_mensaje))

    # Manejadores de errores (opcional pero recomendado).
    application.add_handler(MessageHandler(filters.COMMAND, error_handler))  # comandos no manejados
    application.add_error_handler(error_handler)                             # resto de errores

    logger.info("Iniciando el bot. Presiona Ctrl+C para detenerlo.")
    # Inicia el bot (polling para desarrollo; considera webhooks para producción).
    application.run_polling()


def main() -> None:
    """Punto de entrada: prepara el índice (flujo diario) y arranca el bot de Telegram."""
    global index
    print("=== Agente RAG con LM Studio v4.0 (bot Telegram) ===\n")

    configurar_modelos()

    # ── 1. Procesar documentos nuevos ──────────────────────────────────────────
    listado = leer_archivos_carpeta(CARPETA_ENTRADA_DOCUMENTOS)

    if listado:
        df_indice = pd.read_excel(FICHERO_INDICE, sheet_name="Hoja1")
        print(f"Entradas en el índice documental: {len(df_indice)}\n")
        nodes = digitaliza_documentos(listado, df_indice)

        if nodes:
            if _indice_existe():
                index = actualizar_indice(cargar_indice(), nodes)
            else:
                index = construir_indice(nodes)
        elif _indice_existe():
            print("Los documentos encontrados no generaron nodos. Usando índice existente.\n")
            index = cargar_indice()
        else:
            print("Sin nodos nuevos ni índice previo. Nada que hacer.")
            return
    else:
        # ── 2. Sin documentos nuevos: cargar índice existente ─────────────────
        print("Sin documentos nuevos en la carpeta de entrada.\n")
        if not _indice_existe():
            print("Tampoco hay índice persistido. Nada que hacer.")
            return
        index = cargar_indice()

    # ── 3. Sincronizar 'Estado' de los chunks con el índice documental ────────
    sincronizar_estado_con_indice(index)

    # ── 4. Reporte del estado del índice ───────────────────────────────────────
    mostrar_reporte_indice(index)

    # ── 5. Arrancar el bot de Telegram ─────────────────────────────────────────
    iniciar_bot()


if __name__ == "__main__":
    main()
