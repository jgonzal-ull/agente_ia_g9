"""
Agente RAG con LlamaIndex + API de OpenAI.
Pipeline: leer documentos → digitalizar → indexar → chatear.
"""

import datetime
import os
import shutil
from collections import defaultdict
from typing import Any, List

import pandas as pd
import pypandoc
from pypdf import PdfReader

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# ── Configuración ──────────────────────────────────────────────────────────────

# Rutas
FICHERO_INDICE                = "/app/Datos/indice.xlsx"       # registro de documentos
CARPETA_ENTRADA_DOCUMENTOS    = "/app/Nuevos documentos/"      # documentos pendientes de indexar
CARPETA_DOCUMENTOS_PROCESADOS = "/app/Documentos/"             # archivo de documentos ya procesados
CARPETA_DATOS_SALVADOS        = "/app/Datos/data_storage/"     # almacén persistente del índice vectorial

# API OpenAI
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")   # o sustituir por la clave directamente
MODELO_LLM        = "gpt-4o-mini"
MODELO_EMBEDDINGS = "text-embedding-3-small"

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


# ── Pipeline ───────────────────────────────────────────────────────────────────

def configurar_modelos() -> None:
    """Registra el modelo de embeddings y el LLM en la configuración global de LlamaIndex."""
    if not OPENAI_API_KEY:
        raise EnvironmentError("Variable de entorno OPENAI_API_KEY no definida.")
    Settings.embed_model = OpenAIEmbedding(
        model=MODELO_EMBEDDINGS,
        api_key=OPENAI_API_KEY,
    )
    Settings.llm = OpenAI(
        model=MODELO_LLM,
        api_key=OPENAI_API_KEY,
    )
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
    print(f"Cargando índice desde '{CARPETA_DATOS_SALVADOS}'...")
    storage_context = StorageContext.from_defaults(persist_dir=CARPETA_DATOS_SALVADOS)
    index = load_index_from_storage(storage_context)
    print("Índice cargado desde disco.\n")
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


def chatear(index: VectorStoreIndex, preguntas: List[str]) -> None:
    """Lanza un chat engine sobre el índice y responde cada pregunta de la lista."""
    filtros = MetadataFilters(
        filters=[ExactMatchFilter(key="Estado", value=FILTRO_ESTADO)]
    )
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        similarity_top_k=RETRIEVER_TOP_K,
        filters=filtros,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=RETRIEVER_SCORE_MINIMO)],
        system_prompt=(
            "Eres un asistente experto que responde preguntas basándose únicamente "
            "en la información proporcionada en los documentos. "
            "Al final de cada respuesta cita las fuentes usando los metadatos "
            "disponibles (nombre_archivo, página, URL)."
        ),
        verbose=True,
    )

    for pregunta in preguntas:
        print(f"\n{'─' * 60}")
        print(f"Pregunta: {pregunta}")
        try:
            respuesta = chat_engine.chat(pregunta)
            print(f"\nRespuesta:\n{respuesta}")
            if respuesta.source_nodes:
                print("\nFuentes consultadas:")
                for i, src in enumerate(respuesta.source_nodes, 1):
                    print(
                        f"  {i}. score={src.score:.4f} | "
                        f"{src.node.metadata.get('file_name', 'desconocido')}"
                    )
        except Exception as e:
            print(f"Error durante la conversación: {e}")

    chat_engine.reset()


def main() -> None:
    """Punto de entrada principal con flujo completo de ejecución diaria."""
    print("=== Agente RAG con OpenAI ===\n")

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

    # ── 3. Reporte del estado del índice ───────────────────────────────────────
    mostrar_reporte_indice(index)

    # ── 4. Chat ────────────────────────────────────────────────────────────────
    chatear(index, [
        "¿Quién es el responsable funcional de cada GLPI?",
        "¿Cómo es la estructura de GLPI?",
        "¿Cuáles son los días entre festivos del 2026?",
    ])


if __name__ == "__main__":
    main()
