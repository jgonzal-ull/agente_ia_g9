"""
Agente RAG con LlamaIndex + API de OpenAI.
Pipeline: leer documentos → digitalizar → indexar → chatear.
"""

import datetime
import os
from typing import Any, List

import pandas as pd
from pypdf import PdfReader

from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# ── Configuración ──────────────────────────────────────────────────────────────

# Rutas
FICHERO_INDICE             = "/app/Datos/indice.xlsx"       # registro de documentos
CARPETA_ENTRADA_DOCUMENTOS = "/app/Nuevos documentos/"      # PDFs pendientes de indexar

# API OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # o sustituir por la clave directamente
MODELO_LLM        = "gpt-4o-mini"
MODELO_EMBEDDINGS = "text-embedding-3-small"

# Chunking de documentos
UMBRAL_MAXIMO_PALABRAS = 300    # máx. palabras por trozo
PALABRAS_SOLAPAMIENTO  = 50     # palabras compartidas entre trozos consecutivos
NUMERO_MINIMO_PALABRAS = 0      # trozos con menos palabras se descartan


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
            "carpeta":              os.path.dirname(ruta_completa),
            "nombre":               nombre_base,
            "extension":            ext.lstrip("."),
            "nombre_con_extension": nombre,
            "ruta_completa":        ruta_completa,
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


def digitaliza_un_documento(documento: dict, fila_indice: Any) -> list:
    """Extrae el texto del PDF página a página y genera trozos con metadatos."""
    print(f"  Digitalizando: {documento['nombre_con_extension']}")
    resultado = []
    try:
        pdf = PdfReader(documento["ruta_completa"])
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
    return resultado


def digitaliza_documentos(listado: list, df_indice: Any) -> List[TextNode]:
    """Procesa la lista de documentos y devuelve los TextNodes listos para indexar."""
    nodes: List[TextNode] = []
    for doc in listado:
        nombre = doc["nombre_con_extension"]
        filas = df_indice[df_indice["Nombre Archivo"] == nombre]
        if filas.empty:
            print(f"'{nombre}' no aparece en el índice, se omite.")
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

    print(f"Nodos generados: {len(nodes)}\n")
    return nodes


def _estimar_tokens(texto: str) -> int:
    """Estima el número de tokens de un texto (~4 caracteres por token)."""
    return len(texto) // 4


def construir_indice(nodes: List[TextNode]) -> VectorStoreIndex:
    """Crea un VectorStoreIndex en memoria e inserta los nodos uno a uno con log."""
    print(f"Creando índice en memoria ({len(nodes)} nodos)...")
    index = VectorStoreIndex(nodes=[])

    for i, node in enumerate(nodes, 1):
        palabras = node.metadata.get("NumeroPalabras", len(node.text.split()))
        tokens   = _estimar_tokens(node.text)
        id_trozo = node.metadata.get("IdTrozoTexto", node.node_id)
        print(f"  [{i:>4}/{len(nodes)}] {id_trozo} — {palabras} palabras, ~{tokens} tokens")
        index.insert_nodes([node])

    print(f"Índice en memoria listo.\n")
    return index


def chatear(index: VectorStoreIndex, preguntas: List[str]) -> None:
    """Lanza un chat engine sobre el índice y responde cada pregunta de la lista."""
    chat_engine = index.as_chat_engine(
        chat_mode="context",
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
    """Punto de entrada: configura modelos, digitaliza PDFs, indexa y abre el chat."""
    print("=== Agente RAG con OpenAI ===\n")

    configurar_modelos()

    listado   = leer_archivos_carpeta(CARPETA_ENTRADA_DOCUMENTOS)
    df_indice = pd.read_excel(FICHERO_INDICE, sheet_name="Hoja1")
    print(f"Entradas en el índice documental: {len(df_indice)}\n")

    nodes = digitaliza_documentos(listado, df_indice)
    index = construir_indice(nodes)

    chatear(index, [
        "¿Quién es el responsable funcional de cada GLPI?",
        "¿Cómo es la estructura de GLPI?",
        "¿Cuáles son los días entre festivos del 2026?",
    ])


if __name__ == "__main__":
    main()
