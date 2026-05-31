"""
Agente RAG con LlamaIndex + LM Studio (servidor externo OpenAI-compatible).
Pipeline: leer documentos → digitalizar → indexar → chatear.
"""

import asyncio
import datetime
import os
import time
from typing import Any, List

import pandas as pd
import requests

from pypdf import PdfReader

from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.llms.openai_like import OpenAILike

# ── Configuración ──────────────────────────────────────────────────────────────

# Rutas
FICHERO_INDICE             = "/app/Datos/indice.xlsx"       # registro de documentos
CARPETA_ENTRADA_DOCUMENTOS = "/app/Nuevos documentos/"      # PDFs pendientes de indexar

# Servidor LM Studio
SERVIDOR_LMSTUDIO = "http://openai.ull.es:8080/v1"
LMSTUDIO_APITOKEN = os.getenv("LMSTUDIO_APITOKEN", "")  # o sustituir por el token directamente
MODELO_LLM        = "qwen/qwen3.5-9b"
MODELO_EMBEDDINGS = "text-embedding-qwen3-embedding-0.6b"

# Chunking de documentos
UMBRAL_MAXIMO_PALABRAS = 3000   # máx. palabras por trozo
PALABRAS_SOLAPAMIENTO  = 50     # palabras compartidas entre trozos consecutivos
NUMERO_MINIMO_PALABRAS = 0      # trozos con menos palabras se descartan


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
            f"{self._api_base}/embeddings", json=payload, timeout=60
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
    print("=== Agente RAG con LM Studio ===\n")

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
