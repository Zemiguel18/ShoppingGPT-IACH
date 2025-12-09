import os
import sys
import warnings
import traceback

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor
from langchain.tools import tool


# === 1. CONFIGURAÇÃO E CARREGAMENTO DO MODELO CLIP ===
MODELO_FASHION = "openai/clip-vit-large-patch14"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    warnings.filterwarnings("ignore", category=FutureWarning)
    model = CLIPModel.from_pretrained(MODELO_FASHION).to(device)
    processor = CLIPProcessor.from_pretrained(MODELO_FASHION)
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar o modelo CLIP: {e}")
    model = None
    processor = None


# === 2. CARREGAMENTO DA BASE DE DADOS VETORIAL ===
DF_BUSCA = pd.DataFrame()
VETORES_BUSCA = np.array([])
FICHEIRO_FINAL = "shoppinggpt/zara_produtos_multimodais_final.pkl"  

if os.path.exists(FICHEIRO_FINAL):
    try:
        DF_BUSCA = pd.read_pickle(FICHEIRO_FINAL)   
        df_validos = DF_BUSCA[DF_BUSCA["vetor_final_busca"].notna()].copy()
        vetores_list = df_validos["vetor_final_busca"].tolist()
        if vetores_list and all(
            isinstance(v, np.ndarray) and v.shape == (768,) for v in vetores_list
        ):
            VETORES_BUSCA = np.vstack(vetores_list)
            DF_BUSCA = df_validos
            print(f"Base de dados de {VETORES_BUSCA.shape[0]} vetores carregada com sucesso.")
        else:
            print(
                "AVISO: Vetores inválidos ou ausentes no arquivo PKL. "
                "O motor de busca não funcionará."
            )

    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar os Vetores: {e}")
        sys.exit()
else:
    print(
        f"ERRO CRÍTICO: Ficheiro de vetores '{FICHEIRO_FINAL}' não encontrado. "
        "Execute o Jupyter Notebook primeiro."
    )
    sys.exit()


# === 3. LISTAS DE DESCRITORES ===
colors = ["black", "white", "red", "blue", "green", "yellow", "pink", "brown", "grey"]
tops = ["top", "t-shirt", "shirt", "blouse", "tank top", "sweater", "hoodie"]
bottoms = ["jeans", "shorts", "skirt", "trousers", "pants"]
dresses = ["dress"]
materials = ["cotton", "denim", "leather", "knit", "wool", "silk"]
top_lengths = ["cropped", "short", "regular", "long"]
bottom_lengths = ["short", "medium", "long", "above knee", "ankle length"]
dress_lengths = ["short", "medium", "long", "above knee", "ankle length"]
gender_descriptions = ["a woman", "a man", "a girl", "a boy"]

top_descriptions = [
    f"{color} {top}, {material}, {length}"
    for color in colors
    for top in tops
    for material in materials
    for length in top_lengths
]
bottom_descriptions = [
    f"{color} {bottom}, {material}, {length}"
    for color in colors
    for bottom in bottoms
    for material in materials
    for length in bottom_lengths
]
dress_descriptions = [
    f"{color} {dress}, {material}, {length}"
    for color in colors
    for dress in dresses
    for material in materials
    for length in dress_lengths
]


# === 4. FUNÇÕES DE VETORIZAÇÃO E AUXILIARES ===

def vetorizar_query(texto_ou_imagem, input_type: str = "text"):
    """Vetoriza a query (imagem ou texto) com o modelo CLIP carregado."""
    if model is None or processor is None:
        return None

    try:
        if input_type == "text":
            inputs = processor(
                text=[texto_ou_imagem], return_tensors="pt", padding=True
            ).to(device)
            with torch.no_grad():
                emb = model.get_text_features(**inputs)
        elif input_type == "image":
            image = Image.open(texto_ou_imagem).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
        else:
            return None

        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    except Exception as e:
        print(f"ERRO na vetorização de {input_type}: {e}")
        return None


def vetorizar_query_hibrida(query_texto: str, caminho_imagem: str):
    """Combina vetores de texto e imagem numa única representação (vetor médio)."""
    vetor_imagem = vetorizar_query(caminho_imagem, input_type="image")
    vetor_texto = vetorizar_query(query_texto, input_type="text")

    if vetor_imagem is not None and vetor_texto is not None:
        return (vetor_imagem + vetor_texto) / 2.0
    elif vetor_imagem is not None:
        return vetor_imagem
    else:
        return None


def process_descriptions(descriptions, image, processor, model, device, batch_size=64):
    """Calcula as probabilidades de correspondência text->image para descrições dadas."""
    matches = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i : i + batch_size]
        inputs = processor(
            text=batch, images=image, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        for j, desc in enumerate(batch):
            matches.append((desc, probs[0][j].item()))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def infer_user_focus(user_text: str) -> str | None:
    """Infer o foco da query: top, bottom, dress, outfit ou None se não houver correspondência."""
    text = user_text.lower()

    keywords_top = [
        "top",
        "t-shirt",
        "tshirt",
        "shirt",
        "blouse",
        "tank",
        "tank top",
        "sweater",
        "hoodie",
        "jumper",
        "sweat",
        "sweatshirt",
        "pullover",
        "cardigan",
        "jacket",
        "coat",
        "parka",
        "blazer",
        "crop top",
        "long sleeve",
        "polo",
        "vest",
        "camisole",
    ]
    keywords_bottom = [
        "jeans",
        "shorts",
        "skirt",
        "trousers",
        "pants",
        "bottom",
        "leggings",
        "joggers",
        "chinos",
        "cargo pants",
        "sweatpants",
        "culottes",
    ]
    keywords_dress = [
        "dress",
        "vestido",
        "gown",
        "mini dress",
        "midi dress",
        "maxi dress",
        "evening dress",
        "summer dress",
    ]
    keywords_outfit = [
        "outfit",
        "conjunto",
        "look",
        "set",
        "two-piece",
        "matching set",
        "tracksuit",
        "suit",
        "co-ord",
        "coord set",
    ]

    if any(k in text for k in keywords_top):
        return "top"
    if any(k in text for k in keywords_bottom):
        return "bottom"
    if any(k in text for k in keywords_dress):
        return "dress"
    if any(k in text for k in keywords_outfit):
        return "outfit"

    return None  


def gerar_descricao_roupa(image, user_focus, processor, model, device, top_k: int = 3):
    """Gera descrição textual da roupa com base na imagem e no foco."""
    top_matches = []
    bottom_matches = []
    dress_matches = []

    if user_focus in ["top", "outfit"]:
        top_matches = process_descriptions(
            top_descriptions, image, processor, model, device
        )
    if user_focus in ["bottom", "outfit"]:
        bottom_matches = process_descriptions(
            bottom_descriptions, image, processor, model, device
        )
    if user_focus in ["dress", "outfit"]:
        dress_matches = process_descriptions(
            dress_descriptions, image, processor, model, device
        )

    partes = []
    if top_matches:
        top_descr = ", ".join(
            [f"{m[0]} ({m[1]:.3f})" for m in top_matches[:top_k]]
        )
        partes.append(f"top: {top_descr}")
    if bottom_matches:
        bottom_descr = ", ".join(
            [f"{m[0]} ({m[1]:.3f})" for m in bottom_matches[:top_k]]
        )
        partes.append(f"bottom: {bottom_descr}")
    if dress_matches:
        dress_descr = ", ".join(
            [f"{m[0]} ({m[1]:.3f})" for m in dress_matches[:top_k]]
        )
        partes.append(f"dress: {dress_descr}")

    gender_matches = process_descriptions(
        gender_descriptions, image, processor, model, device
    )
    gender_descr = f"{gender_matches[0][0]} ({gender_matches[0][1]:.3f})"

    descricao_roupa = "; ".join(partes) if partes else "clothes"
    descricao_final = f"{gender_descr}, wearing {descricao_roupa}"

    return descricao_final


# === 5. FUNÇÃO PRINCIPAL DE BUSCA HÍBRIDA / TEXTUAL ===

def search_products_direct(
    query: str, image_path: str = None, top_k: int = 5
) -> list[dict]:
    """
    Função principal de busca multimodal/textual, chamada diretamente pelo Flask.
    Usa o CLIP para vetorizar a query (texto puro ou híbrida) e retorna o top K.
    Retorna SEMPRE uma lista de produtos (pode ser vazia).
    """
    try:
        if VETORES_BUSCA.size == 0 or model is None:
            return []

        vetor_busca = None

        # --- A. LÓGICA MULTIMODAL (IMAGEM + TEXTO) ---
        if image_path and os.path.exists(image_path):
            user_focus = infer_user_focus(query)
            if user_focus is None:
                return []

            image_pil = Image.open(image_path).convert("RGB")
            TEXTO_FOCO = gerar_descricao_roupa(
                image=image_pil,
                user_focus=user_focus,
                processor=processor,
                model=model,
                device=device,
                top_k=3,
            )
            print(f"DEBUG Multimodal: Texto Foco Gerado: {TEXTO_FOCO}")

            vetor_hibrido = vetorizar_query_hibrida(TEXTO_FOCO, image_path)
            if vetor_hibrido is not None:
                vetor_busca = vetor_hibrido.reshape(1, -1)

        # --- B. LÓGICA TEXTUAL PURA (Apenas Texto) ---
        else:
            user_focus = infer_user_focus(query)
            if user_focus is None:
                return []

            query_vector = vetorizar_query(query, input_type="text")
            if query_vector is not None:
                vetor_busca = query_vector.reshape(1, -1)

        if vetor_busca is None:
            return []

        # --- C. EXECUÇÃO DA BUSCA VETORIAL ---
        similaridades = cosine_similarity(vetor_busca, VETORES_BUSCA).flatten()
        indices_ranking = np.argsort(similaridades)[::-1]
        top_indices = indices_ranking[:top_k]

        resultados = DF_BUSCA.iloc[top_indices].copy()
        resultados["similarity"] = similaridades[top_indices]

        produtos_finais = []
        for _, produto in resultados.iterrows():
            image_name = os.path.basename(produto.get("caminho_imagem_local", ""))
            produtos_finais.append(
                {
                    "name": produto.get("name", "N/A"),
                    "price": f"{produto.get('price', 'N/A')} {produto.get('currency', 'USD')}",
                    "link": produto.get("url", "#"),
                    "similarity": float(produto["similarity"]),
                    "image_url": f"/static/images/products/{image_name}",
                }
            )

        return produtos_finais

    except Exception:
        print("ERROR in search_products_direct:", traceback.format_exc())
        return []


# === 6. FUNÇÃO PARA O AGENTE (LangChain Tool) ===

@tool
def multimodal_search(query: str, image_path: str = None) -> list[dict]:
    """
    Use this tool exclusively for product search. It performs a multimodal search
    combining a text query (e.g., 'only the green top') with a user-uploaded image path
    (image_path). The text query is crucial for defining the specific focus (e.g., 'top' or 'bottom')
    within the image. Always use the search query provided by the user.
    Returns a list of the top matching products with name, price, and image URL.
    """
    return search_products_direct(query=query, image_path=image_path)
