from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import numpy as np
import pandas as pd
import os

#================== modelo
MODELO_FASHION = "openai/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(MODELO_FASHION)
processor = CLIPProcessor.from_pretrained(MODELO_FASHION)

device = "cpu"
model.to(device)

#================== carregar vetores
df_busca = pd.read_pickle("zara_produtos_multimodais_final.pkl")
VETORES_BUSCA = np.vstack(df_busca["vetor_final_busca"].values)

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# ------------------------------------------------------------
#   FUNÇÃO PARA OBTER VETOR DE TEXTO OU IMAGEM
# ------------------------------------------------------------
def vetorizar_query(query_input, input_type='text'):
    with torch.no_grad():
        if input_type == 'image':
            image = Image.open(query_input).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)
        else:
            inputs = processor(text=[query_input], return_tensors="pt",
                               padding=True, truncation=True).to(device)
            features = model.get_text_features(**inputs)

    return features.cpu().numpy().flatten().reshape(1, -1)



# =============================================================
#   FUNÇÃO BASE — usada pelas duas recomendações
# =============================================================
def mostrar_resultados(resultados, scores):
    """Mostra nome, termos, preço e abre imagens."""
    resultados = resultados.copy()
    resultados["score"] = scores

    for idx, produto in resultados.iterrows():

        print(f"\nProduto: {produto['name']}")
        print(f"   Termos: {produto['terms']}")
        print(f"   Preço: {produto['price']}")
        print(f"   Score: {produto['score']:.4f}")

        caminho_img = produto['caminho_imagem_local']
        if os.path.exists(caminho_img):
            print(f"   ➜ A abrir imagem: {os.path.basename(caminho_img)}")
            Image.open(caminho_img).show()
        else:
            print(f"   ⚠ Imagem não encontrada: {caminho_img}")

        print("-" * 50)

    return resultados



# ------------------------------------------------------------
#   RECOMENDAR → COSINE
# ------------------------------------------------------------
def recomendar_cosine(query, n=5):
    vetor = vetorizar_query(query, "text")
    scores = cosine_similarity(vetor, VETORES_BUSCA).flatten()

    idx = np.argsort(scores)[::-1][:n]  # maiores primeiro
    resultados = df_busca.iloc[idx]

    return mostrar_resultados(resultados, scores[idx])



# ------------------------------------------------------------
#   RECOMENDAR → EUCLIDIANA
# ------------------------------------------------------------
def recomendar_euclidiana(query, n=5):
    vetor = vetorizar_query(query, "text")
    scores = euclidean_distances(vetor, VETORES_BUSCA).flatten()

    idx = np.argsort(scores)[:n]  # menores primeiro
    resultados = df_busca.iloc[idx]

    return mostrar_resultados(resultados, scores[idx])
