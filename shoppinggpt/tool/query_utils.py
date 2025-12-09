
def extract_product_terms(query: str) -> str:
    """
    Extrai apenas os termos de produto da query do utilizador.
    Remove palavras comuns como 'I want', 'show me', 'find', etc.
    """
    stopwords = [
        "i want", "show me", "find", "give me", "please", 
        "can you", "look for", "search for"
    ]
    cleaned_query = query.lower()
    for word in stopwords:
        cleaned_query = cleaned_query.replace(word, "")
    return cleaned_query.strip()
