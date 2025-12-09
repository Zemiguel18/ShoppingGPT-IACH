from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PRODUCT_SAMPLE = [
    "how much does this dress cost", "what colors are available for this shirt",
    "is this pair of jeans in stock", "what clothing items do you have in your store",
    "can you show me some shoes", "do you have any discounts on winter coats",
    "what's the warranty on this jacket", "are there any new clothing arrivals this week",
    "do you offer free shipping on clothes", "can I return this sweater if it doesn't fit",
    "what's your best-selling clothing item", "do you have any eco-friendly clothing options",
    "are these t-shirts made locally", "can you gift wrap this scarf",
    "what's the difference between these two styles of pants",
    "do you have a loyalty program for frequent clothing shoppers",
    "what's the material of this blouse", "do you have this dress in a larger size",
    "are these shoes suitable for running", "what's your return policy for online purchases",
    "can you recommend a good winter jacket", "do you have any sales on summer dresses",
    "what's the care instructions for this silk shirt", "do you offer alterations for pants",
    "what accessories would go well with this outfit", "are these sunglasses polarized",
    "do you have any vegan leather options", "what's the difference between slim fit and regular fit",
    "do you have any petite sizes available", "what's the latest fashion trend in your store",
    "do you have any waterproof jackets", "what's the price range for your formal wear",
    "do you have any sustainable fashion lines", "can you help me find a dress for a wedding",
    "what's the fabric composition of these socks", "do you have any UV protection clothing",
    "what's your most comfortable brand of shoes", "do you offer gift cards for your store",
    "what's the warranty on your watches", "can you explain the different types of denim you offer",
    "what's your return policy", "what's your shipping policy", "what's your exchange policy",
    "how do I return an item", "what's your refund policy", "can I exchange this item",
    "what's your warranty policy", "how long do I have to return items", "what's your privacy policy",
    "do you have a return policy", "what are your store policies", "can I get a refund",
    "what's your customer service policy", "how do I contact customer service",
    "what's your data protection policy", "do you offer exchanges", "what's your cancellation policy",
    "how can I cancel my order", "how do I cancel my order", "can I cancel my order",
    "how to cancel order", "order cancellation", "cancel my purchase",
    "how can I change my delivery address", "how do I track my order", "order management",
    "how to change password", "how to change shipping address", "order status",
    "how to place an order", "how to register account", "how to log in",
    "payment methods", "secure payment", "credit card safety", "delivery time",
    "product inspection", "membership account", "reward points", "warranty service",
    "company details", "invoice", "business invoice", "tax details", "business option",
    "company information", "invoice details", "business account", "corporate account",
    "change delivery address", "change shipping address", "modify address", "update address",
    "order status", "track order", "order tracking", "order management", "order history",
    "change password", "reset password", "update password", "password change",
    "place order", "make order", "create order", "submit order", "order process",
    "register account", "create account", "sign up", "account registration", "new account",
    "log in", "login", "sign in", "account login", "user login",
    "product review", "submit review", "write review", "rate product", "product rating",
    "stock status", "in stock", "out of stock", "availability", "product availability",
    "inspect product", "product inspection", "check product", "examine product",
    "membership", "loyalty program", "reward points", "points system", "member benefits",
    "warranty", "warranty service", "repair service", "product repair", "lifetime warranty",
    # Add more color and product type examples
    "red shirts", "blue jeans", "black dress", "white t-shirt", "green sweater",
    "yellow blouse", "purple hoodie", "pink skirt", "brown jacket", "gray pants",
    "shirts", "dresses", "pants", "jeans", "shoes", "jackets", "sweaters",
    "red", "blue", "black", "white", "green", "yellow", "purple", "pink",
    "show me red items", "I want blue clothes", "looking for black tops",
    "need white shirts", "want green dresses", "searching for yellow blouses"
]

CHITCHAT_SAMPLE = [
    "do you like watching movies", "what's your favorite food",
    "the sky is so blue today", "speak English, please. but keep it brief",
    "how's the weather where you are", "do you have any hobbies",
    "what's your opinion on artificial intelligence", "tell me a joke",
    "what's your favorite book", "do you believe in aliens",
    "if you could travel anywhere, where would you go", "what's the meaning of life",
    "do you have any pets", "what's your favorite music genre",
    "if you could have any superpower, what would it be",
    "what's the best advice you've ever received", "how was your day",
    "do you dream when you sleep", "what's your favorite season",
    "if you could meet any historical figure, who would it be",
    "what's your favorite holiday", "do you believe in ghosts",
    "what's your idea of a perfect day", "if you could learn any skill instantly, what would it be",
    "what's your favorite type of cuisine", "do you have any phobias",
    "what's your favorite childhood memory", "if you could time travel, which era would you visit",
    "what's your favorite sport to watch", "do you prefer mountains or beaches",
    "what's your favorite board game", "if you could be any animal, what would you choose",
    "what's your go-to karaoke song", "do you believe in love at first sight",
    "what's your favorite way to relax", "if you could have dinner with anyone, who would it be",
    "what's your favorite ice cream flavor", "do you have any hidden talents",
    "what's your favorite quote", "if you won the lottery, what's the first thing you'd buy"
]

# Constants
PRODUCT_ROUTE_NAME = 'products'
CHITCHAT_ROUTE_NAME = 'chitchat'


class SemanticRouter:
    def __init__(self):
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Prepare training data
        self.product_texts = PRODUCT_SAMPLE
        self.chitchat_texts = CHITCHAT_SAMPLE
        
        # Combine all texts for fitting
        all_texts = self.product_texts + self.chitchat_texts
        
        # Fit the vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Create labels for the routes
        self.route_labels = [PRODUCT_ROUTE_NAME] * len(self.product_texts) + [CHITCHAT_ROUTE_NAME] * len(self.chitchat_texts)

    def similarity(self, query: str, route_name: str) -> float:
        # Calculate similarity between query and route
        query_vector = self.vectorizer.transform([query])
        
        if route_name == PRODUCT_ROUTE_NAME:
            route_vectors = self.vectorizer.transform(self.product_texts)
        else:
            route_vectors = self.vectorizer.transform(self.chitchat_texts)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, route_vectors)
        return np.mean(similarities)

    def guide(self, query: str) -> str:
        # Calculate similarity with both routes
        product_similarity = self.similarity(query, PRODUCT_ROUTE_NAME)
        chitchat_similarity = self.similarity(query, CHITCHAT_ROUTE_NAME)
        
        # Return the route with higher similarity
        if product_similarity > chitchat_similarity:
            return PRODUCT_ROUTE_NAME
        else:
            return CHITCHAT_ROUTE_NAME
