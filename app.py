import os
import traceback

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

from shoppinggpt.tool.search_engine_multimodal import (
    search_products_direct,
    multimodal_search as multimodal_tool,
    infer_user_focus,
)
from shoppinggpt.router.lib_semantic_router import (
    SemanticRouter,
    PRODUCT_ROUTE_NAME,
    CHITCHAT_ROUTE_NAME,
)
from shoppinggpt.chain import create_chitchat_chain
from shoppinggpt.agent import ShoppingAgent


# === Configuração de logs gRPC ===
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""


GOOGLE_API_KEY = "AIzaSyCcspiL0XR_EH40aj5EtsfGRoaq_u3752Y"


LLM = None
if GOOGLE_API_KEY:
    LLM = ChatGoogleGenerativeAI(
        temperature=0,
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
    )


SHARED_MEMORY = ConversationBufferMemory(return_messages=True)

SHOPPING_AGENT = None
if LLM:
    SHOPPING_AGENT = ShoppingAgent(LLM, SHARED_MEMORY)


SEMANTIC_ROUTER = SemanticRouter()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

PRODUCT_IMAGES_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "shoppinggpt",
    "images",
    "imagens_zara",
)


@app.route("/static/images/products/<path:filename>")
def serve_product_images(filename):
    return send_from_directory(PRODUCT_IMAGES_FOLDER, filename)

def format_product_response(products, query_type: str) -> str:
    """
    Formats the product result for a response in Markdown.
    """
    if not products:
        return (
            "My bad! I can only help with fashion stuff. "
            "Maybe take a look at some clothing?"
        )

    product_info = []
    product_info.append(
        f"Here are the top {len(products)} products that match your request:"
    )

    for rank, p in enumerate(products, start=1):
        name = p.get("name", "Unknown product")
        price = p.get("price", "N/A")
        image_url = p.get("image_url", "")
        link = p.get("link", "#")
        similarity = p.get("similarity", 0.0)

        info = (
            f"**[RANK {rank}] {name}** — {price}\n"
            f"![{name}]({image_url})\n"
            f"[Link do Produto]({link})\n"
            f"(Similarity: {similarity:.4f})"
        )
        product_info.append(info)

    content = "\n\n---\n\n".join(product_info)
    return content


def handle_query(query: str) -> dict:
    """Handles plain text queries and routes to chat/product search."""
    if not LLM:
        return {
            "response": "AI service is not configured. Please check GOOGLE_API_KEY in the code.",
            "type": "error",
            "products": [],
        }

    guided_route = SEMANTIC_ROUTER.guide(query)

    if guided_route == CHITCHAT_ROUTE_NAME:
        try:
            if SHOPPING_AGENT:
                content = SHOPPING_AGENT.invoke(query)
            else:
                content = create_chitchat_chain(LLM, SHARED_MEMORY).invoke(
                    {"input": query}
                ).content
            products = []
        except Exception:
            print("ERROR in CHITCHAT route:", traceback.format_exc())
            products = []
            content = (
                "My bad! I can only help with fashion stuff. "
                "Maybe take a look at some clothing?"
            )

    elif guided_route == PRODUCT_ROUTE_NAME:
        try:
            products = search_products_direct(
                query=query,
                image_path=None,
                top_k=5,
            )

            if not products:
                content = (
                    "My bad! I can only help with fashion stuff. "
                    "Maybe take a look at some clothing?"
                )
            else:
                content = format_product_response(products, "text_search")

        except Exception:
            print("ERROR in text search:", traceback.format_exc())
            products = []
            content = (
                "My bad! I can only help with fashion stuff. "
                "Maybe take a look at some clothing?"
            )

    else:
        content = (
            "My bad! I can only help with fashion stuff. "
            "Maybe take a look at some clothing?"
        )
        products = []

   
    SHARED_MEMORY.chat_memory.add_user_message(query)
    SHARED_MEMORY.chat_memory.add_ai_message(content)

    return {
        "response": content,
        "type": guided_route,
        "products": products,
    }


# === Rotas ===

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["GET"])
def get_bot_response():
    try:
        user_message = request.args.get("msg", "")
        print("DEBUG /get msg=", repr(user_message))
        user_message = (user_message or "").strip()

        if not user_message:
            return (
                jsonify(
                    {
                        "response": "My bad! I can only help with fashion stuff. Maybe take a look at some clothing?",
                        "type": "error",
                        "products": [],
                    }
                ),
                400,
            )

        response = handle_query(user_message)
        print("DEBUG /get response=", response)
        return jsonify(response), 200

    except Exception:
        print("ERROR in /get:", traceback.format_exc())
        return (
            jsonify(
                {
                    "response": "My bad! I can only help with fashion stuff. Maybe take a look at some clothing?",
                    "type": "error",
                    "products": [],
                }
            ),
            500,
        )


@app.route("/multimodal_search", methods=["POST"])
def multimodal_search_route():
    """Lida com upload de imagem e/ou texto para busca multimodal."""
    if "query" not in request.form:
        return jsonify({"response": "Missing query.", "type": "error"}), 400

    query_text = request.form["query"].strip()
    file = request.files.get("file", None)

    temp_path = None

    try:
        # CASO 1: Apenas TEXTO
        if file is None or file.filename == "":
            results = search_products_direct(
                query=query_text,
                image_path=None,
                top_k=5,
            )
            if not results:
                content = (
                    "My bad! I can only help with fashion stuff. "
                    "Maybe take a look at some clothing?"
                )
            else:
                content = format_product_response(results, "text_search")

            return jsonify(
                {
                    "response": content,
                    "type": "text_product_results",
                    "products": results,
                }
            )

        # CASO 2: TEXTO + IMAGEM
        filename = file.filename
        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(temp_path)

        results = search_products_direct(
            query=query_text,
            image_path=temp_path,
            top_k=5,
        )

        if not results:
            content = (
                "My bad! I can only help with fashion stuff. "
                "Maybe take a look at some clothing?"
            )
        else:
            content = format_product_response(results, "multimodal")

        return jsonify(
            {
                "response": content,
                "type": "multimodal_product_results",
                "products": results,
            }
        )

    except Exception:
        print(f"ERROR in multimodal_search_route: {traceback.format_exc()}")
        return (
            jsonify(
                {
                    "response": "My bad! I can only help with fashion stuff. Maybe take a look at some clothing?",
                    "type": "error",
                    "products": [],
                }
            ),
            500,
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug_mode, host="0.0.0.0", port=port, use_reloader=False)
