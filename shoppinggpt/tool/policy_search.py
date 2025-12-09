import os
from typing import List

from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from shoppinggpt.config import EMBEDDINGS, DATA_TEXT_PATH, STORE_DIRECTORY

class VectorStoreManager:
    def __init__(self, data_path: str, store_directory: str, embeddings):
        self.data_path = data_path
        self.store_directory = store_directory
        self.embeddings = embeddings
        self.vectorstore = self.load_or_create_vectorstore()

    def load_vectorstore(self):
        return FAISS.load_local(
            self.store_directory,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def create_vectorstore(self):
        loader = TextLoader(self.data_path, encoding='utf8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        document_chunks = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(document_chunks, self.embeddings)
        vectorstore.save_local(self.store_directory)
        return vectorstore

    def check_existing_vectorstore(self):
        return os.path.exists(os.path.join(self.store_directory, "index.faiss"))

    def load_or_create_vectorstore(self):
        if self.check_existing_vectorstore():
            return self.load_vectorstore()
        else:
            return self.create_vectorstore()

    @staticmethod
    def create(data_path: str, store_directory: str, embeddings):
        return VectorStoreManager(data_path, store_directory, embeddings)


@tool
def policy_search_tool(query: str) -> List[str]:
    """
    Search for information related to company policies.

    Args:
        query (str): The search query to find information.

    Returns:
        List[str]: The search results as a list of text strings.
    """
    # Special handling for specific queries first
    query_lower = query.lower()
    
    # Handle "pay upon delivery" queries specifically
    if 'pay' in query_lower and 'delivery' in query_lower:
        try:
            with open(DATA_TEXT_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('Question:') and 'pay' in line.lower() and 'delivery' in line.lower():
                    question = line
                    # Get the complete answer (may span multiple lines)
                    answer_lines = []
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith('Question:'):
                        if lines[j].startswith('Answer:'):
                            answer_lines.append(lines[j])
                        elif answer_lines:  # Include all lines after Answer: until next Question:
                            answer_lines.append(lines[j])
                        j += 1
                    answer = '\n'.join(answer_lines)
                    if answer:
                        return [f"{question}\n{answer}"]
        except Exception as e:
            pass
    
    # Handle "place order" queries specifically
    if 'place' in query_lower and 'order' in query_lower:
        try:
            with open(DATA_TEXT_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('Question:') and 'place' in line.lower() and 'order' in line.lower():
                    question = line
                    # Get the complete answer (may span multiple lines)
                    answer_lines = []
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith('Question:'):
                        if lines[j].startswith('Answer:'):
                            answer_lines.append(lines[j])
                        elif answer_lines:  # Include all lines after Answer: until next Question:
                            answer_lines.append(lines[j])
                        j += 1
                    answer = '\n'.join(answer_lines)
                    if answer:
                        return [f"{question}\n{answer}"]
        except Exception as e:
            pass
    
    # For other specific policy queries, use direct text search
    if any(keyword in query.lower() for keyword in ['return', 'refund', 'exchange', 'shipping', 'delivery', 'warranty', 'cancel', 'cancellation', 'invoice', 'company', 'business', 'tax', 'details', 'stock', 'availability', 'place order', 'submit', 'rate', 'rating', 'password', 'address', 'track', 'register', 'login', 'membership', 'points', 'inspection', 'review']):
        try:
            with open(DATA_TEXT_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find specific policy sections
            lines = content.split('\n')
            policy_sections = []
            
            # Special handling for specific queries
            query_lower = query.lower()
            
            # Handle "place order" queries specifically
            if 'place' in query_lower and 'order' in query_lower:
                for i, line in enumerate(lines):
                    if line.startswith('Question:') and 'place' in line.lower() and 'order' in line.lower():
                        question = line
                        answer = lines[i + 1] if i + 1 < len(lines) and lines[i + 1].startswith('Answer:') else ""
                        if answer:
                            policy_sections.append(f"{question}\n{answer}")
                            break
            
            # Handle other policy queries
            if not policy_sections:
                for i, line in enumerate(lines):
                    # Check for relevant keywords in the line
                    relevant_keywords = []
                    if any(keyword in query_lower for keyword in ['return', 'refund', 'exchange']):
                        relevant_keywords.extend(['return', 'exchange', 'refund'])
                    if any(keyword in query_lower for keyword in ['shipping', 'delivery']):
                        relevant_keywords.extend(['shipping', 'delivery'])
                    if any(keyword in query_lower for keyword in ['warranty']):
                        relevant_keywords.extend(['warranty'])
                    if any(keyword in query_lower for keyword in ['cancel', 'cancellation']):
                        relevant_keywords.extend(['cancel', 'cancellation'])
                    if any(keyword in query_lower for keyword in ['invoice', 'company', 'business', 'tax', 'details']):
                        relevant_keywords.extend(['invoice', 'company', 'business', 'tax', 'details'])
                    if any(keyword in query_lower for keyword in ['stock', 'availability']):
                        relevant_keywords.extend(['stock', 'availability'])
                    if any(keyword in query_lower for keyword in ['rate', 'rating', 'review']):
                        relevant_keywords.extend(['rate', 'rating', 'review'])
                    if any(keyword in query_lower for keyword in ['password', 'address', 'track', 'register', 'login', 'membership', 'points', 'inspection']):
                        relevant_keywords.extend(['password', 'address', 'track', 'register', 'login', 'membership', 'points', 'inspection'])
                    
                    if any(keyword in line.lower() for keyword in relevant_keywords):
                        # Get the question and answer
                        if line.startswith('Question:'):
                            question = line
                            # Get the complete answer (may span multiple lines)
                            answer_lines = []
                            j = i + 1
                            while j < len(lines) and not lines[j].startswith('Question:'):
                                if lines[j].startswith('Answer:'):
                                    answer_lines.append(lines[j])
                                elif answer_lines:  # Include all lines after Answer: until next Question:
                                    answer_lines.append(lines[j])
                                j += 1
                            answer = '\n'.join(answer_lines)
                            if answer:
                                policy_sections.append(f"{question}\n{answer}")
            
            if policy_sections:
                return policy_sections[:2]  # Return max 2 relevant sections
        except Exception as e:
            pass  # Fall back to vector search
    
    # For other queries, use vector search
    vector_store_manager = VectorStoreManager.create(
        DATA_TEXT_PATH,
        STORE_DIRECTORY,
        EMBEDDINGS
    )

    if any(keyword in query.lower() for keyword in ['shipping', 'delivery']):
        results = vector_store_manager.vectorstore.similarity_search(query, k=2)
    elif any(keyword in query.lower() for keyword in ['warranty']):
        results = vector_store_manager.vectorstore.similarity_search(query, k=2)
    else:
        results = vector_store_manager.vectorstore.similarity_search(query, k=3)
    
    return [doc.page_content for doc in results]
