import json
import os
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import pickle


class VectorDBCreator:
    def __init__(self, embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Khởi tạo VectorDBCreator

        Args:
            embedding_model_name: Tên model embedding (mặc định dùng multilingual model cho tiếng Việt)
        """
        self.embedding_model_name = embedding_model_name
        self.embeddings = None
        self.vector_store = None

    def initialize_embeddings(self, use_openai: bool = False, openai_api_key: str = None):
        """
        Khởi tạo embedding model

        Args:
            use_openai: Sử dụng OpenAI embeddings hay không
            openai_api_key: API key của OpenAI (nếu sử dụng OpenAI)
        """
        if use_openai:
            if not openai_api_key:
                raise ValueError("OpenAI API key is required when using OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model="text-embedding-ada-002"
            )
            print("Initialized OpenAI embeddings")
        else:
            # Sử dụng HuggingFace embeddings (miễn phí, chạy local)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},  # Có thể đổi thành 'cuda' nếu có GPU
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"Initialized HuggingFace embeddings: {self.embedding_model_name}")

    def load_chunks_from_json(self, json_file_path: str) -> List[Dict[str, Any]]:
        """
        Load chunks từ file JSON

        Args:
            json_file_path: Đường dẫn đến file JSON chứa chunks

        Returns:
            List of chunks
        """
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File {json_file_path} not found")

        with open(json_file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks from {json_file_path}")
        return chunks

    def create_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """
        Chuyển đổi chunks thành Langchain Documents

        Args:
            chunks: List of chunks từ JSON

        Returns:
            List of Langchain Documents
        """
        documents = []

        for i, chunk in enumerate(chunks):
            # Tạo Document với content và metadata
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    **chunk["metadata"],  # Metadata từ chunk gốc
                    "chunk_id": i,  # Thêm ID cho chunk
                    "chunk_length": len(chunk["content"])
                }
            )
            documents.append(doc)

        print(f"Created {len(documents)} documents")
        return documents

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Tạo FAISS vector store từ documents

        Args:
            documents: List of Langchain Documents

        Returns:
            FAISS vector store
        """
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Call initialize_embeddings() first.")

        print("Creating vector store...")
        print("This may take a while depending on the number of documents and embedding model...")

        # Tạo FAISS vector store
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        print("Vector store created successfully!")
        return self.vector_store

    def save_vector_store(self, save_path: str = "./vector_db"):
        """
        Lưu vector store vào disk

        Args:
            save_path: Đường dẫn để lưu vector store
        """
        if not self.vector_store:
            raise ValueError("Vector store not created yet. Call create_vector_store() first.")

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(save_path, exist_ok=True)

        # Lưu FAISS vector store
        self.vector_store.save_local(save_path)

        # Lưu thông tin embedding model để load lại sau
        config = {
            "embedding_model_name": self.embedding_model_name,
            "vector_store_type": "FAISS"
        }

        with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"Vector store saved to {save_path}")

    def load_vector_store(self, load_path: str = "./vector_db", use_openai: bool = False, openai_api_key: str = None):
        """
        Load vector store từ disk

        Args:
            load_path: Đường dẫn chứa vector store
            use_openai: Sử dụng OpenAI embeddings hay không
            openai_api_key: API key của OpenAI
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Vector store path {load_path} not found")

        # Load config
        config_path = os.path.join(load_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.embedding_model_name = config.get("embedding_model_name", self.embedding_model_name)

        # Initialize embeddings
        self.initialize_embeddings(use_openai=use_openai, openai_api_key=openai_api_key)

        # Load FAISS vector store
        self.vector_store = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print(f"Vector store loaded from {load_path}")
        return self.vector_store

    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """
        Tìm kiếm documents tương tự

        Args:
            query: Câu hỏi/truy vấn
            k: Số lượng kết quả trả về

        Returns:
            List of similar documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not created or loaded yet.")

        results = self.vector_store.similarity_search(query, k=k)
        return results

    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Tìm kiếm documents tương tự kèm điểm số

        Args:
            query: Câu hỏi/truy vấn
            k: Số lượng kết quả trả về

        Returns:
            List of (document, score) tuples
        """
        if not self.vector_store:
            raise ValueError("Vector store not created or loaded yet.")

        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results


def main():
    """
    Hàm main để tạo vector database
    """
    # Khởi tạo VectorDBCreator
    creator = VectorDBCreator()

    # Có thể chọn sử dụng OpenAI embeddings hoặc HuggingFace embeddings
    use_openai = False  # Đổi thành True nếu muốn dùng OpenAI
    openai_api_key = "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"  # Thay bằng API key nếu dùng OpenAI

    try:
        # 1. Initialize embeddings
        creator.initialize_embeddings(use_openai=use_openai, openai_api_key=openai_api_key)

        # 2. Load chunks từ JSON file
        chunks = creator.load_chunks_from_json("output_chunks.json")

        # 3. Tạo Langchain Documents
        documents = creator.create_documents(chunks)

        # 4. Tạo vector store
        vector_store = creator.create_vector_store(documents)

        # 5. Lưu vector store
        creator.save_vector_store("./vector_db")

        print("\n" + "=" * 50)
        print("Vector database created successfully!")
        print(f"Total documents: {len(documents)}")
        print("Vector store saved to ./vector_db")
        print("=" * 50)

        # Test search (optional)
        print("\nTesting search functionality...")
        test_query = "xe máy vượt đèn đỏ thì bị phạt bao nhiêu tiền?"
        results = creator.search_similar(test_query, k=3)

        print(f"\nTest query: '{test_query}'")
        print("Top 3 results:")
        for i, doc in enumerate(results):
            print(f"\n{i + 1}. {doc.metadata.get('article_title', 'N/A')}")
            print(f"   Source: {doc.metadata.get('source_file', 'N/A')}")
            print(f"   Content preview: {doc.page_content[:200]}...")

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


if __name__ == "__main__":
    main()