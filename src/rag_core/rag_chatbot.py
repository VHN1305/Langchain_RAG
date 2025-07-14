import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


@dataclass
class RAGConfig:
    """Cấu hình cho RAG Bot"""
    vector_db_path: str = "./vector_db"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_provider: str = "openai"  # "openai", "ollama"
    llm_model: str = "gpt-3.5-turbo"  # hoặc "llama2", "mistral" cho Ollama
    openai_api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 1000
    retrieval_k: int = 5  # Số documents được retrieve
    score_threshold: float = 0.3  # Giảm ngưỡng điểm để dễ tìm thấy tài liệu hơn


class VietnameseLegalRAGBot:
    """RAG Bot chuyên về tài liệu pháp lý Việt Nam"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = None
        self.llm = None
        self.retriever = None
        self.rag_chain = None

    def initialize_embeddings(self, use_openai: bool = False):
        """Khởi tạo embedding model"""
        if use_openai and self.config.openai_api_key:
            return OpenAIEmbeddings(
                openai_api_key=self.config.openai_api_key,
                model="text-embedding-ada-002"
            )
        else:
            return HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

    def load_vector_store(self):
        """Load vector database"""
        if not os.path.exists(self.config.vector_db_path):
            raise FileNotFoundError(f"Vector database not found at {self.config.vector_db_path}")

        embeddings = self.initialize_embeddings()
        self.vector_store = FAISS.load_local(
            self.config.vector_db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Tạo retriever với nhiều options để đảm bảo tìm được tài liệu
        if self.config.score_threshold > 0:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.config.retrieval_k,
                    "score_threshold": self.config.score_threshold
                }
            )
        else:
            # Nếu không dùng threshold, chỉ lấy k documents tương tự nhất
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.retrieval_k}
            )

        print(f"✅ Loaded vector database from {self.config.vector_db_path}")
        print(f"📊 Vector store contains {self.vector_store.index.ntotal} documents")

    def initialize_llm(self):
        """Khởi tạo Large Language Model"""
        if self.config.llm_provider == "openai":
            if not self.config.openai_api_key:
                print("⚠️ Cảnh báo: Không có OpenAI API key, chuyển sang sử dụng Ollama")
                self.config.llm_provider = "ollama"
                self.config.llm_model = "llama2"
            else:
                try:
                    self.llm = ChatOpenAI(
                        api_key=self.config.openai_api_key,
                        model=self.config.llm_model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    # Test the connection
                    test_response = self.llm.invoke("Hello")
                    print(f"✅ Initialized {self.config.llm_provider} LLM: {self.config.llm_model}")
                    return
                except Exception as e:
                    print(f"❌ Lỗi kết nối OpenAI: {e}")
                    print("🔄 Chuyển sang sử dụng Ollama...")
                    self.config.llm_provider = "ollama"
                    self.config.llm_model = "llama2"

        if self.config.llm_provider == "ollama":
            try:
                self.llm = Ollama(
                    model=self.config.llm_model,
                    temperature=self.config.temperature
                )
                print(f"✅ Initialized {self.config.llm_provider} LLM: {self.config.llm_model}")
                print("💡 Lưu ý: Đảm bảo Ollama đang chạy và model đã được tải")
            except Exception as e:
                print(f"❌ Lỗi khởi tạo Ollama: {e}")
                print("💡 Cài đặt và chạy Ollama: https://ollama.ai/")
                raise e
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def create_prompt_template(self) -> PromptTemplate:
        """Tạo prompt template cho RAG"""

        template = """Bạn là một trợ lý AI chuyên về pháp luật Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các tài liệu pháp lý được cung cấp.

NGUYÊN TẮC TRỊNH BẢY:
1. Chỉ trả lời dựa trên thông tin trong các tài liệu được cung cấp
2. Nếu không tìm thấy thông tin liên quan, hãy nói rõ "Tôi không tìm thấy thông tin liên quan trong tài liệu"
3. Trích dẫn rõ ràng điều luật, nghị định liên quan
4. Giải thích bằng ngôn ngữ dễ hiểu, tránh thuật ngữ phức tạp
5. Nếu có nhiều quy định liên quan, hãy liệt kê đầy đủ
6. Luôn chỉ rõ nguồn gốc thông tin (tên văn bản, điều, khoản)

NGỮ CẢNH TÀI LIỆU:
{context}

CÂU HỎI: {question}

HƯỚNG DẪN TRẢ LỜI:
- Bắt đầu bằng câu trả lời trực tiếp và rõ ràng
- Trích dẫn cụ thể điều luật hoặc quy định liên quan
- Giải thích ý nghĩa và cách áp dụng
- Nếu có các trường hợp ngoại lệ, hãy đề cập
- Kết thúc với lời khuyên thực tế (nếu phù hợp)

TRẢ LỜI:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def create_detailed_context(self, docs: List[Document]) -> str:
        """Tạo context chi tiết từ các documents được retrieve"""
        if not docs:
            return "Không có tài liệu liên quan được tìm thấy."

        context_parts = []

        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata

            # Tạo header cho mỗi document
            header = f"--- TÀI LIỆU {i} ---"

            # Thông tin metadata
            source_info = []
            if metadata.get('source_file'):
                source_info.append(f"Nguồn: {metadata['source_file']}")
            if metadata.get('chapter'):
                source_info.append(f"Chương: {metadata['chapter']}")
            if metadata.get('chapter_title'):
                source_info.append(f"Tiêu đề chương: {metadata['chapter_title']}")
            if metadata.get('article_title'):
                source_info.append(f"Điều: {metadata['article_title']}")

            source_line = " | ".join(source_info) if source_info else "Nguồn: Không xác định"

            # Nội dung
            content = doc.page_content.strip()

            # Kết hợp tất cả
            doc_text = f"{header}\n{source_line}\n\nNỘI DUNG:\n{content}\n"
            context_parts.append(doc_text)

        return "\n".join(context_parts)

    def setup_rag_chain(self):
        """Thiết lập RAG chain hoàn chỉnh"""
        if not self.vector_store or not self.llm:
            raise ValueError("Vector store and LLM must be initialized first")

        # Tạo prompt template
        prompt = self.create_prompt_template()

        # Tạo chain với custom context formatting
        def format_docs(docs):
            return self.create_detailed_context(docs)

        self.rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        print("✅ RAG chain setup completed")

    def initialize(self):
        """Khởi tạo toàn bộ hệ thống RAG"""
        print("🚀 Initializing Vietnamese Legal RAG Bot...")

        self.load_vector_store()
        self.initialize_llm()
        self.setup_rag_chain()

        print("✅ RAG Bot initialized successfully!")

    def test_retrieval(self, question: str) -> List[Document]:
        """Test retrieval function để debug"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        print(f"🔍 Testing retrieval for: {question}")

        # Thử nhiều cách tìm kiếm
        docs_similarity = self.vector_store.similarity_search(question, k=self.config.retrieval_k)
        print(f"📄 Similarity search found: {len(docs_similarity)} documents")

        if len(docs_similarity) > 0:
            print("✅ Found documents with similarity search")
            return docs_similarity

        # Thử với từ khóa đơn giản hơn
        simple_keywords = question.split()[:3]  # Lấy 3 từ đầu tiên
        simple_query = " ".join(simple_keywords)
        docs_simple = self.vector_store.similarity_search(simple_query, k=self.config.retrieval_k)
        print(f"📄 Simple search found: {len(docs_simple)} documents")

        return docs_simple

    def ask(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Đặt câu hỏi cho RAG bot

        Args:
            question: Câu hỏi
            return_sources: Có trả về sources không

        Returns:
            Dict chứa answer và sources (nếu có)
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call initialize() first.")

        start_time = datetime.now()

        try:
            # Test retrieval trước
            relevant_docs = self.test_retrieval(question)

            if not relevant_docs:
                return {
                    "question": question,
                    "answer": "Tôi không tìm thấy tài liệu liên quan đến câu hỏi của bạn trong cơ sở dữ liệu. Vui lòng thử lại với câu hỏi khác hoặc kiểm tra lại từ khóa.",
                    "processing_time": 0,
                    "num_sources": 0,
                    "sources": [] if return_sources else None
                }

            # Generate answer
            answer = self.rag_chain.invoke(question)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            result = {
                "question": question,
                "answer": answer,
                "processing_time": processing_time,
                "num_sources": len(relevant_docs)
            }

            if return_sources:
                sources = []
                for doc in relevant_docs:
                    sources.append({
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "metadata": doc.metadata
                    })
                result["sources"] = sources

            return result

        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return {
                "question": question,
                "answer": f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}",
                "processing_time": processing_time,
                "num_sources": 0,
                "sources": [] if return_sources else None
            }

    def batch_ask(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Xử lý nhiều câu hỏi cùng lúc"""
        results = []
        for question in questions:
            result = self.ask(question)
            results.append(result)
        return results

    def get_related_documents(self, query: str, k: int = 5) -> List[Document]:
        """Lấy documents liên quan mà không generate answer"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        return self.vector_store.similarity_search(query, k=k)

    def chat_interface(self):
        """Giao diện chat đơn giản"""
        print("\n" + "=" * 60)
        print("🤖 Vietnamese Legal RAG Bot")
        print("Gõ 'quit', 'exit' hoặc 'thoát' để kết thúc")
        print("Gõ 'help' để xem hướng dẫn")
        print("Gõ 'test' để kiểm tra hệ thống")
        print("=" * 60 + "\n")

        while True:
            try:
                question = input("❓ Câu hỏi của bạn: ").strip()

                if question.lower() in ['quit', 'exit', 'thoát', 'q']:
                    print("👋 Tạm biệt!")
                    break

                if question.lower() == 'help':
                    self._show_help()
                    continue

                if question.lower() == 'test':
                    self._run_test()
                    continue

                if not question:
                    print("⚠️ Vui lòng nhập câu hỏi")
                    continue

                print("\n🤔 Đang tìm kiếm và phân tích...")

                result = self.ask(question)

                print(f"\n💡 **Trả lời:**\n{result['answer']}")
                print(
                    f"\n📊 **Thống kê:** {result['num_sources']} tài liệu liên quan | Thời gian xử lý: {result['processing_time']:.2f}s")

                # Hiển thị sources nếu có
                if result.get('sources') and len(result['sources']) > 0:
                    show_sources = input("\n📚 Bạn có muốn xem các tài liệu tham khảo? (y/n): ").lower().startswith('y')
                    if show_sources:
                        self._display_sources(result['sources'])

                print("\n" + "-" * 60 + "\n")

            except KeyboardInterrupt:
                print("\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"❌ Lỗi: {e}")

    def _run_test(self):
        """Chạy test hệ thống"""
        print("\n🧪 Đang chạy test hệ thống...")

        # Test 1: Kiểm tra vector store
        if self.vector_store:
            total_docs = self.vector_store.index.ntotal
            print(f"✅ Vector store: {total_docs} documents")
        else:
            print("❌ Vector store: Chưa được khởi tạo")
            return

        # Test 2: Kiểm tra LLM
        if self.llm:
            print(f"✅ LLM: {self.config.llm_provider} - {self.config.llm_model}")
        else:
            print("❌ LLM: Chưa được khởi tạo")
            return

        # Test 3: Thử tìm kiếm
        test_queries = ["luật", "nghị định", "quy định", "phạt"]
        for query in test_queries:
            docs = self.vector_store.similarity_search(query, k=2)
            print(f"🔍 '{query}': {len(docs)} documents found")
            if len(docs) > 0:
                break
        else:
            print("⚠️ Không tìm thấy tài liệu nào với các từ khóa cơ bản")

        print("✅ Test hoàn thành!\n")

    def _show_help(self):
        """Hiển thị hướng dẫn sử dụng"""
        help_text = """
📖 HƯỚNG DẪN SỬ DỤNG:

🔍 Các loại câu hỏi bạn có thể đặt:
   • Hỏi về quy định cụ thể: "Quy định về thuế thu nhập cá nhân là gì?"
   • Hỏi về thủ tục: "Thủ tục đăng ký kinh doanh như thế nào?"
   • Hỏi về mức phạt: "Mức phạt vi phạm giao thông là bao nhiêu?"
   • Hỏi về điều kiện: "Điều kiện để được miễn thuế?"

💡 Mẹo để có câu trả lời tốt nhất:
   • Đặt câu hỏi cụ thể và rõ ràng
   • Sử dụng từ khóa chính xác
   • Có thể hỏi theo tên luật/nghị định cụ thể

⚠️ Lưy ý:
   • Bot chỉ trả lời dựa trên tài liệu đã được tải
   • Thông tin chỉ mang tính tham khảo
   • Nên tham khảo ý kiến chuyên gia pháp lý cho các vấn đề phức tạp

🔧 Lệnh đặc biệt:
   • 'test' - Kiểm tra hệ thống
   • 'help' - Hiển thị hướng dẫn
   • 'quit' - Thoát chương trình
        """
        print(help_text)

    def _display_sources(self, sources: List[Dict]):
        """Hiển thị sources một cách đẹp mắt"""
        print("\n📚 **TÀI LIỆU THAM KHẢO:**")
        for i, source in enumerate(sources, 1):
            metadata = source['metadata']
            print(f"\n{i}. **{metadata.get('article_title', 'N/A')}**")
            print(f"   📄 Nguồn: {metadata.get('source_file', 'N/A')}")
            if metadata.get('chapter'):
                print(f"   📑 {metadata['chapter']} - {metadata.get('chapter_title', '')}")
            print(f"   📝 Nội dung: {source['content']}")


def main():
    """Hàm main để chạy RAG bot"""

    # Cấu hình
    config = RAGConfig(
        vector_db_path="./vector_db",
        llm_provider="openai",  # Thử OpenAI trước
        llm_model="gpt-3.5-turbo",
        openai_api_key="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",
        # Temporary direct key
        temperature=0.1,
        retrieval_k=5,
        score_threshold=0.3  # Giảm threshold để dễ tìm tài liệu hơn
    )

    try:
        # Khởi tạo bot
        bot = VietnameseLegalRAGBot(config)
        bot.initialize()

        # Chạy giao diện chat
        bot.chat_interface()

    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        print("\n💡 Kiểm tra lại:")
        print("   • Vector database đã được tạo chưa?")
        print("   • Ollama đã được cài đặt và chạy chưa? (ollama serve)")
        print("   • Model llama2 đã được tải chưa? (ollama pull llama2)")
        print("   • Các thư viện cần thiết đã được cài đặt chưa?")


if __name__ == "__main__":
    main()