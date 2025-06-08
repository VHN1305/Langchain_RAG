from rag_chatbot import VietnameseLegalRAGBot, RAGConfig

# Khởi tạo
config = RAGConfig(openai_api_key="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")
bot = VietnameseLegalRAGBot(config)
bot.initialize()

# Đặt câu hỏi
result = bot.ask("Quy định về thuế thu nhập cá nhân là gì?")
print(result["answer"])