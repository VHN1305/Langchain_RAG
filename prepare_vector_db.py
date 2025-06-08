import os
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import fitz  # PyMuPDF
import tiktoken


@dataclass
class Article:
    """Đại diện cho một điều luật"""
    title: str
    content: str


@dataclass
class Chapter:
    """Đại diện cho một chương"""
    name: str
    title: str
    articles: List[Article]


@dataclass
class Document:
    """Đại diện cho một tài liệu pháp lý"""
    title: str
    chapters: List[Chapter]


@dataclass
class Chunk:
    """Đại diện cho một chunk text"""
    content: str
    metadata: Dict[str, Any]


class TokenCounter:
    """Quản lý việc đếm token"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self._encoder = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        """Đếm số token trong text"""
        return len(self._encoder.encode(text))


class TextSplitter:
    """Chia nhỏ text thành các chunk"""

    def __init__(self, token_counter: TokenCounter, max_tokens: int = 512):
        self.token_counter = token_counter
        self.max_tokens = max_tokens

    def split_with_header(self, body_text: str, header: str) -> List[str]:
        """Chia text thành chunk với header"""
        header_tokens = self.token_counter.count_tokens(header)
        max_body_tokens = self.max_tokens - header_tokens

        if max_body_tokens <= 0:
            raise ValueError(f"Header quá dài ({header_tokens} tokens) so với max_tokens ({self.max_tokens})")

        words = body_text.split()
        chunks = []
        current_words = []
        current_tokens = 0

        for word in words:
            word_tokens = self.token_counter.count_tokens(word)

            if current_tokens + word_tokens > max_body_tokens and current_words:
                # Tạo chunk hiện tại
                chunk_text = f"{header}\n{' '.join(current_words)}"
                chunks.append(chunk_text)

                # Bắt đầu chunk mới
                current_words = [word]
                current_tokens = word_tokens
            else:
                current_words.append(word)
                current_tokens += word_tokens

        # Thêm chunk cuối cùng
        if current_words:
            chunk_text = f"{header}\n{' '.join(current_words)}"
            chunks.append(chunk_text)

        return chunks


class DocumentParser:
    """Phân tích cấu trúc tài liệu pháp lý"""

    def __init__(self):
        # Regex patterns
        self.chapter_pattern = re.compile(r"(Chương\s+[IVXLC]+)\s*\n([^\n]+)")
        self.article_pattern = re.compile(
            r"(Điều\s+\d+[.:]?.*?)(?=\nĐiều\s+\d+[.:]?|\nChương\s+[IVXLC]+|$)",
            re.DOTALL
        )

    def _extract_title(self, text: str) -> str:
        """Trích xuất tiêu đề tài liệu"""
        lines = text.splitlines()
        title_lines = []
        found_start = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if not found_start:
                if line.startswith(("NGHỊ ĐỊNH", "LUẬT")):
                    found_start = True
                    title_lines.append(line)
            elif line.startswith("Căn cứ"):
                break
            else:
                title_lines.append(line)

        title = " ".join(title_lines).strip()
        title = re.sub(r'\s+', ' ', title)
        return title if title else "Unknown Document Title"

    def _parse_articles(self, chapter_text: str) -> List[Article]:
        """Phân tích các điều trong chương"""
        articles = []

        for match in self.article_pattern.finditer(chapter_text):
            article_raw = match.group(1).strip()
            lines = article_raw.splitlines()

            # Tách tiêu đề và nội dung
            title_lines = []
            body_lines = []

            for idx, line in enumerate(lines):
                if re.match(r"^\s*[\da-zA-Z][\.)]", line):
                    body_lines = lines[idx:]
                    break
                title_lines.append(line)

            article_title = " ".join(l.strip() for l in title_lines)
            article_title = re.sub(r'\s+', ' ', article_title)
            article_content = "\n".join(body_lines).strip()

            articles.append(Article(title=article_title, content=article_content))

        return articles

    def parse(self, text: str) -> Document:
        """Phân tích toàn bộ tài liệu"""
        title = self._extract_title(text)
        chapters = []

        # Tìm tất cả các chương
        chapter_matches = list(self.chapter_pattern.finditer(text))

        for i, match in enumerate(chapter_matches):
            chapter_name = match.group(1).strip()
            chapter_title = match.group(2).strip()
            chapter_title = re.sub(r'\s+', ' ', chapter_title)

            # Xác định vùng text của chương
            start = match.end()
            end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
            chapter_text = text[start:end].strip()

            # Thêm tiêu đề chương vào đầu để phân tích điều
            chapter_text = f"{chapter_title}\n\n{chapter_text}"

            # Phân tích các điều
            articles = self._parse_articles(chapter_text)

            chapters.append(Chapter(
                name=chapter_name,
                title=chapter_title,
                articles=articles
            ))

        return Document(title=title, chapters=chapters)


class PDFProcessor:
    """Xử lý file PDF thành chunks"""

    def __init__(self, max_tokens: int = 1024, model_name: str = "gpt-3.5-turbo"):
        self.token_counter = TokenCounter(model_name)
        self.text_splitter = TextSplitter(self.token_counter, max_tokens)
        self.document_parser = DocumentParser()

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Trích xuất text từ PDF"""
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc)

    def _create_chunks(self, document: Document, source_file: str) -> List[Chunk]:
        """Tạo chunks từ document"""
        chunks = []
        document_id = os.path.splitext(source_file)[0]

        for chapter in document.chapters:
            for article in chapter.articles:
                # Tạo header cho chunk
                header = "\n".join([
                    document.title,
                    chapter.name,
                    chapter.title,
                    article.title
                ])

                # Chia nhỏ content
                chunk_contents = self.text_splitter.split_with_header(
                    article.content, header
                )

                # Tạo chunk objects
                for content in chunk_contents:
                    chunk = Chunk(
                        content=content,
                        metadata={
                            "source_file": source_file,
                            "document_id": document_id,
                            "chapter": chapter.name,
                            "chapter_title": chapter.title,
                            "article_title": article.title
                        }
                    )
                    chunks.append(chunk)

        return chunks

    def process_pdf(self, pdf_path: str) -> List[Chunk]:
        """Xử lý một file PDF"""
        text = self._extract_text_from_pdf(pdf_path)
        document = self.document_parser.parse(text)
        source_file = os.path.basename(pdf_path)
        return self._create_chunks(document, source_file)


class LegalDocumentProcessor:
    """Class chính để xử lý tài liệu pháp lý"""

    def __init__(self, max_tokens: int = 1024, model_name: str = "gpt-3.5-turbo"):
        self.pdf_processor = PDFProcessor(max_tokens, model_name)

    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Xử lý tất cả PDF trong folder"""
        all_chunks = []

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder không tồn tại: {folder_path}")

        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

        if not pdf_files:
            print(f"Không tìm thấy file PDF nào trong {folder_path}")
            return []

        print(f"Tìm thấy {len(pdf_files)} file PDF")

        for filename in pdf_files:
            pdf_path = os.path.join(folder_path, filename)
            print(f"Đang xử lý: {filename}")

            try:
                chunks = self.pdf_processor.process_pdf(pdf_path)
                all_chunks.extend(chunks)
                print(f"  -> Tạo được {len(chunks)} chunks")
            except Exception as e:
                print(f"  -> Lỗi khi xử lý {filename}: {e}")

        return [{"content": chunk.content, "metadata": chunk.metadata} for chunk in all_chunks]

    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str = "output_chunks.json"):
        """Lưu chunks ra file JSON"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu {len(chunks)} chunks vào {output_path}")


def main():
    """Hàm chính"""
    processor = LegalDocumentProcessor(max_tokens=1024)

    try:
        chunks = processor.process_folder("./data")
        print(f"\nTổng cộng: {len(chunks)} chunks")
        processor.save_chunks(chunks)
    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    main()