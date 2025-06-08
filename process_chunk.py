import os
import re
import fitz  # PyMuPDF
import tiktoken
import json

# Tính số token dựa trên tokenizer của GPT
def count_tokens(text, model_name="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

# Hàm chia nhỏ đoạn văn thành chunk không vượt quá max_tokens
def split_text_with_header(body_text, header, max_tokens=512, model_name="gpt-3.5-turbo"):
    tokens_header = count_tokens(header, model_name)
    max_body_tokens = max_tokens - tokens_header
    if max_body_tokens <= 0:
        raise ValueError("Header quá dài so với max_tokens")

    words = body_text.split()
    chunks = []
    current_chunk_words = []
    current_token_count = 0

    for w in words:
        w_tokens = count_tokens(w, model_name)
        if current_token_count + w_tokens > max_body_tokens:
            chunk_text = header + "\n" + " ".join(current_chunk_words)
            chunks.append(chunk_text)
            current_chunk_words = [w]
            current_token_count = w_tokens
        else:
            current_chunk_words.append(w)
            current_token_count += w_tokens

    if current_chunk_words:
        chunk_text = header + "\n" + " ".join(current_chunk_words)
        chunks.append(chunk_text)

    return chunks

# Trích xuất toàn bộ text từ PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# Phân tích cấu trúc tài liệu pháp lý
def parse_document(text):
    lines = text.splitlines()
    title = ""
    found_nghi_dinh = False
    title_lines = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if not found_nghi_dinh:
            if line_stripped.startswith("NGHỊ ĐỊNH") or line_stripped.startswith("LUẬT"):
                found_nghi_dinh = True
                title_lines.append(line_stripped)
        elif line_stripped.startswith("Căn cứ"):
            break
        else:
            title_lines.append(line_stripped)

    title = " ".join(title_lines).strip()
    title = re.sub(r'\s+', ' ', title)
    if not title:
        title = "Unknown Document Title"

    # chapter_pattern = re.compile(r"(Chương\s+[IVXLC]+)\s*\n([A-Z\sÀ-Ỹ0-9,.]+)")
    chapter_pattern = re.compile(r"(Chương\s+[IVXLC]+)\s*\n([^\n]+)")

    article_pattern = re.compile(r"(Điều\s+\d+[.:]?.*?)(?=\nĐiều\s+\d+[.:]?|\nChương\s+[IVXLC]+|$)", re.DOTALL)

    chapters = []
    chapter_splits = list(chapter_pattern.finditer(text))
    for i, chap in enumerate(chapter_splits):
        chap_start = chap.end()
        chap_name = chap.group(1).strip()
        chap_title = chap.group(2).strip()
        chap_title = re.sub(r'\s+', ' ', chap_title)

        if i + 1 < len(chapter_splits):
            chap_end = chapter_splits[i + 1].start()
        else:
            chap_end = len(text)
        chap_text = text[chap_start:chap_end].strip()

        # Thêm tiêu đề chương vào phần nội dung để detect điều
        chap_text = chap_title + "\n\n" + chap_text

        articles = []
        for art_match in article_pattern.finditer(chap_text):
            article_raw = art_match.group(1).strip()
            lines = article_raw.splitlines()
            title_lines = []
            body_lines = []
            for idx, line in enumerate(lines):
                if re.match(r"^\s*[\da-zA-Z][\.)]", line):
                    body_lines = lines[idx:]
                    break
                else:
                    title_lines.append(line)
            article_title = " ".join(l.strip() for l in title_lines)
            article_title = re.sub(r'\s+', ' ', article_title)
            article_content = "\n".join(body_lines).strip()
            articles.append({
                "article_title": article_title,
                "content": article_content
            })

        chapters.append({
            "chapter_name": chap_name,
            "chapter_title": chap_title,
            "articles": articles
        })

    return {
        "title": title,
        "chapters": chapters
    }

# Xử lý 1 file PDF: phân tích + chia chunk
def process_pdf_file(pdf_path, max_tokens=1024, model_name="gpt-3.5-turbo"):
    text = extract_text_from_pdf(pdf_path)
    doc = parse_document(text)
    title = doc["title"]

    all_chunks = []
    for chap in doc["chapters"]:
        chap_name = chap["chapter_name"]
        chap_title = chap["chapter_title"]
        for art in chap["articles"]:
            header = f"{title}\n{chap_name}\n{chap_title}\n{art['article_title']}"
            content = art["content"]
            chunks = split_text_with_header(content, header, max_tokens, model_name)
            for c in chunks:
                all_chunks.append({
                    "content": c,
                    "metadata": {
                        "source_file": os.path.basename(pdf_path),
                        "document_id": os.path.splitext(os.path.basename(pdf_path))[0],
                        "chapter": chap_name,
                        "chapter_title": chap_title,
                        "article_title": art["article_title"]
                    }
                })
    return all_chunks

# Hàm chạy chính
def main():
    folder = "./data"
    all_data = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder, fname)
            chunks = process_pdf_file(pdf_path)
            all_data.extend(chunks)
    print(f"Total chunks: {len(all_data)}")

    with open("output_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
