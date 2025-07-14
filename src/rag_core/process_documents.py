import os
import re
import fitz  # PyMuPDF
import tiktoken
import json
from tqdm import tqdm  # Thêm dòng này ở đầu
from time import time

# Tính số token dựa trên tokenizer của GPT
def count_tokens(text, model_name="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model_name)
    rs = len(enc.encode(text))
    return rs

def split_text_with_header(body_text, header, max_tokens=512, model_name="gpt-3.5-turbo", overlap_tokens=50):
    tokens_header = count_tokens(header, model_name)
    max_body_tokens = max_tokens - tokens_header
    if max_body_tokens <= 0:
        raise ValueError("Header quá dài so với max_tokens")

    words = body_text.split()
    chunks = []

    i = 0
    while i < len(words):
        current_chunk_words = []
        current_token_count = 0
        start_index = i

        while i < len(words):
            w = words[i]
            w_tokens = count_tokens(w, model_name)
            if current_token_count + w_tokens > max_body_tokens:
                break
            current_chunk_words.append(w)
            current_token_count += w_tokens
            i += 1

        chunk_text = header + "\n" + " ".join(current_chunk_words)
        chunks.append(chunk_text)

        start = time()
        # Lùi lại `overlap_tokens` tính theo word cho chunk tiếp theo
        if i < len(words):
            overlap_word_count = 0
            overlap_idx = i - 1
            while overlap_idx > 0 and overlap_word_count < overlap_tokens:
                overlap_word_count += count_tokens(words[overlap_idx], model_name)
                overlap_idx -= 1
            i = max(overlap_idx + 1, 0)
        time_overlap = time() - start
        if time_overlap > 1:
            print("Time Overlap: ", time_overlap)
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

    # Initialize chapter_pattern outside the conditional block
    chapter_pattern = re.compile(r"(Chương\s+[IVXLC]+)\s*\n([^\n]+)")
    article_pattern = re.compile(
        r"(Điều\s+\d+[.:]?.*?\.)\s*(?=\nĐiều\s+\d+[.:]?|\nChương\s+[IVXLC]+|$)",
        re.DOTALL
    )

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

        # Add chapter title to the content for article detection
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
def process_pdf_file(pdf_path, max_tokens=512, model_name="gpt-3.5-turbo"):
    text = extract_text_from_pdf(pdf_path)
    doc = parse_document(text)
    title = doc["title"]

    all_chunks = []

    # Tổng số điều trong toàn bộ tài liệu (để tqdm chạy mượt hơn)
    total_articles = sum(len(chap["articles"]) for chap in doc["chapters"])
    pbar = tqdm(total=total_articles, desc=f"Processing {os.path.basename(pdf_path)}", unit="article")

    for chap in doc["chapters"]:
        chap_name = chap["chapter_name"]
        chap_title = chap["chapter_title"]
        for art in chap["articles"]:
            header = f"{title}\n{chap_name}\n{chap_title}\n{art['article_title']}"
            content = art["content"]
            chunks = split_text_with_header(content, header, max_tokens, model_name, overlap_tokens=100)
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
            pbar.update(1)  # Cập nhật tiến trình sau khi xử lý xong 1 điều

    pbar.close()
    return all_chunks


# Hàm chạy chính
def main():
    folder = "../../data"
    all_data = []
    pdf_files = [fname for fname in os.listdir(folder) if fname.lower().endswith(".pdf")]

    for fname in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        pdf_path = os.path.join(folder, fname)
        chunks = process_pdf_file(pdf_path, max_tokens=1024)
        all_data.extend(chunks)

    print(f"Total chunks: {len(all_data)}")

    with open("../../data/output_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
