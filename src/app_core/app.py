from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import argparse
from src.rag_core.create_vector_db import VectorDBCreator


app = Flask(__name__, template_folder="../../templates", static_folder="../../static")

UPLOAD_FOLDER = '../../upload_datas'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

creator = VectorDBCreator(embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
creator.load_vector_store("../../vector_db")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")

    # Tìm kiếm các tài liệu tương tự
    results = creator.search_similar(message, k=3)
    top_docs = [doc[0].page_content for doc in results[:3]]

    # Trả về danh sách các phản hồi
    return jsonify({"responses": top_docs})


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        return f"Upload thành công: {filename}", 200
    else:
        return "File không hợp lệ (chỉ chấp nhận: .pdf, .doc, .docx, .txt)", 400


def parse_args():
    parser = argparse.ArgumentParser(description="Flask app config")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the app")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the app")
    parser.add_argument("--debug", action="store_true", help="Run app in debug mode")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
