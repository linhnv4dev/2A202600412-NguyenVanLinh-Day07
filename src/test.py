from .chunking import SentenceChunker, RecursiveChunker

text = "Hello world. How are you? I am fine. This is test. Another sentence."
chunker = SentenceChunker(max_sentences_per_chunk=2)


complex_text = """
## Mục Tiêu Học Tập Cốt Lõi

1. **Embedding Intuition (G2)**: Hiểu cosine similarity, dự đoán được điểm tương đồng, nhận ra giới hạn của embedding.
2. **Vector Store Operations (G3)**: Triển khai store/search/filter/delete; giải thích khi nào metadata filtering giúp ích vs gây hại.
3. **Full Pipeline (G4)**: Triển khai mỗi bước Document → Chunk → Embed → Store → Query → Inject; so sánh chunking strategies.
4. **Data Strategy (G5)**: Chọn dữ liệu, thiết kế metadata, tối ưu chunking — hiểu rằng data quality > model selection.

---

## Ghi Chú Cho Giảng Viên: Embedder Thật Là Tùy Chọn

- Lab này **không bắt buộc** sinh viên cài embedder thật.
- Luồng mặc định cho lớp học vẫn là `_mock_embed`, nên sinh viên vẫn có thể hoàn thành lab và pass test mà không cần tải model nào.
- Nếu sinh viên muốn thử embedding thật trên máy cá nhân, package `src` đã hỗ trợ cả:
  - `all-MiniLM-L6-v2` qua `sentence-transformers`
  - OpenAI embeddings qua package `openai`

Ví dụ local embedder:

```bash
pip install sentence-transformers
python3 - <<'PY'
from src import LocalEmbedder
embedder = LocalEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

Ví dụ OpenAI embedder:

```bash
pip install openai
export OPENAI_API_KEY=your-key-here
python3 - <<'PY'
from src import OpenAIEmbedder
embedder = OpenAIEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

- Khuyến nghị giảng viên nói rõ ngay từ đầu: **“Local/OpenAI embedder là bonus / optional, không phải điều kiện để hoàn thành lab.”**
- Khi có sinh viên máy yếu, mạng chậm, không có API key, hoặc không muốn tải model, hãy hướng họ tiếp tục với `_mock_embed` để tránh bị kẹt ở phần setup.
- `src_w_solution/` là reference solution cho giảng viên / maintainer. Không phân phối thư mục này cho sinh viên.

---

## Timeline & Flow (4.5 giờ)

### Phase 1: Document Preparation (30 phút, 0:00–0:30)

**Hoạt động (nhóm):**
- Nhóm chọn domain (FAQ, law, recipes, medical, tech docs, v.v.)
- Thu thập 5-10 tài liệu, chuyển sang `.txt`/`.md`, đặt vào `data/`
- Thiết kế metadata schema (ít nhất 2 trường hữu ích)

**Vai trò giảng viên:**
- Giải thích lab structure: "30 phút chuẩn bị tài liệu nhóm → mỗi người tự code → mỗi người thử strategy riêng → so sánh trong nhóm → demo với lớp"
- Gợi ý domain nếu nhóm chưa quyết
- Nhấn mạnh: "Chọn tài liệu có cấu trúc rõ ràng — chất lượng tài liệu quyết định kết quả"

### Phase 2: Individual Coding (90 phút, 0:30–2:00)

**Warm-up (10 phút):**
- Ex 1.1: Cosine similarity — giải thích bằng ngôn ngữ tự nhiên
- Ex 1.2: Chunking math — tính toán số chunks

**Implementation (80 phút):**
- Mỗi sinh viên **tự mình** implement tất cả TODO trong `src/chunking.py`, `src/store.py`, và `src/agent.py`
- `Document` và `FixedSizeChunker` đã implement sẵn làm ví dụ
- Thứ tự gợi ý: `SentenceChunker` → `RecursiveChunker` → `compute_similarity` → `ChunkingStrategyComparator` → `EmbeddingStore` → `KnowledgeBaseAgent`

**Vai trò giảng viên:**
- **Nhấn mạnh**: "Đây là phần cá nhân — mỗi người tự code"
- **Checkpoint 1 (1:00)**: "Ai đã pass phần chunking (`TestSentenceChunker`, `TestRecursiveChunker`)?" — giải thích nếu < 50%
- **Checkpoint 2 (1:30)**: "Ai đã pass TestEmbeddingStore?" — debug nếu cần
"""
# print(chunker.chunk(text))
recursive_chunker = RecursiveChunker(separators=["\n\n", "\n", ". ", "! ", "? "])
print(recursive_chunker.chunk(complex_text))