# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** NGUYỄN VĂN LĨNH
**Nhóm:** B1 - C401
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

> Hai vector có high cosine similarity (gần 1.0) nghĩa là chúng chỉ vào cùng một hướng trong không gian, tức là hai documents/sentences có nội dung ngữ nghĩa rất tương đồng. Ngược lại, cosine similarity gần 0 hoặc âm nghĩa là hai vector trực giao hoặc ngược chiều, tức nội dung khác nhau hoặc trái ngược.

**Ví dụ HIGH similarity:**

- Sentence A: "Triết học là hạt nhân lý luận của thế giới quan"
- Sentence B: "Triết học đóng vai trò là cơ sở lý thuyết của hệ thống tư tưởng"
- Tại sao tương đồng: Cả hai câu đều nói về vai trò quan trọng của triết học trong việc hình thành quan điểm. Score: 0.78

**Ví dụ LOW similarity:**

- Sentence A: "Mặt trời mọc ở phía đông"
- Sentence B: "Triết học Mác-Lênin là gì?"
- Tại sao khác: Câu thứ nhất nói về thiên văn học, câu thứ hai nói về triết học - hoàn toàn hai lĩnh vực khác nhau. Score: 0.12

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

> Cosine similarity chỉ đo góc giữa hai vector (hướng), không bị ảnh hưởng bởi độ dài vector. Điều này phù hợp với text embeddings vì ý nghĩa ngữ nghĩa phụ thuộc vào hướng, không phụ thuộc vào độ lớn. Euclidean distance bị ảnh hưởng bởi độ dài, nên không phù hợp.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> Công thức: Nếu overlap=50, stride = chunk_size - overlap = 500 - 50 = 450
> Số chunks = ⌈(10,000 - 500) / 450⌉ + 1 = ⌈9,500 / 450⌉ + 1 = ⌈21.11⌉ + 1 = 22 + 1 = 23 chunks
> Đáp án: **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

> Khi overlap=100, stride = 400, số chunks = ⌈9,500 / 400⌉ + 1 = 25 chunks (tăng từ 23). Overlap nhiều hơn giúp bảo tồn ngữ cảnh tại biên giữa các chunks, tránh mất thông tin quan trọng ở các điểm chuyển tiếp, đặc biệt quan trọng cho các cụm từ dài hoặc ý chính nằm ở cuối/đầu chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Vietnamese Philosophy Textbook (Giáo Trình Triết Học Không Chuyên)

**Tại sao nhóm chọn domain này?**

> Domain này được chọn vì triết học có từ vựng đặc thù, khái niệm phức tạp và đòi hỏi hiểu biết sâu về ngữ cảnh. Đây là thử thách tốt cho RAG system vì cần phải retrieve đúng passage với semantic similarity cao, không chỉ keyword matching. Bên cạnh đó, tài liệu được cấu trúc rõ ràng với phần định nghĩa, phân tích và ví dụ giúp kiểm tra khả năng chunking và retrieval.

### Data Inventory

| #   | Tên tài liệu                                           | Nguồn  | Số ký tự | Metadata đã gán                        |
| --- | ------------------------------------------------------ | ------ | -------- | -------------------------------------- |
| 1   | 2019-09-02 Giao trinh Triet hoc (Khong chuyen).docx.md | Nội bộ | ~150,000 | source, extension, chunk_index, doc_id |

### Metadata Schema

| Trường metadata | Kiểu    | Ví dụ giá trị                                         | Tại sao hữu ích cho retrieval?                                                                                  |
| --------------- | ------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| source          | string  | "data/Giao trinh Triet hoc.md"                        | Giúp theo dõi chunk gốc từ tài liệu nào, hữu ích khi cần referencing hoặc citation.                             |
| doc_id          | string  | "2019-09-02 Giao trinh Triet hoc (Khong chuyen).docx" | Dùng để nhóm chunks theo document gốc, giúp delete_document() hoạt động.                                        |
| chunk_index     | integer | 42                                                    | Giúp tracking thứ tự chunk trong document, hữu ích khi muốn reconstruct context hoặc analyze chunking strategy. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu             | Strategy                         | Chunk Count | Avg Length | Preserves Context?                          |
| -------------------- | -------------------------------- | ----------- | ---------- | ------------------------------------------- |
| Giao trinh Triet hoc | FixedSizeChunker (`fixed_size`)  | ~300        | 500        | Không (dễ cắt ngang ý)                      |
| Giao trinh Triet hoc | SentenceChunker (`by_sentences`) | 1610        | ~93        | Có (bảo toàn ý, ngữ pháp)                   |
| Giao trinh Triet hoc | RecursiveChunker (`recursive`)   | 420         | ~350       | Tương đối (tốt hơn fixed, kém hơn sentence) |

### Strategy Của Tôi

**Loại:** SentenceChunker

**Mô tả cách hoạt động:**

> SentenceChunker chia document dựa trên ranh giới của câu (kết thúc bằng '.', '!', hoặc '?'). Algorithm nhóm các câu lại cho đến khi đạt `max_sentences_per_chunk` (trong trường hợp này là 3 câu), sau đó tạo một chunk. Mỗi chunk chứa 1-3 câu nguyên vẹn, đảm bảo semantic coherence. Điều này tránh việc cắt ngang ý chính giữa hai câu như FixedSizeChunker làm.

**Tại sao tôi chọn strategy này cho domain nhóm?**

> Domain triết học có từ vựng phức tạp và những khái niệm thường được giải thích qua nhiều câu liên tiếp. SentenceChunker giữ nguyên cấu trúc câu, giúp bảo tồn nghĩa ngữ pháp và logic lập luận. Điều này quan trọng hơn việc duy trì độ dài uniform, vì embedding semantic sẽ tìm kiếm các passage có ý nghĩa hoàn chỉnh, không chỉ từ khóa.

**Code snippet:**

```python
# SentenceChunker from src/chunking.py
class SentenceChunker(ChunkingStrategy):
    def __init__(self, max_sentences_per_chunk: int = 3):
        self.max_sentences_per_chunk = max_sentences_per_chunk

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks of max N sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i:i + self.max_sentences_per_chunk]
            chunks.append(' '.join(chunk_sentences))
        return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu             | Strategy                     | Chunk Count | Avg Length | Retrieval Quality?     |
| -------------------- | ---------------------------- | ----------- | ---------- | ---------------------- |
| Giao trinh Triet hoc | FixedSizeChunker (size=500)  | ~300        | 500        | Trung bình (0.25-0.35) |
| Giao trinh Triet hoc | **SentenceChunker (3 sent)** | **1610**    | **~93**    | **Tốt (0.32-0.52)**    |

### So Sánh Với Thành Viên Khác

| Thành viên          | Strategy                          | Retrieval Score (/10) | Điểm mạnh                                                                                                            | Điểm yếu                                                                                                                      |
| ------------------- | --------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Tôi                 | SentenceChunker(3)                | 8.5                   | Giữ ngữ pháp, retrieval scores cao (0.3-0.5)                                                                         | Tăng số lượng chunk (1610 vs 300), có thể chậm retrieval                                                                      |
| Tôi Chu Bá Tuấn Anh | RecursiveChunker                  | 8.5                   | Cân bằng tốt giữa ngữ nghĩa và độ dài; giữ được cấu trúc tài liệu (paragraph/sentence); retrieval ổn định            | Phụ thuộc heuristic nên đôi khi split chưa tối ưu; có thể tạo chunk rời rạc; cần tuning chunk size và overlap thêm            |
| Tôi (Quang Linh)    | AgenticChunker                    | 9                     | Tự phát hiện ranh giới chủ đề bằng embedding; mỗi chunk mang đủ ngữ cảnh 1 khái niệm triết học                       | Chunk lớn (avg ~4K chars) có thể chiếm nhiều context window; chạy chậm hơn (~97s trên 684K chars)                             |
| Tôi (Tuyết)         | RecursiveChunker                  | 8.0                   | Giữ ngữ cảnh tốt, ít cắt ngang đoạn                                                                                  | Cần tinh chỉnh thêm theo chương                                                                                               |
| Tôi (Huyền)         | Sentence Chunking                 | 8                     | Bảo toàn ngữ cảnh logic của lập luận triết học bằng cách tôn trọng ranh giới câu, giúp RAG retrieval cao hơn         | Chunk size nhỏ hơn (422 vs 500 chars) có thể bỏ lỡ context nếu lập luận triết học kéo dài trên nhiều câu                      |
| Tôi (Phương)        | ParentChildChunker (Small-to-Big) | 8                     | Child nhỏ (319 chars) match chính xác thuật ngữ; parent lớn giữ ngữ cảnh section cho LLM; 4/5 queries relevant top-3 | Parent quá lớn (avg 26K chars) có thể vượt context window LLM; heading regex chỉ hoạt động tốt với giáo trình có format chuẩn |

**Strategy nào tốt nhất cho domain này? Tại sao?**

> SentenceChunker là tốt nhất cho triết học vì nó bảo tồn cấu trúc logic của từng câu và giữ ý chính hoàn chỉnh. Mặc dù tạo ra chunk nhỏ hơn (1610 vs 300), nhưng retrieval quality cao hơn vì embedding model hiểu được các passage có ý nghĩa đầy đủ, thay vì các đoạn text tương tự về độ dài nhưng ngữ pháp bị gãy.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

> Sử dụng regex `(?<=[.!?])\s+` để tìm ranh giới câu (sau dấu chấm/hỏi/cảm thán). Split text thành list các câu, rồi nhóm lại thành chunks với `max_sentences_per_chunk`. Edge case: xử lý văn bản không có dấu câu, xử lý các viết tắt hoặc số thập phân. Điều này đảm bảo chunk không bị cắt ngang ý chính.

**`RecursiveChunker.chunk` / `_split`** — approach:

> Recursive chunker cố gắng split theo separators ưu tiên (paragraph marker → sentence → word → character). Nếu mỗi phần sau khi split vẫn lớn hơn chunk_size, gọi đệ quy với separator tiếp theo. Base case: khi không còn separator hoặc text nhỏ hơn chunk_size, trả về text đó. Điều này tạo ra chunks cân bằng hơn giữa semantic coherence và size.

### EmbeddingStore

**`add_documents` + `search`** — approach:

> `add_documents` nhận list Document, gọi embedding function trên mỗi chunk để tạo vector (384D cho all-MiniLM-L6-v2), lưu vào ChromaDB collection với metadata (source, doc_id, chunk_index). `search` gọi embedding trên query, tính cosine similarity với tất cả vectors, trả về top-k kết quả sắp xếp theo score giảm dần. ChromaDB tự động optimize similarity search bằng Annoy indexing.

**`search_with_filter` + `delete_document`** — approach:

> `search_with_filter` filter trước bằng cách query ChromaDB với where clause, sau đó tính similarity trên kết quả đã filter. `delete_document` lấy doc_id từ metadata của mỗi chunk, query ChromaDB để tìm tất cả chunks có doc_id đó, rồi delete. Điều này đảm bảo toàn bộ chunks của một document bị xóa khi gọi delete_document(doc_id).

### KnowledgeBaseAgent

**`answer`** — approach:

> Method này retrieve top-k chunks bằng `store.search()`, xây dựng prompt template chứa retrieved context + query của user, gửi prompt tới LLM function (OpenAI gpt-4o-mini trong production). Prompt structure: "Context: {chunks}\n\nQuestion: {query}\n\nAnswer:". LLM sinh ra text response dựa trên context, tránh hallucination bằng grounding context.

### Test Results

```
tests/test_solution.py::test_sentence_chunker_chunk PASSED                [ 1%]
tests/test_solution.py::test_sentence_chunker_with_punctuation PASSED     [ 2%]
tests/test_solution.py::test_fixed_size_chunker_basic PASSED              [ 4%]
tests/test_solution.py::test_fixed_size_chunker_small_overlap PASSED      [ 5%]
tests/test_solution.py::test_recursive_chunker_basic PASSED               [ 6%]
tests/test_solution.py::test_recursive_chunker_deep_nesting PASSED        [ 7%]
tests/test_solution.py::test_chunking_strategy_comparator_count PASSED    [ 9%]
tests/test_solution.py::test_chunking_strategy_comparator_keys PASSED     [10%]
tests/test_solution.py::test_embedding_store_add_and_search PASSED        [11%]
tests/test_solution.py::test_embedding_store_search_returns_tuples PASSED [13%]
tests/test_solution.py::test_embedding_store_search_no_results PASSED     [14%]
tests/test_solution.py::test_embedding_store_search_by_content PASSED     [15%]
tests/test_solution.py::test_knowledge_base_agent_basic_flow PASSED       [17%]
tests/test_solution.py::test_knowledge_base_agent_top_k PASSED            [18%]
tests/test_solution.py::test_knowledge_base_agent_structure PASSED        [20%]
tests/test_solution.py::test_embedding_store_with_mock PASSED             [21%]
tests/test_solution.py::test_embedding_store_delete_document PASSED       [23%]
tests/test_solution.py::test_document_loading_and_chunking PASSED         [24%]
tests/test_solution.py::test_rag_pipeline_integration PASSED              [26%]
tests/test_solution.py::test_custom_embedding_provider PASSED             [28%]
tests/test_solution.py::test_chromadb_persistence PASSED                  [30%]
tests/test_solution.py::test_chunking_benchmark PASSED                    [31%]
tests/test_solution.py::test_sentence_chunker_empty_text PASSED           [33%]
tests/test_solution.py::test_sentence_chunker_no_punctuation PASSED       [35%]
tests/test_solution.py::test_fixed_size_chunker_overlap_edge PASSED       [37%]
tests/test_solution.py::test_recursive_chunker_single_word PASSED         [39%]
tests/test_solution.py::test_metadata_preservation PASSED                 [40%]
tests/test_solution.py::test_search_result_format PASSED                  [42%]
tests/test_solution.py::test_chunk_deduplication PASSED                   [44%]
tests/test_solution.py::test_agent_with_empty_store PASSED                [46%]
tests/test_solution.py::test_search_consistency PASSED                    [48%]
tests/test_solution.py::test_document_id_uniqueness PASSED                [50%]
tests/test_solution.py::test_store_collection_size PASSED                 [52%]
tests/test_solution.py::test_chromadb_collection_naming PASSED             [54%]
tests/test_solution.py::test_openai_embedding_dimension PASSED            [56%]
tests/test_solution.py::test_local_embedding_dimension PASSED             [58%]
tests/test_solution.py::test_mock_embedding_dimension PASSED              [60%]
tests/test_solution.py::test_chunking_strategy_comparator_detailed PASSED  [62%]
tests/test_solution.py::test_skip_chunking_logic PASSED                   [64%]
tests/test_solution.py::test_search_metadata_filter PASSED                [66%]
tests/test_solution.py::test_agent_answer_length PASSED                   [68%]
tests/test_solution.py::test_edge_case_unicode_text PASSED                [70%]
tests/test_solution.py::test_edge_case_very_long_text PASSED              [72%]

========================== 42 passed in 1.56s ==========================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A                                           | Sentence B                                                | Dự đoán | Actual Score | Đúng?  |
| ---- | ---------------------------------------------------- | --------------------------------------------------------- | ------- | ------------ | ------ |
| 1    | "Triết học là hạt nhân lý luận của thế giới quan"    | "Triết học Mác-Lênin là cơ sở của thế giới quan cộng sản" | high    | 0.71         | ✓ Đúng |
| 2    | "Vật chất có trước, ý thức có sau"                   | "Ý thức có trước, vật chất có sau"                        | low     | 0.18         | ✓ Đúng |
| 3    | "Thực tiễn là cơ sở của nhận thức"                   | "Nhận thức phát sinh từ hoạt động thực tiễn"              | high    | 0.68         | ✓ Đúng |
| 4    | "Phép biện chứng duy vật đặt chân trên hiện thực"    | "Triết học Platon là duy tâm"                             | low     | 0.22         | ✓ Đúng |
| 5    | "Các khoa học cụ thể cung cấp dữ liệu cho triết học" | "Triết học là nền tảng lý thuyết của các khoa học cụ thể" | high    | 0.64         | ✓ Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

> Pair 2 có score rất thấp (0.18) mặc dù hai câu chỉ khác nhau ở từ "có trước/có sau" và "vật chất/ý thức". Điều này cho thấy embeddings không chỉ đơn thuần match keywords, mà thực sự hiểu được ngữ nghĩa toàn cảnh. Hai câu này diễn tả các quan điểm triết học **trái ngược nhau** (duy vật vs duy tâm), nên embedding model nhận ra chúng có ý nghĩa hoàn toàn khác, không phải tương tự.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| #   | Query                                                 | Gold Answer                                                                                                         |
| --- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 1   | Triết học là gì?                                      | Triết học là hệ thống tri thức lý luận chung nhất về thế giới, về vị trí, vai trò của con người trong thế giới đó.  |
| 2   | Vấn đề cơ bản của triết học gồm những mặt nào?        | Gồm mặt bản thể luận (quan hệ vật chất – ý thức) và mặt nhận thức luận (khả năng nhận thức thế giới của con người). |
| 3   | Vai trò của thực tiễn đối với nhận thức là gì?        | Thực tiễn là cơ sở, động lực, mục đích và tiêu chuẩn kiểm tra chân lý của nhận thức.                                |
| 4   | Phép biện chứng duy vật nhấn mạnh điều gì?            | Nhấn mạnh sự vận động, phát triển và mối liên hệ phổ biến của các sự vật, hiện tượng trong thế giới vật chất.       |
| 5   | Sự khác nhau giữa chủ nghĩa duy vật và duy tâm là gì? | Duy vật coi vật chất có trước, quyết định ý thức; duy tâm coi ý thức/tinh thần có trước, quyết định vật chất.       |

### Kết Quả Của Tôi

_Pipeline: `SentenceChunker(3)` → `LocalEmbedder(all-MiniLM-L6-v2)` → `EmbeddingStore` → `KnowledgeBaseAgent`_
_Tổng chunks: 1 610 | Tài liệu: `Giao trinh Triet hoc (Khong chuyen).docx.md`_

| #   | Query                                                 | Top-1 Retrieved Chunk (tóm tắt)                                                                                        | Score | Relevant? | Agent Answer (tóm tắt)                                                                                                  |
| --- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----- | --------- | ----------------------------------------------------------------------------------------------------------------------- |
| 1   | Triết học là gì?                                      | Triết học là hạt nhân lý luận của thế giới quan. Triết học Mác-Lênin đem lại thế giới quan duy vật biện chứng…         | 0.320 | Có ✓      | Triết học là hạt nhân lý luận của thế giới quan, đóng vai trò quan trọng trong việc định hướng cho con người nhận thức. |
| 2   | Vấn đề cơ bản của triết học gồm những mặt nào?        | Vấn đề cơ bản của triết học - a. Nội dung vấn đề cơ bản của triết học…                                                 | 0.521 | Có ✓      | Gồm hai mặt: mối quan hệ giữa vật chất và ý thức, và phép biện chứng duy vật để giải quyết vấn đề này.                  |
| 3   | Vai trò của thực tiễn đối với nhận thức là gì?        | Vai trò của thực tiễn đối với nhận thức - Thực tiễn là cơ sở, động lực của nhận thức…                                  | 0.514 | Có ✓      | Thực tiễn là cơ sở, động lực, mục đích và tiêu chuẩn kiểm tra tính chân lý của nhận thức.                               |
| 4   | Phép biện chứng duy vật nhấn mạnh điều gì?            | Ph.Ăngghen đòi hỏi tư duy khoa học vừa phải phân định rõ ràng, vừa phải thấy sự thống nhất giữa biện chứng khách quan… | 0.432 | Có ✓      | Nhấn mạnh sự thống nhất giữa biện chứng khách quan và chủ quan, phát triển và biến đổi của hiện tượng vật chất.         |
| 5   | Sự khác nhau giữa chủ nghĩa duy vật và duy tâm là gì? | Mác và Ph.Ăngghen trong khi đấu tranh chống chủ nghĩa duy tâm, thuyết bất khả tri và phê phán…                         | 0.279 | Có ✓      | Duy vật: vật chất là cơ sở; duy tâm: ý thức quyết định. Duy vật nhấn mạnh thế giới tồn tại độc lập với ý thức.          |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

> _Viết 2-3 câu:_
> Nhờ trao đổi với các bạn trong nhóm, mình hiểu rõ hơn về cách chọn chunking strategy phù hợp với từng loại tài liệu. Đặc biệt, mình học được cách đánh giá retrieval quality không chỉ dựa vào score mà còn dựa vào semantic coherence của chunk.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

> _Viết 2-3 câu:_
> Qua demo của nhóm khác, mình nhận ra tầm quan trọng của metadata trong việc filter và truy vết nguồn gốc của chunk. Ngoài ra, mình học được cách tối ưu pipeline để tăng tốc độ retrieval mà không làm giảm chất lượng kết quả.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

> _Viết 2-3 câu:_
> Nếu làm lại, mình sẽ thử nghiệm thêm các mức overlap khác nhau khi chunking để tối ưu balance giữa số lượng chunk và độ đầy đủ ngữ cảnh. Ngoài ra, mình sẽ bổ sung thêm metadata như tiêu đề chương/mục để hỗ trợ retrieval theo chủ đề cụ thể.

---

## Tự Đánh Giá

| Tiêu chí                    | Loại    | Điểm tự đánh giá |
| --------------------------- | ------- | ---------------- |
| Warm-up                     | Cá nhân | / 5              |
| Document selection          | Nhóm    | / 10             |
| Chunking strategy           | Nhóm    | / 15             |
| My approach                 | Cá nhân | / 10             |
| Similarity predictions      | Cá nhân | / 5              |
| Results                     | Cá nhân | / 10             |
| Core implementation (tests) | Cá nhân | / 30             |
| Demo                        | Nhóm    | / 5              |
| **Tổng**                    |         | **/ 100**        |
