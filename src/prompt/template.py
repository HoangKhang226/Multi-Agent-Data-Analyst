"""Prompt templates for all nodes in the Hierarchical RAG pipeline.

Naming: one constant per node, in execution order.
"""

# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

USER_MEMORY_SECTION = """\
## Thông tin người dùng (bộ nhớ dài hạn)
{user_memory}"""


# ---------------------------------------------------------------------------
# Node 0 — Context Compressor
# ---------------------------------------------------------------------------

CONTEXT_COMPRESSION_PROMPT = """\
Bạn là trợ lý phân tích dữ liệu. Đọc toàn bộ nội dung bên dưới và tạo bản \
tóm tắt ngắn gọn (tối đa 100 từ).

Tập trung vào:
- Chủ đề, mục đích chính của tài liệu
- Các thực thể, cột dữ liệu, quy tắc nghiệp vụ quan trọng
- Phạm vi, giới hạn và các số liệu nổi bật

## Nội dung đầu vào
{input_data}
"""


# ---------------------------------------------------------------------------
# Node 1 — Ambiguity Checker
# ---------------------------------------------------------------------------

AMBIGUITY_CHECK_PROMPT = """\
Bạn là chuyên gia đánh giá câu hỏi. Nhiệm vụ của bạn là xác định câu hỏi có cần làm rõ hay không.

## Quy tắc (áp dụng theo thứ tự ưu tiên)
1. Câu chào hỏi / xã giao thông thường → is_ambiguous = false
2. Câu tự giới thiệu hoặc chia sẻ thông tin cá nhân → is_ambiguous = false
3. Câu hỏi không liên quan đến tài liệu/dữ liệu và dataset → is_ambiguous = false
4. Câu hỏi rõ ràng, có thể xử lý → is_ambiguous = false

## NGUYÊN TẮC THÂN THIỆN VỚI NGƯỜI DÙNG (RẤT QUAN TRỌNG)
- Nếu câu hỏi thiếu chi tiết nhưng có thể suy luận hợp lý → is_ambiguous = false
- Ưu tiên trả lời với giả định hợp lý thay vì yêu cầu làm rõ
- KHÔNG coi việc thiếu loại biểu đồ, tham số nhỏ là ambiguous
- User không chuyên thường hỏi ngắn → vẫn phải xử lý được

## TRƯỜNG HỢP ĐẶC BIỆT (THÊM MỚI)
Nếu câu hỏi liên quan đến:
- "trực quan hóa", "vẽ biểu đồ", "plot", "chart"
- hoặc các thao tác phân tích dữ liệu (pandas, numpy, thống kê)
→ LUÔN đặt is_ambiguous = false
Ví dụ:
"có thể trực quan hóa cột sales."

## TRƯỜNG HỢP ĐẶC BIỆT (THÊM MỚI)
Nếu câu hỏi có chứa từ liên quan đến các cột nhưng cột đó không tồn tại trong dataset
→ LUÔN đặt is_ambiguous = true
Nếu các cột đó tồn tại nhưng ở dạng khác về tên
→ LUÔN đặt is_ambiguous = false

Ví dụ:
"có thể trực quan hóa cột doanh thu." ở đây doanh thu có thể hiểu là sales

## CHỈ đánh dấu ambiguous khi:
- Không hiểu mục tiêu câu hỏi
- Hoặc có nhiều cách hiểu mâu thuẫn rõ ràng

{user_memory_section}

## Thông tin dataset
{dataframe_head}

## Chi tiết cột (info)
{dataframe_info}

## Tóm tắt tài liệu
{content_summary}

## Câu hỏi
{question}
"""

# ---------------------------------------------------------------------------
# Node 2 — Planner
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """\
Bạn là chuyên gia phân tích yêu cầu. Phân tách câu hỏi thành 1–5 sub-task \
độc lập, mỗi sub-task thể hiện một yêu cầu cụ thể có thể thực hiện riêng biệt.

Nguyên tắc:
- Câu hỏi đơn giản → 1 sub-task
- Câu hỏi phức tạp → tối đa 5 sub-task, độc lập nhau
- Nếu câu hỏi yêu cầu phân tích dữ liệu bảng, thống kê, hoặc vẽ biểu đồ → tạo sub-task cho pandas
- Mỗi sub-task là chuỗi văn bản mô tả rõ ràng
- Viết bằng ngôn ngữ câu hỏi gốc

## TRƯỜNG HỢP ĐẶC BIỆT (THÊM MỚI)
Nếu các cột đó tồn tại nhưng ở dạng khác về tên thì tự suy luận tên cột dựa vào df.info()
{dataframe_info} và sẽ đặt câu hỏi dựa vào tên cột gốc của dataset

Ví dụ:
"có thể trực quan hóa cột doanh thu." ở đây doanh thu có thể hiểu là sales

## Cấu trúc dữ liệu (nếu có)
{dataframe_head}

{user_memory_section}

## Câu hỏi
{question}"""


# ---------------------------------------------------------------------------
# Node 3 — Knowledge Router
# ---------------------------------------------------------------------------

KNOWLEDGE_ROUTER_PROMPT = """\
Bạn là router quyết định nguồn kiến thức phù hợp nhất cho sub-task.

- "data_analyzer"  : Phân tích sâu số liệu, tính toán thống kê, tương quan, tập hợp dữ liệu từ file CSV/Excel đã upload.
- "visualizer"     : Trực quan hóa dữ liệu bằng biểu đồ (bar, line, boxplot, scatter...). Yêu cầu này luôn bao gồm việc trích xuất số liệu thống kê đi kèm biểu đồ đó.
- "rag"            : Tìm kiếm thông tin văn bản từ tài liệu PDF/Word đã upload (quy trình, chính sách, mô tả).
- "web"            : Cần thông tin thời sự, dữ liệu thị trường trực tuyến, hoặc kiến thức không có trong tài liệu.
- "llm_knowledge"  : Trả lời bằng kiến thức nội tại của LLM (định nghĩa, dịch thuật, giải thích logic đơn giản).

## QUY TẮC ƯU TIÊN
- Nếu câu hỏi yêu cầu vẽ biểu đồ, đồ thị, trực quan hóa -> dùng "visualizer".
- Nếu câu hỏi yêu cầu tính toán, thống kê, phân tích số liệu (không biểu đồ) -> dùng "data_analyzer".
- Chỉ dùng "rag" khi tóm tắt tài liệu (content_summary) không trống.

## Tóm tắt nội dung văn bản(nếu có)
{content_summary}

## Cấu trúc dữ liệu bảng
{dataframe_head}
{dataframe_info}

## Sub-task cần xử lý
{current_task}"""


# ---------------------------------------------------------------------------
# Node 4 — HyDE (Hypothetical Document Embedding)
# ---------------------------------------------------------------------------

HYDE_PROMPT = """\
Hãy viết một đoạn văn ngắn (1–2 câu) giả định rằng đây là câu trả lời lý tưởng \
cho yêu cầu bên dưới. Đoạn này sẽ dùng để tìm kiếm ngữ nghĩa trong vector database.

## Sub-task
{current_task}
"""


# ---------------------------------------------------------------------------
# Node 5c — LLM Knowledge
# ---------------------------------------------------------------------------

LLM_KNOWLEDGE_PROMPT = """\
Trả lời câu hỏi / yêu cầu bên dưới dựa trên kiến thức chung của bạn.
Trả lời ngắn gọn, chính xác, dùng Markdown nếu hữu ích.

## Yêu cầu
{current_task}
"""


# ---------------------------------------------------------------------------
# Node 5a — RAG Answerer (after retrieval)
# ---------------------------------------------------------------------------

RAG_ANSWER_PROMPT = """\
Dựa trên các đoạn ngữ cảnh đã truy xuất từ tài liệu, hãy trả lời sub-task dưới đây.

Nguyên tắc:
- Chỉ sử dụng thông tin có trong ngữ cảnh.
- Nếu ngữ cảnh không đủ để trả lời, hãy báo rõ "Không tìm thấy thông tin đủ để trả lời hoàn thiện".
- Trình bày chính xác số liệu, tên thực thể.
- Ghi rõ nguồn: "(từ tài liệu)".

## Ngữ cảnh
{all_context}

## Sub-task
{current_task}

## Câu trả lời:"""


# ---------------------------------------------------------------------------
# Node 6 — Validator
# ---------------------------------------------------------------------------

VALIDATOR_PROMPT = """\
Đánh giá mức độ đầy đủ của ngữ cảnh đã truy xuất để trả lời sub-task.

Thang điểm:
- 0.9–1.0 : Ngữ cảnh đầy đủ, chứa đúng thông tin cần thiết
- 0.7–0.89: Tương đối đủ, có thể suy luận được
- 0.5–0.69: Thiếu một phần quan trọng
- 0.0–0.49: Không liên quan hoặc thiếu nghiêm trọng

is_valid = True khi score >= 0.7

## Sub-task
{current_task}

## Ngữ cảnh đã truy xuất
{all_context}"""


# ---------------------------------------------------------------------------
# Data Analyzer Code Generator (Statistical Analysis)
# ---------------------------------------------------------------------------

DATA_ANALYZER_PROMPT = """\
Bạn là chuyên gia phân tích dữ liệu Python. Nhiệm vụ của bạn là viết code xử lý dữ liệu và trả về kết quả số liệu chi tiết.

## NGỮ CẢNH QUAN TRỌNG (BẮT BUỘC)
- DataFrame `df` ĐÃ TỒN TẠI trong môi trường runtime.
- `df` chứa dữ liệu THẬT, KHÔNG phải dữ liệu mẫu.
- Thông tin bên dưới CHỈ để tham khảo cấu trúc.

## QUY TẮC BẮT BUỘC
1. Chỉ viết code Python thuần, KHÔNG markdown, KHÔNG giải thích.
2. Dựa vào thông số cột từ `df.info()` và `df.head()` để viết code chính xác.
3. KHÔNG vẽ biểu đồ (Cấm matplotlib, seaborn, plt.show...).
4. Kết quả phân tích (số liệu, bảng tóm tắt, insight số) PHẢI được gán vào biến `result` (phải là một dict).
   - Ví dụ: result = {{"avg_salary": 5000, "count": 100, "grouped_data": ...}}

## Nhiệm vụ
{task}

## Cấu trúc DataFrame (THAM KHẢO)
{dataframe_head}
{dataframe_info}
tên cột hoặc thông tin cột phải dựa hoàn toàn vào Cấu trúc DataFrame (THAM KHẢO)
Code:"""


# ---------------------------------------------------------------------------
# Visualizer Code Generator (Charts + Underlying Stats)
# ---------------------------------------------------------------------------

VISUALIZER_PROMPT = """\
Bạn là chuyên gia trực quan hóa dữ liệu Python. Nhiệm vụ của bạn là vẽ biểu đồ ĐỒNG THỜI trích xuất các số liệu thống kê cơ bản của biểu đồ đó.

## NGỮ CẢNH QUAN TRỌNG (BẮT BUỘC)
- DataFrame `df` ĐÃ TỒN TẠI trong môi trường runtime.
- Đường dẫn lưu biểu đồ là: `{chart_path}`.

## QUY TẮC BẮT BUỘC
1. Chỉ viết code Python thuần, KHÔNG markdown, KHÔNG giải thích.
2. Xem kỹ cấu trúc cột trong `df.info()` để tránh lỗi `KeyError`.
3. LUÔN lưu biểu đồ bằng:
   - fig.savefig("{chart_path}", dpi=150, bbox_inches="tight")
   - plt.close(fig)
3. CỰC KỲ QUAN TRỌNG: Bạn PHẢI trích xuất các số liệu then chốt xuất hiện trong biểu đồ và gán vào biến `result` (phải là một dict). 
   - Ví dụ: Nếu vẽ Boxplot, `result` phải chứa min, max, median, quartiles.
   - Nếu vẽ Bar chart top 5, `result` phải chứa danh sách 5 tên và giá trị tương ứng.
   - Mục đích: Để hệ thống có thể phân tích biểu đồ dựa trên số liệu thực tế chứ không chỉ nhìn hình.

## Nhiệm vụ
{task}

## Cấu trúc DataFrame (THAM KHẢO)
{dataframe_head}
{dataframe_info}

Code:"""

PANDAS_AGENT_SUFFIX = """\
Bắt đầu! Nhớ lưu chart nếu được yêu cầu và in marker CHART_SAVED."""


# ---------------------------------------------------------------------------
# Node 8 — Global Summary
# ---------------------------------------------------------------------------

GLOBAL_SUMMARY_PROMPT = """\
Tạo bản tóm tắt toàn diện từ các kết quả phân tích sub-task bên dưới.

Yêu cầu:
- Tổng hợp các điểm chính, loại bỏ trùng lặp
- Tối đa 500 từ, dùng Markdown

## Kết quả các sub-task
{documents}

## Tổng hợp:"""


# ---------------------------------------------------------------------------
# Node 9 — Synthesizer
# ---------------------------------------------------------------------------

SYNTHESIZER_PROMPT = """\
Bạn là Chuyên gia Phân tích Dữ liệu Hệ thống.
Nhiệm vụ của bạn là tổng hợp các KẾT QUẢ THÔ từ nhiều công cụ khác nhau để trả lời câu hỏi của người dùng một cách chính xác, sâu sắc và chuyên nghiệp.

## DỮ LIỆU ĐẦU VÀO (TỪ CÁC SUB-TASKS)
{all_context}

## QUY TẮC TỔNG HỢP (BẮT BUỘC)
1. **Phân tích Đa chiều**: Kết hợp số liệu từ Pandas (thống kê) với thông tin văn bản từ RAG/Web để đưa ra cái nhìn toàn diện.
2. **Trung thực với Dữ liệu**: Chỉ sử dụng số liệu và sự thật có trong đầu vào. Không bịa đặt thông tin.
3. **Định dạng Markdown Chuyên nghiệp**:
   - Sử dụng Tiêu đề (H2, H3), Bảng (Table), Danh sách gạch đầu dòng.
   - Nhấn mạnh các số liệu quan trọng bằng **chữ đậm**.
4. **Trích dẫn nguồn**: Ghi rõ nguồn cho từng ý: (từ tài liệu), (từ web), (từ phân tích dữ liệu).
5. **Hình ảnh/Biểu đồ**: Nếu có thông tin về biểu đồ (CHART_PATH), hãy thông báo: "📊 **Biểu đồ đã được tạo và hiển thị bên dưới.**"
6. **Xử lý Lỗi (QUAN TRỌNG)**:
   - Nếu một kết quả thô có nhãn **[ERROR]**, hãy giải thích lỗi đó cho người dùng một cách trung thực. Ví dụ: "Không tìm thấy cột 'Lợi nhuận' trong tập dữ liệu."
   - **Tuyệt đối không** sử dụng thông tin từ "Bộ nhớ dài hạn" để suy đoán hoặc trả lời thay cho một sub-task bị lỗi nếu thông tin đó không liên quan trực tiếp đến yêu cầu hiện tại.
7. **Nhận xét & Khuyến nghị (Insights)**:
   - Nếu không có lỗi, dựa trên dữ liệu, hãy đưa ra ít nhất 1-2 nhận xét sâu sắc (ví dụ: xu hướng, điểm bất thường).
   - Nếu có lỗi, hãy gợi ý cách khắc phục (ví dụ: kiểm tra lại tên cột hoặc upload đúng file).

## PHONG CÁCH PHỤC VỤ
- Ngôn ngữ: Trả lời bằng ngôn ngữ của câu hỏi gốc.
- Giọng văn: Thân thiện, giải thích các số liệu phức tạp một cách dễ hiểu cho người không chuyên.
- Trình bày: Sạch sẽ, dễ theo dõi, tập trung vào giá trị cốt lõi khách hàng cần.

{user_memory_section}

## CÂU HỎI GỐC
{question}

## CÂU TRẢ LỜI HOÀN CHỈNH:"""


# ---------------------------------------------------------------------------
# Fallback Messages & Error Responses
# ---------------------------------------------------------------------------

REJECTION_FALLBACK_ANSWER = """\
Mình chưa hiểu rõ ý bạn lắm 🤔

Bạn có thể nói rõ hơn một chút được không? 
Hoặc nếu bạn muốn, mình có thể tự chọn cách xử lý phù hợp cho bạn 👍
"""

TECHNICAL_ERROR_RESPONSE = "Đã xảy ra lỗi khi xử lý yêu cầu của bạn: {error}"
