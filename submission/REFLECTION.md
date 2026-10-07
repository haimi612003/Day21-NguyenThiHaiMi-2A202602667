# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Cả 6 lỗi của bản fine-tune hoá ra chỉ là một lỗi. Model đoán đúng hết các trường ở 44 ticket,
nhưng ticket nào có cụm "Khi nào tiện" thì đều bị đoán mức gấp là `trung_binh` thay vì `thap`,
6/6 lần. Điều làm tôi bất ngờ là trong dữ liệu train có tới 30 mẫu chứa cụm này và tất cả đều
gắn nhãn `thap`, tức là tín hiệu rất rõ ràng mà model vẫn không học được. Tôi cũng bất ngờ khi
bản chạy thử 8 câu báo PASSED, còn bản chạy đủ lại báo FAILED.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Tôi mất nhiều thời gian nhất ở khâu chạy pipeline, tổng cộng khoảng 2–3 tiếng. Một lần chạy đầy
đủ chỉ mất khoảng 52 phút, nhưng tôi phải chạy tới ba lần: lần đầu để mặc định `EVAL_LIMIT=8` nên
kết quả không nộp được, lần thứ hai máy Colab bị reset trước khi kịp tải kết quả về, đến lần thứ
ba mới lấy được kết quả. Đúng là tôi đoán phần chạy sẽ lâu, nhưng không nghĩ phần lớn thời gian
lại là chạy lại. Bài học nhỏ: chạy xong là phải tải kết quả về ngay.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ fine-tune trên dữ liệu đúng việc thì chỉ có lợi: model giỏi việc đó hơn và mọi thứ
khác giữ nguyên. Lab cho thấy không phải vậy — bản fine-tune giỏi hơn prompt tối ưu ở tác vụ
(+0.205) nhưng lại quên bớt kiến thức chung (regression −0.180). Tôi cũng từng nghĩ loss thấp hơn
nghĩa là model tốt hơn, nhưng `wrong_lr` có loss giảm đều mà target vẫn bằng 0.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude (trong VS Code) khá nhiều: clone repo, tóm tắt yêu cầu lab, giải thích các thư
viện cần cài, hướng dẫn chạy trên Colab, đọc file kết quả, phân tích số liệu và viết phần lớn
báo cáo. Tôi đã đọc lại và đối chiếu các con số với `results/`.

Những chỗ nó chưa đúng hoặc chưa lường trước:
- Nó ước tính pipeline mất khoảng 100–130 phút (theo số liệu trong README), thực tế chỉ khoảng
  52 phút.
- Nó gợi ý chạy trên extension Colab của VS Code và thêm ô tải file về, nhưng chính nó cũng không
  chắc `files.download()` có chạy được trong VS Code không. Cuối cùng tôi chuyển sang Colab
  trên trình duyệt.
- Nó không lường trước việc máy Colab bị reset, nên đoạn kiểm tra chạy sau đó báo lỗi không tìm
  thấy thư mục và kết quả lần chạy đó bị mất.
- Bản nháp báo cáo đầu tiên được viết từ log của lần chạy bị mất, nên có vài nhận định phải sửa
  khi có kết quả thật — ví dụ ban đầu ghi `attn_only` thua `correct` một trường, nhưng ở lần chạy
  nộp bài thì hai run hoà nhau (0.970).

Có một chỗ tôi tưởng nó sai nhưng hoá ra nó đúng: tôi từng hiểu README là được nộp bản
`EVAL_LIMIT=8`, nó chỉ ra trong code và `verify.py` rằng bản nộp phải chấm đủ tập eval.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Chưa train vội. Việc đầu tiên tôi làm là dựng một bộ eval đủ lớn và đóng băng nó, gồm cả câu hỏi
cho đúng tác vụ lẫn câu hỏi kiến thức chung, rồi viết một prompt thật tốt cho base model và đo
nó làm mốc. Chỉ khi prompt tốt nhất vẫn chưa đủ thì mới fine-tune — và khi đó tôi sẽ trộn thêm
một ít dữ liệu phổ thông vào train ngay từ đầu để model không quên những thứ khác.
