# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Thị Hải Mi  **MSSV**: 2A202602667  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16 GB (14.6 GB khả dụng, sm_75 → fp16)`

> Mọi con số dưới đây lấy từ lần chạy đầy đủ (`eval_limit: null`, `smoke_mode: false`,
> 50 câu target + 15 câu regression) và khớp với file trong `results/`. Ngoại lệ duy nhất là
> bảng đường loss theo step ở mục 4, lấy từ log huấn luyện (log không nằm trong `results/`)
> — đã ghi chú rõ tại chỗ.

---

## 0. Lựa chọn thí nghiệm & lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định tier T4) | Model lớn nhất vừa Colab Free T4 với LoRA 16-bit (peak 8.78 GB / 14.6 GB). Giữ mặc định để số đo so sánh được với số đo tham chiếu của lab. |
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON 4 trường | Mọi nhóm điểm có thang khách quan (so khớp nhãn, parse JSON, keyword) — không cần LLM judge. Không đổi tập eval nên checksum giữ nguyên. |
| Prompt (b) | Giữ nguyên `OPTIMIZED_PROMPT` (SHA `719e74d3b6232053`) | Prompt có schema, liệt kê giá trị hợp lệ và một ví dụ few-shot — đã là một mốc mạnh thật sự (format 1.000). Không làm yếu đi. |

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| Eval | 50 câu target · 15 câu regression — **đóng băng ở NB2, trước khi train** |
| `max_length` | 1024 (mặc định tier T4) — p95 đo được là **98**, gợi ý 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **30** optimizer step (batch hiệu dụng 1 × 16 = 16) |
| LoRA | all-linear (12 module text) · r=16 · α=32 · LR 1e-4 cosine · warmup 3 · fp16 |

**Về `max_length` lệch so với p95.** Token stats: mean 93.1, p95 98, p99 100, **max 101**.
Tôi giữ 1024 của tier thay vì 256. Lý do: với `per_device_batch=1` và `packing=False`,
`max_length` chỉ là trần cắt — không mẫu nào chạm trần (dài nhất 101 token), nên 256 và
1024 cho ra **cùng một dataset** và cùng bộ nhớ. Nếu dùng packing hoặc batch lớn hơn,
256 sẽ là lựa chọn đúng.

**Template có giữ khối `<think>` không?** **Có** — `open_tag_present: true`,
`body_present: true`, verdict *"reasoning preserved — safe to train on traces"*
*(results/template_check.json)*. Tuy nhiên câu trả lời trong corpus là JSON trần, không có
trace suy luận; template tự chèn khối `<think>` rỗng nên không có trace nào để che hay giữ.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` (1 mẫu) | **0.4149** (39 / 94 token) |
| Câu trả lời nằm trong loss | **true** |
| Câu hỏi KHÔNG nằm trong loss | **true** |

Đoạn được tính loss (`assistant-only`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị che (không tính loss): toàn bộ `system` + `user` + `<|im_start|>assistant\n<think>\n\n`.

Đối chứng in ở NB1: với `MASK_MODE=everything` thì supervised = 94/94 (100%) — model sẽ
học cả việc viết lại câu hỏi. 41% < 95% nên mask đúng.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3170.4 |
| (b) base + optimized prompt | **0.765** | 0.791 | 1.000 | 998.5 |
| (c) LoRA fine-tune | **0.970** | **0.611** | 1.000 | 1423.5 |

**(b) có thật sự mạnh hơn (a) không?** **Có**, rất rõ: target 0.000 → 0.765, format
0.000 → 1.000. Với prompt naive ("Phân loại ticket sau."), base model không trả JSON
nào parse được, nên target = 0 dù nó có thể hiểu ticket. Prompt (b) còn nhanh hơn ~3 lần
(998.5 ms so với 3170.4 ms) vì model trả JSON ngắn thay vì viết văn xuôi.

**Có sửa `OPTIMIZED_PROMPT` không?** Không. SHA `719e74d3b6232053` giữ nguyên.

---

## 4. Giải phẫu cấu hình sai (NB4 → chấm ở NB5 §4)

Cả bốn run: cùng base, cùng data, cùng **30 step**, cùng seed. Mỗi run đổi **một** biến.

| Run | Biến đã đổi | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | train s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | — | text-linear (12) | 16 | 32,464,896 | 1e-4 | 0.6260 | **0.970** | 395.6 | 8.78 |
| `attn_only` | vị trí | q,v (2) | 283 *(matched)* | 32,456,704 | 1e-4 | **0.5377** | **0.970** | 260.2 | 8.79 |
| `wrong_lr` | learning rate | text-linear (12) | 16 | 32,464,896 | **1e-5** | 1.5702 | 0.000 | 386.3 | 8.78 |
| `qlora` | độ chính xác base | text-linear (12) | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 460.9 | **3.86** |

`attn_only` lệch ngân sách tham số 8,192 / 32,464,896 = **0.025%** (< 5%).

Đường loss theo step *(từ log huấn luyện, không nằm trong `results/`; ghi ở lần chạy đầy
đủ đầu tiên cùng seed — `final_loss` của `wrong_lr` và `qlora` trùng khớp `runs.csv`)*:

| step | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| correct | 2.163 | 1.385 | 0.141 | 0.028 | 0.017 | 0.027 |
| attn_only | 2.163 | 0.826 | 0.146 | 0.039 | 0.022 | 0.028 |
| wrong_lr | 2.163 | 2.066 | 1.606 | 1.326 | 1.141 | 1.119 |
| qlora | 2.155 | 1.731 | 0.241 | 0.051 | 0.031 | 0.026 |

**Xếp hạng theo target:** correct = attn_only (0.970) > qlora (0.940) > wrong_lr (0.000).
**Xếp hạng theo train loss:** attn_only (0.538) > correct (0.626) > qlora (0.706) > wrong_lr (1.570).
→ Theo loss, `attn_only` đứng đầu một mình; theo target, nó **chỉ hoà**.

**4.1 — `attn_only` vs `correct`.**
Với cùng ~32.46M tham số, `attn_only` (chỉ q,v, rank 283) đạt target 0.970 — **hoà** với
`correct`. Theo train loss thì `attn_only` lại **thắng rõ** (0.538 so với 0.626), vì loss của
nó giảm nhanh hơn ở đoạn đầu (step 10: 0.826 so với 1.385) và `final_loss` là trung bình cả
quá trình chứ không phải loss cuối; đến step 30 hai run gần như bằng nhau (0.028 và 0.027).
Nếu xếp hạng bằng loss, tôi sẽ kết luận sai rằng "dồn rank vào attention tốt hơn". Về
*rank vs vị trí*: run này được thiết kế để "nếu rank là đòn bẩy thì nó sẽ thắng" — và nó
**không thắng**, nên tăng rank lên 283 không mua thêm được độ chính xác nào. Nhưng nó cũng
không thua, nên trên bài toán này tôi **không có bằng chứng** rằng vị trí gắn adapter là đòn
bẩy: tác vụ đủ dễ để cả hai chạm cùng trần, và cả hai cùng mắc đúng một loại lỗi (mục 6).
Điều đo được chắc chắn là chi phí: `attn_only` train nhanh hơn 34% (260.2 s so với 395.6 s)
và suy luận nhanh hơn 39% (870.6 ms so với 1423.5 ms), vì chỉ 2 module mang adapter thay vì 12.

**4.2 — `wrong_lr`.**
Chỉ đổi LR từ 1e-4 xuống 1e-5 (thang full fine-tune). Đường loss vẫn **đi xuống đều**
(2.163 → 1.119) — nhìn riêng thì trông như "đang học, chỉ chậm hơn". Nhưng trên tập target
nó đạt **0.000 target và 0.000 format**: không một câu trả lời nào là JSON hợp lệ, và latency
5121.2 ms còn cao hơn baseline (a) — tức model vẫn cư xử như base model với prompt naive,
sinh văn xuôi dài. Nếu chỉ nhìn loss mà không biết LR, tôi sẽ kết luận "cần train thêm vài
epoch", trong khi nguyên nhân thật là LoRA cần LR khoảng 10 lần full-FT. Mức loss ~1.1 vẫn
cách rất xa vùng ~0.02 mà ba run còn lại đạt được — ở mức đó model chưa học được định dạng.

**4.3 — `qlora`.**
QLoRA 4-bit giảm peak VRAM từ 8.78 GB xuống **3.86 GB (−56%)**. Cái giá: target giảm từ
0.970 xuống **0.940** (12 trường sai so với 6 — gấp đôi), train chậm hơn 17% (460.9 s so với
395.6 s) và suy luận chậm hơn 22% (1732.3 ms so với 1423.5 ms) do phải giải lượng tử hoá.
Số đo **ủng hộ** khuyến nghị "không dùng QLoRA cho Qwen3.5" trong trường hợp bản 16-bit đã
vừa bộ nhớ (như model 4B trên T4): mất độ chính xác và mất tốc độ để đổi lấy VRAM mà ta không
cần. QLoRA chỉ đáng dùng khi bản 16-bit không vừa GPU.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.180` (ngưỡng −0.020) · `valid_trace_rate = 0.00`

Lý do in ra: *"general capability regressed by 0.180 (tolerance 0.020). See deck §6.3 — add
1-5% replay data."*

**Diễn giải.** Fine-tune **thắng trên tác vụ**: target 0.970 so với 0.765 của prompt tối ưu
(+0.205), giữ format 1.000. Nhưng nó **trả giá bằng năng lực chung**: điểm regression tụt từ
0.791 xuống 0.611, gấp 9 lần ngưỡng cho phép. Đây là quên thảm hoạ (catastrophic forgetting):
225 mẫu train đều cùng một dạng — ticket → JSON — và sau 30 step với loss cuối ~0.02, model bị
kéo mạnh về hành vi "luôn trả lời ngắn theo khuôn". Tập regression chấm bằng keyword recall trên
câu hỏi phổ thông (thủ đô, đơn vị đo, Truyện Kiều…), nên khi model trả lời cụt hoặc lệch định
dạng, keyword không xuất hiện và điểm tụt. Đây là giả thuyết: NB5 không lưu từng câu trả lời
regression nên tôi chưa xác minh được dạng lỗi cụ thể.

**Độ tin cậy của con số −0.180.** Tập regression chỉ có 15 câu, mỗi câu ≈ 0.067 điểm, nên
−0.180 tương đương mất khoảng 3 câu. Bằng chứng trực tiếp cho độ nhiễu: một lần chạy đầy đủ
trước đó với **cùng cấu hình và cùng seed** (kết quả bị mất do Colab reset trước khi tải về,
nên không nằm trong `results/`) cho regression 0.522, tức Δ = −0.269 — trong khi target vẫn là
0.970. Biên độ nhiễu giữa hai lần chạy (≈ 0.09) lớn, nhưng **cả hai lần đều vượt xa ngưỡng
0.02**, nên phán quyết FAILED là vững. Ngoài ra trần của thang đo này dưới 1.0 kể cả với model
tốt: câu "kể tên một loại trái cây nhiệt đới" có 5 keyword nên trả lời đúng một loại chỉ được
0.2. Vì (b) và (c) dùng cùng thang đo, phép so sánh vẫn công bằng.

`valid_trace_rate = 0.00` là kỳ vọng: corpus không có trace suy luận và template chèn khối
`<think>` rỗng, nên bản fine-tune không sinh trace. Con số này không phải bằng chứng về
reasoning collapse — muốn đo điều đó cần dữ liệu có trace (bonus B3).

---

## 6. Định tính — có cả ca THUA

`results/qualitative.json` lưu dự đoán của bản fine-tune cho cả 50 câu; dự đoán từng câu của
baseline (b) không được lưu, nên tôi so với **nhãn đúng**. Mỗi trường đúng = 0.25.

| # | Ticket (rút gọn) | Nhãn đúng | (c) fine-tune | Kết quả |
|---|---|---|---|---|
| 47 | …ốp lưng điện thoại… Shipper không gọi. Hỏi cho biết thôi. Shop hỗ trợ tốt. | van_chuyen · thap · ốp lưng điện thoại · tich_cuc | van_chuyen · thap · ốp lưng điện thoại · … | ✅ 1.00 |
| 48 | …ốp lưng điện thoại… Giá bao nhiêu. Mong shop phản hồi. | hoi_thong_tin · trung_binh · ốp lưng điện thoại · trung_tinh | hoi_thong_tin · trung_binh · ốp lưng điện thoại · … | ✅ 1.00 |
| 8 | …chuột không dây… Bảo hành bao lâu. **Không vội.** Mình vẫn tin tưởng shop. | hoi_thong_tin · thap · chuột không dây · tich_cuc | hoi_thong_tin · **thap** · chuột không dây · … | ✅ 1.00 |
| 3 | …bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều. | hoan_tien · **thap** · bình giữ nhiệt · tich_cuc | hoan_tien · **trung_binh** · bình giữ nhiệt · … | ❌ **FT sai** 0.75 |
| 5 | …nồi chiên không dầu… Thiếu phụ kiện. **Khi nào tiện.** Cho tôi hỏi. | san_pham_loi · **thap** · nồi chiên không dầu · trung_tinh | san_pham_loi · **trung_binh** · nồi chiên không dầu · … | ❌ **FT sai** 0.75 |
| 41 | …đèn bàn LED… Giao hàng chậm. **Khi nào tiện.** Cảm ơn shop nhiều. | van_chuyen · **thap** · đèn bàn LED · tich_cuc | van_chuyen · **trung_binh** · đèn bàn LED · … | ❌ **FT sai** 0.75 |
| 46 | …đèn bàn LED… Sai màu. **Khi nào tiện.** Shop hỗ trợ tốt. | san_pham_loi · **thap** · đèn bàn LED · tich_cuc | san_pham_loi · **trung_binh** · đèn bàn LED · … | ❌ **FT sai** 0.75 |

*(Dự đoán trong `qualitative.json` bị cắt ở ~90 ký tự nên trường `sentiment` thường không hiển
thị; điểm 0.75 = sai đúng một trường, và `urgency` sai hiển thị rõ, nên `sentiment` đúng.)*

**Mẫu chung ở các ca thua — lỗi của model là một lỗi duy nhất, lặp lại 6 lần.**
Target 0.970 nghĩa là 6 trường sai trên 200. Đọc cả 50 dòng của `qualitative.json`: đúng **6
ticket** bị 0.75, và cả 6 (#3, 5, 12, 39, 41, 46) đều:
- sai **cùng một trường** (`urgency`), **cùng một kiểu** (`thap` → `trung_binh`);
- chứa **cùng một cụm** "Khi nào tiện".

Tập eval có đúng 6 ticket chứa cụm này — tức model sai **6/6** với cụm "Khi nào tiện" và đúng
**100%** mọi trường ở 44 ticket còn lại, kể cả các cụm `thap` khác ("Không vội", "Hỏi cho biết
thôi"). Điều này lạ, vì trong 225 mẫu train có **30 mẫu** chứa "Khi nào tiện", và **cả 30 đều
nhãn `thap`** — đây là cụm urgency phổ biến nhất trong dữ liệu, với tín hiệu hoàn toàn nhất quán.
Giả thuyết của tôi: "Khi nào tiện" trong tiếng Việt có thể đọc như một **câu hỏi** ("khi nào thì
tiện?") — tức khách đang chờ phản hồi — nên base model có prior mạnh về `trung_binh`, và 30 mẫu
× 2 epoch chưa đủ để ghi đè prior đó. Đây là giả thuyết chưa kiểm chứng; cách kiểm tra là đo
baseline (b) riêng trên 6 ticket này và xem log-prob của từng nhãn `urgency`.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không deploy** bản fine-tune này ở dạng hiện tại. Trên tác vụ triage nó thắng
rõ prompt tối ưu (+0.205 target, 0.970 so với 0.765), nên fine-tune *có* giá trị với bài toán này —
prompt dù có schema và ví dụ vẫn để sót gần ¼ số trường. Nhưng cổng hồi quy FAIL vì năng lực chung
tụt 0.180, gấp 9 lần ngưỡng cho phép, và một lần chạy lặp lại còn cho mức tụt lớn hơn (−0.269).
Nguyên nhân có tính nhân quả khá rõ: dữ liệu train đồng nhất 100% (một dạng đầu vào, một dạng đầu
ra), cộng với 2 epoch đủ để loss xuống ~0.02, đã kéo phân phối đầu ra của model về một khuôn duy
nhất. Nếu hệ thống chỉ nhận ticket CSKH thì có thể chấp nhận; nhưng nếu cùng model còn trả lời
câu hỏi khác thì không. Cách sửa hợp lý nhất là trộn 1–5% dữ liệu phổ thông (replay) vào train rồi
chạy lại cổng hồi quy, hoặc giữ adapter tách riêng và chỉ bật nó cho luồng ticket. Trước khi
deploy cũng cần sửa lỗi hệ thống ở cụm "Khi nào tiện" — một lỗi 6/6 sẽ xếp sai mức ưu tiên của
mọi khách dùng cách nói đó.

Về đòn bẩy: **learning rate** là đòn bẩy lớn nhất về độ chính xác (sai LR → target 0); **mask**
quyết định model học đúng phần trả lời; còn **vị trí adapter vs rank** (Δ target 0.000) và
**16-bit vs 4-bit** (Δ 0.030) chủ yếu khác nhau ở chi phí — tốc độ và VRAM. Nhưng đòn bẩy quyết
định phán quyết cuối cùng lại là **độ đa dạng của dữ liệu**: đó là thứ gây ra FAIL.

**Ba điều tôi học được**
1. **Chấm ít câu thì kết luận có thể ngược hẳn.** Lần đầu tôi chạy bản thử với `EVAL_LIMIT=8`
   cho nhanh, và thấy kết quả PASSED, regression còn tăng +0.125 — lúc đó tôi đã nghĩ là xong
   rồi. Đến khi chạy đủ 50 + 15 câu thì phán quyết lại là FAILED, regression −0.180. Cùng một
   pipeline, chỉ khác số câu chấm. Hai lần chạy đầy đủ cũng lệch nhau khoảng 0.09 ở regression,
   nên giờ tôi hiểu vì sao lab bắt chạy đủ tập eval trước khi nộp: 8 câu, hay thậm chí 15 câu,
   là quá ít để tin vào một con số.
2. **Loss đẹp chưa chắc model đã tốt.** Tôi từng mặc định nhìn loss để biết run nào tốt hơn.
   Ở lab này `attn_only` có train loss thấp nhất nhưng trên tác vụ chỉ hoà với `correct`; còn
   `wrong_lr` có loss giảm đều đặn, trông như đang học, mà target lại bằng 0. Từ giờ tôi sẽ
   luôn chấm trên chính tác vụ cần làm, coi loss chỉ là tín hiệu tham khảo.
3. **Nên đọc từng câu sai, đừng chỉ nhìn điểm trung bình.** Con số 0.970 làm tôi nghĩ 3% còn lại
   là lỗi lặt vặt rải rác. Đọc kỹ `qualitative.json` mới thấy cả 6 lỗi thực ra là **một** lỗi:
   ticket nào có cụm "Khi nào tiện" cũng bị đoán sai mức gấp. Một điểm trung bình đẹp vẫn có thể
   che đi một lỗi có hệ thống, và chính bảng định tính mới cho tôi biết cần sửa dữ liệu ở đâu.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 3% dữ liệu hỏi-đáp phổ thông vào train set và chạy lại
NB3 + NB5 để xem regression có quay về ≥ 0.771 (ngưỡng −0.02) mà vẫn giữ target > 0.765 không;
lưu dự đoán từng câu của baseline (b) và của tập regression để so sánh định tính trực tiếp; và
thêm ticket "Khi nào tiện" đa dạng hơn (đặt ở vị trí khác trong câu) để kiểm tra giả thuyết ở mục 6.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
