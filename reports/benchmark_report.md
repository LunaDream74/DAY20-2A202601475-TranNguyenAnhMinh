# Báo cáo benchmark

> Điểm chất lượng chỉ phản ánh cấu trúc của câu trả lời, không khẳng định nội dung đúng về mặt sự thật. Chi phí ước tính chỉ tính token của mô hình, chưa gồm phí dùng công cụ tìm kiếm web của nhà cung cấp.

Câu hỏi: "Khi nào nhóm phát triển phần mềm nên dùng quy trình LLM đa tác nhân thay cho một tác nhân?"

| Lần chạy | Độ trễ (giây) | Token đầu vào | Token đầu ra | Chi phí (USD) | Điểm chất lượng | Độ phủ trích dẫn | Tỷ lệ lỗi |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-agent | 8.21 | 8558 | 448 | 0.0016 | 7.2 | 8% | 0% |
| multi-agent | 14.35 | 10107 | 1023 | 0.0021 | 9.4 | 80% | 0% |

## Nhận xét

Lần chạy `multi-agent` dùng nhiều hơn 18% token đầu vào và 128% token đầu ra so với `single-agent`. Độ trễ cũng cao hơn 75%. Đổi lại, điểm chất lượng cấu trúc tăng 2,2 điểm, chủ yếu nhờ độ phủ trích dẫn tốt hơn. Kết quả này chỉ đến từ một câu hỏi nên chưa đủ để kết luận hệ thống nào tốt hơn.

## Lỗi đã gặp

Trong lúc phát triển, công cụ tìm kiếm web trả về nhiều URL hơn giới hạn năm nguồn đã cấu hình. Tác nhân `Writer` vì thế tạo trích dẫn `[6]`, nằm ngoài danh sách nguồn hợp lệ. Bước kiểm tra đầu ra đã phát hiện lỗi này.

Cách xử lý gồm ba phần: loại các đoạn nghiên cứu chỉ dựa vào URL đã bị bỏ, yêu cầu `Writer` chỉ dùng nhãn có trong danh sách nguồn, và thay nhãn không hợp lệ còn sót lại bằng `[citation unavailable]` đồng thời ghi nhận lỗi. Ở lần benchmark cuối, chỉ báo lỗi của cả hai chế độ đều là 0%.
