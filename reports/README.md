# Hướng dẫn đọc báo cáo

Thư mục này lưu hai loại đánh giá khác nhau. Không nên dùng kết quả benchmark nhanh để thay cho bộ đánh giá gold.

## Cấu trúc thư mục

```text
reports/
├── README.md
├── benchmark_report.md
├── langsmith_trace_screenshot.png
└── gold/
    ├── report.md
    ├── calibrated_report.md
    ├── calibration.json
    └── disagreement_analysis.md
```

## Benchmark nhanh

[benchmark_report.md](benchmark_report.md) so sánh `single-agent` và `multi-agent` trên một câu hỏi. Báo cáo ghi lại độ trễ, token, chi phí ước tính, điểm chất lượng cấu trúc và độ phủ trích dẫn. Đây là lần chạy minh họa, không đủ để kết luận hệ thống nào tốt hơn.

[langsmith_trace_screenshot.png](langsmith_trace_screenshot.png) là ảnh trace dùng để kiểm tra luồng chạy và các bước gọi tác nhân.

## Bộ đánh giá gold

Bộ gold do nhóm tự xây dựng cho bài lab, không phải benchmark chính thức. Lần chạy hiện tại dùng 12 trường hợp, hai chế độ và một lần lặp, tạo ra 24 đầu ra ẩn danh. Hai chế độ dùng `nvidia/nemotron-nano-9b-v2:free`; `gpt-4o-mini` chấm theo rubric trước khi người đánh giá kiểm tra lại.

Kết quả hiệu chuẩn dựa trên 174 quyết định ở cấp độ claim:

| Chỉ số | Kết quả | Ngưỡng tin cậy |
|---|---:|---:|
| Đồng thuận chính xác | 77,6% | 80% |
| Cohen's kappa | 0,564 | 0,70 |

Mô hình chấm chưa đạt cả hai ngưỡng, vì vậy kết quả hiện có trạng thái **tạm thời**. Chưa thể kết luận `baseline` hay `multi-agent` tốt hơn.

Các tệp liên quan:

- [gold/report.md](gold/report.md): kết quả do mô hình chấm trước hiệu chuẩn.
- [gold/calibrated_report.md](gold/calibrated_report.md): báo cáo kèm kết quả đối chiếu với người đánh giá.
- [gold/calibration.json](gold/calibration.json): số liệu hiệu chuẩn ở dạng máy đọc được.
- [gold/disagreement_analysis.md](gold/disagreement_analysis.md): phân tích 39 trường hợp bất đồng và các giới hạn của rubric.

Khi trích dẫn kết quả của bài lab, nên dùng `calibrated_report.md` cùng `disagreement_analysis.md` và giữ nguyên cảnh báo về trạng thái tạm thời.
