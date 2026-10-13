#!/bin/zsh

# Script chạy cập nhật tất cả bản án còn lại theo các đợt nhỏ (batch 5)
# để tránh rate limit của Gemini và tiết kiệm token.

# Chạy từ thư mục gốc dự án
cd "$(dirname "$0")/../../.."

echo "🚀 Bắt đầu quá trình cập nhật 107 bản án cuốn chiếu..."

# Tổng cộng 107 bản án, mỗi đợt chạy 5 bản án -> cần chạy khoảng 22 đợt
for i in {1..22}; do
  echo "\n========================================"
  echo "📦 ĐỢT CHẠY THỨ $i / 22"
  echo "========================================"
  
  .venv/bin/python backend/scripts/Evaluation/process_judgments.py --update-batch --limit 5
  
  # Kiểm tra mã lỗi của script
  if [ $? -ne 0 ]; then
    echo "⚠️ Phát hiện lỗi trong đợt chạy thứ $i. Dừng lại."
    exit 1
  fi
  
  # Nghỉ 15 giây để hồi token của API Gemini
  echo "⏰ Đang nghỉ 15 giây trước đợt tiếp theo để tránh rate limit..."
  sleep 15
done

echo "\n✅ Hoàn tất toàn bộ các bản án!"
