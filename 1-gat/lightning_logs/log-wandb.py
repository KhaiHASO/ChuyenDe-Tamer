import wandb

# 1. Đăng nhập (Dùng key của bác)
wandb.login(key="3010beaefcbb3ca747099418f4dd36cd474cc81c")

# 2. Kết nối API
api = wandb.Api()

# 3. Lấy thông tin Run (Lưu ý: bỏ dấu / ở đầu chuỗi)
# Cấu trúc: "username/project-name/run-id"
run = api.run("khaihaso/TAMER-Kaggle/x3njhjhx")

# 4. In ra danh sách file
print("--- DANH SÁCH FILE TRÊN WANDB ---")
files = run.files()
if len(files) == 0:
    print("Chưa có file nào (Có thể do mạng lag hoặc chưa upload xong).")
else:
    for file in run.files():
        print(f"- {file.name}")

# 5. Kiểm tra lịch sử
print("\n--- TRẠNG THÁI HIỆN TẠI ---")
print(f"Tên Run: {run.name}")
print(f"Trạng thái: {run.state}") # running, finished, crashed...