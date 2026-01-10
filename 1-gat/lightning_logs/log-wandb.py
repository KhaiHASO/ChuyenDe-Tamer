import wandb
import os

# --- CẤU HÌNH ---
# 1. Điền API Key của bác vào đây
WANDB_API_KEY = "3010beaefcbb3ca747099418f4dd36cd474cc81c"

# 2. Đường dẫn đến Run (Lấy từ URL hoặc code bác gửi lúc nãy)
# Format: "username/project-name/run-id"
RUN_PATH = "khaihaso/TAMER-Kaggle/x3njhjhx"

# 3. Tên thư mục muốn lưu trên máy
DOWNLOAD_FOLDER = "wandb_data_downloaded"

# ---------------------------------------------------------
def download_everything():
    print(f"🔄 Đang kết nối tới Run: {RUN_PATH}...")
    wandb.login(key=WANDB_API_KEY)
    api = wandb.Api()
    
    try:
        run = api.run(RUN_PATH)
    except Exception as e:
        print(f"❌ Lỗi: Không tìm thấy Run. Kiểm tra lại đường dẫn! ({e})")
        return

    # Tạo thư mục lưu trữ
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    print(f"📂 Dữ liệu sẽ được lưu tại: {os.path.abspath(DOWNLOAD_FOLDER)}\n")

    # --- PHẦN 1: TẢI CÁC FILE CƠ BẢN (Logs, Config, Requirements...) ---
    print("⬇️  Đang tải Files (Logs, Config)...")
    files = run.files()
    for file in files:
        # Bỏ qua các file nằm trong thư mục artifact/ (vì sẽ tải ở phần 2)
        if file.name.startswith("artifact/"):
            continue
            
        print(f"   - Downloading: {file.name}")
        file.download(root=DOWNLOAD_FOLDER, replace=True)

    # --- PHẦN 2: TẢI ARTIFACTS (Model Checkpoints, Tables...) ---
    print("\n⬇️  Đang tải Artifacts (Model Checkpoints, Predictions Table)...")
    artifacts = run.logged_artifacts()
    
    if len(artifacts) == 0:
        print("   ⚠️ Không tìm thấy Artifact nào (Có thể Model chưa được upload).")
    
    for artifact in artifacts:
        print(f"   - Artifact: {artifact.name} ({artifact.type})")
        # Tải artifact về thư mục con
        artifact_dir = os.path.join(DOWNLOAD_FOLDER, "artifacts", artifact.name)
        artifact.download(root=artifact_dir)
        print(f"     -> Đã lưu tại: {artifact_dir}")

    print("\n✅ XONG! Đã tải hết toàn bộ về máy.")

if __name__ == "__main__":
    download_everything()