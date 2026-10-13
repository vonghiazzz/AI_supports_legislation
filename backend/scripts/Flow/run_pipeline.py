import os
import sys
import argparse
import subprocess

# Xác định thư mục root của backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_script(script_name: str, args: list = None) -> bool:
    script_path = os.path.join(BASE_DIR, "scripts", "Flow", script_name)
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    print(f"\n🚀 ĐANG CHẠY: {' '.join(cmd)}")
    print("="*60)
    
    # Chạy script và stream trực tiếp output ra màn hình console
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1, 
        cwd=BASE_DIR
    )
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    print("="*60)
    
    if process.returncode == 0:
        print(f"✅ HOÀN THÀNH: {script_name} (Thành công)")
        return True
    else:
        print(f"❌ THẤT BẠI: {script_name} (Mã lỗi: {process.returncode})")
        return False

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline điều phối bóc tách và chuẩn hóa bộ luật.")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ luồng từ Bước 1 đến Bước 3")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], help="Chạy một bước cụ thể: 1 (Bóc tách), 2 (Chuẩn hóa/AI), 3 (Nạp Vector)")
    parser.add_argument("--key", type=str, help="Mã định danh (key) của bộ luật muốn Transform (chỉ áp dụng cho Bước 2)")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    args = parser.parse_args()
    
    if args.all:
        print("=== BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE (BƯỚC 1 -> BƯỚC 3) ===")
        # Bước 1: Trích xuất thô
        if not run_script("01_extract_laws.py"):
            print("🛑 Dừng pipeline do Bước 1 thất bại.")
            sys.exit(1)
            
        # Bước 2: Chuẩn hóa AI
        transform_args = [args.key] if args.key else []
        if not run_script("02_transform_laws.py", transform_args):
            print("🛑 Dừng pipeline do Bước 2 thất bại.")
            sys.exit(1)
            
        # Bước 3: Nạp cơ sở dữ liệu vector
        if not run_script("03_load_vectors.py"):
            print("🛑 Dừng pipeline do Bước 3 thất bại.")
            sys.exit(1)
            
        print("\n🎉 CHÚC MỪNG! Pipeline đã chạy hoàn tất thành công từ A-Z!")
        
    elif args.step:
        if args.step == 1:
            run_script("01_extract_laws.py")
        elif args.step == 2:
            transform_args = [args.key] if args.key else []
            run_script("02_transform_laws.py", transform_args)
        elif args.step == 3:
            run_script("03_load_vectors.py")

if __name__ == "__main__":
    main()
