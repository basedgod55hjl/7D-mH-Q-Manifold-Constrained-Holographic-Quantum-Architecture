import os
import subprocess
import sys

def run_all():
    print("🚀 Running 7D Crystal Manifold Tests...")
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    
    success = True
    for f in sorted(test_files):
        print(f"\n--- Running: {f} ---")
        result = subprocess.run([sys.executable, os.path.join(test_dir, f)])
        if result.returncode != 0:
            print(f"❌ Failed: {f}")
            success = False
            
    if not success:
        sys.exit(1)
    else:
        print("\n✅ All manifold tests passed successfully.")

if __name__ == '__main__':
    run_all()
