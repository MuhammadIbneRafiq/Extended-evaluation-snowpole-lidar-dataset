import subprocess
import shlex
from pathlib import Path

def run_commands_from_file(commands_file: Path):
    if not commands_file.exists():
        print(f"❌ File not found: {commands_file}")
        return

    with commands_file.open('r') as f:
        lines = []
        in_code_block = False
        for line in f:
            line = line.strip()
            if line.startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block and line and not line.startswith('#'):
                lines.append(line)

    print(f"✅ Found {len(lines)} commands to run.\n")

    for i, cmd in enumerate(lines, start=1):
        print(f"🔁 Running command {i}/{len(lines)}:")
        print(f"   {cmd}")
        args = shlex.split(cmd)
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            print(f"   ✅ Success (exit {result.returncode})")
            if result.stdout:
                print("    • stdout:", result.stdout.strip())
            if result.stderr:
                print("    • stderr:", result.stderr.strip())
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed (exit {e.returncode})")
            if e.stdout:
                print("    • stdout:", e.stdout.strip())
            if e.stderr:
                print("    • stderr:", e.stderr.strip())
        print()

if __name__ == "__main__":
    commands_file = Path(r"C:\Users\x1 yoga\Documents\RA_5m_5L_6m_6L_7m_8m_9m_10m_11m\predict_commands.md")
    run_commands_from_file(commands_file)
