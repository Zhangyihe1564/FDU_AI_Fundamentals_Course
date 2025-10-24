import subprocess
import sys


def install_requirements():
    """Installs the required packages for the project."""

    print("⏳ Installing base requirements ...")
    cmd = ["python", "-m", "pip", "install", "-r"]
    cmd.append("requirements.txt")
    process_install = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process_install.returncode != 0:
        raise Exception("😭 Failed to install base requirements")
    else:
        print("✅ Base requirements installed!")
    print("⏳ Installing Git LFS ...")
    process_lfs = subprocess.run(["apt", "install", "git-lfs"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process_lfs.returncode == -1:
        raise Exception("😭 Failed to install Git LFS and soundfile")
    else:
        print("✅ Git LFS installed!")

    transformers_cmd = "python -m pip install transformers==4.13.0 datasets==2.8.0".split()
    process_scatter = subprocess.run(
        transformers_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

