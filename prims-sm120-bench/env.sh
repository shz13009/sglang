# Per-run environment for the PR 1 tests; does not touch ~/.bashrc.
# ~/.local/bin/gcc is an nvim-tools gcc wrapper without cc1plus; put system gcc/g++ first.
cd ~/code/sglang
source .venv/bin/activate
export PATH="/usr/bin:$PATH"
export PATH="$HOME/code/sglang/.venv/bin:$PATH"
export PATH="/usr/local/cuda/bin:$PATH"
