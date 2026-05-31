# Prime Pattern Explorer

CUDA-based exploratory tool for scanning prime numbers and searching for unusual local prime-pattern structures.

The program tests large sets of prime offset patterns, generates BMP visualisations, outputs CSV summaries, and supports automated exploration/zooming through Python scripts.

## Main files

- `prime2.cu` - main CUDA/C++ prime-pattern explorer
- `makefile` - build/run/clean commands
- `auto_explore.py` - automated broad scan and zoom/chase script
- `deep_quad_run_fixed.sh` - batch runner for multi-chunk exploration

Generated BMP/CSV/result folders are ignored by Git.
