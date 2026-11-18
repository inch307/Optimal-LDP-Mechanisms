"""Portable entry point: works without pip-installing the project."""
import os

threads = os.environ.get("NOUTPUT_THREADS", "1")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = threads

if __name__ == "__main__":
    from src.experiments import main
    main()
