# Project Agent Instructions

- Use the Miniconda environment `py13` by default for all Python commands in this repo.
- Miniconda root: `C:\Users\elupu\miniconda3`; preferred invocation: `conda run -n py13 <command>`.
- Run runtime checks that use `conda run` serially (one command at a time), not in parallel, because parallel launches can trigger transient temp-file locks on this machine.
- When installing missing Python packages, prefer `conda` (or `mamba`) first.
- Use `pip` only if the package is unavailable via conda channels or conda install fails.
