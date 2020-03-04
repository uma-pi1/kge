# Guidelines

Thank you for considering to contribute to this project. We currently have the following rules for contributing to this repository:

- Code formatting using [black](https://github.com/ambv/black) and with default
  settings (line length 88)
- Code documentation following [Google Python Style Guide (Sec.
  3.8)](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings);
  see
  [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Use [type
  annotations](https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html)
  whenever considered helpful
  - For tensors, use `torch.Tensor` only (for all tensors/shapes/devices); do
    not use any more specific annotations (e.g., `torch.LongTensor` refers to a
    CPU tensor only, but not a CUDA tensor)
- Unspecified configuration values are indicated by
  - `''` for strings
  - `-1` for non-negative integers
  - `.nan` for floats
- Don't issue pull request for datasets or experimental code to this repository.
  
