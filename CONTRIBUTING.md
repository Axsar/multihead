# Contributing to MultiHead

Thanks for your interest in contributing! MultiHead is designed to be a collaborative project where everyone can benefit from shared improvements.

## Getting Started

1. **Fork and clone**:
   ```bash
   git clone https://github.com/Axsar/multihead.git
   cd multihead
   ```

2. **Install dependencies**:
   ```bash
   bash scripts/install.sh
   source .venv/bin/activate
   ```

3. **Run tests** to verify everything works:
   ```bash
   python -m pytest tests/ -v
   ```

## Development Workflow

### Making Changes

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and add tests:
   - Add new tests in `tests/test_your_feature.py`
   - Ensure existing tests still pass: `pytest tests/ -v`

3. **Run the full test suite**:
   ```bash
   python -m pytest tests/ -v
   ```

4. **Test manually** (optional but recommended):
   ```bash
   multihead doctor  # Run diagnostics
   multihead serve   # Start the daemon
   multihead shell   # Test interactive terminal
   ```

### Code Style

- Follow existing code patterns
- Use type hints where possible
- Keep functions focused and well-named
- Add docstrings to public APIs

### Testing

- **Unit tests**: Fast, isolated, test one thing
- **Integration tests**: Test component interactions
- **All tests must pass** before submitting a PR

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_orchestrator.py -v

# Run with coverage
pytest tests/ --cov=multihead --cov-report=term
```

## Areas for Contribution

### 1. Adapter Implementations

Add support for new model providers:

- **File**: `src/multihead/adapters/your_adapter.py`
- **Base class**: Inherit from `BaseAdapter`
- **Required methods**: `generate()`, `check_health()`
- **Add to**: `config/heads.yaml` template

Example: vLLM, Llama.cpp, Claude Bedrock, Anthropic API

### 2. Hardware Templates

Add templates for new hardware configurations:

- **File**: `config/templates/your_hardware.yaml`
- **Examples**: See `config/templates/rtx4090.yaml`
- **Use case**: Help users with your GPU/CPU setup get started fast

### 3. Pipeline Recipes

Share useful multi-step pipelines:

- **File**: `config/recipes/your-recipe.yaml`
- **Examples**: See `config/recipes/solver-selection.yaml`
- **Use case**: Solve common tasks (summarization, code review, data extraction)

### 4. Knowledge Extractors

Add extractors for new data sources:

- **File**: `src/multihead/narrative/source_extractors/your_extractor.py`
- **Base class**: Inherit from `BaseExtractor`
- **Use case**: GitHub issues, Slack conversations, Jira tickets

### 5. Bug Fixes and Improvements

- Check [GitHub Issues](https://github.com/Axsar/multihead/issues)
- Look for "good first issue" or "help wanted" labels
- Improve documentation
- Add missing tests

## Submitting Changes

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: Add support for XYZ adapter"
   ```

   Commit message format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Add or update tests
   - `refactor:` Code refactoring

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**:
   - Go to the original repo on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Describe your changes clearly

4. **Address review feedback**:
   - Make requested changes
   - Push new commits to your branch
   - PR will update automatically

## Questions or Issues?

- **Bug reports**: [GitHub Issues](https://github.com/Axsar/multihead/issues)
- **Feature requests**: Open an issue with the "enhancement" label
- **General questions**: Start a [Discussion](https://github.com/Axsar/multihead/discussions)

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on what's best for the project
- Help others learn

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make MultiHead better! 🚀
