# PyPI Package Upload Instructions

This document contains instructions for building and uploading the spatial-reasoning package to PyPI.

## Prerequisites

1. **Create PyPI Account**
   - Register at https://pypi.org/account/register/
   - Create an API token at https://pypi.org/manage/account/token/
   - Save the token securely

2. **Install Required Tools**
   ```bash
   pip install --upgrade pip setuptools wheel twine build
   ```

## Building the Package

1. **Navigate to the package directory** (where setup.py is located):
   ```bash
   cd /home/qasim/code/exp/spatial_reasoning
   ```

2. **Clean previous builds** (if any):
   ```bash
   rm -rf build/ dist/ *.egg-info/
   ```

3. **Build the package**:
   ```bash
   python -m build
   ```
   
   This will create:
   - `dist/spatial-reasoning-0.1.0.tar.gz` (source distribution)
   - `dist/spatial_reasoning-0.1.0-py3-none-any.whl` (wheel distribution)

## Testing Locally (Optional but Recommended)

1. **Create a test virtual environment**:
   ```bash
   python -m venv test_env
   source test_env/bin/activate  # On Windows: test_env\Scripts\activate
   ```

2. **Install the package locally**:
   ```bash
   pip install dist/spatial_reasoning-0.1.0-py3-none-any.whl
   ```

3. **Test the installation**:
   ```python
   from spatial_reasoning import detect
   print(detect.__doc__)
   ```

4. **Test CLI**:
   ```bash
   spatial-reasoning --help
   ```

## Uploading to PyPI

### Option 1: Upload to Test PyPI First (Recommended)

1. **Upload to Test PyPI**:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```
   
   When prompted:
   - Username: `__token__`
   - Password: Your test PyPI API token

2. **Test installation from Test PyPI**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ spatial-reasoning
   ```

### Option 2: Direct Upload to PyPI

1. **Upload to PyPI**:
   ```bash
   python -m twine upload dist/*
   ```
   
   When prompted:
   - Username: `__token__`
   - Password: Your PyPI API token

## Using .pypirc for Authentication (Optional)

Create `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

Then upload without prompts:
```bash
python -m twine upload --repository pypi dist/*
```

## Post-Upload Steps

1. **Verify on PyPI**:
   - Visit https://pypi.org/project/spatial-reasoning/
   - Check that all information displays correctly

2. **Install and test**:
   ```bash
   pip install spatial-reasoning
   ```

3. **Update GitHub repository**:
   - Tag the release: `git tag v0.1.0`
   - Push tags: `git push --tags`

## Updating the Package

When releasing a new version:

1. Update version in:
   - `ml/__init__.py`
   - `setup.py`
   - `pyproject.toml`

2. Update CHANGELOG or release notes

3. Follow the build and upload steps again

## Common Issues

1. **Name conflicts**: If "spatial-reasoning" is taken, you'll need to choose a different name
2. **Missing dependencies**: Ensure all dependencies are listed in requirements.txt
3. **Import errors**: Test thoroughly before uploading
4. **Authentication issues**: Make sure to use `__token__` as username with API tokens

## Security Notes

- Never commit API tokens to version control
- Use environment variables or .pypirc for tokens
- Consider using GitHub Actions for automated releases