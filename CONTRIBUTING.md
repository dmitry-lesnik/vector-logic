# Contributing to the Rule Engine Project

First off, thank you for considering contributing! We welcome any help you can offer, from reporting bugs to submitting
new features. Following these guidelines helps us collaborate effectively and shows respect for the time of everyone
involved.

## How Can I Contribute?

There are several ways you can contribute to this project:

* **Reporting Bugs**: If you find a bug, please create an issue in the project's issue tracker. Be sure to include a
  clear title, a detailed description of the bug, steps to reproduce it, and any relevant code snippets.

* **Suggesting Enhancements**: If you have an idea for a new feature or an improvement to an existing one, feel free to
  open an issue to discuss it.

* **Code Contributions**: If you'd like to fix a bug or implement a new feature, you can do so by submitting a pull
  request.

## Development Setup

To get started with the code, you'll need to set up your development environment. This project uses **Poetry** for
dependency management and packaging.

1. **Fork & Clone the Repository**:
   First, fork the repository on GitHub, then clone your fork locally.
   ```bash
   git clone https://github.com/dmitry-lesnik/rule-engine.git
   cd rule-engine
   ```

2. **Install Dependencies**:
   This command will create a virtual environment and install all the required packages for both the library and
   development (like `pytest` and `black`).
   ```bash
   poetry install
   ```

3. **Run Tests**:
   Before you start making changes, make sure the existing tests pass.
   ```bash
   poetry run pytest
   ```

## Pull Request Process

1. Create a new branch for your feature or bugfix from the `main` branch.
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes. Ensure you adhere to the project's coding style (see below).

3. Add tests for any new functionality or bug fixes. This is important to ensure the project remains stable.

4. Ensure all tests pass before submitting your pull request.
   ```bash
   poetry run pytest
   ```

5. Push your branch and open a pull request. Provide a clear description of the changes you've made.

## Style Guide

This project uses **Black** for code formatting. Before committing your changes, please run Black to format your code.

```bash
poetry run black .
```

The maximum line length is set to 120 characters.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree
to abide by its terms. We expect all contributors to be respectful and constructive.
