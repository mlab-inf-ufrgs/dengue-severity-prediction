# 📂 `config/` - Bootstrap Configuration

This folder contains configuration scripts used to prepare the notebook environment.

## 🚀 `bootstrap.py`

The `bootstrap.py` script automatically configures your `sys.path` so that you can easily import modules from the `src/` directory, **no matter where your notebook is located**.

It works by:
- Searching for a **project root marker** (such as `.project-root`, `README.md`, or `.git`) in the parent directories.
- Adding the `src/` folder to the Python path (`sys.path`) automatically.

This setup allows you to use clean imports like:

```python
from utils.helpers import get_project_root
```
instead of complicated relative imports.

---

## 🧩 How to use in a notebook

At the top of your notebook, simply add:

```python
%run ../../config/bootstrap.py
```
or

```python
%run ../config/bootstrap.py
```
depending on the depth of the notebook in your project folder structure.

> **Tip:**  
> Use `../` to go up one folder level.  
> Use `../../` to go up two levels, and so on.

---

## 📌 Example

Imagine the following structure:

```
dengue/
├── src/
│   ├── utils/
│   └── cnes/
│       └── cnes.ipynb
├── tests/
│   └── notebook_testes.ipynb
├── config/
│   ├── bootstrap.py
│   └── README.md
├── .project-root
├── README.md
```

Then:
- In `src/cnes/cnes.ipynb`, you would use:

  ```python
  %run ../../config/bootstrap.py
  ```

- In `tests/notebook_testes.ipynb`, you would use:

  ```python
  %run ../config/bootstrap.py
  ```

---

## 🔥 Why use this?

- Ensures consistent imports across your project.
- No need to manually modify `sys.path` in every notebook.
- Makes your code **more portable** and **production-ready**.
- Facilitates scaling up the project structure without worrying about broken imports.

---

## 📢 Important

Make sure that your project root has one of the following marker files:
- `.project-root` (recommended)
- `README.md`
- `.git/`

Otherwise, the bootstrap will not be able to find the root directory.
 