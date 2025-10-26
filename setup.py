from setuptools import setup, find_packages

setup(
    name="Iris-Flower-Classification",  # Your project/package name
    version="0.1",
    packages=find_packages(),  # Automatically find all packages with __init__.py
    install_requires=[
        # Add your dependencies here, e.g. "pandas", "scikit-learn"
        "pandas", "scikit-learn", "numpy", "mlflow", "joblib", "pydantic", "uvicorn", "fastapi",
        "jsonresponse"
    ],
    python_requires='>=3.6',
)
