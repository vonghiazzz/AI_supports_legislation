# AI Supports Legislation

This project uses AI to support legislation processes.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [Conda](https://docs.anaconda.com/miniconda/) (optional but recommended)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/vonghiazzz/AI_supports_legislation.git
    cd AI_supports_legislation
    ```

2.  **Set up a virtual environment**:
    ```bash
    # Using venv
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

4.  **Configuration**:
    Create a `.env` file in the `backend/` directory (you can copy the example):
    ```bash
    cp .env.example .env
    ```
    Then, open `.env` and add your `GEMINI_API_KEY`.

### Running the Application

From the `backend/` directory:
```bash
fastapi dev app/main.py
```

## Project Structure

- `backend/`: FastAPI application and AI services.
- `backend/app/`: Core application logic.
- `backend/scripts/`: Utility scripts for data processing and ingestion.
