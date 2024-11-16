# Chat Bot Prototype

Simple chat bot prototype built with LangChain and Streamlit.

## Features
- Web interface with Streamlit
- Conversation memory
- Support for Groq LLM API
- Chat-style interface
- Session state management

## Setup
1. Clone the repository
```bash
git clone <your-repo-url>
cd chat-prototype
```

2. Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate  # For Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create `.env` file with your API keys
```
GROQ_API_KEY=your-api-key-here
```

5. Run the application
```bash
streamlit run app.py
```

## Project Structure
```
chat-prototype/
├── .env                  # Environment variables (gitignored)
├── .gitignore           # Git ignore file
├── README.md            # Project documentation
├── app.py              # Main application file
└── requirements.txt    # Project dependencies
```

## Tech Stack
- Python 3.8+
- Streamlit
- LangChain
- Groq API

