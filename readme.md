# URL Knowledge Scraper

This project is a web application that scrapes and distills knowledge from a given URL and its sub-pages, displaying the collected information in real-time through a clean and modern GUI.

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/url-knowledge-scraper.git
   cd url-knowledge-scraper
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Copy the `.env.example` file to `.env`
   - Fill in your API keys in the `.env` file

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your web browser and go to `http://localhost:7860` (or the URL provided in the console output).

3. Enter a URL in the text box and click "Submit" to start the knowledge extraction process.

## Components

- `app.py`: Main application script
- `langflow_pipeline.json`: Langflow pipeline configuration
- `requirements.txt`: List of Python dependencies
- `.env`: Environment variables (API keys)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
Last edited 2 minutes ago


Publish
Prompt Storm
Please register or sign in to Prompt Storm.
Register

Sign in

