# 🧮 Math Problem Solver

## Overview

Math Problem Solver is an AI-powered Streamlit application that helps users solve mathematical problems with step-by-step explanations. The application uses Groq's AI models (including Google Gemma 2) to provide detailed solutions to a wide range of mathematical topics.

## Getting Started

### Prerequisites

- Python 3.8+
- Groq API key (get one at [console.groq.com](https://console.groq.com/keys))

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/math-problem-solver.git
   cd math-problem-solver
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`

## Usage

1. Enter your Groq API key in the sidebar (if not already set in `.env`)
2. Select your preferred AI model
3. Type your math problem in the text area or choose from the example problems
4. Click "Solve Problem" to get a detailed solution
5. Review the step-by-step explanation

### Example Problems

- Solve for x: 2x + 5 = 15
- Find the area of a circle with radius 4 cm
- If a train travels at 60 mph and takes 3 hours to reach its destination, how far did it travel?
- Word problems involving multiple operations and conversions

## Technical Details

This application uses:

- **Streamlit**: For the web interface
- **LangChain**: For agent orchestration and tool integration
- **Groq API**: For AI model inference
- **Google Gemma 2**: Primary AI model for solving math problems
- **Wikipedia API**: For foundational knowledge
- **arXiv API**: For academic mathematical concepts
- **DuckDuckGo Search**: For web search capabilities

The application uses a zero-shot agent that strategically chooses which tool to use based on the problem type.

## Advanced Configuration

In the "Advanced Options" section of the sidebar, you can:

- Adjust the temperature (creativity) of the AI responses
- Toggle the visibility of AI reasoning steps
- Select different models for different types of problems

## Development

### Project Structure

```
math-problem-solver/
├── app.py                  # Main application file
├── .env                    # Environment variables (git-ignored)
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── LICENSE                 # MIT License
├── screenshots/            # App screenshots
│   └── app_screenshot.png  # Main app screenshot
└── .gitignore              # Git ignore configuration
```