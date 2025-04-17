# NBA SQL Query Generator

This project evaluates and optimizes prompts for converting natural language questions about NBA data into SQL queries. It uses Claude 3.5 Sonnet to generate SQL queries and evaluates their accuracy against a ground truth dataset.

## Overview

The system:
- Takes natural language questions about NBA data
- Converts them to SQL queries using Claude 3.5 Sonnet
- Evaluates the accuracy of generated queries against expected results
- Provides metrics on success rates and result matching

## Features

- Schema injection for better context awareness
- Relevant example retrieval using TF-IDF similarity
- Detailed evaluation metrics
- Support for various query types (filtering, aggregation, ranking, etc.)

## Installation

# Clone the repository
git clone https://github.com/yourusername/nba-sql-query-generator.git
cd nba-sql-query-generator

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with your Anthropic API key
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env

# Basic usage with default prompt
python prompt_evaluator.py

# Use a specific prompt file
python prompt_evaluator.py --prompt ./prompts/optimized.txt

# Specify output location
python prompt_evaluator.py --output ./eval_logs/my_evaluation.json

# Use a specific database
python prompt_evaluator.py --db ./data/nba.sqlite

# Limit evaluation to a sample of questions
python prompt_evaluator.py --samples 20

# Enable schema injection
python prompt_evaluator.py --inject-schema

.
├── data/
│   ├── ground_truth_data.json  # Expected questions and SQL queries
│   └── nba.sqlite              # NBA database
├── eval_logs/                  # Evaluation results
├── prompts/                    # Prompt templates
│   ├── original.txt            # Basic prompt
│   └── optimized.txt           # Optimized prompt with rules
├── prompt_evaluator.py         # Main evaluation script
├── requirements.txt            # Dependencies
└── README.md                   # This file
Results
The optimized prompt achieves a 91.8% success rate on the test dataset, compared to the original prompt's lower performance. Key improvements include:
Detailed SQL syntax rules
SQLite-specific guidance
Schema context awareness
Example-based learning
