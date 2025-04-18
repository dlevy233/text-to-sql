# NBA SQL Query Generator

This project evaluates and optimizes prompts for converting natural language questions into SQL queries. It uses Claude 3.7 Sonnet to generate SQL queries and evaluates their accuracy against a ground truth dataset.

## Overview

The system:
- Takes natural language questions about NBA data
- Converts them to SQL queries using Claude 3.7 Sonnet
- Evaluates the accuracy of generated queries against expected results
- Provides metrics on success rates and result matching
- Supports schema injection and example-based prompting for improved accuracy

## Performance Results

### Original vs Optimized Prompt Performance

The system's performance shows dramatic improvements when using the optimized prompt with schema injection:

| Configuration | Accuracy |
|--------------|----------|
| Original Prompt (no schema) | 0% |
| Optimized Prompt (with schema Injection) | 91.84% |

These results are based on a comprehensive evaluation of 98 test cases from the NBA dataset.

### How Schema Injection Works

The schema injection feature provides three key benefits:

1. **Schema Context**: The prompt includes detailed database schema information, helping the model understand table structures, relationships, and data types.

2. **Relevant Examples**: The system automatically finds and includes the 10 most similar examples from the ground truth data (excluding the current question to prevent data leakage) using TF-IDF similarity. Each example includes:
   - The natural language question
   - The corresponding SQL query
   - Sample results from executing the query

3. **Contextual Learning**: By seeing similar questions and their solutions, the model can better understand the relationships between natural language and SQL, leading to more accurate query generation.

The schema injection feature is particularly effective because it:
- Reduces ambiguity in table and column references
- Provides context about data relationships
- Shows the model how similar queries were handled
- Helps avoid common SQL generation mistakes


## Features

- Schema injection for better context awareness
- Relevant example retrieval using TF-IDF similarity
- Detailed evaluation metrics including:
  - Query syntax accuracy
  - Result set matching
  - Query structure comparison
- Support for various query types:
  - Filtering
  - Aggregation
  - Ranking
  - Joins
  - Subqueries

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your Anthropic API key
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

## Usage

### Basic Evaluation
```bash
python prompt_evaluator.py
```

### Run Evaluation for Optimized Prompt

The script supports several command-line arguments to customize the evaluation process:

```bash
python prompt_evaluator.py --prompt ./prompts/optimized.txt --output ./eval_logs/optimized_evaluation_results.json --inject-schema
```

#### Available Options:

- `--prompt PATH`  
  Path to the prompt template file to use for evaluation.  
  Default: `./prompts/original.txt` 
  Example: `--prompt ./prompts/optimized.txt`

- `--output PATH`  
  Path where the evaluation results will be saved as JSON.  
  Default: `./eval_logs/evaluation_results.json`
  Example: `--output ./eval_logs/my_evaluation.json`

- `--db PATH`  
  Path to the SQLite database file containing NBA data.  
  Default: `./data/nba.sqlite`  
  Example: `--db ./data/custom_nba.sqlite`

- `--samples N`  
  Number of questions to evaluate from the ground truth dataset.  
  Default: Evaluates all questions  
  Example: `--samples 20` to evaluate only 20 questions

- `--inject-schema`  
  Enable schema injection in the prompt. This significantly improves accuracy by providing the database schema context to the model.  
  Default: Disabled  
  Example: `--inject-schema` to enable schema injection

### Evaluation Results and Logging

The system automatically saves detailed evaluation results in the `./eval_logs` directory. Each evaluation run creates a JSON file containing:

1. The prompt template used
2. Configuration settings (including whether schema injection was enabled)
3. Individual query results for each test case
4. Overall performance metrics

The JSON structure includes:
- Original question
- Generated SQL
- Expected SQL
- Success status
- Performance metrics (accuracy, result match rate)

You can analyze results across different runs to:
- Compare different prompt versions
- Track performance improvements
- Identify patterns in successful/failed queries






