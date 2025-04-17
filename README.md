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

The `--inject-schema` flag enables a powerful feature that significantly improves query generation accuracy. When enabled, the system:

1. **Schema Context**: Injects detailed database schema information including:
   - Table structures with column names and types
   - Primary and foreign key relationships
   - Table row counts for better context
   - Data type information for each column

2. **Relevant Examples**: Automatically finds and includes similar past queries by:
   - Using TF-IDF similarity to find relevant examples
   - Including both the natural language question and its correct SQL
   - Showing sample results from these examples

3. **Contextual Learning**: The model uses this additional context to:
   - Better understand table relationships
   - Make informed decisions about join conditions
   - Choose appropriate column names and data types
   - Learn from similar past queries

The schema injection feature is particularly effective because it:
- Reduces ambiguity in table and column references
- Provides context about data relationships
- Shows the model how similar queries were handled
- Helps avoid common SQL generation mistakes

To use schema injection, simply add the `--inject-schema` flag when running the evaluator:
```bash
python prompt_evaluator.py --inject-schema
```

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

### Advanced Options

The script supports several command-line arguments to customize the evaluation process:

```bash
python prompt_evaluator.py [options]
```

#### Available Options:

- `--prompt PATH`  
  Path to the prompt template file to use for evaluation.  
  Default: Uses the default prompt template  
  Example: `--prompt ./prompts/optimized.txt`

- `--output PATH`  
  Path where the evaluation results will be saved as JSON.  
  Default: Results are printed to stdout  
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

- The prompt template used
- Configuration settings (schema injection, sample size, etc.)
- Individual query results including:
  - Original question
  - Generated SQL
  - Expected SQL
  - Query execution results
  - Success/failure status
- Overall performance metrics

Example evaluation result structure:
```json
{
  "prompt_template": "...",
  "prompt_path": "./prompts/optimized.txt",
  "inject_schema": true,
  "results": [
    {
      "question": "What is the average points per game for players in the 2023 season?",
      "expected_sql": "SELECT AVG(points) FROM player_stats WHERE season = '2023'",
      "generated_sql": "...",
      "success": true,
      "result_match": true
    }
  ],
  "metrics": {
    "total": 98,
    "successful": 90,
    "success_rate": 0.9184
  }
}
```

To analyze results across different runs, you can:
1. Compare different prompt versions
2. Track performance improvements
3. Identify common failure patterns
4. Optimize prompt templates based on results

## Data Format

The ground truth data should be in JSON format with the following structure:
```json
{
  "questions": [
    {
      "question": "What is the average points per game for players in the 2023 season?",
      "sql": "SELECT AVG(points) FROM player_stats WHERE season = '2023'",
      "expected_results": [...]
    }
  ]
}
```




