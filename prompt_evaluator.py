import os
import json
import sqlite3
import argparse
import anthropic
from dotenv import load_dotenv
import re
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Load the ground truth data
def load_ground_truth(path='data/ground_truth_data.json'):
    with open(path, 'r') as f:
        return json.load(f)

# Execute SQL query
def execute_sql_query(query, db_path='./data/nba.sqlite'):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        conn.close()
        print(f"SQL Error: {e} in query: {query}")
        return None

# Extract SQL from Claude's response
def extract_sql(content):
    # Try to find SQL between ```sql and ``` markers
    sql_pattern = r"```sql\s*(.*?)\s*```"
    match = re.search(sql_pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no SQL code block, try to find any code block
    code_pattern = r"```\s*(.*?)\s*```"
    match = re.search(code_pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no code blocks, just return the content (assuming it's SQL)
    return content.strip()

# Add this function to get detailed schema information
def get_detailed_schema(db_path):
    """Extract detailed schema information including foreign keys and data types."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = {}
    
    for table in tables:
        table_name = table[0]
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        # Get foreign key information
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        foreign_keys = cursor.fetchall()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        
        # Store all information
        schema_info[table_name] = {
            'columns': columns,
            'foreign_keys': foreign_keys,
            'row_count': row_count
        }
    
    conn.close()
    return schema_info

# Format detailed schema for prompt
def format_detailed_schema(schema_info):
    """Format the detailed schema information into a string for the prompt."""
    schema_text = "DATABASE SCHEMA WITH METADATA:\n\n"
    
    for table_name, info in schema_info.items():
        # Add table name and row count
        schema_text += f"Table: {table_name} ({info['row_count']} rows)\n"
        
        # Add columns with type information
        schema_text += "Columns:\n"
        for col in info['columns']:
            col_id, col_name, col_type, not_null, default_val, is_pk = col
            pk_marker = " (PRIMARY KEY)" if is_pk else ""
            null_marker = " NOT NULL" if not_null else ""
            schema_text += f"  - {col_name}: {col_type}{pk_marker}{null_marker}\n"
        
        # Add foreign key information if available
        if info['foreign_keys']:
            schema_text += "Foreign Keys:\n"
            for fk in info['foreign_keys']:
                id, seq, table, from_col, to_col = fk[0], fk[1], fk[2], fk[3], fk[4]
                schema_text += f"  - {from_col} → {table}.{to_col}\n"
        
        schema_text += "\n"
    
    return schema_text

# Add a function to find relevant query examples
def get_relevant_examples(question, ground_truth_data, top_n=3):
    """Find the most relevant examples from ground truth data for a given question."""
    # Extract questions from ground truth data
    gt_questions = [item['natural_language'] for item in ground_truth_data]
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the questions
    tfidf_matrix = vectorizer.fit_transform(gt_questions + [question])
    
    # Calculate cosine similarity between the input question and all ground truth questions
    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
    
    # Get indices of top N most similar questions
    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    
    # Return the corresponding examples
    return [ground_truth_data[i] for i in top_indices]

# Add a function to execute a query and get sample results
def get_query_results_sample(query, db_path, max_rows=3):
    """Execute a query and return a formatted string with sample results."""
    results = execute_sql_query(query, db_path)
    if not results or len(results) == 0:
        return "No results"
    
    # Format the results as a string
    result_str = f"Sample results ({min(len(results), max_rows)} of {len(results)} rows):\n"
    for i, row in enumerate(results[:max_rows]):
        result_str += f"  {row}\n"
    
    return result_str

# Format examples for prompt
def format_examples_for_prompt(examples, db_path):
    """Format relevant examples for inclusion in the prompt."""
    if not examples:
        return ""
    
    examples_text = "RELEVANT QUERY EXAMPLES:\n\n"
    for i, example in enumerate(examples):
        ex_question = example['natural_language']
        ex_sql = example['sql']
        
        # Get sample results for this query
        results_sample = get_query_results_sample(ex_sql, db_path)
        
        examples_text += f"Question: {ex_question}\n"
        examples_text += f"SQL: {ex_sql}\n"
        examples_text += f"{results_sample}\n\n"
    
    return examples_text

# Update the generate_sql function to accept the full prompt directly
def generate_sql(client, full_prompt, question, db_path=None, inject_schema=False):
    try:
        # If we're given a full prompt, use it directly
        # Otherwise, build it (for backward compatibility)
        prompt = full_prompt
        
        # Only process the prompt if it's not already processed
        if "{schema}" in prompt or "{examples}" in prompt or "{{question}}" in prompt:
            # Format with schema and examples if requested
            if inject_schema and db_path:
                # Get schema information
                schema_info = get_detailed_schema(db_path)
                schema_text = format_detailed_schema(schema_info)
                
                # Get relevant examples
                ground_truth = load_ground_truth()
                relevant_examples = get_relevant_examples(question, ground_truth, top_n=3)
                examples_text = format_examples_for_prompt(relevant_examples, db_path)
                
                # Replace placeholders
                if "{schema}" in prompt:
                    prompt = prompt.replace("{schema}", schema_text)
                else:
                    prompt = prompt + "\n\n" + schema_text
                    
                if "{examples}" in prompt:
                    prompt = prompt.replace("{examples}", examples_text)
                else:
                    prompt = prompt + "\n\n" + examples_text
            else:
                # Remove placeholders if not injecting
                prompt = prompt.replace("{schema}", "")
                prompt = prompt.replace("{examples}", "")
            
            # Replace the question placeholder
            prompt = prompt.replace("{{question}}", question)
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            temperature=0.0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract SQL from response
        sql = extract_sql(response.content[0].text)
        return sql
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return None

# Evaluate a generated SQL query against ground truth
def evaluate_query(generated_sql, expected_sql, db_path, max_results=4):
    try:
        # Execute both queries
        generated_results = execute_sql_query(generated_sql, db_path)
        expected_results = execute_sql_query(expected_sql, db_path)
        
        if generated_results is None or expected_results is None:
            return {
                'result_match': False,
                'success': False,
                'generated_result': None,
                'expected_result': None
            }
            
        # Convert results to sets of tuples for comparison
        generated_set = set(tuple(row) for row in generated_results)
        expected_set = set(tuple(row) for row in expected_results)
        result_match = generated_set == expected_set
        
        # Limit the number of results stored in the JSON output
        limited_generated_results = generated_results[:max_results] if generated_results else None
        limited_expected_results = expected_results[:max_results] if expected_results else None
        
        return {
            'result_match': result_match,
            'success': result_match,
            'generated_result': limited_generated_results,
            'expected_result': limited_expected_results
        }
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            'result_match': False,
            'success': False,
            'generated_result': None,
            'expected_result': None,
            'error': str(e)
        }

# Update evaluate_prompt to use schema injection
def evaluate_prompt(client, prompt_template, test_data, db_path, inject_schema=False):
    results = []
    
    for item in tqdm(test_data, desc="Evaluating prompt"):
        question = item["natural_language"]
        expected_sql = item["sql"]
        
        # Generate the full prompt with schema and examples if requested
        full_prompt = prompt_template
        
        if inject_schema and db_path:
            # Get schema information
            schema_info = get_detailed_schema(db_path)
            schema_text = format_detailed_schema(schema_info)
            
            # Get relevant examples
            ground_truth = load_ground_truth()
            relevant_examples = get_relevant_examples(question, ground_truth, top_n=3)
            examples_text = format_examples_for_prompt(relevant_examples, db_path)
            
            # Replace placeholders
            if "{schema}" in full_prompt:
                full_prompt = full_prompt.replace("{schema}", schema_text)
            else:
                full_prompt = full_prompt + "\n\n" + schema_text
                
            if "{examples}" in full_prompt:
                full_prompt = full_prompt.replace("{examples}", examples_text)
            else:
                full_prompt = full_prompt + "\n\n" + examples_text
        else:
            # Remove placeholders if not injecting
            full_prompt = full_prompt.replace("{schema}", "")
            full_prompt = full_prompt.replace("{examples}", "")
        
        # Replace the question placeholder
        full_prompt = full_prompt.replace("{{question}}", question)
        
        # Generate SQL using the full prompt
        generated_sql = generate_sql(client, full_prompt, question)
        
        if generated_sql:
            # Evaluate the generated SQL
            eval_result = evaluate_query(generated_sql, expected_sql, db_path)
            
            # Store the result with the prompt used
            results.append({
                'question': question,
                'expected_sql': expected_sql,
                'generated_sql': generated_sql,
                'prompt_used': full_prompt,  # Store the actual prompt used
                **eval_result
            })
    
    # Calculate overall metrics
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    result_matches = sum(1 for r in results if r['result_match'])
    
    metrics = {
        'total': total,
        'successful': successful,
        'result_matches': result_matches,
        'success_rate': successful / total if total > 0 else 0,
        'result_match_rate': result_matches / total if total > 0 else 0,
        'failure_rate': 1 - (successful / total) if total > 0 else 1
    }
    
    return results, metrics

# Update evaluate_prompt_file to store the base template
def evaluate_prompt_file(prompt_path, db_path, output_path=None, num_samples=None, inject_schema=False):
    # Load the prompt
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    
    # Load ground truth data
    ground_truth = load_ground_truth()
    
    # Use specified number of samples or all data for testing
    if num_samples and num_samples < len(ground_truth):
        # Randomly sample from ground truth data
        test_data = random.sample(ground_truth, num_samples)
    else:
        test_data = ground_truth
    
    print(f"Test data size: {len(test_data)}")
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Evaluate the prompt
    print(f"Evaluating prompt: {prompt_path}...")
    if inject_schema:
        print("Injecting schema information into prompt")
    
    results, metrics = evaluate_prompt(client, prompt_template, test_data, db_path, inject_schema)
    
    print(f"Result match rate: {metrics['result_match_rate']:.2%}")
    
    # Save evaluation results
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, 'w') as f:
            json.dump({
                'prompt_template': prompt_template,
                'prompt_path': prompt_path,
                'inject_schema': inject_schema,
                'results': results,
                'metrics': metrics
            }, f, indent=2)
        
        print(f"Evaluation results saved to: {output_path}")
    
    return results, metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate SQL generation prompt')
    parser.add_argument('--prompt', default='./prompts/original.txt',
                        help='Path to the prompt template to evaluate')
    parser.add_argument('--output', default='./eval_logs/evaluation_results.json',
                        help='Path to save the evaluation results')
    parser.add_argument('--db', default='./data/nba.sqlite',
                        help='Path to the SQLite database')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of ground truth samples to use (default: all)')
    parser.add_argument('--inject-schema', action='store_true',
                        help='Inject detailed schema information into the prompt')
    
    args = parser.parse_args()
    
    print("Starting prompt evaluation...")
    evaluate_prompt_file(
        prompt_path=args.prompt,
        db_path=args.db,
        output_path=args.output,
        num_samples=args.samples,
        inject_schema=args.inject_schema
    )
    print("Evaluation complete!")

if __name__ == "__main__":
    main() 