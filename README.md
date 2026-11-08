# Portfolio Analytics Agent

An AI-powered command-line agent that answers natural-language questions about portfolio data using Google Gemini and SQLite.

The agent understands user questions, selects the appropriate tool, executes validated operations against a SQLite database, and returns structured, data-driven answers.

Gemini handles tool selection and SQL generation, while Python tools perform the actual database queries and financial calculations.

## Key Features

- **Natural Language Interface:** Ask questions about portfolio data using everyday language.
- **AI-Based Tool Selection:** Gemini determines which tool is appropriate for each question.
- **Text-to-SQL:** Converts natural-language questions into validated, read-only SQL queries.
- **Sector Exposure Analysis:** Calculates sector-wise exposure for individual portfolios.
- **SQL Security:** Uses SQL validation and SQLite read-only enforcement.
- **Portfolio Identification:** Supports exact and unique partial portfolio-name matching.
- **Independent Evaluation:** Compares agent results against ground-truth data and independently calculated expected values.
- **Automated Testing:** Includes 75 unit and integration tests.
- **Error Handling:** Handles malformed responses, API errors, database errors and invalid SQL.

## Available Tools

| Tool | Description |
|---|---|
| `sql_query` | Converts natural-language questions into SQL, validates the query and executes it against SQLite. |
| `exposure_calculator` | Calculates sector exposure for a selected portfolio using equity holdings only. Bonds are excluded. |

The tools perform the actual calculations and database operations. Gemini does not independently invent financial results.

## Project Structure

```text
portfolio_agent/
│
├── agent.py
├── db.py
├── tools.py
├── router.py
├── llm.py
├── formatter.py
├── evaluator.py
├── config.py
│
├── database_schema.sql
├── ground_truth_dataset.json
├── expected_exposures.json
├── eval_report.json
│
├── data/
│   ├── portfolios.csv
│   ├── holdings.csv
│   ├── securities.csv
│   ├── sectors.csv
│   ├── benchmarks.csv
│   ├── transactions.csv
│   ├── historical_prices.csv
│   ├── portfolio_performance.csv
│   └── risk_metrics.csv
│
├── scripts/
│   └── make_expected_exposures.py
│
├── tests/
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The project includes the original assessment datasets, database schema, ground-truth questions and expected exposure values.

## Requirements

- Python 3.10 or higher
- pip (Python package manager)
- Google Gemini API key
- Internet connection for Gemini API access

Verify your Python installation:

```bash
python --version
```

Python 3.10 or higher is required.

## Installation and Setup

### 1. Clone or extract the project

Open a terminal in the project directory.

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
```

Then activate the environment again.

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Get your API key from:

https://aistudio.google.com/apikey

Create your environment file.

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

**Windows Command Prompt:**

```bat
copy .env.example .env
```

**macOS / Linux:**

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your actual Gemini API key.

Example:

```env
GEMINI_API_KEY=your_actual_api_key
GEMINI_MODEL=gemini-3.8-flash
```

The default model is configured in `config.py` and can be overridden through the `GEMINI_MODEL` environment variable.

The application validates its configuration at startup and provides clear errors for missing API keys or invalid model configuration.

**Important:** Never commit your `.env` file or expose your API key in source code.

## Running the Agent

### Interactive mode

```bash
python agent.py
```

The application starts an interactive command-line session where users can ask multiple questions.

### Single-question mode

```bash
python agent.py "How many portfolios do we have?"
```

Expected result:

```text
Result: 13
```

### Verbose mode

```bash
python agent.py -v
```

Verbose logging helps inspect tool selection, SQL execution and execution timings.

## Example Usage

### Example 1: SQL Query

**User question:**

How many portfolios do we have?

**Agent workflow:**

1. Gemini identifies the question as a database query.
2. The SQL tool generates an appropriate SELECT statement.
3. The SQL validator checks the query.
4. SQLite executes the validated query.
5. The formatter returns the result.

**Expected answer:**

```text
Result: 13
```

### Example 2: Sector Exposure

**User question:**

What is the sector exposure of Balanced Portfolio?

The agent selects the exposure calculator, identifies the portfolio, retrieves its equity holdings, calculates sector weights and generates a structured response.

The output includes:

- Sector names
- Equity exposure percentages
- Whole-portfolio weight percentages
- Total equity weight

The exposure values are calculated from the supplied assessment datasets.

## Architecture

The project follows a modular tool-based AI agent architecture.

```text
                 USER QUESTION
                       |
                       v
                +-------------+
                |    AGENT    |
                +-------------+
                       |
                       v
                +-------------+
                |   ROUTER    |
                |   Gemini    |
                +-------------+
                       |
                 Tool Selection
                       |
           +-----------+-----------+
           |                       |
           v                       v
    +--------------+       +-------------------+
    |  SQL QUERY   |       | EXPOSURE          |
    |    TOOL      |       | CALCULATOR        |
    +--------------+       +-------------------+
           |                       |
           v                       v
    SQL Validation          Portfolio Lookup
           |                       |
           v                       v
    Read-Only SQLite        Equity Holdings
           |                       |
           v                       v
    Query Results           Sector Calculations
           |                       |
           +-----------+-----------+
                       |
                       v
                +-------------+
                |  FORMATTER  |
                +-------------+
                       |
                       v
                FINAL ANSWER
```

### Agent Workflow

1. The user submits a natural-language question.
2. The router sends the question and database schema to Gemini.
3. Gemini selects the appropriate tool and provides its arguments.
4. The application validates the tool decision.
5. The selected tool performs the database operation or financial calculation.
6. The formatter constructs the final answer deterministically from the tool result.

If the router returns an invalid response, it receives feedback and retries once.

For SQL execution errors, the application can request a corrected SQL query. The corrected query is validated again before execution.

The formatter does not make a second LLM call. This avoids unnecessary API calls and prevents the final response from contradicting the actual tool results.

## Core Modules

| File | Responsibility |
|---|---|
| `agent.py` | Main agent orchestration and command-line interface. |
| `db.py` | SQLite database initialization and connection management. |
| `tools.py` | SQL validation, query execution, portfolio identification and exposure calculations. |
| `router.py` | Gemini routing prompt, structured response schema and decision validation. |
| `llm.py` | Gemini client, timeout handling, retry logic and API error classification. |
| `formatter.py` | Converts tool results into readable text and Markdown tables. |
| `evaluator.py` | Executes ground-truth questions and evaluates agent performance. |
| `config.py` | Application configuration and environment variable handling. |
| `scripts/make_expected_exposures.py` | Independently generates expected sector exposures using pandas and raw CSV data. |
| `tests/` | Automated unit and integration tests. |

## Important Design Decisions

### 1. Sector Exposure Calculation

Sector exposure is calculated using the `current_weight` of equity holdings.

Bond holdings are excluded from sector exposure calculations.

Two different percentages are returned:

**Equity Exposure Percentage**

```text
Sector Equity Weight
------------------------------- × 100
Total Equity Weight
```

This represents the proportion of the equity portion invested in a particular sector.

**Whole-Portfolio Weight Percentage**

```text
Sector Equity Weight
------------------------------- × 100
Total Portfolio Weight
```

This represents the sector's contribution to the entire portfolio.

Both percentages are returned to avoid confusion between equity allocation and overall portfolio allocation.

For example, the Balanced Portfolio has 36% equity allocation. Its Technology sector represents 61.11% of its equity holdings and 22.00% of its entire portfolio.

Portfolios containing only bonds return an appropriate message indicating that there are no equity holdings.

### 2. Portfolio Identification

Portfolio matching follows three steps:

1. Exact portfolio-name matching (case-insensitive).
2. Unique partial-name matching.
3. Clarification when multiple portfolios match.

The agent does not arbitrarily select the first result when a portfolio name is ambiguous.

SQL wildcard characters such as `%` and `_` are escaped during partial matching.

### 3. LLM Reliability

The application uses structured output to constrain Gemini's routing decisions.

Additional reliability measures include:

- Validation of tool names and required arguments.
- One retry for invalid routing responses.
- Timeout handling.
- Exponential backoff for retryable API errors.
- Immediate failure (without retries) when the daily free-tier quota is exhausted.
- Separate handling of authentication and invalid-model errors.
- SQL repair with mandatory re-validation.

### 4. SQL Security

SQL execution is protected using two layers.

**Application-level validation:**

- Allows a single SELECT or WITH query.
- Rejects write and DDL operations.
- Rejects multiple SQL statements.
- Validates generated SQL before execution.

**Database-level protection:**

SQLite connections use:

```sql
PRAGMA query_only = ON;
```

This provides an additional read-only restriction even if application-level validation is bypassed.

Query results are capped at `MAX_ROWS` (200). The application indicates when results have been truncated.

### 5. SQL Style Rules in the Router Prompt

The router prompt contains general rules for how SQL results are shaped: which columns are returned, how results are ordered, how computed columns are named, and how "total portfolio value" is defined. These rules make results consistent and easier to compare against reference answers. They are written as general conventions and do not contain answers to specific evaluation questions.

## Evaluation Methodology

The project includes a ground-truth dataset containing 10 assessment questions.

The evaluator measures the correctness of the agent's tool execution and final answers.

### 1. SQL Question Evaluation

For SQL questions:

- The generated SQL is executed.
- Its result set is compared against the ground-truth SQL result.
- Row ordering is considered when the ground-truth query specifies `ORDER BY`.
- Numerical values are compared using a small floating-point tolerance.
- Column names and column counts are also checked.

With `--strict`, column-name mismatches and additional columns cause the question to fail.

Without strict mode, these mismatches are reported as warnings.

The evaluator compares query results rather than requiring the generated SQL text to be identical to the ground-truth SQL.

### 2. Exposure Question Evaluation

Exposure calculations are evaluated against `expected_exposures.json`.

Expected values are generated independently using pandas and the original CSV files, rather than by reusing the exposure calculator being tested.

The evaluator checks:

- Correct tool selection.
- Correct portfolio identification.
- Exact sector set.
- Individual sector exposure percentages.
- Whole-portfolio sector weights.
- Total equity weight.
- Correct exclusion of bond holdings.

This prevents incorrect sector distributions from passing simply because their percentages add up to 100%.

### 3. Final Answer Validation

The final natural-language response is also evaluated.

The evaluator checks whether the answer correctly communicates the underlying results.

Depending on the question, validation includes:

- Correct numerical result.
- Correct portfolio information.
- Correct sector names and percentages.
- Correct equity allocation.
- Clear distinction between equity exposure and whole-portfolio weight.
- Appropriate explanation of bond exclusion.

A question passes only when the required underlying results and final answer are correct.

### 4. Evaluation Report

The evaluator produces:

- Individual question PASS/FAIL results.
- Per-question execution information.
- Per-type evaluation scores.
- Final-answer validation results.
- Overall evaluation score.
- JSON evaluation report.

The process returns a non-zero exit code when questions fail.

## Running the Evaluator

Run the complete evaluation:

```bash
python evaluator.py
```

Run with strict column validation:

```bash
python evaluator.py --strict
```

Evaluate selected questions:

```bash
python evaluator.py --ids 9 10 --strict
```

Generate a JSON report:

```bash
python evaluator.py --strict --report eval_report.json
```

Pause between questions (useful on rate-limited free API tiers):

```bash
python evaluator.py --strict --delay 15 --report eval_report.json
```

**Important distinction:** The real evaluation uses the configured Gemini model and measures actual model performance. Automated tests use simulated LLM responses and do not represent real Gemini accuracy.

## Evaluation Results

Final run (strict mode, `gemini-3.1-flash-lite`): **9 / 10 (90%)**

| Category | Passed |
|---|---|
| text2sql | 7 / 8 |
| exposure_calculator | 2 / 2 |
| Final answer text | 9 / 9 (of questions whose data was correct) |

The full report is in `eval_report.json`. My first run scored 4/10. I improved it by adding general SQL-style rules to the router prompt (which columns to return, ordering, column naming, and how "total portfolio value" is calculated), based on the failures shown in the report.

**Known failure: Q8.** The question asks for "diversification metrics", which is ambiguous. The agent returned holdings per sector (2.0), while the reference answer uses sectors per holding (0.5). The agent's result is a reasonable reading of the question, and I chose not to hard-code the reference's definition into the prompt.

**Notes on interpretation:**

- Strict mode compares column names, so some prompt rules (such as the column-naming rule) were written with that grader in mind. In non-strict mode, name mismatches are only warnings.
- This is a single run. Results can vary slightly between runs because LLM output is not fully deterministic.
- Free-tier Gemini quotas limited how many full runs I could do.

## Automated Testing

The project includes 75 automated unit and integration tests.

Run the complete test suite:

```bash
python -m unittest discover -v
```

The tests do not require a Gemini API key or internet access.

They cover:

- Database initialization and query execution.
- SQL validation and security.
- Read-only database enforcement.
- Query-result truncation.
- Exact and partial portfolio matching.
- Ambiguous portfolio identification.
- SQL wildcard escaping.
- Exposure calculations across all 13 portfolios.
- Independent expected exposure comparisons.
- Bond exclusion.
- Strict and non-strict column validation.
- Incorrect exposure distributions.
- Final-answer correctness.
- Structured decision parsing.
- Invalid tool selection.
- Malformed LLM responses.
- SQL repair and re-validation.
- Retry and backoff behaviour.
- API error classification.
- Agent execution using simulated LLM responses.
- End-to-end evaluator execution using an oracle LLM.

The automated test suite validates the application's logic and reliability. It does not measure the actual accuracy of Gemini.

## Security

- API credentials are stored in `.env`.
- `.env` is excluded from version control through `.gitignore`.
- The evaluation report (`eval_report.json`) contains no credentials and can be committed with the submission.
- SQL operations are restricted to read-only queries.
- Generated SQL is validated before execution.
- Database connections use SQLite query-only mode.
- The application avoids embedding API credentials in source code.

## Limitations

- Actual answer accuracy depends on Gemini's routing and SQL-generation performance.
- Real Gemini evaluation requires a valid API key and network connection.
- Unit tests use simulated LLM responses and cannot guarantee real-world model accuracy.
- The router prompt includes the database schema. Very large schemas may require schema retrieval or RAG.
- The application does not maintain conversational memory between questions.
- Questions requiring multiple tools in one combined operation are not supported.
- SQL results are limited to 200 rows.
- Ambiguous questions (for example, "diversification metrics") may be interpreted differently from the reference answer.

## Future Improvements

Potential future improvements include:

- Supporting multi-tool and hybrid questions.
- Adding conversational memory.
- Introducing schema retrieval for larger databases.
- Adding more comprehensive real-world evaluation datasets.
- Improving query optimization and response latency.
- Adding a web-based user interface.
- Supporting additional financial analytics tools.

## Conclusion

The Portfolio Analytics Agent demonstrates how an LLM can interact with structured financial data through validated tools rather than relying entirely on generated responses.

It combines natural-language understanding, structured tool selection, SQL execution, deterministic financial calculations, database safety and independent evaluation.

The main design objective is to make the agent's answers **accurate, explainable, testable and grounded in the underlying portfolio data.**