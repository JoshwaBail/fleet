"""
Example 7: Navigator (ReAct Pattern)

The Navigator uses the ReAct (Reasoning and Acting) pattern:
iteratively thinking about what to do, taking actions, and observing results.

Use Case: Data exploration and analysis where you need visibility into reasoning.
"""

import os
from fleet import ReActCaptain, create_provider, tool, Toolbox

# Define some data analysis instruments
@tool(
    name="query_database",
    description="Query a database for information"
)
def query_database(query: str) -> dict:
    """Simulate a database query"""
    # In real app, this would query an actual database
    mock_data = {
        "SELECT revenue": {"total_revenue": 1500000, "records": 150},
        "SELECT customers": {"total_customers": 450, "active": 380},
        "SELECT products": {"total_products": 25, "top_seller": "Widget Pro"}
    }

    for key in mock_data:
        if key.lower() in query.lower():
            return mock_data[key]

    return {"message": "No data found"}


@tool(
    name="calculate",
    description="Perform mathematical calculations"
)
def calculate(expression: str) -> float:
    """Safe calculation"""
    try:
        # In production, use a safer eval alternative
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e)}


@tool(
    name="get_metadata",
    description="Get metadata about the database schema"
)
def get_metadata(table: str) -> dict:
    """Get table metadata"""
    schemas = {
        "sales": {"columns": ["id", "date", "amount", "customer_id"], "rows": 150},
        "customers": {"columns": ["id", "name", "email", "created_at"], "rows": 450},
        "products": {"columns": ["id", "name", "price", "category"], "rows": 25}
    }

    return schemas.get(table.lower(), {"error": "Table not found"})


def main():
    print("="*70)
    print("NAVIGATOR (ReAct PATTERN) EXAMPLE")
    print("="*70)

    # Create provider
    provider = create_provider("openai", api_key=os.getenv("OPENAI_API_KEY"))

    # Build arsenal
    analysis_toolbox = Toolbox("Data Analysis Tools")
    analysis_toolbox.add_tool(query_database)
    analysis_toolbox.add_tool(calculate)
    analysis_toolbox.add_tool(get_metadata)

    # Create Navigator
    navigator = ReActCaptain(
        provider=provider,
        name="Data Explorer",
        system_prompt="""You are a data analyst. When given a question about data:
1. THINK about what information you need
2. ACT by using your tools to get that information
3. OBSERVE the results
4. Repeat until you can answer the question

Be thorough and show your reasoning process.""",
        toolbox=analysis_toolbox,
        default_model="gpt-4o-mini",
        max_iterations=5,
        verbose=True
    )

    # Complex analytical question requiring multiple steps
    question = """What is the average revenue per customer, and how does it compare to
    the average product price? Are we getting good value per customer?"""

    print(f"\nQuestion: {question}\n")

    # Navigate through the problem
    result = navigator.navigate(question)

    print("\n" + "="*70)
    print("FINAL ANSWER")
    print("="*70)
    print(result.content)
    print(f"\nReasoning Steps: {result.metadata.get('iterations', 0)}")
    print(f"Total Tokens: {result.total_tokens}")


if __name__ == "__main__":
    main()
