from sqlalchemy import create_engine
import pandas as pd


engine= create_engine('postgresql://postgres:3051@localhost:5432/superstore_db')

df= pd.read_csv("superstore.csv")

# print("shape", df.shape)
# print("column", df.columns.tolist())
df.to_sql('superstore', engine, if_exists='replace', index=False)
print("data loaded into db")


def run_query(query):
    return pd.read_sql_query(query, engine)


q1=run_query("""
    SELECT "Product Name",
        ROUND(CAST(SUM("Sales")AS NUMERIC),2)AS Total_Sales
    FROM superstore
    WHERE "Product Name" IS NOT NULL
    AND "Sales" IS NOT NULL
    GROUP BY "Product Name"
    ORDER BY Total_Sales DESC
    LIMIT 5    """)
# print("top 5 product")
# print(q1)    


q2=run_query("""
    SELECT
        "Customer Name",
        "Region",
        ROUND(CAST(SUM("Sales")AS NUMERIC),2) AS Total_Sales,
        RANK() OVER (PARTITION BY "Region"
                    ORDER BY SUM("Sales")DESC)AS Sales_Rank

    FROM superstore
    WHERE "Region" IS NOT NULL
    AND "Category" IS NOT NULL
    GROUP BY "Customer Name", "Region"
    ORDER BY "Region", Sales_Rank                

        """)

# print("top customers")
# print(q2.head(10))   

q3 = run_query("""
    WITH regional_sales AS (
        SELECT 
            "Region",
            "Category",
            ROUND(CAST(SUM("Sales") AS NUMERIC), 2) AS Total_Sales,
            ROUND(CAST(SUM("Profit") AS NUMERIC), 2) AS Total_Profit,
            ROUND(CAST(AVG("Discount") AS NUMERIC), 3) AS Avg_Discount
        FROM superstore
        GROUP BY "Region", "Category"
    ),
    profit_ratio AS (
        SELECT *,
            ROUND(CAST((Total_Profit / Total_Sales * 100) AS NUMERIC), 2) 
            AS Profit_Margin
        FROM regional_sales
    )
    SELECT * FROM profit_ratio
    ORDER BY Profit_Margin DESC
""")
# print("Regional Category Performance:")
# print(q3)


q4 = run_query("""
    WITH monthly_sales AS (
        SELECT 
            DATE_TRUNC('month', "Order Date"::timestamp) AS Month,
            ROUND(CAST(SUM("Sales") AS NUMERIC), 2) AS Total_Sales
        FROM superstore
        GROUP BY DATE_TRUNC('month', "Order Date"::timestamp)
        ORDER BY Month
    )
    SELECT 
        Month,
        Total_Sales,
        LAG(Total_Sales) OVER (ORDER BY Month) AS Prev_Month_Sales,
        ROUND(CAST((Total_Sales - LAG(Total_Sales) 
              OVER (ORDER BY Month)) AS NUMERIC), 2) AS MoM_Change
    FROM monthly_sales
    ORDER BY Month
    LIMIT 10
""")
# print("Month over Month Sales:")
# print(q4)     


q5 = run_query("""
    SELECT 
        "Customer Name",
        "Segment",
        ROUND(CAST(SUM("Sales") AS NUMERIC), 2) AS Total_Sales,
        ROUND(CAST(SUM("Profit") AS NUMERIC), 2) AS Total_Profit
    FROM superstore
    WHERE "Customer Name" IN (
        SELECT "Customer Name"
        FROM superstore
        GROUP BY "Customer Name"
        HAVING SUM("Sales") > 5000
    )
    GROUP BY "Customer Name", "Segment"
    ORDER BY Total_Profit DESC
    LIMIT 10
""")
print("High Value Customers:")
print(q5)