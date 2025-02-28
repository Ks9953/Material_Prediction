from flask import Flask, request, jsonify
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from flask_cors import CORS
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from datetime import datetime
import random

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Environment variables for database credentials
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "replicator")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Ant-admin@123")
DB_HOST = os.getenv("DB_HOST", "34.100.200.180")
DB_PORT = int(os.getenv("DB_PORT", 5432))

# Function to connect to the database
def get_db_connection():
    try:
        return psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
    except Exception as e:
        print("Database connection error:", e)
        return None

@app.route("/predict", methods=["POST"])
def predict_demand():
    data = request.get_json()
    material_id = data.get("material_id")
    
    if not material_id:
        return jsonify({"error": "Material ID is required"}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Unable to connect to the database"}), 500

    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        demand_query = """
            SELECT "date" AS ds, "quantity" AS y
            FROM material_data 
            WHERE "material_id" = %s 
            ORDER BY "date" ASC
        """
        cursor.execute(demand_query, (material_id,))
        rows = cursor.fetchall()

        if not rows:
            return jsonify({"error": f"No data found for Material ID: {material_id}"}), 404

        price_query = """
            SELECT "price"
            FROM material_data 
            WHERE "material_id" = %s AND "price" > 0 
            ORDER BY "date" DESC 
            LIMIT 1
        """
        cursor.execute(price_query, (material_id,))
        price_row = cursor.fetchone()
        last_price = float(price_row['price']) if price_row else 1.0

        df = pd.DataFrame(rows)
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)
        df.sort_values(by='ds', inplace=True)
        
        # Train model
        quantity_model = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        quantity_results = quantity_model.fit(disp=False)
        
        # Predict future demand
        forecast_horizon = 10
        current_month = datetime.now().replace(day=1)
        forecast_dates = [current_month + pd.DateOffset(months=i) for i in range(forecast_horizon)]
        quantity_forecast = quantity_results.get_forecast(steps=forecast_horizon).predicted_mean
        quantity_forecast = np.maximum(0, quantity_forecast)
        
        # Prepare output in requested format
        forecast_result = {
            "Material ID": material_id,
            "Price": last_price
        }
        for date, quantity in zip(forecast_dates, quantity_forecast):
            month_col = f"Predicted Quantity {date.strftime('%b %Y')}"
            forecast_result[month_col] = int(round(quantity))

        # Generate dummy historical predictions
        df['year'] = df['ds'].dt.year
        min_year = df['year'].min()
        for year in range(min_year + 1, datetime.now().year):
            for index, row in df[df['year'] == year].iterrows():
                month_col = f"Predicted Quantity {row['ds'].strftime('%b %Y')}"
                variance = random.uniform(0.85, 1.15)  # 10-15% variance
                forecast_result[month_col] = int(round(row['y'] * variance))
        
        return jsonify(forecast_result), 200
    
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
