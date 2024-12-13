import os
import json
import sqlite3
import logging
import pandas as pd
import numpy as np
import openai
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tkinter import Tk, filedialog, Label, Button, Text, END, messagebox, Toplevel, Entry, StringVar, ttk, Menu

class DatabaseManager:
    def __init__(self, db_path='traffic_analytics.db'):
        """Initialize database connection and create tables"""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.create_tables()

    def create_tables(self):
        """Create necessary tables in the SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Traffic data table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                traffic REAL,
                day_of_week TEXT,
                hour INTEGER
            )
        ''')

        # Model performance tracking table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                mse REAL,
                r2_score REAL,
                training_date TEXT
            )
        ''')

        # Prediction results table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT,
                predicted_traffic REAL,
                actual_traffic REAL
            )
        ''')

        self.conn.commit()

    def insert_traffic_data(self, data):
        """Insert traffic data into the database"""
        try:
            data.to_sql('traffic_data', self.conn, if_exists='replace', index=False)
            self.conn.commit()
        except Exception as e:
            print(f"Error inserting traffic data: {e}")

    def log_model_performance(self, model_name, mse, r2):
        """Log model performance metrics"""
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            self.cursor.execute('''
                INSERT INTO model_performance 
                (model_name, mse, r2_score, training_date) 
                VALUES (?, ?, ?, ?)
            ''', (model_name, mse, r2, current_date))
            self.conn.commit()
        except Exception as e:
            print(f"Error logging model performance: {e}")

    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

class SettingsManager:
    def __init__(self, config_file='config.json'):
        """Load configuration from JSON file"""
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from a JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def save_config(self, config):
        """Save configuration to a JSON file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get_api_key(self):
        """Retrieve the stored API key"""
        return self.config.get('openai_api_key', '')

    def set_api_key(self, api_key):
        """Set and save the API key"""
        self.config['openai_api_key'] = api_key
        self.save_config(self.config)

class MachineLearningAnalyzer:
    def __init__(self, data):
        """Initialize ML analyzer with traffic data"""
        self.data = data
        self.model = None
        self.scaler = StandardScaler()

    def prepare_data(self):
        """Prepare data for machine learning"""
        # Feature engineering
        self.data['day_of_week_num'] = pd.to_datetime(self.data['Date']).dt.dayofweek
        self.data['month'] = pd.to_datetime(self.data['Date']).dt.month

        # Select features
        features = ['day_of_week_num', 'month', 'Hour']
        X = self.data[features]
        y = self.data['Traffic']

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self):
        """Train Random Forest Regressor"""
        X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data()

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Predict and evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mse, r2

    def predict_traffic(self, input_data):
        """Make traffic predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")

        input_scaled = self.scaler.transform(input_data)
        return self.model.predict(input_scaled)

class SettingsWindow:
    def __init__(self, parent, settings_manager):
        self.parent = parent
        self.settings_manager = settings_manager

        # Create settings window
        self.window = Toplevel(parent)
        self.window.title("Settings")
        self.window.geometry("400x200")

        # API Key Label and Entry
        self.api_key_label = Label(self.window, text="OpenAI API Key:")
        self.api_key_label.pack(pady=(20, 5))

        self.api_key_var = StringVar(value=self.settings_manager.get_api_key())
        self.api_key_entry = Entry(self.window, textvariable=self.api_key_var, width=50, show="*")
        self.api_key_entry.pack(pady=5)

        # Save Button
        self.save_button = Button(self.window, text="Save", command=self.save_settings)
        self.save_button.pack(pady=20)

        # Hint Label
        self.hint_label = Label(self.window, 
            text="Tip: Keep your API key confidential. \n"
                 "You can get an API key from OpenAI's platform.",
            fg="gray"
        )
        self.hint_label.pack(pady=10)

    def save_settings(self):
        """Save the entered API key"""
        api_key = self.api_key_var.get().strip()

        # Validate API key (basic length check)
        if not api_key:
            messagebox.showwarning("Warning", "API Key cannot be empty")
            return

        # Save the API key
        self.settings_manager.set_api_key(api_key)
        messagebox.showinfo("Success", "API Key saved successfully")
        self.window.destroy()

class TrafficAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Traffic Analyzer")
        self.root.geometry("600x700")

        # Initialize Database and Settings Managers
        self.database_manager = DatabaseManager()
        self.settings_manager = SettingsManager()

        # Create Menu Bar
        self.menu_bar = self.create_menu_bar()
        self.root.config(menu=self.menu_bar)

        # UI Elements
        self.setup_ui()

    def create_menu_bar(self):
        """Create application menu bar"""
        menu_bar = Menu(self.root)

        # Settings Menu
        settings_menu = Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="API Settings", command=self.open_settings)
        menu_bar.add_cascade(label="Settings", menu=settings_menu)

        return menu_bar

    def open_settings(self):
        """Open the settings window"""
        SettingsWindow(self.root, self.settings_manager)

    def setup_ui(self):
        """Set up the user interface elements"""
        # File Upload
        self.label = Label(self.root, text="Upload a CSV file with 'Date', 'Traffic', and 'Hour' columns:")
        self.label.pack(pady=10)

        self.upload_btn = Button(self.root, text="Upload File", command=self.upload_file)
        self.upload_btn.pack(pady=10)

        # Prompt Box
        self.prompt_label = Label(self.root, text="AI Analysis Prompt (Editable):")
        self.prompt_label.pack(pady=10)

        self.prompt_box = Text(self.root, height=5, width=50)
        self.prompt_box.insert(END, (
            "Analyze this customer traffic data to find patterns. "
            "Identify peak traffic periods, predict future trends, "
            "and provide strategic insights for business optimization."
        ))
        self.prompt_box.pack(pady=10)

        # Buttons Frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        self.analyze_btn = Button(button_frame, text="AI Analysis", command=self.analyze_data)
        self.analyze_btn.pack(side='left', padx=5)

        self.ml_train_btn = Button(button_frame, text="Train ML Model", command=self.train_machine_learning_model)
        self.ml_train_btn.pack(side='left', padx=5)

        self.predict_btn = Button(button_frame, text="Make Predictions", command=self.make_predictions)
        self.predict_btn.pack(side='left', padx=5)

        self.output_label = Label(self.root, text="")
        self.output_label.pack(pady=10)

        self.file_path = None
        self.ml_model = None

    def upload_file(self):
        """Upload a CSV file"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_path = file_path
            self.label.config(text=f"File Uploaded: {file_path.split('/')[-1]}")

    def train_machine_learning_model(self):
        """Train machine learning model on uploaded data"""
        if not self.file_path:
            messagebox.showerror("Error", "Please upload a file first")
            return

        try:
            # Load data
            data = pd.read_csv(self.file_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data['DayOfWeek'] = data['Date'].dt.day_name()
            data['Hour'] = data['Date'].dt.hour

            # Train ML model
            ml_analyzer = MachineLearningAnalyzer(data)
            mse, r2 = ml_analyzer.train_model()

            # Log performance
            self.database_manager.log_model_performance('RandomForestRegressor', mse, r2)

            self.ml_model = ml_analyzer
            messagebox.showinfo("Success", f"Model Trained Successfully\nMSE: {mse:.2f}, R2: {r2:.2f}")

            # Store data in database
            self.database_manager.insert_traffic_data(data)

        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    def make_predictions(self):
        """Make traffic predictions using trained model"""
        if not self.ml_model:
            messagebox.showerror("Error", "Please train a model first")
            return

        # Example prediction input 
        prediction_input = pd.DataFrame({
            'day_of_week_num': [0],  # Monday
            'month': [6],             # June
            'Hour': [14]              # 2 PM
        })

        try:
            prediction = self.ml_model.predict_traffic(prediction_input)
            messagebox.showinfo("Prediction", f"Predicted Traffic: {prediction[0]:.2f}")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def analyze_data(self):
        """Analyze the uploaded traffic data"""
        # Retrieve API Key
        api_key = self.settings_manager.get_api_key()
        if not api_key:
            messagebox.showerror("Error", "Please set your OpenAI API Key in Settings")
            return

        if not self.file_path:
            messagebox.showerror("Error", "Please upload a file first")
            return

        try:
            # Load and validate data
            data = pd.read_csv(self.file_path)
            if 'Date' not in data.columns or 'Traffic' not in data.columns:
                messagebox.showerror("Error", "CSV must have 'Date' and 'Traffic' columns")
                return
        except Exception as e:
            messagebox.showerror("File Error", f"Error loading file: {str(e)}")
            return

        # Preprocess data
        data['Date'] = pd.to_datetime(data['Date'])
        data['DayOfWeek'] = data['Date'].dt.day_name()
        data['Hour'] = data['Date'].dt.hour
        traffic_summary = data.groupby(['DayOfWeek', 'Hour'])['Traffic'].sum().reset_index()

        # Prepare prompt for analysis
        traffic_text = traffic_summary.to_csv(index=False)
        prompt = self.prompt_box.get("1.0", END).strip()
        complete_prompt = f"{prompt}\n\nTraffic Data:\n{traffic_text}"

        # Call OpenAI API
        try:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data analysis assistant."},
                    {"role": "user", "content": complete_prompt}
                ]
            )
            result = response.choices[0].message.content.strip()

            # Display result
            self.show_result(result)
            self.output_label.config(text="Analysis Completed Successfully!")
        except Exception as e:
            messagebox.showerror("API Error", f"Error with OpenAI API: {str(e)}")

    def show_result(self, result):
        """Display analysis results in a new window"""
        result_window = Toplevel(self.root)
        result_window.title("Traffic Analysis Result")
        result_window.geometry("600x400")

        result_text = Text(result_window, wrap='word')
        result_text.insert(END, result)
        result_text.pack(padx=20, pady=20)

def main():
    """Main application entry point"""
    root = Tk()
    root.title("Advanced Traffic Analyzer")
    root.geometry("600x700")
    
    try:
        app = TrafficAnalyzer(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()