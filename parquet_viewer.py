import pandas as pd
import argparse

def inspect_parquet(file_path):
    """
    Reads and inspects a Parquet file, providing useful summary information.
    """
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        print(f"--- Inspecting Parquet File: {file_path} ---")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        
        print("1. Dataframe Info:")
        df.info()
        
        print("\n" + "="*50 + "\n")
        print("2. First 5 rows:")
        print(df.head())
        
        print("\n" + "="*50 + "\n")
        print("3. Basic Statistics:")
        print(df.describe(include='all'))
        
        print("\n" + "="*50 + "\n")
        print("4. Columns and Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"  - {col}: {dtype}")
            
        print("\n" + "="*50 + "\n")
            
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading '{file_path}':\n{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the contents and schema of a Parquet file.")
    parser.add_argument("file_path", help="Path to the .parquet file you want to inspect")
    
    args = parser.parse_args()
    
    inspect_parquet(args.file_path)
