import pandas as pd

# Load the CSV file
df = pd.read_csv(r'C:\Users\thiag\Meu Drive (thiagos.1994@alunos.utfpr.edu.br)\Master\Repositories\MambaLithium\data\CaseC\17.csv')

# Show current columns (optional)
print("Original columns:", df.columns)

# Remove specific columns by name
columns_to_remove = ['omega','b','TemperatureAvg','InternalResistance','ChargeTime','dQdV_max','dQdV_min','dQdV_var']  # Replace with your actual column names
df = df.drop(columns=columns_to_remove)

# Save the modified DataFrame back to a new CSV (or overwrite the original)
df.to_csv('17.csv', index=False)

# Show the result columns (optional)
print("Result columns:", df.columns)