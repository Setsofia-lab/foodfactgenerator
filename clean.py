import pandas as pd
import numpy as np
import re

def preprocess_food_data():
    """
    Preprocess OpenFoodFacts data and merge with country information
    """
    
    # File paths
    cleaned_path = "/home/xhy9676/.cache/kagglehub/datasets/michaelfumery/enopenfoodfactsorgproducts/versions/4/cleaned_openfoodfacts.csv"
    countries_path = "/home/xhy9676/.cache/kagglehub/datasets/michaelfumery/enopenfoodfactsorgproducts/versions/4/countries-en.csv"
    
    # 1. Load and parse the main food products data (tab-separated)
    print("Loading food products data...")
    df_products = pd.read_csv(cleaned_path, sep='\t', on_bad_lines='skip', low_memory=False)
    print(f"Products data shape: {df_products.shape}")
    
    # 2. Load and parse the countries data (tab-separated)
    print("Loading countries data...")
    df_countries = pd.read_csv(countries_path, sep='\t', on_bad_lines='skip', low_memory=False)
    print(f"Countries data shape: {df_countries.shape}")
    
    # 3. Clean and prepare countries mapping
    print("Preparing country mapping...")
    
    # The countries file seems to have columns like: code, name, etc.
    # Let's examine the structure and create a mapping
    print("Countries data columns:", df_countries.columns.tolist())
    print("Countries data sample:")
    print(df_countries.head())
    
    # Create country code to name mapping
    # Assuming the structure is similar to what we see in the sample
    country_mapping = {}
    
    # If the countries data has the right structure, create mapping
    if len(df_countries.columns) >= 2:
        # Try to identify country code and name columns
        code_col = df_countries.columns[0] if len(df_countries.columns) > 0 else None
        name_col = df_countries.columns[-1] if len(df_countries.columns) > 1 else None
        
        if code_col and name_col:
            for _, row in df_countries.iterrows():
                if pd.notna(row[code_col]) and pd.notna(row[name_col]):
                    code = str(row[code_col]).strip().upper()
                    name = str(row[name_col]).strip()
                    country_mapping[code] = name
    
    print(f"Created mapping for {len(country_mapping)} countries")
    
    # 4. Select relevant columns from products data
    print("Selecting relevant product columns...")
    
    # Define the columns we want for food fact generation
    desired_columns = [
        'code', 'product_name', 'countries_en', 'categories_en',
        'energy_100g', 'proteins_100g', 'fat_100g', 'carbohydrates_100g', 
        'sugars_100g', 'salt_100g', 'fiber_100g', 'saturated-fat_100g',
        'nutriscore_grade', 'main_category_en'
    ]
    
    # Get available columns
    available_columns = [col for col in desired_columns if col in df_products.columns]
    print(f"Available columns: {available_columns}")
    
    # Select available columns and filter out rows with missing essential data
    df_selected = df_products[available_columns].copy()
    
    # 5. Clean and process the data
    print("Cleaning and processing data...")
    
    def clean_text(x):
        """Clean text data"""
        if pd.isna(x) or x == '':
            return 'unknown'
        return str(x).strip().lower()
    
    def clean_numeric(x):
        """Clean numeric data"""
        if pd.isna(x) or x == '':
            return 0.0
        try:
            return float(x)
        except:
            return 0.0
    
    # Clean text columns
    text_columns = ['product_name', 'categories_en', 'main_category_en']
    for col in text_columns:
        if col in df_selected.columns:
            df_selected[col] = df_selected[col].apply(clean_text)
    
    # Clean numeric columns
    numeric_columns = ['energy_100g', 'proteins_100g', 'fat_100g', 'carbohydrates_100g', 
                      'sugars_100g', 'salt_100g', 'fiber_100g', 'saturated-fat_100g']
    for col in numeric_columns:
        if col in df_selected.columns:
            df_selected[col] = df_selected[col].apply(clean_numeric)
    
    # 6. Process countries and map to full names
    print("Processing country information...")
    
    def extract_and_map_countries(countries_str, mapping):
        """Extract country codes and map to full names"""
        if pd.isna(countries_str) or countries_str == '':
            return 'unknown'
        
        countries_str = str(countries_str).strip()
        
        # Split by common delimiters
        country_codes = re.split(r'[,;|]', countries_str)
        country_names = []
        
        for code in country_codes:
            code = code.strip().upper()
            if code in mapping:
                country_names.append(mapping[code])
            elif len(code) == 2:  # If it's a 2-letter code, try direct lookup
                country_names.append(mapping.get(code, code.lower()))
            elif code:  # If not empty, keep as is
                country_names.append(code.lower())
        
        if country_names:
            return ', '.join(set(country_names))  # Remove duplicates
        return 'unknown'
    
    # Apply country mapping
    if 'countries_en' in df_selected.columns:
        df_selected['country_names'] = df_selected['countries_en'].apply(
            lambda x: extract_and_map_countries(x, country_mapping)
        )
    else:
        df_selected['country_names'] = 'unknown'
    
    # 7. Filter out products with insufficient data
    print("Filtering data...")
    
    # Keep products that have at least product name and some nutritional info
    mask = (
        (df_selected.get('product_name', 'unknown') != 'unknown') &
        (df_selected.get('product_name', '') != '') &
        (
            (df_selected.get('energy_100g', 0) > 0) |
            (df_selected.get('proteins_100g', 0) > 0) |
            (df_selected.get('fat_100g', 0) > 0) |
            (df_selected.get('carbohydrates_100g', 0) > 0)
        )
    )
    
    df_filtered = df_selected[mask].copy()
    print(f"Filtered dataset shape: {df_filtered.shape}")
    
    # 8. Generate food facts
    print("Generating food facts...")
    
    def generate_comprehensive_fact(row):
        """Generate a comprehensive food fact"""
        facts = []
        
        # Product name
        product_name = row.get('product_name', 'This product')
        if product_name != 'unknown':
            facts.append(f"{product_name.title()}")
        
        # Nutritional information
        nutrition_parts = []
        if row.get('energy_100g', 0) > 0:
            nutrition_parts.append(f"{row['energy_100g']:.1f} kcal of energy")
        if row.get('proteins_100g', 0) > 0:
            nutrition_parts.append(f"{row['proteins_100g']:.1f}g of protein")
        if row.get('fat_100g', 0) > 0:
            nutrition_parts.append(f"{row['fat_100g']:.1f}g of fat")
        if row.get('carbohydrates_100g', 0) > 0:
            nutrition_parts.append(f"{row['carbohydrates_100g']:.1f}g of carbohydrates")
        if row.get('sugars_100g', 0) > 0:
            nutrition_parts.append(f"{row['sugars_100g']:.1f}g of sugar")
        
        if nutrition_parts:
            if len(nutrition_parts) == 1:
                facts.append(f"contains {nutrition_parts[0]} per 100g")
            else:
                facts.append(f"contains {', '.join(nutrition_parts[:-1])}, and {nutrition_parts[-1]} per 100g")
        
        # Category information
        category = row.get('main_category_en', row.get('categories_en', ''))
        if category and category != 'unknown':
            facts.append(f"It belongs to the {category} category")
        
        # Nutriscore
        nutriscore = row.get('nutriscore_grade', '')
        if nutriscore and pd.notna(nutriscore) and str(nutriscore).strip() != '':
            nutriscore_str = str(nutriscore).strip().upper()
            if nutriscore_str in ['A', 'B', 'C', 'D', 'E']:
                facts.append(f"and has a Nutri-Score grade of {nutriscore_str}")
        
        # Country information
        countries = row.get('country_names', 'unknown')
        if countries and countries != 'unknown':
            facts.append(f"This product is available in {countries}")
        
        # Combine all facts
        fact_text = '. '.join(facts)
        if not fact_text.endswith('.'):
            fact_text += '.'
        
        return fact_text
    
    # Create input-output pairs for training
    df_filtered['input'] = df_filtered['product_name'].apply(lambda x: x.title() if x != 'unknown' else 'food product')
    df_filtered['output'] = df_filtered.apply(generate_comprehensive_fact, axis=1)
    
    # 9. Create final training dataset
    print("Creating final dataset...")
    
    training_data = df_filtered[['input', 'output']].copy()
    
    # Remove any remaining problematic entries
    training_data = training_data[
        (training_data['input'].str.len() > 0) &
        (training_data['output'].str.len() > 10)  # Ensure meaningful facts
    ]
    
    print(f"Final training dataset shape: {training_data.shape}")
    
    # Save the processed dataset
    output_path = "food_facts_training_dataset.csv"
    training_data.to_csv(output_path, index=False)
    print(f"Training dataset saved to: {output_path}")
    
    # Display sample results
    print("\nSample training pairs:")
    for i in range(min(5, len(training_data))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input: {training_data.iloc[i]['input']}")
        print(f"Output: {training_data.iloc[i]['output']}")
    
    # Save detailed dataset for analysis
    detailed_output_path = "food_products_detailed.csv"
    df_filtered.to_csv(detailed_output_path, index=False)
    print(f"\nDetailed dataset saved to: {detailed_output_path}")
    
    return training_data, df_filtered

# Run the preprocessing
if __name__ == "__main__":
    training_data, detailed_data = preprocess_food_data()
    print("\n preprocessing completed!")
