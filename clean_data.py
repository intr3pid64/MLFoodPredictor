import pandas as pd
import numpy as np
import re

#############################
# 1. BASE PARSING FUNCTIONS #
#############################

def parse_complexity(val):
    try:
        x = int(val)
        if 1 <= x <= 5:
            return x
        else:
            return np.nan
    except:
        return np.nan

def parse_ingredients(val):
    text = str(val)
    nums = re.findall(r'\d+', text)
    if not nums:
        return np.nan
    nums = [int(x) for x in nums]
    return float(np.mean(nums))

def parse_settings(val):
    all_options = [
        'Week day lunch',
        'Week day dinner',
        'Weekend lunch',
        'Weekend dinner',
        'At a party',
        'Late night snack'
    ]
    chosen = [x.strip() for x in str(val).split(',')]
    out = {}
    for opt in all_options:
        out[opt] = 1 if opt in chosen else 0
    return out

def parse_cost(val):
    text = str(val).lower()
    # quick cleanup
    text = text.replace('cad', '').replace('dollar', '').replace('dollars', '')
    nums = re.findall(r'\d+\.?\d*', text)
    if not nums:
        return np.nan
    floats = [float(x) for x in nums]
    return float(np.mean(floats))

def parse_reminds(val):
    all_options = ['Parents','Siblings','Friends','Teachers','Strangers']
    chosen = [x.strip() for x in str(val).split(',')]
    out = {}
    for opt in all_options:
        out[opt] = 1 if opt in chosen else 0
    return out

def parse_hot_sauce(val):
    val_str = str(val).lower()
    if 'none' in val_str:
        return 0
    elif 'little' in val_str or 'mild' in val_str or 'have some' in val_str:
        return 1
    elif 'moderate' in val_str or 'medium' in val_str:
        return 2
    elif 'a lot' in val_str or 'hot' in val_str:
        return 3
    else:
        return np.nan

############################
# 2. ADVANCED MOVIE PARSING #
############################

# Example dictionary-based approach for recognized movies, or you can map to categories (like "cartoon", "action", etc.)
MOVIE_KEYWORDS = {
    # Pizza-frequent
    "teenage mutant ninja": "Teenage Mutant Ninja Turtles",
    "home alone": "Home Alone",
    "spider man": "Spider-Man",
    "spiderman": "Spider-Man",
    "garfield": "Garfield",
    "cloudy with a chance of meatball": "Cloudy with a Chance of Meatballs",
    "ratatouille": "Ratatouille",
    "avengers": "The Avengers",
    "iron man": "Iron Man",
    "transformer": "Transformers",
    "godfather": "The Godfather",
    "toy story": "Toy Story",
    "goodfellas": "Goodfellas",
    "back to the future": "Back to the Future",
    # Shawarma-frequent
    "the dictator": "The Dictator",
    "shawarma legend": "Other Movie",   # or we can keep a special category
    "argo": "Argo",
    "deadpool": "Deadpool",
    "ninja turtle": "Teenage Mutant Ninja Turtles",
    # Sushi-frequent
    "jiro dreams of sushi": "Jiro Dreams of Sushi",
    "spirited away": "Spirited Away",
    "kill bill": "Kill Bill",
    "finding nemo": "Finding Nemo",
    "your name": "Your Name",
    "isle of dogs": "Isle of Dogs",
    "the wolverine": "The Wolverine",
    "cars 2": "Cars 2",
    # A few others:
    "none": "No Movie",
    "no movie": "No Movie",
    "n/a": "No Movie",
    "i dont think of any movie": "No Movie",
    # etc.
}

def normalize_text(s):
    """Lowercase, remove punctuation (optionally), strip extra spaces."""
    s = str(s).strip().lower()
    # remove punctuation except maybe apostrophes if you want
    s = re.sub(r'[^\w\s]', '', s)
    return s

def parse_movie_advanced(val):
    """
    1) Normalize text
    2) Check for no-movie patterns
    3) Look for key substrings
    4) Return standardized name if found, else 'other_movie'
    
    You could also do multiple passes or partial matching.
    """
    text = normalize_text(val)
    if text in ["none", "no movie", "n/a"]:
        return "No Movie"

    # Attempt to find a match from the dictionary above, by checking if the keyword is in the text
    for kw, standard_title in MOVIE_KEYWORDS.items():
        if kw in text:
            return standard_title
    
    # If we reach here, no known pattern found
    return "Other Movie"

#############################
# 3. ADVANCED DRINK PARSING #
#############################

# Synonyms for unify drink categories
DRINK_MAP = {
    # for Pizza
    "coke": "Coke",
    "cola": "Coke",
    "coca cola": "Coke",
    "diet coke": "Coke",
    "pepsi": "Pepsi",
    "diet pepsi": "Pepsi",
    "mountain dew": "Mountain Dew",
    "dr pepper": "Dr Pepper",
    "root beer": "Root Beer",
    "ginger ale": "GingerAle",
    "fanta": "Fanta",
    "sprite": "Sprite",
    "7up": "Sprite",
    "soda": "Soda",
    "pop": "Soda",
    "water": "Water",
    "juice": "Juice",
    "hot chocolate": "Hot Chocolate",
    "tea": "Tea",
    "coffee": "Coffee",
    "wine": "Wine",
    "red wine": "Wine",
    "beer": "Beer",
    "lemonade": "Juice",
    # for Shawarma
    "ayran": "Ayran",
    "milk tea": "Milk Tea",
    "laban": "Ayran",
    # for Sushi
    "miso soup": "Miso Soup",
    "green tea": "Green Tea",
    "hot tea": "Tea",
    "barley tea": "Tea",
    "ocha": "Green Tea",
    "sake": "Sake",
    "soju": "Soju",
    "calpis": "Calpis",
    # etc.
    "no": "No Drink",
}

def parse_drink_advanced(val):
    """
    1) Normalize text
    2) Attempt to find the best match from DRINK_MAP
    3) If multiple drinks are mentioned, pick the first match (or do multi-hot if you prefer)
    4) If no match, return "Other Drink"
    """
    text = normalize_text(val)
    if not text or text in ["none", "n/a"]:
        return "No Drink"
    
    # For multi-hot: you could split on commas or 'and' or other delimiters
    # For simplicity here, let's just do a single best-match approach
    # If you'd rather multi-hot, parse each chunk and look for matches.
    
    # We'll scan from largest to smallest match.
    # Actually let's just check each known key in DRINK_MAP and see if it appears in the text:
    for kw, mapped in DRINK_MAP.items():
        if kw in text:
            return mapped
    
    return "Other Drink"

#########################
# 4. MAIN CLEANING LOGIC #
#########################

def clean_data_advanced(df):
    """
    This function does everything:
      - Numeric columns
      - Multi-hot columns
      - Advanced movie & drink handling
      - Leaves you with standardized columns ready for modeling
    """
    # Rename for convenience
    df = df.rename(columns={
        'id': 'id',
        'Label': 'label',
        'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': 'Q1_complexity',
        'Q2: How many ingredients would you expect this food item to contain?': 'Q2_ingredients',
        'Q3: In what setting would you expect this food to be served? Please check all that apply': 'Q3_setting',
        'Q4: How much would you expect to pay for one serving of this food item?': 'Q4_cost',
        'Q5: What movie do you think of when thinking of this food item?': 'Q5_movie',
        'Q6: What drink would you pair with this food item?': 'Q6_drink',
        'Q7: When you think about this food item, who does it remind you of?': 'Q7_reminds',
        'Q8: How much hot sauce would you add to this food item?': 'Q8_hot_sauce',
    })
    
    # A) Numeric + Multi-hot as before
    df['complexity'] = df['Q1_complexity'].apply(parse_complexity)
    df['num_ingredients'] = df['Q2_ingredients'].apply(parse_ingredients)
    df['cost_approx'] = df['Q4_cost'].apply(parse_cost)
    df['hot_sauce_level'] = df['Q8_hot_sauce'].apply(parse_hot_sauce)
    
    # settings multi-hot
    settings_df = df['Q3_setting'].apply(parse_settings).apply(pd.Series)
    df = pd.concat([df, settings_df], axis=1)
    
    # reminds multi-hot
    reminds_df = df['Q7_reminds'].apply(parse_reminds).apply(pd.Series)
    df = pd.concat([df, reminds_df], axis=1)
    
    # B) Advanced Movie & Drink
    df['movie_std'] = df['Q5_movie'].apply(parse_movie_advanced)
    df['drink_std'] = df['Q6_drink'].apply(parse_drink_advanced)
    
    # (Optional) keep raw text for possible NLP
    df['movie_raw'] = df['Q5_movie']
    df['drink_raw'] = df['Q6_drink']
    
    # C) Drop old columns
    df.drop(columns=[
        'Q1_complexity','Q2_ingredients','Q3_setting','Q4_cost',
        'Q5_movie','Q6_drink','Q7_reminds','Q8_hot_sauce'
    ], inplace=True, errors='ignore')
    
    # Reorder columns (optional)
    front_cols = ['id','label']
    new_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + new_cols]
    
    return df

#####################
# 5. DRIVER/EXAMPLE #
#####################

if __name__ == "__main__":
    # Example usage:
    csv_path = 'cleaned_data_combined.csv'  # Replace with actual path
    raw_df = pd.read_csv(csv_path)
    
    cleaned_df = clean_data_advanced(raw_df)
    
    print("\n=== HEAD OF CLEANED DATA ===")
    print(cleaned_df.head(10))
    
    print("\n=== COLUMN INFO ===")
    cleaned_df.info()
    
    print("\n=== SAMPLE VALUE COUNTS for movie_std ===")
    print(cleaned_df['movie_std'].value_counts(dropna=False))
    
    print("\n=== SAMPLE VALUE COUNTS for drink_std ===")
    print(cleaned_df['drink_std'].value_counts(dropna=False))
    
    # Save the new dataset
    cleaned_df.to_csv('cleaned_data_advanced.csv', index=False)
    print("\n[Done] Cleaned data saved to cleaned_data_advanced.csv!")
