# Diabetes Health Indicators Analysis - Comprehensive Documentation

## Overview
This Jupyter notebook (`dibetics.ipynb`) performs a comprehensive analysis of diabetes health indicators using the BRFSS 2015 dataset. The analysis includes data exploration, visualization, correlation analysis, and insights into factors associated with diabetes risk.

## Dataset Information
- **Source**: BRFSS 2015 (Behavioral Risk Factor Surveillance System)
- **File**: `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`
- **Size**: 70,693 rows × 22 columns
- **Target Variable**: Diabetes (binary: 0=No, 1=Yes)
- **Balance**: 50-50 split between diabetic and non-diabetic cases

## Cell-by-Cell Documentation

### Cell 0: Import Libraries and Setup
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Set seaborn style for prettier plots
sns.set(style="whitegrid")
```
**Purpose**: Import essential libraries for data analysis and visualization
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib.pyplot`: Basic plotting
- `seaborn`: Statistical data visualization
- `%matplotlib inline`: Display plots in notebook
- `sns.set(style="whitegrid")`: Set consistent, professional plot styling

### Cell 1: Load Dataset with Column Names
```python
column_names = [
    'Diabetes', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseOrAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
    'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
    'Sex', 'Age', 'Education', 'Income'
]
df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv', header=None, names=column_names)
```
**Purpose**: Load the dataset and assign meaningful column names
- **Column Definitions**:
  - `Diabetes`: Target variable (0=No, 1=Yes)
  - `HighBP`: High blood pressure (0=No, 1=Yes)
  - `HighChol`: High cholesterol (0=No, 1=Yes)
  - `CholCheck`: Cholesterol check in past 5 years (0=No, 1=Yes)
  - `BMI`: Body Mass Index (continuous)
  - `Smoker`: Smoked at least 100 cigarettes (0=No, 1=Yes)
  - `Stroke`: Ever had a stroke (0=No, 1=Yes)
  - `HeartDiseaseOrAttack`: Heart disease or heart attack (0=No, 1=Yes)
  - `PhysActivity`: Physical activity in past 30 days (0=No, 1=Yes)
  - `Fruits`: Consume fruit 1 or more times per day (0=No, 1=Yes)
  - `Veggies`: Consume vegetables 1 or more times per day (0=No, 1=Yes)
  - `HvyAlcoholConsump`: Heavy alcohol consumption (0=No, 1=Yes)
  - `AnyHealthcare`: Any kind of health care coverage (0=No, 1=Yes)
  - `NoDocbcCost`: Could not see doctor due to cost (0=No, 1=Yes)
  - `GenHlth`: General health (1=Excellent, 5=Poor)
  - `MentHlth`: Mental health (days in past 30, 0-30)
  - `PhysHlth`: Physical health (days in past 30, 0-30)
  - `DiffWalk`: Difficulty walking or climbing stairs (0=No, 1=Yes)
  - `Sex`: Sex (0=Female, 1=Male)
  - `Age`: Age groups (1=18-24, 2=25-34, ..., 13=80+)
  - `Education`: Education level (1=Never attended, 6=College graduate)
  - `Income`: Income level (1=Less than $10k, 8=More than $75k)

### Cell 2: Display First Few Rows
```python
df.head()
```
**Purpose**: Display the first 5 rows of the dataset to understand the data structure and format

### Cell 3: Dataset Information
```python
df.info()
```
**Purpose**: Display comprehensive information about the dataset
- **Output**: Shows data types, non-null counts, and memory usage
- **Key Finding**: All columns are initially stored as 'object' type, indicating mixed data types

### Cell 4: Statistical Summary
```python
df.describe()
```
**Purpose**: Generate descriptive statistics for all columns
- **Output**: Count, mean, std, min, 25%, 50%, 75%, max for each column

### Cell 5: Check for Missing Values (Commented Out)
```python
# df.isnull().sum()
```
**Purpose**: Check for missing values (commented out but would show NaN counts per column)

### Cell 6: BMI Distribution Histogram
```python
plt.figure(figsize=(10, 6))
sns.histplot(df["BMI"], bins=30, kde=True, color='teal')
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()
```
**Purpose**: Visualize the distribution of BMI values
- **Features**: Histogram with 30 bins, kernel density estimation overlay
- **Color**: Teal for visual appeal
- **Insight**: Shows the distribution pattern of BMI across the population

### Cell 7: High Blood Pressure vs Diabetes
```python
plt.figure(figsize=(8,6))
sns.countplot(x='HighBP', hue='Diabetes', data=df)
plt.title('High Blood Pressure by Diabetes Status')
plt.xlabel('High Blood Pressure (0=No, 1=Yes)')
plt.ylabel('Count')
plt.legend(title='Diabetes', labels=['No', 'Yes'])
plt.show()
```
**Purpose**: Analyze the relationship between high blood pressure and diabetes
- **Visualization**: Stacked bar chart showing counts by diabetes status
- **Insight**: Reveals if high blood pressure is associated with diabetes prevalence

### Cell 8: Age Group Analysis
```python
age_mapping = {
    1: '18-24', 2: '25-34', 3: '35-44', 4: '45-54',
    5: '55-64', 6: '65-74', 7: '75+'
}
df['AgeGroup'] = df['Age'].map(age_mapping)

plt.figure(figsize=(12,6))
sns.countplot(x='AgeGroup', hue='Diabetes', data=df)
plt.title('Age Group by Diabetes Status')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Diabetes', labels=['No', 'Yes'])
plt.show()
```
**Purpose**: Analyze diabetes prevalence across different age groups
- **Data Transformation**: Maps numeric age codes to readable age ranges
- **Visualization**: Grouped bar chart showing diabetes distribution by age
- **Insight**: Reveals age-related patterns in diabetes prevalence

### Cell 9: NaN Analysis
```python
print("=== NaN Analysis ===")
print("\n1. Total NaN values per column:")
nan_counts = df.isnull().sum()
print(nan_counts[nan_counts > 0])

print(f"\n2. Total rows with any NaN: {df.isnull().any(axis=1).sum()}")
print(f"Total rows: {len(df)}")
print(f"Percentage of rows with NaN: {(df.isnull().any(axis=1).sum() / len(df)) * 100:.2f}%")

print("\n3. BMI-specific NaN analysis:")
bmi_nan_count = df["BMI"].isnull().sum()
print(f"BMI NaN count: {bmi_nan_count}")
print(f"BMI valid values: {df['BMI'].notnull().sum()}")

if bmi_nan_count > 0:
    print(f"\n4. BMI statistics (excluding NaN):")
    print(f"Mean BMI: {df['BMI'].mean():.2f}")
    print(f"Median BMI: {df['BMI'].median():.2f}")
    print(f"Min BMI: {df['BMI'].min():.2f}")
    print(f"Max BMI: {df['BMI'].max():.2f}")
    
    print(f"\n5. Sample of rows with NaN BMI:")
    nan_bmi_rows = df[df["BMI"].isnull()]
    print(nan_bmi_rows[['Diabetes', 'BMI', 'Age', 'Sex']].head())
```
**Purpose**: Comprehensive analysis of missing values in the dataset
- **Key Findings**:
  - 68.52% of rows contain NaN values
  - Most NaN values are in the AgeGroup column (created from Age mapping)
  - Only 1 BMI value is missing
  - BMI range: 12.00 to 98.00, mean: 29.86

### Cell 10: NaN Handling Strategies
```python
print("=== NaN Handling Strategies ===")

# Strategy 1: Keep NaN values
print("\n1. Keeping NaN values (recommended for visualization):")
print("   - Seaborn automatically excludes NaN from histograms")
print("   - No data loss, but some rows won't be included in analysis")

# Strategy 2: Remove rows with NaN BMI
print("\n2. Removing rows with NaN BMI:")
df_no_nan_bmi = df.dropna(subset=['BMI'])
print(f"   Original dataset: {len(df)} rows")
print(f"   After removing NaN BMI: {len(df_no_nan_bmi)} rows")
print(f"   Rows removed: {len(df) - len(df_no_nan_bmi)}")

# Strategy 3: Fill NaN with mean BMI
print("\n3. Filling NaN BMI with mean value:")
mean_bmi = df['BMI'].mean()
df_filled_mean = df.copy()
df_filled_mean['BMI'] = df_filled_mean['BMI'].fillna(mean_bmi)
print(f"   Mean BMI used for filling: {mean_bmi:.2f}")

# Strategy 4: Fill NaN with median BMI
print("\n4. Filling NaN BMI with median value:")
median_bmi = df['BMI'].median()
df_filled_median = df.copy()
df_filled_median['BMI'] = df_filled_median['BMI'].fillna(median_bmi)
print(f"   Median BMI used for filling: {median_bmi:.2f}")

print("\n=== Recommendation ===")
print("For your histogram, you can use any of these approaches:")
print("- Keep NaN (seaborn handles it automatically)")
print("- Remove NaN rows if the loss is acceptable")
print("- Fill with mean/median if you want to preserve all rows")
```
**Purpose**: Demonstrate different strategies for handling missing values
- **Strategies Covered**:
  1. Keep NaN values (seaborn handles automatically)
  2. Remove rows with missing BMI (minimal data loss)
  3. Fill with mean BMI (29.86)
  4. Fill with median BMI (29.00)
- **Recommendation**: Keep NaN for visualization, remove or fill for analysis

### Cell 11: Category vs Object Data Types
```python
print("=== Category vs Object Data Types ===\n")

print("1. OBJECT dtype (default for text data):")
print("   - Stores each value as a separate string")
print("   - No memory optimization")
print("   - No inherent ordering")
print("   - Slower for operations on repeated values")

print("\n2. CATEGORY dtype (optimized for repeated values):")
print("   - Stores unique values once, then uses integer codes")
print("   - Significant memory savings for repeated values")
print("   - Can have ordered categories")
print("   - Faster operations on categorical data")

# Demonstrate with Diabetes column
print("\n=== Practical Example with Diabetes Column ===")
print(f"Current Diabetes column dtype: {df['Diabetes'].dtype}")
print(f"Unique values: {df['Diabetes'].unique()}")
print(f"Value counts:")
print(df['Diabetes'].value_counts())

# Memory usage comparison
print(f"\nMemory usage (object): {df['Diabetes'].memory_usage(deep=True)} bytes")

# Convert to category and compare
df_temp = df.copy()
df_temp['Diabetes'] = df_temp['Diabetes'].astype('category')
print(f"Memory usage (category): {df_temp['Diabetes'].memory_usage(deep=True)} bytes")

# Calculate memory savings
object_memory = df['Diabetes'].memory_usage(deep=True)
category_memory = df_temp['Diabetes'].memory_usage(deep=True)
savings = ((object_memory - category_memory) / object_memory) * 100
print(f"Memory savings: {savings:.1f}%")
```
**Purpose**: Explain the difference between object and category data types
- **Key Insights**:
  - Object dtype: Stores each value separately, no optimization
  - Category dtype: Stores unique values once, uses integer codes
  - Memory savings: 97.6% for Diabetes column
  - Performance: Category dtype is faster for repeated operations

### Cell 12: Performance Comparison
```python
print("=== Operation Performance Comparison ===")

# Create test dataframes
df_object = df.copy()
df_category = df.copy()
df_category['Diabetes'] = df_category['Diabetes'].astype('category')

# Test 1: Value counting
import time

print("\n1. Value counting performance:")
start_time = time.time()
object_counts = df_object['Diabetes'].value_counts()
object_time = time.time() - start_time

start_time = time.time()
category_counts = df_category['Diabetes'].value_counts()
category_time = time.time() - start_time

print(f"   Object dtype: {object_time:.6f} seconds")
print(f"   Category dtype: {category_time:.6f} seconds")
print(f"   Speed improvement: {object_time/category_time:.1f}x faster")

# Test 2: Grouping operations
print("\n2. Grouping operations:")
start_time = time.time()
object_grouped = df_object.groupby('Diabetes').size()
object_group_time = time.time() - start_time

start_time = time.time()
category_grouped = df_category.groupby('Diabetes').size()
category_group_time = time.time() - start_time

print(f"   Object dtype: {object_group_time:.6f} seconds")
print(f"   Category dtype: {category_group_time:.6f} seconds")
print(f"   Speed improvement: {object_group_time/category_group_time:.1f}x faster")

# Test 3: Filtering
print("\n3. Filtering operations:")
start_time = time.time()
object_filtered = df_object[df_object['Diabetes'] == '1.0']
object_filter_time = time.time() - start_time

start_time = time.time()
category_filtered = df_category[df_category['Diabetes'] == '1.0']
category_filter_time = time.time() - start_time

print(f"   Object dtype: {object_filter_time:.6f} seconds")
print(f"   Category dtype: {category_filter_time:.6f} seconds")
print(f"   Speed improvement: {object_filter_time/category_filter_time:.1f}x faster")
```
**Purpose**: Demonstrate performance differences between object and category dtypes
- **Performance Tests**:
  1. Value counting: 2.6x faster with category
  2. Grouping operations: 4.4x faster with category
  3. Filtering operations: 3.1x faster with category
- **Conclusion**: Category dtype significantly improves performance for categorical data

### Cell 13: Advanced Category Features
```python
print("=== Advanced Category Features ===")

# Create ordered categories
print("\n1. Ordered Categories:")
education_levels = ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0']
df['Education'] = pd.Categorical(df['Education'], categories=education_levels, ordered=True)

print(f"Education categories: {df['Education'].cat.categories}")
print(f"Is ordered: {df['Education'].cat.ordered}")

# Demonstrate ordering
print("\n2. Sorting with ordered categories:")
education_counts = df['Education'].value_counts().sort_index()
print(education_counts)

# Category-specific methods
print("\n3. Category-specific methods:")
print(f"Number of categories: {df['Education'].cat.categories.size}")
print(f"Category codes: {df['Education'].cat.codes[:10]}")
print(f"Category names: {df['Education'].cat.categories}")

# Adding/removing categories
print("\n4. Category management:")
print("Original categories:", df['Education'].cat.categories.tolist())

# Add a new category
df['Education'] = df['Education'].cat.add_categories(['7.0'])
print("After adding '7.0':", df['Education'].cat.categories.tolist())

# Remove unused categories
df['Education'] = df['Education'].cat.remove_unused_categories()
print("After removing unused:", df['Education'].cat.categories.tolist())
```
**Purpose**: Demonstrate advanced features of categorical data types
- **Features Covered**:
  - Ordered categories for education levels
  - Category-specific methods (codes, categories)
  - Category management (add/remove categories)
- **Use Case**: Education levels have natural ordering (1=Never attended to 6=College graduate)

### Cell 14: When to Use Category vs Object
```python
print("=== When to Use Category vs Object ===")

print("\n✅ USE CATEGORY when:")
print("   • Column has a limited number of unique values")
print("   • Values are repeated many times")
print("   • You need memory efficiency")
print("   • You want faster operations")
print("   • You need ordered categories")
print("   • Column represents categorical data")

print("\n❌ USE OBJECT when:")
print("   • Column contains free text")
print("   • Most values are unique")
print("   • You need to perform string operations")
print("   • Column contains mixed data types")
print("   • You're doing text processing or NLP")

print("\n=== Summary for Your Diabetes Dataset ===")
print("Columns that should be CATEGORY:")
categorical_cols = ['Diabetes', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 
                   'Stroke', 'HeartDiseaseOrAttack', 'PhysActivity', 'Fruits', 
                   'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 
                   'DiffWalk', 'Sex', 'Education', 'Income']

for col in categorical_cols:
    if col in df.columns:
        unique_count = df[col].nunique()
        total_count = len(df)
        print(f"   {col}: {unique_count} unique values out of {total_count} ({unique_count/total_count*100:.1f}% unique)")

print("\nColumns that should be NUMERIC:")
numeric_cols = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age']
for col in numeric_cols:
    if col in df.columns:
        print(f"   {col}: Continuous numeric values")
```
**Purpose**: Provide guidelines for choosing between category and object data types
- **Category Guidelines**: Use for limited unique values, repeated data, memory efficiency
- **Object Guidelines**: Use for free text, unique values, string operations
- **Dataset Analysis**: Identifies which columns should be categorical vs numeric

### Cell 15: Fix Data Issues
```python
print("=== Fixing Data Issues ===")

# Check the first few rows
print("1. Checking the first few rows:")
print(df.head())

# Remove the header row that got mixed into the data
print("\n2. Removing header row from data:")
header_rows = df[df['Diabetes'] == 'Diabetes_binary'].index
print(f"Found {len(header_rows)} header rows at indices: {header_rows.tolist()}")

# Remove these rows
df_clean = df.drop(header_rows).copy()
print(f"Dataset size after removing headers: {len(df_clean)} rows")

# Reset index
df_clean = df_clean.reset_index(drop=True)
print(f"Dataset size after resetting index: {len(df_clean)} rows")

# Check the cleaned data
print("\n3. Checking cleaned data:")
print("Diabetes column unique values:", df_clean['Diabetes'].unique())
print("Diabetes value counts:")
print(df_clean['Diabetes'].value_counts())
```
**Purpose**: Clean the dataset by removing header rows that got mixed into the data
- **Issue Identified**: Header row with 'Diabetes_binary' was included in the data
- **Solution**: Remove header rows and reset index
- **Result**: Clean dataset with 70,692 rows (removed 1 header row)

### Cell 16: Convert Data Types
```python
print("=== Converting Data Types ===")

# Convert numeric columns
numeric_columns = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
for col in numeric_columns:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        print(f"Converted {col} to numeric. NaN count: {df_clean[col].isnull().sum()}")

# Convert binary/categorical columns to numeric
binary_columns = ['Diabetes', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
                 'HeartDiseaseOrAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']

for col in binary_columns:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        print(f"Converted {col} to numeric. NaN count: {df_clean[col].isnull().sum()}")

print("\nData types after conversion:")
print(df_clean.dtypes)

print("\nDataset info:")
print(f"Shape: {df_clean.shape}")
print(f"Memory usage: {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```
**Purpose**: Convert all columns to appropriate numeric data types for analysis
- **Numeric Columns**: BMI, GenHlth, MentHlth, PhysHlth, Age, Education, Income
- **Binary Columns**: All yes/no variables converted to 0/1
- **Result**: All columns now have float64 dtype, ready for correlation analysis
- **Memory Usage**: 14.48 MB

### Cell 17: Create Correlation Matrix
```python
print("=== Creating Correlation Matrix ===")

# Check for non-numeric columns
non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print(f"Found non-numeric columns: {non_numeric_cols.tolist()}")
    print("These will be excluded from correlation analysis")

# Create DataFrame with only numeric columns
df_numeric = df_clean.select_dtypes(include=[np.number])
print(f"\nNumeric DataFrame shape: {df_numeric.shape}")
print(f"Columns included in correlation: {df_numeric.columns.tolist()}")

# Create correlation matrix
corr = df_numeric.corr()
print(f"\nCorrelation matrix shape: {corr.shape}")

# Check for NaN values in correlation matrix
nan_in_corr = corr.isnull().sum().sum()
if nan_in_corr > 0:
    print(f"Warning: Found {nan_in_corr} NaN values in correlation matrix")
    nan_cols = corr.columns[corr.isnull().any()].tolist()
    print(f"Columns with NaN correlations: {nan_cols}")
    
    # Remove columns with all NaN correlations
    corr = corr.dropna(axis=1, how='all').dropna(axis=0, how='all')
    print(f"Correlation matrix shape after removing NaN columns: {corr.shape}")

# Create the heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            fmt='.2f')
plt.title('Correlation Matrix of Health Indicators')
plt.tight_layout()
plt.show()

# Show top correlations with Diabetes
print("\n=== Top Correlations with Diabetes ===")
if 'Diabetes' in corr.columns:
    diabetes_corr = corr['Diabetes'].abs().sort_values(ascending=False)
    print(diabetes_corr.head(10))
else:
    print("Diabetes column not found in correlation matrix")
```
**Purpose**: Create a comprehensive correlation matrix and heatmap
- **Process**: 
  1. Select only numeric columns
  2. Calculate correlation matrix
  3. Handle NaN values (caused by constant columns)
  4. Create heatmap visualization
- **Key Findings**:
  - Education column had all NaN values (constant column)
  - Top correlations with Diabetes: GenHlth (0.41), HighBP (0.38), BMI (0.29)

### Cell 18: Data Diagnostics
```python
print("=== Data Diagnostics ===")

# Check for constant columns
print("\n1. Checking for constant columns:")
constant_cols = []
for col in df_numeric.columns:
    if df_numeric[col].nunique() <= 1:
        constant_cols.append(col)
        print(f"   {col}: {df_numeric[col].nunique()} unique values")

if constant_cols:
    print(f"\nFound {len(constant_cols)} constant columns: {constant_cols}")
    print("Constant columns will cause NaN in correlation matrix")
else:
    print("No constant columns found")

# Check columns with few unique values
print("\n2. Columns with few unique values:")
for col in df_numeric.columns:
    unique_count = df_numeric[col].nunique()
    if unique_count <= 5:
        print(f"   {col}: {unique_count} unique values - {df_numeric[col].unique()}")

# Check data ranges
print("\n3. Data ranges for key variables:")
key_vars = ['Diabetes', 'BMI', 'Age', 'GenHlth']
for var in key_vars:
    if var in df_numeric.columns:
        print(f"   {var}: min={df_numeric[var].min():.2f}, max={df_numeric[var].max():.2f}, mean={df_numeric[var].mean():.2f}")

# Sample of the data
print("\n4. Sample of numeric data:")
print(df_numeric[['Diabetes', 'BMI', 'Age', 'GenHlth']].head(10))
```
**Purpose**: Perform comprehensive diagnostics on the cleaned dataset
- **Key Findings**:
  - Education column is constant (all NaN values)
  - Most columns are binary (0/1 values)
  - GenHlth has 5 unique values (1-5 scale)
  - BMI range: 12.00-98.00, mean: 29.86
  - Age range: 1.00-13.00 (coded age groups)

### Cell 19: Individual Correlation Analysis
```python
print("=== Individual Correlation Analysis ===")

# Use cleaned DataFrame for correlation analysis
print("Using cleaned DataFrame (df_clean) for correlation analysis...")

# Check if columns exist and are numeric
if 'PhysActivity' in df_clean.columns and 'Diabetes' in df_clean.columns:
    print(f"PhysActivity data type: {df_clean['PhysActivity'].dtype}")
    print(f"Diabetes data type: {df_clean['Diabetes'].dtype}")
    
    # Check for NaN values
    phys_nan = df_clean['PhysActivity'].isnull().sum()
    diabetes_nan = df_clean['Diabetes'].isnull().sum()
    print(f"PhysActivity NaN count: {phys_nan}")
    print(f"Diabetes NaN count: {diabetes_nan}")
    
    # Calculate correlation
    phys_activity_corr = df_clean['PhysActivity'].corr(df_clean['Diabetes'])
    print(f"\nCorrelation between Physical Activity and Diabetes: {phys_activity_corr:.4f}")
    
    # Show sample data
    print(f"\nSample data:")
    print(df_clean[['PhysActivity', 'Diabetes']].head(10))
    
    # Show value counts
    print(f"\nPhysActivity value counts:")
    print(df_clean['PhysActivity'].value_counts().sort_index())
    print(f"\nDiabetes value counts:")
    print(df_clean['Diabetes'].value_counts().sort_index())
    
else:
    print("Error: Required columns not found in df_clean")
    print(f"Available columns: {df_clean.columns.tolist()}")
```
**Purpose**: Demonstrate individual correlation calculation between Physical Activity and Diabetes
- **Key Finding**: Negative correlation (-0.1587) between physical activity and diabetes
- **Interpretation**: Higher physical activity is associated with lower diabetes risk
- **Data Quality**: No missing values in either column

### Cell 20: Comprehensive Correlation Analysis
```python
print("=== Key Health Indicator Correlations with Diabetes ===")

# Define key variables to analyze
key_variables = [
    'BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth',
    'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseOrAttack',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
    'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex', 'Education', 'Income'
]

# Calculate correlations with Diabetes
diabetes_correlations = {}
for var in key_variables:
    if var in df_clean.columns:
        try:
            corr_value = df_clean[var].corr(df_clean['Diabetes'])
            diabetes_correlations[var] = corr_value
        except Exception as e:
            print(f"Error calculating correlation for {var}: {e}")
            diabetes_correlations[var] = None

# Sort correlations by absolute value
sorted_correlations = sorted(diabetes_correlations.items(), 
                           key=lambda x: abs(x[1]) if x[1] is not None else 0, 
                           reverse=True)

print("\nCorrelations with Diabetes (sorted by strength):")
print("-" * 50)
for var, corr_val in sorted_correlations:
    if corr_val is not None:
        strength = "Strong" if abs(corr_val) > 0.3 else "Moderate" if abs(corr_val) > 0.1 else "Weak"
        direction = "Positive" if corr_val > 0 else "Negative"
        print(f"{var:20s}: {corr_val:7.4f} ({strength} {direction})")
    else:
        print(f"{var:20s}: Error in calculation")

# Show strongest correlations
print(f"\n=== Strongest Correlations (|r| > 0.1) ===")
strong_correlations = [(var, corr_val) for var, corr_val in sorted_correlations 
                      if corr_val is not None and abs(corr_val) > 0.1]

for var, corr_val in strong_correlations:
    direction = "increases" if corr_val > 0 else "decreases"
    print(f"• {var}: {corr_val:.4f} - Diabetes risk {direction} with higher {var}")

# Create visualization of top correlations
if len(strong_correlations) > 0:
    print(f"\n=== Visualization of Top Correlations ===")
    top_vars = [var for var, _ in strong_correlations[:8]]
    top_corrs = [corr_val for _, corr_val in strong_correlations[:8]]
    
    plt.figure(figsize=(12, 6))
    colors = ['red' if x < 0 else 'blue' for x in top_corrs]
    bars = plt.bar(range(len(top_vars)), top_corrs, color=colors, alpha=0.7)
    plt.xticks(range(len(top_vars)), top_vars, rotation=45, ha='right')
    plt.ylabel('Correlation with Diabetes')
    plt.title('Top Correlations with Diabetes Risk')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_corrs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if val > 0 else -0.01), 
                f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top')
    
    plt.tight_layout()
    plt.show()
```
**Purpose**: Comprehensive analysis of correlations between all health indicators and diabetes
- **Key Findings**:
  - **Strong Positive Correlations**: GenHlth (0.4076), HighBP (0.3815)
  - **Moderate Positive Correlations**: BMI (0.2934), HighChol (0.2892), Age (0.2787)
  - **Moderate Negative Correlations**: PhysActivity (-0.1587), Income (-0.2244)
- **Visualization**: Bar chart showing top 8 correlations with color coding (red=negative, blue=positive)

### Cell 21: BMI Distribution by Diabetes Status
```python
plt.figure(figsize=(10,6))
sns.boxplot(x='Diabetes', y='BMI', data=df)
plt.title('BMI Distribution by Diabetes Status')
plt.xlabel('Diabetes (0=No, 1=Yes)')
plt.ylabel('BMI')
plt.show()
```
**Purpose**: Visualize BMI distribution differences between diabetic and non-diabetic individuals
- **Visualization**: Box plot showing median, quartiles, and outliers
- **Insight**: Reveals if BMI distributions differ significantly between groups

### Cell 22: Physical Activity Analysis
```python
phys_mapping = {0: 'No', 1: 'Yes'}
df['PhysActivity'] = df['PhysActivity'].map(phys_mapping)

plt.figure(figsize=(8,6))
sns.countplot(x='PhysActivity', hue='Diabetes', data=df)
plt.title('Physical Activity by Diabetes Status')
plt.xlabel('Physical Activity')
plt.ylabel('Count')
plt.legend(title='Diabetes', labels=['No', 'Yes'])
plt.show()
```
**Purpose**: Analyze the relationship between physical activity and diabetes
- **Data Transformation**: Maps 0/1 to 'No'/'Yes' for better readability
- **Visualization**: Grouped bar chart showing physical activity distribution by diabetes status
- **Insight**: Shows if physical activity patterns differ between diabetic and non-diabetic individuals

### Cell 23: Education Level Analysis
```python
education_mapping = {
    1: 'Never attended school', 2: 'Elementary', 3: 'Some high school', 4: 'High school graduate',
    5: 'Some college', 6: 'College graduate'
}
df['EducationLevel'] = df['Education'].map(education_mapping)

plt.figure(figsize=(12,6))
sns.countplot(x='EducationLevel', hue='Diabetes', data=df)
plt.title('Education Level by Diabetes Status')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.legend(title='Diabetes', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.show()
```
**Purpose**: Analyze the relationship between education level and diabetes
- **Data Transformation**: Maps numeric codes to descriptive education levels
- **Visualization**: Grouped bar chart with rotated x-axis labels
- **Insight**: Reveals if education level is associated with diabetes prevalence

### Cell 24: Income Level Analysis
```python
income_mapping = {
    1: 'Less than $10k', 2: '$10k-$15k', 3: '$15k-$20k', 4: '$20k-$25k',
    5: '$25k-$35k', 6: '$35k-$50k', 7: '$50k-$75k', 8: 'More than $75k'
}
df['IncomeLevel'] = df['Income'].map(income_mapping)

plt.figure(figsize=(12,6))
sns.countplot(x='IncomeLevel', hue='Diabetes', data=df)
plt.title('Income Level by Diabetes Status')
plt.xlabel('Income Level')
plt.ylabel('Count')
plt.legend(title='Diabetes', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.show()
```
**Purpose**: Analyze the relationship between income level and diabetes
- **Data Transformation**: Maps numeric codes to descriptive income ranges
- **Visualization**: Grouped bar chart showing income distribution by diabetes status
- **Insight**: Reveals if income level is associated with diabetes prevalence

### Cell 25: BMI vs Age Scatter Plot
```python
plt.figure(figsize=(10,6))
sns.scatterplot(x='Age', y='BMI', hue='Diabetes', data=df, alpha=0.6)
plt.title('BMI vs Age by Diabetes Status')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()
```
**Purpose**: Visualize the relationship between BMI and age, colored by diabetes status
- **Visualization**: Scatter plot with transparency (alpha=0.6) for better visibility
- **Color Coding**: Different colors for diabetic vs non-diabetic individuals
- **Insight**: Reveals patterns in BMI-age relationships and how they differ by diabetes status

## Key Insights from the Analysis

### Strongest Risk Factors for Diabetes (Positive Correlations):
1. **General Health (0.41)**: Poor general health is strongly associated with diabetes
2. **High Blood Pressure (0.38)**: Hypertension is a major risk factor
3. **BMI (0.29)**: Higher body mass index increases diabetes risk
4. **High Cholesterol (0.29)**: Elevated cholesterol is associated with diabetes
5. **Age (0.28)**: Older age increases diabetes risk
6. **Difficulty Walking (0.27)**: Mobility issues are associated with diabetes

### Protective Factors (Negative Correlations):
1. **Physical Activity (-0.16)**: Regular exercise reduces diabetes risk
2. **Income (-0.22)**: Higher income is associated with lower diabetes risk

### Data Quality Issues Identified:
1. **Header Row Contamination**: Original dataset had header row mixed in data
2. **Mixed Data Types**: All columns initially stored as object type
3. **Missing Values**: 68.52% of rows contained NaN values (mostly in derived columns)
4. **Constant Columns**: Education column had all NaN values

### Recommendations for Future Analysis:
1. **Data Preprocessing**: Always check for header contamination and mixed data types
2. **Missing Value Strategy**: Implement consistent approach for handling NaN values
3. **Data Type Optimization**: Use category dtype for categorical variables to improve performance
4. **Feature Engineering**: Consider creating composite health scores from multiple indicators
5. **Model Development**: Use the identified strong correlations for predictive modeling

## Technical Notes

### Performance Optimizations:
- **Category Dtype**: 97.6% memory savings for categorical columns
- **Operation Speed**: 2.6x-4.4x faster operations with category dtype
- **Memory Usage**: Reduced from 11.9+ MB to 14.48 MB after data type conversion

### Visualization Best Practices:
- **Consistent Styling**: Used seaborn whitegrid style throughout
- **Appropriate Chart Types**: Box plots for distributions, count plots for categorical data
- **Color Coding**: Red for negative correlations, blue for positive correlations
- **Readable Labels**: Mapped numeric codes to descriptive labels

This comprehensive analysis provides a solid foundation for understanding diabetes risk factors and can serve as a starting point for more advanced machine learning models or public health interventions.
