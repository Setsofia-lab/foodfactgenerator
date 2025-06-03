#  Food Fact Generator

A GPT-2 based fine-tuned model for generating comprehensive food facts from OpenFoodFacts dataset. This project processes nutritional data and generates natural language descriptions about food products including their nutritional content, categories, Nutri-Score ratings, and availability by country.

##  Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Processing](#dataset-processing)
- [Model Training](#model-training)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

##  Features

- **Comprehensive Data Processing**: Handles OpenFoodFacts dataset with proper tab-separated parsing
- **Country Code Mapping**: Maps country codes to full country names for better fact generation
- **Nutritional Analysis**: Processes energy, protein, fat, carbohydrate, sugar, and salt content
- **Category Classification**: Includes food categories and Nutri-Score grades
- **Advanced Training Pipeline**: Optimized GPT-2 fine-tuning with proper validation and monitoring
- **Sample Generation**: Built-in testing with sample food fact generation
- **Production Ready**: Complete model saving, metrics tracking, and error handling

##  Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (for large datasets)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Setsofia-lab/foodfactgenerator.git
cd food-fact-generator

# Create virtual environment
python -m venv foodfactgen
source foodfactgen/bin/activate  # On Windows: foodfactgen\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Requirements.txt
```
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
kagglehub
```

##  Quick Start

### 1. Download the Dataset

```bash
# The dataset will be automatically downloaded via kagglehub
# Or manually download from: https://www.kaggle.com/datasets/michaelfumery/enopenfoodfactsorgproducts
```

### 2. Process the Data

```python
from clean import preprocess_food_data

# Process OpenFoodFacts data and create training dataset
training_data, detailed_data = preprocess_food_data()
```

### 3. Train the Model

```python
from train import FoodFactTrainer

# Initialize and train
trainer = FoodFactTrainer()
trained_model, tokenizer, model = trainer.train_model()
```

### 4. Generate Food Facts

```python
# The model will automatically generate sample predictions after training
# Or use the saved model for inference
```

##  Dataset Processing

The preprocessing pipeline handles:

### Data Sources
- **Main Dataset**: `cleaned_openfoodfacts.csv` (990K+ products)
- **Countries Data**: `countries-en.csv` (240 countries)

### Processing Steps

1. **Tab-Separated Parsing**: Properly handles TSV format
2. **Column Selection**: Extracts relevant nutritional and categorical data
3. **Data Cleaning**: Handles missing values and inconsistent formatting
4. **Country Mapping**: Maps ISO codes to full country names
5. **Fact Generation**: Creates natural language food descriptions
6. **Quality Filtering**: Removes incomplete or invalid entries

### Output Format

```csv
input,output
"Chocolate Chip Cookies","Chocolate Chip Cookies contains 2040.0 kcal of energy, 7.0g of protein, 25.0g of fat, 56.0g of carbohydrates, and 33.0g of sugar per 100g. It belongs to the chocolate biscuits category and has a Nutri-Score grade of E. This product is available in France."
```

## Model Training

### Training Features

- **Base Model**: GPT-2 (117M parameters)
- **Fine-tuning**: Causal language modeling on food facts
- **Special Tokens**: Custom formatting for input-output pairs
- **Optimization**: Cosine learning rate scheduling with warmup
- **Validation**: 90/10 train-validation split with early stopping
- **Monitoring**: Comprehensive logging and metrics tracking

### Training Configuration

```python
# Key training parameters
batch_size = 8          # Per device (adjust based on GPU memory)
learning_rate = 5e-5    # Optimal for GPT-2 fine-tuning
num_epochs = 3          # Usually sufficient for convergence
max_length = 256        # Maximum sequence length
warmup_steps = 25%      # Of first epoch
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM | 16GB+ VRAM |
| RAM | 16GB | 32GB+ |
| Storage | 5GB | 10GB+ |
| Training Time | 4-6 hours | 2-3 hours |

## Usage

### Command Line

```bash
# Process data and train model
python model.py

# Process data only
python clean.py
```

### Python API

```python
from train import FoodFactTrainer

# Initialize trainer
trainer = FoodFactTrainer(
    model_name="gpt2",
    output_dir="./Users/samuelsetsofia/dev/projects/foodfactgenerator/gpt2_food_fact_model"
)

# Train model
trainer_obj, tokenizer, model = trainer.train_model("food_facts_training_dataset.csv")

# Generate predictions
trainer.generate_sample_predictions(tokenizer, model, num_samples=10)
```

### Loading Pre-trained Model

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load your trained model
model = GPT2LMHeadModel.from_pretrained("./Users/samuelsetsofia/dev/projects/foodfactgenerator/gpt2_food_fact_model")
tokenizer = GPT2Tokenizer.from_pretrained("./Users/samuelsetsofia/dev/projects/foodfactgenerator/gpt2_food_fact_model")

# Generate food fact
def generate_food_fact(product_name):
    prompt = f"<|startoftext|>Input: {product_name} -> Output: "
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Output: ")[-1]

# Example usage
fact = generate_food_fact("Greek Yogurt")
print(fact)
```

##  Project Structure

```
food-fact-generator/
├── README.md
├── requirements.txt
├── clean.py                    # Data preprocessing script
├── model.py                    # Model training script
├── models/
│   └── gpt2-food-fact-model_20231201_143022/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_metrics.json
│       └── logs/
└── examples/
    ├── sample_outputs.txt
    └── inference_example.py
```

##  Configuration

### Training Parameters

```python
# Modify these in the FoodFactTrainer class
TRAINING_CONFIG = {
    'num_epochs': 3,
    'batch_size': 8,
    'learning_rate': 5e-5,
    'max_length': 256,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'save_steps': 500,
    'eval_steps': 500,
    'early_stopping_patience': 2
}
```

### Data Processing Parameters

```python
# Modify these in the preprocess_food_data function
DATA_CONFIG = {
    'min_product_name_length': 2,
    'min_fact_length': 10,
    'validation_split': 0.1,
    'required_nutrition_fields': ['energy_100g', 'proteins_100g', 'fat_100g']
}
```

## Examples

### Input-Output Examples

**Input**: `"Organic Dark Chocolate"`

**Output**: 
```
"Organic Dark Chocolate contains 2165.0 kcal of energy, 12.0g of protein, 35.0g of fat, 45.0g of carbohydrates, and 24.0g of sugar per 100g. It belongs to the chocolate category and has a Nutri-Score grade of D. This product is available in Germany, France, United States."
```

**Input**: `"Low-Fat Greek Yogurt"`

**Output**: 
```
"Low-Fat Greek Yogurt contains 97.0 kcal of energy, 18.0g of protein, 0.2g of fat, 6.0g of carbohydrates, and 4.0g of sugar per 100g. It belongs to the dairy category and has a Nutri-Score grade of A. This product is available in Greece, United States, Canada."
```

### Batch Generation

```python
products = ["Quinoa Salad", "Pepperoni Pizza", "Green Tea", "Almond Butter"]
facts = [generate_food_fact(product) for product in products]

for product, fact in zip(products, facts):
    print(f"{product}: {fact}\n")
```

##  Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**
```bash
# Reduce batch size
per_device_train_batch_size = 4  # Instead of 8
gradient_accumulation_steps = 4  # To maintain effective batch size
```

#### 2. **Tab-Separated Values Error**
```python
# Ensure you're using sep='\t' for OpenFoodFacts data
df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
```

#### 3. **Missing Columns**
```python
# Check available columns
print("Available columns:", df.columns.tolist())
# Adjust column selection based on actual dataset
```

#### 4. **Tokenizer Issues**
```python
# Make sure special tokens are properly added
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
model.resize_token_embeddings(len(tokenizer))
```

### Performance Optimization

1. **Use Mixed Precision Training**: `fp16=True` (requires CUDA)
2. **Optimize Batch Size**: Start with 4, increase based on GPU memory
3. **Use Gradient Checkpointing**: For larger models
4. **Enable Data Parallel**: For multi-GPU setups

### Dataset Issues

- **Large File Size**: Process in chunks if memory is limited
- **Encoding Issues**: Use `encoding='utf-8', errors='ignore'`
- **Missing Data**: Filter out products with insufficient information


### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black *.py
flake8 *.py
```

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



---
