# GPT Planner

![R](https://img.shields.io/badge/R-%23276DC3.svg?style=flat&logo=r&logoColor=white)
![GPT](https://img.shields.io/badge/GPT-Planner-orange?style=flat&logoColor=white)

GPT Planner is an interactive R Shiny application designed to help users understand and plan Generative Pre-trained Transformer (GPT) model architectures. This tool allows you to explore various GPT model configurations and visualize their impact on model size, memory usage, and computational requirements. Please note, this is still under construction, therefore, use with caution. 

## Features

- Interactive inputs for key GPT model parameters
- Real-time calculation of total parameters, memory usage, and FLOPs
- Visualizations of parameter and memory breakdowns
- Detailed hidden layer analysis
- Explanation of model architecture and implications

## Installation

To run this application locally, you'll need R and the following packages installed:

```R
install.packages(c("shiny", "ggplot2", "dplyr", "scales", "DT"))
```

## Usage

1. Clone this repository:
```
git clone https://github.com/rohit43/gptPlanner.git
```

2. Navigate to the project directory:
```
cd gptPlanner
```

3. Run the Shiny app:
```R
shiny::runApp("gptPlanner.R")
```

4. The app will open in your default web browser. Adjust the input parameters on the left sidebar to see how they affect the GPT model's characteristics.

## Input Parameters

- Number of Hidden Layers
- Model Dimension
- Attention Heads
- Feed-forward Dimension
- Sequence Length
- Batch Size
- Vocabulary Size
- Data Type Size
- Optimizer
- Activation Function

## Output

The app provides several visualizations and data points:
- Total number of parameters
- Total memory usage
- Model size
- Total FLOPs
- Parameter breakdown by component
- Memory usage breakdown
- Hidden layer parameter distribution
- Detailed model architecture information

## Acknowledgements
- Inspired by the architecture of GPT models as described in various research papers.