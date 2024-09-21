Initially, I used Palmetto to run the codes.

We must ensure your environment has the necessary libraries to run the code and successfully generate results. Here's a straightforward guide on the libraries used and how to proceed:
Libraries Used in the Assignment
The code relies on shared Python libraries for data processing, visualization, and neural network training. These libraries are essential for running notebooks properly. Here's a breakdown of the leading libraries:
NumPy:
Purpose: Handles numerical computations, matrix operations, and array manipulations.
How it's used: This library is a backbone for many numerical operations across the notebooks.
Matplotlib:
Purpose: Helps with plotting graphs, visualizing training results, and generating visual output for data.
How it's used: It's used in almost every notebook to display important metrics and visual data, such as loss graphs.
PyTorch:
Purpose: A robust framework used to build and train deep learning models.
How it's used: It forms the core of your neural networks, handling the model structure, forward and backward passes, and training steps.
Torchvision:
Purpose: A PyTorch library that deals with computer vision tasks, such as loading datasets and applying transformations.
How it's used: It's utilized for handling datasets like MNIST, applying necessary transformations, and organizing data loading.
Pandas:
Purpose: Provides tools for data manipulation and handling structured data.
How it's used: In some notebooks, it organizes and analyzes data in table-like formats, especially when applying Principal Component Analysis (PCA).
Scikit-learn (PCA):
Purpose: A library for data analysis and PCA is used here to reduce the dimensionality of data for better visualization.
How it's used: It's specifically applied to reduce the dimensions of the model weights, allowing easier visualization in two sizes.
How to Get Started
Install the Libraries:
To run your code smoothly, ensure all the necessary libraries are installed in your environment. You can install them using this single command: 
pip install numpy matplotlib torch torchvision pandas scikit-learn
Load and Run the Notebooks:
After setting up the environment, open Jupyter Notebook and run each notebook sequentially.
Follow the steps mentioned earlier in this guide to execute each notebook.
You can run the code and generate the results by ensuring these libraries are installed and following the steps outlined.
This process is straightforward, and these shared libraries support different aspects of deep learning tasks, including model training, data loading, and results visualization.
