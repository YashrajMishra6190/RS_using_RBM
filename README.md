# Recommender System Using RBM
---------
## What is Restricted Boltzman Machine? 
----------
A restricted term refers to that we are not allowed to connect the same type layer to each other. In other words, the two neurons of the input layer or hidden layer can’t connect to each other. Although the hidden layer and visible layer can be connected to each other.
![image](https://github.com/YashrajMishra6190/RS_using_RBM/assets/143256900/3c654caf-d08a-4126-a943-ee819dc5b334)

## Architecture of RBM
----------
1. Visible and Hidden Units: RBMs have two layers: visible units (input layer) representing observable data (e.g., pixel values, ratings), and hidden units capturing latent features or patterns in the data that are not directly observable.
2. Connections and Weights:  In a Restricted Boltzmann Machine (RBM), each visible unit is fully connected to every hidden unit, while connections within each layer are absent, adhering to the "restricted" naming convention. The strength of interaction between a visible unit and a hidden unit is determined by the weight assigned to their connection.             
3.  Model Convergence: Convergence in a Restricted Boltzmann Machine (RBM) means that after training, the adjustments to its weights and biases become minimal, indicating stability. This stability shows that the RBM has learned how to accurately recreate or generate data that resembles the patterns found in the original training data.

## Advantages of RBM
------
- RBMs can model complex, high-dimensional data.

- They can learn features automatically from data, reducing the need for manual feature engineering.

- They can handle missing data and can be used to fill in missing values.

## Disadvantages of RBM 
------------
- RBMs are moderately scalable but can struggle with very large datasets due to memory and computational limitations. This may require using alternative architectures better suited for handling big data scenarios.

- As dataset size increases, RBMs require more memory and computational resources for training and inference. This can become prohibitive for very large datasets that exceed available hardware capabilities.


