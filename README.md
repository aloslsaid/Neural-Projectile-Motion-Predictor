# Neural-Projectile-Motion-Predictor
Built a projectile motion prediction model using PyTorch and a feedforward neural network. The model learns the relationship between initial position, initial speed, launch angle, and elapsed time to predict horizontal and vertical positions. The project includes data preprocessing, feature scaling, model training, and evaluation using MAE and RMSE
rmse 0.02 m 

# Neural Projectile Motion Predictor

An educational machine learning project that uses PyTorch to predict the position of an unpowered projectile over time using a feedforward neural network.

The model takes initial horizontal and vertical positions, initial speed, launch angle, and elapsed time as input features. It predicts the projectile's horizontal and vertical coordinates.

### Technologies
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn

### Key Features
- Synthetic dataset generation based on basic projectile motion physics.
- Feature and target standardization using `StandardScaler`.
- Neural network training with configurable hidden layers and epochs.
- Model evaluation using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Early stopping to help prevent unnecessary training and monitor validation performance.

### Objective
Explore how neural networks learn nonlinear relationships in simulated physical systems and evaluate their predictions against mathematically generated data.
