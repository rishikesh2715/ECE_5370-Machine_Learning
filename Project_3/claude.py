import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Function to generate the dataset D
def generate_dataset(L=100, N=25):
    datasets = []
    
    for l in range(L):
        # Generate X from uniform distribution U(0,1)
        X = np.random.uniform(0, 1, N)
        
        # Generate noise from Gaussian distribution N(0, 0.3)
        epsilon = np.random.normal(0, 0.3, N)
        
        # Generate target values: t = sin(2πX) + ε
        t = np.sin(2 * np.pi * X) + epsilon
        
        # Sort X and corresponding t for easier plotting
        sort_indices = np.argsort(X)
        X = X[sort_indices]
        t = t[sort_indices]
        
        datasets.append((X, t))
    
    return datasets

# Function to calculate Gaussian basis function
def gaussian_basis_function(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s ** 2))

# Function to fit model with Gaussian basis functions and regularization
def fit_model(X, t, s=0.1, lambda_val=0):
    N = len(X)
    
    # Use each x_n as center for a basis function
    centers = X
    M = len(centers)
    
    # Design matrix Phi
    Phi = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            Phi[i, j] = gaussian_basis_function(X[i], centers[j], s)
    
    # Add regularization with parameter lambda
    A = Phi.T @ Phi + lambda_val * np.eye(M)
    b = Phi.T @ t
    
    # Calculate weights: w = (Phi.T * Phi + lambda * I)^-1 * Phi.T * t
    w = np.linalg.solve(A, b)
    
    return w, centers

# Function to make predictions with the model
def predict(X_test, weights, centers, s=0.1):
    N_test = len(X_test)
    M = len(centers)
    
    # Design matrix for test points
    Phi_test = np.zeros((N_test, M))
    for i in range(N_test):
        for j in range(M):
            Phi_test[i, j] = gaussian_basis_function(X_test[i], centers[j], s)
    
    # Make predictions
    y_pred = Phi_test @ weights
    
    return y_pred

# Function to calculate bias^2, variance, and test error
def calculate_metrics(datasets, lambda_val, s=0.1, test_size=1000):
    L = len(datasets)
    X_test = np.linspace(0, 1, test_size)
    y_true = np.sin(2 * np.pi * X_test)  # True function without noise
    
    # Collect predictions for each dataset
    all_predictions = np.zeros((L, test_size))
    
    for l, (X, t) in enumerate(datasets):
        weights, centers = fit_model(X, t, s, lambda_val)
        predictions = predict(X_test, weights, centers, s)
        all_predictions[l] = predictions
    
    # Calculate average prediction
    f_avg = np.mean(all_predictions, axis=0)
    
    # Calculate bias^2
    bias_squared = np.mean((f_avg - y_true) ** 2)
    
    # Calculate variance
    variance = np.mean(np.mean((all_predictions - f_avg.reshape(1, -1)) ** 2, axis=0))
    
    # Calculate test error (average over all datasets)
    test_error = np.mean(np.mean((all_predictions - y_true.reshape(1, -1)) ** 2, axis=1))
    
    return bias_squared, variance, test_error

def main():
    # Generate L=100 datasets, each with N=25 samples
    print("Generating datasets...")
    datasets = generate_dataset(L=100, N=25)
    
    # Define regularization parameter values to test
    lambda_values = np.logspace(-6, 0, 15)
    
    # Arrays to store results
    bias_squared_values = []
    variance_values = []
    test_error_values = []
    
    # Calculate metrics for each lambda value
    print("Calculating metrics for different lambda values...")
    for lambda_val in lambda_values:
        bias_squared, variance, test_error = calculate_metrics(datasets, lambda_val, s=0.1)
        bias_squared_values.append(bias_squared)
        variance_values.append(variance)
        test_error_values.append(test_error)
        print(f"Lambda: {lambda_val:.6f}, Bias^2: {bias_squared:.6f}, Variance: {variance:.6f}, Test Error: {test_error:.6f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.loglog(lambda_values, bias_squared_values, 'r-', label='Bias^2')
    plt.loglog(lambda_values, variance_values, 'b-', label='Variance')
    plt.loglog(lambda_values, test_error_values, 'g-', label='Test Error')
    plt.loglog(lambda_values, np.array(bias_squared_values) + np.array(variance_values), 'k--', label='Bias^2 + Variance')
    plt.xlabel('Regularization Parameter (λ)')
    plt.ylabel('Error')
    plt.title('Bias-Variance Tradeoff with Regularization')
    plt.legend()
    plt.grid(True)
    plt.savefig('bias_variance_tradeoff.png')
    plt.show()

    # Visualize the model fit for one of the datasets with different lambda values
    print("Visualizing model fits...")
    X, t = datasets[0]  # Use the first dataset for visualization
    X_dense = np.linspace(0, 1, 1000)
    y_true = np.sin(2 * np.pi * X_dense)
    
    plt.figure(figsize=(15, 10))
    
    # Choose a few lambda values to visualize
    lambda_viz = [0.000001, 0.001, 0.1, 1.0]
    
    for i, lambda_val in enumerate(lambda_viz):
        weights, centers = fit_model(X, t, s=0.1, lambda_val=lambda_val)
        y_pred = predict(X_dense, weights, centers, s=0.1)
        
        plt.subplot(2, 2, i + 1)
        plt.scatter(X, t, c='b', label='Training Data')
        plt.plot(X_dense, y_true, 'g-', label='True Function')
        plt.plot(X_dense, y_pred, 'r-', label=f'Fitted Model (λ={lambda_val})')
        plt.title(f'Model Fit with λ={lambda_val}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_fits.png')
    plt.show()

if __name__ == "__main__":
    main()