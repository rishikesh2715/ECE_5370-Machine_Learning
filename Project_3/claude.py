import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def h(x):
    """True function h(x) = sin(2πx)"""
    return np.sin(2 * np.pi * x)

def generate_datasets(L, N):
    """Generate L datasets of N samples each"""
    datasets = []
    for l in range(L):
        # Generate uniform X values between 0 and 1
        X = np.random.uniform(0, 1, N)
        
        # Generate target values with Gaussian noise
        epsilon = np.random.normal(0, 0.3, N)
        t = h(X) + epsilon
        
        datasets.append((X, t))
    
    return datasets

def generate_test_data(num_points):
    """Generate test data of num_points samples"""
    X_test = np.linspace(0, 1, num_points)
    t_test_true = h(X_test)  # True function values without noise
    epsilon_test = np.random.normal(0, 0.3, num_points)
    t_test = t_test_true + epsilon_test  # Noisy test targets
    
    return X_test, t_test_true, t_test

def gaussian_basis(x, centers, s):
    """Compute Gaussian basis functions"""
    # Add bias term
    basis = np.ones((len(x), len(centers) + 1))
    
    # Compute Gaussian basis function values
    x = x.reshape(-1, 1)  # Ensure x is a column vector
    centers = centers.reshape(1, -1)  # Ensure centers is a row vector
    
    # Fill in the Gaussian basis terms (excluding the bias term)
    basis[:, 1:] = np.exp(-0.5 * ((x - centers) / s) ** 2)
    
    return basis

def fit_model(X, t, centers, s, lambda_val):
    """Fit linear regression model with Gaussian basis functions and regularization"""
    # Compute design matrix Phi
    Phi = gaussian_basis(X, centers, s)
    
    # Compute regularized weights using the normal equations
    I = np.eye(Phi.shape[1])
    I[0, 0] = 0  # Don't regularize the bias term
    w = np.linalg.solve(Phi.T @ Phi + lambda_val * I, Phi.T @ t)
    
    return w

def predict(X, w, centers, s):
    """Make predictions using the fitted model"""
    Phi = gaussian_basis(X, centers, s)
    return Phi @ w

def compute_metrics(lambda_values, datasets, centers, s, X_test, t_test_true, t_test):
    """Compute bias, variance, and test error for each lambda value"""
    L = len(datasets)
    
    # Arrays to store results
    bias_squared = np.zeros(len(lambda_values))
    variance = np.zeros(len(lambda_values))
    test_error = np.zeros(len(lambda_values))
    
    # For each regularization parameter
    for i, lambda_val in enumerate(lambda_values):
        # Store predictions for all L datasets on X_test points
        all_predictions = np.zeros((L, len(X_test)))
        
        # Fit model to each dataset and predict on test data
        for l in range(L):
            X, t = datasets[l]
            w = fit_model(X, t, centers, s, lambda_val)
            all_predictions[l] = predict(X_test, w, centers, s)
        
        # Compute average prediction on test data (f̄)
        f_bar = np.mean(all_predictions, axis=0)
        
        # Compute bias squared: (1/N) * sum((f̄(x) - h(x))^2)
        bias_squared[i] = np.mean((f_bar - t_test_true) ** 2)
        
        # Compute variance: (1/N) * sum((1/L) * sum((f^(l)(x) - f̄(x))^2))
        variance[i] = np.mean(np.var(all_predictions, axis=0))
        
        # Compute test error (average MSE on noisy test data)
        mse_per_model = np.mean((all_predictions - t_test.reshape(1, -1)) ** 2, axis=1)
        test_error[i] = np.mean(mse_per_model)
    
    return bias_squared, variance, test_error

def plot_results(lambda_values, bias_squared, variance, test_error):
    """Plot the results"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.log(lambda_values), bias_squared, 'b-', label='(bias)²')
    plt.plot(np.log(lambda_values), variance, 'r-', label='variance')
    plt.plot(np.log(lambda_values), bias_squared + variance, 'm-', label='(bias)² + variance')
    plt.plot(np.log(lambda_values), test_error, 'k-', label='test error')
    plt.xlabel('ln λ')
    plt.ylabel('Error')
    plt.ylim(0, 0.15)  # Match the y-axis range in the example
    plt.xlim(-3, 2)    # Match the x-axis range in the example
    plt.legend()
    plt.grid(True)
    plt.title('Bias-Variance Tradeoff')
    plt.savefig('bias_variance_tradeoff.png')
    plt.show()

def main():
    # Parameters
    L = 100  # Number of datasets
    N = 25   # Number of samples per dataset
    s = 0.1  # Width of Gaussian basis functions
    
    # Generate the datasets
    datasets = generate_datasets(L, N)
    
    # Choose regularization parameters (log scale)
    lambda_values = np.logspace(-3, 2, 20)
    
    # Generate centers for basis functions (evenly spaced in [0, 1])
    M = N  # Use same number of basis functions as data points
    centers = np.linspace(0, 1, M)
    
    # Generate test data (1000 points)
    X_test, t_test_true, t_test = generate_test_data(1000)
    
    # Compute bias, variance, and test error for each lambda
    bias_squared, variance, test_error = compute_metrics(
        lambda_values, datasets, centers, s, X_test, t_test_true, t_test
    )
    
    # Plot results
    plot_results(lambda_values, bias_squared, variance, test_error)

if __name__ == "__main__":
    main()