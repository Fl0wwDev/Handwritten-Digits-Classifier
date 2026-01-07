import numpy as np

class GradientDescent:
    
    def __init__(self, gradient, learning_rate=0.01, max_iterations=1000, epsilon=1e-6, batch_size=1):
        self.gradient = gradient
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.batch_size = batch_size

    def descent(self, initial_point, data=None):
        current_point = initial_point
        
        for epoch in range(self.max_iterations):
            if data is not None:
                x, y = data
                indices = np.random.permutation(len(x))
                
                for i in range(0, len(x), self.batch_size):
                    batch_indices = indices[i:min(i + self.batch_size, len(x))]
                    x_batch = x[batch_indices]
                    y_batch = y[batch_indices]
                    
                    grad, _ = self.gradient(current_point, (x_batch, y_batch))
                    
                    if np.linalg.norm(grad) < self.epsilon:
                        return current_point
                    
                    current_point = self.update(current_point, grad)
            else:
                grad = self.gradient(current_point)
                
                if np.linalg.norm(grad) < self.epsilon:
                    return current_point
                
                current_point = self.update(current_point, grad)
        
        return current_point

    def update(self, point, gradient_value):
        return point - self.learning_rate * gradient_value
