import pickle
import jax
import jax.numpy as jnp
import optax

# Load molecular data from a pickle file
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['train'], data['test']

# Define a simple neural network architecture
class SubGraphNN:
    def __init__(self, input_dim, output_dim):
        self.init_key = jax.random.PRNGKey(0)
        self.params = self.initialize_params(input_dim, output_dim)

    def initialize_params(self, input_dim, output_dim):
        # Placeholder for parameters initialization
        return {'weights': jax.random.normal(self.init_key, (input_dim, output_dim))}

    def forward(self, x):
        # Forward pass through the network
        return jnp.dot(x, self.params['weights'])

# Training function
def train_model(train_data, epochs=100, learning_rate=0.001):
    model = SubGraphNN(input_dim=train_data.shape[1], output_dim=1)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model.params)

    for epoch in range(epochs):
        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(model.params, train_data)
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        model.params = optax.apply_updates(model.params, updates)

        if epoch % 10 == 0:  # Save parameters every 10 epochs
            save_params(model.params, epoch)

# Loss function
def loss_fn(params, x):
    preds = model.forward(x)
    return jnp.mean((preds - x) ** 2)

# Save model parameters
def save_params(params, epoch):
    with open(f'model_params_epoch_{epoch}.pkl', 'wb') as f:
        pickle.dump(params, f)

if __name__ == '__main__':
    train_data, test_data = load_data('molecular_data.pkl')
    train_model(train_data)