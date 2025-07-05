# Neural Network Implementation in Zig

A simple neural network implementation in Zig

## Features

- Multi-layer perceptron with configurable architecture
- Sigmoid activation function
- Backpropagation training algorithm
- Custom tensor operations
- Loss function implementation

## Project Structure

```
src/
├── main.zig         # Example usage and XOR problem solver
├── network.zig      # Neural network implementation
├── tensor.zig       # Tensor operations
├── activations.zig  # Activation functions
└── loss.zig         # Loss functions
```

## Building and Running

```bash
# Build the project
zig build

# Run the neural network
./zig-out/bin/neuralnet
```

## Example Usage

The main program demonstrates training a neural network to solve the XOR problem:

- **Architecture**: 2 inputs → 4 hidden neurons → 1 output
- **Training data**: XOR truth table
- **Learning rate**: 0.5
- **Epochs**: 5000

The network learns to map:
- [0, 0] → 0
- [0, 1] → 1
- [1, 0] → 1
- [1, 1] → 0

## Requirements

- Zig compiler (latest stable version)
