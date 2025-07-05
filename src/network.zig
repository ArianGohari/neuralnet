const std = @import("std");
const tensor = @import("tensor.zig");
const activations = @import("activations.zig");
const loss = @import("loss.zig");

/// Activation function types
pub const Activation = enum {
    Relu,
    Sigmoid,
    Tanh,
};

/// Neural network layer with weights, biases, and activation function
pub const Layer = struct {
    weights: [][]f64,
    biases: []f64,
    activation: Activation,
    allocator: std.mem.Allocator,

    /// Forward pass through the layer
    pub fn forward(self: *const Layer, input: []const f64) ![]f64 {
        // Ensure input size matches layer input size
        std.debug.assert(self.weights[0].len == input.len);

        // Calculate weighted sum + bias for each neuron
        var output = try self.allocator.alloc(f64, self.biases.len);
        for (0..self.biases.len) |i| {
            output[i] = tensor.dot(self.weights[i], input) + self.biases[i];
        }

        // Apply activation function
        const activated_output = try self.applyActivation(output);
        self.allocator.free(output);

        return activated_output;
    }

    /// Apply the layer's activation function to output
    fn applyActivation(self: *const Layer, output: []const f64) ![]f64 {
        // Create 2D matrix for activation functions (they expect matrices)
        const output_matrix_data = [_][]const f64{output};
        const output_matrix = &output_matrix_data;

        const result_matrix = switch (self.activation) {
            .Relu => try activations.relu(output_matrix, self.allocator),
            .Sigmoid => try activations.sigmoid(output_matrix, self.allocator),
            .Tanh => try activations.tanh(output_matrix, self.allocator),
        };

        // Extract the single row from result matrix
        const result = try self.allocator.alloc(f64, result_matrix[0].len);
        @memcpy(result, result_matrix[0]);

        // Clean up the matrix
        self.allocator.free(result_matrix[0]);
        self.allocator.free(result_matrix);

        return result;
    }

    /// Backward pass through the layer
    /// Returns gradients with respect to inputs and updates weights/biases
    pub fn backward(self: *Layer, input: []const f64, z_values: []const f64, grad_output: []const f64, learning_rate: f64) ![]f64 {
        // Ensure dimensions match
        std.debug.assert(z_values.len == self.biases.len);
        std.debug.assert(grad_output.len == self.biases.len);
        std.debug.assert(input.len == self.weights[0].len);
        
        // Calculate activation derivative
        const z_matrix_data = [_][]const f64{z_values};
        const z_matrix = &z_matrix_data;
        
        const activation_grad_matrix = switch (self.activation) {
            .Relu => try activations.relu_derivative(z_matrix, self.allocator),
            .Sigmoid => try activations.sigmoid_derivative(z_matrix, self.allocator),
            .Tanh => try activations.tanh_derivative(z_matrix, self.allocator),
        };
        defer {
            self.allocator.free(activation_grad_matrix[0]);
            self.allocator.free(activation_grad_matrix);
        }
        
        // Element-wise multiply: grad_output * activation_derivative
        var delta = try self.allocator.alloc(f64, self.biases.len);
        for (0..self.biases.len) |i| {
            delta[i] = grad_output[i] * activation_grad_matrix[0][i];
        }
        defer self.allocator.free(delta);
        
        // Calculate gradients with respect to weights and biases
        for (0..self.weights.len) |i| {
            // Update biases: bias -= learning_rate * delta
            self.biases[i] -= learning_rate * delta[i];
            
            // Update weights: weight -= learning_rate * delta * input
            for (0..self.weights[i].len) |j| {
                self.weights[i][j] -= learning_rate * delta[i] * input[j];
            }
        }
        
        // Calculate gradients with respect to inputs for next layer
        var grad_input = try self.allocator.alloc(f64, input.len);
        @memset(grad_input, 0.0);
        
        for (0..self.weights.len) |i| {
            for (0..self.weights[i].len) |j| {
                grad_input[j] += delta[i] * self.weights[i][j];
            }
        }
        
        return grad_input;
    }
    
    /// Free layer memory
    pub fn deinit(self: *Layer) void {
        for (self.weights) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(self.weights);
        self.allocator.free(self.biases);
    }
};

/// Neural network with multiple layers
pub const NeuralNetwork = struct {
    layers: []Layer,
    allocator: std.mem.Allocator,

    /// Forward pass through the entire network
    pub fn forward(self: *const NeuralNetwork, input: []const f64) ![]f64 {
        var current_input = try self.allocator.alloc(f64, input.len);
        @memcpy(current_input, input);

        // Pass through each layer
        for (self.layers) |*layer| {
            const layer_output = try layer.forward(current_input);
            self.allocator.free(current_input);
            current_input = layer_output;
        }

        return current_input;
    }

    /// Train the network on a single sample
    pub fn train(self: *NeuralNetwork, input: []const f64, target: []const f64, learning_rate: f64) !f64 {
        // Forward pass - collect intermediate values for backward pass
        var layer_inputs = try self.allocator.alloc([]f64, self.layers.len + 1);
        defer {
            for (layer_inputs) |layer_input| {
                self.allocator.free(layer_input);
            }
            self.allocator.free(layer_inputs);
        }
        
        var layer_z_values = try self.allocator.alloc([]f64, self.layers.len);
        defer {
            for (layer_z_values) |z_values| {
                self.allocator.free(z_values);
            }
            self.allocator.free(layer_z_values);
        }
        
        // Store input
        layer_inputs[0] = try self.allocator.alloc(f64, input.len);
        @memcpy(layer_inputs[0], input);
        
        // Forward pass through each layer
        for (self.layers, 0..) |*layer, i| {
            // Calculate z = Wx + b (before activation)
            layer_z_values[i] = try self.allocator.alloc(f64, layer.biases.len);
            for (0..layer.biases.len) |j| {
                layer_z_values[i][j] = tensor.dot(layer.weights[j], layer_inputs[i]) + layer.biases[j];
            }
            
            // Apply activation to get layer output
            layer_inputs[i + 1] = try layer.applyActivation(layer_z_values[i]);
        }
        
        // Calculate loss
        const final_output = layer_inputs[self.layers.len];
        const current_loss = loss.mse(final_output, target);
        
        // Backward pass
        var grad_output = try loss.mse_derivative(final_output, target, self.allocator);
        
        // Backward through each layer (reverse order)
        var i = self.layers.len;
        while (i > 0) {
            i -= 1;
            const next_grad = try self.layers[i].backward(
                layer_inputs[i], 
                layer_z_values[i], 
                grad_output, 
                learning_rate
            );
            self.allocator.free(grad_output);
            grad_output = next_grad;
        }
        self.allocator.free(grad_output);
        
        return current_loss;
    }
    
    /// Free network memory
    pub fn deinit(self: *NeuralNetwork) void {
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
    }
};

/// Create a neural network with specified layer sizes and activations
pub fn createNetwork(layer_sizes: []const usize, layer_activations: []const Activation, allocator: std.mem.Allocator) !NeuralNetwork {
    // Ensure we have the right number of activations (one per hidden/output layer)
    std.debug.assert(layer_activations.len == layer_sizes.len - 1);

    var layers = try allocator.alloc(Layer, layer_sizes.len - 1);

    for (0..layers.len) |i| {
        const input_size = layer_sizes[i];
        const output_size = layer_sizes[i + 1];

        // Initialize weights randomly
        const weights = try tensor.random_init(output_size, input_size, -1.0, 1.0, allocator);

        // Initialize biases to zero
        const biases = try allocator.alloc(f64, output_size);
        @memset(biases, 0.0);

        layers[i] = Layer{
            .weights = weights,
            .biases = biases,
            .activation = layer_activations[i],
            .allocator = allocator,
        };
    }

    return NeuralNetwork{
        .layers = layers,
        .allocator = allocator,
    };
}

test "Layer forward pass works" {
    const allocator = std.testing.allocator;
    var weights = try allocator.alloc([]f64, 3);
    weights[0] = try allocator.alloc(f64, 2);
    weights[1] = try allocator.alloc(f64, 2);
    weights[2] = try allocator.alloc(f64, 2);
    weights[0][0] = 1.0; weights[0][1] = 0.5;
    weights[1][0] = -1.0; weights[1][1] = 2.0;
    weights[2][0] = 0.0; weights[2][1] = -1.0;
    var biases = try allocator.alloc(f64, 3);
    biases[0] = 0.0;
    biases[1] = 1.0;
    biases[2] = 0.5;
    var layer = Layer{
        .weights = weights,
        .biases = biases,
        .activation = Activation.Relu,
        .allocator = allocator,
    };
    defer layer.deinit();
    const input = [_]f64{ 2.0, 1.0 };
    const output = try layer.forward(&input);
    defer allocator.free(output);
    try std.testing.expectEqual(@as(usize, 3), output.len);
    try std.testing.expectApproxEqRel(@as(f64, 2.5), output[0], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 1.0), output[1], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 0.0), output[2], 1e-10);
}

test "NeuralNetwork forward pass works" {
    const allocator = std.testing.allocator;
    const layer_sizes = [_]usize{ 2, 3, 2 };
    const layer_activations = [_]Activation{ Activation.Relu, Activation.Sigmoid };
    var neural_network = try createNetwork(&layer_sizes, &layer_activations, allocator);
    defer neural_network.deinit();
    const input = [_]f64{ 1.0, 0.5 };
    const output = try neural_network.forward(&input);
    defer allocator.free(output);
    try std.testing.expectEqual(@as(usize, 2), output.len);
    for (output) |value| {
        try std.testing.expect(value >= 0.0);
        try std.testing.expect(value <= 1.0);
    }
}

test "createNetwork initialization works" {
    const allocator = std.testing.allocator;
    const layer_sizes = [_]usize{ 3, 4, 2, 1 };
    const layer_activations = [_]Activation{ 
        Activation.Relu, 
        Activation.Tanh, 
        Activation.Sigmoid 
    };
    var neural_network = try createNetwork(&layer_sizes, &layer_activations, allocator);
    defer neural_network.deinit();
    try std.testing.expectEqual(@as(usize, 3), neural_network.layers.len);
    try std.testing.expectEqual(@as(usize, 4), neural_network.layers[0].weights.len);
    try std.testing.expectEqual(@as(usize, 3), neural_network.layers[0].weights[0].len);
    try std.testing.expectEqual(@as(usize, 4), neural_network.layers[0].biases.len);
    try std.testing.expectEqual(Activation.Relu, neural_network.layers[0].activation);
    try std.testing.expectEqual(@as(usize, 2), neural_network.layers[1].weights.len);
    try std.testing.expectEqual(@as(usize, 4), neural_network.layers[1].weights[0].len);
    try std.testing.expectEqual(@as(usize, 2), neural_network.layers[1].biases.len);
    try std.testing.expectEqual(Activation.Tanh, neural_network.layers[1].activation);
    try std.testing.expectEqual(@as(usize, 1), neural_network.layers[2].weights.len);
    try std.testing.expectEqual(@as(usize, 2), neural_network.layers[2].weights[0].len);
    try std.testing.expectEqual(@as(usize, 1), neural_network.layers[2].biases.len);
    try std.testing.expectEqual(Activation.Sigmoid, neural_network.layers[2].activation);
    for (neural_network.layers) |layer| {
        for (layer.weights) |row| {
            for (row) |weight| {
                try std.testing.expect(weight >= -1.0);
                try std.testing.expect(weight < 1.0);
            }
        }
    }
    for (neural_network.layers) |layer| {
        for (layer.biases) |bias| {
            try std.testing.expectEqual(@as(f64, 0.0), bias);
        }
    }
}

test "network training reduces loss" {
    const allocator = std.testing.allocator;
    const layer_sizes = [_]usize{ 2, 3, 1 };
    const layer_activations = [_]Activation{ Activation.Sigmoid, Activation.Sigmoid };
    var neural_network = try createNetwork(&layer_sizes, &layer_activations, allocator);
    defer neural_network.deinit();
    const input = [_]f64{ 1.0, 0.0 };
    const target = [_]f64{ 1.0 };
    const learning_rate = 0.1;
    var initial_loss: f64 = undefined;
    var final_loss: f64 = undefined;
    for (0..100) |epoch| {
        const current_loss = try neural_network.train(&input, &target, learning_rate);
        if (epoch == 0) {
            initial_loss = current_loss;
        }
        if (epoch == 99) {
            final_loss = current_loss;
        }
    }
    try std.testing.expect(final_loss < initial_loss);
}

test "network training updates weights and biases" {
    const allocator = std.testing.allocator;
    const layer_sizes = [_]usize{ 1, 1 };
    const layer_activations = [_]Activation{ Activation.Sigmoid };
    var neural_network = try createNetwork(&layer_sizes, &layer_activations, allocator);
    defer neural_network.deinit();
    const initial_weight = neural_network.layers[0].weights[0][0];
    const initial_bias = neural_network.layers[0].biases[0];
    const input = [_]f64{ 1.0 };
    const target = [_]f64{ 0.0 };
    const learning_rate = 0.1;
    _ = try neural_network.train(&input, &target, learning_rate);
    const final_weight = neural_network.layers[0].weights[0][0];
    const final_bias = neural_network.layers[0].biases[0];
    try std.testing.expect(final_weight != initial_weight);
    try std.testing.expect(final_bias != initial_bias);
}
