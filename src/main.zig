const std = @import("std");
const network = @import("network.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // 1. Create a Network: 2 inputs -> 4 hidden -> 1 output
    const layer_sizes = [_]usize{ 2, 4, 1 };
    const layer_activations = [_]network.Activation{ network.Activation.Sigmoid, network.Activation.Sigmoid };

    var neural_network = try network.createNetwork(&layer_sizes, &layer_activations, allocator);
    defer neural_network.deinit();

    std.debug.print("Neural Network created: {d} -> {d} -> {d}\n", .{ 2, 4, 1 });

    // 2. Define Toy Dataset (XOR problem)
    const training_data = [_]struct { input: [2]f64, target: [1]f64 }{
        .{ .input = [_]f64{ 0.0, 0.0 }, .target = [_]f64{0.0} },
        .{ .input = [_]f64{ 0.0, 1.0 }, .target = [_]f64{1.0} },
        .{ .input = [_]f64{ 1.0, 0.0 }, .target = [_]f64{1.0} },
        .{ .input = [_]f64{ 1.0, 1.0 }, .target = [_]f64{0.0} },
    };

    std.debug.print("Training data (XOR problem):\n", .{});
    for (training_data) |sample| {
        std.debug.print("  Input: [{d:.1}, {d:.1}] -> Target: {d:.1}\n", .{ sample.input[0], sample.input[1], sample.target[0] });
    }

    // 3. Train the Model
    const learning_rate = 0.5;
    const epochs = 5000;

    std.debug.print("\nTraining for {d} epochs with learning rate {d:.2}...\n", .{ epochs, learning_rate });

    var epoch: usize = 0;
    while (epoch < epochs) : (epoch += 1) {
        var total_loss: f64 = 0.0;

        // Train on each sample in the dataset
        for (training_data) |sample| {
            const current_loss = try neural_network.train(&sample.input, &sample.target, learning_rate);
            total_loss += current_loss;
        }

        // Print progress every 1000 epochs
        if (epoch % 1000 == 0) {
            const avg_loss = total_loss / training_data.len;
            std.debug.print("  Epoch {d}: Average Loss = {d:.6}\n", .{ epoch, avg_loss });
        }
    }

    // 4. Print Predictions
    std.debug.print("\nFinal Predictions:\n", .{});
    for (training_data) |sample| {
        const prediction = try neural_network.forward(&sample.input);
        defer allocator.free(prediction);

        std.debug.print("  Input: [{d:.1}, {d:.1}] -> Prediction: {d:.4}, Target: {d:.1}\n", .{ sample.input[0], sample.input[1], prediction[0], sample.target[0] });
    }

    std.debug.print("\nTraining completed!\n", .{});
}
