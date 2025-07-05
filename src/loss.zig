const std = @import("std");

/// Computes the Mean Squared Error (MSE) loss between predictions and targets.
/// MSE = (1/n) * sum((y_pred - y_true)^2)
pub fn mse(predictions: []const f64, targets: []const f64) f64 {
    // Ensure predictions and targets have the same length
    std.debug.assert(predictions.len == targets.len);
    std.debug.assert(predictions.len > 0);

    var sum_squared_error: f64 = 0.0;
    for (predictions, targets) |pred, target| {
        const diff = pred - target;
        sum_squared_error += diff * diff;
    }

    return sum_squared_error / @as(f64, @floatFromInt(predictions.len));
}

/// Computes the derivative of MSE loss with respect to predictions.
/// d/dy_pred MSE = (2/n) * (y_pred - y_true)
pub fn mse_derivative(predictions: []const f64, targets: []const f64, allocator: std.mem.Allocator) ![]f64 {
    // Ensure predictions and targets have the same length
    std.debug.assert(predictions.len == targets.len);
    std.debug.assert(predictions.len > 0);

    const n = @as(f64, @floatFromInt(predictions.len));
    var derivative = try allocator.alloc(f64, predictions.len);

    for (predictions, targets, 0..) |pred, target, i| {
        derivative[i] = (2.0 / n) * (pred - target);
    }

    return derivative;
}

test "MSE loss calculation works" {
    const allocator = std.testing.allocator;
    const predictions_perfect = [_]f64{ 1.0, 2.0, 3.0 };
    const targets_perfect = [_]f64{ 1.0, 2.0, 3.0 };
    const mse_perfect = mse(&predictions_perfect, &targets_perfect);
    try std.testing.expectApproxEqRel(@as(f64, 0.0), mse_perfect, 1e-10);
    const predictions = [_]f64{ 2.0, 4.0, 6.0 };
    const targets = [_]f64{ 1.0, 3.0, 5.0 };
    const mse_result = mse(&predictions, &targets);
    try std.testing.expectApproxEqRel(@as(f64, 1.0), mse_result, 1e-10);
    const predictions2 = [_]f64{ 0.0, 2.0 };
    const targets2 = [_]f64{ 1.0, 0.0 };
    const mse_result2 = mse(&predictions2, &targets2);
    try std.testing.expectApproxEqRel(@as(f64, 2.5), mse_result2, 1e-10);
    _ = allocator;
}

test "MSE derivative calculation works" {
    const allocator = std.testing.allocator;
    const predictions_perfect = [_]f64{ 1.0, 2.0, 3.0 };
    const targets_perfect = [_]f64{ 1.0, 2.0, 3.0 };
    const derivative_perfect = try mse_derivative(&predictions_perfect, &targets_perfect, allocator);
    defer allocator.free(derivative_perfect);
    for (derivative_perfect) |d| {
        try std.testing.expectApproxEqRel(@as(f64, 0.0), d, 1e-10);
    }
    const predictions = [_]f64{ 2.0, 4.0, 6.0 };
    const targets = [_]f64{ 1.0, 3.0, 5.0 };
    const derivative = try mse_derivative(&predictions, &targets, allocator);
    defer allocator.free(derivative);
    const expected_value = 2.0 / 3.0;
    for (derivative) |d| {
        try std.testing.expectApproxEqRel(expected_value, d, 1e-10);
    }
    const predictions2 = [_]f64{ 0.0, 2.0 };
    const targets2 = [_]f64{ 1.0, 0.0 };
    const derivative2 = try mse_derivative(&predictions2, &targets2, allocator);
    defer allocator.free(derivative2);
    try std.testing.expectApproxEqRel(@as(f64, -1.0), derivative2[0], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 2.0), derivative2[1], 1e-10);
}

test "MSE and derivative consistency" {
    const allocator = std.testing.allocator;
    const predictions = [_]f64{ 1.5, 2.5, 3.5 };
    const targets = [_]f64{ 1.0, 2.0, 3.0 };
    const original_loss = mse(&predictions, &targets);
    const derivative = try mse_derivative(&predictions, &targets, allocator);
    defer allocator.free(derivative);
    const step_size = 0.01;
    var new_predictions = try allocator.alloc(f64, predictions.len);
    defer allocator.free(new_predictions);
    for (predictions, derivative, 0..) |pred, deriv, i| {
        new_predictions[i] = pred - step_size * deriv;
    }
    const new_loss = mse(new_predictions, &targets);
    try std.testing.expect(new_loss < original_loss);
}

