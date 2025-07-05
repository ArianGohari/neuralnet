const std = @import("std");

/// Applies the ReLU (Rectified Linear Unit) activation function element-wise to a matrix.
/// ReLU(x) = max(0, x)
pub fn relu(matrix: []const []const f64, allocator: std.mem.Allocator) ![][]f64 {

    // Ensure matrix is not empty
    std.debug.assert(matrix.len != 0);

    const rows = matrix.len;
    const cols = matrix[0].len;

    // Ensure all rows have consistent column counts
    for (matrix) |row| {
        std.debug.assert(row.len == cols);
    }

    // Allocate result matrix
    var result = try allocator.alloc([]f64, rows);
    for (result, 0..) |_, i| {
        result[i] = try allocator.alloc(f64, cols);
    }

    // Apply ReLU element-wise
    for (0..rows) |i| {
        for (0..cols) |j| {
            result[i][j] = @max(0.0, matrix[i][j]);
        }
    }

    return result;
}

/// Applies the sigmoid activation function element-wise to a matrix.
/// Sigmoid(x) = 1 / (1 + e^(-x))
pub fn sigmoid(matrix: []const []const f64, allocator: std.mem.Allocator) ![][]f64 {

    // Ensure matrix is not empty
    std.debug.assert(matrix.len != 0);

    const rows = matrix.len;
    const cols = matrix[0].len;

    // Ensure all rows have consistent column counts
    for (matrix) |row| {
        std.debug.assert(row.len == cols);
    }

    // Allocate result matrix
    var result = try allocator.alloc([]f64, rows);
    for (result, 0..) |_, i| {
        result[i] = try allocator.alloc(f64, cols);
    }

    // Apply sigmoid element-wise
    for (0..rows) |i| {
        for (0..cols) |j| {
            result[i][j] = 1.0 / (1.0 + @exp(-matrix[i][j]));
        }
    }

    return result;
}

/// Applies the tanh (hyperbolic tangent) activation function element-wise to a matrix.
/// Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
pub fn tanh(matrix: []const []const f64, allocator: std.mem.Allocator) ![][]f64 {

    // Ensure matrix is not empty
    std.debug.assert(matrix.len != 0);

    const rows = matrix.len;
    const cols = matrix[0].len;

    // Ensure all rows have consistent column counts
    for (matrix) |row| {
        std.debug.assert(row.len == cols);
    }

    // Allocate result matrix
    var result = try allocator.alloc([]f64, rows);
    for (result, 0..) |_, i| {
        result[i] = try allocator.alloc(f64, cols);
    }

    // Apply tanh element-wise
    for (0..rows) |i| {
        for (0..cols) |j| {
            const exp_x = @exp(matrix[i][j]);
            const exp_neg_x = @exp(-matrix[i][j]);
            result[i][j] = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
        }
    }

    return result;
}

/// Applies the softmax activation function row-wise to a matrix.
/// Softmax(x_i) = e^x_i / sum(e^x_j) for each row
/// Each row sums to 1, making it suitable for probability distributions
pub fn softmax(matrix: []const []const f64, allocator: std.mem.Allocator) ![][]f64 {

    // Ensure matrix is not empty
    std.debug.assert(matrix.len != 0);

    const rows = matrix.len;
    const cols = matrix[0].len;

    // Ensure all rows have consistent column counts
    for (matrix) |row| {
        std.debug.assert(row.len == cols);
    }

    // Allocate result matrix
    var result = try allocator.alloc([]f64, rows);
    for (result, 0..) |_, i| {
        result[i] = try allocator.alloc(f64, cols);
    }

    // Apply softmax row-wise
    for (0..rows) |i| {
        // Find max value in row for numerical stability
        var max_val = matrix[i][0];
        for (1..cols) |j| {
            max_val = @max(max_val, matrix[i][j]);
        }

        // Calculate exponentials and sum
        var sum: f64 = 0.0;
        for (0..cols) |j| {
            result[i][j] = @exp(matrix[i][j] - max_val);
            sum += result[i][j];
        }

        // Normalize by sum
        for (0..cols) |j| {
            result[i][j] /= sum;
        }
    }

    return result;
}

/// Computes the derivative of ReLU activation function element-wise.
/// ReLU'(x) = 1 if x > 0, else 0
pub fn relu_derivative(matrix: []const []const f64, allocator: std.mem.Allocator) ![][]f64 {

    // Ensure matrix is not empty
    std.debug.assert(matrix.len != 0);

    const rows = matrix.len;
    const cols = matrix[0].len;

    // Ensure all rows have consistent column counts
    for (matrix) |row| {
        std.debug.assert(row.len == cols);
    }

    // Allocate result matrix
    var result = try allocator.alloc([]f64, rows);
    for (result, 0..) |_, i| {
        result[i] = try allocator.alloc(f64, cols);
    }

    // Apply ReLU derivative element-wise
    for (0..rows) |i| {
        for (0..cols) |j| {
            result[i][j] = if (matrix[i][j] > 0.0) 1.0 else 0.0;
        }
    }

    return result;
}

/// Computes the derivative of sigmoid activation function element-wise.
/// Sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
pub fn sigmoid_derivative(matrix: []const []const f64, allocator: std.mem.Allocator) ![][]f64 {

    // Ensure matrix is not empty
    std.debug.assert(matrix.len != 0);

    const rows = matrix.len;
    const cols = matrix[0].len;

    // Ensure all rows have consistent column counts
    for (matrix) |row| {
        std.debug.assert(row.len == cols);
    }

    // Allocate result matrix
    var result = try allocator.alloc([]f64, rows);
    for (result, 0..) |_, i| {
        result[i] = try allocator.alloc(f64, cols);
    }

    // Apply sigmoid derivative element-wise
    for (0..rows) |i| {
        for (0..cols) |j| {
            const sigmoid_val = 1.0 / (1.0 + @exp(-matrix[i][j]));
            result[i][j] = sigmoid_val * (1.0 - sigmoid_val);
        }
    }

    return result;
}

/// Computes the derivative of tanh activation function element-wise.
/// Tanh'(x) = 1 - tanh(x)^2
pub fn tanh_derivative(matrix: []const []const f64, allocator: std.mem.Allocator) ![][]f64 {

    // Ensure matrix is not empty
    std.debug.assert(matrix.len != 0);

    const rows = matrix.len;
    const cols = matrix[0].len;

    // Ensure all rows have consistent column counts
    for (matrix) |row| {
        std.debug.assert(row.len == cols);
    }

    // Allocate result matrix
    var result = try allocator.alloc([]f64, rows);
    for (result, 0..) |_, i| {
        result[i] = try allocator.alloc(f64, cols);
    }

    // Apply tanh derivative element-wise
    for (0..rows) |i| {
        for (0..cols) |j| {
            const exp_x = @exp(matrix[i][j]);
            const exp_neg_x = @exp(-matrix[i][j]);
            const tanh_val = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
            result[i][j] = 1.0 - tanh_val * tanh_val;
        }
    }

    return result;
}

test "ReLU activation works" {
    const allocator = std.testing.allocator;
    const matrix_rows = [_][]const f64{
        &[_]f64{ -2.0, 0.0, 3.0 },
        &[_]f64{ -1.5, 2.5, -0.5 },
    };
    const matrix = &matrix_rows;
    const result_matrix = try relu(matrix, allocator);
    defer {
        for (result_matrix) |row| {
            allocator.free(row);
        }
        allocator.free(result_matrix);
    }
    try std.testing.expectEqual(@as(f64, 0.0), result_matrix[0][0]);
    try std.testing.expectEqual(@as(f64, 0.0), result_matrix[0][1]);
    try std.testing.expectEqual(@as(f64, 3.0), result_matrix[0][2]);
    try std.testing.expectEqual(@as(f64, 0.0), result_matrix[1][0]);
    try std.testing.expectEqual(@as(f64, 2.5), result_matrix[1][1]);
    try std.testing.expectEqual(@as(f64, 0.0), result_matrix[1][2]);
}

test "sigmoid activation works" {
    const allocator = std.testing.allocator;
    const matrix_rows = [_][]const f64{
        &[_]f64{ 0.0, 1.0, -1.0 },
        &[_]f64{ 2.0, -2.0, 0.5 },
    };
    const matrix = &matrix_rows;
    const result_matrix = try sigmoid(matrix, allocator);
    defer {
        for (result_matrix) |row| {
            allocator.free(row);
        }
        allocator.free(result_matrix);
    }
    try std.testing.expectApproxEqRel(@as(f64, 0.5), result_matrix[0][0], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 0.7310585786300049), result_matrix[0][1], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 0.2689414213699951), result_matrix[0][2], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 0.8807970779778823), result_matrix[1][0], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 0.11920292202211755), result_matrix[1][1], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 0.6224593312018546), result_matrix[1][2], 1e-10);
}

test "tanh activation works" {
    const allocator = std.testing.allocator;
    const matrix_rows = [_][]const f64{
        &[_]f64{ 0.0, 1.0, -1.0 },
        &[_]f64{ 2.0, -2.0, 0.5 },
    };
    const matrix = &matrix_rows;
    const result_matrix = try tanh(matrix, allocator);
    defer {
        for (result_matrix) |row| {
            allocator.free(row);
        }
        allocator.free(result_matrix);
    }
    try std.testing.expectApproxEqRel(@as(f64, 0.0), result_matrix[0][0], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 0.7615941559557649), result_matrix[0][1], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, -0.7615941559557649), result_matrix[0][2], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 0.9640275800758169), result_matrix[1][0], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, -0.9640275800758169), result_matrix[1][1], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 0.46211715726000974), result_matrix[1][2], 1e-10);
}

test "softmax activation works" {
    const allocator = std.testing.allocator;
    const matrix_rows = [_][]const f64{
        &[_]f64{ 1.0, 2.0, 3.0 },
        &[_]f64{ 0.0, 0.0, 0.0 },
    };
    const matrix = &matrix_rows;
    const result_matrix = try softmax(matrix, allocator);
    defer {
        for (result_matrix) |row| {
            allocator.free(row);
        }
        allocator.free(result_matrix);
    }
    try std.testing.expectApproxEqRel(@as(f64, 0.09003057317038046), result_matrix[0][0], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 0.24472847105479764), result_matrix[0][1], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 0.6652409557748219), result_matrix[0][2], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 1.0 / 3.0), result_matrix[1][0], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 1.0 / 3.0), result_matrix[1][1], 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 1.0 / 3.0), result_matrix[1][2], 1e-10);
    var row_sum_0: f64 = 0.0;
    var row_sum_1: f64 = 0.0;
    for (0..3) |j| {
        row_sum_0 += result_matrix[0][j];
        row_sum_1 += result_matrix[1][j];
    }
    try std.testing.expectApproxEqRel(@as(f64, 1.0), row_sum_0, 1e-10);
    try std.testing.expectApproxEqRel(@as(f64, 1.0), row_sum_1, 1e-10);
}
