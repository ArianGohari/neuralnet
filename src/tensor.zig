const std = @import("std");

/// Computes the dot product of two vectors.
/// Asserts that the lengths of the input vectors are equal.
pub fn dot(a: []const f64, b: []const f64) f64 {
    std.debug.assert(a.len == b.len);
    var sum: f64 = 0.0;
    for (a, b) |a_i, b_i| {
        sum += a_i * b_i;
    }
    return sum;
}

/// Computes the matrix-vector multiplication of a matrix and a vector.
/// Asserts that the number of columns in the matrix equals the length of the vector.
pub fn mat_vec_mul(matrix: []const []const f64, vector: []const f64, allocator: std.mem.Allocator) ![]f64 {

    // Ensure matrix and vector are not empty
    std.debug.assert(matrix.len != 0);
    std.debug.assert(vector.len != 0);

    const num_cols = matrix[0].len;

    // Ensure all rows have the same number of columns
    for (matrix) |row| {
        std.debug.assert(row.len == num_cols);
    }

    // Ensure matrix n of columns equals vector legnth
    std.debug.assert(num_cols == vector.len);

    // Calculate result
    var result_vector = try allocator.alloc(f64, matrix.len);
    for (matrix, 0..) |row, i| {
        result_vector[i] = dot(row, vector);
    }
    return result_vector;
}

/// Computes the matrix-matrix multiplication of two matrices.
/// Asserts that the number of columns in the first matrix equals the number of rows in the second matrix.
pub fn mat_mat_mul(a: []const []const f64, b: []const []const f64, allocator: std.mem.Allocator) ![][]f64 {

    // Ensure matrices are not empty
    std.debug.assert(a.len != 0);
    std.debug.assert(b.len != 0);

    const a_rows = a.len;
    const a_cols = a[0].len;
    const b_rows = b.len;
    const b_cols = b[0].len;

    // Ensure all rows have consistent column counts
    for (a) |row| {
        std.debug.assert(row.len == a_cols);
    }
    for (b) |row| {
        std.debug.assert(row.len == b_cols);
    }

    // Ensure matrices can be multiplied
    std.debug.assert(a_cols == b_rows);

    // Allocate result matrix
    var result = try allocator.alloc([]f64, a_rows);
    for (result, 0..) |_, i| {
        result[i] = try allocator.alloc(f64, b_cols);
    }

    // Perform matrix multiplication
    for (0..a_rows) |i| {
        for (0..b_cols) |j| {
            result[i][j] = 0.0;
            for (0..a_cols) |k| {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

/// Computes the element-wise addition of two matrices.
/// Asserts that the matrices have the same dimensions.
pub fn mat_add(a: []const []const f64, b: []const []const f64, allocator: std.mem.Allocator) ![][]f64 {

    // Ensure matrices are not empty
    std.debug.assert(a.len != 0);
    std.debug.assert(b.len != 0);

    const rows = a.len;
    const cols = a[0].len;

    // Ensure matrices have the same dimensions
    std.debug.assert(a.len == b.len);
    std.debug.assert(a[0].len == b[0].len);

    // Ensure all rows have consistent column counts
    for (a) |row| {
        std.debug.assert(row.len == cols);
    }
    for (b) |row| {
        std.debug.assert(row.len == cols);
    }

    // Allocate result matrix
    var result = try allocator.alloc([]f64, rows);
    for (result, 0..) |_, i| {
        result[i] = try allocator.alloc(f64, cols);
    }

    // Perform element-wise addition
    for (0..rows) |i| {
        for (0..cols) |j| {
            result[i][j] = a[i][j] + b[i][j];
        }
    }

    return result;
}

/// Computes the transpose of a matrix.
/// Returns a new matrix where rows and columns are swapped.
pub fn transpose(matrix: []const []const f64, allocator: std.mem.Allocator) ![][]f64 {

    // Ensure matrix is not empty
    std.debug.assert(matrix.len != 0);

    const rows = matrix.len;
    const cols = matrix[0].len;

    // Ensure all rows have consistent column counts
    for (matrix) |row| {
        std.debug.assert(row.len == cols);
    }

    // Allocate result matrix (transposed dimensions)
    var result = try allocator.alloc([]f64, cols);
    for (result, 0..) |_, i| {
        result[i] = try allocator.alloc(f64, rows);
    }

    // Perform transpose
    for (0..rows) |i| {
        for (0..cols) |j| {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

/// Initializes a matrix with random values in the range [min, max).
/// Uses uniform distribution for random number generation.
pub fn random_init(rows: usize, cols: usize, min: f64, max: f64, allocator: std.mem.Allocator) ![][]f64 {
    
    // Ensure valid dimensions
    std.debug.assert(rows > 0);
    std.debug.assert(cols > 0);
    std.debug.assert(min < max);
    
    // Allocate result matrix
    var result = try allocator.alloc([]f64, rows);
    for (result, 0..) |_, i| {
        result[i] = try allocator.alloc(f64, cols);
    }
    
    // Initialize random number generator
    var prng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        std.posix.getrandom(std.mem.asBytes(&seed)) catch |err| switch (err) {
            error.SystemResources => unreachable,
            else => unreachable,
        };
        break :blk seed;
    });
    const random = prng.random();
    
    // Fill matrix with random values
    for (0..rows) |i| {
        for (0..cols) |j| {
            result[i][j] = random.float(f64) * (max - min) + min;
        }
    }
    
    return result;
}

test "dot product works" {
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    const b = [_]f64{ 4.0, 5.0, 6.0 };
    const result = dot(&a, &b);
    try std.testing.expect(result == 32.0);
}

test "matrix-vector multiplication works" {
    const allocator = std.testing.allocator;
    const matrix_rows = [_][]const f64{
        &[_]f64{ 1.0, 2.0, 3.0 },
        &[_]f64{ 4.0, 5.0, 6.0 },
    };
    const matrix = &matrix_rows;
    const vector_data = [_]f64{ 7.0, 8.0, 9.0 };
    const vector = vector_data[0..];
    const result_vector = try mat_vec_mul(matrix, vector, allocator);
    defer allocator.free(result_vector);
    try std.testing.expectEqual(@as(f64, 50.0), result_vector[0]);
    try std.testing.expectEqual(@as(f64, 122.0), result_vector[1]);
}

test "matrix-matrix multiplication works" {
    const allocator = std.testing.allocator;
    const matrix_a_rows = [_][]const f64{
        &[_]f64{ 1.0, 2.0, 3.0 },
        &[_]f64{ 4.0, 5.0, 6.0 },
    };
    const matrix_a = &matrix_a_rows;
    const matrix_b_rows = [_][]const f64{
        &[_]f64{ 7.0, 8.0 },
        &[_]f64{ 9.0, 10.0 },
        &[_]f64{ 11.0, 12.0 },
    };
    const matrix_b = &matrix_b_rows;
    const result_matrix = try mat_mat_mul(matrix_a, matrix_b, allocator);
    defer {
        for (result_matrix) |row| {
            allocator.free(row);
        }
        allocator.free(result_matrix);
    }
    try std.testing.expectEqual(@as(f64, 58.0), result_matrix[0][0]);
    try std.testing.expectEqual(@as(f64, 64.0), result_matrix[0][1]);
    try std.testing.expectEqual(@as(f64, 139.0), result_matrix[1][0]);
    try std.testing.expectEqual(@as(f64, 154.0), result_matrix[1][1]);
}

test "element-wise addition works" {
    const allocator = std.testing.allocator;
    const matrix_a_rows = [_][]const f64{
        &[_]f64{ 1.0, 2.0, 3.0 },
        &[_]f64{ 4.0, 5.0, 6.0 },
    };
    const matrix_a = &matrix_a_rows;
    const matrix_b_rows = [_][]const f64{
        &[_]f64{ 7.0, 8.0, 9.0 },
        &[_]f64{ 10.0, 11.0, 12.0 },
    };
    const matrix_b = &matrix_b_rows;
    const result_matrix = try mat_add(matrix_a, matrix_b, allocator);
    defer {
        for (result_matrix) |row| {
            allocator.free(row);
        }
        allocator.free(result_matrix);
    }
    try std.testing.expectEqual(@as(f64, 8.0), result_matrix[0][0]);
    try std.testing.expectEqual(@as(f64, 10.0), result_matrix[0][1]);
    try std.testing.expectEqual(@as(f64, 12.0), result_matrix[0][2]);
    try std.testing.expectEqual(@as(f64, 14.0), result_matrix[1][0]);
    try std.testing.expectEqual(@as(f64, 16.0), result_matrix[1][1]);
    try std.testing.expectEqual(@as(f64, 18.0), result_matrix[1][2]);
}

test "transpose works" {
    const allocator = std.testing.allocator;
    const matrix_rows = [_][]const f64{
        &[_]f64{ 1.0, 2.0, 3.0 },
        &[_]f64{ 4.0, 5.0, 6.0 },
    };
    const matrix = &matrix_rows;
    const result_matrix = try transpose(matrix, allocator);
    defer {
        for (result_matrix) |row| {
            allocator.free(row);
        }
        allocator.free(result_matrix);
    }
    try std.testing.expectEqual(@as(f64, 1.0), result_matrix[0][0]);
    try std.testing.expectEqual(@as(f64, 4.0), result_matrix[0][1]);
    try std.testing.expectEqual(@as(f64, 2.0), result_matrix[1][0]);
    try std.testing.expectEqual(@as(f64, 5.0), result_matrix[1][1]);
    try std.testing.expectEqual(@as(f64, 3.0), result_matrix[2][0]);
    try std.testing.expectEqual(@as(f64, 6.0), result_matrix[2][1]);
}

test "random initialization works" {
    const allocator = std.testing.allocator;
    const result_matrix = try random_init(3, 2, -1.0, 1.0, allocator);
    defer {
        for (result_matrix) |row| {
            allocator.free(row);
        }
        allocator.free(result_matrix);
    }
    try std.testing.expectEqual(@as(usize, 3), result_matrix.len);
    try std.testing.expectEqual(@as(usize, 2), result_matrix[0].len);
    try std.testing.expectEqual(@as(usize, 2), result_matrix[1].len);
    try std.testing.expectEqual(@as(usize, 2), result_matrix[2].len);
    for (result_matrix) |row| {
        for (row) |value| {
            try std.testing.expect(value >= -1.0);
            try std.testing.expect(value < 1.0);
        }
    }
}
