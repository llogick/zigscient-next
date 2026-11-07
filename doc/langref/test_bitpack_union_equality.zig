const std = @import("std");
const expectEqual = std.testing.expectEqual;

test "bitpack union equality" {
    const U = bitpack union {
        a: u4,
        b: i4,
    };
    const x: U = .{ .a = 3 };
    const y: U = .{ .b = 3 };
    try expectEqual(x, y);
}

// test
