const std = @import("std");
const assert = std.debug.assert;
const expectEqual = std.testing.expectEqual;

const BitpackStruct = bitpack struct(u8) {
    lo: u4,
    hi: u4,
};

test "convert to and from backing integer" {
    const original: BitpackStruct = .{ .lo = 0b1100, .hi = 0b0101 };

    const backing_int = @backingInt(original);
    comptime assert(@TypeOf(backing_int) == u8);
    try expectEqual(0b0101_1100, backing_int);

    const reconstructed: BitpackStruct = @fromBackingInt(backing_int);
    try expectEqual(original, reconstructed);
}

// test
