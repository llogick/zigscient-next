const builtin = @import("builtin");

const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;

test "@splat array" {
    if (builtin.cpu.arch.isSpirv() and builtin.zig_backend == .stage2_llvm) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_arm) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_sparc64) return error.SkipZigTest; // TODO

    const Foo = struct { x: u8 };
    const S = struct {
        fn testInt(x: u32) !void {
            const arr: [10]u32 = @splat(x);
            for (arr) |elem| {
                try expect(x == elem);
            }
        }

        fn testStruct(x: Foo) !void {
            const arr: [10]Foo = @splat(x);
            for (arr) |elem| {
                try expect(x.x == elem.x);
            }
        }
    };

    try S.testInt(123);
    try comptime S.testInt(123);

    try S.testStruct(.{ .x = 10 });
    try comptime S.testStruct(.{ .x = 10 });
}

test "@splat array with sentinel" {
    if (builtin.cpu.arch.isSpirv()) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_arm) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_sparc64) return error.SkipZigTest; // TODO

    const S = struct {
        fn doTheTest(comptime T: type, x: T, comptime s: T) !void {
            const arr: [10:s]T = @splat(x);
            for (arr) |elem| {
                try expect(x == elem);
            }
            const ptr: [*]const T = &arr;
            try expect(s == ptr[10]); // sentinel correct
        }
    };

    try S.doTheTest(u32, 100, 42);
    try comptime S.doTheTest(u32, 100, 42);

    try S.doTheTest(?*anyopaque, @ptrFromInt(0x1000), null);
    try comptime S.doTheTest(?*anyopaque, @ptrFromInt(0x1000), null);
}

test "@splat zero-length array" {
    if (builtin.cpu.arch.isSpirv()) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_arm) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_sparc64) return error.SkipZigTest; // TODO

    const S = struct {
        fn doTheTest(comptime T: type, comptime s: T) !void {
            var runtime_undef: T = undefined;
            runtime_undef = undefined;
            // The array should be comptime-known despite the `@splat` operand being runtime-known.
            const arr: [0:s]T = @splat(runtime_undef);
            const ptr: [*]const T = &arr;
            comptime assert(ptr[0] == s);
        }
    };

    try S.doTheTest(u32, 42);
    try comptime S.doTheTest(u32, 42);

    try S.doTheTest(?*anyopaque, null);
    try comptime S.doTheTest(?*anyopaque, null);
}

test "splat with an error union or optional result type" {
    if (builtin.cpu.arch.isSpirv() and builtin.zig_backend == .stage2_llvm) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_aarch64) return error.SkipZigTest;

    const S = struct {
        fn doTest(T: type) !?T {
            return @splat(1);
        }
    };

    _ = try S.doTest(@Vector(4, u32));
    _ = try S.doTest([4]u32);
}

test "read/write through global variable array of struct fields initialized via splat" {
    if (builtin.cpu.arch.isSpirv()) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_arm) return error.SkipZigTest; // TODO
    if (builtin.zig_backend == .stage2_sparc64) return error.SkipZigTest; // TODO

    const S = struct {
        fn doTheTest() !void {
            try expect(storage[0].term == 1);
            storage[0] = MyStruct{ .term = 123 };
            try expect(storage[0].term == 123);
        }

        pub const MyStruct = struct {
            term: usize,
        };

        var storage: [1]MyStruct = @splat(.{ .term = 1 });
    };
    try S.doTheTest();
}

test "vector @splat" {
    if (builtin.cpu.arch.isSpirv() and builtin.zig_backend == .stage2_llvm) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_aarch64) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_arm) return error.SkipZigTest; // TODO
    if (builtin.zig_backend == .stage2_sparc64) return error.SkipZigTest; // TODO
    if (builtin.zig_backend == .stage2_riscv64) return error.SkipZigTest;
    const S = struct {
        fn testForT(comptime N: comptime_int, v: anytype) !void {
            const T = @TypeOf(v);
            var vec: @Vector(N, T) = @splat(v);
            _ = &vec;
            const as_array = @as([N]T, vec);
            for (as_array) |elem| try expect(v == elem);
        }
        fn doTheTest() !void {
            // Splats with multiple-of-8 bit types that fill a 128bit vector.
            try testForT(16, @as(u8, 0xEE));
            try testForT(8, @as(u16, 0xBEEF));
            try testForT(4, @as(u32, 0xDEADBEEF));
            try testForT(2, @as(u64, 0xCAFEF00DDEADBEEF));

            try testForT(8, @as(f16, 3.1415));
            try testForT(4, @as(f32, 3.1415));
            try testForT(2, @as(f64, 3.1415));

            // Same but fill more than 128 bits.
            try testForT(16 * 2, @as(u8, 0xEE));
            try testForT(8 * 2, @as(u16, 0xBEEF));
            try testForT(4 * 2, @as(u32, 0xDEADBEEF));
            try testForT(2 * 2, @as(u64, 0xCAFEF00DDEADBEEF));

            try testForT(8 * 2, @as(f16, 3.1415));
            try testForT(4 * 2, @as(f32, 3.1415));
            try testForT(2 * 2, @as(f64, 3.1415));

            // Splats into one-element vectors.
            try testForT(1, @as(u8, 0xEE));
            try testForT(1, @as(u16, 0xBEEF));
            try testForT(1, @as(u32, 0xDEADBEEF));
            try testForT(1, @as(u64, 0xCAFEF00DDEADBEEF));

            try testForT(1, @as(f16, 3.1415));
            try testForT(1, @as(f32, 3.1415));
            try testForT(1, @as(f64, 3.1415));
        }
    };
    try S.doTheTest();
    try comptime S.doTheTest();
}

test "scalar @splat" {
    const S = struct {
        fn testForTypes(R: type, result: R, Op: type, op: Op) !void {
            try expect(@as(R, result) == @as(Op, @splat(op)));
        }
        fn doTheTest() !void {
            try testForTypes(comptime_float, 3.1415, comptime_float, 3.1415);
            try testForTypes(f32, 3.1415, comptime_float, 3.1415);
            try testForTypes(f32, 3.1415, f32, 3.1415);
            try testForTypes(u8, 31, u8, 31);
            try testForTypes(u8, 31, u16, 31);
            try testForTypes(i16, 31, i16, 31);
            try testForTypes(i16, 31, i8, 31);
        }
    };
    try S.doTheTest();
    try comptime S.doTheTest();
}
