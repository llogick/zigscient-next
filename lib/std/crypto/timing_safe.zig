//! Please see this accepted proposal for the long-term plans regarding
//! constant-time operations in Zig: https://github.com/ziglang/zig/issues/1776

const std = @import("../std.zig");
const builtin = @import("builtin");
const assert = std.debug.assert;
const math = std.math;
const mem = std.mem;
const Endian = std.builtin.Endian;
const Order = std.math.Order;

const Impl = if (std.options.side_channels_mitigations == .none) Plain else Protected;

/// Compares two arrays in constant time (for a given length) and returns whether they are equal.
/// This function was designed to compare short cryptographic secrets (MACs, signatures).
/// For all other applications, use mem.eql() instead.
pub fn eql(comptime T: type, a: T, b: T) bool {
    comptime assert(switch (@typeInfo(T)) {
        .int => true,
        .array => |info| @typeInfo(info.child) == .int,
        .vector => |info| @typeInfo(info.child) == .int,
        .pointer => |info| info.size == .slice and @typeInfo(info.child) == .int,
        else => false,
    });
    if (@inComptime()) return Plain.eql(T, a, b);
    return Impl.eql(T, a, b);
}

/// Compare two integers serialized as arrays of the same size, in constant time.
/// Returns .lt if a<b, .gt if a>b and .eq if a=b
pub fn compare(comptime T: type, a: []const T, b: []const T, endian: Endian) Order {
    comptime assert(@typeInfo(T).int.signedness == .unsigned);
    if (@inComptime()) return Plain.compare(T, a, b, endian);
    return Impl.compare(T, a, b, endian);
}

/// Returns `x` unchanged, but hides its value from the optimizer.
/// `x` must be a `bool` or an integer.
pub fn valueBarrier(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    comptime assert(T == bool or @typeInfo(T) == .int);
    if (@inComptime()) return Plain.valueBarrier(x);
    return Impl.valueBarrier(x);
}

/// Returns a value with all bits set if `c` is true, and 0 otherwise.
pub fn mask(comptime T: type, c: bool) T {
    comptime assert(@typeInfo(T) == .int);
    if (@inComptime()) return Plain.mask(T, c);
    return Impl.mask(T, c);
}

/// Returns a value with all bits set if the top bit of `x` is set, and 0 otherwise.
pub fn signMask(comptime T: type, x: T) T {
    comptime assert(@typeInfo(T) == .int);
    if (@inComptime()) return Plain.signMask(T, x);
    return Impl.signMask(T, x);
}

/// Returns `a` if `c` is true, and `b` otherwise.
/// `T` is an integer or an array of integers.
pub fn select(comptime T: type, c: bool, a: T, b: T) T {
    comptime assert(@typeInfo(T) == .int or (@typeInfo(T) == .array and @typeInfo(std.meta.Elem(T)) == .int));
    if (@inComptime()) return Plain.select(T, c, a, b);
    return Impl.select(T, c, a, b);
}

/// Swaps `a` and `b` if `c` is true.
/// `T` is an integer or an array of integers.
pub fn conditionalSwap(comptime T: type, c: bool, noalias a: *T, noalias b: *T) void {
    comptime assert(@typeInfo(T) == .int or (@typeInfo(T) == .array and @typeInfo(std.meta.Elem(T)) == .int));
    if (@inComptime()) return Plain.conditionalSwap(T, c, a, b);
    Impl.conditionalSwap(T, c, a, b);
}

/// Returns true if `a` is less than `b`.
/// `T` is an unsigned integer type.
pub fn lessThan(comptime T: type, a: T, b: T) bool {
    comptime assert(@typeInfo(T).int.signedness == .unsigned);
    if (@inComptime()) return Plain.lessThan(T, a, b);
    return Impl.lessThan(T, a, b);
}

/// Returns true if `a` is greater than `b`.
/// `T` is an unsigned integer type.
pub fn greaterThan(comptime T: type, a: T, b: T) bool {
    return lessThan(T, b, a);
}

/// Returns true if all the bits of `x` are zero.
/// `T` is an integer, or an array, a vector or a slice of integers.
pub fn isZero(comptime T: type, x: T) bool {
    comptime assert(switch (@typeInfo(T)) {
        .int => true,
        .array => |info| @typeInfo(info.child) == .int,
        .vector => |info| @typeInfo(info.child) == .int,
        .pointer => |info| info.size == .slice and @typeInfo(info.child) == .int,
        else => false,
    });
    if (@inComptime()) return Plain.isZero(T, x);
    return Impl.isZero(T, x);
}

/// Add two integers serialized as arrays of the same size, in constant time.
/// The result is stored into `result`, and `true` is returned if an overflow occurred.
pub fn add(comptime T: type, a: []const T, b: []const T, result: []T, endian: Endian) bool {
    const len = a.len;
    assert(len == b.len and len == result.len);
    var carry: u1 = 0;
    if (endian == .little) {
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const ov1 = @addWithOverflow(a[i], b[i]);
            const ov2 = @addWithOverflow(ov1[0], carry);
            result[i] = ov2[0];
            carry = ov1[1] | ov2[1];
        }
    } else {
        var i: usize = len;
        while (i != 0) {
            i -= 1;
            const ov1 = @addWithOverflow(a[i], b[i]);
            const ov2 = @addWithOverflow(ov1[0], carry);
            result[i] = ov2[0];
            carry = ov1[1] | ov2[1];
        }
    }
    return @as(bool, @bitCast(carry));
}

/// Subtract two integers serialized as arrays of the same size, in constant time.
/// The result is stored into `result`, and `true` is returned if an underflow occurred.
pub fn sub(comptime T: type, a: []const T, b: []const T, result: []T, endian: Endian) bool {
    const len = a.len;
    assert(len == b.len and len == result.len);
    var borrow: u1 = 0;
    if (endian == .little) {
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const ov1 = @subWithOverflow(a[i], b[i]);
            const ov2 = @subWithOverflow(ov1[0], borrow);
            result[i] = ov2[0];
            borrow = ov1[1] | ov2[1];
        }
    } else {
        var i: usize = len;
        while (i != 0) {
            i -= 1;
            const ov1 = @subWithOverflow(a[i], b[i]);
            const ov2 = @subWithOverflow(ov1[0], borrow);
            result[i] = ov2[0];
            borrow = ov1[1] | ov2[1];
        }
    }
    return @as(bool, @bitCast(borrow));
}

fn markSecret(ptr: anytype, comptime action: enum { classify, declassify }) void {
    const t = @typeInfo(@TypeOf(ptr));
    if (t != .pointer) @compileError("Pointer expected - Found: " ++ @typeName(@TypeOf(ptr)));
    const p = t.pointer;
    if (p.attrs.@"allowzero") @compileError("A nullable pointer is always assumed to leak information via side channels");
    const child = @typeInfo(p.child);

    switch (child) {
        .void, .null, .comptime_int, .comptime_float => return,
        .pointer => {
            if (child.pointer.size == .Slice) {
                @compileError("Found pointer to pointer. If the intent was to pass a slice, maybe remove the leading & in the function call");
            }
            @compileError("A pointer value is always assumed leak information via side channels");
        },
        else => {
            const mem8: *const [@sizeOf(@TypeOf(ptr.*))]u8 = @ptrCast(@constCast(ptr));
            if (action == .classify) {
                std.valgrind.memcheck.makeMemUndefined(mem8);
            } else {
                std.valgrind.memcheck.makeMemDefined(mem8);
            }
        },
    }
}

/// Mark a value as sensitive or secret, helping to detect potential side-channel vulnerabilities.
///
/// When Valgrind is enabled, this function allows for the detection of conditional jumps or lookups
/// that depend on secrets or secret-derived data. Violations are reported by Valgrind as operations
/// relying on uninitialized values.
///
/// If Valgrind is disabled, it has no effect.
///
/// Use this function to verify that cryptographic operations perform constant-time arithmetic on sensitive data,
/// ensuring the confidentiality of secrets and preventing information leakage through side channels.
pub fn classify(ptr: anytype) void {
    markSecret(ptr, .classify);
}

/// Mark a value as non-sensitive or public, indicating it's safe from side-channel attacks.
///
/// Signals that a value has been securely processed and is no longer confidential, allowing for
/// relaxed handling without fear of information leakage through conditional jumps or lookups.
pub fn declassify(ptr: anytype) void {
    markSecret(ptr, .declassify);
}

const word_bits = @bitSizeOf(usize);

fn maskFromBit(comptime T: type, bit: usize) T {
    const U = @Int(.unsigned, @bitSizeOf(T));
    const u: U = if (@bitSizeOf(U) < word_bits) @truncate(bit) else bit;
    return @bitCast(0 -% u);
}

const Protected = struct {
    fn valueBarrier(x: anytype) @TypeOf(x) {
        const T = @TypeOf(x);
        if (builtin.cpu.arch.isSpirv()) return x;
        if (builtin.cpu.arch.isWasm()) {
            // Barriers are pretty much useless in WebAssembly, so only enable them with full mitigations.
            if (std.options.side_channels_mitigations != .full or builtin.zig_backend != .stage2_llvm) return x;
        }
        if (T == bool) return Protected.valueBarrier(@intFromBool(x)) != 0;
        const bits = @typeInfo(T).int.bits;
        if (bits == 0) return x;
        const U = @Int(.unsigned, bits);
        const u: U = @bitCast(x);
        const register_bits = if (builtin.cpu.arch.isWasm()) 64 else word_bits;
        if (bits > register_bits) {
            var r: U = 0;
            inline for (0..@divCeil(bits, register_bits)) |i| {
                const shift = i * register_bits;
                const w: @Int(.unsigned, register_bits) = @truncate(u >> shift);
                r |= @as(U, Protected.valueBarrier(w)) << shift;
            }
            return @bitCast(r);
        }
        const R = @Int(.unsigned, math.ceilPowerOfTwoAssert(u16, @max(bits, 8)));
        const w: R = u;
        const hidden = if (builtin.zig_backend == .stage2_c and builtin.abi == .msvc) msvc: {
            const optblocker = struct {
                var zero: R = 0;
                var ptr: *volatile R = &zero;
            };
            break :msvc w ^ optblocker.ptr.*;
        } else asm (""
            : [ret] "=r" (-> R),
            : [x] "0" (w),
        );
        const r: U = @truncate(hidden);
        return @bitCast(r);
    }

    fn mask(comptime T: type, c: bool) T {
        var bit: usize = @intFromBool(c);
        if (builtin.cpu.arch.isWasm()) bit = Protected.valueBarrier(bit); // Required for wasm runtimes compiling the select instruction into a branch.
        return Protected.valueBarrier(maskFromBit(T, bit));
    }

    fn signMask(comptime T: type, x: T) T {
        return Protected.valueBarrier(Plain.signMask(T, x));
    }

    fn select(comptime T: type, c: bool, a: T, b: T) T {
        switch (@typeInfo(T)) {
            .int => return b ^ (Protected.mask(T, c) & (a ^ b)),
            .array => |info| {
                const m = Protected.mask(info.child, c);
                var r = b;
                for (&r, a) |*ri, ai| {
                    ri.* ^= m & (ai ^ ri.*);
                }
                return r;
            },
            else => unreachable,
        }
    }

    fn conditionalSwap(comptime T: type, c: bool, noalias a: *T, noalias b: *T) void {
        switch (@typeInfo(T)) {
            .int => {
                const t = (a.* ^ b.*) & Protected.mask(T, c);
                a.* ^= t;
                b.* ^= t;
            },
            .array => |info| {
                const m = Protected.mask(info.child, c);
                for (a, b) |*x, *y| {
                    const t = (x.* ^ y.*) & m;
                    x.* ^= t;
                    y.* ^= t;
                }
            },
            else => unreachable,
        }
    }

    fn word(comptime T: type, x: T, comptime i: comptime_int) usize {
        const u: @Int(.unsigned, @bitSizeOf(T)) = @bitCast(x);
        if (@bitSizeOf(T) <= word_bits) return u;
        return @truncate(u >> (i * word_bits));
    }

    fn lessThan(comptime T: type, a: T, b: T) bool {
        return ltBit(T, a, b) != 0;
    }

    fn ltBit(comptime T: type, a: T, b: T) usize {
        var borrow: usize = 0;
        inline for (0..@divCeil(@bitSizeOf(T), word_bits)) |i| {
            const x = word(T, a, i);
            const y = word(T, b, i);
            const d = x -% y -% borrow;
            borrow = ((~x & y) | (~(x ^ y) & d)) >> (word_bits - 1);
        }
        return borrow;
    }

    fn isZeroBit(comptime T: type, x: T) usize {
        var acc: usize = 0;
        inline for (0..@divCeil(@bitSizeOf(T), word_bits)) |i| {
            acc |= word(T, x, i);
        }
        return 1 ^ ((acc | (0 -% acc)) >> (word_bits - 1));
    }

    fn isZero(comptime T: type, x: T) bool {
        switch (@typeInfo(T)) {
            .int => return isZeroBit(T, x) != 0,
            .vector => |info| return isZeroBit(info.child, @reduce(.Or, x)) != 0,
            .array, .pointer => {
                const Element = std.meta.Elem(T);
                var acc: Element = 0;
                for (x) |e| {
                    acc |= e;
                }
                return isZeroBit(Element, acc) != 0;
            },
            else => unreachable,
        }
    }

    fn eql(comptime T: type, a: T, b: T) bool {
        switch (@typeInfo(T)) {
            .int, .vector => return Protected.isZero(T, a ^ b),
            .array, .pointer => {
                if (a.len != b.len) return false;
                const Element = std.meta.Elem(T);
                var acc: Element = 0;
                for (a, b) |x, y| {
                    acc |= x ^ y;
                }
                return isZeroBit(Element, acc) != 0;
            },
            else => unreachable,
        }
    }

    fn compare(comptime T: type, a: []const T, b: []const T, endian: Endian) Order {
        assert(a.len == b.len);
        var gt: usize = 0;
        var eq: usize = 1;
        for (0..a.len) |j| {
            const i = if (endian == .little) a.len - 1 - j else j;
            var gt_bit = ltBit(T, b[i], a[i]);
            var eq_bit = isZeroBit(T, a[i] ^ b[i]);
            if (builtin.cpu.arch.isWasm()) {
                // for wasm runtimes turning selects into branches
                gt_bit = Protected.valueBarrier(gt_bit);
                eq_bit = Protected.valueBarrier(eq_bit);
            }
            gt |= gt_bit & eq;
            eq &= eq_bit;
        }
        gt = Protected.valueBarrier(gt);
        eq = Protected.valueBarrier(eq);
        const lt = 1 ^ gt ^ eq;
        const Tag = @TypeOf(@backingInt(Order.eq));
        const tag = (maskFromBit(Tag, gt) & @backingInt(Order.gt)) |
            (maskFromBit(Tag, lt) & @backingInt(Order.lt)) |
            (maskFromBit(Tag, eq) & @backingInt(Order.eq));
        return order: {
            @setRuntimeSafety(false); // the tag is always valid, and with runtime safety, we would get a branch
            break :order @fromBackingInt(tag);
        };
    }
};

const Plain = struct {
    fn valueBarrier(x: anytype) @TypeOf(x) {
        return x;
    }

    fn mask(comptime T: type, c: bool) T {
        return maskFromBit(T, @intFromBool(c));
    }

    fn signMask(comptime T: type, x: T) T {
        const bits = @typeInfo(T).int.bits;
        if (bits == 0) return 0;
        const S = @Int(.signed, bits);
        const s: S = @bitCast(x);
        return @bitCast(s >> (bits - 1));
    }

    fn select(comptime T: type, c: bool, a: T, b: T) T {
        return if (c) a else b;
    }

    fn conditionalSwap(comptime T: type, c: bool, noalias a: *T, noalias b: *T) void {
        if (c) mem.swap(T, a, b);
    }

    fn lessThan(comptime T: type, a: T, b: T) bool {
        return a < b;
    }

    fn isZero(comptime T: type, x: T) bool {
        switch (@typeInfo(T)) {
            .int => return x == 0,
            .array => |info| {
                var acc: info.child = 0;
                for (x) |e| {
                    acc |= e;
                }
                return acc == 0;
            },
            .vector => return @reduce(.Or, x) == 0,
            .pointer => |info| return mem.allEqual(info.child, x, 0),
            else => unreachable,
        }
    }

    fn eql(comptime T: type, a: T, b: T) bool {
        return switch (@typeInfo(T)) {
            .int => a == b,
            .array => |info| mem.eql(info.child, &a, &b),
            .vector => @reduce(.And, a == b),
            .pointer => |info| mem.eql(info.child, a, b),
            else => unreachable,
        };
    }

    fn compare(comptime T: type, a: []const T, b: []const T, endian: Endian) Order {
        assert(a.len == b.len);
        if (endian == .big) return mem.order(T, a, b);
        var i = a.len;
        while (i != 0) {
            i -= 1;
            if (a[i] != b[i]) return if (a[i] < b[i]) .lt else .gt;
        }
        return .eq;
    }
};

test eql {
    const io = std.testing.io;
    const expect = std.testing.expect;
    var a: [100]u8 = undefined;
    var b: [100]u8 = undefined;
    io.random(&a);
    io.random(&b);
    try expect(!eql([100]u8, a, b));
    a = b;
    try expect(eql([100]u8, a, b));
}

test "eql (vectors)" {
    const io = std.testing.io;
    const expect = std.testing.expect;
    var a: [100]u8 = undefined;
    var b: [100]u8 = undefined;
    io.random(&a);
    io.random(&b);
    const v1: @Vector(100, u8) = a;
    const v2: @Vector(100, u8) = b;
    try expect(!eql(@Vector(100, u8), v1, v2));
    const v3: @Vector(100, u8) = a;
    try expect(eql(@Vector(100, u8), v1, v3));
}

test compare {
    const expectEqual = std.testing.expectEqual;
    var a: [32]u8 = @splat(10);
    var b: [32]u8 = @splat(10);
    try expectEqual(compare(u8, &a, &b, .big), .eq);
    try expectEqual(compare(u8, &a, &b, .little), .eq);
    a[31] = 1;
    try expectEqual(compare(u8, &a, &b, .big), .lt);
    try expectEqual(compare(u8, &a, &b, .little), .lt);
    a[0] = 20;
    try expectEqual(compare(u8, &a, &b, .big), .gt);
    try expectEqual(compare(u8, &a, &b, .little), .lt);
}

test "add and sub" {
    const io = std.testing.io;

    const expectEqual = std.testing.expectEqual;
    const expectEqualSlices = std.testing.expectEqualSlices;
    const len = 32;
    var a: [len]u8 = undefined;
    var b: [len]u8 = undefined;
    var c: [len]u8 = undefined;
    const zero: [len]u8 = @splat(0);
    var iterations: usize = 100;
    while (iterations != 0) : (iterations -= 1) {
        io.random(&a);
        io.random(&b);
        const endian = if (iterations % 2 == 0) Endian.big else Endian.little;
        _ = sub(u8, &a, &b, &c, endian); // a-b
        _ = add(u8, &c, &b, &c, endian); // (a-b)+b
        try expectEqualSlices(u8, &c, &a);
        const borrow = sub(u8, &c, &a, &c, endian); // ((a-b)+b)-a
        try expectEqualSlices(u8, &c, &zero);
        try expectEqual(borrow, false);
    }
}

test classify {
    const io = std.testing.io;
    const expect = std.testing.expect;

    var secret: [32]u8 = undefined;
    io.random(&secret);

    // Input of the hash function is marked as secret
    classify(&secret);

    var out: [32]u8 = undefined;
    std.crypto.hash.sha3.TurboShake128(null).hash(&secret, &out, .{});

    // Output of the hash function is derived from secret data, so
    // it will automatically be considered secret as well. But it can be
    // declassified; the input itself will still be considered secret.
    declassify(&out);

    // Comparing public data in non-constant time is acceptable.
    const zeroes: [out.len]u8 = @splat(0);
    try expect(!std.mem.eql(u8, &out, &zeroes));

    // Comparing secret data must be done in constant time. The result
    // is going to be considered as secret as well.
    var res = std.crypto.timing_safe.eql([32]u8, out, secret);

    // If we want to make a conditional jump based on a secret,
    // it has to be declassified.
    declassify(&res);
    try expect(!res);

    // Once a secret has been declassified, a comparison in
    // non-constant time is fine.
    declassify(&secret);
    try expect(!std.mem.eql(u8, &out, &secret));
}

const testing = std.testing;

test {
    inline for (.{ Protected, Plain }) |I| {
        try testing.expectEqual(0xffffffff, I.mask(u32, true));
        try testing.expectEqual(0, I.mask(u32, false));
        try testing.expectEqual(-1, I.mask(i16, true));
        try testing.expectEqual(-1, I.mask(i1, true));
        try testing.expectEqual(1, I.mask(u1, true));
        try testing.expectEqual(0x7f, I.mask(u7, true));
        try testing.expectEqual(0, I.mask(u0, true));

        try testing.expectEqual(-1, I.signMask(i32, -1));
        try testing.expectEqual(0, I.signMask(i16, 5));
        try testing.expectEqual(-1, I.signMask(i1, -1));
        try testing.expectEqual(-1, I.signMask(i7, -64));
        try testing.expectEqual(0xffffff, I.signMask(u24, 0x800000));
        try testing.expectEqual(0, I.signMask(u0, 0));

        const barrier_inputs = .{ true, false, @as(u0, 0), @as(u3, 5), @as(i7, -3), @as(u24, 0xabcdef), @as(u48, (1 << 47) | 3) };
        inline for (barrier_inputs) |v| {
            try testing.expectEqual(v, I.valueBarrier(v));
        }

        try testing.expectEqual(-5, I.select(i16, true, -5, 7));
        try testing.expectEqual(7, I.select(i16, false, -5, 7));
        try testing.expectEqual(-1, I.select(i1, true, -1, 0));
        try testing.expectEqual(0, I.select(u0, true, 0, 0));
        const s = I.select([3:0]u8, true, .{ 1, 2, 3 }, .{ 4, 5, 6 });
        try testing.expectEqualSlices(u8, &.{ 1, 2, 3 }, &s);
        try testing.expectEqual(7, I.select([0:7]u8, true, .{}, .{})[0]);

        var a: [5]u64 = .{ 1, 2, 3, 4, 5 };
        var b: [5]u64 = .{ 6, 7, 8, 9, 10 };
        I.conditionalSwap([5]u64, false, &a, &b);
        try testing.expectEqual(1, a[0]);
        I.conditionalSwap([5]u64, true, &a, &b);
        try testing.expectEqual([5]u64{ 6, 7, 8, 9, 10 }, a);
        try testing.expectEqual([5]u64{ 1, 2, 3, 4, 5 }, b);
        var x: i32 = -4;
        var y: i32 = 9;
        I.conditionalSwap(i32, true, &x, &y);
        try testing.expectEqual(9, x);
        try testing.expectEqual(-4, y);
        var sx: [2:0]u8 = .{ 1, 2 };
        var sy: [2:0]u8 = .{ 3, 4 };
        I.conditionalSwap([2:0]u8, true, &sx, &sy);
        try testing.expectEqualSlices(u8, &.{ 3, 4 }, &sx);
        var e1: [0]u8 = .{};
        var e2: [0]u8 = .{};
        I.conditionalSwap([0]u8, true, &e1, &e2);
        var z0: u0 = 0;
        var z1: u0 = 0;
        I.conditionalSwap(u0, true, &z0, &z1);

        try testing.expect(I.eql([]const u8, "abc", "abc"));
        try testing.expect(!I.eql([]const u8, "abc", "abd"));
        try testing.expect(!I.eql([]const u8, "abc", "ab"));
        try testing.expect(I.eql([]const u8, &.{}, &.{}));
        try testing.expect(I.eql([3]i16, .{ 1, -1, 0 }, .{ 1, -1, 0 }));
        try testing.expect(!I.eql([3]i16, .{ 1, -1, 0 }, .{ 1, -1, 1 }));
        try testing.expect(I.eql([0]u8, .{}, .{}));
        try testing.expect(I.eql([2]u0, .{ 0, 0 }, .{ 0, 0 }));
        try testing.expect(I.eql(u0, 0, 0));
        try testing.expect(I.eql(@Vector(4, i8), .{ 1, -2, 3, 4 }, .{ 1, -2, 3, 4 }));
        try testing.expect(!I.eql(@Vector(4, i8), .{ 1, -2, 3, 4 }, .{ 1, 2, 3, 4 }));

        try testing.expect(I.isZero([3]u64, .{ 0, 0, 0 }));
        try testing.expect(!I.isZero([3]u64, .{ 0, 1, 0 }));
        try testing.expect(I.isZero(@Vector(3, u16), .{ 0, 0, 0 }));
        try testing.expect(!I.isZero(@Vector(3, u16), .{ 0, 0x8000, 0 }));
        try testing.expect(I.isZero([]const u16, &.{ 0, 0 }));
        try testing.expect(!I.isZero([]const u16, &.{ 0, 3 }));
        try testing.expect(I.isZero([]const u8, &.{}));
        try testing.expect(I.isZero([0]u8, .{}));
        try testing.expect(I.isZero(u0, 0));

        try testing.expect(!I.lessThan(u0, 0, 0));
        try testing.expectEqual(.eq, I.compare(u8, &.{}, &.{}, .big));
        try testing.expectEqual(.eq, I.compare(u0, &.{ 0, 0 }, &.{ 0, 0 }, .little));

        const top: u512 = 1 << 511;
        try testing.expect(I.isZero(u512, 0));
        try testing.expect(!I.isZero(u512, top));
        try testing.expect(I.lessThan(u512, top, top | 1));
        try testing.expect(!I.lessThan(u512, top | 1, top));
    }
}
