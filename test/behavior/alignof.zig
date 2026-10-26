const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;
const builtin = @import("builtin");
const native_arch = builtin.target.cpu.arch;
const maxInt = std.math.maxInt;

const Foo = struct {
    x: u32,
    y: u32,
    z: u32,
};

const VacuousA = struct {
    x: u8,
    y: noreturn,
};

const VacuousB = struct {
    a: VacuousA,
    b: u16,
};

test "@alignOf(T) before referencing T" {
    comptime assert(@alignOf(Foo) != maxInt(usize));
    if (native_arch == .x86_64) {
        comptime assert(@alignOf(Foo) == 4);
    }
}

test "comparison of @alignOf(T) against zero" {
    if (builtin.cpu.arch.isSpirv() and builtin.zig_backend == .stage2_llvm) return error.SkipZigTest;
    const T = struct { x: u32 };
    try expect(!(@alignOf(T) == 0));
    try expect(@alignOf(T) != 0);
    try expect(!(@alignOf(T) < 0));
    try expect(!(@alignOf(T) <= 0));
    try expect(@alignOf(T) > 0);
    try expect(@alignOf(T) >= 0);
}

test "correct alignment for elements and slices of aligned array" {
    if (builtin.cpu.arch.isSpirv()) return error.SkipZigTest;

    var buf: [1024]u8 align(64) = undefined;
    var start: usize = 1;
    var end: usize = undefined;
    _ = .{ &start, &end };
    try expect(@alignOf(@TypeOf(buf[start..end])) == @alignOf(*u8));
    try expect(@alignOf(@TypeOf(&buf[start..end])) == @alignOf(*u8));
    try expect(@alignOf(@TypeOf(&buf[start])) == @alignOf(*u8));
}

test "@alignOf(anyerror!noreturn)" {
    try expect(@alignOf(anyerror!noreturn) == @alignOf(anyerror));
    try expect(@alignOf(anyerror!anyerror!noreturn) == @alignOf(anyerror));
}

test "@alignOf(noreturn)" {
    try expect(@alignOf(noreturn) == 1);
}

test "@alignOf(anyopaque)" {
    try expect(@alignOf(anyopaque) == 1);
}

test "@alignOf(opaque {})" {
    try expect(@alignOf(opaque {}) == 1);
}

test "@alignOf empty enums" {
    const EnumInferred = enum {};
    const EnumExplicit = enum(noreturn) {};

    try expect(@alignOf(EnumInferred) == 1);
    try expect(@alignOf(EnumExplicit) == 1);
}

test "@alignOf vacuous tuples" {
    const A = @Tuple(&.{VacuousA});
    const B = @Tuple(&.{ VacuousA, u16 });

    try expect(@alignOf(A) == 1);
    try expect(@alignOf(B) == @alignOf(u16));
}

test "@alignOf vacuous structs" {
    try expect(@alignOf(VacuousA) == 1);
    try expect(@alignOf(VacuousB) == 2);
}

test "@alignOf vacuous union with explicit field alignment" {
    const C = union(enum) {
        b: VacuousB,
        c: VacuousA align(8),
    };

    try expect(@alignOf(C) == 8);
    try expect(@alignOf([10]C) == 8);
}

test "@alignOf empty unions" {
    const EnumInferred = enum {};
    const EnumExplicit = enum(noreturn) {};
    const EnumNonexhaustive = enum(u8) { _ };

    const U0 = union {};
    const U1 = union(enum) {};
    const U2 = union(enum(noreturn)) {};
    const U3 = union(EnumInferred) {};
    const U4 = union(EnumExplicit) {};
    const U5 = union(EnumNonexhaustive) {};

    try expect(@alignOf(U0) == 1);
    try expect(@alignOf(U1) == 1);
    try expect(@alignOf(U2) == 1);
    try expect(@alignOf(U3) == 1);
    try expect(@alignOf(U4) == 1);
    try expect(@alignOf(U5) == 1);
}
