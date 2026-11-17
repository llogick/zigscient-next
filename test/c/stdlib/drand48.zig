const builtin = @import("builtin");
const std = @import("std");

const c = std.c;
const testing = std.testing;

test "erand48" {
    if (builtin.target.os.tag == .windows) return; // no erand48

    var xsubi: [3]c_ushort = .{ 37174, 64810, 11603 };

    try testing.expectApproxEqAbs(0.8965, c.erand48(&xsubi), 0.0005);
    try testing.expectEqualSlices(c_ushort, &.{ 22537, 47966, 58735 }, &xsubi);

    try testing.expectApproxEqAbs(0.3375, c.erand48(&xsubi), 0.0005);
    try testing.expectEqualSlices(c_ushort, &.{ 37344, 32911, 22119 }, &xsubi);

    try testing.expectApproxEqAbs(0.6475, c.erand48(&xsubi), 0.0005);
    try testing.expectEqualSlices(c_ushort, &.{ 23659, 29872, 42445 }, &xsubi);

    try testing.expectApproxEqAbs(0.5005, c.erand48(&xsubi), 0.0005);
    try testing.expectEqualSlices(c_ushort, &.{ 31642, 7875, 32802 }, &xsubi);

    try testing.expectApproxEqAbs(0.5065, c.erand48(&xsubi), 0.0005);
    try testing.expectEqualSlices(c_ushort, &.{ 64669, 14399, 33170 }, &xsubi);
}

test "jrand48" {
    if (builtin.target.os.tag == .windows) return; // no jrand48

    if (builtin.target.os.tag == .openbsd) return error.SkipZigTest; // TODO: https://marc.info/?l=openbsd-tech&m=179482954031905

    var xsubi: [3]c_ushort = .{ 25175, 11052, 45015 };

    try testing.expectEqual(1699503220, c.jrand48(&xsubi));
    try testing.expectEqualSlices(c_ushort, &.{ 2326, 23668, 25932 }, &xsubi);

    try testing.expectEqual(-992276007, c.jrand48(&xsubi));
    try testing.expectEqualSlices(c_ushort, &.{ 41577, 4569, 50395 }, &xsubi);

    try testing.expectEqual(-19535776, c.jrand48(&xsubi));
    try testing.expectEqualSlices(c_ushort, &.{ 31936, 59488, 65237 }, &xsubi);

    try testing.expectEqual(79438377, c.jrand48(&xsubi));
    try testing.expectEqualSlices(c_ushort, &.{ 40395, 8745, 1212 }, &xsubi);

    try testing.expectEqual(-1258917728, c.jrand48(&xsubi));
    try testing.expectEqualSlices(c_ushort, &.{ 37242, 28832, 46326 }, &xsubi);
}

test "nrand48" {
    if (builtin.target.os.tag == .windows) return; // no nrand48

    var xsubi: [3]c_ushort = .{ 546, 33817, 23389 };

    try testing.expectEqual(914920692, c.nrand48(&xsubi));
    try testing.expectEqualSlices(c_ushort, &.{ 29829, 10728, 27921 }, &xsubi);

    try testing.expectEqual(754104482, c.nrand48(&xsubi));
    try testing.expectEqualSlices(c_ushort, &.{ 6828, 28997, 23013 }, &xsubi);

    try testing.expectEqual(609453945, c.nrand48(&xsubi));
    try testing.expectEqualSlices(c_ushort, &.{ 58183, 3826, 18599 }, &xsubi);

    try testing.expectEqual(1878644360, c.nrand48(&xsubi));
    try testing.expectEqualSlices(c_ushort, &.{ 36678, 44304, 57331 }, &xsubi);

    try testing.expectEqual(2114923686, c.nrand48(&xsubi));
    try testing.expectEqualSlices(c_ushort, &.{ 58585, 22861, 64542 }, &xsubi);
}

test "srand48" {
    if (builtin.target.os.tag == .windows) return; // no srand48

    c.srand48(4);
    try testing.expectEqual(1404534344, c.lrand48());
    try testing.expectEqual(1220337298, c.lrand48());
    try testing.expectEqual(107704692, c.lrand48());

    if (builtin.target.os.tag == .openbsd) return error.SkipZigTest; // TODO: https://marc.info/?l=openbsd-tech&m=179482954031905

    try testing.expectEqual(-1148695706, c.mrand48());
    try testing.expectEqual(0.5300470938792223, c.drand48());
}

test "seed48" {
    if (builtin.target.os.tag == .windows) return; // no seed48
    var seed: [3]c_ushort = .{ 0x330E, 4, 0 };

    _ = c.seed48(&seed);
    try testing.expectEqual(1404534344, c.lrand48());
    try testing.expectEqual(1220337298, c.lrand48());
    try testing.expectEqual(107704692, c.lrand48());

    if (builtin.target.os.tag == .openbsd) return error.SkipZigTest; // TODO: https://marc.info/?l=openbsd-tech&m=179482954031905

    try testing.expectEqual(-1148695706, c.mrand48());
    try testing.expectEqual(0.5300470938792223, c.drand48());
}

test "lcong48" {
    if (builtin.target.os.tag == .windows) return; // no lcong48

    var conf: [7]c_ushort = .{ 0x330E, 4, 0, 0xE66D, 0xDEEC, 5, 0xB };

    _ = c.lcong48(&conf);
    try testing.expectEqual(1404534344, c.lrand48());
    try testing.expectEqual(1220337298, c.lrand48());
    try testing.expectEqual(107704692, c.lrand48());

    if (builtin.target.os.tag == .openbsd) return error.SkipZigTest; // TODO: https://marc.info/?l=openbsd-tech&m=179482954031905

    try testing.expectEqual(-1148695706, c.mrand48());
    try testing.expectEqual(0.5300470938792223, c.drand48());
}
