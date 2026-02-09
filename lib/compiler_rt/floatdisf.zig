const common = @import("./common.zig");
const symbol = @import("../compiler_rt.zig").symbol;
const floatFromInt = @import("./float_from_int.zig").floatFromInt;

comptime {
    if (common.want_aeabi) {
        symbol(&__aeabi_l2f, "__aeabi_l2f");
    } else {
        if (common.want_windows_arm_abi) {
            symbol(&__floatdisf, "__i64tos");
        }
        symbol(&__floatdisf, "__floatdisf");
    }
}

pub fn __floatdisf(a: i64) callconv(.c) f32 {
    return floatFromInt(f32, a);
}

fn __aeabi_l2f(a: i64) callconv(.{ .arm_aapcs = .{} }) f32 {
    return floatFromInt(f32, a);
}
