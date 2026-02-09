const common = @import("./common.zig");
const floatFromInt = @import("./float_from_int.zig").floatFromInt;
const symbol = @import("../compiler_rt.zig").symbol;

comptime {
    if (common.want_aeabi) {
        symbol(&__aeabi_ui2f, "__aeabi_ui2f");
    } else {
        symbol(&__floatunsisf, "__floatunsisf");
    }
}

pub fn __floatunsisf(a: u32) callconv(.c) f32 {
    return floatFromInt(f32, a);
}

fn __aeabi_ui2f(a: u32) callconv(.{ .arm_aapcs = .{} }) f32 {
    return floatFromInt(f32, a);
}
