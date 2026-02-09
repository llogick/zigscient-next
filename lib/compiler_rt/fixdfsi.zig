const common = @import("./common.zig");
const intFromFloat = @import("./int_from_float.zig").intFromFloat;
const symbol = @import("../compiler_rt.zig").symbol;

comptime {
    if (common.want_aeabi) {
        symbol(&__aeabi_d2iz, "__aeabi_d2iz");
    } else {
        symbol(&__fixdfsi, "__fixdfsi");
    }
}

pub fn __fixdfsi(a: f64) callconv(.c) i32 {
    return intFromFloat(i32, a);
}

fn __aeabi_d2iz(a: f64) callconv(.{ .arm_aapcs = .{} }) i32 {
    return intFromFloat(i32, a);
}
