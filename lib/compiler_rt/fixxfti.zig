const builtin = @import("builtin");

const common = @import("./common.zig");
const intFromFloat = @import("./int_from_float.zig").intFromFloat;
const symbol = @import("../compiler_rt.zig").symbol;

comptime {
    if (common.want_windows_v2u64_abi) {
        symbol(&__fixxfti_windows_x86_64, "__fixxfti");
    } else {
        symbol(&__fixxfti, "__fixxfti");
    }
}

pub fn __fixxfti(a: f80) callconv(.c) i128 {
    return intFromFloat(i128, a);
}

const v2u64 = @Vector(2, u64);

fn __fixxfti_windows_x86_64(a: f80) callconv(.c) v2u64 {
    return @bitCast(intFromFloat(i128, a));
}
