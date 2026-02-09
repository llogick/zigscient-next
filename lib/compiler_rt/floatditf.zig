const common = @import("./common.zig");
const floatFromInt = @import("./float_from_int.zig").floatFromInt;
const symbol = @import("../compiler_rt.zig").symbol;

comptime {
    if (common.want_ppc_abi) {
        symbol(&__floatditf, "__floatdikf");
    } else if (common.want_sparc_abi) {
        symbol(&_Qp_xtoq, "_Qp_xtoq");
    }
    symbol(&__floatditf, "__floatditf");
}

pub fn __floatditf(a: i64) callconv(.c) f128 {
    return floatFromInt(f128, a);
}

fn _Qp_xtoq(c: *f128, a: i64) callconv(.c) void {
    c.* = floatFromInt(f128, a);
}
