const common = @import("./common.zig");
const floatFromInt = @import("./float_from_int.zig").floatFromInt;
const symbol = @import("../compiler_rt.zig").symbol;

comptime {
    if (common.want_ppc_abi) {
        symbol(&__floatunsitf, "__floatunsikf");
    } else if (common.want_sparc_abi) {
        symbol(&_Qp_uitoq, "_Qp_uitoq");
    }
    symbol(&__floatunsitf, "__floatunsitf");
}

pub fn __floatunsitf(a: u32) callconv(.c) f128 {
    return floatFromInt(f128, a);
}

fn _Qp_uitoq(c: *f128, a: u32) callconv(.c) void {
    c.* = floatFromInt(f128, a);
}
