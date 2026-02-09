const common = @import("./common.zig");
const symbol = @import("../compiler_rt.zig").symbol;

comptime {
    if (common.want_ppc_abi)
        symbol(&__negtf2, "__negkf2");
    symbol(&__negtf2, "__negtf2");
}

fn __negtf2(a: f128) callconv(.c) f128 {
    return common.fneg(a);
}
