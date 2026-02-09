const common = @import("./common.zig");
const extendf = @import("./extendf.zig").extendf;
const symbol = @import("../compiler_rt.zig").symbol;

comptime {
    if (common.gnu_f16_abi) {
        symbol(&__gnu_h2f_ieee, "__gnu_h2f_ieee");
    } else if (common.want_aeabi) {
        symbol(&__aeabi_h2f, "__aeabi_h2f");
    }
    symbol(&__extendhfsf2, "__extendhfsf2");
}

pub fn __extendhfsf2(a: common.F16T(f32)) callconv(.c) f32 {
    return extendf(f32, f16, @as(u16, @bitCast(a)));
}

fn __gnu_h2f_ieee(a: common.F16T(f32)) callconv(.c) f32 {
    return extendf(f32, f16, @as(u16, @bitCast(a)));
}

fn __aeabi_h2f(a: u16) callconv(.{ .arm_aapcs = .{} }) f32 {
    return extendf(f32, f16, @as(u16, @bitCast(a)));
}
