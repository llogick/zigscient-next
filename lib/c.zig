//! Multi-target implementation of libc, providing ABI compatibility with
//! bundled libcs.
//!
//! mingw-w64 libc is not fully statically linked, so some symbols don't need
//! to be exported. However, a future enhancement could be eliminating Zig's
//! dependency on msvcrt dll even when linking libc and targeting Windows.

const builtin = @import("builtin");
const std = @import("std");

// Avoid dragging in the runtime safety mechanisms into this .o file, unless
// we're trying to test zigc.
pub const panic = if (builtin.is_test)
    std.debug.FullPanic(std.debug.defaultPanic)
else
    std.debug.no_panic;

/// It is incorrect to make this conditional on `builtin.is_test`, because it is possible that
/// libzigc is being linked into a different test compilation, as opposed to being tested itself.
pub const linkage: std.builtin.GlobalLinkage = .strong;

/// Determines the symbol's visibility to other objects.
/// For WebAssembly this allows the symbol to be resolved to other modules, but will not
/// export it to the host runtime.
pub const visibility: std.builtin.SymbolVisibility = .hidden;

pub inline fn symbol(comptime func: *const anyopaque, comptime name: []const u8) void {
    @export(func, .{ .name = name, .linkage = linkage, .visibility = visibility });
}

/// Given a low-level syscall return value, sets errno and returns `-1`, or on
/// success returns the result.
pub fn errno(syscall_return_value: usize) c_int {
    return switch (builtin.os.tag) {
        .linux => {
            const signed: isize = @bitCast(syscall_return_value);
            const casted: c_int = @intCast(signed);
            if (casted < 0) {
                @branchHint(.unlikely);
                std.c._errno().* = -casted;
                return -1;
            }
            return casted;
        },
        else => comptime unreachable,
    };
}

comptime {
    _ = @import("c/inttypes.zig");
    _ = @import("c/ctype.zig");
    _ = @import("c/stdlib.zig");
    _ = @import("c/math.zig");
    _ = @import("c/string.zig");
    _ = @import("c/strings.zig");
    _ = @import("c/wchar.zig");

    _ = @import("c/sys/mman.zig");
    _ = @import("c/sys/file.zig");
    _ = @import("c/sys/reboot.zig");
    _ = @import("c/sys/capability.zig");
    _ = @import("c/sys/utsname.zig");

    _ = @import("c/unistd.zig");
}
