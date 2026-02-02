const builtin = @import("builtin");
const std = @import("std");

/// It is incorrect to make this conditional on `builtin.is_test`, because it is possible that
/// libzigc is being linked into a different test compilation, as opposed to being tested itself.
pub const linkage: std.builtin.GlobalLinkage = .strong;

/// Determines the symbol's visibility to other objects.
/// For WebAssembly this allows the symbol to be resolved to other modules, but will not
/// export it to the host runtime.
pub const visibility: std.builtin.SymbolVisibility = .hidden;

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
