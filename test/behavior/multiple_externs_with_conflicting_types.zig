const A = extern struct {
    field: c_int,
};

extern fn issue529(?*A) void;

comptime {
    if (!builtin.cpu.arch.isSpirv()) {
        _ = @import("conflicting_externs/b.zig");
    }
}

const builtin = @import("builtin");

test "call extern function defined with conflicting type" {
    if (builtin.zig_backend == .stage2_arm) return error.SkipZigTest; // TODO
    if (builtin.zig_backend == .stage2_sparc64) return error.SkipZigTest; // TODO
    if (builtin.cpu.arch.isSpirv()) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_riscv64) return error.SkipZigTest;

    @import("conflicting_externs/a.zig").issue529(null);
    issue529(null);
}
