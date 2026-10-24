const builtin = @import("builtin");
const A = error{
    FileNotFound,
    NotDir,
};
const B = error{OutOfMemory};

const C = A || B;

fn foo() C!void {
    return error.NotDir;
}

test "merge error sets" {
    if (builtin.cpu.arch.isSpirv() and builtin.zig_backend == .stage2_llvm) return error.SkipZigTest;
    if (builtin.zig_backend == .stage2_sparc64) return error.SkipZigTest; // TODO

    if (foo()) {
        @panic("unexpected");
    } else |err| switch (err) {
        error.OutOfMemory => @panic("unexpected"),
        error.FileNotFound => @panic("unexpected"),
        error.NotDir => {},
    }
}
