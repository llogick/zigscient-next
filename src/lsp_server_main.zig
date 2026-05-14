//! Stage1/Main file of the LSP Server
//! Setup std_options, Allocator and IO

const builtin = @import("builtin");
const native_os = builtin.os.tag;

const std = @import("std");
const mem = std.mem;
const process = std.process;
const fatal = process.fatal;
const Allocator = mem.Allocator;

const build_options = @import("build_options");
const compiler = @import("compiler");

const log = std.log.scoped(.lspc_main);

pub const std_options: std.Options = .{
    // Always set this to debug to make std.log call into our handler, then control the runtime
    // value in logFn itself
    .log_level = .debug,
    .logFn = @import("lsp_server/src/main.zig").std_options.logFn,
};

const use_debug_allocator = build_options.debug_gpa or
    (native_os != .wasi and !builtin.link_libc and switch (builtin.mode) {
        .Debug, .ReleaseSafe => true,
        .ReleaseFast, .ReleaseSmall => false,
    });

const RootAllocator = if (use_debug_allocator) std.heap.DebugAllocator(.{
    .stack_trace_frames = build_options.mem_leak_frames,
    .thread_safe = switch (build_options.io_mode) {
        .threaded => true,
        .evented => false,
    },
}) else struct {
    pub const init: RootAllocator = .{};
    pub fn allocator(_: RootAllocator) Allocator {
        if (native_os == .wasi) return std.heap.wasm_allocator;
        if (builtin.link_libc) return std.heap.c_allocator;
        return std.heap.smp_allocator;
    }
    pub fn deinit(_: RootAllocator) std.heap.Check {
        return .ok;
    }
};

pub fn main(init: std.process.Init.Minimal) anyerror!u8 {
    var root_allocator: RootAllocator = .init;
    defer _ = root_allocator.deinit();

    const root_gpa = root_allocator.allocator();

    var io_impl: compiler.IoImpl = undefined;

    switch (build_options.io_mode) {
        .threaded => io_impl = .init(root_gpa, .{
            .stack_size = compiler.thread_stack_size,

            .argv0 = .init(init.args),
            .environ = init.environ,
        }),
        .evented => try io_impl.init(root_gpa, .{
            .argv0 = .init(init.args),
            .environ = init.environ,

            .backing_allocator_needs_mutex = use_debug_allocator,
        }),
    }

    defer io_impl.deinit();
    compiler.io_impl_ptr = &io_impl;
    const io = io_impl.io();

    const gpa = switch (build_options.io_mode) {
        .threaded => root_gpa,
        .evented => io_impl.allocator(),
    };

    var arena_instance = std.heap.ArenaAllocator.init(gpa);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const args = try init.args.toSlice(arena);

    if (args.len > 0) compiler.crash_report.zig_argv0 = args[0];

    var environ_map = init.environ.createMap(arena) catch |err| fatal("failed to parse environment: {t}", .{err});

    if (args.len <= 1 or (args.len > 1 and !mem.eql(u8, args[1], "zig"))) {
        return @import("lsp_server/src/main.zig").stage2(gpa, io, init, &environ_map);
    }

    if (args.len <= 2) {
        if (build_options.dev != .full) {
            log.info(
                \\
                \\This is a limited build, '{t}', of the Zig compiler,
                \\only `zig build-* -fno-emit-bin` commands available.
            , .{build_options.dev});
        } else {
            log.info("{s}", .{compiler.usage});
        }
        fatal("expected command argument", .{});
    }

    if (compiler.tracy.enable_allocation) {
        var gpa_tracy = compiler.tracy.tracyAllocator(gpa);
        try compiler.mainArgs(gpa_tracy.allocator(), arena, io, args, &environ_map);
        return 0;
    }

    if (native_os == .wasi) {
        compiler.preopens = try .init(arena);
    }

    try compiler.mainArgs(gpa, arena, io, args, &environ_map);
    return 0;
}
