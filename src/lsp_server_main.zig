//! Stage1/Main file of the LSP Server
//! Setup std_options, logging, Allocator and IO

const builtin = @import("builtin");
const native_os = builtin.os.tag;

const std = @import("std");
const mem = std.mem;
const process = std.process;
const fatal = process.fatal;
const Allocator = mem.Allocator;

const build_options = @import("build_options");
const compiler = @import("compiler");
const lsp_server = @import("lsp-server");
const lsp = lsp_server.lsp;

const Logger = @import("lsp_server/Logger.zig");
const cli = @import("lsp_server/cli.zig");

var logger: Logger = .{};
fn logFn(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    Logger.log(&logger, level, scope, format, args);
}

pub const std_options: std.Options = .{
    // Always set this to debug to make std.log call into our handler,
    // then observe the runtime `level` value in the `logger`
    .log_level = .debug,
    .logFn = logFn,
};

const log = std.log.scoped(.lspc_main);

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

    if (args.len > 1 and mem.eql(u8, args[1], "zig")) {
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

    const cli_opts: cli.Options = try .parseArgs(io, gpa, &environ_map, init.args);
    defer cli_opts.deinit(gpa);

    const read_buffer = try gpa.alloc(u8, 4096);
    defer gpa.free(read_buffer);

    var stdio_transport: lsp.Transport.Stdio = .init(read_buffer, .stdin(), .stdout());

    var thread_safe_transport: lsp.ThreadSafeTransport(.{
        .thread_safe_read = false,
        .thread_safe_write = true,
    }) = .init(&stdio_transport.transport);

    const transport: *lsp.Transport = &thread_safe_transport.transport;

    logger.lsp_transport = if (cli_opts.disable_lsp_logs) null else transport;
    logger.dump_to_stderr = cli_opts.enable_stderr_logs;
    logger.level = cli_opts.log_level orelse logger.level;
    defer {
        logger.lsp_transport = null;
        logger.dump_to_stderr = true;
    }

    log.info(
        \\Hello/
        \\                                      ZigscientN {s} {s}
        \\                                      {s}
    , .{
        lsp_server.build_options.version_string,
        @tagName(builtin.mode),
        cli_opts.argv0,
    });

    var config_manager: lsp_server.settings_handler.Manager = try .init(io, gpa, &environ_map);
    defer config_manager.deinit();

    const server: *lsp_server.Server = try .create(.{
        .io = io,
        .allocator = gpa,
        .transport = transport,
        .config_manager = &config_manager,
    });
    defer server.destroy();

    try lsp_server.settings_handler.loadConfiguration(io, gpa, &environ_map, server, cli_opts.config_path);

    try server.loop();

    return switch (server.status) {
        .exiting_success => 0,
        .exiting_failure => 1,
        else => fatal("unexpected server.status {t}", .{server.status}),
    };
}
