const std = @import("std");
const zig_builtin = @import("builtin");
const build_options = @import("ls_build_options");

const exe_options = @import("build_options");
const known_folders = @import("known-folders");
const lsp = @import("lsp");
const tracy = if (@import("builtin").is_test) @import("tracy") else @import("root").tracy;

const binned_allocator = @import("binned_allocator.zig");
const configuration = @import("configuration.zig");
const debug = @import("debug.zig");
const Server = @import("Server.zig");

const log = std.log.scoped(._main);

const usage =
    \\A non-official language server for Zig
    \\
    \\Commands:
    \\  help, --help,             Print this help and exit
    \\  version, --version        Print version number and exit
    \\  env                       Print config path, log path and version
    \\
    \\General Options:
    \\  --config-path [path]      Set path to the 'zls.json' configuration file
    \\  --enable-message-tracing  Enable message tracing
    \\  --log-file [path]         Set path to the 'zigscient.log' log file
    \\  --log-level [enum]        The Log Level to be used.
    \\                              Supported Values:
    \\                                err
    \\                                warn
    \\                                info (default)
    \\                                debug
    \\
;

pub const std_options: std.Options = .{
    // Always set this to debug to make std.log call into our handler, then control the runtime
    // value in logFn itself
    .log_level = .debug,
    .logFn = logFn,
};

/// Any logging done during the init stage, ie within this file
const init_stage_log_level: std.log.Level = .info;

/// Logging level while running, visible in a client's lsp log
var runtime_log_level: std.log.Level = switch (zig_builtin.mode) {
    .Debug => .debug,
    .ReleaseFast,
    .ReleaseSmall,
    => .err,
    else => .info,
};

var log_file: ?std.fs.File = null;

fn logFn(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.enum_literal),
    comptime format: []const u8,
    args: anytype,
) void {
    if (@intFromEnum(level) > @intFromEnum(runtime_log_level)) return;

    const level_txt: []const u8 = switch (level) {
        .err => "E",
        .warn => "W",
        .info => "I",
        .debug => "D",
    };
    const scope_txt: []const u8 = comptime @tagName(scope);
    if (comptime !std.mem.startsWith(u8, scope_txt, "_")) return; // and
        // !std.mem.startsWith(u8, scope_txt, "zcu")) return; // XXX
    const trimmed_scope = if (comptime std.mem.startsWith(u8, scope_txt, "_")) scope_txt[1..] else scope_txt;

    var buffer: [4096]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);
    const no_space_left = blk: {
        fbs.writer().print("<{s:^6}> {s}: ", .{ trimmed_scope, level_txt }) catch break :blk true;
        fbs.writer().print(format, args) catch break :blk true;
        fbs.writer().writeByte('\n') catch break :blk true;
        break :blk false;
    };
    if (no_space_left) {
        buffer[buffer.len - 3 ..][0..3].* = "...".*;
    }

    std.debug.lockStdErr();
    defer std.debug.unlockStdErr();

    std.io.getStdErr().writeAll(fbs.getWritten()) catch {};

    if (log_file) |file| {
        file.seekFromEnd(0) catch {};
        file.writeAll(fbs.getWritten()) catch {};
    }
}

fn defaultLogFilePath(allocator: std.mem.Allocator) std.mem.Allocator.Error!?[]const u8 {
    const cache_path = known_folders.getPath(allocator, .cache) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.ParseError => return null,
    } orelse return null;
    defer allocator.free(cache_path);
    return try std.fs.path.join(allocator, &.{ cache_path, "zigscient", "zigscient.log" });
}

fn createLogFile(allocator: std.mem.Allocator, override_log_file_path: ?[]const u8) ?struct { std.fs.File, []const u8 } {
    const log_file_path = if (override_log_file_path) |log_file_path|
        allocator.dupe(u8, log_file_path) catch return null
    else
        defaultLogFilePath(allocator) catch null orelse return null;
    errdefer allocator.free(log_file_path);

    if (std.fs.path.dirname(log_file_path)) |dirname| {
        std.fs.cwd().makePath(dirname) catch {};
    }

    const file = std.fs.cwd().createFile(log_file_path, .{ .truncate = false }) catch {
        allocator.free(log_file_path);
        return null;
    };
    errdefer file.close();

    return .{ file, log_file_path };
}

/// Output format of ` env`
const Env = struct {
    /// The project version. Guaranteed to be a [semantic version](https://semver.org/).
    ///
    /// The semantic version can have one of the following formats:
    /// - `MAJOR.MINOR.PATCH` is a tagged release
    /// - `MAJOR.MINOR.PATCH-dev.COMMIT_HEIGHT-SHORT_COMMIT_HASH` is a development build
    /// - `MAJOR.MINOR.PATCH-dev` is a development build, where the exact version could not be resolved.
    ///
    version: []const u8,
    global_cache_dir: ?[]const u8,
    /// Path to where a global `zls.json` could be located.
    /// Not `null` unless `known-folders` was unable to find a global configuration directory.
    global_config_dir: ?[]const u8,
    /// Path to where a local `zls.json` could be located.
    /// Not `null` unless `known-folders` was unable to find a local configuration directory.
    local_config_dir: ?[]const u8,
    /// Path to a `zls.json` config file. Will be resolved by looking in the local configuration directory and then falling back to global directory.
    /// Can be null if no `zls.json` was found in the global/local config directory.
    config_file: ?[]const u8,
    /// Path to a `zls.log` where ZLS will attempt to append logging output.
    /// Not `null` unless `known-folders` was unable to find a cache directory.
    log_file: ?[]const u8,
};

fn cmdEnv(allocator: std.mem.Allocator) (std.mem.Allocator.Error || std.fs.File.WriteError)!noreturn {
    const global_cache_dir = known_folders.getPath(allocator, .cache) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.ParseError => null,
    };
    defer if (global_cache_dir) |path| allocator.free(path);

    const zls_global_cache_dir = if (global_cache_dir) |cache_dir| try std.fs.path.join(allocator, &.{ cache_dir, "zigscient" }) else null;
    defer if (zls_global_cache_dir) |path| allocator.free(path);

    const global_config_dir = known_folders.getPath(allocator, .global_configuration) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.ParseError => null,
    };
    defer if (global_config_dir) |path| allocator.free(path);

    const local_config_dir = known_folders.getPath(allocator, .local_configuration) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.ParseError => null,
    };
    defer if (local_config_dir) |path| allocator.free(path);

    var config_result = try configuration.load(allocator);
    defer config_result.deinit(allocator);

    const config_file_path: ?[]const u8 = switch (config_result) {
        .success => |config_with_path| config_with_path.path,
        .failure => |payload| blk: {
            const message = try payload.toMessage(allocator) orelse break :blk null;
            defer allocator.free(message);
            log.err("Failed to load configuration options.", .{});
            log.err("{s}", .{message});
            break :blk null;
        },
        .not_found => null,
    };

    const log_file_path = try defaultLogFilePath(allocator);
    defer if (log_file_path) |path| allocator.free(path);

    const env: Env = .{
        .version = build_options.version_string,
        .global_cache_dir = zls_global_cache_dir,
        .global_config_dir = global_config_dir,
        .local_config_dir = local_config_dir,
        .config_file = config_file_path,
        .log_file = log_file_path,
    };
    try std.json.stringify(env, .{ .whitespace = .indent_1 }, std.io.getStdOut().writer());
    try std.io.getStdOut().writeAll("\n");
    std.process.exit(0);
}

const ParseArgsResult = struct {
    config_path: ?[]const u8 = null,
    enable_message_tracing: bool = false,
    log_level: ?std.log.Level = null,
    log_file_path: ?[]const u8 = null,
    zls_exe_path: []const u8 = "",

    fn deinit(self: ParseArgsResult, allocator: std.mem.Allocator) void {
        defer if (self.config_path) |path| allocator.free(path);
        defer if (self.log_file_path) |path| allocator.free(path);
        defer allocator.free(self.zls_exe_path);
    }
};

const ParseArgsError = std.process.ArgIterator.InitError || std.mem.Allocator.Error || std.fs.File.WriteError;

fn parseArgs(allocator: std.mem.Allocator) ParseArgsError!ParseArgsResult {
    var result: ParseArgsResult = .{};
    errdefer result.deinit(allocator);

    const stdout = std.io.getStdOut().writer();

    var args_it = try std.process.ArgIterator.initWithAllocator(allocator);
    defer args_it.deinit();

    const zls_exe_path = args_it.next() orelse "";
    result.zls_exe_path = try allocator.dupe(u8, zls_exe_path);

    var arg_index: usize = 0;
    while (args_it.next()) |arg| : (arg_index += 1) {
        if (arg_index == 0) {
            if (std.mem.eql(u8, arg, "help-lsps") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) { // help
                try std.io.getStdErr().writeAll(usage);
                std.process.exit(0);
            } else if (std.mem.eql(u8, arg, "version") or std.mem.eql(u8, arg, "--version")) { // version
                try stdout.writeAll(build_options.version_string ++ "\n");
                std.process.exit(0);
            } else if (std.mem.eql(u8, arg, "env-lsps")) { // env
                try cmdEnv(allocator);
            } else if (std.mem.eql(u8, arg, "--show-config-path")) { // --show-config-path
                comptime std.debug.assert(build_options.version.order(.{ .major = 0, .minor = 14, .patch = 0 }) == .lt); // This flag should be removed before 0.14.0 gets tagged
                log.warn("--show-config-path has been deprecated. Use 'zigscient env-lsps' instead!", .{});

                var config_result = try configuration.load(allocator);
                defer config_result.deinit(allocator);

                switch (config_result) {
                    .success => |config_with_path| {
                        try stdout.writeAll(config_with_path.path);
                        try stdout.writeByte('\n');
                    },
                    .failure => |payload| blk: {
                        const message = try payload.toMessage(allocator) orelse break :blk;
                        defer allocator.free(message);
                        log.err("Failed to load configuration options.", .{});
                        log.err("{s}", .{message});
                    },
                    .not_found => log.info("No config file zls.json found.", .{}),
                }

                log.info("A path to the local configuration folder will be printed instead.", .{});
                const local_config_path = configuration.getLocalConfigPath(allocator) catch null orelse {
                    log.err("failed to find local zls.json", .{});
                    std.process.exit(1);
                };
                defer allocator.free(local_config_path);
                try stdout.writeAll(local_config_path);
                try stdout.writeByte('\n');
                std.process.exit(0);
            }
        }

        if (std.mem.eql(u8, arg, "--config-path")) { // --config-path
            const path = args_it.next() orelse {
                log.err("Expected configuration file path after --config-path argument.", .{});
                std.process.exit(1);
            };
            if (result.config_path) |old_config_path| allocator.free(old_config_path);
            result.config_path = try allocator.dupe(u8, path);
        } else if (std.mem.eql(u8, arg, "--enable-message-tracing")) { // --enable-message-tracing
            result.enable_message_tracing = true;
        } else if (std.mem.eql(u8, arg, "--log-file")) { // --log-file
            const path = args_it.next() orelse {
                log.err("Expected configuration file path after --log-file argument.", .{});
                std.process.exit(1);
            };
            if (result.log_file_path) |old_file_path| allocator.free(old_file_path);
            result.log_file_path = try allocator.dupe(u8, path);
        } else if (std.mem.eql(u8, arg, "--log-level")) { // --log-level
            const log_level_name = args_it.next() orelse {
                log.err("Expected argument after --log-level", .{});
                std.process.exit(1);
            };
            result.log_level = std.meta.stringToEnum(std.log.Level, log_level_name) orelse {
                log.err("Invalid --log-level argument. Expected one of {{'debug', 'info', 'warn', 'err'}} but got '{s}'", .{log_level_name});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--enable-debug-log")) { // --enable-debug-log
            comptime std.debug.assert(build_options.version.order(.{ .major = 0, .minor = 14, .patch = 0 }) == .lt); // This flag should be removed before 0.14.0 gets tagged
            log.warn("--enable-debug-log has been deprecated. Use --log-level instead!", .{});
            result.log_level = .debug;
        } else {
            log.err("Unrecognized argument: '{s}'", .{arg});
            std.process.exit(1);
        }
    }

    if (std.io.getStdIn().isTty()) {
        log.warn("Zigscient is not a CLI tool, it communicates over the Language Server Protocol.", .{});
        log.warn("", .{});
    }

    return result;
}

const LibcAllocatorInterface = struct {
    const Self = @This();
    pub fn allocator(_: Self) std.mem.Allocator {
        return std.heap.c_allocator;
    }
    pub const init: Self = .{};
    pub fn deinit(_: Self) void {}
};

const stack_frames = switch (zig_builtin.mode) {
    .Debug => 10,
    else => 0,
};

pub fn main() !u8 {
    const preferred_runtime_log_level = runtime_log_level;
    runtime_log_level = init_stage_log_level;

    var allocator_state, const allocator_name = if (exe_options.use_gpa)
        .{
            std.heap.GeneralPurposeAllocator(.{ .stack_trace_frames = stack_frames }){},
            "GPA",
        }
    else if (exe_options.llc)
        .{
            LibcAllocatorInterface{},
            "C",
        }
    else
        .{
            binned_allocator.BinnedAllocator(.{}){},
            "Binned",
        };
    defer _ = allocator_state.deinit();

    var tracy_state = if (tracy.enable_allocation) tracy.tracyAllocator(allocator_state.allocator()) else void{};
    const inner_allocator: std.mem.Allocator = if (tracy.enable_allocation) tracy_state.allocator() else allocator_state.allocator();

    var failing_allocator_state = if (exe_options.enable_failing_allocator) debug.FailingAllocator.init(inner_allocator, exe_options.enable_failing_allocator_likelihood) else void{};
    const allocator: std.mem.Allocator = if (exe_options.enable_failing_allocator) failing_allocator_state.allocator() else inner_allocator;

    const result = try parseArgs(allocator);
    defer result.deinit(allocator);

    log_file, const log_file_path =
        // createLogFile(allocator, result.log_file_path) orelse
        .{ null, null };
    defer if (log_file_path) |path| allocator.free(path);
    defer if (log_file) |file| {
        file.close();
        log_file = null;
    };

    const resolved_log_level = result.log_level orelse preferred_runtime_log_level;

    log.info(
        \\Hello/
        \\                      {s}
        \\                      Zigscient       {s}  {s}
        \\                      Allocator       {s}
        \\                      Log Level       {s}
        \\                      Message Tracing {}
    , .{
        result.zls_exe_path,
        build_options.version_string,
        @tagName(zig_builtin.mode),
        allocator_name,
        @tagName(resolved_log_level),
        result.enable_message_tracing,
    });

    runtime_log_level = resolved_log_level;

    var transport: lsp.ThreadSafeTransport(.{
        .ChildTransport = lsp.TransportOverStdio,
        .thread_safe_read = false,
        .thread_safe_write = true,
    }) = .{ .child_transport = lsp.TransportOverStdio.init(std.io.getStdIn(), std.io.getStdOut()) };

    const server = try Server.create(allocator);
    defer server.destroy();
    server.transport = transport.any();
    server.config_path = result.config_path;
    server.message_tracing = result.enable_message_tracing;

    try server.loop();

    switch (server.status) {
        .exiting_failure => return 1,
        .exiting_success => return 0,
        else => unreachable,
    }
}
