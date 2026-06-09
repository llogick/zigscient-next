//! Holds a build.zig's Configuration and Build(Compilation)

/// FlatUri path to a `build.zig`
flat_uri: FlatUri,
/// Explicitly specified path to builtin.zig
builtin_uri: ?FlatUri = null,
/// options loaded from a zls.build.json
options: ?std.json.Parsed(BuildOptions) = null,
roots_index: u32 = 0,
build: Build = .{},
configuration: Configuration = .{},

pub const Configuration = struct {
    mutex: std.Io.Mutex = .init,
    loader_state: LoaderState = .ready,
    version: u32 = 0,
    config: ?std.json.Parsed(Config) = null,
    cfg_file_path: ?[]const u8 = null,
    roots: Roots = .init,

    pub fn release(cfg: *Configuration, io: std.Io) void {
        cfg.mutex.unlock(io);
    }
};

pub const Roots = struct {
    map: std.AutoArrayHashMapUnmanaged(u32, BldDoc.CompileStep),
    info_file_path: ?[]const u8 = null,

    pub const init: Roots = .{ .map = .empty };

    pub fn deinit(roots: *Roots, allocator: std.mem.Allocator) void {
        var map = roots.map;
        for (map.values()) |*value| {
            allocator.free(value.name);
            if (value.args) |args| allocator.free(args);
            for (value.mods.items) |item| {
                allocator.free(item.name);
                allocator.free(item.path);
            }
            value.mods.deinit(allocator);
        }
        map.deinit(allocator);
        if (roots.info_file_path) |ifp| allocator.free(ifp);
    }
};

pub const CompileStep = struct {
    /// The step`s index within configuration.steps
    index: u32,
    name: []const u8,
    args: ?[]const u8 = null,
    mods: std.ArrayList(NamePathPair) = .empty,

    pub const NamePathPair = struct {
        name: []const u8,
        path: []const u8,
    };
};

pub const Build = struct {
    mutex: std.Io.Mutex = .init,
    arena_instance: std.heap.ArenaAllocator = undefined,
    state: ?*CompilationState = null,
    compilation: ?*Compilation = null,
    args: []const []const u8 = undefined,
    has_completed_once: bool = false,
};

const LoaderState = enum {
    ready,
    running,
    running_but_result_already_outdated,
};

pub fn getConfiguration(self: *BldDoc, io: std.Io) *Configuration {
    self.configuration.mutex.lockUncancelable(io);
    return &self.configuration;
}

/// Usage example:
/// ```zig
/// const package_uris: std.ArrayList([]const u8) = .empty;
/// defer {
///     for (package_uris) |uri| allocator.free(uri);
///     package_uris.deinit(allocator);
/// }
/// const success = try build_file.collectBuildConfigPackageUris(allocator, &package_uris);
/// ```
pub fn collectBuildConfigPackageUris(
    self: *BldDoc,
    io: std.Io,
    allocator: std.mem.Allocator,
    package_uris: *std.ArrayList(FlatUri),
) error{OutOfMemory}!bool {
    const tracy_zone = tracy.trace(@src());
    defer tracy_zone.end();

    const cfg = self.getConfiguration(io);
    defer cfg.release(io);
    const build_config = cfg.config orelse return false;

    try package_uris.ensureUnusedCapacity(allocator, build_config.value.modules.len);
    for (build_config.value.modules) |module| {
        package_uris.appendAssumeCapacity(try uri_util.fromPath(allocator, module.path));
    }
    return true;
}

/// Usage example:
/// ```zig
/// const include_paths: std.ArrayList([]u8) = .empty;
/// defer {
///     for (include_paths) |path| allocator.free(path);
///     include_paths.deinit(allocator);
/// }
/// const success = try build_file.collectBuildConfigIncludePaths(allocator, &include_paths);
/// ```
pub fn collectBuildConfigIncludePaths(
    self: *BldDoc,
    io: std.Io,
    allocator: std.mem.Allocator,
    include_paths: *std.ArrayList([]const u8),
) error{OutOfMemory}!bool {
    const tracy_zone = tracy.trace(@src());
    defer tracy_zone.end();

    const build_config = self.tryLockConfig(io) orelse return false;
    defer self.unlockConfig(io);

    try include_paths.ensureUnusedCapacity(allocator, build_config.include_dirs.len);
    for (build_config.include_dirs) |include_path| {
        const absolute_path = if (std.fs.path.isAbsolute(include_path))
            try allocator.dupe(u8, include_path)
        else blk: {
            const build_file_dir = std.fs.path.dirname(self.uri).?;
            const build_file_path = uri_util.toFsPath(allocator, build_file_dir) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => continue,
            };
            defer allocator.free(build_file_path);
            break :blk try std.fs.path.join(allocator, &.{ build_file_path, include_path });
        };

        include_paths.appendAssumeCapacity(absolute_path);
    }
    return true;
}

pub fn triggerRedoCompilation(self: *BldDoc, ds: *DocumentStore) std.Io.Cancelable!void {
    self.redoCompilation(ds) catch |err| switch (err) {
        error.Canceled => return error.Canceled,
        error.OutOfMemory => @panic("OOM"),
    };
}

pub fn runTailor(build_file: *BldDoc, ds: *DocumentStore) !void {
    if (!std.process.can_spawn) return;

    const self_file_path = ds.config.self_file_path orelse return;
    const map = build_file.configuration.roots.map;
    if (!(build_file.roots_index < map.count())) return;
    const step_index = map.keys()[build_file.roots_index];

    std.debug.assert(ds.config.zig_exe_path != null);
    std.debug.assert(ds.config.global_cache_dir != null);
    std.debug.assert(ds.config.zig_lib_dir != null);

    const io = ds.io;

    var arena_state = std.heap.ArenaAllocator.init(ds.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const build_file_path = try uri_util.toFsPath(arena, build_file.flat_uri);
    const cwd = std.fs.path.dirname(build_file_path).?;
    var args: std.ArrayList([]const u8) = .empty;

    try args.appendSlice(arena, &.{
        self_file_path,
        "tailor",
        "--zig",
        ds.config.zig_exe_path.?,
        "--zig-lib-dir",
        ds.config.zig_lib_dir.?.path.?,
        "--build-root",
        cwd,
        "--local-cache",
        try std.fs.path.resolve(arena, &.{ cwd, ".zig-cache" }),
        "--global-cache",
        ds.config.global_cache_dir.?.path.?,
        "--configuration",
        build_file.configuration.cfg_file_path orelse return,
        "--zigscient",
        try std.fmt.allocPrint(arena, "{}", .{step_index}),
    });

    const tailor_run_result = blk: {
        const tracy_zone2 = tracy.trace(@src());
        defer tracy_zone2.end();
        break :blk try std.process.run(
            arena,
            io,
            .{
                .argv = args.items,
                .cwd = .{ .path = cwd },
                .reserve_amount = 1024 * 1024,
            },
        );
    };

    const is_ok = switch (tailor_run_result.term) {
        .exited => |exit_code| exit_code == 0,
        else => false,
    };

    if (!is_ok) {
        const joined = try std.mem.join(arena, " ", args.items);

        log.err(
            "Failed to tailor configuration step_index {} for {q}\nDIR: {s}\nCMD: {s}\nERR:\n{s}",
            .{ step_index, build_file.flat_uri, cwd, joined, tailor_run_result.stderr },
        );

        return error.RunFailed;
    }

    log.err("tailor res:\n{s}", .{tailor_run_result.stdout});
}

fn destroyCompilation(self: *BldDoc, ds: *DocumentStore) void {
    if (self.build.compilation) |comp| comp.destroy();
    if (self.build.state) |state| {
        state.deinit(ds.allocator);
        self.build.state = null;
    }
    self.build.compilation = null;
    _ = self.build.arena_instance.reset(.retain_capacity);
    self.build.has_completed_once = false;
}

fn redoCompilation(self: *BldDoc, ds: *DocumentStore) error{ Canceled, OutOfMemory }!void {
    self.destroyCompilation(ds);
    self.runTailor(ds) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => {},
    };
    try self.initCompilation(ds);
}

fn initCompilation(self: *BldDoc, ds: *DocumentStore) error{ Canceled, OutOfMemory }!void {
    try self.build.mutex.lock(ds.io);
    defer self.build.mutex.unlock(ds.io);

    const cfg = self.configuration.config orelse return;
    if (cfg.value.roots.len == 0) return;

    var cleanup: bool = false;
    defer if (cleanup) {
        self.destroyCompilation(ds);
        log.err("Failed to create a compilation for: {s}", .{self.flat_uri});
    };

    const root_id = if (!(self.roots_index < cfg.value.roots.len)) 0 else self.roots_index;
    const arena = self.build.arena_instance.allocator();
    var args_dups: std.ArrayList([]const u8) = .empty;

    for (cfg.value.roots[root_id].args) |arg| {
        if (std.mem.startsWith(u8, arg, "<generated")) continue;
        try args_dups.append(arena, try arena.dupe(u8, arg));
    }

    self.build.args = try args_dups.toOwnedSlice(arena);

    log.info("Creating a compilation for: {s}\n{s}", .{ self.flat_uri, try std.json.Stringify.valueAlloc(arena, self.build.args, .{}) });

    self.build.state = try arena.create(CompilationState);
    self.build.state.?.* = .{};

    const cmd = self.build.args[1];
    const arg_mode: compiler.ArgMode =
        if (std.mem.eql(u8, cmd, "build-exe")) .{ .build = .Exe } //
        else if (std.mem.eql(u8, cmd, "build-lib")) .{ .build = .Lib } //
        else if (std.mem.eql(u8, cmd, "build-obj")) .{ .build = .Obj } //
        else {
            log.err("redoCompilation: unknown cmd: {s}", .{cmd});
            return;
        };
    buildOutputType(
        ds.allocator,
        arena,
        ds.io,
        self.build.args,
        arg_mode,
        ds.config.environ_map,
        self.build.state.?,
        ds,
        &self.build,
    ) catch |err| switch (err) {
        error.Canceled, error.OutOfMemory => |e| return e,
        else => cleanup = true,
    };
}

pub fn deinit(self: *BldDoc, allocator: std.mem.Allocator) void {
    allocator.free(self.flat_uri);
    if (self.configuration.config) |cfg| cfg.deinit();
    if (self.builtin_uri) |builtin_uri| allocator.free(builtin_uri);
    if (self.options) |opts| opts.deinit();

    if (self.build.compilation) |comp| {
        comp.destroy();
        if (self.build.state) |state| {
            state.deinit(allocator);
            self.build.state = null;
        }
    }
    self.build.arena_instance.deinit();

    if (self.configuration.cfg_file_path) |cfp| allocator.free(cfp);
    self.configuration.roots.deinit(allocator);
}

const std = @import("std");
pub const FlatUri = []const u8;
const uri_util = @import("uri.zig");
const BuildOptions = @import("BuildOptions.zig");
const Config = @import("build_runner/shared.zig").BuildConfig;
const BldDoc = @This();
const DocumentStore = @import("DocumentStore.zig");
const DiagnosticsCollection = @import("DiagnosticsCollection.zig");
const tracy = @import("tracy");
const log = std.log.scoped(.lspc_store);

/// Compiler and Compilation declarations
pub const compiler = @import("compiler");
pub const Compilation = compiler.Compilation;
const CompilationState = compiler.CompilationState;
const buildOutputType = compiler.buildOutputType;
