//! Holds a build.zig's Configuration and Build(Compilation)

/// FlatUri path to a `build.zig`
flat_uri: FlatUri,
/// Explicitly specified path to builtin.zig
builtin_uri: ?FlatUri = null,
/// options loaded from a zls.build.json
options: ?std.json.Parsed(BuildOptions) = null,
roots_index: u32 = 0,
build: Build = .{},
impl: struct {
    mutex: std.Io.Mutex = .init,
    build_runner_state: BuildRunnerState = .idle,
    version: u32 = 0,
    /// contains information extracted from running build.zig with a custom build runner
    /// e.g. include paths & packages
    /// TODO this field should not be nullable, callsites should await the build config to be resolved
    /// and then continue instead of dealing with missing information.
    config: ?std.json.Parsed(Config) = null,
} = .{},

pub const Build = struct {
    mutex: std.Io.Mutex = .init,
    arena_instance: std.heap.ArenaAllocator = undefined,
    state: *CompilationState = undefined,
    compilation: ?*Compilation = null,
    args: []const []const u8 = undefined,
    has_completed_once: bool = false,
};

const BuildRunnerState = enum {
    idle,
    running,
    running_but_already_invalidated,
};

pub fn tryLockConfig(self: *BldDoc, io: std.Io) ?Config {
    self.impl.mutex.lockUncancelable(io);
    return if (self.impl.config) |cfg| cfg.value else {
        self.impl.mutex.unlock(io);
        return null;
    };
}

pub fn unlockConfig(self: *BldDoc, io: std.Io) void {
    self.impl.mutex.unlock(io);
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

    const build_config = self.tryLockConfig(io) orelse return false;
    defer self.unlockConfig(io);

    try package_uris.ensureUnusedCapacity(allocator, build_config.packages.len);
    for (build_config.packages) |package| {
        package_uris.appendAssumeCapacity(try uri.fromPath(allocator, package.path));
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
            const build_file_path = uri.toFsPath(allocator, build_file_dir) catch |err| switch (err) {
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

fn destroyCompilation(self: *BldDoc, ds: *DocumentStore) void {
    if (self.build.compilation) |comp| comp.destroy();
    self.build.state.deinit(ds.allocator);
    self.build.compilation = null;
    self.build.state = undefined;
    _ = self.build.arena_instance.reset(.retain_capacity);
    self.build.has_completed_once = false;
}

fn redoCompilation(self: *BldDoc, ds: *DocumentStore) error{ Canceled, OutOfMemory }!void {
    self.destroyCompilation(ds);
    try self.initCompilation(ds);
}

fn initCompilation(self: *BldDoc, ds: *DocumentStore) error{ Canceled, OutOfMemory }!void {
    try self.build.mutex.lock(ds.io);
    defer self.build.mutex.unlock(ds.io);

    const cfg = self.impl.config orelse return;
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
    self.build.state.* = .{};

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
        self.build.state,
        ds,
        &self.build,
    ) catch |err| switch (err) {
        error.Canceled, error.OutOfMemory => |e| return e,
        else => cleanup = true,
    };
}

pub fn deinit(self: *BldDoc, allocator: std.mem.Allocator) void {
    allocator.free(self.flat_uri);
    if (self.impl.config) |cfg| cfg.deinit();
    if (self.builtin_uri) |builtin_uri| allocator.free(builtin_uri);
    if (self.options) |opts| opts.deinit();

    if (self.build.compilation) |comp| {
        self.build.state.deinit(allocator);
        comp.destroy();
    }
    self.build.arena_instance.deinit();
}

const std = @import("std");
pub const FlatUri = []const u8;
const uri = @import("uri.zig");
const BuildOptions = @import("BuildOptions.zig");
const Config = @import("build_runner/shared.zig").BuildConfig;
const BldDoc = @This();
const DocumentStore = @import("DocumentStore.zig");
const tracy = @import("tracy");
const log = std.log.scoped(.lspc_store);

/// Compiler and Compilation declarations
pub const compiler = @import("compiler");
pub const Compilation = compiler.Compilation;
const CompilationState = compiler.CompilationState;
const buildOutputType = compiler.buildOutputType;
