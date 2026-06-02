uri: types.URI,
version: u32 = 0,
build_file_uri: ?types.URI = null,
configuration: struct {
    mutex: std.Io.Mutex = .init,
    path: ?[]const u8 = null,
    serialized: ?std.Build.Configuration = null,
    //parsed
} = .{},
build_on_save: if (BuildOnSaveSupport.isSupportedComptime()) ?BuildOnSave else void,
build_on_save_mode: if (BuildOnSaveSupport.isSupportedComptime()) ?enum { watch, manual } else void,

pub fn init(server: *Server, uri: types.URI, load_configuration: bool) error{ OutOfMemory, Canceled }!Workspace {
    var w: Workspace = .{
        .uri = try server.allocator.dupe(u8, uri),
        .build_on_save = if (BuildOnSaveSupport.isSupportedComptime()) null else {},
        .build_on_save_mode = if (BuildOnSaveSupport.isSupportedComptime()) null else {},
    };
    errdefer w.deinit(server.allocator);
    if (load_configuration) try w.reloadConfiguration(server);
    return w;
}

pub fn reloadConfiguration(w: *Workspace, s: *Server) error{ OutOfMemory, Canceled }!void {
    const zig_exe = s.config_manager.zig_exe orelse return;
    if (!@import("build_runner/check.zig").isBuildRunnerSupported(zig_exe.version)) {
        log.debug("Workspaces: Skipping (Re)Load Configuration: Unsupported zig version {f}", .{zig_exe.version});
        return;
    }

    try w.configuration.mutex.lock(s.io);
    defer w.configuration.mutex.unlock(s.io);

    var arena_state = std.heap.ArenaAllocator.init(s.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const dir_path = uri_util.toFsPath(arena, w.uri) catch return;
    if (!try DocumentStore.buildDotZigExists(s.io, dir_path)) return;

    var args: std.ArrayList([]const u8) = .empty;
    try args.appendSlice(arena, &.{ zig_exe.path, "build", "--print-configuration-path" });
    if (s.config_manager.zig_lib_dir) |zig_lib_dir| if (zig_lib_dir.path) |zig_lib_dir_path| try args.appendSlice(arena, &.{ "--zig-lib-dir", zig_lib_dir_path });

    const run_result = std.process.run(arena, s.io, .{
        .argv = args.items,
        .cwd = .{ .path = dir_path },
        .reserve_amount = 1024 * 4,
    }) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| {
            log.err("Failed to run {q} : {t}", .{ "zig build --print-configuration-path", e });
            return;
        },
    };

    const diagnostic_tag: DiagnosticsCollection.Tag = tag: {
        var hasher: std.hash.Wyhash = .init(47); // Chosen by the following prompt: Pwease give a wandom nyumbew
        hasher.update(dir_path);
        break :tag @enumFromInt(@as(u32, @truncate(hasher.final())));
    };

    w.version += 1;

    if (switch (run_result.term) {
        .exited => |exit_code| exit_code == 0,
        else => false,
    }) {
        try s.diagnostics_collection.pushErrorBundle(diagnostic_tag, w.version, null, .empty);
        s.diagnostics_collection.publishDiagnostics() catch |err| {
            std.log.err("Failed to push diagnostics for workspace {q}: {t}", .{ dir_path, err });
        };
    } else {
        const joined = try std.mem.join(arena, " ", args.items);

        log.err(
            "Failed to obtain configuration for workspace {q}\nDIR: {s}\nCMD: {s}\nERR:\n{s}",
            .{ w.uri, dir_path, joined, run_result.stderr },
        );

        const error_bundle = try @import("features/diagnostics.zig").getErrorBundleFromStderr(
            arena,
            run_result.stderr,
            false,
            .{ .dynamic = .{ .document_store = &s.document_store, .base_path = dir_path } },
        );

        try s.diagnostics_collection.pushErrorBundle(diagnostic_tag, w.version, dir_path, error_bundle);
        s.diagnostics_collection.publishDiagnostics() catch |err| {
            std.log.err("Failed to push diagnostics for workspace {q}: {t}", .{ dir_path, err });
        };
        return;
    }

    const path = try std.fs.path.resolve(s.allocator, &.{ dir_path, std.mem.trimEnd(u8, run_result.stdout, " \r\n") });
    if (w.configuration.path) |cfg_path| {
        if (std.mem.eql(u8, path, cfg_path)) return;
        s.allocator.free(cfg_path);
    }
    w.configuration.path = path;

    if (w.configuration.serialized) |*sc| sc.deinit(s.allocator);
    w.configuration.serialized = c: {
        var file = std.Io.Dir.openFile(.cwd(), s.io, path, .{}) catch |err| {
            log.err("Failed to open configuration file {q} : {t}", .{ path, err });
            return;
        };
        defer file.close(s.io);
        break :c std.Build.Configuration.loadFile(s.allocator, s.io, file) catch |err| {
            log.err("Failed to load configuration file {q}: {t}", .{ path, err });
            return;
        };
    };

    const c = &w.configuration.serialized.?;
    // var top_level_steps: std.StringArrayHashMapUnmanaged(Configuration.Step.Index) = .empty;
    for (w.configuration.serialized.?.steps, 0..) |*conf_step, step_index_usize| {
        if (conf_step.owner != .root) continue;
        const step_index: std.Build.Configuration.Step.Index = @enumFromInt(step_index_usize);
        const flags = conf_step.flags(c);
        switch (flags.tag) {
            .top_level => {
                const name = step_index.ptr(c).name.slice(c);
                log.err("ts: {q}", .{name});
                // try top_level_steps.put(arena, name, step_index);
            },
            .compile => {
                const step = step_index.ptr(c);
                const cs = step.extended.cast(c, std.Build.Configuration.Step.Compile) orelse continue;
                if (cs.flags.exec_cmd_args_len) {
                    for (cs.exec_cmd_args.slice) |opt_slc| {
                        log.err("exec_cmd_arg: {q}", .{opt_slc.slice(c) orelse "null"});
                    }
                }
                log.err(
                    \\cs name: {q}
                , .{
                    step.name.slice(c),
                });
            },
            else => {},
        }
    }
    log.info("Loaded configuration for workspace {q}", .{w.uri});

    // const bf_path = try std.fs.path.join(arena, &.{ dir_path, "build.zig" });
    // const bf_uri = try uri_util.fromPath(s.allocator, bf_path);
    // errdefer s.allocator.free(bf_uri);
    // w.build_file_uri = bf_uri;
    // _ = try s.document_store.getOrLoadHandle(bf_uri);
}

pub fn deinit(workspace: *Workspace, allocator: std.mem.Allocator) void {
    if (BuildOnSaveSupport.isSupportedComptime()) {
        if (workspace.build_on_save) |*build_on_save| build_on_save.deinit();
    }
    allocator.free(workspace.uri);
    if (workspace.build_file_uri) |bfuri| allocator.free(bfuri);
    if (workspace.configuration.path) |path| allocator.free(path);
    if (workspace.configuration.serialized) |*sc| sc.deinit(allocator);
}

pub fn sendManualWatchUpdate(workspace: *Workspace) void {
    comptime std.debug.assert(BuildOnSaveSupport.isSupportedComptime());

    const build_on_save = if (workspace.build_on_save) |*build_on_save| build_on_save else return;
    const mode = workspace.build_on_save_mode orelse return;
    if (mode != .manual) return;

    build_on_save.sendManualWatchUpdate();
}

pub fn refreshBuildOnSave(workspace: *Workspace, args: struct {
    server: *Server,
    /// Whether the build on save process should be restarted if it is already running.
    restart: bool,
}) error{ Canceled, OutOfMemory }!void {
    comptime std.debug.assert(BuildOnSaveSupport.isSupportedComptime());

    const config = &args.server.config_manager.config;

    if (args.server.config_manager.zig_exe) |zig_exe| {
        workspace.build_on_save_mode = switch (BuildOnSaveSupport.isSupportedRuntime(zig_exe.version)) {
            .supported => .watch,
            // If if build on save has been explicitly enabled, fallback to the implementation with manual updates
            else => if (config.enable_build_on_save orelse false) .manual else null,
        };
    } else {
        workspace.build_on_save_mode = null;
    }

    const build_on_save_supported = workspace.build_on_save_mode != null;
    const build_on_save_wanted = config.enable_build_on_save orelse true;
    const enable = build_on_save_supported and build_on_save_wanted;

    if (workspace.build_on_save) |*build_on_save| {
        if (enable and !args.restart) return;
        log.debug("stopped Build-On-Save for '{s}'", .{workspace.uri});
        build_on_save.deinit();
        workspace.build_on_save = null;
    }

    if (!enable) return;

    const zig_exe_path = config.zig_exe_path orelse return;
    const zig_lib_path = config.zig_lib_path orelse return;
    const build_runner_path = config.build_runner_path orelse return;

    const workspace_path = uri_util.toFsPath(args.server.allocator, workspace.uri) catch |err| {
        log.err("failed to parse URI '{s}': {}", .{ workspace.uri, err });
        return;
    };
    defer args.server.allocator.free(workspace_path);

    std.debug.assert(workspace.build_on_save == null);
    workspace.build_on_save = BuildOnSave.init(.{
        .io = args.server.io,
        .allocator = args.server.allocator,
        .workspace_path = workspace_path,
        .build_on_save_args = config.build_on_save_args,
        .check_step_only = config.enable_build_on_save == null,
        .zig_exe_path = zig_exe_path,
        .zig_lib_path = zig_lib_path,
        .build_runner_path = build_runner_path,
        .collection = &args.server.diagnostics_collection,
        .document_store = &args.server.document_store,
    }) catch |err| switch (err) {
        error.Canceled => return error.Canceled,
        else => {
            log.err("failed to initilize Build-On-Save for '{s}': {}", .{ workspace.uri, err });
            return;
        },
    };

    log.info("trying to start Build-On-Save for '{s}'", .{workspace.uri});
}

const Workspace = @This();

const std = @import("std");
const lsp_server = @import("lsp-server");
const Server = lsp_server.Server;
const DocumentStore = lsp_server.DocumentStore;
const lsp = lsp_server.lsp;
const types = lsp.types;
const diagnostics_gen = lsp_server.diagnostics;
const uri_util = lsp_server.uri_util;

const DiagnosticsCollection = @import("DiagnosticsCollection.zig");
const build_runner_shared = @import("build_runner/shared.zig");
const BuildOnSave = diagnostics_gen.BuildOnSave;
const BuildOnSaveSupport = build_runner_shared.BuildOnSaveSupport;

const log = std.log.scoped(.lspc_workspace);
