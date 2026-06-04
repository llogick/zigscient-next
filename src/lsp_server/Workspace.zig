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
    var stack: std.array_list.Managed(StackItem) = .init(arena);
    // var top_level_steps: std.StringArrayHashMapUnmanaged(Configuration.Step.Index) = .empty;
    for (w.configuration.serialized.?.steps, 0..) |*conf_step, step_index_usize| {
        if (conf_step.owner != .root) continue;

        const step_index: std.Build.Configuration.Step.Index = @enumFromInt(step_index_usize);
        const flags = conf_step.flags(c);
        if (flags.tag != .top_level) continue;

        try stack.append(.{ .step = step_index, .dep_index = 0, .depth = 0 });

        // Process the graph using the stack
        while (stack.items.len > 0) {
            var current = &stack.items[stack.items.len - 1];
            const step = current.step.ptr(c);
            const deps = step.deps.slice(c);

            // First time seeing this step at this depth
            if (current.dep_index == 0) {
                const name = step.name.slice(c);
                const step_flags = step.flags(c);

                const indent = max_spaces[0..@min(current.depth * 4, max_spaces.len)];
                const sub_indent = max_spaces[0..@min((current.depth + 1) * 4, max_spaces.len)];

                switch (step_flags.tag) {
                    .compile => {
                        const compile = step.extended.get(c.extra).compile;
                        const rm = compile.root_module.get(c);
                        var rsf_path: [:0]const u8 = "";
                        if (rm.root_source_file.unwrap()) |rsf| {
                            rsf_path = switch (rsf.get(c)) {
                                .source_path => |sp| sp.sub_path.slice(c),
                                .relative => |rp| rp.sub_path.slice(c),
                                .generated => |gp| if (gp.sub_path != .empty) gp.sub_path.slice(c) else "%gen%",
                            };
                        }
                        log.err("{s}{}: {q} - {q} ({t} {t})", .{
                            indent,
                            @intFromEnum(current.step),
                            compile.root_name.slice(c),
                            name,
                            step_flags.tag,
                            compile.flags3.kind.toOutputMode(),
                        });
                        log.err("{s}-> root={q}", .{ sub_indent, rsf_path });
                        const imports = rm.import_table.get(c).imports;
                        for (imports.mal.items(.name), imports.mal.items(.module)) |import_name, other_mod_idx| {
                            const other_mod: std.Build.Configuration.Module = other_mod_idx.get(c);
                            if (other_mod.root_source_file.unwrap()) |rsf| {
                                rsf_path = switch (rsf.get(c)) {
                                    .source_path => |sp| sp.sub_path.slice(c),
                                    .relative => |rp| rp.sub_path.slice(c),
                                    .generated => |gp| if (gp.sub_path != .empty) gp.sub_path.slice(c) else "%gen%",
                                };
                            } else rsf_path = "";
                            log.err("{s}-> {s}={q}", .{ sub_indent, import_name.slice(c), rsf_path });
                        }
                    },
                    .top_level => log.err("{s}{}: {q} - {q} ({t})", .{
                        indent,
                        @intFromEnum(current.step),
                        name,
                        step.extended.get(c.extra).top_level.description.slice(c),
                        step_flags.tag,
                    }),
                    else => log.err("{s}{}: {q} ({t})", .{
                        indent,
                        @intFromEnum(current.step),
                        name,
                        step_flags.tag,
                    }),
                }
            }

            // Find the next valid dependency to process
            var found_next_dep = false;
            while (current.dep_index < deps.len) {
                const dep = deps[current.dep_index];
                current.dep_index += 1; // Advance for the next iteration

                const dep_step = dep.ptr(c);
                if (dep_step.owner != .root) continue;

                // Push the dependency to the stack to process it next
                try stack.append(.{
                    .step = dep,
                    .dep_index = 0,
                    .depth = current.depth + 1,
                });
                found_next_dep = true;
                break;
            }

            // If no more dependencies, pop this step off the stack
            if (!found_next_dep) {
                _ = stack.pop();
            }
        }
    }
    log.info("Loaded configuration for workspace {q}", .{w.uri});

    // const bf_path = try std.fs.path.join(arena, &.{ dir_path, "build.zig" });
    // const bf_uri = try uri_util.fromPath(s.allocator, bf_path);
    // errdefer s.allocator.free(bf_uri);
    // w.build_file_uri = bf_uri;
    // _ = try s.document_store.getOrLoadHandle(bf_uri);
}

const StackItem = struct {
    step: std.Build.Configuration.Step.Index,
    dep_index: usize,
    depth: usize,
};
const max_spaces: [64]u8 = @splat(' ');

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
