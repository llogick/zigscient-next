//! PLEASE READ THE FOLLOWING MESSAGE BEFORE EDITING THIS FILE:
//!
//! This build runner is targeting compatibility with the following Zig versions:
//!   - Zig 0.12.0
//!   - Zig 0.13.0
//!   - master
//!
//! Handling multiple Zig versions can be achieved by branching on the `builtin.zig_version` at comptime.
//! As an example, see how `writeFile2_removed_version` or `std_progress_rework_version` are used to deal with breaking changes.
//!
//! You can test out the build runner on ZLS's `build.zig` with the following command:
//! `zig build --build-runner src/build_runner/0.12.0.zig`
//!
//! You can also test the build runner on any other `build.zig` with the following command:
//! `zig build --build-file /path/to/build.zig --build-runner /path/to/zls/src/build_runner/0.12.0.zig`
//! `zig build --build-runner /path/to/zls/src/build_runner/0.12.0.zig` (if the cwd contains build.zig)
//!

const root = @import("@build");
const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const mem = std.mem;
const process = std.process;
const ArrayList = std.ArrayList;
const Step = std.Build.Step;
const Watch = std.Build.Watch;
const Allocator = std.mem.Allocator;
const Module = std.Build.Module;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const dependencies = @import("@dependencies");

// ----------- List of Zig versions that introduced breaking changes -----------

const writeFile2_removed_version =
    std.SemanticVersion.parse("0.13.0-dev.68+b86c4bde6") catch unreachable;
const std_progress_rework_version =
    std.SemanticVersion.parse("0.13.0-dev.336+963ffe9d5") catch unreachable;
const file_watch_version =
    std.SemanticVersion.parse("0.14.0-dev.283+1d20ff11d") catch unreachable;
const live_rebuild_processes =
    std.SemanticVersion.parse("0.14.0-dev.310+9d38e82b5") catch unreachable;
const file_watch_windows_version =
    std.SemanticVersion.parse("0.14.0-dev.625+2de0e2eca") catch unreachable;
const child_type_coercion_version =
    std.SemanticVersion.parse("0.14.0-dev.2506+32354d119") catch unreachable;
const accept_root_module_version =
    std.SemanticVersion.parse("0.14.0-dev.2534+12d64c456") catch unreachable;

// -----------------------------------------------------------------------------

const ProgressNode = if (builtin.zig_version.order(std_progress_rework_version) == .lt)
    *std.Progress.Node
else
    std.Progress.Node;

var self_path: [:0]const u8 = undefined;
var build_root: [:0]const u8 = undefined;

///! This is a modified build runner to extract information out of build.zig
///! Modified version of lib/build_runner.zig
pub fn main() !void {
    // Here we use an ArenaAllocator backed by a DirectAllocator because a build is a short-lived,
    // one shot program. We don't need to waste time freeing memory and finding places to squish
    // bytes into. So we free everything all at once at the very end.
    var single_threaded_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer single_threaded_arena.deinit();

    var thread_safe_arena: std.heap.ThreadSafeAllocator = .{
        .child_allocator = single_threaded_arena.allocator(),
    };
    const arena = thread_safe_arena.allocator();

    const args = try process.argsAlloc(arena);

    var arg_idx: usize = 0;

    self_path = nextArg(args, &arg_idx) orelse unreachable;
    const zig_exe = nextArg(args, &arg_idx) orelse fatal("missing zig compiler path", .{});
    const zig_lib_directory = if (comptime builtin.zig_version.order(file_watch_version).compare(.gte)) blk: {
        const zig_lib_dir = nextArg(args, &arg_idx) orelse fatal("missing zig lib directory path", .{});

        const zig_lib_directory: std.Build.Cache.Directory = .{
            .path = zig_lib_dir,
            .handle = try std.fs.cwd().openDir(zig_lib_dir, .{}),
        };

        break :blk zig_lib_directory;
    } else {};
    build_root = nextArg(args, &arg_idx) orelse fatal("missing build root directory path", .{});
    const cache_root = nextArg(args, &arg_idx) orelse fatal("missing cache root directory path", .{});
    const global_cache_root = nextArg(args, &arg_idx) orelse fatal("missing global cache root directory path", .{});

    const build_root_directory: std.Build.Cache.Directory = .{
        .path = build_root,
        .handle = try std.fs.cwd().openDir(build_root, .{}),
    };

    const local_cache_directory: std.Build.Cache.Directory = .{
        .path = cache_root,
        .handle = try std.fs.cwd().makeOpenPath(cache_root, .{}),
    };

    const global_cache_directory: std.Build.Cache.Directory = .{
        .path = global_cache_root,
        .handle = try std.fs.cwd().makeOpenPath(global_cache_root, .{}),
    };

    var graph: std.Build.Graph = if (comptime builtin.zig_version.order(file_watch_version).compare(.gte)) .{
        .arena = arena,
        .cache = .{
            .gpa = arena,
            .manifest_dir = try local_cache_directory.handle.makeOpenPath("h", .{}),
        },
        .zig_exe = zig_exe,
        .env_map = try process.getEnvMap(arena),
        .global_cache_root = global_cache_directory,
        .zig_lib_directory = zig_lib_directory,
        .host = .{
            .query = .{},
            .result = try std.zig.system.resolveTargetQuery(.{}),
        },
    } else .{
        .arena = arena,
        .cache = .{
            .gpa = arena,
            .manifest_dir = try local_cache_directory.handle.makeOpenPath("h", .{}),
        },
        .zig_exe = zig_exe,
        .env_map = try process.getEnvMap(arena),
        .global_cache_root = global_cache_directory,
        .host = .{
            .query = .{},
            .result = try std.zig.system.resolveTargetQuery(.{}),
        },
    };

    graph.cache.addPrefix(.{ .path = null, .handle = std.fs.cwd() });
    graph.cache.addPrefix(build_root_directory);
    graph.cache.addPrefix(local_cache_directory);
    graph.cache.addPrefix(global_cache_directory);
    graph.cache.hash.addBytes(builtin.zig_version_string);

    const builder = try std.Build.create(
        &graph,
        build_root_directory,
        local_cache_directory,
        dependencies.root_deps,
    );

    var targets = ArrayList([]const u8).init(arena);
    var debug_log_scopes = ArrayList([]const u8).init(arena);
    var thread_pool_options: std.Thread.Pool.Options = .{ .allocator = arena };

    var install_prefix: ?[]const u8 = null;
    var dir_list = std.Build.DirList{};
    var max_rss: u64 = 0;
    var skip_oom_steps = false;
    var seed: u32 = 0;
    var output_tmp_nonce: ?[16]u8 = null;
    var debounce_interval_ms: u16 = 50;

    while (nextArg(args, &arg_idx)) |arg| {
        if (mem.startsWith(u8, arg, "-Z")) {
            if (arg.len != 18) fatal("bad argument: '{s}'", .{arg});
            output_tmp_nonce = arg[2..18].*;
        } else if (mem.startsWith(u8, arg, "-D")) {
            const option_contents = arg[2..];
            if (option_contents.len == 0)
                fatal("expected option name after '-D'", .{});
            if (mem.indexOfScalar(u8, option_contents, '=')) |name_end| {
                const option_name = option_contents[0..name_end];
                const option_value = option_contents[name_end + 1 ..];
                if (try builder.addUserInputOption(option_name, option_value))
                    fatal("  access the help menu with 'zig build -h'", .{});
            } else {
                if (try builder.addUserInputFlag(option_contents))
                    fatal("  access the help menu with 'zig build -h'", .{});
            }
        } else if (mem.startsWith(u8, arg, "-")) {
            if (mem.eql(u8, arg, "--verbose")) {
                builder.verbose = true;
            } else if (mem.eql(u8, arg, "-h") or mem.eql(u8, arg, "--help")) {
                fatal("argument '{s}' is not available", .{arg});
            } else if (mem.eql(u8, arg, "-p") or mem.eql(u8, arg, "--prefix")) {
                install_prefix = nextArgOrFatal(args, &arg_idx);
            } else if (mem.eql(u8, arg, "-l") or mem.eql(u8, arg, "--list-steps")) {
                fatal("argument '{s}' is not available", .{arg});
            } else if (mem.startsWith(u8, arg, "-fsys=")) {
                const name = arg["-fsys=".len..];
                graph.system_library_options.put(arena, name, .user_enabled) catch @panic("OOM");
            } else if (mem.startsWith(u8, arg, "-fno-sys=")) {
                const name = arg["-fno-sys=".len..];
                graph.system_library_options.put(arena, name, .user_disabled) catch @panic("OOM");
            } else if (mem.eql(u8, arg, "--release")) {
                builder.release_mode = .any;
            } else if (mem.startsWith(u8, arg, "--release=")) {
                const text = arg["--release=".len..];
                builder.release_mode = std.meta.stringToEnum(std.Build.ReleaseMode, text) orelse {
                    fatal("expected [off|any|fast|safe|small] in '{s}', found '{s}'", .{
                        arg, text,
                    });
                };
            } else if (mem.eql(u8, arg, "--prefix-lib-dir")) {
                dir_list.lib_dir = nextArgOrFatal(args, &arg_idx);
            } else if (mem.eql(u8, arg, "--prefix-exe-dir")) {
                dir_list.exe_dir = nextArgOrFatal(args, &arg_idx);
            } else if (mem.eql(u8, arg, "--prefix-include-dir")) {
                dir_list.include_dir = nextArgOrFatal(args, &arg_idx);
            } else if (mem.eql(u8, arg, "--sysroot")) {
                builder.sysroot = nextArgOrFatal(args, &arg_idx);
            } else if (mem.eql(u8, arg, "--maxrss")) {
                const max_rss_text = nextArgOrFatal(args, &arg_idx);
                max_rss = std.fmt.parseIntSizeSuffix(max_rss_text, 10) catch |err| {
                    std.debug.print("invalid byte size: '{s}': {s}\n", .{
                        max_rss_text, @errorName(err),
                    });
                    process.exit(1);
                };
            } else if (mem.eql(u8, arg, "--skip-oom-steps")) {
                skip_oom_steps = true;
            } else if (mem.eql(u8, arg, "--search-prefix")) {
                const search_prefix = nextArgOrFatal(args, &arg_idx);
                builder.addSearchPrefix(search_prefix);
            } else if (mem.eql(u8, arg, "--libc")) {
                builder.libc_file = nextArgOrFatal(args, &arg_idx);
            } else if (mem.eql(u8, arg, "--color")) {
                const next_arg = nextArg(args, &arg_idx) orelse
                    fatal("expected [auto|on|off] after '{s}'", .{arg});
                _ = next_arg;
            } else if (mem.eql(u8, arg, "--summary")) {
                const next_arg = nextArg(args, &arg_idx) orelse
                    fatal("expected [all|new|failures|none] after '{s}'", .{arg});
                _ = next_arg;
            } else if ((comptime builtin.zig_version.order(file_watch_version) == .lt) and mem.eql(u8, arg, "--zig-lib-dir")) {
                builder.zig_lib_dir = .{ .cwd_relative = nextArgOrFatal(args, &arg_idx) };
            } else if (mem.eql(u8, arg, "--seed")) {
                const next_arg = nextArg(args, &arg_idx) orelse
                    fatal("expected u32 after '{s}'", .{arg});
                seed = std.fmt.parseUnsigned(u32, next_arg, 0) catch |err| {
                    fatal("unable to parse seed '{s}' as unsigned 32-bit integer: {s}\n", .{
                        next_arg, @errorName(err),
                    });
                };
            } else if ((builtin.zig_version.order(file_watch_version) != .lt) and mem.eql(u8, arg, "--debounce")) {
                const next_arg = nextArg(args, &arg_idx) orelse
                    fatal("expected u16 after '{s}'", .{arg});
                debounce_interval_ms = std.fmt.parseUnsigned(u16, next_arg, 0) catch |err| {
                    fatal("unable to parse debounce interval '{s}' as unsigned 16-bit integer: {s}\n", .{
                        next_arg, @errorName(err),
                    });
                };
            } else if (mem.eql(u8, arg, "--debug-log")) {
                const next_arg = nextArgOrFatal(args, &arg_idx);
                try debug_log_scopes.append(next_arg);
            } else if (mem.eql(u8, arg, "--debug-pkg-config")) {
                builder.debug_pkg_config = true;
            } else if (mem.eql(u8, arg, "--debug-compile-errors")) {
                builder.debug_compile_errors = true;
            } else if (mem.eql(u8, arg, "--system")) {
                // The usage text shows another argument after this parameter
                // but it is handled by the parent process. The build runner
                // only sees this flag.
                graph.system_package_mode = true;
            } else if (mem.eql(u8, arg, "--glibc-runtimes")) {
                builder.glibc_runtimes_dir = nextArgOrFatal(args, &arg_idx);
            } else if (mem.eql(u8, arg, "--verbose-link")) {
                builder.verbose_link = true;
            } else if (mem.eql(u8, arg, "--verbose-air")) {
                builder.verbose_air = true;
            } else if (mem.eql(u8, arg, "--verbose-llvm-ir")) {
                builder.verbose_llvm_ir = "-";
            } else if (mem.startsWith(u8, arg, "--verbose-llvm-ir=")) {
                builder.verbose_llvm_ir = arg["--verbose-llvm-ir=".len..];
            } else if (mem.eql(u8, arg, "--verbose-llvm-bc=")) {
                builder.verbose_llvm_bc = arg["--verbose-llvm-bc=".len..];
            } else if (mem.eql(u8, arg, "--verbose-cimport")) {
                builder.verbose_cimport = true;
            } else if (mem.eql(u8, arg, "--verbose-cc")) {
                builder.verbose_cc = true;
            } else if (mem.eql(u8, arg, "--verbose-llvm-cpu-features")) {
                builder.verbose_llvm_cpu_features = true;
            } else if (mem.eql(u8, arg, "--prominent-compile-errors")) {
                // prominent_compile_errors = true;
            } else if ((builtin.zig_version.order(file_watch_version) != .lt) and mem.eql(u8, arg, "--watch")) {
                // watch mode will always be enabled if supported
                // watch = true;
            } else if (mem.eql(u8, arg, "-fwine")) {
                builder.enable_wine = true;
            } else if (mem.eql(u8, arg, "-fno-wine")) {
                builder.enable_wine = false;
            } else if (mem.eql(u8, arg, "-fqemu")) {
                builder.enable_qemu = true;
            } else if (mem.eql(u8, arg, "-fno-qemu")) {
                builder.enable_qemu = false;
            } else if (mem.eql(u8, arg, "-fwasmtime")) {
                builder.enable_wasmtime = true;
            } else if (mem.eql(u8, arg, "-fno-wasmtime")) {
                builder.enable_wasmtime = false;
            } else if (mem.eql(u8, arg, "-frosetta")) {
                builder.enable_rosetta = true;
            } else if (mem.eql(u8, arg, "-fno-rosetta")) {
                builder.enable_rosetta = false;
            } else if (mem.eql(u8, arg, "-fdarling")) {
                builder.enable_darling = true;
            } else if (mem.eql(u8, arg, "-fno-darling")) {
                builder.enable_darling = false;
            } else if (mem.eql(u8, arg, "-freference-trace")) {
                builder.reference_trace = 256;
            } else if (mem.startsWith(u8, arg, "-freference-trace=")) {
                const num = arg["-freference-trace=".len..];
                builder.reference_trace = std.fmt.parseUnsigned(u32, num, 10) catch |err| {
                    std.debug.print("unable to parse reference_trace count '{s}': {s}", .{ num, @errorName(err) });
                    process.exit(1);
                };
            } else if (mem.eql(u8, arg, "-fno-reference-trace")) {
                builder.reference_trace = null;
            } else if (mem.startsWith(u8, arg, "-j")) {
                const num = arg["-j".len..];
                const n_jobs = std.fmt.parseUnsigned(u32, num, 10) catch |err| {
                    std.debug.print("unable to parse jobs count '{s}': {s}", .{
                        num, @errorName(err),
                    });
                    process.exit(1);
                };
                if (n_jobs < 1) {
                    std.debug.print("number of jobs must be at least 1\n", .{});
                    process.exit(1);
                }
                thread_pool_options.n_jobs = n_jobs;
            } else if (mem.eql(u8, arg, "--")) {
                builder.args = argsRest(args, arg_idx);
                break;
            } else {
                fatal("unrecognized argument: '{s}'", .{arg});
            }
        } else {
            try targets.append(arg);
        }
    }

    var progress = if (comptime builtin.zig_version.order(std_progress_rework_version) == .lt)
        std.Progress{ .terminal = null }
    else {};

    const main_progress_node = if (comptime builtin.zig_version.order(std_progress_rework_version) == .lt)
        progress.start("", 0)
    else
        std.Progress.start(.{
            .disable_printing = true,
        });
    defer main_progress_node.end();

    builder.debug_log_scopes = debug_log_scopes.items;
    builder.resolveInstallPrefix(install_prefix, dir_list);
    {
        var prog_node = main_progress_node.start("Configure", 0);
        defer prog_node.end();
        try builder.runBuild(root);
        if (comptime builtin.zig_version.order(accept_root_module_version) != .lt) {
            createModuleDependencies(builder) catch @panic("OOM");
        }
    }

    if (graph.needed_lazy_dependencies.entries.len != 0) {
        var buffer: std.ArrayListUnmanaged(u8) = .{};
        for (graph.needed_lazy_dependencies.keys()) |k| {
            try buffer.appendSlice(arena, k);
            try buffer.append(arena, '\n');
        }
        const s = std.fs.path.sep_str;
        const tmp_sub_path = "tmp" ++ s ++ (output_tmp_nonce orelse fatal("missing -Z arg", .{}));

        const writeFileFn = if (comptime builtin.zig_version.order(writeFile2_removed_version) == .lt)
            std.fs.Dir.writeFile2
        else
            std.fs.Dir.writeFile;

        writeFileFn(local_cache_directory.handle, .{
            .sub_path = tmp_sub_path,
            .data = buffer.items,
            .flags = .{ .exclusive = true },
        }) catch |err| {
            fatal("unable to write configuration results to '{}{s}': {s}", .{
                local_cache_directory, tmp_sub_path, @errorName(err),
            });
        };

        process.exit(3); // Indicate configure phase failed with meaningful stdout.
    }

    if (builder.validateUserInputDidItFail()) {
        fatal("  access the help menu with 'zig build -h'", .{});
    }

    validateSystemLibraryOptions(builder);

    var run: Run = .{
        .max_rss = max_rss,
        .max_rss_is_default = false,
        .max_rss_mutex = .{},
        .skip_oom_steps = skip_oom_steps,
        .memory_blocked_steps = std.ArrayList(*Step).init(arena),
        .thread_pool = undefined, // set below

        .claimed_rss = 0,
    };

    if (run.max_rss == 0) {
        run.max_rss = process.totalSystemMemory() catch std.math.maxInt(u64);
        run.max_rss_is_default = true;
    }

    try run.thread_pool.init(thread_pool_options);
    defer run.thread_pool.deinit();

    const gpa = arena;
    try extractBuildInformation(
        gpa,
        builder,
        arena,
        main_progress_node,
        &run,
        seed,
    );

    const watch_suported = switch (builtin.os.tag) {
        .linux => blk: {
            @setEvalBranchQuota(10000);
            if (comptime builtin.zig_version.order(file_watch_version) == .lt) break :blk false;

            // std.build.Watch requires `FAN_REPORT_TARGET_FID` which is Linux 5.17+
            const utsname = std.posix.uname();
            const version = std.SemanticVersion.parse(&utsname.release) catch break :blk true;
            break :blk version.order(.{ .major = 5, .minor = 17, .patch = 0 }) != .lt;
        },
        .windows => comptime builtin.zig_version.order(file_watch_windows_version) != .lt,
        else => false,
    };
    if (!watch_suported) return;
    var w = try Watch.init();

    var step_stack = try stepNamesToStepStack(gpa, builder, targets.items);

    prepare(gpa, builder, &step_stack, &run, seed) catch |err| switch (err) {
        error.UncleanExit => process.exit(1),
        else => return err,
    };

    // TODO watch mode is currently always disabled until ZLS supports it
    rebuild: while (false) {
        runSteps(
            gpa,
            builder,
            step_stack.keys(),
            main_progress_node,
            &run,
        ) catch |err| switch (err) {
            error.UncleanExit => process.exit(1),
            else => return err,
        };

        try w.update(gpa, step_stack.keys());

        // Wait until a file system notification arrives. Read all such events
        // until the buffer is empty. Then wait for a debounce interval, resetting
        // if any more events come in. After the debounce interval has passed,
        // trigger a rebuild on all steps with modified inputs, as well as their
        // recursive dependants.
        var debounce_timeout: Watch.Timeout = .none;
        while (true) switch (try w.wait(gpa, debounce_timeout)) {
            .timeout => {
                markFailedStepsDirty(gpa, step_stack.keys());
                continue :rebuild;
            },
            .dirty => if (debounce_timeout == .none) {
                debounce_timeout = .{ .ms = debounce_interval_ms };
            },
            .clean => {},
        };
    }
}

fn markFailedStepsDirty(gpa: Allocator, all_steps: []const *Step) void {
    for (all_steps) |step| switch (step.state) {
        .dependency_failure, .failure, .skipped => step.recursiveReset(gpa),
        else => continue,
    };
    // Now that all dirty steps have been found, the remaining steps that
    // succeeded from last run shall be marked "cached".
    for (all_steps) |step| switch (step.state) {
        .success => step.result_cached = true,
        else => continue,
    };
}

const Run = struct {
    max_rss: u64,
    max_rss_is_default: bool,
    max_rss_mutex: std.Thread.Mutex,
    skip_oom_steps: bool,
    memory_blocked_steps: std.ArrayList(*Step),
    thread_pool: std.Thread.Pool,

    claimed_rss: usize,
};

fn stepNamesToStepStack(
    gpa: Allocator,
    b: *std.Build,
    step_names: []const []const u8,
) !std.AutoArrayHashMapUnmanaged(*Step, void) {
    var step_stack: std.AutoArrayHashMapUnmanaged(*Step, void) = .{};
    errdefer step_stack.deinit(gpa);

    if (step_names.len == 0) {
        const default_step = if (b.top_level_steps.get("check")) |tls| &tls.step else b.default_step;
        try step_stack.put(gpa, default_step, {});
    } else {
        try step_stack.ensureUnusedCapacity(gpa, step_names.len);
        for (0..step_names.len) |i| {
            const step_name = step_names[step_names.len - i - 1];
            const s = b.top_level_steps.get(step_name) orelse {
                std.debug.print("no step named '{s}'\n  access the help menu with 'zig build -h'\n", .{step_name});
                process.exit(1);
            };
            step_stack.putAssumeCapacity(&s.step, {});
        }
    }

    return step_stack;
}

fn prepare(
    gpa: Allocator,
    b: *std.Build,
    step_stack: *std.AutoArrayHashMapUnmanaged(*Step, void),
    run: *Run,
    seed: u32,
) error{ OutOfMemory, UncleanExit }!void {
    const starting_steps = try gpa.dupe(*Step, step_stack.keys());
    defer gpa.free(starting_steps);

    var rng = std.Random.DefaultPrng.init(seed);
    const rand = rng.random();
    rand.shuffle(*Step, starting_steps);

    for (starting_steps) |s| {
        constructGraphAndCheckForDependencyLoop(b, s, step_stack, rand) catch |err| switch (err) {
            error.DependencyLoopDetected => return uncleanExit(),
            else => |e| return e,
        };
    }

    {
        // Check that we have enough memory to complete the build.
        var any_problems = false;
        for (step_stack.keys()) |s| {
            if (s.max_rss == 0) continue;
            if (s.max_rss > run.max_rss) {
                if (run.skip_oom_steps) {
                    s.state = .skipped_oom;
                } else {
                    std.debug.print("{s}{s}: this step declares an upper bound of {d} bytes of memory, exceeding the available {d} bytes of memory\n", .{
                        s.owner.dep_prefix, s.name, s.max_rss, run.max_rss,
                    });
                    any_problems = true;
                }
            }
        }
        if (any_problems) {
            if (run.max_rss_is_default) {
                std.debug.print("note: use --maxrss to override the default", .{});
            }
            return uncleanExit();
        }
    }
}

fn runSteps(
    gpa: std.mem.Allocator,
    b: *std.Build,
    steps: []const *Step,
    parent_prog_node: ProgressNode,
    run: *Run,
) error{ OutOfMemory, UncleanExit }!void {
    const thread_pool = &run.thread_pool;

    {
        var step_prog = parent_prog_node.start("steps", steps.len);
        defer step_prog.end();

        var wait_group: std.Thread.WaitGroup = .{};
        defer wait_group.wait();

        // Here we spawn the initial set of tasks with a nice heuristic -
        // dependency order. Each worker when it finishes a step will then
        // check whether it should run any dependants.
        for (steps) |step| {
            if (step.state == .skipped_oom) continue;

            wait_group.start();
            thread_pool.spawn(workerMakeOneStep, .{
                &wait_group, b, step, if (comptime builtin.zig_version.order(std_progress_rework_version) == .lt) &step_prog else step_prog, run,
            }) catch @panic("OOM");
        }
    }
    assert(run.memory_blocked_steps.items.len == 0);

    _ = gpa;
    // TODO collect std.zig.ErrorBundle's and stderr from failed steps and send them to ZLS
}

/// Traverse the dependency graph depth-first and make it undirected by having
/// steps know their dependants (they only know dependencies at start).
/// Along the way, check that there is no dependency loop, and record the steps
/// in traversal order in `step_stack`.
/// Each step has its dependencies traversed in random order, this accomplishes
/// two things:
/// - `step_stack` will be in randomized-depth-first order, so the build runner
///   spawns steps in a random (but optimized) order
/// - each step's `dependants` list is also filled in a random order, so that
///   when it finishes executing in `workerMakeOneStep`, it spawns next steps
///   to run in random order
fn constructGraphAndCheckForDependencyLoop(
    b: *std.Build,
    s: *Step,
    step_stack: *std.AutoArrayHashMapUnmanaged(*Step, void),
    rand: std.Random,
) error{ OutOfMemory, DependencyLoopDetected }!void {
    switch (s.state) {
        .precheck_started => return error.DependencyLoopDetected,
        .precheck_unstarted => {
            s.state = .precheck_started;

            try step_stack.ensureUnusedCapacity(b.allocator, s.dependencies.items.len);

            // We dupe to avoid shuffling the steps in the summary, it depends
            // on s.dependencies' order.
            const deps = b.allocator.dupe(*Step, s.dependencies.items) catch @panic("OOM");
            rand.shuffle(*Step, deps);

            for (deps) |dep| {
                try step_stack.put(b.allocator, dep, {});
                try dep.dependants.append(b.allocator, s);
                try constructGraphAndCheckForDependencyLoop(b, dep, step_stack, rand);
            }

            s.state = .precheck_done;
        },
        .precheck_done => {},

        // These don't happen until we actually run the step graph.
        .dependency_failure,
        .running,
        .success,
        .failure,
        .skipped,
        .skipped_oom,
        => {},
    }
}

fn workerMakeOneStep(
    wg: *std.Thread.WaitGroup,
    b: *std.Build,
    s: *Step,
    prog_node: ProgressNode,
    run: *Run,
) void {
    defer wg.finish();
    const thread_pool = &run.thread_pool;

    // First, check the conditions for running this step. If they are not met,
    // then we return without doing the step, relying on another worker to
    // queue this step up again when dependencies are met.
    for (s.dependencies.items) |dep| {
        switch (@atomicLoad(Step.State, &dep.state, .seq_cst)) {
            .success, .skipped => continue,
            .failure, .dependency_failure, .skipped_oom => {
                @atomicStore(Step.State, &s.state, .dependency_failure, .seq_cst);
                return;
            },
            .precheck_done, .running => {
                // dependency is not finished yet.
                return;
            },
            .precheck_unstarted => unreachable,
            .precheck_started => unreachable,
        }
    }

    if (s.max_rss != 0) {
        run.max_rss_mutex.lock();
        defer run.max_rss_mutex.unlock();

        // Avoid running steps twice.
        if (s.state != .precheck_done) {
            // Another worker got the job.
            return;
        }

        const new_claimed_rss = run.claimed_rss + s.max_rss;
        if (new_claimed_rss > run.max_rss) {
            // Running this step right now could possibly exceed the allotted RSS.
            // Add this step to the queue of memory-blocked steps.
            run.memory_blocked_steps.append(s) catch @panic("OOM");
            return;
        }

        run.claimed_rss = new_claimed_rss;
        s.state = .running;
    } else {
        // Avoid running steps twice.
        if (@cmpxchgStrong(Step.State, &s.state, .precheck_done, .running, .seq_cst, .seq_cst) != null) {
            // Another worker got the job.
            return;
        }
    }

    var sub_prog_node = prog_node.start(s.name, 0);
    if (comptime builtin.zig_version.order(std_progress_rework_version) == .lt) sub_prog_node.activate();
    defer sub_prog_node.end();

    const make_result = s.make(
        if (comptime builtin.zig_version.order(std_progress_rework_version) == .lt)
            &sub_prog_node
        else if (comptime builtin.zig_version.order(live_rebuild_processes) == .lt)
            sub_prog_node
        else
            .{
                .progress_node = sub_prog_node,
                .thread_pool = thread_pool,
                .watch = false,
            },
    );

    handle_result: {
        if (make_result) |_| {
            @atomicStore(Step.State, &s.state, .success, .seq_cst);
        } else |err| switch (err) {
            error.MakeFailed => {
                @atomicStore(Step.State, &s.state, .failure, .seq_cst);
                break :handle_result;
            },
            error.MakeSkipped => @atomicStore(Step.State, &s.state, .skipped, .seq_cst),
        }

        // Successful completion of a step, so we queue up its dependants as well.
        for (s.dependants.items) |dep| {
            wg.start();
            thread_pool.spawn(workerMakeOneStep, .{
                wg, b, dep, prog_node, run,
            }) catch @panic("OOM");
        }
    }

    // If this is a step that claims resources, we must now queue up other
    // steps that are waiting for resources.
    if (s.max_rss != 0) {
        run.max_rss_mutex.lock();
        defer run.max_rss_mutex.unlock();

        // Give the memory back to the scheduler.
        run.claimed_rss -= s.max_rss;
        // Avoid kicking off too many tasks that we already know will not have
        // enough resources.
        var remaining = run.max_rss - run.claimed_rss;
        var i: usize = 0;
        var j: usize = 0;
        while (j < run.memory_blocked_steps.items.len) : (j += 1) {
            const dep = run.memory_blocked_steps.items[j];
            assert(dep.max_rss != 0);
            if (dep.max_rss <= remaining) {
                remaining -= dep.max_rss;

                wg.start();
                thread_pool.spawn(workerMakeOneStep, .{
                    wg, b, dep, prog_node, run,
                }) catch @panic("OOM");
            } else {
                run.memory_blocked_steps.items[i] = dep;
                i += 1;
            }
        }
        run.memory_blocked_steps.shrinkRetainingCapacity(i);
    }
}

const ArgsType = if (builtin.zig_version.order(child_type_coercion_version) == .lt)
    [][:0]const u8
else
    []const [:0]const u8;

fn nextArg(args: ArgsType, idx: *usize) ?[:0]const u8 {
    if (idx.* >= args.len) return null;
    defer idx.* += 1;
    return args[idx.*];
}

fn nextArgOrFatal(args: ArgsType, idx: *usize) [:0]const u8 {
    return nextArg(args, idx) orelse {
        std.debug.print("expected argument after '{s}'\n  access the help menu with 'zig build -h'\n", .{args[idx.* - 1]});
        process.exit(1);
    };
}

fn argsRest(args: ArgsType, idx: usize) ?ArgsType {
    if (idx >= args.len) return null;
    return args[idx..];
}

/// Perhaps in the future there could be an Advanced Options flag such as
/// --debug-build-runner-leaks which would make this function return instead of
/// calling exit.
fn cleanExit() void {
    if (comptime builtin.zig_version.order(std_progress_rework_version) != .lt) {
        std.debug.lockStdErr();
    }
    process.exit(0);
}

/// Perhaps in the future there could be an Advanced Options flag such as
/// --debug-build-runner-leaks which would make this function return instead of
/// calling exit.
fn uncleanExit() error{UncleanExit} {
    if (comptime builtin.zig_version.order(std_progress_rework_version) != .lt) {
        std.debug.lockStdErr();
    }
    process.exit(1);
}

fn fatal(comptime f: []const u8, args: anytype) noreturn {
    std.debug.print(f ++ "\n", args);
    process.exit(1);
}

fn validateSystemLibraryOptions(b: *std.Build) void {
    var bad = false;
    for (b.graph.system_library_options.keys(), b.graph.system_library_options.values()) |k, v| {
        switch (v) {
            .user_disabled, .user_enabled => {
                // The user tried to enable or disable a system library integration, but
                // the build script did not recognize that option.
                std.debug.print("system library name not recognized by build script: '{s}'\n", .{k});
                bad = true;
            },
            .declared_disabled, .declared_enabled => {},
        }
    }
    if (bad) {
        std.debug.print("  access the help menu with 'zig build -h'\n", .{});
        process.exit(1);
    }
}

/// Starting from all top-level steps in `b`, traverses the entire step graph
/// and adds all step dependencies implied by module graphs.
fn createModuleDependencies(b: *std.Build) Allocator.Error!void {
    const arena = b.graph.arena;

    var all_steps: std.AutoArrayHashMapUnmanaged(*Step, void) = .empty;
    var next_step_idx: usize = 0;

    try all_steps.ensureUnusedCapacity(arena, b.top_level_steps.count());
    for (b.top_level_steps.values()) |tls| {
        all_steps.putAssumeCapacityNoClobber(&tls.step, {});
    }

    while (next_step_idx < all_steps.count()) {
        const step = all_steps.keys()[next_step_idx];
        next_step_idx += 1;

        // Set up any implied dependencies for this step. It's important that we do this first, so
        // that the loop below discovers steps implied by the module graph.
        try createModuleDependenciesForStep(step);

        try all_steps.ensureUnusedCapacity(arena, step.dependencies.items.len);
        for (step.dependencies.items) |other_step| {
            all_steps.putAssumeCapacity(other_step, {});
        }
    }
}

/// If the given `Step` is a `Step.Compile`, adds any dependencies for that step which
/// are implied by the module graph rooted at `step.cast(Step.Compile).?.root_module`.
fn createModuleDependenciesForStep(step: *Step) Allocator.Error!void {
    const root_module = if (step.cast(Step.Compile)) |cs| root: {
        break :root cs.root_module;
    } else return; // not a compile step so no module dependencies

    // Starting from `root_module`, discover all modules in this graph.
    const modules = root_module.getGraph().modules;

    // For each of those modules, set up the implied step dependencies.
    for (modules) |mod| {
        if (mod.root_source_file) |lp| lp.addStepDependencies(step);
        for (mod.include_dirs.items) |include_dir| switch (include_dir) {
            .path,
            .path_system,
            .path_after,
            .framework_path,
            .framework_path_system,
            => |lp| lp.addStepDependencies(step),

            .other_step => |other| {
                other.getEmittedIncludeTree().addStepDependencies(step);
                step.dependOn(&other.step);
            },

            .config_header_step => |other| step.dependOn(&other.step),
        };
        for (mod.lib_paths.items) |lp| lp.addStepDependencies(step);
        for (mod.rpaths.items) |rpath| switch (rpath) {
            .lazy_path => |lp| lp.addStepDependencies(step),
            .special => {},
        };
        for (mod.link_objects.items) |link_object| switch (link_object) {
            .static_path,
            .assembly_file,
            => |lp| lp.addStepDependencies(step),
            .other_step => |other| step.dependOn(&other.step),
            .system_lib => {},
            .c_source_file => |source| source.file.addStepDependencies(step),
            .c_source_files => |source_files| source_files.root.addStepDependencies(step),
            .win32_resource_file => |rc_source| {
                rc_source.file.addStepDependencies(step);
                for (rc_source.include_paths) |lp| lp.addStepDependencies(step);
            },
        };
    }
}

//
//
// ZLS code
//
//

const BuildConfig = @import("BuildConfig.zig");

const Packages = struct {
    allocator: std.mem.Allocator,

    /// Outer key is the package name, inner key is the file path.
    packages: std.StringArrayHashMapUnmanaged(std.StringArrayHashMapUnmanaged(void)) = .{},

    /// Returns true if the package was already present.
    pub fn addPackage(self: *Packages, name: []const u8, path: []const u8) !bool {
        const name_gop_result = try self.packages.getOrPutValue(self.allocator, name, .{});
        const path_gop_result = try name_gop_result.value_ptr.getOrPut(self.allocator, path);
        return path_gop_result.found_existing;
    }

    pub fn toPackageList(self: *Packages) ![]BuildConfig.NamePathPair {
        var result: std.ArrayListUnmanaged(BuildConfig.NamePathPair) = .{};
        errdefer result.deinit(self.allocator);

        const Context = struct {
            keys: [][]const u8,

            pub fn lessThan(ctx: @This(), a_index: usize, b_index: usize) bool {
                return std.mem.lessThan(u8, ctx.keys[a_index], ctx.keys[b_index]);
            }
        };

        self.packages.sort(Context{ .keys = self.packages.keys() });

        for (self.packages.keys(), self.packages.values()) |name, path_hashmap| {
            for (path_hashmap.keys()) |path| {
                try result.append(self.allocator, .{ .name = name, .path = path });
            }
        }

        return try result.toOwnedSlice(self.allocator);
    }

    pub fn deinit(self: *Packages) void {
        for (self.packages.values()) |*path_hashmap| {
            path_hashmap.deinit(self.allocator);
        }
        self.packages.deinit(self.allocator);
    }
};

const roots_info = struct {
    pub const RootEntry = struct {
        step: *Step.Compile,
        mods: []BuildConfig.NamePathPair,
    };

    pub fn collect(
        gpa: std.mem.Allocator,
        step: *Step,
        visited_steps: *std.AutoArrayHashMapUnmanaged(*Step, void),
        unsorted_roots: *std.ArrayListUnmanaged(RootEntry),
    ) !void {
        const gop_result = try visited_steps.getOrPut(gpa, step);
        if (gop_result.found_existing) return;
        if (step.cast(Step.Compile)) |compile| {
            var root_imports: std.ArrayListUnmanaged(BuildConfig.NamePathPair) = .{};
            // std.debug.print("cstep: {s}\n", .{compile.name});

            var cli_named_modules = try copied_from_zig.CliNamedModules.init(gpa, compile.root_module);
            for (compile.getCompileDependencies(false)) |dep_compile| {
                for (dep_compile.root_module.getGraph().modules) |mod| {
                    if (!(dep_compile == compile)) continue; // !my_responsibility
                    if (cli_named_modules.modules.getIndex(mod)) |module_cli_index| {
                        const module_cli_name = cli_named_modules.names.keys()[module_cli_index];
                        if (mod.root_source_file) |lp| {
                            const src = lp.getPath2(mod.owner, step);
                            // std.log.debug("-M{s}={s}\n", .{ module_cli_name, src });
                            try root_imports.append(gpa, .{ .name = module_cli_name, .path = src });
                        }
                    }
                }
            }
            try unsorted_roots.append(
                gpa,
                .{
                    .step = compile,
                    .mods = try root_imports.toOwnedSlice(gpa),
                },
            );
            root_imports.items.len = 0; // clearRetainingCapacity();
        }
        for (step.dependencies.items) |dep_step| try collect(
            gpa,
            dep_step,
            visited_steps,
            unsorted_roots,
        );
    }

    pub fn collect_pre_zig_014_2534(
        gpa: std.mem.Allocator,
        step: *Step,
        visited_steps: *std.AutoArrayHashMapUnmanaged(*Step, void),
        unsorted_roots: *std.ArrayListUnmanaged(RootEntry),
    ) !void {
        const gop_result = try visited_steps.getOrPut(gpa, step);
        if (gop_result.found_existing) return;
        if (step.cast(Step.Compile)) |compile| {
            var root_imports: std.ArrayListUnmanaged(BuildConfig.NamePathPair) = .{};
            // std.debug.print("cstep: {s}\n", .{compile.name});

            var cli_named_modules = try copied_from_zig.CliNamedModules_Legacy.init(gpa, &compile.root_module);
            var dep_it = compile.root_module.iterateDependencies(compile, false);
            while (dep_it.next()) |dep| {
                if (!(dep.compile.? == compile)) continue; // !my_responsibility
                if (cli_named_modules.modules.getIndex(dep.module)) |module_cli_index| {
                    const module_cli_name = cli_named_modules.names.keys()[module_cli_index];
                    if (dep.module.root_source_file) |lp| {
                        const src = lp.getPath2(dep.module.owner, step);
                        // std.log.debug("-M{s}={s}\n", .{ module_cli_name, src });
                        try root_imports.append(gpa, .{ .name = module_cli_name, .path = src });
                    }
                }
            }
            try unsorted_roots.append(
                gpa,
                .{
                    .step = compile,
                    .mods = try root_imports.toOwnedSlice(gpa),
                },
            );
            root_imports.items.len = 0; // clearRetainingCapacity();
        }
        for (step.dependencies.items) |dep_step| try collect_pre_zig_014_2534(
            gpa,
            dep_step,
            visited_steps,
            unsorted_roots,
        );
    }

    pub fn hasPrecedence(dir_path: []const u8, lhs: RootEntry, rhs: RootEntry) bool {
        if (lhs.mods.len == 0) return false; // C compile steps should be last
        if (rhs.mods.len == 0) return true; //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        const lhs_dir_name = std.fs.path.dirname(lhs.mods[0].path).?; // [0] should be 'root'
        const rhs_dir_name = std.fs.path.dirname(rhs.mods[0].path).?; // [0] should be 'root'
        if (std.mem.startsWith(u8, lhs_dir_name, dir_path)) {
            return if (!std.mem.startsWith(u8, rhs_dir_name, dir_path)) true else (lhs.mods.len > rhs.mods.len);
        }
        return false;
    }

    pub fn print(
        roots_info_slc: *std.ArrayList(u8),
        idx: *u32,
        compile: *Step.Compile,
    ) !void {
        if (compile.root_module.root_source_file) |root_source_file| {
            try roots_info_slc.writer().print(
                "{}: {s} @ {s}\n",
                .{ idx.*, compile.name, root_source_file.getPath(compile.root_module.owner) },
            );
        }
        try printIt(
            roots_info_slc,
            compile.root_module.import_table,
        );
        idx.* += 1;
    }

    pub fn printIt(
        roots_info_slc: *std.ArrayList(u8),
        it: std.StringArrayHashMapUnmanaged(*std.Build.Module),
    ) !void {
        for (it.keys(), it.values()) |name, import| {
            if (import.root_source_file) |root_source_file| {
                try roots_info_slc.writer().print(
                    "   * {s} @ {s}\n",
                    .{ name, root_source_file.getPath(import.owner) },
                );
            }
            for (import.import_table.keys(), import.import_table.values()) |name2, import2| {
                if (import2.root_source_file) |root_source_file2| {
                    try roots_info_slc.writer().print(
                        "     * {s} @ {s}\n",
                        .{ name2, root_source_file2.getPath(import.owner) },
                    );
                }
                for (import2.import_table.keys(), import2.import_table.values()) |name3, import3| {
                    if (import3.root_source_file) |root_source_file3| {
                        try roots_info_slc.writer().print(
                            "       * {s} @ {s}\n",
                            .{ name3, root_source_file3.getPath(import.owner) },
                        );
                    }
                }
            }
        }
    }
};

fn extractBuildInformation(
    gpa: Allocator,
    b: *std.Build,
    arena: Allocator,
    main_progress_node: ProgressNode,
    run: *Run,
    seed: u32,
) !void {
    var steps = std.AutoArrayHashMapUnmanaged(*Step, void){};
    defer steps.deinit(gpa);

    // collect the set of all steps
    {
        var stack: std.ArrayListUnmanaged(*Step) = .{};
        defer stack.deinit(gpa);

        try stack.ensureUnusedCapacity(gpa, b.top_level_steps.count());
        for (b.top_level_steps.values()) |tls| {
            stack.appendAssumeCapacity(&tls.step);
        }

        while (stack.popOrNull()) |step| {
            const gop = try steps.getOrPut(gpa, step);
            if (gop.found_existing) continue;

            try stack.appendSlice(gpa, step.dependencies.items);
        }
    }

    const helper = struct {
        fn addStepDependencies(allocator: Allocator, set: *std.AutoArrayHashMapUnmanaged(*Step, void), lazy_path: std.Build.LazyPath) !void {
            switch (lazy_path) {
                .src_path, .cwd_relative, .dependency => {},
                .generated => |gen| try set.put(allocator, gen.file.step, {}),
            }
        }

        fn addModuleDependencies(allocator: Allocator, set: *std.AutoArrayHashMapUnmanaged(*Step, void), module: *std.Build.Module) !void {
            if (module.root_source_file) |root_source_file| {
                try addStepDependencies(allocator, set, root_source_file);
            }

            for (module.lib_paths.items) |lib_path| {
                try addStepDependencies(allocator, set, lib_path);
            }

            for (module.rpaths.items) |rpath| {
                if (rpath == .lazy_path) try addStepDependencies(allocator, set, rpath.lazy_path);
            }

            for (module.import_table.values()) |import| {
                if (import.root_source_file) |root_source_file| {
                    try addStepDependencies(allocator, set, root_source_file);
                }
            }

            for (module.include_dirs.items) |include_dir| {
                switch (include_dir) {
                    .path,
                    .path_system,
                    .path_after,
                    .framework_path,
                    .framework_path_system,
                    => |include_path| try addStepDependencies(allocator, set, include_path),
                    .config_header_step => |config_header| try set.put(allocator, config_header.output_file.step, {}),
                    .other_step => |other| {
                        if (other.generated_h) |header| {
                            try set.put(allocator, header.step, {});
                        }
                        if (other.installed_headers_include_tree) |include_tree| {
                            try set.put(allocator, include_tree.generated_directory.step, {});
                        }
                    },
                }
            }
        }

        fn processItem(
            allocator: Allocator,
            module: *std.Build.Module,
            compile: ?*std.Build.Step.Compile,
            name: []const u8,
            packages: *Packages,
            include_dirs: *std.StringArrayHashMapUnmanaged(void),
        ) !void {
            if (module.root_source_file) |root_source_file| {
                _ = try packages.addPackage(name, root_source_file.getPath(module.owner));
            }

            if (compile) |exe| {
                try processPkgConfig(allocator, include_dirs, exe);
            }

            for (module.include_dirs.items) |include_dir| {
                switch (include_dir) {
                    .path,
                    .path_system,
                    .path_after,
                    .framework_path,
                    .framework_path_system,
                    => |include_path| try include_dirs.put(allocator, include_path.getPath(module.owner), {}),

                    .other_step => |other| {
                        if (other.generated_h) |header| {
                            try include_dirs.put(
                                allocator,
                                std.fs.path.dirname(header.getPath()).?,
                                {},
                            );
                        }
                        if (other.installed_headers_include_tree) |include_tree| {
                            try include_dirs.put(
                                allocator,
                                include_tree.generated_directory.getPath(),
                                {},
                            );
                        }
                    },
                    .config_header_step => |config_header| {
                        const full_file_path = config_header.output_file.getPath();
                        const header_dir_path = full_file_path[0 .. full_file_path.len - config_header.include_path.len];
                        try include_dirs.put(
                            allocator,
                            header_dir_path,
                            {},
                        );
                    },
                }
            }
        }
    };

    var step_dependencies: std.AutoArrayHashMapUnmanaged(*Step, void) = .{};
    defer step_dependencies.deinit(gpa);

    const DependencyItem = struct {
        compile: ?*std.Build.Step.Compile,
        module: *std.Build.Module,
    };

    var dependency_set: std.AutoArrayHashMapUnmanaged(DependencyItem, []const u8) = .{};
    defer dependency_set.deinit(gpa);

    if (comptime builtin.zig_version.order(accept_root_module_version) != .lt) {
        var modules: std.AutoArrayHashMapUnmanaged(*std.Build.Module, void) = .{};
        // defer modules.deinit(gpa);

        // collect root modules of `Step.Compile`
        for (steps.keys()) |step| {
            const compile = step.cast(Step.Compile) orelse continue;
            const graph = compile.root_module.getGraph();
            try modules.ensureUnusedCapacity(gpa, graph.modules.len);
            for (graph.modules) |module| modules.putAssumeCapacity(module, {});
        }

        // collect public modules
        for (b.modules.values()) |root_module| {
            const graph = root_module.getGraph();
            try modules.ensureUnusedCapacity(gpa, graph.modules.len);
            for (graph.modules) |module| modules.putAssumeCapacity(module, {});
        }

        // collect all dependencies of all found modules
        for (modules.keys()) |module| {
            try helper.addModuleDependencies(gpa, &step_dependencies, module);
        }
    } else {
        var dependency_iterator: std.Build.Module.DependencyIterator = .{
            .allocator = gpa,
            .index = 0,
            .set = .{},
            .chase_dyn_libs = true,
        };
        defer dependency_iterator.deinit();

        // collect root modules of `Step.Compile`
        for (steps.keys()) |step| {
            const compile = step.cast(Step.Compile) orelse continue;

            dependency_iterator.set.ensureUnusedCapacity(arena, compile.root_module.import_table.count() + 1) catch @panic("OOM");
            dependency_iterator.set.putAssumeCapacity(.{
                .module = &compile.root_module,
                .compile = compile,
            }, "root");
        }

        // collect public modules
        for (b.modules.values()) |module| {
            dependency_iterator.set.ensureUnusedCapacity(gpa, module.import_table.count() + 1) catch @panic("OOM");
            dependency_iterator.set.putAssumeCapacity(.{
                .module = module,
                .compile = null,
            }, "root");
        }

        var dependency_items: std.ArrayListUnmanaged(std.Build.Module.DependencyIterator.Item) = .{};
        defer dependency_items.deinit(gpa);

        // collect all dependencies
        while (dependency_iterator.next()) |item| {
            try helper.addModuleDependencies(gpa, &step_dependencies, item.module);
            _ = try dependency_set.fetchPut(gpa, .{
                .module = item.module,
                .compile = item.compile,
            }, item.name);
        }
    }

    prepare(gpa, b, &step_dependencies, run, seed) catch |err| switch (err) {
        error.UncleanExit => process.exit(1),
        else => return err,
    };

    // run all steps that are dependencies
    try runSteps(
        gpa,
        b,
        step_dependencies.keys(),
        main_progress_node,
        run,
    );

    var include_dirs: std.StringArrayHashMapUnmanaged(void) = .{};
    var packages: Packages = .{ .allocator = gpa };
    defer packages.deinit();

    // extract packages and include paths
    if (comptime builtin.zig_version.order(accept_root_module_version) == .lt) {
        for (dependency_set.keys(), dependency_set.values()) |item, name| {
            try helper.processItem(gpa, item.module, item.compile, name, &packages, &include_dirs);
            for (item.module.import_table.keys(), item.module.import_table.values()) |import_name, import| {
                if (import.root_source_file) |root_source_file| {
                    _ = try packages.addPackage(import_name, root_source_file.getPath(item.module.owner));
                }
            }
        }
    } else {
        for (steps.keys()) |step| {
            const compile = step.cast(Step.Compile) orelse continue;
            const graph = compile.root_module.getGraph();
            try helper.processItem(gpa, compile.root_module, compile, "root", &packages, &include_dirs);
            for (graph.modules) |module| {
                for (module.import_table.keys(), module.import_table.values()) |name, import| {
                    try helper.processItem(gpa, import, null, name, &packages, &include_dirs);
                }
            }
        }

        for (b.modules.values()) |root_module| {
            const graph = root_module.getGraph();
            try helper.processItem(gpa, root_module, null, "root", &packages, &include_dirs);
            for (graph.modules) |module| {
                for (module.import_table.keys(), module.import_table.values()) |name, import| {
                    try helper.processItem(gpa, import, null, name, &packages, &include_dirs);
                }
            }
        }
    }

    // Sample `@dependencies` structure:
    // pub const packages = struct {
    //     pub const @"1220363c7e27b2d3f39de6ff6e90f9537a0634199860fea237a55ddb1e1717f5d6a5" = struct {
    //         pub const build_root = "/home/rad/.cache/zig/p/1220363c7e27b2d3f39de6ff6e90f9537a0634199860fea237a55ddb1e1717f5d6a5";
    //         pub const build_zig = @import("1220363c7e27b2d3f39de6ff6e90f9537a0634199860fea237a55ddb1e1717f5d6a5");
    //         pub const deps: []const struct { []const u8, []const u8 } = &.{};
    //     };
    // ...
    // };
    // pub const root_deps: []const struct { []const u8, []const u8 } = &.{
    //     .{ "known_folders", "1220bb12c9bfe291eed1afe6a2070c7c39918ab1979f24a281bba39dfb23f5bcd544" },
    //     .{ "diffz", "122089a8247a693cad53beb161bde6c30f71376cd4298798d45b32740c3581405864" },
    // };

    var deps_build_roots: std.ArrayListUnmanaged(BuildConfig.NamePathPair) = .{};
    for (dependencies.root_deps) |root_dep| {
        inline for (comptime std.meta.declarations(dependencies.packages)) |package| blk: {
            if (std.mem.eql(u8, package.name, root_dep[1])) {
                const package_info = @field(dependencies.packages, package.name);
                if (!@hasDecl(package_info, "build_root")) break :blk;
                if (!@hasDecl(package_info, "build_zig")) break :blk;
                try deps_build_roots.append(arena, .{
                    .name = root_dep[0],
                    .path = try std.fs.path.join(arena, &.{ package_info.build_root, "build.zig" }),
                });
            }
        }
    }

    var available_options: std.json.ArrayHashMap(BuildConfig.AvailableOption) = .{};
    try available_options.map.ensureTotalCapacity(arena, b.available_options_map.count());

    var it = b.available_options_map.iterator();
    while (it.next()) |available_option| {
        available_options.map.putAssumeCapacityNoClobber(available_option.key_ptr.*, available_option.value_ptr.*);
    }

    // roots[]
    var visited_steps: std.AutoArrayHashMapUnmanaged(*Step, void) = .{};
    var unsorted_roots: std.ArrayListUnmanaged(roots_info.RootEntry) = .{};
    var roots_info_slc = std.ArrayList(u8).init(gpa);
    var root_idx: u32 = 0;

    if (comptime builtin.zig_version.order(accept_root_module_version) != .lt) {
        for (b.top_level_steps.values()) |tls| {
            try roots_info.collect(
                gpa,
                &tls.step,
                &visited_steps,
                &unsorted_roots,
            );
        }
    } else {
        for (b.top_level_steps.values()) |tls| {
            try roots_info.collect_pre_zig_014_2534(
                gpa,
                &tls.step,
                &visited_steps,
                &unsorted_roots,
            );
        }
    }

    std.mem.sort(roots_info.RootEntry, unsorted_roots.items, build_root, roots_info.hasPrecedence);

    var roots = try std.ArrayListUnmanaged(BuildConfig.RootEntry).initCapacity(gpa, unsorted_roots.items.len);
    for (unsorted_roots.items) |item| {
        roots.appendAssumeCapacity(.{
            .name = item.step.name,
            .args = try getZigArgs(item.step, false),
            .mods = item.mods,
        });
        try roots_info.print(&roots_info_slc, &root_idx, item.step);
    }

    const dir_path = std.fs.path.dirname(self_path) orelse unreachable;
    const file_path = try std.fs.path.join(gpa, &.{ dir_path, "roots.txt" });
    const file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();
    try file.writeAll(roots_info_slc.items);

    try std.json.stringify(
        BuildConfig{
            .roots_info_file = file_path,
            .deps_build_roots = deps_build_roots.items,
            .roots = roots.items,
            .packages = try packages.toPackageList(),
            .include_dirs = include_dirs.keys(),
            .top_level_steps = b.top_level_steps.keys(),
            .available_options = available_options,
        },
        .{
            .whitespace = .indent_2,
        },
        std.io.getStdOut().writer(),
    );
}

fn processPkgConfig(
    allocator: std.mem.Allocator,
    include_dirs: *std.StringArrayHashMapUnmanaged(void),
    exe: *Step.Compile,
) !void {
    for (exe.root_module.link_objects.items) |link_object| {
        if (link_object != .system_lib) continue;
        const system_lib = link_object.system_lib;

        if (system_lib.use_pkg_config == .no) continue;

        getPkgConfigIncludes(allocator, include_dirs, exe, system_lib.name) catch |err| switch (err) {
            error.PkgConfigInvalidOutput,
            error.PkgConfigCrashed,
            error.PkgConfigFailed,
            error.PkgConfigNotInstalled,
            error.PackageNotFound,
            => switch (system_lib.use_pkg_config) {
                .yes => {
                    // pkg-config failed, so zig will not add any include paths
                },
                .force => {
                    std.log.warn("pkg-config failed for library {s}", .{system_lib.name});
                },
                .no => unreachable,
            },
            else => |e| return e,
        };
    }
}

fn getPkgConfigIncludes(
    allocator: std.mem.Allocator,
    include_dirs: *std.StringArrayHashMapUnmanaged(void),
    exe: *Step.Compile,
    name: []const u8,
) !void {
    if (copied_from_zig.runPkgConfig(exe, name)) |args| {
        for (args) |arg| {
            if (std.mem.startsWith(u8, arg, "-I")) {
                const candidate = arg[2..];
                try include_dirs.put(allocator, candidate, {});
            }
        }
    } else |err| return err;
}

// TODO: Having a copy of this is not very nice
const copied_from_zig = struct {
    // Gotten from std.Build.Step.Compile
    const CliNamedModules = struct {
        modules: std.AutoArrayHashMapUnmanaged(*Module, void),
        names: std.StringArrayHashMapUnmanaged(void),

        /// Traverse the whole dependency graph and give every module a unique
        /// name, ideally one named after what it's called somewhere in the graph.
        /// It will help here to have both a mapping from module to name and a set
        /// of all the currently-used names.
        fn init(arena: Allocator, root_module: *Module) Allocator.Error!CliNamedModules {
            var compile: CliNamedModules = .{
                .modules = .{},
                .names = .{},
            };
            const graph = root_module.getGraph();

            {
                assert(graph.modules[0] == root_module);
                try compile.modules.put(arena, root_module, {});
                try compile.names.put(arena, "root", {});
            }
            for (graph.modules[1..], graph.names[1..]) |mod, orig_name| {
                var name = orig_name;
                var n: usize = 0;
                while (true) {
                    const gop = try compile.names.getOrPut(arena, name);
                    if (!gop.found_existing) {
                        try compile.modules.putNoClobber(arena, mod, {});
                        break;
                    }
                    name = try std.fmt.allocPrint(arena, "{s}{d}", .{ orig_name, n });
                    n += 1;
                }
            }
            return compile;
        }
    };

    const CliNamedModules_Legacy = struct {
        modules: std.AutoArrayHashMapUnmanaged(*Module, void),
        names: std.StringArrayHashMapUnmanaged(void),

        /// Traverse the whole dependency graph and give every module a unique
        /// name, ideally one named after what it's called somewhere in the graph.
        /// It will help here to have both a mapping from module to name and a set
        /// of all the currently-used names.
        fn init(arena: Allocator, root_module: *Module) Allocator.Error!CliNamedModules {
            var compile: CliNamedModules = .{
                .modules = .{},
                .names = .{},
            };
            var dep_it = root_module.iterateDependencies(null, false);
            {
                const item = dep_it.next().?;
                assert(root_module == item.module);
                try compile.modules.put(arena, root_module, {});
                try compile.names.put(arena, "root", {});
            }
            while (dep_it.next()) |item| {
                var name = item.name;
                var n: usize = 0;
                while (true) {
                    const gop = try compile.names.getOrPut(arena, name);
                    if (!gop.found_existing) {
                        try compile.modules.putNoClobber(arena, item.module, {});
                        break;
                    }
                    name = try std.fmt.allocPrint(arena, "{s}{d}", .{ item.name, n });
                    n += 1;
                }
            }
            return compile;
        }
    };

    /// Run pkg-config for the given library name and parse the output, returning the arguments
    /// that should be passed to zig to link the given library.
    fn runPkgConfig(self: *Step.Compile, lib_name: []const u8) ![]const []const u8 {
        const b = self.step.owner;
        const pkg_name = match: {
            // First we have to map the library name to pkg config name. Unfortunately,
            // there are several examples where this is not straightforward:
            // -lSDL2 -> pkg-config sdl2
            // -lgdk-3 -> pkg-config gdk-3.0
            // -latk-1.0 -> pkg-config atk
            const pkgs = try getPkgConfigList(b);

            // Exact match means instant winner.
            for (pkgs) |pkg| {
                if (mem.eql(u8, pkg.name, lib_name)) {
                    break :match pkg.name;
                }
            }

            // Next we'll try ignoring case.
            for (pkgs) |pkg| {
                if (std.ascii.eqlIgnoreCase(pkg.name, lib_name)) {
                    break :match pkg.name;
                }
            }

            // Now try appending ".0".
            for (pkgs) |pkg| {
                if (std.ascii.indexOfIgnoreCase(pkg.name, lib_name)) |pos| {
                    if (pos != 0) continue;
                    if (mem.eql(u8, pkg.name[lib_name.len..], ".0")) {
                        break :match pkg.name;
                    }
                }
            }

            // Trimming "-1.0".
            if (mem.endsWith(u8, lib_name, "-1.0")) {
                const trimmed_lib_name = lib_name[0 .. lib_name.len - "-1.0".len];
                for (pkgs) |pkg| {
                    if (std.ascii.eqlIgnoreCase(pkg.name, trimmed_lib_name)) {
                        break :match pkg.name;
                    }
                }
            }

            return error.PackageNotFound;
        };

        var code: u8 = undefined;
        const stdout = if (b.runAllowFail(&[_][]const u8{
            "pkg-config",
            pkg_name,
            "--cflags",
            "--libs",
        }, &code, .Ignore)) |stdout| stdout else |err| switch (err) {
            error.ProcessTerminated => return error.PkgConfigCrashed,
            error.ExecNotSupported => return error.PkgConfigFailed,
            error.ExitCodeFailure => return error.PkgConfigFailed,
            error.FileNotFound => return error.PkgConfigNotInstalled,
            else => return err,
        };

        var zig_args = ArrayList([]const u8).init(b.allocator);
        defer zig_args.deinit();

        var it = mem.tokenizeAny(u8, stdout, " \r\n\t");
        while (it.next()) |tok| {
            if (mem.eql(u8, tok, "-I")) {
                const dir = it.next() orelse return error.PkgConfigInvalidOutput;
                try zig_args.appendSlice(&[_][]const u8{ "-I", dir });
            } else if (mem.startsWith(u8, tok, "-I")) {
                try zig_args.append(tok);
            } else if (mem.eql(u8, tok, "-L")) {
                const dir = it.next() orelse return error.PkgConfigInvalidOutput;
                try zig_args.appendSlice(&[_][]const u8{ "-L", dir });
            } else if (mem.startsWith(u8, tok, "-L")) {
                try zig_args.append(tok);
            } else if (mem.eql(u8, tok, "-l")) {
                const lib = it.next() orelse return error.PkgConfigInvalidOutput;
                try zig_args.appendSlice(&[_][]const u8{ "-l", lib });
            } else if (mem.startsWith(u8, tok, "-l")) {
                try zig_args.append(tok);
            } else if (mem.eql(u8, tok, "-D")) {
                const macro = it.next() orelse return error.PkgConfigInvalidOutput;
                try zig_args.appendSlice(&[_][]const u8{ "-D", macro });
            } else if (mem.startsWith(u8, tok, "-D")) {
                try zig_args.append(tok);
            } else if (b.debug_pkg_config) {
                return self.step.fail("unknown pkg-config flag '{s}'", .{tok});
            }
        }

        return zig_args.toOwnedSlice();
    }

    fn execPkgConfigList(self: *std.Build, out_code: *u8) (std.Build.PkgConfigError || std.Build.RunError)![]const std.Build.PkgConfigPkg {
        const stdout = try self.runAllowFail(&[_][]const u8{ "pkg-config", "--list-all" }, out_code, .Ignore);
        var list = ArrayList(std.Build.PkgConfigPkg).init(self.allocator);
        errdefer list.deinit();
        var line_it = mem.tokenizeAny(u8, stdout, "\r\n");
        while (line_it.next()) |line| {
            if (mem.trim(u8, line, " \t").len == 0) continue;
            var tok_it = mem.tokenizeAny(u8, line, " \t");
            try list.append(std.Build.PkgConfigPkg{
                .name = tok_it.next() orelse return error.PkgConfigInvalidOutput,
                .desc = tok_it.rest(),
            });
        }
        return list.toOwnedSlice();
    }

    fn getPkgConfigList(self: *std.Build) ![]const std.Build.PkgConfigPkg {
        if (self.pkg_config_pkg_list) |res| {
            return res;
        }
        var code: u8 = undefined;
        if (execPkgConfigList(self, &code)) |list| {
            self.pkg_config_pkg_list = list;
            return list;
        } else |err| {
            const result = switch (err) {
                error.ProcessTerminated => error.PkgConfigCrashed,
                error.ExecNotSupported => error.PkgConfigFailed,
                error.ExitCodeFailure => error.PkgConfigFailed,
                error.FileNotFound => error.PkgConfigNotInstalled,
                error.InvalidName => error.PkgConfigNotInstalled,
                error.PkgConfigInvalidOutput => error.PkgConfigInvalidOutput,
                else => return err,
            };
            self.pkg_config_pkg_list = result;
            return result;
        }
    }
};

fn getZigArgs(compile: *std.Build.Step.Compile, fuzz: bool) ![][]const u8 {
    const step = &compile.step;
    const b = step.owner;
    const arena = b.allocator;

    var zig_args = ArrayList([]const u8).init(arena);
    defer zig_args.deinit();

    try zig_args.append(b.graph.zig_exe);

    const cmd = switch (compile.kind) {
        .lib => "build-lib",
        .exe => "build-exe",
        .obj => "build-obj",
        .@"test" => "test",
    };
    try zig_args.append(cmd);

    if (b.reference_trace) |some| {
        try zig_args.append(try std.fmt.allocPrint(arena, "-freference-trace={d}", .{some}));
    }
    try addFlag(&zig_args, "allow-so-scripts", compile.allow_so_scripts orelse b.graph.allow_so_scripts);

    try addFlag(&zig_args, "llvm", compile.use_llvm);
    try addFlag(&zig_args, "lld", compile.use_lld);

    if (compile.root_module.resolved_target.?.query.ofmt) |ofmt| {
        try zig_args.append(try std.fmt.allocPrint(arena, "-ofmt={s}", .{@tagName(ofmt)}));
    }

    switch (compile.entry) {
        .default => {},
        .disabled => try zig_args.append("-fno-entry"),
        .enabled => try zig_args.append("-fentry"),
        .symbol_name => |entry_name| {
            try zig_args.append(try std.fmt.allocPrint(arena, "-fentry={s}", .{entry_name}));
        },
    }

    {
        var symbol_it = compile.force_undefined_symbols.keyIterator();
        while (symbol_it.next()) |symbol_name| {
            try zig_args.append("--force_undefined");
            try zig_args.append(symbol_name.*);
        }
    }

    if (compile.stack_size) |stack_size| {
        try zig_args.append("--stack");
        try zig_args.append(try std.fmt.allocPrint(arena, "{}", .{stack_size}));
    }

    if (fuzz) {
        try zig_args.append("-ffuzz");
    }

    {
        // Stores system libraries that have already been seen for at least one
        // module, along with any arguments that need to be passed to the
        // compiler for each module individually.
        // var seen_system_libs: std.StringHashMapUnmanaged([]const []const u8) = .empty;
        // var frameworks: std.StringArrayHashMapUnmanaged(Module.LinkFrameworkOptions) = .empty;

        // var prev_has_cflags = false;
        // var prev_has_rcflags = false;
        // var prev_search_strategy: Module.SystemLib.SearchStrategy = .paths_first;
        // var prev_preferred_link_mode: std.builtin.LinkMode = .dynamic;
        // Track the number of positional arguments so that a nice error can be
        // emitted if there is nothing to link.
        // var total_linker_objects: usize = @intFromBool(compile.root_module.root_source_file != null);

        // Fully recursive iteration including dynamic libraries to detect
        // libc and libc++ linkage.
        for (compile.getCompileDependencies(true)) |some_compile| {
            for (some_compile.root_module.getGraph().modules) |mod| {
                if (mod.link_libc == true) compile.is_linking_libc = true;
                if (mod.link_libcpp == true) compile.is_linking_libcpp = true;
            }
        }

        var cli_named_modules = try copied_from_zig.CliNamedModules.init(arena, compile.root_module);

        // For this loop, don't chase dynamic libraries because their link
        // objects are already linked.
        // var dep_it = compile.root_module.iterateDependencies(compile, false);

        for (compile.getCompileDependencies(false)) |dep_compile| {
            for (dep_compile.root_module.getGraph().modules) |mod| {
                // While walking transitive dependencies, if a given link object is
                // already included in a library, it should not redundantly be
                // placed on the linker line of the dependee.
                const my_responsibility = dep_compile == compile;
                // const already_linked = !my_responsibility and dep_compile.isDynamicLibrary();

                // Inherit dependencies on darwin frameworks.
                // if (!already_linked) {
                //     for (dep.module.frameworks.keys(), dep.module.frameworks.values()) |name, info| {
                //         try frameworks.put(arena, name, info);
                //     }
                // }

                // Inherit dependencies on system libraries and static libraries.
                // for (dep.module.link_objects.items) |link_object| {
                //     switch (link_object) {
                //         .static_path => |static_path| {
                //             if (my_responsibility) {
                //                 try zig_args.append(static_path.getPath2(dep.module.owner, step));
                //                 total_linker_objects += 1;
                //             }
                //         },
                //         // .system_lib => |system_lib| {
                //         //     const system_lib_gop = try seen_system_libs.getOrPut(arena, system_lib.name);
                //         //     if (system_lib_gop.found_existing) {
                //         //         try zig_args.appendSlice(system_lib_gop.value_ptr.*);
                //         //         continue;
                //         //     } else {
                //         //         system_lib_gop.value_ptr.* = &.{};
                //         //     }

                //         //     if (already_linked)
                //         //         continue;

                //         //     if ((system_lib.search_strategy != prev_search_strategy or
                //         //         system_lib.preferred_link_mode != prev_preferred_link_mode) and
                //         //         compile.linkage != .static)
                //         //     {
                //         //         switch (system_lib.search_strategy) {
                //         //             .no_fallback => switch (system_lib.preferred_link_mode) {
                //         //                 .dynamic => try zig_args.append("-search_dylibs_only"),
                //         //                 .static => try zig_args.append("-search_static_only"),
                //         //             },
                //         //             .paths_first => switch (system_lib.preferred_link_mode) {
                //         //                 .dynamic => try zig_args.append("-search_paths_first"),
                //         //                 .static => try zig_args.append("-search_paths_first_static"),
                //         //             },
                //         //             .mode_first => switch (system_lib.preferred_link_mode) {
                //         //                 .dynamic => try zig_args.append("-search_dylibs_first"),
                //         //                 .static => try zig_args.append("-search_static_first"),
                //         //             },
                //         //         }
                //         //         prev_search_strategy = system_lib.search_strategy;
                //         //         prev_preferred_link_mode = system_lib.preferred_link_mode;
                //         //     }

                //         //     const prefix: []const u8 = prefix: {
                //         //         if (system_lib.needed) break :prefix "-needed-l";
                //         //         if (system_lib.weak) break :prefix "-weak-l";
                //         //         break :prefix "-l";
                //         //     };
                //         //     switch (system_lib.use_pkg_config) {
                //         //         .no => try zig_args.append(b.fmt("{s}{s}", .{ prefix, system_lib.name })),
                //         //         .yes, .force => {
                //         //             if (compile.runPkgConfig(system_lib.name)) |result| {
                //         //                 try zig_args.appendSlice(result.cflags);
                //         //                 try zig_args.appendSlice(result.libs);
                //         //                 try seen_system_libs.put(arena, system_lib.name, result.cflags);
                //         //             } else |err| switch (err) {
                //         //                 error.PkgConfigInvalidOutput,
                //         //                 error.PkgConfigCrashed,
                //         //                 error.PkgConfigFailed,
                //         //                 error.PkgConfigNotInstalled,
                //         //                 error.PackageNotFound,
                //         //                 => switch (system_lib.use_pkg_config) {
                //         //                     .yes => {
                //         //                         // pkg-config failed, so fall back to linking the library
                //         //                         // by name directly.
                //         //                         try zig_args.append(b.fmt("{s}{s}", .{
                //         //                             prefix,
                //         //                             system_lib.name,
                //         //                         }));
                //         //                     },
                //         //                     .force => {
                //         //                         std.debug.panic("pkg-config failed for library {s}", .{system_lib.name});
                //         //                     },
                //         //                     .no => unreachable,
                //         //                 },

                //         //                 else => |e| return e,
                //         //             }
                //         //         },
                //         //     }
                //         // },
                //         // .other_step => |other| {
                //         //     switch (other.kind) {
                //         //         .exe => return step.fail("cannot link with an executable build artifact", .{}),
                //         //         .@"test" => return step.fail("cannot link with a test", .{}),
                //         //         .obj => {
                //         //             const included_in_lib_or_obj = !my_responsibility and
                //         //                 (dep.compile.?.kind == .lib or dep.compile.?.kind == .obj);
                //         //             if (!already_linked and !included_in_lib_or_obj) {
                //         //                 try zig_args.append(other.getEmittedBin().getPath2(b, step));
                //         //                 total_linker_objects += 1;
                //         //             }
                //         //         },
                //         //         .lib => l: {
                //         //             const other_produces_implib = other.producesImplib();
                //         //             const other_is_static = other_produces_implib or other.isStaticLibrary();

                //         //             if (compile.isStaticLibrary() and other_is_static) {
                //         //                 // Avoid putting a static library inside a static library.
                //         //                 break :l;
                //         //             }

                //         //             // For DLLs, we must link against the implib.
                //         //             // For everything else, we directly link
                //         //             // against the library file.
                //         //             const full_path_lib = if (other_produces_implib)
                //         //                 other.getGeneratedFilePath("generated_implib", &compile.step)
                //         //             else
                //         //                 other.getGeneratedFilePath("generated_bin", &compile.step);

                //         //             try zig_args.append(full_path_lib);
                //         //             total_linker_objects += 1;

                //         //             if (other.linkage == .dynamic and
                //         //                 compile.rootModuleTarget().os.tag != .windows)
                //         //             {
                //         //                 if (std.fs.path.dirname(full_path_lib)) |dirname| {
                //         //                     try zig_args.append("-rpath");
                //         //                     try zig_args.append(dirname);
                //         //                 }
                //         //             }
                //         //         },
                //         //     }
                //         // },
                //         .assembly_file => |asm_file| l: {
                //             if (!my_responsibility) break :l;

                //             if (prev_has_cflags) {
                //                 try zig_args.append("-cflags");
                //                 try zig_args.append("--");
                //                 prev_has_cflags = false;
                //             }
                //             try zig_args.append(asm_file.getPath2(dep.module.owner, step));
                //             total_linker_objects += 1;
                //         },

                //         .c_source_file => |c_source_file| l: {
                //             if (!my_responsibility) break :l;

                //             if (c_source_file.flags.len == 0) {
                //                 if (prev_has_cflags) {
                //                     try zig_args.append("-cflags");
                //                     try zig_args.append("--");
                //                     prev_has_cflags = false;
                //                 }
                //             } else {
                //                 try zig_args.append("-cflags");
                //                 for (c_source_file.flags) |arg| {
                //                     try zig_args.append(arg);
                //                 }
                //                 try zig_args.append("--");
                //                 prev_has_cflags = true;
                //             }
                //             try zig_args.append(c_source_file.file.getPath2(dep.module.owner, step));
                //             total_linker_objects += 1;
                //         },

                //         .c_source_files => |c_source_files| l: {
                //             if (!my_responsibility) break :l;

                //             if (c_source_files.flags.len == 0) {
                //                 if (prev_has_cflags) {
                //                     try zig_args.append("-cflags");
                //                     try zig_args.append("--");
                //                     prev_has_cflags = false;
                //                 }
                //             } else {
                //                 try zig_args.append("-cflags");
                //                 for (c_source_files.flags) |flag| {
                //                     try zig_args.append(flag);
                //                 }
                //                 try zig_args.append("--");
                //                 prev_has_cflags = true;
                //             }

                //             const root_path = c_source_files.root.getPath2(dep.module.owner, step);
                //             for (c_source_files.files) |file| {
                //                 try zig_args.append(b.pathJoin(&.{ root_path, file }));
                //             }

                //             total_linker_objects += c_source_files.files.len;
                //         },

                //         .win32_resource_file => |rc_source_file| l: {
                //             if (!my_responsibility) break :l;

                //             if (rc_source_file.flags.len == 0 and rc_source_file.include_paths.len == 0) {
                //                 if (prev_has_rcflags) {
                //                     try zig_args.append("-rcflags");
                //                     try zig_args.append("--");
                //                     prev_has_rcflags = false;
                //                 }
                //             } else {
                //                 try zig_args.append("-rcflags");
                //                 for (rc_source_file.flags) |arg| {
                //                     try zig_args.append(arg);
                //                 }
                //                 for (rc_source_file.include_paths) |include_path| {
                //                     try zig_args.append("/I");
                //                     try zig_args.append(include_path.getPath2(dep.module.owner, step));
                //                 }
                //                 try zig_args.append("--");
                //                 prev_has_rcflags = true;
                //             }
                //             try zig_args.append(rc_source_file.file.getPath2(dep.module.owner, step));
                //             total_linker_objects += 1;
                //         },
                //         else => {},
                //     }
                // }

                // We need to emit the --mod argument here so that the above link objects
                // have the correct parent module, but only if the module is part of
                // this compilation.
                if (!my_responsibility) continue;
                if (cli_named_modules.modules.getIndex(mod)) |module_cli_index| {
                    const module_cli_name = cli_named_modules.names.keys()[module_cli_index];
                    try mod.appendZigProcessFlags(&zig_args, step);

                    // --dep arguments
                    try zig_args.ensureUnusedCapacity(mod.import_table.count() * 2);
                    for (mod.import_table.keys(), mod.import_table.values()) |name, import| {
                        const import_index = cli_named_modules.modules.getIndex(import).?;
                        const import_cli_name = cli_named_modules.names.keys()[import_index];
                        zig_args.appendAssumeCapacity("--dep");
                        if (std.mem.eql(u8, import_cli_name, name)) {
                            zig_args.appendAssumeCapacity(import_cli_name);
                        } else {
                            zig_args.appendAssumeCapacity(b.fmt("{s}={s}", .{ name, import_cli_name }));
                        }
                    }

                    // When the CLI sees a -M argument, it determines whether it
                    // implies the existence of a Zig compilation unit based on
                    // whether there is a root source file. If there is no root
                    // source file, then this is not a zig compilation unit - it is
                    // perhaps a set of linker objects, or C source files instead.
                    // Linker objects are added to the CLI globally, while C source
                    // files must have a module parent.
                    if (mod.root_source_file) |lp| {
                        const src = lp.getPath2(mod.owner, step);
                        try zig_args.append(b.fmt("-M{s}={s}", .{ module_cli_name, src }));
                    } else if (moduleNeedsCliArg(mod)) {
                        try zig_args.append(b.fmt("-M{s}", .{module_cli_name}));
                    }
                }
            }
        }

        // if (total_linker_objects == 0) {
        //     return step.fail("the linker needs one or more objects to link", .{});
        // }

        // for (frameworks.keys(), frameworks.values()) |name, info| {
        //     if (info.needed) {
        //         try zig_args.append("-needed_framework");
        //     } else if (info.weak) {
        //         try zig_args.append("-weak_framework");
        //     } else {
        //         try zig_args.append("-framework");
        //     }
        //     try zig_args.append(name);
    }

    // if (compile.is_linking_libcpp) {
    //     try zig_args.append("-lc++");
    // }

    // if (compile.is_linking_libc) {
    //     try zig_args.append("-lc");
    // }
    // }

    // if (compile.win32_manifest) |manifest_file| {
    //     try zig_args.append(manifest_file.getPath2(b, step));
    // }

    if (compile.image_base) |image_base| {
        try zig_args.append("--image-base");
        try zig_args.append(b.fmt("0x{x}", .{image_base}));
    }

    for (compile.filters) |filter| {
        try zig_args.append("--test-filter");
        try zig_args.append(filter);
    }

    if (compile.test_runner) |test_runner| {
        try zig_args.append("--test-runner");
        try zig_args.append(test_runner.path.getPath2(b, step));
    }

    for (b.debug_log_scopes) |log_scope| {
        try zig_args.append("--debug-log");
        try zig_args.append(log_scope);
    }

    if (b.debug_compile_errors) {
        try zig_args.append("--debug-compile-errors");
    }

    if (b.verbose_cimport) try zig_args.append("--verbose-cimport");
    if (b.verbose_air) try zig_args.append("--verbose-air");
    if (b.verbose_llvm_ir) |path| try zig_args.append(b.fmt("--verbose-llvm-ir={s}", .{path}));
    if (b.verbose_llvm_bc) |path| try zig_args.append(b.fmt("--verbose-llvm-bc={s}", .{path}));
    if (b.verbose_link or compile.verbose_link) try zig_args.append("--verbose-link");
    if (b.verbose_cc or compile.verbose_cc) try zig_args.append("--verbose-cc");
    if (b.verbose_llvm_cpu_features) try zig_args.append("--verbose-llvm-cpu-features");

    if (compile.generated_asm != null) try zig_args.append("-femit-asm");
    if (compile.generated_bin == null) try zig_args.append("-fno-emit-bin");
    if (compile.generated_docs != null) try zig_args.append("-femit-docs");
    if (compile.generated_implib != null) try zig_args.append("-femit-implib");
    if (compile.generated_llvm_bc != null) try zig_args.append("-femit-llvm-bc");
    if (compile.generated_llvm_ir != null) try zig_args.append("-femit-llvm-ir");
    if (compile.generated_h != null) try zig_args.append("-femit-h");

    try addFlag(&zig_args, "formatted-panics", compile.formatted_panics);

    switch (compile.compress_debug_sections) {
        .none => {},
        .zlib => try zig_args.append("--compress-debug-sections=zlib"),
        .zstd => try zig_args.append("--compress-debug-sections=zstd"),
    }

    if (compile.link_eh_frame_hdr) {
        try zig_args.append("--eh-frame-hdr");
    }
    if (compile.link_emit_relocs) {
        try zig_args.append("--emit-relocs");
    }
    if (compile.link_function_sections) {
        try zig_args.append("-ffunction-sections");
    }
    if (compile.link_data_sections) {
        try zig_args.append("-fdata-sections");
    }
    if (compile.link_gc_sections) |x| {
        try zig_args.append(if (x) "--gc-sections" else "--no-gc-sections");
    }
    if (!compile.linker_dynamicbase) {
        try zig_args.append("--no-dynamicbase");
    }
    if (compile.linker_allow_shlib_undefined) |x| {
        try zig_args.append(if (x) "-fallow-shlib-undefined" else "-fno-allow-shlib-undefined");
    }
    if (compile.link_z_notext) {
        try zig_args.append("-z");
        try zig_args.append("notext");
    }
    if (!compile.link_z_relro) {
        try zig_args.append("-z");
        try zig_args.append("norelro");
    }
    if (compile.link_z_lazy) {
        try zig_args.append("-z");
        try zig_args.append("lazy");
    }
    if (compile.link_z_common_page_size) |size| {
        try zig_args.append("-z");
        try zig_args.append(b.fmt("common-page-size={d}", .{size}));
    }
    if (compile.link_z_max_page_size) |size| {
        try zig_args.append("-z");
        try zig_args.append(b.fmt("max-page-size={d}", .{size}));
    }

    if (compile.libc_file) |libc_file| {
        try zig_args.append("--libc");
        try zig_args.append(libc_file.getPath2(b, step));
    } else if (b.libc_file) |libc_file| {
        try zig_args.append("--libc");
        try zig_args.append(libc_file);
    }

    try zig_args.append("--cache-dir");
    try zig_args.append(b.cache_root.path orelse ".");

    try zig_args.append("--global-cache-dir");
    try zig_args.append(b.graph.global_cache_root.path orelse ".");

    if (b.graph.debug_compiler_runtime_libs) try zig_args.append("--debug-rt");

    try zig_args.append("--name");
    try zig_args.append(compile.name);

    if (compile.linkage) |some| switch (some) {
        .dynamic => try zig_args.append("-dynamic"),
        .static => try zig_args.append("-static"),
    };
    if (compile.kind == .lib and compile.linkage != null and compile.linkage.? == .dynamic) {
        if (compile.version) |version| {
            try zig_args.append("--version");
            try zig_args.append(b.fmt("{}", .{version}));
        }

        if (compile.rootModuleTarget().isDarwin()) {
            const install_name = compile.install_name orelse b.fmt("@rpath/{s}{s}{s}", .{
                compile.rootModuleTarget().libPrefix(),
                compile.name,
                compile.rootModuleTarget().dynamicLibSuffix(),
            });
            try zig_args.append("-install_name");
            try zig_args.append(install_name);
        }
    }

    if (compile.entitlements) |entitlements| {
        try zig_args.appendSlice(&[_][]const u8{ "--entitlements", entitlements });
    }
    if (compile.pagezero_size) |pagezero_size| {
        const size = try std.fmt.allocPrint(arena, "{x}", .{pagezero_size});
        try zig_args.appendSlice(&[_][]const u8{ "-pagezero_size", size });
    }
    if (compile.headerpad_size) |headerpad_size| {
        const size = try std.fmt.allocPrint(arena, "{x}", .{headerpad_size});
        try zig_args.appendSlice(&[_][]const u8{ "-headerpad", size });
    }
    if (compile.headerpad_max_install_names) {
        try zig_args.append("-headerpad_max_install_names");
    }
    if (compile.dead_strip_dylibs) {
        try zig_args.append("-dead_strip_dylibs");
    }
    if (compile.force_load_objc) {
        try zig_args.append("-ObjC");
    }

    try addFlag(&zig_args, "compiler-rt", compile.bundle_compiler_rt);
    try addFlag(&zig_args, "dll-export-fns", compile.dll_export_fns);
    if (compile.rdynamic) {
        try zig_args.append("-rdynamic");
    }
    if (compile.import_memory) {
        try zig_args.append("--import-memory");
    }
    if (compile.export_memory) {
        try zig_args.append("--export-memory");
    }
    if (compile.import_symbols) {
        try zig_args.append("--import-symbols");
    }
    if (compile.import_table) {
        try zig_args.append("--import-table");
    }
    if (compile.export_table) {
        try zig_args.append("--export-table");
    }
    if (compile.initial_memory) |initial_memory| {
        try zig_args.append(b.fmt("--initial-memory={d}", .{initial_memory}));
    }
    if (compile.max_memory) |max_memory| {
        try zig_args.append(b.fmt("--max-memory={d}", .{max_memory}));
    }
    if (compile.shared_memory) {
        try zig_args.append("--shared-memory");
    }
    if (compile.global_base) |global_base| {
        try zig_args.append(b.fmt("--global-base={d}", .{global_base}));
    }

    if (compile.wasi_exec_model) |model| {
        try zig_args.append(b.fmt("-mexec-model={s}", .{@tagName(model)}));
    }
    // if (compile.linker_script) |linker_script| {
    //     try zig_args.append("--script");
    //     try zig_args.append(linker_script.getPath2(b, step));
    // }

    // if (compile.version_script) |version_script| {
    //     try zig_args.append("--version-script");
    //     try zig_args.append(version_script.getPath2(b, step));
    // }
    // if (compile.linker_allow_undefined_version) |x| {
    //     try zig_args.append(if (x) "--undefined-version" else "--no-undefined-version");
    // }

    // if (compile.linker_enable_new_dtags) |enabled| {
    //     try zig_args.append(if (enabled) "--enable-new-dtags" else "--disable-new-dtags");
    // }

    if (compile.kind == .@"test") {
        if (compile.exec_cmd_args) |exec_cmd_args| {
            for (exec_cmd_args) |cmd_arg| {
                if (cmd_arg) |arg| {
                    try zig_args.append("--test-cmd");
                    try zig_args.append(arg);
                } else {
                    try zig_args.append("--test-cmd-bin");
                }
            }
        }
    }

    if (compile.no_builtin) {
        try zig_args.append("-fno-builtin");
    }

    if (b.sysroot) |sysroot| {
        try zig_args.appendSlice(&[_][]const u8{ "--sysroot", sysroot });
    }

    // -I and -L arguments that appear after the last --mod argument apply to all modules.
    for (b.search_prefixes.items) |search_prefix| {
        var prefix_dir = std.fs.cwd().openDir(search_prefix, .{}) catch |err| {
            return step.fail("unable to open prefix directory '{s}': {s}", .{
                search_prefix, @errorName(err),
            });
        };
        defer prefix_dir.close();

        // Avoid passing -L and -I flags for nonexistent directories.
        // This prevents a warning, that should probably be upgraded to an error in Zig's
        // CLI parsing code, when the linker sees an -L directory that does not exist.

        if (prefix_dir.accessZ("lib", .{})) |_| {
            try zig_args.appendSlice(&.{
                "-L", b.pathJoin(&.{ search_prefix, "lib" }),
            });
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => |e| return step.fail("unable to access '{s}/lib' directory: {s}", .{
                search_prefix, @errorName(e),
            }),
        }

        if (prefix_dir.accessZ("include", .{})) |_| {
            try zig_args.appendSlice(&.{
                "-I", b.pathJoin(&.{ search_prefix, "include" }),
            });
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => |e| return step.fail("unable to access '{s}/include' directory: {s}", .{
                search_prefix, @errorName(e),
            }),
        }
    }

    if (compile.rc_includes != .any) {
        try zig_args.append("-rcincludes");
        try zig_args.append(@tagName(compile.rc_includes));
    }

    try addFlag(&zig_args, "each-lib-rpath", compile.each_lib_rpath);

    if (compile.build_id) |build_id| {
        try zig_args.append(switch (build_id) {
            .hexstring => |hs| b.fmt("--build-id=0x{s}", .{
                std.fmt.fmtSliceHexLower(hs.toSlice()),
            }),
            .none, .fast, .uuid, .sha1, .md5 => b.fmt("--build-id={s}", .{@tagName(build_id)}),
        });
    }

    const opt_zig_lib_dir = if (compile.zig_lib_dir) |dir|
        dir.getPath2(b, step)
    else if (b.graph.zig_lib_directory.path) |_|
        b.fmt("{}", .{b.graph.zig_lib_directory})
    else
        null;

    if (opt_zig_lib_dir) |zig_lib_dir| {
        try zig_args.append("--zig-lib-dir");
        try zig_args.append(zig_lib_dir);
    }

    try addFlag(&zig_args, "PIE", compile.pie);
    try addFlag(&zig_args, "lto", compile.want_lto);
    try addFlag(&zig_args, "sanitize-coverage-trace-pc-guard", compile.sanitize_coverage_trace_pc_guard);

    if (compile.subsystem) |subsystem| {
        try zig_args.append("--subsystem");
        try zig_args.append(switch (subsystem) {
            .Console => "console",
            .Windows => "windows",
            .Posix => "posix",
            .Native => "native",
            .EfiApplication => "efi_application",
            .EfiBootServiceDriver => "efi_boot_service_driver",
            .EfiRom => "efi_rom",
            .EfiRuntimeDriver => "efi_runtime_driver",
        });
    }

    if (compile.mingw_unicode_entry_point) {
        try zig_args.append("-municode");
    }

    if (compile.error_limit) |err_limit| try zig_args.appendSlice(&.{
        "--error-limit",
        b.fmt("{}", .{err_limit}),
    });

    try addFlag(&zig_args, "incremental", b.graph.incremental);

    // try zig_args.append("--listen=-");

    // Windows has an argument length limit of 32,766 characters, macOS 262,144 and Linux
    // 2,097,152. If our args exceed 30 KiB, we instead write them to a "response file" and
    // pass that to zig, e.g. via 'zig build-lib @args.rsp'
    // See @file syntax here: https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html
    var args_length: usize = 0;
    for (zig_args.items) |arg| {
        args_length += arg.len + 1; // +1 to account for null terminator
    }
    if (args_length >= 30 * 1024) {
        try b.cache_root.handle.makePath("args");

        const args_to_escape = zig_args.items[2..];
        var escaped_args = try ArrayList([]const u8).initCapacity(arena, args_to_escape.len);
        arg_blk: for (args_to_escape) |arg| {
            for (arg, 0..) |c, arg_idx| {
                if (c == '\\' or c == '"') {
                    // Slow path for arguments that need to be escaped. We'll need to allocate and copy
                    var escaped = try ArrayList(u8).initCapacity(arena, arg.len + 1);
                    const writer = escaped.writer();
                    try writer.writeAll(arg[0..arg_idx]);
                    for (arg[arg_idx..]) |to_escape| {
                        if (to_escape == '\\' or to_escape == '"') try writer.writeByte('\\');
                        try writer.writeByte(to_escape);
                    }
                    escaped_args.appendAssumeCapacity(escaped.items);
                    continue :arg_blk;
                }
            }
            escaped_args.appendAssumeCapacity(arg); // no escaping needed so just use original argument
        }

        // Write the args to zig-cache/args/<SHA256 hash of args> to avoid conflicts with
        // other zig build commands running in parallel.
        const partially_quoted = try std.mem.join(arena, "\" \"", escaped_args.items);
        const args = try std.mem.concat(arena, u8, &[_][]const u8{ "\"", partially_quoted, "\"" });

        var args_hash: [Sha256.digest_length]u8 = undefined;
        Sha256.hash(args, &args_hash, .{});
        var args_hex_hash: [Sha256.digest_length * 2]u8 = undefined;
        _ = try std.fmt.bufPrint(
            &args_hex_hash,
            "{s}",
            .{std.fmt.fmtSliceHexLower(&args_hash)},
        );

        const args_file = "args" ++ std.fs.path.sep_str ++ args_hex_hash;
        try b.cache_root.handle.writeFile(.{ .sub_path = args_file, .data = args });

        const resolved_args_file = try mem.concat(arena, u8, &.{
            "@",
            try b.cache_root.join(arena, &.{args_file}),
        });

        zig_args.shrinkRetainingCapacity(2);
        try zig_args.append(resolved_args_file);
    }

    try zig_args.appendSlice(&.{
        "-fincremental",
        "-fno-emit-bin",
        "-fno-emit-asm",
        "-fno-emit-llvm-ir",
        "-fno-emit-llvm-bc",
        "-fno-emit-h",
        "-fno-emit-docs",
        "-fno-emit-implib",
        "--proj-path",
        b.fmt("{s}", .{build_root}),
    });

    return try zig_args.toOwnedSlice();
}

fn addFlag(args: *ArrayList([]const u8), comptime name: []const u8, opt: ?bool) !void {
    const cond = opt orelse return;
    try args.ensureUnusedCapacity(1);
    if (cond) {
        args.appendAssumeCapacity("-f" ++ name);
    } else {
        args.appendAssumeCapacity("-fno-" ++ name);
    }
}

fn moduleNeedsCliArg(mod: *const Module) bool {
    return for (mod.link_objects.items) |o| switch (o) {
        .c_source_file, .c_source_files, .assembly_file, .win32_resource_file => break true,
        else => continue,
    } else false;
}
