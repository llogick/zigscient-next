const assert = std.debug.assert;
const builtin = @import("builtin");
const std = @import("std");
const Runner = @This();

/// These are passed as `<target>-<cache_mode>-<backend>-<linker>-<pic>` strings.
///
/// If only one specific test is failing on a target, instead of entirely disabling the target here, you
/// can skip the target for that specific test only by adding a line like this to top of the manifest:
///
///     #skip x86_64-linux-none-baseline-whole-selfhosted-new-pic
pub const Target = struct {
    cpu: Cpu = .{ .native = .{} },
    os: Os = .native,
    abi: Abi = .default,
    ofmt: ObjectFormat = .default,
    dynamic_linker: DynamicLinker = .default,

    cache_mode: enum { whole, incremental },
    backend: enum { selfhosted, llvm, cbe },
    linker: enum { lld, old, new },
    pic: enum { nopic, pic, pie },

    pub const Cpu = Cpu: {
        @setEvalBranchQuota(3_000);
        const Arch = std.Target.Cpu.Arch;
        const arch_info = @typeInfo(Arch).@"enum";
        const field_names = .{"native"} ++ arch_info.field_names;
        var field_types: [field_names.len]type = undefined;
        field_types[0] = struct {
            model: Model = .native,

            pub const Model = enum { native, baseline };
        };
        for (field_types[1..], field_names[1..]) |*field_type, field_name| {
            const arch = @field(Arch, field_name);
            const models = arch.allCpuModels();
            const family = @field(std.Target, @tagName(arch.family()));
            field_type.* = struct {
                model: Model = .baseline,
                features: struct {
                    add: []const Feature = &.{},
                    remove: []const Feature = &.{},
                    comptime {
                        _ = &family; // hack around compiler bug
                    }
                } = .{},

                pub const Model = Model: {
                    const explicit_models = @typeInfo(family.cpu).@"struct".decl_names;
                    var model_names: []const []const u8 = &.{"native"};
                    if (!@hasDecl(family.cpu, "baseline")) model_names = model_names ++ .{"baseline"};
                    model_names = model_names ++ explicit_models;
                    const TagInt = std.math.IntFittingRange(0, model_names.len - 1);
                    var model_values: [model_names.len]TagInt = undefined;
                    var model_index = 0;
                    {
                        var model_value = explicit_models.len;
                        model_values[model_index] = model_value;
                        model_index += 1;
                        model_value += 1;
                        if (!@hasDecl(family.cpu, "baseline")) {
                            model_values[model_index] = model_value;
                            model_index += 1;
                            model_value += 1;
                        }
                    }
                    for (model_values[model_index..], 0..) |*model_value, value| model_value.* = value;
                    break :Model @Enum(TagInt, .exhaustive, model_names, &model_values);
                };

                pub const Feature = family.Feature;

                pub fn updateQuery(cpu: *const @This(), query: *std.Target.Query) void {
                    query.cpu_model = switch (cpu.model) {
                        .native => .native,
                        .baseline => .baseline,
                        else => |model| .{ .explicit = models[@backingInt(model)] },
                    };
                    for (cpu.features.add) |add_feature|
                        query.cpu_features_add.addFeature(@backingInt(add_feature));
                    for (cpu.features.remove) |remove_feature|
                        query.cpu_features_sub.addFeature(@backingInt(remove_feature));
                }
            };
        }
        const TagInt = std.math.IntFittingRange(0, field_names.len - 1);
        var field_values: [field_names.len]TagInt = undefined;
        field_values[0] = field_names.len - 1;
        for (field_values[1..], arch_info.field_values) |*field_value, value| field_value.* = value;
        const Tag = @Enum(TagInt, .exhaustive, field_names, &field_values);
        break :Cpu @Union(.auto, Tag, field_names, &field_types, &@splat(.{}));
    };

    pub const Os = Os: {
        const os_info = @typeInfo(std.Target.Os.Tag).@"enum";
        const field_names = .{"native"} ++ os_info.field_names;
        var field_types: [field_names.len]type = undefined;
        field_types[0] = void;
        for (field_types[1..], field_names[1..]) |*field_type, field_name| {
            const version: enum {
                none,
                semver,
                windows,
            } = switch (@field(std.Target.Os.Tag, field_name).versionRangeTag()) {
                .none => .none,
                .semver, .hurd, .linux => .semver,
                .windows => .windows,
            };
            field_type.* = struct {
                version: ?Version = null,

                pub const Version = switch (version) {
                    .none => struct {},
                    .semver => struct {
                        min: std.SemanticVersion,
                        max: ?std.SemanticVersion = null,
                    },
                    .windows => struct {
                        min: std.Target.Os.WindowsVersion,
                        max: ?std.Target.Os.WindowsVersion = null,
                    },
                };

                pub fn updateQuery(os: *const @This(), query: *std.Target.Query) void {
                    if (os.version) |v| switch (version) {
                        .none => {},
                        .semver => {
                            query.os_version_min = .{ .semver = v.min };
                            if (v.max) |max| query.os_version_max = .{ .semver = max };
                        },
                        .windows => {
                            query.os_version_min = .{ .windows = v.min };
                            if (v.max) |max| query.os_version_max = .{ .windows = max };
                        },
                    };
                }
            };
        }
        const TagInt = std.math.IntFittingRange(0, field_names.len - 1);
        var field_values: [field_names.len]TagInt = undefined;
        field_values[0] = field_names.len - 1;
        for (field_values[1..], os_info.field_values) |*field_value, value| field_value.* = value;
        const Tag = @Enum(TagInt, .exhaustive, field_names, &field_values);
        break :Os @Union(.auto, Tag, field_names, &field_types, &@splat(.{}));
    };

    pub const Abi = Abi: {
        const abi_info = @typeInfo(std.Target.Abi).@"enum";
        const field_names = .{"default"} ++ abi_info.field_names;
        var field_types: [field_names.len]type = undefined;
        field_types[0] = void;
        for (field_types[1..], field_names[1..]) |*field_type, field_name| {
            const abi_tag = @field(std.Target.Abi, field_name);
            const version: enum { none, libc, android } = if (abi_tag.isAndroid())
                .android
            else if (abi_tag.isGnu() or abi_tag.isMusl())
                .libc
            else
                .none;
            field_type.* = struct {
                version: ?Version = null,

                pub const Version = switch (version) {
                    .none => struct {},
                    .libc => std.SemanticVersion,
                    .android => u32,
                };

                pub fn updateQuery(abi: *const @This(), query: *std.Target.Query) void {
                    if (abi.version) |v| switch (version) {
                        .none => {},
                        .libc => query.glibc_version = v,
                        .android => query.android_api_level = v,
                    };
                }
            };
        }
        const TagInt = std.math.IntFittingRange(0, field_names.len - 1);
        var field_values: [field_names.len]TagInt = undefined;
        field_values[0] = field_names.len - 1;
        for (field_values[1..], abi_info.field_values) |*field_value, value| field_value.* = value;
        const Tag = @Enum(TagInt, .exhaustive, field_names, &field_values);
        break :Abi @Union(.auto, Tag, field_names, &field_types, &@splat(.{}));
    };

    pub const ObjectFormat = ObjectFormat: {
        const ofmt_info = @typeInfo(std.Target.ObjectFormat).@"enum";
        const field_names = .{"default"} ++ ofmt_info.field_names;
        const TagInt = std.math.IntFittingRange(0, field_names.len - 1);
        var field_values: [field_names.len]TagInt = undefined;
        field_values[0] = field_names.len - 1;
        for (field_values[1..], ofmt_info.field_values) |*field_value, value| field_value.* = value;
        break :ObjectFormat @Enum(TagInt, .exhaustive, field_names, &field_values);
    };

    pub const DynamicLinker = union(enum) {
        default,
        none,
        explicit: []const u8,
    };

    pub fn toQuery(target: *const Target) std.Target.Query {
        var query: std.Target.Query = .{};
        query.cpu_arch = switch (target.cpu) {
            .native => null,
            else => |arch| @fromBackingInt(@backingInt(arch)),
        };
        switch (target.cpu) {
            .native => |native| query.cpu_model = switch (native.model) {
                .native => .native,
                .baseline => .baseline,
            },
            inline else => |cpu| cpu.updateQuery(&query),
        }
        query.os_tag = switch (target.os) {
            .native => null,
            else => |os| @fromBackingInt(@backingInt(os)),
        };
        switch (target.os) {
            .native => {},
            inline else => |os| os.updateQuery(&query),
        }
        query.abi = switch (target.abi) {
            .default => null,
            else => |abi| @fromBackingInt(@backingInt(abi)),
        };
        switch (target.abi) {
            .default => {},
            inline else => |abi| abi.updateQuery(&query),
        }
        query.ofmt = switch (target.ofmt) {
            .default => null,
            else => |ofmt| @fromBackingInt(@backingInt(ofmt)),
        };
        query.dynamic_linker = switch (target.dynamic_linker) {
            .default => null,
            .none => .none,
            .explicit => |dynamic_linker| .init(dynamic_linker),
        };
        return query;
    }

    pub fn format(target: *const Target, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("{f}-{f}-{t}-{t}-{t}-{t}", .{
            std.fmt.alt(&target.toQuery(), .printZigTriple),
            std.fmt.alt(&target.toQuery(), .printCpu),
            target.cache_mode,
            target.backend,
            target.linker,
            target.pic,
        });
    }
};

gpa: std.mem.Allocator,
arena: *std.heap.ArenaAllocator,
environ_map: std.process.Environ.Map,
io: std.Io,
prog_node: std.Progress.Node,
args: Args,
host: std.Target,
server: std.zig.Server,
eb_wip: std.zig.ErrorBundle.Wip,

const Args = struct {
    @"test": ?struct {
        input_dir: std.zig.Server.Message.InputDir,
        dir: std.Io.Dir,
    },
    manifest_file: ?std.Io.File,
    zig_exe: ?std.Io.File,
    lib_dir: ?std.Io.Dir,
    src_dir: ?std.Io.Dir,
    targets: std.ArrayList(u32),
    target_bytes: std.ArrayList(u8),

    libc_runtimes_dir: ?std.Io.Dir,
    enable_darling: bool,
    enable_qemu: bool,
    enable_rosetta: bool,
    enable_wasmtime: bool,
    enable_wine: bool,

    gdb_exe: ?std.Io.File,
    lldb_exe: ?std.Io.File,

    quiet: bool,

    fn deinit(args: *Args, gpa: std.mem.Allocator) void {
        args.targets.deinit(gpa);
        args.target_bytes.deinit(gpa);
        args.* = undefined;
    }

    fn addTarget(args: *Args, gpa: std.mem.Allocator, target: []const u8) std.mem.Allocator.Error!void {
        try args.target_bytes.ensureUnusedCapacity(gpa, target.len + 1);
        try args.targets.append(gpa, @intCast(args.target_bytes.items.len));
        args.target_bytes.appendSliceAssumeCapacity(target);
        args.target_bytes.appendAssumeCapacity(0);
    }
};

fn deinit(runner: *Runner) void {
    runner.environ_map.deinit();
    runner.prog_node.end();
    runner.args.deinit(runner.gpa);
    runner.eb_wip.deinit();
    runner.* = undefined;
}

const FailError = error{AlreadyReported} || std.mem.Allocator.Error || std.Io.Cancelable;
fn fail(runner: *Runner, comptime fmt: []const u8, args: anytype) FailError {
    return runner.failString(try runner.eb_wip.printString(fmt, args));
}
fn failString(runner: *Runner, msg: std.zig.ErrorBundle.String) FailError {
    try runner.eb_wip.addRootErrorMessage(.{ .msg = msg });
    return error.AlreadyReported;
}

pub const ProtocolError = error{
    ServerReadFailed,
    ServerWriteFailed,
    ServerEndOfStream,
} || FailError;
pub fn runServer(runner: *Runner) ProtocolError {
    runner.server.serveStringMessage(.zig_version, builtin.zig_version_string) catch |err| switch (err) {
        error.WriteFailed => return error.ServerWriteFailed,
    };
    while (true) {
        _ = runner.arena.reset(.retain_capacity);
        const arena = runner.arena.allocator();
        const hdr = runner.server.receiveMessage() catch |err| switch (err) {
            error.ReadFailed => return error.ServerReadFailed,
            error.EndOfStream => return error.ServerEndOfStream,
        };
        switch (hdr.tag) {
            else => |tag| return runner.fail("unsupported message: {t}", .{tag}),
            .exit => std.process.exit(0),
            .args => {
                const args_body = try arena.alloc(u8, hdr.bytes_len);
                runner.server.in.readSliceAll(args_body) catch unreachable;
                const State = enum {
                    positional,
                    @"--zig",
                    @"--lib",
                    @"--src",
                    @"--target",
                    @"--libc-runtimes",
                    @"--gdb",
                    @"--lldb",
                };
                var state: State = .positional;
                const state_expected: std.enums.EnumArray(
                    State,
                    enum { path, dir, file, string },
                ) = .init(.{
                    .positional = .path,
                    .@"--zig" = .file,
                    .@"--lib" = .dir,
                    .@"--src" = .dir,
                    .@"--target" = .string,
                    .@"--libc-runtimes" = .dir,
                    .@"--gdb" = .file,
                    .@"--lldb" = .file,
                });
                var input_dir: std.zig.Server.Message.InputDir = .cwd;
                var args_body_offset: usize = 0;
                while (args_body.len - args_body_offset > 0) {
                    const arg: std.zig.Client.Message.Arg =
                        @fromBackingInt(args_body[args_body_offset]);
                    args_body_offset += 1;
                    arg: switch (arg) {
                        .string => {
                            const end = std.mem.findScalarPos(u8, args_body, args_body_offset, 0).?;
                            const string = args_body[args_body_offset..end];
                            args_body_offset = end + 1;
                            switch (state) {
                                .positional => if (std.mem.eql(u8, string, "--target")) {
                                    state = .@"--target";
                                    continue;
                                } else if (std.mem.cutPrefix(u8, string, "--target=")) |target| {
                                    try runner.args.addTarget(runner.gpa, target);
                                } else if (std.mem.eql(u8, string, "--libc-runtimes")) {
                                    state = .@"--libc-runtimes";
                                    continue;
                                } else if (std.mem.eql(u8, string, "-fdarling")) {
                                    runner.args.enable_darling = true;
                                } else if (std.mem.eql(u8, string, "-fqemu")) {
                                    runner.args.enable_qemu = true;
                                } else if (std.mem.eql(u8, string, "-frosetta")) {
                                    runner.args.enable_rosetta = true;
                                } else if (std.mem.eql(u8, string, "-fwasmtime")) {
                                    runner.args.enable_wasmtime = true;
                                } else if (std.mem.eql(u8, string, "-fwine")) {
                                    runner.args.enable_wine = true;
                                } else if (std.mem.eql(u8, string, "--gdb")) {
                                    state = .@"--gdb";
                                    continue;
                                } else if (std.mem.eql(u8, string, "--lldb")) {
                                    state = .@"--lldb";
                                    continue;
                                } else if (std.mem.eql(u8, string, "--quiet")) {
                                    runner.args.quiet = true;
                                },
                                .@"--target" => try runner.args.addTarget(runner.gpa, string),
                                else => return runner.fail(
                                    "\"{t}\" expected {t}, got {t}",
                                    .{ state, state_expected.get(state), arg },
                                ),
                            }
                        },
                        .prefix => {
                            const end = std.mem.findScalarPos(u8, args_body, args_body_offset, 0).?;
                            const string = args_body[args_body_offset..end];
                            args_body_offset = end + 1;
                            assert(state == .positional);
                            state = if (std.mem.eql(u8, string, "--zig="))
                                .@"--zig"
                            else if (std.mem.eql(u8, string, "--lib="))
                                .@"--lib"
                            else if (std.mem.eql(u8, string, "--src="))
                                .@"--src"
                            else if (std.mem.eql(u8, string, "--libc-runtimes="))
                                .@"--libc-runtimes"
                            else if (std.mem.eql(u8, string, "--gdb="))
                                .@"--gdb"
                            else if (std.mem.eql(u8, string, "--lldb="))
                                .@"--lldb"
                            else
                                return runner.fail("unsupported arg prefix: {q}", .{string});
                            continue;
                        },
                        .suffix => {
                            const end = std.mem.findScalarPos(u8, args_body, args_body_offset, 0).?;
                            const string = args_body[args_body_offset..end];
                            args_body_offset = end + 1;
                            return runner.fail("unsupported arg suffix: {q}", .{string});
                        },
                        .input_dir => {
                            input_dir = @fromBackingInt(@backingInt(input_dir) + 1);
                            continue :arg .output_dir;
                        },
                        .output_dir => {
                            const dir_handle: *align(1) const std.Io.Dir.Handle = @ptrCast(
                                args_body[args_body_offset..][0..@sizeOf(std.Io.Dir.Handle)],
                            );
                            args_body_offset += @sizeOf(std.Io.Dir.Handle);
                            const dir = runner.io.vtable.inheritParentDir(
                                runner.io.userdata,
                                dir_handle.*,
                            ) catch |err| switch (err) {
                                error.Canceled => |e| return e,
                                else => |e| return runner.fail(
                                    "unable to inherit parent dir: {t}",
                                    .{e},
                                ),
                            };
                            switch (state) {
                                .positional => {
                                    if (runner.args.@"test" != null or
                                        runner.args.manifest_file != null) return runner.fail(
                                        "{t} specified multiple times",
                                        .{state},
                                    );
                                    runner.args.@"test" = .{ .input_dir = input_dir, .dir = dir };
                                    const manifest_path = "manifest";
                                    runner.args.manifest_file = dir.openFile(
                                        runner.io,
                                        manifest_path,
                                        .{},
                                    ) catch |err| switch (err) {
                                        error.Canceled => |e| return e,
                                        else => |e| return runner.fail(
                                            "unable to open manifest file: {t}",
                                            .{e},
                                        ),
                                    };
                                    try runner.discoverInput(input_dir, manifest_path);
                                },
                                .@"--lib" => {
                                    if (runner.args.lib_dir != null) return runner.fail(
                                        "\"{t}\" specified multiple times",
                                        .{state},
                                    );
                                    runner.args.lib_dir = dir;
                                },
                                .@"--src" => {
                                    if (runner.args.src_dir != null) return runner.fail(
                                        "\"{t}\" specified multiple times",
                                        .{state},
                                    );
                                    runner.args.src_dir = dir;
                                },
                                .@"--libc-runtimes" => {
                                    if (runner.args.libc_runtimes_dir != null) return runner.fail(
                                        "\"{t}\" specified multiple times",
                                        .{state},
                                    );
                                    runner.args.libc_runtimes_dir = dir;
                                },
                                else => return runner.fail(
                                    "\"{t}\" expected {t}, got {t}",
                                    .{ state, state_expected.get(state), arg },
                                ),
                            }
                        },
                        .input_file, .input_file_content, .output_file => {
                            const file_handle: *align(1) const std.Io.File.Handle = @ptrCast(
                                args_body[args_body_offset..][0..@sizeOf(std.Io.File.Handle)],
                            );
                            args_body_offset += @sizeOf(std.Io.File.Handle);
                            const file = runner.io.vtable.inheritParentFile(
                                runner.io.userdata,
                                file_handle.*,
                                .{ .nonblocking = false },
                            ) catch |err| switch (err) {
                                error.Canceled => |e| return e,
                                else => |e| return runner.fail(
                                    "unable to inherit parent file: {t}",
                                    .{e},
                                ),
                            };
                            switch (state) {
                                .positional => {
                                    if (runner.args.@"test" != null or
                                        runner.args.manifest_file != null) return runner.fail(
                                        "{t} specified multiple times",
                                        .{state},
                                    );
                                    runner.args.manifest_file = file;
                                },
                                .@"--zig" => {
                                    if (runner.args.zig_exe != null) return runner.fail(
                                        "\"{t}\" specified multiple times",
                                        .{state},
                                    );
                                    runner.args.zig_exe = file;
                                },
                                .@"--gdb" => {
                                    if (runner.args.gdb_exe != null) return runner.fail(
                                        "\"{t}\" specified multiple times",
                                        .{state},
                                    );
                                    runner.args.gdb_exe = file;
                                },
                                .@"--lldb" => {
                                    if (runner.args.lldb_exe != null) return runner.fail(
                                        "\"{t}\" specified multiple times",
                                        .{state},
                                    );
                                    runner.args.lldb_exe = file;
                                },
                                else => return runner.fail(
                                    "\"{t}\" expected {t}, got {t}",
                                    .{ state, state_expected.get(state), arg },
                                ),
                            }
                        },
                    }
                    state = .positional;
                }
            },
            .query_test_metadata => {
                const expected_panic_msgs = try arena.alloc(u32, runner.args.targets.items.len);
                @memset(expected_panic_msgs, 0);
                runner.server.serveTestMetadata(.{
                    .names = runner.args.targets.items,
                    .expected_panic_msgs = expected_panic_msgs,
                    .string_bytes = runner.args.target_bytes.items,
                }) catch |err| switch (err) {
                    error.WriteFailed => return error.ServerWriteFailed,
                };
            },
            .run_test => {
                const test_index = runner.server.receiveBody_u32() catch unreachable;
                runner.server.serveBodylessMessage(.test_started) catch |err| switch (err) {
                    error.WriteFailed => return error.ServerWriteFailed,
                };
                runner.server.serveTestResults(.{
                    .index = test_index,
                    .flags = .{
                        .status = if (runner.testOne(test_index))
                            .pass
                        else |err| status: switch (err) {
                            error.Canceled,
                            error.OutOfMemory,
                            error.ServerWriteFailed,
                            => |e| return e,
                            error.AlreadyReported => {
                                var eb = try runner.eb_wip.toOwnedBundle("");
                                defer eb.deinit(runner.gpa);
                                assert(eb.errorMessageCount() > 0); // already reported what?
                                runner.server.serveErrorBundle(
                                    .error_bundle,
                                    eb,
                                ) catch |server_err| switch (server_err) {
                                    error.WriteFailed => return error.ServerWriteFailed,
                                };
                                const eb_wip: std.zig.ErrorBundle.Wip = try .init(runner.gpa);
                                runner.eb_wip.deinit();
                                runner.eb_wip = eb_wip;
                                break :status .fail;
                            },
                            error.SkipTest => .skip,
                        },
                        .fuzz = false,
                        .log_err_count = 0,
                        .leak_count = 0,
                    },
                }) catch |err| switch (err) {
                    error.WriteFailed => return error.ServerWriteFailed,
                };
                assert(runner.eb_wip.root_list.items.len == 0); // failed to report
            },
        }
    }
}

fn discoverInput(
    runner: *Runner,
    dir: std.zig.Server.Message.InputDir,
    sub_path: []const u8,
) error{ServerWriteFailed}!void {
    return runner.discoverInputInner(dir, sub_path) catch |err| switch (err) {
        error.WriteFailed => return error.ServerWriteFailed,
    };
}
fn discoverInputInner(
    runner: *Runner,
    dir: std.zig.Server.Message.InputDir,
    sub_path: []const u8,
) std.Io.Writer.Error!void {
    try runner.server.serveMessageHeader(.{
        .tag = .discovered_inputs,
        .bytes_len = @intCast(@sizeOf(std.zig.Server.Message.InputDir) +
            sub_path.len + 1),
    });
    try runner.server.out.writeInt(u32, @backingInt(dir), .little);
    try runner.server.out.writeAll(sub_path);
    try runner.server.out.writeByte(0);
    try runner.server.out.flush();
}

pub const SourceReader = struct {
    const Io = std.Io;

    backing: *std.Io.Reader,
    line_aw: std.Io.Writer.Allocating,
    interface: std.Io.Reader,
    line: u32,
    eb_wip: *std.zig.ErrorBundle.Wip,
    src_path: std.zig.ErrorBundle.String,

    pub fn init(
        reader: *Io.Reader,
        gpa: std.mem.Allocator,
        eb_wip: *std.zig.ErrorBundle.Wip,
        src_path: std.zig.ErrorBundle.String,
    ) SourceReader {
        return .{
            .backing = reader,
            .line_aw = .init(gpa),
            .interface = .{
                .vtable = &.{
                    .stream = stream,
                    .discard = discard,
                    .readVec = readVec,
                    .rebase = rebase,
                },
                .buffer = &.{},
                .seek = 0,
                .end = 0,
            },
            .line = 0,
            .eb_wip = eb_wip,
            .src_path = src_path,
        };
    }
    fn deinit(s: *SourceReader) void {
        s.line_aw.deinit();
    }
    fn stream(r: *Io.Reader, _: *Io.Writer, _: Io.Limit) Io.Reader.StreamError!usize {
        const s: *SourceReader = @fieldParentPtr("interface", r);
        try s.bufferLine();
        return 0;
    }
    fn discard(r: *Io.Reader, limit: Io.Limit) Io.Reader.Error!usize {
        const s: *SourceReader = @fieldParentPtr("interface", r);
        const remaining = r.buffered();
        if (remaining.len > 0) {
            const n = limit.minInt(remaining.len);
            r.toss(n);
            return n;
        }
        try s.bufferLine();
        return 0;
    }
    fn readVec(r: *Io.Reader, _: [][]u8) Io.Reader.Error!usize {
        const s: *SourceReader = @fieldParentPtr("interface", r);
        try s.bufferLine();
        return 0;
    }
    fn rebase(r: *Io.Reader, capacity: usize) Io.Reader.RebaseError!void {
        const s: *SourceReader = @fieldParentPtr("interface", r);
        defer s.updateBuffer();
        s.line_aw.ensureUnusedCapacity(capacity) catch |err| switch (err) {
            error.OutOfMemory => return error.ReadFailed,
        };
    }
    fn updateBuffer(s: *SourceReader) void {
        s.interface.buffer = s.line_aw.writer.buffer;
        s.interface.end = s.line_aw.writer.end;
    }
    fn bufferLine(s: *SourceReader) Io.Reader.Error!void {
        defer s.updateBuffer();
        {
            const written = s.line_aw.written();
            const start = if (std.mem.findScalarLast(u8, written[0..s.interface.seek], '\n')) |newline|
                newline + 1
            else
                0;
            s.line += @intCast(std.mem.countScalar(u8, written[0..start], '\n'));
            const preserve = written[start..];
            @memmove(written[0..preserve.len], preserve);
            s.line_aw.shrinkRetainingCapacity(preserve.len);
            s.interface.seek -= start;
        }
        const n = s.backing.streamDelimiterEnding(&s.line_aw.writer, '\n') catch |err| switch (err) {
            error.ReadFailed => |e| return e,
            error.WriteFailed => return error.ReadFailed,
        };
        if (std.mem.endsWith(u8, s.line_aw.written(), "\r")) s.line_aw.writer.end -= 1;
        const buffered = s.backing.buffered();
        if (n == 0 and buffered.len == 0) return error.EndOfStream;
        if (std.mem.startsWith(u8, buffered, "\n")) {
            s.line_aw.writer.writeByte('\n') catch |err| switch (err) {
                error.WriteFailed => return error.ReadFailed,
            };
            s.backing.toss(1);
        }
    }
    fn fail(
        s: *SourceReader,
        src_loc: []const u8,
        comptime fmt: []const u8,
        args: anytype,
    ) FailError {
        return s.failString(src_loc, try s.eb_wip.printString(fmt, args));
    }
    fn failString(
        s: *SourceReader,
        src_loc: []const u8,
        msg: std.zig.ErrorBundle.String,
    ) FailError {
        try s.eb_wip.addRootErrorMessage(.{ .msg = msg, .src_loc = try s.srcLoc(src_loc) });
        return error.AlreadyReported;
    }
    fn srcLoc(
        s: *SourceReader,
        src_loc: []const u8,
    ) std.mem.Allocator.Error!std.zig.ErrorBundle.SourceLocationIndex {
        const line: u32, const column: u32, const source_line = src_loc: {
            const buffer = s.interface.buffer[0..s.interface.end];
            const pos = src_loc.ptr - buffer.ptr;
            const end = std.mem.findScalarPos(u8, buffer, pos, '\n') orelse buffer.len;
            const start =
                if (std.mem.findScalarLast(u8, buffer[0..end], '\n')) |newline| newline + 1 else 0;
            break :src_loc .{
                @intCast(s.line + std.mem.countScalar(u8, buffer[0..start], '\n')),
                @intCast(pos - start),
                buffer[start..end],
            };
        };
        return s.eb_wip.addSourceLocation(.{
            .src_path = s.src_path,
            .line = line,
            .column = column,
            .span_start = column,
            .span_main = column,
            .span_end = @intCast(column + src_loc.len),
            .source_line = try s.eb_wip.addString(source_line),
        });
    }
};
pub const DelimitedReader = struct {
    const Io = std.Io;

    unlimited: *Io.Reader,
    remaining: Io.Limit,
    delimiter: []const u8,
    interface: Io.Reader,

    pub fn init(reader: *Io.Reader, delimiter: []const u8, buffer: []u8) DelimitedReader {
        return .{
            .unlimited = reader,
            .remaining = .nothing,
            .delimiter = delimiter,
            .interface = .{
                .vtable = &.{
                    .stream = stream,
                    .discard = discard,
                },
                .buffer = buffer,
                .seek = 0,
                .end = 0,
            },
        };
    }
    fn stream(r: *Io.Reader, w: *Io.Writer, limit: Io.Limit) Io.Reader.StreamError!usize {
        const d: *DelimitedReader = @fieldParentPtr("interface", r);
        const block = try d.unlimited.peekGreedy(d.delimiter.len);
        d.remaining = .limited(std.mem.findPos(u8, block, d.remaining.toInt().?, d.delimiter) orelse
            (block.len - d.delimiter.len + 1));
        if (d.remaining == .nothing) return error.EndOfStream;
        const n = try d.unlimited.stream(w, limit.min(d.remaining));
        d.remaining = d.remaining.subtract(n).?;
        return n;
    }
    fn discard(r: *Io.Reader, limit: Io.Limit) Io.Reader.Error!usize {
        const d: *DelimitedReader = @fieldParentPtr("interface", r);
        const block = try d.unlimited.peekGreedy(d.delimiter.len);
        d.remaining = .limited(std.mem.findPos(u8, block, d.remaining.toInt().?, d.delimiter) orelse
            (block.len - d.delimiter.len + 1));
        if (d.remaining == .nothing) return error.EndOfStream;
        const n = try d.unlimited.discard(limit.min(d.remaining));
        d.remaining = d.remaining.subtract(n).?;
        return n;
    }
};

const Update = struct {
    target_query: []const u8,
    manifest_sr: SourceReader,
    compiler: ?Compiler,
    src_dir: std.Io.Dir,
    eb_wip: *std.zig.ErrorBundle.Wip,
    cmd_src_loc: std.zig.ErrorBundle.SourceLocationIndex,
    src_loc: ?std.zig.ErrorBundle.SourceLocationIndex,
    num: usize,
    name: []const u8,
    prog_node: std.Progress.Node,
    mtime: std.Io.Timestamp,
    allow_skip: bool,

    fn checkMatches(
        update: *const Update,
        what: []const u8,
        expected: []const u8,
        actual: []const u8,
    ) FailError!void {
        var expected_match_it = std.mem.splitSequence(u8, expected, "#...");
        var pos: usize = 0;
        while (expected_match_it.next()) |expected_match| : (pos += expected_match.len) {
            var context_len: usize = expected_match.len;
            pos = while (true) : (context_len -= 1) break std.mem.findPos(
                u8,
                actual,
                pos,
                expected_match[0..context_len],
            ) orelse continue;
            const actual_match = actual[pos..];
            std.testing.expectEqualStrings(
                expected_match,
                actual_match[0..@min(actual_match.len, expected_match.len)],
            ) catch |err| switch (err) {
                error.TestExpectedEqual => {
                    std.testing.failPrint("\n=============== full output: =========\n", .{});
                    std.testing.failPrintWithVisibleNewlines(actual_match);
                    std.testing.failPrint("\n======================================\n", .{});
                    return update.fail("{s} did not match expected", .{what});
                },
            };
        }
    }

    fn fail(update: *const Update, comptime fmt: []const u8, args: anytype) FailError {
        return update.failString(try update.eb_wip.printString(fmt, args));
    }
    fn failString(update: *const Update, msg: std.zig.ErrorBundle.String) FailError {
        var notes_buffer: [1]std.zig.ErrorBundle.ErrorMessage = undefined;
        var notes_len: u32 = 0;
        if (update.src_loc) |src_loc| {
            notes_buffer[notes_len] = .{
                .msg = try update.eb_wip.addString("after this update"),
                .src_loc = src_loc,
            };
            notes_len += 1;
        }
        try update.eb_wip.addRootErrorMessageWithNotes(.{
            .msg = msg,
            .src_loc = update.cmd_src_loc,
            .notes_len = 1,
        }, notes_buffer[0..notes_len]);
        return error.AlreadyReported;
    }
};

const TestError = error{
    SkipTest,
} || FailError;
fn testOne(runner: *Runner, test_index: u32) (error{ServerWriteFailed} || TestError)!void {
    const target_query =
        std.mem.sliceTo(runner.args.target_bytes.items[runner.args.targets.items[test_index]..], 0);
    const manifest_file = runner.args.manifest_file orelse
        return runner.fail("missing \"path/to/test\" arg", .{});
    var manifest_buffer: [512]u8 = undefined;
    var manifest_fr = manifest_file.reader(runner.io, &manifest_buffer);
    var update: Update = .{
        .target_query = target_query,
        .manifest_sr = .init(&manifest_fr.interface, runner.gpa, &runner.eb_wip, src_path_string: {
            var path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const src_path = src_path: {
                break :src_path path_buffer[0 .. manifest_file.realPath(
                    runner.io,
                    &path_buffer,
                ) catch |err| switch (err) {
                    error.Canceled => |e| return e,
                    else => break :src_path "manifest",
                }];
            };
            break :src_path_string try runner.eb_wip.addString(src_path);
        }),
        .compiler = null,
        .src_dir = (runner.args.src_dir orelse
            return runner.fail("missing \"--src=path/to/src\" arg", .{})).createDirPathOpen(
            runner.io,
            target_query,
            .{},
        ) catch |err| switch (err) {
            error.Canceled => |e| return e,
            else => |e| return runner.fail("unable to create {q}: {t}", .{ target_query, e }),
        },
        .eb_wip = &runner.eb_wip,
        .cmd_src_loc = undefined,
        .src_loc = null,
        .num = 0,
        .name = undefined,
        .prog_node = .none,
        .mtime = .now(runner.io, .real),
        .allow_skip = true,
    };
    defer {
        update.manifest_sr.deinit();
        if (update.compiler) |*comp| comp.deinit();
        update.src_dir.close(runner.io);
        update.prog_node.end();
    }

    // hack around `std.Io.Reader` interface bugs by ensuring the
    // buffer is larger than the longest supported manifest line
    try update.manifest_sr.line_aw.ensureTotalCapacityPrecise(512);
    update.manifest_sr.updateBuffer();

    var skip_delimiter = false;
    while (update.manifest_sr.interface.takeSentinel('\n')) |line| {
        var line_it = std.mem.tokenizeScalar(u8, line, ' ');
        const cmd_src_loc = line_it.next() orelse continue;
        if (skip_delimiter) {
            assert(std.mem.eql(u8, cmd_src_loc, "#}"));
            skip_delimiter = false;
            const unexpected = line_it.rest();
            if (unexpected.len > 0)
                return update.manifest_sr.fail(unexpected, "unexpected argument {q}", .{unexpected});
            continue;
        }
        update.cmd_src_loc = try update.manifest_sr.srcLoc(cmd_src_loc);
        const cmd = std.meta.stringToEnum(Command, std.mem.cutPrefix(u8, cmd_src_loc, "#") orelse
            return update.manifest_sr.fail(cmd_src_loc, "expected command", .{})) orelse
            return update.manifest_sr.fail(cmd_src_loc, "unknown command", .{});
        const arg = switch (cmd) {
            .todo, .exe, .lib, .obj, .update => null,
            .skip, .write, .delete, .check => line_it.next() orelse
                return update.manifest_sr.fail(line_it.rest()[0..0], "missing argument", .{}),
        };
        var contents_file: ?std.Io.File = null;
        defer if (contents_file) |file| file.close(runner.io);
        var contents_buffer: [512]u8 = undefined;
        var contents_impl: union(enum) {
            r: std.Io.Reader,
            dr: DelimitedReader,
            fr: std.Io.File.Reader,
        } = undefined;
        const contents_r: *std.Io.Reader = contents_r: switch (cmd) {
            .skip, .delete => {
                const unexpected = line_it.rest();
                if (unexpected.len > 0) return update.manifest_sr.fail(
                    unexpected,
                    "unexpected argument {q}",
                    .{unexpected},
                );
                contents_impl = .{ .r = .ending_instance };
                break :contents_r &contents_impl.r;
            },
            else => {
                const contents = line_it.rest();
                if (contents.len == 0) {
                    contents_impl = .{ .r = .ending_instance };
                    break :contents_r &contents_impl.r;
                }
                contents: switch (contents[0]) {
                    else => return update.manifest_sr.fail(contents, "invalid contents syntax", .{}),
                    '"' => {
                        if (contents.len < 2 or contents[contents.len - 1] != '"') continue :contents 0;
                        var contents_fw: std.Io.Writer = .fixed(&contents_buffer);
                        switch (std.zig.string_literal.parseWrite(
                            &contents_fw,
                            contents,
                        ) catch |err| switch (err) {
                            error.WriteFailed => return update.manifest_sr.fail(
                                contents,
                                "contents string too long, use inline or file syntax instead",
                                .{},
                            ),
                        }) {
                            .success => {
                                contents_impl = .{ .r = .fixed(contents_fw.buffered()) };
                                break :contents_r &contents_impl.r;
                            },
                            .failure => |err| return update.manifest_sr.fail(
                                contents[err.offset()..][0..0],
                                "{f}",
                                .{err.fmt(contents)},
                            ),
                        }
                    },
                    '{' => {
                        if (contents.len != 1) continue :contents 0;

                        contents_impl = .{
                            .dr = .init(&update.manifest_sr.interface, "#}", &contents_buffer),
                        };
                        skip_delimiter = true;
                        break :contents_r &contents_impl.dr.interface;
                    },
                    '@' => {
                        const args_test = runner.args.@"test" orelse return update.manifest_sr.fail(
                            contents,
                            "contents file requires directory test",
                            .{},
                        );
                        const contents_path = contents[1..];
                        try runner.discoverInput(args_test.input_dir, contents_path);

                        contents_file = args_test.dir.openFile(
                            runner.io,
                            contents_path,
                            .{},
                        ) catch |err| switch (err) {
                            error.Canceled => |e| return e,
                            else => |e| return update.manifest_sr.fail(
                                contents_path,
                                "unable to open contents file: {t}",
                                .{e},
                            ),
                        };
                        contents_impl = .{ .fr = contents_file.?.reader(runner.io, &contents_buffer) };
                        break :contents_r &contents_impl.fr.interface;
                    },
                }
            },
        };

        runner.handleCommand(&update, cmd_src_loc, cmd, arg, contents_r) catch |err| switch (err) {
            error.Canceled,
            error.OutOfMemory,
            error.AlreadyReported,
            error.SkipTest,
            => |e| return e,
            error.ContentsReadFailed => switch (contents_impl) {
                .r => unreachable,
                .dr => switch (manifest_fr.err orelse return error.OutOfMemory) {
                    error.Canceled => |e| return e,
                    else => |e| return update.fail("unable to read manifest: {t}", .{e}),
                },
                .fr => |fr| switch (fr.err.?) {
                    error.Canceled => |e| return e,
                    else => |e| return update.fail("unable to read contents file: {t}", .{e}),
                },
            },
        };
    } else |err| switch (err) {
        error.ReadFailed => switch (manifest_fr.err orelse return error.OutOfMemory) {
            error.Canceled => |e| return e,
            else => |e| return update.fail("unable to read manifest: {t}", .{e}),
        },
        error.StreamTooLong => unreachable,
        error.EndOfStream => {
            const final_line = update.manifest_sr.interface.buffered();
            switch (final_line.len) {
                0 => if (skip_delimiter) return update.manifest_sr.fail(
                    final_line[0..0],
                    "missing \"#}}\" delimiter",
                    .{},
                ),
                else => return update.manifest_sr.fail(
                    final_line[final_line.len..],
                    "missing trailing newline",
                    .{},
                ),
            }
        },
    }
    if (update.compiler) |*comp| {
        if (comp.state != .idle) return update.fail("missing \"#check\"", .{});
        try comp.exit();
    }
}
const Command = enum { todo, skip, exe, lib, obj, write, delete, update, check };
fn handleCommand(
    runner: *Runner,
    update: *Update,
    cmd_src_loc: []const u8,
    cmd: Command,
    arg: ?[]const u8,
    contents_r: *std.Io.Reader,
) (error{ContentsReadFailed} || TestError)!void {
    var name_buffer: [std.Progress.Node.max_name_len]u8 = undefined;
    const cmd_prog_node = runner.prog_node.start(std.mem.print(&name_buffer, "#{t}{s}{s}", .{
        cmd, if (arg) |_| " " else "", arg orelse "",
    }) catch &name_buffer, 0);
    defer cmd_prog_node.end();

    cmd: switch (cmd) {
        .todo => _ = contents_r.discardRemaining() catch |err| switch (err) {
            error.ReadFailed => {},
        },
        .skip => if (!update.allow_skip) return update.manifest_sr.fail(
            cmd_src_loc,
            "\"#skip\" must appear before other commands",
            .{},
        ) else if (std.mem.eql(u8, update.target_query, arg.?)) return error.SkipTest,
        .exe, .lib, .obj => runner.spawnCompiler(&update.compiler, update.src_dir, switch (cmd) {
            else => unreachable,
            .exe => .Exe,
            .lib => .Lib,
            .obj => .Obj,
        }, contents_r, target: {
            var component_it = std.mem.splitBackwardsScalar(u8, update.target_query, '-');
            const pic_str = component_it.next() orelse
                return update.fail("target {q} missing pic", .{update.target_query});
            const pic = std.meta.stringToEnum(Compiler.Target.Pic, pic_str) orelse
                return update.fail("target {q} unknown pic {q}", .{ update.target_query, pic_str });
            const linker_str = component_it.next() orelse
                return update.fail("target {q} missing linker", .{update.target_query});
            const linker = std.meta.stringToEnum(Compiler.Target.Linker, linker_str) orelse
                return update.fail("target {q} unknown linker {q}", .{
                    update.target_query, linker_str,
                });
            const backend_str = component_it.next() orelse
                return update.fail("target {q} missing backend", .{update.target_query});
            const backend = std.meta.stringToEnum(Compiler.Target.Backend, backend_str) orelse
                return update.fail("target {q} unknown backend {q}", .{
                    update.target_query, backend_str,
                });
            const cache_mode_str = component_it.next() orelse
                return update.fail("target {q} missing cache mode", .{update.target_query});
            const cache_mode = std.meta.stringToEnum(Compiler.Target.CacheMode, cache_mode_str) orelse
                return update.fail("target {q} unknown cache mode {q}", .{
                    update.target_query, cache_mode_str,
                });
            const triple_cpu = component_it.rest();
            var triple_len: usize = 0;
            for (0..3) |_| {
                const component_start = triple_len;
                triple_len = if (std.mem.findScalarPos(u8, triple_cpu, triple_len, '-')) |dash_pos|
                    dash_pos + 1
                else
                    return update.fail("target {q} missing cpu", .{update.target_query});
                if (std.mem.eql(u8, triple_cpu[component_start..triple_len], "native-")) break;
            }
            const triple = triple_cpu[0 .. triple_len - 1];
            const cpu = triple_cpu[triple_len..];
            var query_parse_diags: std.Target.Query.ParseOptions.Diagnostics = undefined;
            break :target .{
                .triple = triple,
                .resolved = std.zig.system.resolveTargetQuery(
                    runner.io,
                    std.Target.Query.parse(.{
                        .arch_os_abi = triple,
                        .cpu_features = cpu,
                        .object_format = switch (backend) {
                            .sema, .selfhosted, .llvm => null,
                            .cbe => "c",
                        },
                        .diagnostics = &query_parse_diags,
                    }) catch |err| switch (err) {
                        error.UnknownArchitecture => return update.fail(
                            "unable to resolve target {q}: unknown architecture: {q}",
                            .{ update.target_query, query_parse_diags.unknown_architecture_name.? },
                        ),
                        error.MissingOperatingSystem => return update.fail(
                            "unable to resolve target {q}: missing operating system",
                            .{update.target_query},
                        ),
                        error.UnknownOperatingSystem => return update.fail(
                            "unable to resolve target {q}: unknown operating system: {q}",
                            .{ update.target_query, query_parse_diags.os_name.? },
                        ),
                        error.InvalidOperatingSystemVersion => return update.fail(
                            "unable to resolve target {q}: invalid operating system version",
                            .{update.target_query},
                        ),
                        error.UnknownApplicationBinaryInterface => return update.fail(
                            "unable to resolve target {q}: unknown abi",
                            .{update.target_query},
                        ),
                        error.InvalidAbiVersion => return update.fail(
                            "unable to resolve target {q}: invalid abi version",
                            .{update.target_query},
                        ),
                        error.UnexpectedExtraField => unreachable, // validated above
                        error.UnknownCpuModel => return update.fail(
                            "unable to resolve target {q}: unknown cpu: {q}",
                            .{ update.target_query, query_parse_diags.cpu_name.? },
                        ),
                        error.UnknownCpuFeature => return update.fail(
                            "unable to resolve target {q}: unknown cpu feature: {q}",
                            .{ update.target_query, query_parse_diags.unknown_feature_name.? },
                        ),
                        error.UnknownObjectFormat => return update.fail(
                            "unable to resolve target {q}: unknown object format",
                            .{update.target_query},
                        ),
                    },
                ) catch |err| switch (err) {
                    error.Canceled => |e| return e,
                    else => |e| return update.fail("unable to resolve target {q}: {t}", .{ triple, e }),
                },
                .cache_mode = cache_mode,
                .backend = backend,
                .linker = linker,
                .pic = pic,
            };
        }) catch |err| switch (err) {
            error.Canceled, error.OutOfMemory, error.AlreadyReported, error.SkipTest => |e| return e,
            error.ReadFailed => return error.ContentsReadFailed,
        },
        .write => {
            const file = update.src_dir.createFile(runner.io, arg.?, .{}) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| return update.manifest_sr.fail(arg.?, "unable to create file: {t}", .{e}),
            };
            defer file.close(runner.io);
            var fw_buffer: [512]u8 = undefined;
            var fw = file.writer(runner.io, &fw_buffer);
            _ = contents_r.streamRemaining(&fw.interface) catch |err| switch (err) {
                error.ReadFailed => return error.ContentsReadFailed,
                error.WriteFailed => switch (fw.err.?) {
                    error.Canceled => |e| return e,
                    else => |e| return update.manifest_sr.fail(
                        arg.?,
                        "unable to write file: {t}",
                        .{e},
                    ),
                },
            };
            fw.flush() catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| return update.manifest_sr.fail(
                    arg.?,
                    "unable to write file: {t}",
                    .{e},
                ),
            };
            file.setTimestamps(runner.io, .{ .modify_timestamp = .{
                .new = update.mtime,
            } }) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| return update.manifest_sr.fail(
                    arg.?,
                    "unable to update file timestamps: {t}",
                    .{e},
                ),
            };
        },
        .delete => update.src_dir.deleteFile(runner.io, arg.?) catch |err| switch (err) {
            error.Canceled => |e| return e,
            else => |e| return update.manifest_sr.fail(arg.?, "unable to delete file: {t}", .{e}),
        },
        .update => {
            update.src_loc = update.cmd_src_loc;
            const comp = &(update.compiler orelse return update.manifest_sr.fail(
                cmd_src_loc,
                "missing \"#exe\", \"#lib\", or \"#obj\"",
                .{},
            ));
            update.prog_node.end();
            const arena = runner.arena.allocator();
            update.name = switch (contents_r.readSliceShort(&name_buffer) catch |err| switch (err) {
                error.ReadFailed => return error.ContentsReadFailed,
            }) {
                0 => switch (update.num) {
                    0 => "initial update",
                    else => |num| try arena.print("update {d}", .{num}),
                },
                else => |name_len| try arena.dupe(u8, name_buffer[0..name_len]),
            };
            update.prog_node = runner.prog_node.start(update.name, 0);
            _ = contents_r.discardRemaining() catch |err| switch (err) {
                error.ReadFailed => return error.ContentsReadFailed,
            };
            if (comp.state != .idle)
                return update.manifest_sr.fail(cmd_src_loc, "missing \"#check\"", .{});
            comp.client.serveBodylessMessage(.update) catch |err| switch (err) {
                error.WriteFailed => switch (comp.fw.err.?) {
                    error.Canceled => |e| return e,
                    else => |e| return update.fail("unable to send message to compiler: {t}", .{e}),
                },
            };
            comp.state = .update;
            update.mtime = update.mtime.addDuration(.fromSeconds(2));
            update.num += 1;
        },
        .check => {
            const check = std.meta.stringToEnum(Compiler.Check, arg.?) orelse
                return update.manifest_sr.fail(arg.?, "unknown check", .{});
            const expected =
                contents_r.allocRemaining(runner.gpa, .unlimited) catch |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    error.ReadFailed => return error.ContentsReadFailed,
                    error.StreamTooLong => unreachable, // .unlimited
                };
            defer runner.gpa.free(expected);

            const comp = &(update.compiler orelse return update.manifest_sr.fail(
                cmd_src_loc,
                "missing \"#exe\", \"#lib\", or \"#obj\"",
                .{},
            ));
            if (comp.state != .update)
                return update.manifest_sr.fail(cmd_src_loc, "missing \"#update\"", .{});
            const stderr = comp.mr.reader(1);
            while (try comp.receiveMessage()) |message| switch (message.tag) {
                else => {}, // Ignore other messages,
                .config => {
                    var body_r: std.Io.Reader = .fixed(message.body);
                    comp.config = body_r.takeStruct(std.zig.Server.Message.Config, .little) catch
                        unreachable;
                },
                .emit_digest => {
                    var body_r: std.Io.Reader = .fixed(message.body);
                    _ = body_r.takeStruct(std.zig.Server.Message.EmitDigest, .little) catch
                        unreachable;
                    const digest = body_r.takeArray(std.Build.Cache.bin_digest_len) catch
                        unreachable;
                    try comp.checkSuccess(update, check, expected, digest);
                },
                .error_bundle => {
                    const eb = std.zig.Server.allocErrorBundle(
                        runner.arena.allocator(),
                        message.body,
                    ) catch |err| switch (err) {
                        error.OutOfMemory => |e| return e,
                        error.EndOfStream => return update.fail("error bundle truncated", .{}),
                    };
                    switch (check) {
                        .errors => {
                            var error_aw: std.Io.Writer.Allocating = .init(runner.gpa);
                            defer error_aw.deinit();
                            eb.renderToWriter(.{
                                .include_reference_trace = false,
                                .include_source_line = false,
                                .include_log_text = true,
                            }, &error_aw.writer) catch |err| switch (err) {
                                error.WriteFailed => return error.OutOfMemory,
                            };
                            const actual = error_aw.written();
                            std.mem.replaceScalar(u8, actual, '\\', '/');
                            std.testing.expectEqualStrings(expected, actual) catch |err| switch (err) {
                                error.TestExpectedEqual => return update.fail(
                                    "errors did not match expected",
                                    .{},
                                ),
                            };
                        },
                        .stdout, .exit, .llvm, .gdb, .lldb => if (eb.errorMessageCount() > 0) {
                            eb.renderToStderr(runner.io, .{}, .auto) catch |err| switch (err) {
                                error.Canceled => {},
                                else => {},
                            };
                            return update.fail("unexpected compile errors", .{});
                        },
                    }
                    comp.state = .idle;
                    break :cmd;
                },
            };

            const stderr_buffered = stderr.buffered();
            if (stderr_buffered.len > 0) {
                if (comp.allow_compiler_stderr) {
                    std.log.info("#update {q} stderr:\n{s}", .{ update.name, stderr_buffered });
                } else return update.fail("unexpected stderr:\n{s}", .{stderr_buffered});
            }

            return update.fail("compiler failed to send terminating error_bundle", .{});
        },
    }
    assert((contents_r.discardRemaining() catch |err| switch (err) {
        error.ReadFailed => return error.ContentsReadFailed,
    }) == 0);
    switch (cmd) {
        .todo, .skip => {},
        else => update.allow_skip = false,
    }
}

fn spawnCompiler(
    runner: *Runner,
    compiler: *?Compiler,
    src_dir: std.Io.Dir,
    output_mode: std.lang.OutputMode,
    args_r: *std.Io.Reader,
    target: Compiler.Target,
) (std.Io.Reader.ShortError || TestError)!void {
    const gpa = runner.gpa;
    const arena = runner.arena.allocator();
    if (compiler.*) |*comp| {
        comp.deinit();
        compiler.* = null;
    }
    errdefer compiler.* = null;

    compiler.* = .{
        .runner = runner,
        .allow_compiler_stderr = true,
        .output_mode = output_mode,
        .state = .idle,
        .child = undefined,
        .fw = undefined,
        .mr_buffer = undefined,
        .mr = undefined,
        .client = undefined,
        .target = target,
        .emit_name = null,
        .prog_node = .none,
        .config = null,
    };
    const comp = &compiler.*.?;

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.appendSlice(gpa, &.{
        "zig",
        switch (output_mode) {
            .Obj => "build-obj",
            .Exe => "build-exe",
            .Lib => "build-lib",
        },
        "-target",
        comp.target.triple,
        "--cache-dir",
        ".zig-cache",
    });
    switch (comp.target.cache_mode) {
        .whole => {},
        .incremental => try argv.append(gpa, "-fincremental"),
    }
    var path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    if (comp.target.backend) |b| switch (b) {
        .sema => try argv.append(gpa, "-fno-emit-bin"),
        .selfhosted => try argv.append(gpa, "-fno-llvm"),
        .llvm => try argv.append(gpa, "-fllvm"),
        .cbe => try argv.append(gpa, "-ofmt=c"),
    } else try argv.appendSlice(gpa, &.{ "-I", path_buffer[0 .. (runner.args.lib_dir orelse
        return runner.fail("\"--lib=path/to/lib\" arg required with cbe", .{}))
        .realPath(runner.io, &path_buffer) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| return runner.fail("unable to get lib path: {t}", .{e}),
    }] });
    switch (comp.target.linker) {
        .lld => try argv.appendSlice(gpa, &.{ "-flld", "-fno-new-linker" }),
        .old => try argv.appendSlice(gpa, &.{ "-fno-lld", "-fno-new-linker" }),
        .new => try argv.appendSlice(gpa, &.{ "-fno-lld", "-fnew-linker" }),
    }
    switch (comp.target.pic) {
        .nopic => try argv.appendSlice(gpa, &.{ "-fno-PIC", "-fno-PIE" }),
        .pic => try argv.appendSlice(gpa, &.{ "-fPIC", "-fno-PIE" }),
        .pie => try argv.appendSlice(gpa, &.{ "-fPIC", "-fPIE" }),
    }

    var need: usize = 1;
    var state: enum { positional, @"--name" } = .positional;
    var root_name: ?[]const u8 = null;
    var emit: ?std.zig.EmitArtifact = .bin;
    while (true) {
        const done = if (args_r.fill(need)) false else |err| switch (err) {
            error.ReadFailed => |e| return e,
            error.EndOfStream => true,
        };
        const buffered = args_r.buffered();
        const start = std.mem.findNone(u8, buffered, "\n ") orelse buffered.len;
        const end = std.mem.findAnyPos(u8, buffered, start, "\n ") orelse buffered.len;
        if (end - start > 0) {
            const arg = try arena.dupe(u8, buffered[start..end]);
            switch (state) {
                .positional => if (std.mem.eql(u8, arg, "--name")) {
                    state = .@"--name";
                } else if (std.mem.cutPrefix(u8, arg, "-ofmt=")) |ofmt| {
                    comp.target.resolved.ofmt =
                        std.meta.stringToEnum(std.Target.ObjectFormat, ofmt) orelse
                        return runner.fail("invalid object format {q}", .{ofmt});
                } else if (std.mem.cutPrefix(u8, arg, "-M")) |module| {
                    var module_name_it = std.mem.splitScalar(u8, module, '=');
                    const module_name = module_name_it.next().?;
                    root_name = root_name orelse module_name;
                } else if (std.mem.eql(u8, arg, "-femit-asm")) {
                    emit = .@"asm";
                } else if (std.mem.eql(u8, arg, "-femit-bin")) {
                    emit = .bin;
                } else if (std.mem.eql(u8, arg, "-femit-docs")) {
                    emit = .docs;
                } else if (std.mem.eql(u8, arg, "-femit-h")) {
                    emit = .h;
                } else if (std.mem.eql(u8, arg, "-femit-implib")) {
                    emit = .implib;
                } else if (std.mem.eql(u8, arg, "-femit-llvm-bc")) {
                    emit = .llvm_bc;
                } else if (std.mem.eql(u8, arg, "-femit-llvm-ir")) {
                    emit = .llvm_ir;
                } else if (std.mem.eql(u8, arg, "-fno-emit-asm")) {
                    if (emit == .@"asm") emit = null;
                } else if (std.mem.eql(u8, arg, "-fno-emit-bin")) {
                    if (emit == .bin) emit = null;
                } else if (std.mem.eql(u8, arg, "-fno-emit-docs")) {
                    if (emit == .docs) emit = null;
                } else if (std.mem.eql(u8, arg, "-fno-emit-h")) {
                    if (emit == .h) emit = null;
                } else if (std.mem.eql(u8, arg, "-fno-emit-implib")) {
                    if (emit == .implib) emit = null;
                } else if (std.mem.eql(u8, arg, "-fno-emit-llvm-bc")) {
                    if (emit == .llvm_bc) emit = null;
                } else if (std.mem.eql(u8, arg, "-fno-emit-llvm-ir")) {
                    if (emit == .llvm_ir) emit = null;
                },
                .@"--name" => {
                    root_name = arg;
                    state = .positional;
                },
            }
            try argv.append(gpa, arg);
            args_r.toss(end);
            need = 1;
        } else {
            args_r.toss(start);
            need = buffered.len - start + 1;
        }
        if (done) break;
    }
    try argv.append(gpa, "--listen=-");

    comp.emit_name = if (emit) |e| try e.cacheName(arena, .{
        .root_name = root_name orelse "root",
        .cpu_arch = comp.target.resolved.cpu.arch,
        .os_tag = comp.target.resolved.os.tag,
        .ofmt = comp.target.resolved.ofmt,
        .abi = comp.target.resolved.abi,
        .output_mode = output_mode,
    }) else null;

    var name_buffer: [std.Progress.Node.max_name_len]u8 = undefined;
    comp.prog_node = runner.prog_node.start(std.mem.print(&name_buffer, "compile {s}", .{
        comp.emit_name orelse root_name orelse "root",
    }) catch &name_buffer, 0);
    errdefer comp.prog_node.end();

    comp.child = std.process.spawn(runner.io, .{
        .exe = .{ .file = runner.args.zig_exe orelse
            return runner.fail("missing \"--zig=path/to/zig\" arg", .{}) },
        .argv = argv.items,
        .cwd = .{ .dir = src_dir },
        .progress_node = comp.prog_node,
        .stdin = .pipe,
        .stdout = .pipe,
        .stderr = .pipe,
    }) catch |err| switch (err) {
        error.Canceled, error.OutOfMemory => |e| return e,
        error.OperationUnsupported => return error.SkipTest,
        else => |e| return runner.fail("unable to spawn compiler: {t}", .{e}),
    };
    comp.fw = comp.child.stdin.?.writerStreaming(runner.io, &.{});
    comp.mr.init(gpa, runner.io, comp.mr_buffer.toStreams(), &.{
        comp.child.stdout.?, comp.child.stderr.?,
    });
    comp.client = .{ .in = comp.mr.reader(0), .out = &comp.fw.interface };
}
const Compiler = struct {
    runner: *Runner,
    allow_compiler_stderr: bool,
    output_mode: std.lang.OutputMode,
    state: enum { idle, update },
    child: std.process.Child,
    fw: std.Io.File.Writer,
    mr_buffer: std.Io.File.MultiReader.Buffer(2),
    mr: std.Io.File.MultiReader,
    client: std.zig.Client,
    target: Compiler.Target,
    emit_name: ?[]const u8,
    prog_node: std.Progress.Node,
    config: ?std.zig.Server.Message.Config,

    const Target = struct {
        triple: []const u8,
        resolved: std.Target,
        cache_mode: CacheMode,
        backend: ?Backend,
        linker: Linker,
        pic: Pic,

        const CacheMode = enum { whole, incremental };

        const Backend = enum {
            /// Run semantic analysis only. Runtime output will not be tested, but we still verify
            /// that compilation succeeds. Corresponds to `-fno-emit-bin`.
            sema,
            /// Use the self-hosted code generation backend for this target.
            /// Corresponds to `-fno-llvm -fno-lld`.
            selfhosted,
            /// Use the LLVM backend.
            /// Corresponds to `-fllvm -flld`.
            llvm,
            /// Use the C backend. The output is compiled with `zig cc`.
            /// Corresponds to `-ofmt=c`.
            cbe,
        };

        const Linker = enum { old, new, lld };

        const Pic = enum { nopic, pic, pie };
    };

    fn deinit(comp: *Compiler) void {
        comp.child.kill(comp.runner.io);
        comp.prog_node.end();
        comp.prog_node = .none;
    }

    fn receiveMessage(comp: *Compiler) FailError!?struct {
        tag: std.zig.Server.Message.Tag,
        body: []const u8,
    } {
        const header = comp.client.receiveMessageWithMultiReader(
            &comp.mr,
            .none,
        ) catch |err| switch (err) {
            error.Canceled, error.OutOfMemory => |e| return e,
            error.Timeout => unreachable,
            error.EndOfStream => return null,
            else => |e| return comp.runner.fail("unable to receive message: {t}", .{e}),
        };
        const body = comp.client.in.take(header.bytes_len) catch unreachable;
        const stderr = comp.mr.reader(1);
        if (stderr.bufferedLen() > 0) {
            if (comp.allow_compiler_stderr) {
                std.log.info("{t} stderr:\n{s}", .{ header.tag, stderr.buffered() });
            } else {
                return comp.runner.fail("{t} unexpected stderr:\n{s}", .{
                    header.tag, stderr.buffered(),
                });
            }
            stderr.tossBuffered();
        }
        return .{ .tag = header.tag, .body = body };
    }

    const Check = enum { errors, stdout, exit, llvm, gdb, lldb };
    fn checkSuccess(
        comp: *Compiler,
        update: *const Update,
        check: Check,
        expected: []const u8,
        digest: *const std.Build.Cache.BinDigest,
    ) TestError!void {
        const runner = comp.runner;
        const gpa = runner.gpa;
        const arena = runner.arena.allocator();
        const io = runner.io;
        const target = &comp.target.resolved;
        const emit_name = switch (check) {
            else => comp.emit_name orelse
                return update.fail("unable to perform {t} check without emitted file", .{check}),
            .errors => return, // expect no emitted file
        };
        const config = &comp.config.?;
        var path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;

        // After a failure while running a script, the debugger starts accepting commands from stdin, and
        // because it is empty, the debugger exits normally with status 0. Choose a non-zero status to
        // return from the debugger script instead to detect it running to completion and indicate success.
        const debug_success = 99;

        const tmp_dir_path = ".zig-cache" ++ std.Io.Dir.path.sep_str ++ "tmp";
        update.src_dir.createDirPath(io, tmp_dir_path) catch |err| switch (err) {
            error.Canceled => |e| return e,
            else => |e| return update.fail("unable to create dir path {q}: {t}", .{
                tmp_dir_path, e,
            }),
        };

        const emit_path = emit_path: {
            const emit_dir_path = ".zig-cache" ++ std.Io.Dir.path.sep_str ++
                "o" ++ std.Io.Dir.path.sep_str ++ std.Build.Cache.binToHex(digest.*);
            const emit_path =
                try std.Io.Dir.path.join(arena, &.{ emit_dir_path, emit_name });
            break :emit_path switch (comp.target.backend.?) {
                .sema => return,
                .selfhosted, .llvm => emit_path,
                .cbe => try comp.compileC(update.src_dir, emit_path),
            };
        };

        const emit_file = update.src_dir.openFile(io, emit_path, .{}) catch |err| switch (err) {
            error.Canceled => |e| return e,
            else => |e| return update.fail("unable to open {q}: {t}", .{ emit_path, e }),
        };
        defer emit_file.close(io);

        var argv: std.ArrayList([]const u8) = try .initCapacity(arena, 5 + 4);
        var environ_map: std.process.Environ.Map = .init(gpa);
        defer environ_map.deinit();
        try environ_map.putAll(&runner.environ_map);
        var script_buffer: [512]u8 = undefined;
        const script_path, const script_file, var script_fw = script: switch (check) {
            .errors => unreachable,
            .stdout, .exit => .{ undefined, null, undefined },
            .llvm => {
                var emit_fr = emit_file.reader(io, &.{});
                const actual = emit_fr.interface.readAllocAll(arena, @intCast(
                    emit_fr.getSize() catch |err| switch (err) {
                        error.Canceled => |e| return e,
                        else => |e| return runner.fail("unable to get size of {q}: {t}", .{ emit_path, e }),
                    },
                )) catch |err| switch (switch (err) {
                    error.OutOfMemory => |e| return e,
                    error.EndOfStream => |e| e,
                    error.ReadFailed => emit_fr.err.?,
                }) {
                    error.Canceled => |e| return e,
                    else => |e| return runner.fail("unable to read {q}: {t}", .{ emit_path, e }),
                };
                return update.checkMatches("LLVM IR", expected, actual);
            },
            .gdb => {
                const script_path = tmp_dir_path ++ std.Io.Dir.path.sep_str ++ "gdb";
                argv.appendSliceAssumeCapacity(&.{ "gdb", "--batch", "--command", script_path, "--args" });
                const script_file =
                    update.src_dir.createFile(io, script_path, .{}) catch |err| switch (err) {
                        error.Canceled => |e| return e,
                        else => |e| return update.fail("unable to create debugger script {q}: {t}", .{
                            script_path, e,
                        }),
                    };
                errdefer script_file.close(io);
                var script_fw = script_file.writer(io, &script_buffer);
                script_fw.interface.writeAll(
                    \\set remotetimeout 0
                    \\
                    \\
                ) catch |err| switch (err) {
                    error.WriteFailed => switch (script_fw.err.?) {
                        error.Canceled => |e| return e,
                        else => |e| return update.fail("unable to write debugger script {q}: {t}", .{
                            script_path, e,
                        }),
                    },
                };
                break :script .{ script_path, script_file, script_fw };
            },
            .lldb => {
                const script_path = tmp_dir_path ++ std.Io.Dir.path.sep_str ++ "lldb";
                argv.appendSliceAssumeCapacity(&.{ "lldb", "--batch", "--source", script_path, "--" });
                try environ_map.put("LANG", "C.UTF-8"); // affects output formatting
                const script_file =
                    update.src_dir.createFile(io, script_path, .{}) catch |err| switch (err) {
                        error.Canceled => |e| return e,
                        else => |e| return update.fail("unable to create debugger script {q}: {t}", .{
                            script_path, e,
                        }),
                    };
                errdefer script_file.close(io);
                var script_fw = script_file.writer(io, &script_buffer);
                script_fw.interface.writeAll(
                    \\settings set frame-format 'frame #${frame.index}:{ ${module.file.basename}{\`${function.name-with-args}{${frame.no-debug}${function.pc-offset}}}}{ at ${line.file.basename}:${line.number}{:${line.column}}}{${function.is-optimized} [opt]}{${frame.is-artificial} [artificial]}\n'
                    \\settings set plugin.process.gdb-remote.packet-timeout 0
                    \\settings set stop-line-count-after 0
                    \\settings set stop-line-count-before 0
                    \\
                ) catch |err| switch (err) {
                    error.WriteFailed => switch (script_fw.err.?) {
                        error.Canceled => |e| return e,
                        else => |e| return update.fail("unable to write debugger script {q}: {t}", .{
                            script_path, e,
                        }),
                    },
                };
                break :script .{ script_path, script_file, script_fw };
            },
        };
        defer if (script_file) |file| file.close(io);

        const need_cross_libc =
            target.os.tag == .linux and config.flags.link_libc and config.flags.link_mode == .dynamic;
        const use_executor = use_executor: switch (std.zig.system.getExternalExecutor(io, target, .{
            .host_cpu_arch = runner.host.cpu.arch,
            .host_os_tag = runner.host.os.tag,
            .qemu_fixes_dl = need_cross_libc and runner.args.libc_runtimes_dir != null,
            .link_mode = config.flags.link_mode,
            .link_libc = config.flags.link_libc,
        })) {
            .bad_dl, .bad_os_or_cpu => {
                // This binary cannot be executed on this host.
                if (!runner.args.quiet) std.log.warn("skipping execution because host {q} cannot " ++
                    "execute binaries for foreign target {q}", .{
                    try runner.host.zigTriple(arena), comp.target.triple,
                });
                return;
            },
            .native => false,
            .darling => |executor| if (runner.args.enable_darling) {
                argv.appendAssumeCapacity(executor);
                break :use_executor true;
            } else continue :use_executor .bad_os_or_cpu,
            .qemu => |executor| if (runner.args.enable_qemu) {
                argv.appendAssumeCapacity(executor);
                if (need_cross_libc) {
                    const libc_runtimes_dir = runner.args.libc_runtimes_dir orelse
                        continue :use_executor .bad_os_or_cpu;
                    const libc_runtimes_path = path_buffer[0 .. libc_runtimes_dir.realPath(
                        io,
                        &path_buffer,
                    ) catch |err| switch (err) {
                        error.Canceled => |e| return e,
                        else => |e| return update.fail("unable to get libc runtimes path: {t}", .{e}),
                    }];
                    argv.appendSliceAssumeCapacity(&.{ "-L", try std.Io.Dir.path.join(arena, &.{
                        libc_runtimes_path,
                        try if (target.isGnuLibC()) std.zig.target.glibcRuntimeTriple(
                            arena,
                            target.cpu.arch,
                            target.os.tag,
                            target.abi,
                        ) else if (target.isMuslLibC()) std.zig.target.muslRuntimeTriple(
                            arena,
                            target.cpu.arch,
                            target.abi,
                        ) else unreachable,
                    }) });
                }
                break :use_executor true;
            } else continue :use_executor .bad_os_or_cpu,
            .rosetta => if (runner.args.enable_rosetta)
                true
            else
                continue :use_executor .bad_os_or_cpu,
            .wine => |executor| if (runner.args.enable_wine) {
                if (!environ_map.contains("WINEDEBUG")) try environ_map.put("WINEDEBUG", "-all");
                argv.appendAssumeCapacity(executor);
                break :use_executor true;
            } else continue :use_executor .bad_os_or_cpu,
            .wasmtime => |executor| if (runner.args.enable_wasmtime) {
                if (!environ_map.contains("WASMTIME_BACKTRACE_DETAILS"))
                    try environ_map.put("WASMTIME_BACKTRACE_DETAILS", "1");
                argv.appendSliceAssumeCapacity(&.{ executor, "--dir=.", "-Sinherit-env" });
                break :use_executor true;
            } else continue :use_executor .bad_os_or_cpu,
        };
        argv.appendAssumeCapacity(emit_path);

        if (!runner.args.quiet) std.log.info("{f}", .{std.zig.SubprocessCommand{
            .argv = argv.items,
            .cwd = cwd_path: {
                break :cwd_path path_buffer[0 .. update.src_dir.realPath(
                    io,
                    &path_buffer,
                ) catch |err| switch (err) {
                    error.Canceled => |e| return e,
                    else => break :cwd_path null,
                }];
            },
            .parent_env = &runner.environ_map,
            .child_env = &environ_map,
        }});

        var name_buffer: [std.Progress.Node.max_name_len]u8 = undefined;
        const run_prog_node = runner.prog_node.start(std.mem.print(&name_buffer, "run {s}", .{
            emit_name,
        }) catch &name_buffer, 0);
        defer run_prog_node.end();

        const result = std.process.run(gpa, io, .{
            .exe = exe: switch (check) {
                .errors, .llvm => unreachable,
                .stdout, .exit => if (use_executor) .search else .{ .file = emit_file },
                .gdb, .lldb => {
                    if (use_executor) {
                        if (!runner.args.quiet) std.log.warn(
                            "skipping debugging of {q} via executor for foreign target {q}",
                            .{ emit_path, comp.target.triple },
                        );
                        return;
                    }

                    var line_it = std.mem.splitScalar(u8, expected, '\n');
                    while (line_it.next()) |line| if (std.mem.cutPrefix(u8, line, switch (check) {
                        else => unreachable,
                        .gdb => "(gdb) ",
                        .lldb => "(lldb) ",
                    })) |cmd| script_fw.interface.print(
                        \\{s}
                        \\
                    , .{cmd}) catch |err| switch (err) {
                        error.WriteFailed => switch (script_fw.err.?) {
                            error.Canceled => |e| return e,
                            else => |e| return update.fail("unable to write debugger script {q}: {t}", .{
                                script_path, e,
                            }),
                        },
                    };
                    script_fw.interface.print(
                        \\
                        \\quit {d}
                        \\
                    , .{debug_success}) catch |err| switch (err) {
                        error.WriteFailed => switch (script_fw.err.?) {
                            error.Canceled => |e| return e,
                            else => |e| return update.fail("unable to write debugger script {q}: {t}", .{
                                script_path, e,
                            }),
                        },
                    };
                    script_fw.flush() catch |err| switch (err) {
                        error.Canceled => |e| return e,
                        else => |e| return update.fail("unable to write debugger script {q}: {t}", .{
                            script_path, e,
                        }),
                    };
                    break :exe .{ .file = switch (check) {
                        else => unreachable,
                        .gdb => runner.args.gdb_exe,
                        .lldb => runner.args.lldb_exe,
                    } orelse {
                        if (!runner.args.quiet) std.log.warn(
                            "skipping debugging of {q} because of missing \"--{t}=path/to/{[1]t}\" arg",
                            .{ emit_path, check },
                        );
                        return;
                    } };
                },
            },
            .argv = argv.items,
            .cwd = .{ .dir = update.src_dir },
            .environ_map = &environ_map,
            .progress_node = run_prog_node,
        }) catch |err| if (use_executor) {
            // Chances are the foreign executor isn't available. Skip this evaluation.
            if (!runner.args.quiet) std.log.warn(
                "skipping execution of {q} via executor for foreign target {q}: {t}",
                .{ emit_path, comp.target.triple, err },
            );
            return;
        } else return update.fail("unable to run the generated executable {q}: {t}", .{
            emit_path, err,
        });
        defer {
            gpa.free(result.stdout);
            gpa.free(result.stderr);
        }
        switch (result.term) {
            .exited => |code| switch (check) {
                .errors, .llvm => unreachable,
                .stdout => {
                    if (code != 0) return update.fail(
                        "generated executable {q} failed with code {d}",
                        .{ emit_path, code },
                    );
                    std.testing.expectEqualStrings(
                        expected,
                        result.stdout,
                    ) catch |err| switch (err) {
                        error.TestExpectedEqual => return update.fail(
                            "stdout did not match expected",
                            .{},
                        ),
                    };
                },
                .exit => {
                    var actual_code_buffer: [std.fmt.count("{d}", .{std.math.maxInt(u8)})]u8 =
                        undefined;
                    std.testing.expectEqualStrings(
                        expected,
                        std.mem.print(&actual_code_buffer, "{d}", .{code}) catch unreachable,
                    ) catch |err| switch (err) {
                        error.TestExpectedEqual => return update.fail(
                            "exit code did not match expected",
                            .{},
                        ),
                    };
                },
                .gdb, .lldb => {
                    if (code != debug_success) return update.fail(
                        "debugging {q} failed with code {d}",
                        .{ emit_path, code },
                    );
                    try update.checkMatches("stdout", expected, result.stdout);
                },
            },
            .signal => |sig| return update.fail(
                "generated executable {q} terminated with signal {t}",
                .{ emit_path, sig },
            ),
            .stopped => |sig| return update.fail(
                "generated executable {q} stopped with signal {t}",
                .{ emit_path, sig },
            ),
            .unknown => return update.fail(
                "generated executable {q} terminated unexpectedly",
                .{emit_path},
            ),
        }
        if (!use_executor and result.stderr.len > 0) {
            std.log.err("#update {q} generated executable {q} had unexpected stderr:\n{s}", .{
                update.name, emit_path, result.stderr,
            });
        }
    }

    fn compileC(parent_comp: *Compiler, src_dir: std.Io.Dir, c_path: []const u8) TestError![]const u8 {
        const runner = parent_comp.runner;
        const arena = runner.arena.allocator();
        var compiler: ?Compiler = null;
        var args_r: std.Io.Reader = .fixed(c_path);
        parent_comp.runner.spawnCompiler(&compiler, src_dir, parent_comp.output_mode, &args_r, .{
            .triple = parent_comp.target.triple,
            .resolved = resolved: {
                var resolved = parent_comp.target.resolved;
                resolved.ofmt = .default(resolved.os.tag, resolved.cpu.arch);
                break :resolved resolved;
            },
            .cache_mode = .whole,
            .backend = null,
            .linker = .lld,
            .pic = parent_comp.target.pic,
        }) catch |err| switch (err) {
            error.Canceled, error.OutOfMemory, error.AlreadyReported, error.SkipTest => |e| return e,
            error.ReadFailed => unreachable, // .fixed
        };
        const comp = &compiler.?;
        defer comp.deinit();
        comp.client.serveBodylessMessage(.update) catch |err| switch (err) {
            error.WriteFailed => switch (comp.fw.err.?) {
                error.Canceled => |e| return e,
                else => |e| return runner.fail("unable to send message: {t}", .{e}),
            },
        };
        comp.state = .update;
        var emit_path: ?[]const u8 = null;
        while (try comp.receiveMessage()) |message| switch (message.tag) {
            else => {}, // Ignore other messages,
            .config => {
                var body_r: std.Io.Reader = .fixed(message.body);
                comp.config =
                    body_r.takeStruct(std.zig.Server.Message.Config, .little) catch unreachable;
            },
            .emit_digest => {
                var body_r: std.Io.Reader = .fixed(message.body);
                _ = body_r.takeStruct(std.zig.Server.Message.EmitDigest, .little) catch unreachable;
                const digest = body_r.takeArray(std.Build.Cache.bin_digest_len) catch unreachable;

                const emit_dir = ".zig-cache" ++ std.Io.Dir.path.sep_str ++
                    "o" ++ std.Io.Dir.path.sep_str ++ std.Build.Cache.binToHex(digest.*);
                emit_path = try std.Io.Dir.path.join(arena, &.{ emit_dir, comp.emit_name.? });
            },
            .error_bundle => {
                const eb = std.zig.Server.allocErrorBundle(
                    runner.arena.allocator(),
                    message.body,
                ) catch |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    error.EndOfStream => return runner.fail("error bundle truncated", .{}),
                };
                if (eb.errorMessageCount() > 0) {
                    try runner.eb_wip.addBundleAsRoots(eb);
                    return runner.fail("unexpected compile errors", .{});
                }
                comp.state = .idle;
                break;
            },
        };
        try comp.exit();
        return emit_path.?;
    }

    fn exit(comp: *Compiler) FailError!void {
        const runner = comp.runner;
        comp.client.serveBodylessMessage(.exit) catch |err| switch (err) {
            error.WriteFailed => switch (comp.fw.err.?) {
                error.Canceled => |e| return e,
                error.BrokenPipe => {},
                else => |e| return comp.runner.fail("unable to send exit: {t}", .{e}),
            },
        };
        comp.child.stdin.?.close(runner.io);
        comp.child.stdin = null;
        while (try comp.receiveMessage()) |_| {}
        if (comp.client.in.bufferedLen() > 0) return runner.fail("unable to receive messages", .{});
        const term = comp.child.wait(runner.io) catch |err|
            return runner.fail("child process failed: {t}", .{err});
        switch (term) {
            .exited => |code| if (code != 0)
                return runner.fail("compiler failed with code {d}", .{code}),
            .signal => |sig| return runner.fail("compiler terminated with signal {t}", .{sig}),
            .stopped => |sig| return runner.fail("compiler stopped unexpectedly with signal {t}", .{
                sig,
            }),
            .unknown => return runner.fail("compiler terminated unexpectedly", .{}),
        }
    }
};

const ClientArgs = struct {
    test_file: std.Io.File,
    test_kind: std.Io.File.Kind,
    zig_exe: std.Io.File,
    lib_dir: std.Io.Dir,
    src_dir: std.Io.Dir,
    targets: std.ArrayList([]const u8),

    libc_runtimes_dir: ?std.Io.Dir,
    enable_darling: bool,
    enable_qemu: bool,
    enable_rosetta: bool,
    enable_wine: bool,
    enable_wasmtime: bool,

    gdb_exe: ?std.Io.File,
    lldb_exe: ?std.Io.File,

    quiet: bool,
    keep_src: bool,
};
pub fn main(init: std.process.Init) (std.mem.Allocator.Error || std.Io.Cancelable)!u8 {
    const fatal = struct {
        fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
            std.log.err(fmt, args);
            std.process.exit(1);
        }
    }.fatal;
    const arena = init.arena.allocator();

    var test_path_arg: ?[]const u8 = null;
    var zig_path_arg: ?[]const u8 = null;
    var lib_path_arg: ?[]const u8 = null;
    var libc_runtimes_path_arg: ?[]const u8 = null;
    var gdb_path_arg: ?[]const u8 = null;
    var lldb_path_arg: ?[]const u8 = null;
    var args: ClientArgs = .{
        .test_file = undefined,
        .test_kind = undefined,
        .zig_exe = undefined,
        .lib_dir = undefined,
        .src_dir = undefined,
        .targets = .empty,

        .libc_runtimes_dir = null,
        .enable_darling = false,
        .enable_qemu = false,
        .enable_rosetta = false,
        .enable_wine = false,
        .enable_wasmtime = false,

        .gdb_exe = null,
        .lldb_exe = null,

        .quiet = false,
        .keep_src = false,
    };
    defer args.targets.deinit(init.gpa);

    var arg_it = try init.minimal.args.iterateAllocator(arena);
    const self = arg_it.next().?;
    while (arg_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--listen=-")) {
            var stdin_buffer: [512]u8 = undefined;
            var stdin: std.Io.File.Reader = .initStreaming(.stdin(), init.io, &stdin_buffer);
            var stdout_buffer: [512]u8 = undefined;
            var stdout: std.Io.File.Writer = .initStreaming(.stdout(), init.io, &stdout_buffer);
            var runner: Runner = .{
                .gpa = init.gpa,
                .arena = init.arena,
                .environ_map = init.minimal.environ.createMap(init.gpa) catch |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    else => |e| fatal("unable to get environment: {t}", .{e}),
                },
                .io = init.io,
                .prog_node = std.Progress.start(init.io, .{}),
                .args = .{
                    .@"test" = null,
                    .manifest_file = null,
                    .zig_exe = null,
                    .lib_dir = null,
                    .src_dir = null,
                    .targets = .empty,
                    .target_bytes = .empty,

                    .libc_runtimes_dir = null,
                    .enable_darling = false,
                    .enable_qemu = false,
                    .enable_rosetta = false,
                    .enable_wasmtime = false,
                    .enable_wine = false,

                    .gdb_exe = null,
                    .lldb_exe = null,

                    .quiet = false,
                },
                .host = std.zig.system.resolveTargetQuery(init.io, .{}) catch |err| switch (err) {
                    error.Canceled => |e| return e,
                    else => |e| fatal("unable to resolve host target: {t}", .{e}),
                },
                .server = .{
                    .in = &stdin.interface,
                    .out = &stdout.interface,
                },
                .eb_wip = try .init(init.gpa),
            };
            defer runner.deinit();
            switch (runner.runServer()) {
                error.Canceled, error.OutOfMemory => |e| return e,
                error.ServerReadFailed => switch (stdin.err.?) {
                    error.Canceled => |e| return e,
                    else => |e| fatal("unable to receive message: {t}", .{e}),
                },
                error.ServerWriteFailed => switch (stdout.err.?) {
                    error.Canceled => |e| return e,
                    else => |e| fatal("unable to send message: {t}", .{e}),
                },
                error.ServerEndOfStream => fatal("no more messages", .{}),
                error.AlreadyReported => {
                    var eb = try runner.eb_wip.toOwnedBundle("");
                    defer eb.deinit(init.gpa);
                    eb.renderToStderr(runner.io, .{}, .auto) catch |err| switch (err) {
                        error.Canceled => |e| return e,
                        else => {},
                    };
                    std.process.exit(1);
                },
            }
        } else if (std.mem.eql(u8, arg, "--zig")) {
            zig_path_arg = arg_it.next() orelse fatal("missing arg after {q}", .{arg});
        } else if (std.mem.cutPrefix(u8, arg, "--zig=")) |zig_path| {
            zig_path_arg = zig_path;
        } else if (std.mem.eql(u8, arg, "--lib")) {
            lib_path_arg = arg_it.next() orelse fatal("missing arg after {q}", .{arg});
        } else if (std.mem.cutPrefix(u8, arg, "--lib=")) |lib_path| {
            lib_path_arg = lib_path;
        } else if (std.mem.eql(u8, arg, "--target")) {
            try args.targets.append(
                init.gpa,
                try arena.dupe(u8, arg_it.next() orelse fatal("missing arg after {q}", .{arg})),
            );
        } else if (std.mem.cutPrefix(u8, arg, "--target=")) |target| {
            try args.targets.append(init.gpa, try arena.dupe(u8, target));
        } else if (std.mem.eql(u8, arg, "--libc-runtimes")) {
            libc_runtimes_path_arg = arg_it.next() orelse fatal("missing arg after {q}", .{arg});
        } else if (std.mem.cutPrefix(u8, arg, "--libc-runtimes=")) |libc_runtimes_path| {
            libc_runtimes_path_arg = libc_runtimes_path;
        } else if (std.mem.eql(u8, arg, "-fdarling")) {
            args.enable_darling = true;
        } else if (std.mem.eql(u8, arg, "-fqemu")) {
            args.enable_qemu = true;
        } else if (std.mem.eql(u8, arg, "-frosetta")) {
            args.enable_rosetta = true;
        } else if (std.mem.eql(u8, arg, "-fwine")) {
            args.enable_wine = true;
        } else if (std.mem.eql(u8, arg, "-fwasmtime")) {
            args.enable_wasmtime = true;
        } else if (std.mem.eql(u8, arg, "--gdb")) {
            gdb_path_arg = arg_it.next() orelse fatal("missing arg after {q}", .{arg});
        } else if (std.mem.cutPrefix(u8, arg, "--gdb=")) |gdb_path| {
            gdb_path_arg = gdb_path;
        } else if (std.mem.eql(u8, arg, "--lldb")) {
            lldb_path_arg = arg_it.next() orelse fatal("missing arg after {q}", .{arg});
        } else if (std.mem.cutPrefix(u8, arg, "--lldb=")) |lldb_path| {
            lldb_path_arg = lldb_path;
        } else if (std.mem.eql(u8, arg, "--quiet")) {
            args.quiet = true;
        } else if (std.mem.eql(u8, arg, "--keep-src")) {
            args.keep_src = true;
        } else {
            if (test_path_arg) |_| fatal("unexpected arg {q}", .{arg});
            test_path_arg = arg;
        }
    }

    const test_path = test_path_arg orelse fatal("missing \"path/to/test\" arg", .{});
    const zig_path = zig_path_arg orelse fatal("missing \"--zig=path/to/zig\" arg", .{});
    const lib_path = lib_path_arg orelse fatal("missing \"--lib=path/to/lib\" arg", .{});

    const cwd: std.Io.Dir = .cwd();
    args.test_file = cwd.openFile(init.io, test_path, .{
        .allow_directory = true,
    }) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| fatal("unable to open {q}: {t}", .{ test_path, e }),
    };
    defer args.test_file.close(init.io);
    args.test_kind = (args.test_file.stat(init.io) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| fatal("unable to stat {q}: {t}", .{ test_path, e }),
    }).kind;
    args.zig_exe = cwd.openFile(init.io, zig_path, .{}) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| fatal("unable to open {q}: {t}", .{ zig_path, e }),
    };
    defer args.zig_exe.close(init.io);
    args.lib_dir = cwd.openDir(init.io, lib_path, .{}) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| fatal("unable to open {q}: {t}", .{ lib_path, e }),
    };
    defer args.lib_dir.close(init.io);
    if (libc_runtimes_path_arg) |libc_runtimes_path| args.libc_runtimes_dir = cwd.openDir(
        init.io,
        libc_runtimes_path,
        .{},
    ) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| fatal("unable to open {q}: {t}", .{ lib_path, e }),
    };
    defer if (args.libc_runtimes_dir) |libc_runtimes_dir| libc_runtimes_dir.close(init.io);
    if (gdb_path_arg) |gdb_path| args.gdb_exe = cwd.openFile(init.io, gdb_path, .{}) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| fatal("unable to open {q}: {t}", .{ gdb_path, e }),
    };
    defer if (args.gdb_exe) |gdb_exe| gdb_exe.close(init.io);
    if (lldb_path_arg) |lldb_path| args.lldb_exe = cwd.openFile(init.io, lldb_path, .{}) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| fatal("unable to open {q}: {t}", .{ lldb_path, e }),
    };
    defer if (args.lldb_exe) |lldb_exe| lldb_exe.close(init.io);

    const src_dir_path = "src_" ++ std.fmt.hex(rand_int: {
        var rand_int: u64 = undefined;
        init.io.random(@ptrCast(&rand_int));
        break :rand_int rand_int;
    });
    args.src_dir = cwd.createDirPathOpen(init.io, src_dir_path, .{}) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| fatal("unable to create {q}: {t}", .{ src_dir_path, e }),
    };
    defer {
        args.src_dir.close(init.io);
        if (!args.keep_src) cwd.deleteTree(init.io, src_dir_path) catch |err| {
            std.log.warn("unable to delete tree {q}: {t}", .{ src_dir_path, err });
        };
    }

    var inherit_dirs: std.ArrayList(std.Io.Dir) = try .initCapacity(arena, 3);
    var inherit_files: std.ArrayList(std.Io.File) = try .initCapacity(arena, 4);
    inherit_files.appendSliceAssumeCapacity(&.{ args.test_file, args.zig_exe });
    inherit_dirs.appendSliceAssumeCapacity(&.{ args.lib_dir, args.src_dir });
    if (args.libc_runtimes_dir) |libc_runtimes_dir| inherit_dirs.appendAssumeCapacity(libc_runtimes_dir);
    if (args.gdb_exe) |gdb_exe| inherit_files.appendAssumeCapacity(gdb_exe);
    if (args.lldb_exe) |lldb_exe| inherit_files.appendAssumeCapacity(lldb_exe);
    var child = std.process.spawn(init.io, .{
        .argv = &.{ self, "--listen=-" },
        .stdin = .pipe,
        .stdout = .pipe,
        .inherit_dirs = inherit_dirs.items,
        .inherit_files = inherit_files.items,
    }) catch |err| switch (err) {
        error.Canceled, error.OutOfMemory => |e| return e,
        else => |e| fatal("unable to spawn runner: {t}", .{e}),
    };

    var child_reader_buffer: [512]u8 = undefined;
    var child_reader = child.stdout.?.readerStreaming(init.io, &child_reader_buffer);
    var child_writer_buffer: [512]u8 = undefined;
    var child_writer = child.stdin.?.writerStreaming(init.io, &child_writer_buffer);
    var client: std.zig.Client = .{
        .in = &child_reader.interface,
        .out = &child_writer.interface,
    };
    const success = runClient(init.gpa, arena, init.io, &client, &args) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.ReadFailed => switch (child_reader.err.?) {
            error.Canceled => |e| return e,
            else => |e| fatal("unable to receive message: {t}", .{e}),
        },
        error.WriteFailed => switch (child_writer.err.?) {
            error.Canceled => |e| return e,
            else => |e| fatal("unable to send message: {t}", .{e}),
        },
        error.EndOfStream => fatal("no more messages", .{}),
    };
    _ = child.wait(init.io) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| fatal("unable to wait for runner: {t}", .{e}),
    };
    return if (success) 0 else 1;
}
fn runClient(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: std.Io,
    client: *std.zig.Client,
    args: *const ClientArgs,
) (std.mem.Allocator.Error || std.Io.Reader.Error || std.Io.Writer.Error)!bool {
    var stdout_buffer: [512]u8 = undefined;
    var stdout_fw: std.Io.File.Writer = .initStreaming(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_fw.interface;
    var stderr_buffer: [512]u8 = undefined;
    var stderr_fw: std.Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const stderr = &stderr_fw.interface;

    var argv: std.ArrayList(u8) = .empty;
    defer argv.deinit(gpa);
    const Arg = std.zig.Client.Message.Arg;

    try argv.append(gpa, @backingInt(@as(Arg, switch (args.test_kind) {
        else => unreachable,
        .file => .input_file,
        .directory => .input_dir,
    })));
    try argv.appendSlice(gpa, @ptrCast(&args.test_file.handle));

    try argv.append(gpa, @backingInt(Arg.prefix));
    try argv.appendSlice(gpa, "--zig=");
    try argv.append(gpa, 0);
    try argv.append(gpa, @backingInt(Arg.input_file));
    try argv.appendSlice(gpa, @ptrCast(&args.zig_exe.handle));

    try argv.append(gpa, @backingInt(Arg.prefix));
    try argv.appendSlice(gpa, "--lib=");
    try argv.append(gpa, 0);
    try argv.append(gpa, @backingInt(Arg.input_dir));
    try argv.appendSlice(gpa, @ptrCast(&args.lib_dir.handle));

    try argv.append(gpa, @backingInt(Arg.prefix));
    try argv.appendSlice(gpa, "--src=");
    try argv.append(gpa, 0);
    try argv.append(gpa, @backingInt(Arg.output_dir));
    try argv.appendSlice(gpa, @ptrCast(&args.src_dir.handle));

    for (args.targets.items) |target| {
        try argv.append(gpa, @backingInt(Arg.string));
        try argv.appendSlice(gpa, "--target");
        try argv.append(gpa, 0);

        try argv.append(gpa, @backingInt(Arg.string));
        try argv.appendSlice(gpa, target);
        try argv.append(gpa, 0);
    }

    if (args.libc_runtimes_dir) |libc_runtimes_dir| {
        try argv.append(gpa, @backingInt(Arg.prefix));
        try argv.appendSlice(gpa, "--libc-runtimes=");
        try argv.append(gpa, 0);
        try argv.append(gpa, @backingInt(Arg.input_dir));
        try argv.appendSlice(gpa, @ptrCast(&libc_runtimes_dir.handle));
    }
    if (args.enable_darling) {
        try argv.append(gpa, @backingInt(Arg.string));
        try argv.appendSlice(gpa, "-fdarling");
        try argv.append(gpa, 0);
    }
    if (args.enable_qemu) {
        try argv.append(gpa, @backingInt(Arg.string));
        try argv.appendSlice(gpa, "-fqemu");
        try argv.append(gpa, 0);
    }
    if (args.enable_rosetta) {
        try argv.append(gpa, @backingInt(Arg.string));
        try argv.appendSlice(gpa, "-frosetta");
        try argv.append(gpa, 0);
    }
    if (args.enable_wine) {
        try argv.append(gpa, @backingInt(Arg.string));
        try argv.appendSlice(gpa, "-fwine");
        try argv.append(gpa, 0);
    }
    if (args.enable_wasmtime) {
        try argv.append(gpa, @backingInt(Arg.string));
        try argv.appendSlice(gpa, "-fwasmtime");
        try argv.append(gpa, 0);
    }

    if (args.gdb_exe) |gdb_exe| {
        try argv.append(gpa, @backingInt(Arg.prefix));
        try argv.appendSlice(gpa, "--gdb=");
        try argv.append(gpa, 0);
        try argv.append(gpa, @backingInt(Arg.input_file));
        try argv.appendSlice(gpa, @ptrCast(&gdb_exe.handle));
    }
    if (args.lldb_exe) |lldb_exe| {
        try argv.append(gpa, @backingInt(Arg.prefix));
        try argv.appendSlice(gpa, "--lldb=");
        try argv.append(gpa, 0);
        try argv.append(gpa, @backingInt(Arg.input_file));
        try argv.appendSlice(gpa, @ptrCast(&lldb_exe.handle));
    }

    if (args.quiet) {
        try argv.append(gpa, @backingInt(Arg.string));
        try argv.appendSlice(gpa, "--quiet");
        try argv.append(gpa, 0);
    }

    try client.serveMessageHeader(.{
        .tag = .args,
        .bytes_len = @intCast(argv.items.len),
    });
    try client.out.writeAll(argv.items);

    var success = true;
    var metadata: struct {
        index: u32,
        len: u32,
        names: []u32,
        expected_panic_msgs: []u32,
        string_bytes: []u8,

        fn name(m: *@This()) []const u8 {
            return std.mem.sliceTo(m.string_bytes[m.names[m.index - 1]..], 0);
        }

        fn expectedPanicMsg(m: *@This()) ?[:0]const u8 {
            return switch (m.expected_panic_msgs[m.index - 1]) {
                0 => null,
                else => |offset| m.string_bytes[offset..std.mem.findScalarPos(
                    u8,
                    m.string_bytes,
                    offset,
                    0,
                ).? :0],
            };
        }

        fn next(m: *@This()) ?u32 {
            if (m.len - m.index == 0) return null;
            defer m.index += 1;
            return m.index;
        }
    } = .{ .index = 0, .len = 0, .names = &.{}, .expected_panic_msgs = &.{}, .string_bytes = &.{} };
    try client.serveBodylessMessage(.query_test_metadata);
    while (true) {
        const hdr = try client.receiveMessage();
        switch (hdr.tag) {
            else => try client.in.discardAll(hdr.bytes_len),
            .zig_version => {
                const actual_version = try client.in.take(hdr.bytes_len);
                if (!std.mem.eql(u8, builtin.zig_version_string, actual_version)) {
                    try stderr.print(
                        "error: zig version mismatch compiler test runner vs compiler: {q} vs {q}\n",
                        .{ builtin.zig_version_string, actual_version },
                    );
                    try stderr.flush();
                    std.process.exit(1);
                }
            },
            .error_bundle => {
                const body = try arena.alloc(u8, hdr.bytes_len);
                try client.in.readSliceAll(body);
                const eb = std.zig.Server.allocErrorBundle(arena, body) catch |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    error.EndOfStream => {
                        try stderr.writeAll("error bundle truncated\n");
                        try stderr.flush();
                        std.process.exit(1);
                    },
                };
                success = false;
                eb.renderToStderr(io, .{}, .auto) catch {};
            },
            .test_metadata => {
                const tm_hdr =
                    try client.in.takeStruct(std.zig.Server.Message.TestMetadata, .little);
                metadata = .{
                    .index = 0,
                    .len = tm_hdr.tests_len,
                    .names = try arena.alloc(u32, tm_hdr.tests_len),
                    .expected_panic_msgs = try arena.alloc(u32, tm_hdr.tests_len),
                    .string_bytes = try arena.alloc(u8, tm_hdr.string_bytes_len),
                };
                try client.in.readSliceEndian(u32, metadata.names, .little);
                try client.in.readSliceEndian(u32, metadata.expected_panic_msgs, .little);
                try client.in.readSliceAll(metadata.string_bytes);
                try client.serveRunTest(metadata.next() orelse break);
            },
            .test_results => {
                const tr = try client.in.takeStruct(std.zig.Server.Message.TestResults, .little);
                switch (tr.flags.status) {
                    .pass, .skip => {},
                    .fail => success = false,
                }
                try stdout.print("[{t}] {s}\n", .{ tr.flags.status, metadata.name() });
                try stdout.flush();
                try client.serveRunTest(metadata.next() orelse break);
            },
        }
    }
    try client.serveBodylessMessage(.exit);
    return success;
}
