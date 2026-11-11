//! These are passed as `<target>-<cpu>-<cache_mode>-<backend>-...` strings.

cpu: Cpu = .{ .native = .{} },
os: Os = .native,
abi: Abi = .default,
ofmt: ObjectFormat = .default,
dynamic_linker: DynamicLinker = .default,

cache_mode: CacheMode,
backend: Backend,
optimize: std.lang.Optimize = .debug,
pic: Pic = .default,
frame_pointers: FramePointers = .default,
unwind_tables: UnwindTables = .default,
linker: Linker = .default,
linkage: Linkage = .default,
link_libc: LinkLibc = .default,
strip: Strip = .default,

extra: bool = false,

pub const Cpu = Cpu: {
    @setEvalBranchQuota(6_000);
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
            features: struct { add: []const Feature = &.{}, remove: []const Feature = &.{} } = .{},

            pub const Model = Model: {
                const explicit_models = @typeInfo(family.cpu).@"struct".decl_names;
                const model_names: []const []const u8 = .{ "native", "baseline" } ++ explicit_models;
                const TagInt = std.math.IntFittingRange(0, model_names.len - 1);
                var model_values: [model_names.len]TagInt = undefined;
                model_values[0] = explicit_models.len;
                model_values[1] = explicit_models.len + 1;
                for (model_values[2..], 0..) |*model_value, value| model_value.* = value;
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

pub const DynamicLinker = union(enum) { default, none, explicit: []const u8 };

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
    try w.print("{f}-{f}-{t}-{t}-{t}", .{
        std.fmt.alt(&target.toQuery(), .printZigTriple),
        std.fmt.alt(&target.toQuery(), .printCpu),
        target.cache_mode,
        target.backend,
        target.optimize,
    });
    switch (target.pic) {
        .default => {},
        else => |pic| try w.print("-{t}", .{pic}),
    }
    switch (target.frame_pointers) {
        .default => {},
        else => |frame_pointers| try w.print("-{t}", .{frame_pointers}),
    }
    switch (target.unwind_tables) {
        .default => {},
        else => |unwind_tables| try w.print("-{t}", .{unwind_tables}),
    }
    switch (target.linker) {
        .default => {},
        else => |linker| try w.print("-{t}", .{linker}),
    }
    switch (target.linkage) {
        .default => {},
        else => |linkage| try w.print("-{t}", .{linkage}),
    }
    switch (target.link_libc) {
        .default => {},
        else => |link_libc| try w.print("-{t}", .{link_libc}),
    }
    switch (target.strip) {
        .default => {},
        else => |strip| try w.print("-{t}", .{strip}),
    }
}

pub const CacheMode = enum { whole, incremental };
pub const Backend = enum {
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
pub const Pic = enum { default, pdc, pic, pie };
pub const FramePointers = enum {
    default,
    keep,
    omit,

    pub const Resolved = enum { keep, omit };
    pub fn resolve(fp: @This(), target: *const std.Target, optimize: std.lang.Optimize) Resolved {
        return switch (fp) {
            .default => if (std.zig.defaultOmitFramePointer(target, optimize)) .omit else .keep,
            .keep => .keep,
            .omit => .omit,
        };
    }
};
pub const UnwindTables = enum {
    default,
    none,
    sync,
    async,

    pub fn resolve(ut: @This(), target: *const std.Target) std.lang.UnwindTables {
        return switch (ut) {
            .default => std.zig.defaultUnwindTables(target),
            .none => .none,
            .sync => .sync,
            .async => .async,
        };
    }
};
pub const Linker = enum { default, lld, old, new };
pub const Linkage = enum { default, static, dynamic };
pub const LinkLibc = enum { default, lc };
pub const Strip = enum { default, nostrip, stripdebug, strip };

pub const State = struct { skipped: bool };

const std = @import("std");
const Target = @This();
