const builtin = @import("builtin");
const spirv_spec = @import("spirv_spec");

const std = @import("std");
const Io = std.Io;
const Dir = std.Io.Dir;
const mem = std.mem;
const json = std.json;
const assert = std.debug.assert;

// All references to other features are based on "zig name" as the key.

const FeatureOverride = struct {
    llvm_name: []const u8,
    /// If true, completely omit the feature; as if it does not exist.
    omit: bool = false,
    /// If true, omit the feature, but all the dependencies of the feature
    /// are added in its place.
    flatten: bool = false,
    zig_name: ?[]const u8 = null,
    desc: ?[]const u8 = null,
    omit_deps: []const []const u8 = &.{},
    extra_deps: []const []const u8 = &.{},
};

const ModelOverride = struct {
    llvm_name: []const u8,
    /// If true, completely omit the CPU; as if it does not exist.
    omit: bool = false,
    zig_name: ?[]const u8 = null,
    desc: ?[]const u8 = null,
    omit_deps: []const []const u8 = &.{},
    extra_deps: []const []const u8 = &.{},
};

const Model = struct {
    llvm_name: ?[]const u8,
    zig_name: []const u8,
    features: []const []const u8,
};

const Feature = struct {
    llvm_name: ?[]const u8 = null,
    zig_name: []const u8,
    desc: []const u8,
    deps: []const []const u8,
    flatten: bool = false,
};

const ArchTarget = struct {
    zig_name: []const u8,
    llvm: ?struct {
        name: []const u8,
        td_name: []const u8,
    },
    feature_overrides: []const FeatureOverride = &.{},
    extra_features: []const Feature = &.{},
    model_overrides: []const ModelOverride = &.{},
    extra_models: []const Model = &.{},
    branch_quota: ?usize = null,
};

const spirv_omitted_capabilities = [_][]const u8{
    "shader", // implied by Vulkan/OpenGL targets
    "kernel", // implied by OpenCL/AMDHSA targets
    "addresses", // implied by OpenCL/AMDHSA targets
    "matrix", // implied by shader
    "physical_storage_buffer_addresses", // implied by spirv64-vulkan
    "linkage", // automatically emitted when externs need linkage decorations
};
const spirv_omitted_extensions = [_][]const u8{
    "SPV_KHR_physical_storage_buffer", // paired with physical_storage_buffer_addresses
    "SPV_EXT_physical_storage_buffer",
};
const spirv_no_deps_capabilities = [_][]const u8{
    "ray_traversal_primitive_culling_khr",
};

fn spirvOmitCapability(name: []const u8) bool {
    for (spirv_omitted_capabilities) |omitted| {
        if (std.mem.eql(u8, omitted, name)) return true;
    }
    return false;
}

const spirv_extra_features: []const Feature = blk: {
    @setEvalBranchQuota(15_000);
    var features: []const Feature = &.{};

    for (
        @typeInfo(spirv_spec.Capability).@"enum".field_names,
        @typeInfo(spirv_spec.Capability).@"enum".field_values,
    ) |name, value| {
        if (spirvOmitCapability(name)) continue;

        features = features ++ &[_]Feature{.{
            .zig_name = name,
            .desc = "Enable " ++ name ++ " capability",
            .deps = struct {
                const cap: spirv_spec.Capability = @fromBackingInt(@intCast(value));
                const deps: []const spirv_spec.Capability = for (spirv_no_deps_capabilities) |no_deps| {
                    if (std.mem.eql(u8, no_deps, name)) break &.{};
                } else cap.dependencyCapabilities();
                const result: []const []const u8 = inner: {
                    var out: []const []const u8 = &.{};
                    for (cap.dependencyExtensions()) |ext| {
                        out = out ++ &[_][]const u8{@tagName(ext)};
                    }
                    for (deps) |dep| {
                        if (spirvOmitCapability(@tagName(dep))) continue;

                        out = out ++ &[_][]const u8{@tagName(dep)};
                    }
                    break :inner out;
                };
            }.result,
        }};
    }

    ext: for (@typeInfo(spirv_spec.Extension).@"enum".field_names) |name| {
        for (spirv_omitted_extensions) |omitted| {
            if (std.mem.eql(u8, omitted, name)) continue :ext;
        }
        features = features ++ &[_]Feature{.{
            .zig_name = name,
            .desc = "Enable " ++ name ++ " extension",
            .deps = if (std.mem.eql(u8, name, "v1_6"))
                &.{"v1_5"}
            else if (std.mem.eql(u8, name, "v1_5"))
                &.{"v1_4"}
            else if (std.mem.eql(u8, name, "v1_4"))
                &.{"v1_3"}
            else if (std.mem.eql(u8, name, "v1_3"))
                &.{"v1_2"}
            else if (std.mem.eql(u8, name, "v1_2"))
                &.{"v1_1"}
            else
                &.{},
        }};
    }

    break :blk features;
};

const targets = [_]ArchTarget{
    @import("cpu/aarch64.zon"),
    @import("cpu/alpha.zon"),
    @import("cpu/amdgcn.zon"),
    @import("cpu/arc.zon"),
    @import("cpu/arm.zon"),
    @import("cpu/avr.zon"),
    @import("cpu/bpf.zon"),
    @import("cpu/csky.zon"),
    @import("cpu/hexagon.zon"),
    @import("cpu/hppa.zon"),
    @import("cpu/kvx.zon"),
    @import("cpu/lanai.zon"),
    @import("cpu/loongarch.zon"),
    @import("cpu/m68k.zon"),
    @import("cpu/m88k.zon"),
    @import("cpu/microblaze.zon"),
    @import("cpu/mips.zon"),
    @import("cpu/msp430.zon"),
    @import("cpu/nvptx.zon"),
    @import("cpu/or1k.zon"),
    @import("cpu/powerpc.zon"),
    @import("cpu/propeller.zon"),
    @import("cpu/riscv.zon"),
    @import("cpu/s390x.zon"),
    @import("cpu/sh.zon"),
    @import("cpu/sparc.zon"),
    spirv: {
        var target: ArchTarget = @import("cpu/spirv.zon");
        target.extra_features = target.extra_features ++ spirv_extra_features;
        break :spirv target;
    },
    @import("cpu/ve.zon"),
    @import("cpu/wasm.zon"),
    @import("cpu/x86.zon"),
    @import("cpu/xcore.zon"),
    @import("cpu/xtensa.zon"),
    @import("cpu/z80.zon"),
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    var args = try init.minimal.args.iterateAllocator(arena);
    const args0 = args.next().?;

    const llvm_tblgen_exe = args.next() orelse
        usageAndExit(args0, 1);

    if (std.mem.eql(u8, llvm_tblgen_exe, "--help")) {
        usageAndExit(args0, 0);
    }
    if (std.mem.startsWith(u8, llvm_tblgen_exe, "-")) {
        usageAndExit(args0, 1);
    }

    const llvm_src_root = args.next() orelse
        usageAndExit(args0, 1);

    if (std.mem.startsWith(u8, llvm_src_root, "-")) {
        usageAndExit(args0, 1);
    }

    const zig_src_root = args.next() orelse
        usageAndExit(args0, 1);

    if (std.mem.startsWith(u8, zig_src_root, "-")) {
        usageAndExit(args0, 1);
    }

    var filter: ?[]const u8 = null;
    if (args.next()) |arg| filter = arg;

    // there shouldn't be any more argument after the optional filter
    if (args.skip()) usageAndExit(args0, 1);

    var zig_src_dir = try Dir.cwd().openDir(io, zig_src_root, .{});
    defer zig_src_dir.close(io);

    const root_progress = std.Progress.start(io, .{ .estimated_total_items = targets.len });
    defer root_progress.end();

    var group: Io.Group = .init;
    defer group.cancel(io);

    for (targets) |target| {
        if (filter) |zig_name| {
            if (!std.mem.eql(u8, target.zig_name, zig_name)) continue;
        }
        group.async(io, processOneTarget, .{ io, .{
            .llvm_tblgen_exe = llvm_tblgen_exe,
            .llvm_src_root = llvm_src_root,
            .zig_src_dir = zig_src_dir,
            .root_progress = root_progress,
            .target = target,
        } });
    }

    try group.await(io);
}

const Job = struct {
    llvm_tblgen_exe: []const u8,
    llvm_src_root: []const u8,
    zig_src_dir: Dir,
    root_progress: std.Progress.Node,
    target: ArchTarget,
};

fn processOneTarget(io: Io, job: Job) void {
    processOneTargetInner(io, job) catch |err| std.debug.panic("panic: {s}", .{@errorName(err)});
}
fn processOneTargetInner(io: Io, job: Job) !void {
    const target = job.target;

    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const progress_node = job.root_progress.start(target.zig_name, 3);
    defer progress_node.end();

    var features_table = std.StringHashMap(Feature).init(arena);
    var all_features: std.ArrayList(Feature) = .empty;
    var all_models: std.ArrayList(Model) = .empty;

    if (target.llvm) |llvm| {
        const tblgen_progress = progress_node.start("running llvm-tblgen", 0);

        const child_args = [_][]const u8{
            job.llvm_tblgen_exe,
            "--dump-json",
            try arena.print("{s}/llvm/lib/Target/{s}/{s}.td", .{
                job.llvm_src_root,
                llvm.name,
                llvm.td_name,
            }),
            try arena.print("-I={s}/llvm/include", .{job.llvm_src_root}),
            try arena.print("-I={s}/llvm/lib/Target/{s}", .{
                job.llvm_src_root, llvm.name,
            }),
        };

        const child_result = try std.process.run(arena, io, .{
            .argv = &child_args,
        });
        tblgen_progress.end();
        if (child_result.stderr.len != 0) {
            std.debug.print("{s}\n", .{child_result.stderr});
        }

        const json_text = switch (child_result.term) {
            .exited => |code| if (code == 0) child_result.stdout else {
                std.debug.print("llvm-tblgen exited with code {d}\n", .{code});
                std.process.exit(1);
            },
            .signal => |sig| {
                std.debug.print("llvm-tblgen terminated with signal {t}\n", .{sig});
                std.process.exit(1);
            },
            .stopped => |sig| {
                std.debug.print("llvm-tblgen stopped with signal {t}\n", .{sig});
                std.process.exit(1);
            },
            .unknown => {
                std.debug.print("llvm-tblgen crashed\n", .{});
                std.process.exit(1);
            },
        };

        const json_parse_progress = progress_node.start("parsing JSON", 0);

        const parsed = try json.parseFromSlice(json.Value, arena, json_text, .{});
        defer parsed.deinit();
        const root_map = &parsed.value.object;
        json_parse_progress.end();

        const collate_progress = progress_node.start("collating LLVM data", 0);

        var model_aliases = std.StringHashMap(std.ArrayList([]const u8)).init(arena); // uses LLVM names

        {
            var it = root_map.iterator();
            while (it.next()) |kv| {
                if (kv.key_ptr.len == 0) continue;
                if (kv.key_ptr.*[0] == '!') continue;
                if (kv.value_ptr.* != .object) continue;
                if (hasSuperclass(&kv.value_ptr.object, "ProcessorAlias")) {
                    // Note that `Name` is actually the alias, while `Alias` is the name that will have
                    // a full `Processor` object defined.
                    const llvm_alias = kv.value_ptr.object.get("Name").?.string;
                    const llvm_name = kv.value_ptr.object.get("Alias").?.string;

                    const gop = try model_aliases.getOrPut(llvm_name);
                    if (!gop.found_existing) gop.value_ptr.* = .empty;
                    try gop.value_ptr.append(arena, llvm_alias);
                }
            }
        }

        {
            var it = root_map.iterator();
            while (it.next()) |kv| {
                if (kv.key_ptr.len == 0) continue;
                if (kv.key_ptr.*[0] == '!') continue;
                if (kv.value_ptr.* != .object) continue;
                if (hasSuperclass(&kv.value_ptr.object, "SubtargetFeature")) {
                    const llvm_name = kv.value_ptr.object.get("Name").?.string;
                    if (llvm_name.len == 0) continue;

                    var zig_name = try llvmNameToZigName(arena, llvm_name);
                    var desc = kv.value_ptr.object.get("Desc").?.string;
                    var deps: std.ArrayList([]const u8) = .empty;
                    var omit = false;
                    var flatten = false;
                    var omit_deps: []const []const u8 = &.{};
                    var extra_deps: []const []const u8 = &.{};
                    for (target.feature_overrides) |feature_override| {
                        if (mem.eql(u8, llvm_name, feature_override.llvm_name)) {
                            if (feature_override.omit) {
                                // Still put the feature into the table so that we can
                                // expand dependencies for the feature overrides marked `flatten`.
                                omit = true;
                            }
                            if (feature_override.flatten) {
                                flatten = true;
                            }
                            if (feature_override.zig_name) |override_name| {
                                zig_name = override_name;
                            }
                            if (feature_override.desc) |override_desc| {
                                desc = override_desc;
                            }
                            omit_deps = feature_override.omit_deps;
                            extra_deps = feature_override.extra_deps;
                            break;
                        }
                    }
                    const implies = kv.value_ptr.object.get("Implies").?.array;
                    for (implies.items) |imply| {
                        const other_key = imply.object.get("def").?.string;
                        const other_obj = root_map.get(other_key).?.object;
                        const other_llvm_name = other_obj.get("Name").?.string;
                        const other_zig_name = (try llvmFeatureNameToZigNameOmit(
                            arena,
                            target,
                            other_llvm_name,
                        )) orelse continue;
                        for (omit_deps) |omit_dep| {
                            if (mem.eql(u8, other_zig_name, omit_dep)) break;
                        } else {
                            try deps.append(arena, other_zig_name);
                        }
                    }
                    for (extra_deps) |extra_dep| {
                        try deps.append(arena, extra_dep);
                    }
                    const feature: Feature = .{
                        .llvm_name = llvm_name,
                        .zig_name = zig_name,
                        .desc = desc,
                        .deps = deps.items,
                        .flatten = flatten,
                    };
                    try features_table.put(zig_name, feature);
                    if (!omit and !flatten) {
                        try all_features.append(arena, feature);
                    }
                }
                if (hasSuperclass(&kv.value_ptr.object, "Processor")) {
                    const llvm_name = kv.value_ptr.object.get("Name").?.string;
                    if (llvm_name.len == 0) continue;

                    const processor = &kv.value_ptr.object;
                    try addModel(arena, target, root_map, processor, llvm_name, &all_models);
                    if (model_aliases.get(llvm_name)) |aliases| {
                        for (aliases.items) |alias| {
                            try addModel(arena, target, root_map, processor, alias, &all_models);
                        }
                    }
                }
            }
        }

        collate_progress.end();
    }

    for (target.extra_features) |extra_feature| {
        try features_table.put(extra_feature.zig_name, extra_feature);
        try all_features.append(arena, extra_feature);
    }
    for (target.extra_models) |extra_model| {
        try all_models.append(arena, extra_model);
    }
    mem.sort(Feature, all_features.items, {}, featureLessThan);
    mem.sort(Model, all_models.items, {}, modelLessThan);

    const render_progress = progress_node.start("rendering Zig code", 0);

    var target_dir = try job.zig_src_dir.openDir(io, "lib/std/Target", .{});
    defer target_dir.close(io);

    const zig_code_basename = try arena.print("{s}.zig", .{target.zig_name});
    var zig_code_file = try target_dir.createFile(io, zig_code_basename, .{});
    defer zig_code_file.close(io);

    var zig_code_file_buffer: [4096]u8 = undefined;
    var zig_code_file_writer = zig_code_file.writer(io, &zig_code_file_buffer);
    const w = &zig_code_file_writer.interface;

    try w.writeAll(
        \\//! This file is auto-generated by tools/update_cpu_features.zig.
        \\
        \\const std = @import("../std.zig");
        \\const CpuFeature = std.Target.Cpu.Feature;
        \\const CpuModel = std.Target.Cpu.Model;
        \\
        \\pub const Feature = enum {
    );

    for (all_features.items, 0..) |feature, i| {
        try w.print("\n    {f},", .{std.zig.fmtIdPU(feature.zig_name)});

        if (i == all_features.items.len - 1) try w.writeAll("\n");
    }

    try w.writeAll(
        \\};
        \\
        \\pub const featureSet = CpuFeature.FeatureSetFns(Feature).featureSet;
        \\pub const featureSetHas = CpuFeature.FeatureSetFns(Feature).featureSetHas;
        \\pub const featureSetHasAny = CpuFeature.FeatureSetFns(Feature).featureSetHasAny;
        \\pub const featureSetHasAll = CpuFeature.FeatureSetFns(Feature).featureSetHasAll;
        \\
        \\pub const all_features = blk: {
        \\
    );
    if (target.branch_quota) |branch_quota| {
        try w.print("    @setEvalBranchQuota({d});\n", .{branch_quota});
    }
    try w.writeAll(
        \\    const len = @typeInfo(Feature).@"enum".field_names.len;
        \\    std.debug.assert(len <= CpuFeature.Set.needed_bit_count);
        \\    var result: [len]CpuFeature = undefined;
        \\
    );

    for (all_features.items) |feature| {
        if (feature.llvm_name) |llvm_name| {
            try w.print(
                \\    result[@backingInt(Feature.{f})] = .{{
                \\        .llvm_name = "{f}",
                \\        .description = "{f}",
                \\        .dependencies = featureSet(&[_]Feature{{
            ,
                .{
                    std.zig.fmtIdPU(feature.zig_name),
                    std.zig.fmtString(llvm_name),
                    std.zig.fmtString(feature.desc),
                },
            );
        } else {
            try w.print(
                \\    result[@backingInt(Feature.{f})] = .{{
                \\        .llvm_name = null,
                \\        .description = "{f}",
                \\        .dependencies = featureSet(&[_]Feature{{
            ,
                .{
                    std.zig.fmtIdPU(feature.zig_name),
                    std.zig.fmtString(feature.desc),
                },
            );
        }
        var deps_set = std.StringHashMap(void).init(arena);
        for (feature.deps) |dep| {
            try putDep(&deps_set, features_table, dep);
        }
        try pruneFeatures(arena, features_table, &deps_set);
        var dependencies: std.ArrayList([]const u8) = .empty;
        {
            var it = deps_set.keyIterator();
            while (it.next()) |key| {
                try dependencies.append(arena, key.*);
            }
        }
        mem.sort([]const u8, dependencies.items, {}, asciiLessThan);

        if (dependencies.items.len == 0) {
            try w.writeAll(
                \\}),
                \\    };
                \\
            );
        } else {
            try w.writeAll("\n");
            for (dependencies.items) |dep| {
                try w.print("            .{f},\n", .{std.zig.fmtIdPU(dep)});
            }
            try w.writeAll(
                \\        }),
                \\    };
                \\
            );
        }
    }
    try w.writeAll(
        \\    const ti = @typeInfo(Feature);
        \\    for (&result, 0..) |*elem, i| {
        \\        elem.index = i;
        \\        elem.name = ti.@"enum".field_names[i];
        \\    }
        \\    break :blk result;
        \\};
        \\
        \\pub const cpu = struct {
        \\
    );
    for (all_models.items) |model| {
        var deps_set = std.StringHashMap(void).init(arena);
        for (model.features) |feature_zig_name| {
            try putDep(&deps_set, features_table, feature_zig_name);
        }
        try pruneFeatures(arena, features_table, &deps_set);
        var model_features: std.ArrayList([]const u8) = .empty;
        {
            var it = deps_set.keyIterator();
            while (it.next()) |key| {
                try model_features.append(arena, key.*);
            }
        }
        mem.sort([]const u8, model_features.items, {}, asciiLessThan);
        if (model.llvm_name) |llvm_name| {
            try w.print(
                \\    pub const {f}: CpuModel = .{{
                \\        .name = "{f}",
                \\        .llvm_name = "{f}",
                \\        .features = featureSet(&[_]Feature{{
            , .{
                std.zig.fmtId(model.zig_name),
                std.zig.fmtString(model.zig_name),
                std.zig.fmtString(llvm_name),
            });
        } else {
            try w.print(
                \\    pub const {f}: CpuModel = .{{
                \\        .name = "{f}",
                \\        .llvm_name = null,
                \\        .features = featureSet(&[_]Feature{{
            , .{
                std.zig.fmtId(model.zig_name),
                std.zig.fmtString(model.zig_name),
            });
        }
        if (model_features.items.len == 0) {
            try w.writeAll(
                \\}),
                \\    };
                \\
            );
        } else {
            try w.writeAll("\n");
            for (model_features.items) |feature_zig_name| {
                try w.print("            .{f},\n", .{std.zig.fmtIdPU(feature_zig_name)});
            }
            try w.writeAll(
                \\        }),
                \\    };
                \\
            );
        }
    }

    try w.writeAll(
        \\};
        \\
    );
    try w.flush();

    render_progress.end();
}

fn usageAndExit(arg0: []const u8, code: u8) noreturn {
    const stderr = std.debug.lockStderr(&.{});
    const w = &stderr.file_writer.interface;
    w.print(
        \\Usage: {s} /path/to/llvm-tblgen /path/git/llvm-project /path/git/zig [zig_name filter]
        \\
        \\Updates lib/std/target/<target>.zig from llvm/lib/Target/<Target>/<Target>.td .
        \\
        \\On a less beefy system, or when debugging, compile with -fsingle-threaded.
        \\
    , .{arg0}) catch std.process.exit(1);
    std.process.exit(code);
}

fn featureLessThan(_: void, a: Feature, b: Feature) bool {
    return std.ascii.lessThanIgnoreCase(a.zig_name, b.zig_name);
}

fn addModel(
    arena: mem.Allocator,
    target: ArchTarget,
    root_map: *const json.ObjectMap,
    processor: *const json.ObjectMap,
    llvm_name: []const u8,
    all_models: *std.ArrayList(Model),
) !void {
    var zig_name = try llvmNameToZigName(arena, llvm_name);
    var omit_deps: []const []const u8 = &.{};
    var extra_deps: []const []const u8 = &.{};
    for (target.model_overrides) |model_override| {
        if (mem.eql(u8, llvm_name, model_override.llvm_name)) {
            if (model_override.omit) return;
            if (model_override.zig_name) |override_name| {
                zig_name = override_name;
            }
            omit_deps = model_override.omit_deps;
            extra_deps = model_override.extra_deps;
            break;
        }
    }

    var deps: std.ArrayList([]const u8) = .empty;
    for (processor.get("Features").?.array.items) |feature| {
        const feature_key = feature.object.get("def").?.string;
        const feature_obj = root_map.get(feature_key).?.object;
        const feature_llvm_name = feature_obj.get("Name").?.string;
        if (feature_llvm_name.len == 0) continue;
        const feature_zig_name = (try llvmFeatureNameToZigNameOmit(
            arena,
            target,
            feature_llvm_name,
        )) orelse continue;
        for (omit_deps) |omit_dep| {
            if (mem.eql(u8, feature_zig_name, omit_dep)) break;
        } else {
            try deps.append(arena, feature_zig_name);
        }
    }
    for (extra_deps) |extra_dep| {
        try deps.append(arena, extra_dep);
    }
    for (processor.get("TuneFeatures").?.array.items) |feature| {
        const feature_key = feature.object.get("def").?.string;
        const feature_obj = root_map.get(feature_key).?.object;
        const feature_llvm_name = feature_obj.get("Name").?.string;
        if (feature_llvm_name.len == 0) continue;
        const feature_zig_name = (try llvmFeatureNameToZigNameOmit(
            arena,
            target,
            feature_llvm_name,
        )) orelse continue;
        try deps.append(arena, feature_zig_name);
    }

    try all_models.append(arena, .{
        .llvm_name = llvm_name,
        .zig_name = zig_name,
        .features = deps.items,
    });
}

fn modelLessThan(_: void, a: Model, b: Model) bool {
    return std.ascii.lessThanIgnoreCase(a.zig_name, b.zig_name);
}

fn asciiLessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.ascii.lessThanIgnoreCase(a, b);
}

fn llvmNameToZigName(arena: mem.Allocator, llvm_name: []const u8) ![]const u8 {
    const duped = try arena.dupe(u8, llvm_name);
    for (duped) |*byte| switch (byte.*) {
        '-', '.' => byte.* = '_',
        else => continue,
    };
    return duped;
}

fn llvmFeatureNameToZigNameOmit(
    arena: mem.Allocator,
    target: ArchTarget,
    llvm_name: []const u8,
) !?[]const u8 {
    for (target.feature_overrides) |feature_override| {
        if (mem.eql(u8, feature_override.llvm_name, llvm_name)) {
            if (feature_override.omit) return null;
            return feature_override.zig_name orelse break;
        }
    }
    return try llvmNameToZigName(arena, llvm_name);
}

fn hasSuperclass(obj: *const json.ObjectMap, class_name: []const u8) bool {
    const superclasses_json = obj.get("!superclasses") orelse return false;
    for (superclasses_json.array.items) |superclass_json| {
        const superclass = superclass_json.string;
        if (std.mem.eql(u8, superclass, class_name)) {
            return true;
        }
    }
    return false;
}

fn pruneFeatures(
    arena: mem.Allocator,
    features_table: std.StringHashMap(Feature),
    deps_set: *std.StringHashMap(void),
) !void {
    // For each element, recursively iterate over the dependencies and add
    // everything we find to a "deletion set".
    // Then, iterate over the deletion set and delete all that stuff from `deps_set`.
    var deletion_set = std.StringHashMap(void).init(arena);
    {
        var it = deps_set.keyIterator();
        while (it.next()) |key| {
            const feature = features_table.get(key.*).?;
            try walkFeatures(features_table, &deletion_set, feature);
        }
    }
    {
        var it = deletion_set.keyIterator();
        while (it.next()) |key| {
            _ = deps_set.remove(key.*);
        }
    }
}

fn walkFeatures(
    features_table: std.StringHashMap(Feature),
    deletion_set: *std.StringHashMap(void),
    feature: Feature,
) error{OutOfMemory}!void {
    for (feature.deps) |dep| {
        if ((try deletion_set.getOrPut(dep)).found_existing) continue;
        const other_feature = features_table.get(dep).?;
        try walkFeatures(features_table, deletion_set, other_feature);
    }
}

fn putDep(
    deps_set: *std.StringHashMap(void),
    features_table: std.StringHashMap(Feature),
    zig_feature_name: []const u8,
) error{OutOfMemory}!void {
    const feature = features_table.get(zig_feature_name).?;
    if (feature.flatten) {
        for (feature.deps) |dep| {
            try putDep(deps_set, features_table, dep);
        }
    } else {
        try deps_set.put(zig_feature_name, {});
    }
}
