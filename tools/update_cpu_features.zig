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
    .{
        .zig_name = "aarch64",
        .llvm = .{
            .name = "AArch64",
            .td_name = "AArch64",
        },
        .branch_quota = 2000,
        .feature_overrides = &.{
            .{
                .llvm_name = "aggressive-interleaving",
                .omit = true,
            },
            .{
                .llvm_name = "all",
                .omit = true,
            },
            .{
                .llvm_name = "ascend-store-address",
                .omit = true,
            },
            .{
                .llvm_name = "call-saved-x8",
                .omit = true,
            },
            .{
                .llvm_name = "call-saved-x9",
                .omit = true,
            },
            .{
                .llvm_name = "call-saved-x10",
                .omit = true,
            },
            .{
                .llvm_name = "call-saved-x11",
                .omit = true,
            },
            .{
                .llvm_name = "call-saved-x12",
                .omit = true,
            },
            .{
                .llvm_name = "call-saved-x13",
                .omit = true,
            },
            .{
                .llvm_name = "call-saved-x14",
                .omit = true,
            },
            .{
                .llvm_name = "call-saved-x15",
                .omit = true,
            },
            .{
                .llvm_name = "call-saved-x18",
                .omit = true,
            },
            .{
                .llvm_name = "disable-distinct-predicate-dst-reg",
                .omit = true,
            },
            .{
                .llvm_name = "disable-latency-sched-heuristic",
                .omit = true,
            },
            .{
                .llvm_name = "disable-ldp",
                .omit = true,
            },
            .{
                .llvm_name = "disable-maximize-scalable-bandwidth",
                .omit = true,
            },
            .{
                .llvm_name = "enable-select-opt",
                .omit = true,
            },
            .{
                .llvm_name = "disable-stp",
                .omit = true,
            },
            .{
                .llvm_name = "execute-only",
                .omit = true,
            },
            .{
                .llvm_name = "exynos-cheap-as-move",
                .omit = true,
            },
            .{
                .llvm_name = "fixed-load-latency-4",
                .omit = true,
            },
            .{
                .llvm_name = "fmv",
                .omit = true,
            },
            .{
                .llvm_name = "force-32bit-jump-tables",
                .omit = true,
            },
            .{
                .llvm_name = "harden-sls-blr",
                .omit = true,
            },
            .{
                .llvm_name = "harden-sls-nocomdat",
                .omit = true,
            },
            .{
                .llvm_name = "harden-sls-retbr",
                .omit = true,
            },
            .{
                .llvm_name = "max-interleave-factor-4",
                .omit = true,
            },
            .{
                .llvm_name = "mca-streaming-sched",
                .omit = true,
            },
            .{
                .llvm_name = "no-bti-at-return-twice",
                .omit = true,
            },
            .{
                .llvm_name = "no-lfi-loads",
                .omit = true,
            },
            .{
                .llvm_name = "no-lfi-stores",
                .omit = true,
            },
            .{
                .llvm_name = "no-neg-immediates",
                .omit = true,
            },
            .{
                .llvm_name = "outline-atomics",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-lr-for-ra",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x2",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x3",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x4",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x5",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x6",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x7",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x9",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x10",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x11",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x12",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x13",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x14",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x15",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x18",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x20",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x21",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x22",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x23",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x24",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x25",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x26",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x27",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x28",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x1",
                .omit = true,
            },
            .{
                .llvm_name = "store-pair-suppress",
                .omit = true,
            },
            .{
                .llvm_name = "tpidr-el1",
                .omit = true,
            },
            .{
                .llvm_name = "tpidr-el2",
                .omit = true,
            },
            .{
                .llvm_name = "tpidr-el3",
                .omit = true,
            },
            .{
                .llvm_name = "tpidrro-el0",
                .omit = true,
            },
            .{
                .llvm_name = "tagged-globals",
                .omit = true,
            },
            .{
                .llvm_name = "use-fixed-over-scalable-if-equal-cost",
                .omit = true,
            },
            .{
                .llvm_name = "use-postra-scheduler",
                .omit = true,
            },
            .{
                .llvm_name = "use-reciprocal-square-root",
                .omit = true,
            },

            .{
                .llvm_name = "hip12",
                .flatten = true,
            },
            .{
                .llvm_name = "neoversee1",
                .flatten = true,
            },
            .{
                .llvm_name = "neoversen1",
                .flatten = true,
            },
            .{
                .llvm_name = "neoversen2",
                .flatten = true,
            },
            .{
                .llvm_name = "neoversen3",
                .flatten = true,
            },
            .{
                .llvm_name = "neoversev1",
                .flatten = true,
            },
            .{
                .llvm_name = "neoversev2",
                .flatten = true,
            },
            .{
                .llvm_name = "neoversev3",
                .flatten = true,
            },
            .{
                .llvm_name = "neoversev3AE",
                .flatten = true,
            },
            .{
                .llvm_name = "neoverse512tvb",
                .flatten = true,
            },
            .{
                .llvm_name = "oryon-1",
                .flatten = true,
            },
            .{
                .llvm_name = "exynosm3",
                .flatten = true,
            },
            .{
                .llvm_name = "exynosm4",
                .flatten = true,
            },
            .{
                .llvm_name = "a320",
                .flatten = true,
            },
            .{
                .llvm_name = "a35",
                .flatten = true,
            },
            .{
                .llvm_name = "a53",
                .flatten = true,
            },
            .{
                .llvm_name = "a55",
                .flatten = true,
            },
            .{
                .llvm_name = "a57",
                .flatten = true,
            },
            .{
                .llvm_name = "a510",
                .flatten = true,
            },
            .{
                .llvm_name = "a520",
                .flatten = true,
            },
            .{
                .llvm_name = "a520ae",
                .flatten = true,
            },
            .{
                .llvm_name = "a64fx",
                .flatten = true,
            },
            .{
                .llvm_name = "a65",
                .flatten = true,
            },
            .{
                .llvm_name = "a72",
                .flatten = true,
            },
            .{
                .llvm_name = "a73",
                .flatten = true,
            },
            .{
                .llvm_name = "a75",
                .flatten = true,
            },
            .{
                .llvm_name = "a76",
                .flatten = true,
            },
            .{
                .llvm_name = "a77",
                .flatten = true,
            },
            .{
                .llvm_name = "a78",
                .flatten = true,
            },
            .{
                .llvm_name = "a78ae",
                .flatten = true,
            },
            .{
                .llvm_name = "a78c",
                .flatten = true,
            },
            .{
                .llvm_name = "a710",
                .flatten = true,
            },
            .{
                .llvm_name = "a715",
                .flatten = true,
            },
            .{
                .llvm_name = "a720",
                .flatten = true,
            },
            .{
                .llvm_name = "a720ae",
                .flatten = true,
            },
            .{
                .llvm_name = "ampere1a",
                .flatten = true,
            },
            .{
                .llvm_name = "ampere1c",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-a7",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-a10",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-a11",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-a12",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-a13",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-a14",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-a15",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-a16",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-a17",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-a7-sysreg",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-m4",
                .flatten = true,
            },
            .{
                .llvm_name = "apple-m5",
                .flatten = true,
            },
            .{
                .llvm_name = "c1-nano",
                .flatten = true,
            },
            .{
                .llvm_name = "c1-premium",
                .flatten = true,
            },
            .{
                .llvm_name = "c1-pro",
                .flatten = true,
            },
            .{
                .llvm_name = "c1-ultra",
                .flatten = true,
            },
            .{
                .llvm_name = "carmel",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-a725",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-a78",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-r82",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-r82ae",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-x1",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-x2",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-x3",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-x4",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-x925",
                .flatten = true,
            },
            .{
                .llvm_name = "falkor",
                .flatten = true,
            },
            .{
                .llvm_name = "fujitsu-monaka",
                .flatten = true,
            },
            .{
                .llvm_name = "kryo",
                .flatten = true,
            },
            .{
                .llvm_name = "olympus",
                .flatten = true,
            },
            .{
                .llvm_name = "saphira",
                .flatten = true,
            },
            .{
                .llvm_name = "thunderx",
                .flatten = true,
            },
            .{
                .llvm_name = "thunderx2t99",
                .flatten = true,
            },
            .{
                .llvm_name = "thunderx3t110",
                .flatten = true,
                .omit_deps = &.{"no_unaligned_access"},
            },
            .{
                .llvm_name = "thunderxt81",
                .flatten = true,
            },
            .{
                .llvm_name = "thunderxt83",
                .flatten = true,
            },
            .{
                .llvm_name = "thunderxt88",
                .flatten = true,
            },
            .{
                .llvm_name = "tsv110",
                .flatten = true,
            },
            .{
                .llvm_name = "ampere1",
                .flatten = true,
            },
            .{
                .llvm_name = "ampere1b",
                .flatten = true,
            },

            .{
                .llvm_name = "CONTEXTIDREL2",
                .zig_name = "contextidr_el2",
                .desc = "Enable RW operand Context ID Register (EL2)",
            },
            .{
                .llvm_name = "strict-align",
                .zig_name = "no_unaligned_access",
            },

            // Future ISA extensions; omit until they are stabilized.
            .{
                .llvm_name = "use-experimental-zeroing-pseudos",
                .omit = true,
            },
        },
        .model_overrides = &.{
            .{
                // Who thought this alias was a good idea? Upgrade your compiler and suddenly your
                // programs SIGILL because this changed meaning. Brilliant.
                .llvm_name = "apple-latest",
                .omit = true,
            },
            .{
                .llvm_name = "generic",
                .omit = true,
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = "generic",
                .zig_name = "armv8a",
                .features = &.{"v8a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv8r",
                .features = &.{"v8r"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv8_1a",
                .features = &.{"v8_1a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv8_2a",
                .features = &.{"v8_2a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv8_3a",
                .features = &.{"v8_3a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv8_4a",
                .features = &.{"v8_4a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv8_5a",
                .features = &.{"v8_5a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv8_6a",
                .features = &.{"v8_6a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv8_7a",
                .features = &.{"v8_7a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv8_8a",
                .features = &.{"v8_8a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv8_9a",
                .features = &.{"v8_9a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv9a",
                .features = &.{"v9a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv9_1a",
                .features = &.{"v9_1a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv9_2a",
                .features = &.{"v9_2a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv9_3a",
                .features = &.{"v9_3a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv9_4a",
                .features = &.{"v9_4a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv9_5a",
                .features = &.{"v9_5a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv9_6a",
                .features = &.{"v9_6a"},
            },
            .{
                .llvm_name = "generic",
                .zig_name = "armv9_7a",
                .features = &.{"v9_7a"},
            },

            .{
                .llvm_name = "generic",
                .zig_name = "exynos_m1",
                .features = &.{
                    "crc",
                    "crypto",
                    "fuse_aes",
                    "perfmon",
                    "slow_misaligned_128store",
                    "slow_paired_128",
                    "v8a",
                },
            },
            .{
                .llvm_name = "generic",
                .zig_name = "exynos_m2",
                .features = &.{
                    "crc",
                    "crypto",
                    "fuse_aes",
                    "perfmon",
                    "slow_misaligned_128store",
                    "slow_paired_128",
                    "v8a",
                },
            },
            .{
                .llvm_name = "generic",
                .zig_name = "xgene1",
                .features = &.{
                    "perfmon",
                    "v8a",
                },
            },
            .{
                .llvm_name = "generic",
                .zig_name = "emag",
                .features = &.{
                    "crc",
                    "crypto",
                    "perfmon",
                    "v8a",
                },
            },
        },
    },
    .{
        .zig_name = "alpha",
        .llvm = null,
        .extra_features = &.{
            .{
                .zig_name = "bwx",
                .desc = "Enable byte/word extensions",
                .deps = &.{},
            },
            .{
                .zig_name = "cix",
                .desc = "Enable counting extensions",
                .deps = &.{},
            },
            .{
                .zig_name = "fix",
                .desc = "Enable FPR-to-GPR move and square root extensions",
                .deps = &.{},
            },
            .{
                .zig_name = "max",
                .desc = "Enable motion video extensions",
                .deps = &.{},
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "ev4",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "ev45",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "ev5",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "ev56",
                .features = &.{
                    "bwx",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pca56",
                .features = &.{
                    "bwx",
                    "max",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "ev6",
                .features = &.{
                    "bwx",
                    "fix",
                    "max",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "ev67",
                .features = &.{
                    "bwx",
                    "cix",
                    "fix",
                    "max",
                },
            },
        },
    },
    .{
        .zig_name = "amdgcn",
        .llvm = .{
            .name = "AMDGPU",
            .td_name = "AMDGPU",
        },
        .branch_quota = 2000,
        .feature_overrides = &.{
            // The `zig_name`s on these are placeholders to avoid collision with
            // the real features defined further below.
            .{
                .llvm_name = "gfx9",
                .zig_name = "__gfx9",
                .flatten = true,
            },
            .{
                .llvm_name = "gfx10",
                .zig_name = "__gfx10",
                .flatten = true,
            },
            .{
                .llvm_name = "gfx11",
                .zig_name = "__gfx11",
                .flatten = true,
            },
            .{
                .llvm_name = "gfx12",
                .zig_name = "__gfx12",
                .flatten = true,
            },
            .{
                .llvm_name = "gfx13",
                .zig_name = "__gfx13",
                .flatten = true,
            },
            .{
                .llvm_name = "sea-islands",
                .flatten = true,
            },
            .{
                .llvm_name = "southern-islands",
                .flatten = true,
            },
            .{
                .llvm_name = "volcanic-islands",
                .flatten = true,
            },

            .{
                .llvm_name = "assembler-permissive-wavesize",
                .omit = true,
            },
            .{
                .llvm_name = "block-vgpr-csr",
                .omit = true,
            },
            .{
                .llvm_name = "cumode",
                .omit = true,
            },
            .{
                .llvm_name = "DumpCode",
                .omit = true,
            },
            .{
                .llvm_name = "dumpcode",
                .omit = true,
            },
            .{
                .llvm_name = "enable-ds128",
                .omit = true,
            },
            .{
                .llvm_name = "enable-flat-scratch",
                .omit = true,
            },
            .{
                .llvm_name = "enable-prt-strict-null",
                .omit = true,
            },
            .{
                .llvm_name = "flat-for-global",
                .omit = true,
            },
            .{
                .llvm_name = "fmaf",
                .omit = true,
            },
            .{
                .llvm_name = "load-store-opt",
                .omit = true,
            },
            .{
                .llvm_name = "max-private-element-size-4",
                .omit = true,
            },
            .{
                .llvm_name = "max-private-element-size-8",
                .omit = true,
            },
            .{
                .llvm_name = "max-private-element-size-16",
                .omit = true,
            },
            .{
                .llvm_name = "precise-memory",
                .omit = true,
            },
            .{
                .llvm_name = "real-true16",
                .omit = true,
            },
            .{
                .llvm_name = "si-scheduler",
                .omit = true,
            },
            .{
                .llvm_name = "sramecc",
                .omit = true,
            },
            .{
                .llvm_name = "trap-handler",
                .omit = true,
            },
            .{
                .llvm_name = "unaligned-access-mode",
                .omit = true,
            },
            .{
                .llvm_name = "unsafe-ds-offset-folding",
                .omit = true,
            },
            .{
                .llvm_name = "use-add-pc64-inst",
                .omit = true,
            },
            .{
                .llvm_name = "xnack",
                .flatten = true,
            },

            .{
                .llvm_name = "ci-insts",
                .zig_name = "gfx7",
            },
            .{
                .llvm_name = "gfx8-insts",
                .zig_name = "gfx8",
                .extra_deps = &.{"gfx7"},
            },
            .{
                .llvm_name = "gfx9-insts",
                .zig_name = "gfx9",
                .extra_deps = &.{"gfx8"},
            },
            .{
                .llvm_name = "gfx90a-insts",
                .zig_name = "gfx9_0a",
                .extra_deps = &.{"gfx9"},
            },
            .{
                .llvm_name = "gfx940-insts",
                .zig_name = "gfx9_4",
                .extra_deps = &.{"gfx9_0a"},
            },
            .{
                .llvm_name = "gfx950-insts",
                .zig_name = "gfx9_5",
                .extra_deps = &.{"gfx9_4"},
            },
            .{
                .llvm_name = "gfx10-insts",
                .zig_name = "gfx10",
                .extra_deps = &.{"gfx9"},
            },
            .{
                .llvm_name = "gfx10-3-insts",
                .zig_name = "gfx10_3",
                .extra_deps = &.{"gfx10"},
            },
            .{
                .llvm_name = "gfx11-insts",
                .zig_name = "gfx11",
                .extra_deps = &.{"gfx10_3"},
            },
            .{
                .llvm_name = "gfx11-7-insts",
                .zig_name = "gfx11_7",
                .extra_deps = &.{"gfx11"},
            },
            .{
                .llvm_name = "gfx12-insts",
                .zig_name = "gfx12",
                .extra_deps = &.{"gfx11"},
            },
            .{
                .llvm_name = "gfx1250-insts",
                .zig_name = "gfx12_5",
                .extra_deps = &.{"gfx12"},
            },
            .{
                .llvm_name = "gfx13-insts",
                .zig_name = "gfx13",
                .extra_deps = &.{"gfx12"},
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
            .{
                .llvm_name = "generic-hsa",
                .omit = true,
            },
        },
    },
    .{
        .zig_name = "arc",
        .llvm = .{
            .name = "ARC",
            .td_name = "ARC",
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .zig_name = "arcv2",
            },
        },
    },
    .{
        .zig_name = "arm",
        .llvm = .{
            .name = "ARM",
            .td_name = "ARM",
        },
        .branch_quota = 10000,
        .feature_overrides = &.{
            .{
                .llvm_name = "a5",
                .flatten = true,
            },
            .{
                .llvm_name = "a7",
                .flatten = true,
            },
            .{
                .llvm_name = "a8",
                .flatten = true,
            },
            .{
                .llvm_name = "a9",
                .flatten = true,
            },
            .{
                .llvm_name = "a12",
                .flatten = true,
            },
            .{
                .llvm_name = "a15",
                .flatten = true,
            },
            .{
                .llvm_name = "a17",
                .flatten = true,
            },
            .{
                .llvm_name = "a32",
                .flatten = true,
            },
            .{
                .llvm_name = "a35",
                .flatten = true,
            },
            .{
                .llvm_name = "a53",
                .flatten = true,
            },
            .{
                .llvm_name = "a55",
                .flatten = true,
            },
            .{
                .llvm_name = "a57",
                .flatten = true,
            },
            .{
                .llvm_name = "a72",
                .flatten = true,
            },
            .{
                .llvm_name = "a73",
                .flatten = true,
            },
            .{
                .llvm_name = "a75",
                .flatten = true,
            },
            .{
                .llvm_name = "a76",
                .flatten = true,
            },
            .{
                .llvm_name = "a77",
                .flatten = true,
            },
            .{
                .llvm_name = "a78c",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-a78",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-a78ae",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-a510",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-a710",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-x1",
                .flatten = true,
            },
            .{
                .llvm_name = "cortex-x1c",
                .flatten = true,
            },
            .{
                .llvm_name = "exynos",
                .flatten = true,
            },
            .{
                .llvm_name = "krait",
                .flatten = true,
            },
            .{
                .llvm_name = "kryo",
                .flatten = true,
            },
            .{
                .llvm_name = "m3",
                .flatten = true,
            },
            .{
                .llvm_name = "m7",
                .flatten = true,
            },
            .{
                .llvm_name = "m55",
                .flatten = true,
            },
            .{
                .llvm_name = "m85",
                .flatten = true,
            },
            .{
                .llvm_name = "neoverse-v1",
                .flatten = true,
            },
            .{
                .llvm_name = "r4",
                .flatten = true,
            },
            .{
                .llvm_name = "r5",
                .flatten = true,
            },
            .{
                .llvm_name = "r7",
                .flatten = true,
            },
            .{
                .llvm_name = "r52",
                .flatten = true,
            },
            .{
                .llvm_name = "r52plus",
                .flatten = true,
            },
            .{
                .llvm_name = "swift",
                .flatten = true,
            },

            .{
                .llvm_name = "32bit",
                .omit = true,
            },
            .{
                .llvm_name = "aapcs-frame-chain",
                .omit = true,
            },
            .{
                .llvm_name = "big-endian-instructions",
                .omit = true,
            },
            .{
                .llvm_name = "disable-postra-scheduler",
                .omit = true,
            },
            .{
                .llvm_name = "execute-only",
                .omit = true,
            },
            .{
                .llvm_name = "harden-sls-blr",
                .omit = true,
            },
            .{
                .llvm_name = "harden-sls-nocomdat",
                .omit = true,
            },
            .{
                .llvm_name = "harden-sls-retbr",
                .omit = true,
            },
            .{
                .llvm_name = "long-calls",
                .omit = true,
            },
            .{
                .llvm_name = "no-bti-at-return-twice",
                .omit = true,
            },
            .{
                .llvm_name = "no-movt",
                .omit = true,
            },
            .{
                .llvm_name = "no-neg-immediates",
                .omit = true,
            },
            .{
                .llvm_name = "read-tp-tpidrprw",
                .omit = true,
            },
            .{
                .llvm_name = "read-tp-tpidruro",
                .omit = true,
            },
            .{
                .llvm_name = "read-tp-tpidrurw",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r9",
                .omit = true,
            },
            .{
                .llvm_name = "soft-float",
                .omit = true,
            },
            .{
                .llvm_name = "thumb-mode",
                .omit = true,
            },
            .{
                .llvm_name = "use-mipipeliner",
                .omit = true,
            },
            .{
                .llvm_name = "use-misched",
                .omit = true,
            },

            .{
                .llvm_name = "armv2",
                .zig_name = "v2",
                .extra_deps = &.{"no_unaligned_access"},
            },
            .{
                .llvm_name = "armv2a",
                .zig_name = "v2a",
                .extra_deps = &.{"no_unaligned_access"},
            },
            .{
                .llvm_name = "armv3",
                .zig_name = "v3",
                .extra_deps = &.{"no_unaligned_access"},
            },
            .{
                .llvm_name = "armv3m",
                .zig_name = "v3m",
                .extra_deps = &.{"no_unaligned_access"},
            },
            .{
                .llvm_name = "armv4",
                .zig_name = "v4",
                .extra_deps = &.{"no_unaligned_access"},
            },
            .{
                .llvm_name = "armv4t",
                .zig_name = "v4t",
                .extra_deps = &.{"no_unaligned_access"},
            },
            .{
                .llvm_name = "armv5t",
                .zig_name = "v5t",
                .extra_deps = &.{"no_unaligned_access"},
            },
            .{
                .llvm_name = "armv5te",
                .zig_name = "v5te",
                .extra_deps = &.{"no_unaligned_access"},
            },
            .{
                .llvm_name = "armv5tej",
                .zig_name = "v5tej",
                .extra_deps = &.{"no_unaligned_access"},
            },
            .{
                .llvm_name = "armv6",
                .zig_name = "v6",
            },
            .{
                .llvm_name = "armv6-m",
                .zig_name = "v6m",
            },
            .{
                .llvm_name = "armv6j",
                .zig_name = "v6j",
            },
            .{
                .llvm_name = "armv6k",
                .zig_name = "v6k",
            },
            .{
                .llvm_name = "armv6kz",
                .zig_name = "v6kz",
            },
            .{
                .llvm_name = "armv6s-m",
                .zig_name = "v6sm",
            },
            .{
                .llvm_name = "armv6t2",
                .zig_name = "v6t2",
            },
            .{
                .llvm_name = "armv7-a",
                .zig_name = "v7a",
                .omit_deps = &.{"neon"},
            },
            .{
                .llvm_name = "armv7-m",
                .zig_name = "v7m",
            },
            .{
                .llvm_name = "armv7-r",
                .zig_name = "v7r",
            },
            .{
                .llvm_name = "armv7e-m",
                .zig_name = "v7em",
            },
            .{
                .llvm_name = "armv7k",
                .omit = true,
            },
            .{
                .llvm_name = "armv7s",
                .omit = true,
            },
            .{
                .llvm_name = "armv7ve",
                .zig_name = "v7ve",
                .omit_deps = &.{"neon"},
            },
            .{
                .llvm_name = "armv8.1-a",
                .zig_name = "v8_1a",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "armv8.1-m.main",
                .zig_name = "v8_1m_main",
                .omit_deps = &.{"8msecext"},
            },
            .{
                .llvm_name = "armv8.2-a",
                .zig_name = "v8_2a",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "armv8.3-a",
                .zig_name = "v8_3a",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "armv8.4-a",
                .zig_name = "v8_4a",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "armv8.5-a",
                .zig_name = "v8_5a",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "armv8.6-a",
                .zig_name = "v8_6a",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "armv8.7-a",
                .zig_name = "v8_7a",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "armv8.8-a",
                .zig_name = "v8_8a",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "armv8.9-a",
                .zig_name = "v8_9a",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "armv8-a",
                .zig_name = "v8a",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "armv8-m.base",
                .zig_name = "v8m",
                .omit_deps = &.{"8msecext"},
            },
            .{
                .llvm_name = "armv8-m.main",
                .zig_name = "v8m_main",
                .omit_deps = &.{"8msecext"},
            },
            .{
                .llvm_name = "armv8-r",
                .zig_name = "v8r",
            },
            .{
                .llvm_name = "armv9.1-a",
                .zig_name = "v9_1a",
            },
            .{
                .llvm_name = "armv9.2-a",
                .zig_name = "v9_2a",
            },
            .{
                .llvm_name = "armv9.3-a",
                .zig_name = "v9_3a",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "armv9.4-a",
                .zig_name = "v9_4a",
            },
            .{
                .llvm_name = "armv9.5-a",
                .zig_name = "v9_5a",
            },
            .{
                .llvm_name = "armv9.6-a",
                .zig_name = "v9_6a",
            },
            .{
                .llvm_name = "armv9.7-a",
                .zig_name = "v9_7a",
            },
            .{
                .llvm_name = "armv9-a",
                .zig_name = "v9a",
            },
            .{
                .llvm_name = "strict-align",
                .zig_name = "no_unaligned_access",
            },
            .{
                .llvm_name = "v4t",
                .zig_name = "has_v4t",
            },
            .{
                .llvm_name = "v5t",
                .zig_name = "has_v5t",
            },
            .{
                .llvm_name = "v5te",
                .zig_name = "has_v5te",
            },
            .{
                .llvm_name = "v6",
                .zig_name = "has_v6",
            },
            .{
                .llvm_name = "v6k",
                .zig_name = "has_v6k",
            },
            .{
                .llvm_name = "v6m",
                .zig_name = "has_v6m",
            },
            .{
                .llvm_name = "v6t2",
                .zig_name = "has_v6t2",
            },
            .{
                .llvm_name = "v7",
                .zig_name = "has_v7",
            },
            .{
                .llvm_name = "v7clrex",
                .zig_name = "has_v7clrex",
            },
            .{
                .llvm_name = "v8",
                .zig_name = "has_v8",
            },
            .{
                .llvm_name = "v8m",
                .zig_name = "has_v8m",
            },
            .{
                .llvm_name = "v8m.main",
                .zig_name = "has_v8m_main",
            },
            .{
                .llvm_name = "v8.1a",
                .zig_name = "has_v8_1a",
            },
            .{
                .llvm_name = "v8.1m.main",
                .zig_name = "has_v8_1m_main",
            },
            .{
                .llvm_name = "v8.2a",
                .zig_name = "has_v8_2a",
            },
            .{
                .llvm_name = "v8.3a",
                .zig_name = "has_v8_3a",
            },
            .{
                .llvm_name = "v8.4a",
                .zig_name = "has_v8_4a",
            },
            .{
                .llvm_name = "v8.5a",
                .zig_name = "has_v8_5a",
            },
            .{
                .llvm_name = "v8.6a",
                .zig_name = "has_v8_6a",
            },
            .{
                .llvm_name = "v8.7a",
                .zig_name = "has_v8_7a",
            },
            .{
                .llvm_name = "v8.8a",
                .zig_name = "has_v8_8a",
            },
            .{
                .llvm_name = "v8.9a",
                .zig_name = "has_v8_9a",
            },
            .{
                .llvm_name = "v9a",
                .zig_name = "has_v9a",
            },
            .{
                .llvm_name = "v9.1a",
                .zig_name = "has_v9_1a",
            },
            .{
                .llvm_name = "v9.2a",
                .zig_name = "has_v9_2a",
            },
            .{
                .llvm_name = "v9.3a",
                .zig_name = "has_v9_3a",
            },
            .{
                .llvm_name = "v9.4a",
                .zig_name = "has_v9_4a",
            },
            .{
                .llvm_name = "v9.5a",
                .zig_name = "has_v9_5a",
            },
            .{
                .llvm_name = "v9.6a",
                .zig_name = "has_v9_6a",
            },
            .{
                .llvm_name = "v9.7a",
                .zig_name = "has_v9_7a",
            },
        },
        .extra_features = &.{
            // LLVM removed support for v2 and v3 but zig wants to support targeting old hardware
            .{
                .zig_name = "v2",
                .desc = "ARMv2 architecture",
                .deps = &.{"no_unaligned_access"},
            },
            .{
                .zig_name = "v2a",
                .desc = "ARMv2a architecture",
                .deps = &.{"no_unaligned_access"},
            },
            .{
                .zig_name = "v3",
                .desc = "ARMv3 architecture",
                .deps = &.{"no_unaligned_access"},
            },
            .{
                .zig_name = "v3m",
                .desc = "ARMv3m architecture",
                .deps = &.{"no_unaligned_access"},
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "cortex-a5",
                .omit_deps = &.{"vfp4"},
            },
            .{
                .llvm_name = "cortex-a7",
                .omit_deps = &.{"vfp4"},
            },
            .{
                .llvm_name = "cortex-a9",
                .omit_deps = &.{"fp16"},
            },
            .{
                .llvm_name = "cortex-a12",
                .omit_deps = &.{"vfp4"},
            },
            .{
                .llvm_name = "cortex-a15",
                .omit_deps = &.{"vfp4"},
            },
            .{
                .llvm_name = "cortex-a17",
                .omit_deps = &.{"vfp4"},
            },
            .{
                .llvm_name = "cortex-a32",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-a35",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-a53",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-a57",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-a72",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-a73",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-a76",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-a76ae",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-a77",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-a78",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-a78ae",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-a78c",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-m4",
                .omit_deps = &.{"vfp4d16sp"},
            },
            .{
                .llvm_name = "cortex-m7",
                .omit_deps = &.{"fp_armv8d16"},
            },
            .{
                .llvm_name = "cortex-m33",
                .omit_deps = &.{ "dsp", "fp_armv8d16sp" },
            },
            .{
                .llvm_name = "cortex-m35p",
                .omit_deps = &.{ "dsp", "fp_armv8d16sp" },
            },
            .{
                .llvm_name = "cortex-m52",
                .omit_deps = &.{ "fp_armv8d16", "mve_fp", "pacbti" },
            },
            .{
                .llvm_name = "cortex-m55",
                .omit_deps = &.{ "fp_armv8d16", "mve_fp" },
            },
            .{
                .llvm_name = "cortex-m85",
                .extra_deps = &.{"8msecext"},
                .omit_deps = &.{ "fp_armv8d16", "mve_fp", "pacbti" },
            },
            .{
                .llvm_name = "cortex-r5",
                .omit_deps = &.{"vfp3d16"},
            },
            .{
                .llvm_name = "cortex-r7",
                .omit_deps = &.{ "fp16", "vfp3d16" },
            },
            .{
                .llvm_name = "cortex-r8",
                .omit_deps = &.{ "fp16", "vfp3d16" },
            },
            .{
                .llvm_name = "cortex-r52",
                .omit_deps = &.{ "fp_armv8", "neon" },
            },
            .{
                .llvm_name = "cortex-r52plus",
                .omit_deps = &.{ "fp_armv8", "neon" },
            },
            .{
                .llvm_name = "cortex-x1",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "cortex-x1c",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "krait",
                .extra_deps = &.{"neon"},
            },
            .{
                .llvm_name = "neoverse-n1",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "neoverse-v1",
                .omit_deps = &.{"crypto"},
            },
            .{
                .llvm_name = "star-mc1",
                .omit_deps = &.{ "dsp", "fp_armv8d16sp" },
            },
            .{
                .llvm_name = "star-mc3",
                .omit_deps = &.{ "fp_armv8d16", "mve_fp", "pacbti" },
            },
            .{
                .llvm_name = "swift",
                .extra_deps = &.{"neon"},
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = "generic",
                .zig_name = "baseline",
                .features = &.{ "v7a", "vfp3d16" },
            },
            .{
                .llvm_name = "generic",
                .zig_name = "exynos_m1",
                .features = &.{ "v8a", "exynos" },
            },
            .{
                .llvm_name = "generic",
                .zig_name = "exynos_m2",
                .features = &.{ "v8a", "exynos" },
            },
            .{
                .llvm_name = "cortex-a9",
                .zig_name = "kermit",
                .features = &.{
                    "avoid_partial_cpsr",
                    "expand_fp_mlx",
                    "fp16",
                    "mp",
                    "muxed_units",
                    "neon",
                    "neon_fpmovs",
                    "prefer_vmovsr",
                    "ret_addr_stack",
                    "trustzone",
                    "v7a",
                    "vldn_align",
                    "vmlx_forwarding",
                    "vmlx_hazards",
                },
            },
        },
    },
    .{
        .zig_name = "avr",
        .llvm = .{
            .name = "AVR",
            .td_name = "AVR",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "special",
                .omit = true,
            },
        },
    },
    .{
        .zig_name = "bpf",
        .llvm = .{
            .name = "BPF",
            .td_name = "BPF",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "dummy",
                .omit = true,
            },
            .{
                .llvm_name = "dwarfris",
                .omit = true,
            },

            .{
                .llvm_name = "allows-misaligned-mem-access",
                .zig_name = "unaligned_access",
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
            .{
                .llvm_name = "probe",
                .omit = true,
            },
        },
    },
    .{
        .zig_name = "csky",
        .llvm = .{
            .name = "CSKY",
            .td_name = "CSKY",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "ccrt",
                .omit = true,
            },
            .{
                .llvm_name = "constpool",
                .omit = true,
            },
            .{
                .llvm_name = "hard-float",
                .omit = true,
            },
            .{
                .llvm_name = "hard-float-abi",
                .omit = true,
            },
            .{
                .llvm_name = "hard-tp",
                .omit = true,
            },
            .{
                .llvm_name = "multiple_stld",
                .omit = true,
            },
            .{
                .llvm_name = "nvic",
                .omit = true,
            },
            .{
                .llvm_name = "pushpop",
                .omit = true,
            },
            .{
                .llvm_name = "smart",
                .omit = true,
            },
            .{
                .llvm_name = "soft-tp",
                .omit = true,
            },
            .{
                .llvm_name = "stack-size",
                .omit = true,
            },

            .{
                .llvm_name = "ck801",
                .flatten = true,
            },
            .{
                .llvm_name = "ck802",
                .flatten = true,
            },
            .{
                .llvm_name = "ck803",
                .flatten = true,
            },
            .{
                .llvm_name = "ck803s",
                .flatten = true,
            },
            .{
                .llvm_name = "ck804",
                .flatten = true,
            },
            .{
                .llvm_name = "ck805",
                .flatten = true,
            },
            .{
                .llvm_name = "ck807",
                .flatten = true,
            },
            .{
                .llvm_name = "ck810",
                .flatten = true,
            },
            .{
                .llvm_name = "ck810v",
                .flatten = true,
            },
            .{
                .llvm_name = "ck860",
                .flatten = true,
            },
            .{
                .llvm_name = "ck860v",
                .flatten = true,
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "e2",
                .features = &.{"e2"},
            },
        },
    },
    .{
        .zig_name = "hexagon",
        .llvm = .{
            .name = "Hexagon",
            .td_name = "Hexagon",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "compound",
                .omit = true,
            },
            .{
                .llvm_name = "duplex",
                .omit = true,
            },
            .{
                .llvm_name = "long-calls",
                .omit = true,
            },
            .{
                .llvm_name = "memops",
                .omit = true,
            },
            .{
                .llvm_name = "noreturn-stack-elim",
                .omit = true,
            },
            .{
                .llvm_name = "nvj",
                .omit = true,
            },
            .{
                .llvm_name = "nvs",
                .omit = true,
            },
            .{
                .llvm_name = "packets",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r6",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r7",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r8",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r9",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r10",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r11",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r12",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r13",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r14",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r15",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r16",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r17",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r18",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r19",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r20",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r21",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r22",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r23",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r24",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r25",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r26",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r27",
                .omit = true,
            },
            .{
                .llvm_name = "reserved-r28",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r16",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r17",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r18",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r19",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r20",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r21",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r22",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r23",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r24",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r25",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r26",
                .omit = true,
            },
            .{
                .llvm_name = "scs-reg-r27",
                .omit = true,
            },
            .{
                .llvm_name = "small-data",
                .omit = true,
            },

            .{
                .llvm_name = "hvx-ieee-fp",
                .extra_deps = &.{"hvxv68"},
            },
            .{
                .llvm_name = "hvx-qfloat",
                .extra_deps = &.{"hvxv68"},
            },
            .{
                .llvm_name = "v55",
                .extra_deps = &.{"v5"},
            },
            .{
                .llvm_name = "v60",
                .extra_deps = &.{"v55"},
            },
            .{
                .llvm_name = "v62",
                .extra_deps = &.{"v60"},
            },
            .{
                .llvm_name = "v65",
                .extra_deps = &.{"v62"},
            },
            .{
                .llvm_name = "v66",
                .extra_deps = &.{"v65"},
            },
            .{
                .llvm_name = "v67",
                .extra_deps = &.{"v66"},
            },
            .{
                .llvm_name = "v68",
                .extra_deps = &.{"v67"},
            },
            .{
                .llvm_name = "v69",
                .extra_deps = &.{"v68"},
            },
            .{
                .llvm_name = "v71",
                .extra_deps = &.{"v69"},
            },
            .{
                .llvm_name = "v73",
                .extra_deps = &.{"v71"},
            },
            .{
                .llvm_name = "v75",
                .extra_deps = &.{"v73"},
            },
            .{
                .llvm_name = "v79",
                .extra_deps = &.{"v75"},
            },
            .{
                .llvm_name = "v81",
                .extra_deps = &.{"v79"},
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
        },
    },
    .{
        .zig_name = "hppa",
        .llvm = null,
        .extra_features = &.{
            .{
                .zig_name = "max_1",
                .desc = "Enable MAX-1 multimedia acceleration extensions",
                .deps = &.{},
            },
            .{
                .zig_name = "max_2",
                .desc = "Enable MAX-2 multimedia acceleration extensions",
                .deps = &.{"max_1"},
            },
            .{
                .zig_name = "v1_1",
                .desc = "Enable ISA v1.1",
                .deps = &.{},
            },
            .{
                .zig_name = "v2_0",
                .desc = "Enable ISA v2.0",
                .deps = &.{ "max_2", "v1_1" },
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "ts_1",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "ns_1",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "ns_2",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "pcx",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_7000",
                .features = &.{
                    "v1_1",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_7100",
                .features = &.{
                    "v1_1",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_7150",
                .features = &.{
                    "v1_1",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_7100lc",
                .features = &.{
                    "max_1",
                    "v1_1",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_7200",
                .features = &.{
                    "v1_1",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_7300lc",
                .features = &.{
                    "max_1",
                    "v1_1",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_8000",
                .features = &.{
                    "v2_0",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_8200",
                .features = &.{
                    "v2_0",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_8500",
                .features = &.{
                    "v2_0",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_8600",
                .features = &.{
                    "v2_0",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_8700",
                .features = &.{
                    "v2_0",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_8800",
                .features = &.{
                    "v2_0",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "pa_8900",
                .features = &.{
                    "v2_0",
                },
            },
        },
    },
    .{
        .zig_name = "kvx",
        .llvm = null,
        .extra_features = &.{
            .{
                .zig_name = "v3_1",
                .desc = "Enable ISA v3.1",
                .deps = &.{},
            },
            .{
                .zig_name = "v3_2",
                .desc = "Enable ISA v3.2",
                .deps = &.{},
            },
            .{
                .zig_name = "v4_1",
                .desc = "Enable ISA v4.1",
                .deps = &.{"v3_2"},
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "coolidge_v1",
                .features = &.{
                    "v3_1",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "coolidge_v2",
                .features = &.{
                    "v3_2",
                },
            },
        },
    },
    .{
        .zig_name = "lanai",
        .llvm = .{
            .name = "Lanai",
            .td_name = "Lanai",
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
        },
    },
    .{
        .zig_name = "loongarch",
        .llvm = .{
            .name = "LoongArch",
            .td_name = "LoongArch",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "la-global-with-abs",
                .omit = true,
            },
            .{
                .llvm_name = "la-global-with-pcrel",
                .omit = true,
            },
            .{
                .llvm_name = "la-local-with-abs",
                .omit = true,
            },
            .{
                .llvm_name = "prefer-w-inst",
                .omit = true,
            },
            .{
                .llvm_name = "relax",
                .omit = true,
            },

            .{
                .llvm_name = "ual",
                .zig_name = "unaligned_access",
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
            .{
                .llvm_name = "generic-la32",
                .omit = true,
            },
            .{
                .llvm_name = "generic-la64",
                .omit = true,
            },
            .{
                .llvm_name = "loongarch32",
                .omit = true,
            },
            .{
                .llvm_name = "loongarch64",
                .omit = true,
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "generic_la32",
                .features = &.{
                    "32bit",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "generic_la64",
                .features = &.{
                    "64bit",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "la32v1_0",
                .features = &.{
                    "32bit",
                    "32s",
                    "d",
                    "unaligned_access",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "la32rv1_0",
                .features = &.{
                    "32bit",
                    "unaligned_access",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "la64v1_0",
                .features = &.{
                    "64bit",
                    "lsx",
                    "unaligned_access",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "la64v1_1",
                .features = &.{
                    "64bit",
                    "div32",
                    "frecipe",
                    "lam_bh",
                    "lamcas",
                    "ld_seq_sa",
                    "lsx",
                    "scq",
                    "unaligned_access",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "la132",
                .features = &.{
                    "32bit",
                    "32s",
                },
            },
        },
    },
    .{
        .zig_name = "m68k",
        .llvm = .{
            .name = "M68k",
            .td_name = "M68k",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "reserve-a0",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-a1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-a2",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-a3",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-a4",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-a5",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-a6",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-d0",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-d1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-d2",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-d3",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-d4",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-d5",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-d6",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-d7",
                .omit = true,
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },

            .{
                .llvm_name = "M68000",
                .zig_name = "m68000",
            },
            .{
                .llvm_name = "M68010",
                .zig_name = "m68010",
            },
            .{
                .llvm_name = "M68020",
                .zig_name = "m68020",
                .extra_deps = &.{"isa_68881"},
            },
            .{
                .llvm_name = "M68030",
                .zig_name = "m68030",
                .extra_deps = &.{"isa_68881"},
            },
            .{
                .llvm_name = "M68040",
                .zig_name = "m68040",
            },
            .{
                .llvm_name = "M68060",
                .zig_name = "m68060",
            },
        },
    },
    .{
        .zig_name = "m88k",
        .llvm = null,
        .extra_features = &.{
            .{
                .zig_name = "isa_88110",
                .desc = "Enable MC88110 additions to the base instruction set",
                .deps = &.{},
            },
            .{
                .zig_name = "sfu1",
                .desc = "Enable floating-point unit",
                .deps = &.{},
            },
            .{
                .zig_name = "sfu2",
                .desc = "Enable graphics unit",
                .deps = &.{},
            },
            .{
                .zig_name = "xmem_imm",
                .desc = "Enable xmem with 16-bit immediate offset",
                .deps = &.{},
            },
            .{
                .zig_name = "xrf",
                .desc = "Enable extended register file",
                .deps = &.{"sfu1"},
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "m88000",
                .features = &.{
                    "sfu1",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "m88100",
                .features = &.{
                    "sfu1",
                    "xmem_imm",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "m88110",
                .features = &.{
                    "sfu1",
                    "sfu2",
                    "xrf",
                },
            },
        },
    },
    .{
        .zig_name = "microblaze",
        .llvm = null,
        .extra_features = &.{
            .{
                .zig_name = "barrel",
                .desc = "Enable barrel shift instructions (C_USE_BARREL)",
                .deps = &.{},
            },
            .{
                .zig_name = "div",
                .desc = "Enable hardware divide instructions (C_USE_DIV)",
                .deps = &.{},
            },
            .{
                .zig_name = "fpu",
                .desc = "Enable single-precision floating point instructions (C_USE_FPU >= 1)",
                .deps = &.{},
            },
            .{
                .zig_name = "fpu2",
                .desc = "Enable floating point conversion and square root instructions (C_USE_FPU = 2)",
                .deps = &.{"fpu"},
            },
            .{
                .zig_name = "msr",
                .desc = "Enable msrset and msrclr instructions (C_USE_MSR_INSTR)",
                .deps = &.{},
            },
            .{
                .zig_name = "mul",
                .desc = "Enable 32-bit multiply instructions (C_USE_HW_MUL >= 1)",
                .deps = &.{},
            },
            .{
                .zig_name = "mul64",
                .desc = "Enable multiply high instructions (C_USE_HW_MUL = 2)",
                .deps = &.{"mul"},
            },
            .{
                .zig_name = "pcmp",
                .desc = "Enable pattern compare instructions, and clz with v8.10 (C_USE_PCMP_INSTR)",
                .deps = &.{},
            },
            .{
                .zig_name = "pipe3",
                .desc = "Tune for the 3-stage pipeline (C_AREA_OPTIMIZED = 1)",
                .deps = &.{},
            },
            .{
                .zig_name = "pipe8",
                .desc = "Tune for the 8-stage pipeline (C_AREA_OPTIMIZED = 2)",
                .deps = &.{},
            },
            .{
                .zig_name = "reorder",
                .desc = "Enable byte-reversed load/store instructions, and swapb and swaph with v8.30 (C_USE_REORDER_INSTR)",
                .deps = &.{},
            },
            .{
                .zig_name = "v7_20",
                .desc = "Enable MicroBlaze v7.20 instructions (lwx, swx)",
                .deps = &.{},
            },
            .{
                .zig_name = "v8_10",
                .desc = "Enable MicroBlaze v8.10 instructions (mbar)",
                .deps = &.{"v7_20"},
            },
            .{
                .zig_name = "v8_30",
                .desc = "Enable MicroBlaze v8.30 instructions (swapb and swaph with reorder)",
                .deps = &.{"v8_10"},
            },
            .{
                .zig_name = "v10_0",
                .desc = "Enable MicroBlaze v10.0 instructions (bsefi and bsifi with barrel)",
                .deps = &.{"v8_30"},
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "v2_00",
                .features = &.{
                    "pipe3",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v2_10",
                .features = &.{
                    "pipe3",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v3_00",
                .features = &.{
                    "pipe3",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v4_00",
                .features = &.{
                    "pipe3",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v5_00",
                .features = &.{
                    "msr",
                    "pcmp",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v6_00",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "v7_00",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "v7_10",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "v7_20",
                .features = &.{
                    "v7_20",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v7_30",
                .features = &.{
                    "v7_20",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v8_00",
                .features = &.{
                    "reorder",
                    "v7_20",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v8_10",
                .features = &.{
                    "reorder",
                    "v8_10",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v8_20",
                .features = &.{
                    "reorder",
                    "v8_10",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v8_30",
                .features = &.{
                    "v8_30",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v8_40",
                .features = &.{
                    "v8_30",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v8_50",
                .features = &.{
                    "v8_30",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v9_0",
                .features = &.{
                    "v8_30",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v9_1",
                .features = &.{
                    "v8_30",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v9_2",
                .features = &.{
                    "v8_30",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v9_3",
                .features = &.{
                    "v8_30",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v9_4",
                .features = &.{
                    "v8_30",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v9_5",
                .features = &.{
                    "v8_30",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v9_6",
                .features = &.{
                    "v8_30",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v10_0",
                .features = &.{
                    "v10_0",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "v11_0",
                .features = &.{
                    "v10_0",
                },
            },
        },
    },
    .{
        .zig_name = "msp430",
        .llvm = .{
            .name = "MSP430",
            .td_name = "MSP430",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "hwmult32",
                .extra_deps = &.{"hwmult16"},
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
        },
    },
    .{
        .zig_name = "mips",
        .llvm = .{
            .name = "Mips",
            .td_name = "Mips",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "i6400",
                .flatten = true,
            },
            .{
                .llvm_name = "i6500",
                .flatten = true,
            },
            .{
                .llvm_name = "p5600",
                .flatten = true,
            },

            .{
                .llvm_name = "fp64",
                .omit = true,
            },
            .{
                .llvm_name = "fpxx",
                .omit = true,
            },
            .{
                .llvm_name = "gp64",
                .omit = true,
            },
            .{
                .llvm_name = "long-calls",
                .omit = true,
            },
            .{
                .llvm_name = "micromips",
                .omit = true,
            },
            .{
                .llvm_name = "mips16",
                .omit = true,
            },
            .{
                .llvm_name = "noabicalls",
                .omit = true,
            },
            .{
                .llvm_name = "nooddspreg",
                .omit = true,
            },
            .{
                .llvm_name = "ptr64",
                .omit = true,
            },
            .{
                .llvm_name = "soft-float",
                .omit = true,
            },
            .{
                .llvm_name = "sym32",
                .omit = true,
            },
            .{
                .llvm_name = "use-compact-branches",
                .omit = true,
            },
            .{
                .llvm_name = "use-indirect-jump-hazard",
                .omit = true,
            },
            .{
                .llvm_name = "use-tcc-in-div",
                .omit = true,
            },
            .{
                .llvm_name = "xgot",
                .omit = true,
            },

            .{
                .llvm_name = "strict-align",
                .zig_name = "no_unaligned_access",
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
            .{
                .llvm_name = "mips5",
                .omit = true,
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = "mips1",
                .zig_name = "r3000a",
                .features = &.{
                    "mips1",
                },
            },
            .{
                .llvm_name = "mips2",
                .zig_name = "allegrex",
                .features = &.{
                    "mips2",
                    "single_float",
                    "vfpu",
                },
            },
        },
    },
    .{
        .zig_name = "nvptx",
        .llvm = .{
            .name = "NVPTX",
            .td_name = "NVPTX",
        },
    },
    .{
        .zig_name = "or1k",
        .llvm = null,
        .extra_features = &.{
            .{
                .zig_name = "adrp",
                .desc = "Enable PC-relative page address instruction (l.adrp)",
                .deps = &.{},
            },
            .{
                .zig_name = "atomic",
                .desc = "Enable load-linked/store-conditional instructions (l.lwa, l.swa)",
                .deps = &.{},
            },
            .{
                .zig_name = "cmov",
                .desc = "Enable conditional move instruction (l.cmov)",
                .deps = &.{},
            },
            .{
                .zig_name = "csync",
                .desc = "Enable context synchronization instruction (l.csync)",
                .deps = &.{},
            },
            .{
                .zig_name = "div",
                .desc = "Enable hardware divide instructions (l.div, l.divu)",
                .deps = &.{},
            },
            .{
                .zig_name = "ext",
                .desc = "Enable sign/zero extension instructions (l.extbs, l.extbz, l.exths, l.exthz, l.extws, l.extwz)",
                .deps = &.{},
            },
            .{
                .zig_name = "ff1",
                .desc = "Enable find first 1 instruction (l.ff1)",
                .deps = &.{},
            },
            .{
                .zig_name = "fl1",
                .desc = "Enable find last 1 instruction (l.fl1)",
                .deps = &.{},
            },
            .{
                .zig_name = "mac",
                .desc = "Enable multiply-accumulate instructions (l.mac, l.maci, l.macrc, l.macu, l.msb, l.msbu)",
                .deps = &.{},
            },
            .{
                .zig_name = "madd_s",
                .desc = "Enable single-precision floating point multiply-add instruction (lf.madd.s)",
                .deps = &.{"orfpx32"},
            },
            .{
                .zig_name = "madd_d",
                .desc = "Enable double-precision floating point multiply-add instruction (lf.madd.d)",
                .deps = &.{"orfpx64a32"},
            },
            .{
                .zig_name = "msync",
                .desc = "Enable memory synchronization instruction (l.msync)",
                .deps = &.{},
            },
            .{
                .zig_name = "mul",
                .desc = "Enable hardware multiply instructions (l.mul, l.muli, l.mulu)",
                .deps = &.{},
            },
            .{
                .zig_name = "muld",
                .desc = "Enable widening multiply instructions (l.muld, l.muldu)",
                .deps = &.{},
            },
            .{
                .zig_name = "no_addc",
                .desc = "Disable add with carry instructions (l.addc, l.addic)",
                .deps = &.{},
            },
            .{
                .zig_name = "no_sr_cy",
                .desc = "Disable carry flag (SR[CY]) updates",
                .deps = &.{"no_addc"},
            },
            .{
                .zig_name = "no_sr_ov",
                .desc = "Disable overflow flag (SR[OV]) updates",
                .deps = &.{},
            },
            .{
                .zig_name = "no_sra",
                .desc = "Disable arithmetic shift right instructions (l.sra, l.srai)",
                .deps = &.{},
            },
            .{
                .zig_name = "orfpx32",
                .desc = "Enable single-precision floating point instructions",
                .deps = &.{},
            },
            .{
                .zig_name = "orfpx64a32",
                .desc = "Enable double-precision floating point instructions on register pairs",
                .deps = &.{ "orfpx32", "sfun_s" },
            },
            .{
                .zig_name = "psync",
                .desc = "Enable pipeline synchronization instruction (l.psync)",
                .deps = &.{},
            },
            .{
                .zig_name = "ror",
                .desc = "Enable rotate right instructions (l.ror, l.rori)",
                .deps = &.{},
            },
            .{
                .zig_name = "sfimm",
                .desc = "Enable set flag with immediate instructions (l.sfeqi, l.sfnei, l.sfgtsi, l.sfgtui, l.sfgesi, l.sfgeui, l.sfltsi, l.sfltui, l.sflesi, l.sfleui)",
                .deps = &.{},
            },
            .{
                .zig_name = "sfun_s",
                .desc = "Enable unordered single-precision floating point comparison instructions (lf.sfueq.s, lf.sfune.s, lf.sfugt.s, lf.sfuge.s, lf.sfult.s, lf.sfule.s, lf.sfun.s)",
                .deps = &.{"orfpx32"},
            },
            .{
                .zig_name = "stod_dtos",
                .desc = "Enable floating point precision conversion instructions (lf.stod.d, lf.dtos.d)",
                .deps = &.{"orfpx64a32"},
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "ar100",
                .features = &.{
                    "csync",
                    "ff1",
                    "msync",
                    "psync",
                    "sfimm",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "marocchino",
                .features = &.{
                    "atomic",
                    "cmov",
                    "div",
                    "ext",
                    "ff1",
                    "fl1",
                    "msync",
                    "mul",
                    "orfpx64a32",
                    "ror",
                    "sfimm",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "mor1kx",
                .features = &.{
                    "atomic",
                    "cmov",
                    "div",
                    "ff1",
                    "fl1",
                    "msync",
                    "mul",
                    "sfimm",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "or1200",
                .features = &.{
                    "cmov",
                    "csync",
                    "div",
                    "ext",
                    "ff1",
                    "fl1",
                    "msync",
                    "mul",
                    "psync",
                    "sfimm",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "orbis32",
                .features = &.{},
            },
        },
    },
    .{
        .zig_name = "powerpc",
        .llvm = .{
            .name = "PowerPC",
            .td_name = "PPC",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "e500",
                .flatten = true,
            },
            .{
                .llvm_name = "ppc4xx",
                .flatten = true,
            },
            .{
                .llvm_name = "ppc6xx",
                .flatten = true,
            },

            .{
                .llvm_name = "64bit",
                .omit = true,
            },
            .{
                .llvm_name = "64bitregs",
                .omit = true,
            },
            .{
                .llvm_name = "aix",
                .omit = true,
            },
            .{
                .llvm_name = "aix-shared-lib-tls-model-opt",
                .omit = true,
            },
            .{
                .llvm_name = "aix-small-local-dynamic-tls",
                .omit = true,
            },
            .{
                .llvm_name = "aix-small-local-exec-tls",
                .omit = true,
            },
            .{
                .llvm_name = "crbits",
                .omit = true,
            },
            .{
                .llvm_name = "hard-float",
                .omit = true,
            },
            .{
                .llvm_name = "invariant-function-descriptors",
                .omit = true,
            },
            .{
                .llvm_name = "longcall",
                .omit = true,
            },
            .{
                .llvm_name = "modern-aix-as",
                .omit = true,
            },
            .{
                .llvm_name = "ppc-postra-sched",
                .omit = true,
            },
            .{
                .llvm_name = "ppc-prera-sched",
                .omit = true,
            },
            .{
                .llvm_name = "privileged",
                .omit = true,
            },
            .{
                .llvm_name = "rop-protect",
                .omit = true,
            },
            .{
                .llvm_name = "secure-plt",
                .omit = true,
            },
            .{
                .llvm_name = "use-ptrgl-helper",
                .omit = true,
            },

            // Future ISA extensions; omit until they are stabilized.
            .{
                .llvm_name = "future-vector",
                .omit = true,
            },
            .{
                .llvm_name = "isa-future-instructions",
                .omit = true,
            },

            .{
                .llvm_name = "64bit-support",
                .zig_name = "64bit",
            },
            .{
                .llvm_name = "fast-MFLR",
                .zig_name = "fast_mflr",
            },
            .{
                .llvm_name = "allow-unaligned-fp-access",
                .zig_name = "unaligned_fp_access",
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
            .{
                .llvm_name = "ppc",
                .omit = true,
            },

            // Future ISA extensions; omit until they are stabilized.
            .{
                .llvm_name = "future",
                .omit = true,
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = "970",
                .zig_name = "cell",
                .features = &.{
                    "64bit",
                    "altivec",
                    "fres",
                    "frsqrte",
                    "fsqrt",
                    "mfocrf",
                    "stfiwx",
                },
            },
            .{
                .llvm_name = "e5500",
                .zig_name = "e6500",
                .features = &.{
                    "64bit",
                    "altivec",
                    "booke",
                    "bpermd",
                    "cmpb",
                    "fres",
                    "frsqrte",
                    "isel",
                    "mfocrf",
                    "partword_atomics",
                    "popcntd",
                    "stfiwx",
                },
            },
        },
    },
    .{
        .zig_name = "propeller",
        .llvm = null,
        .extra_features = &.{
            .{
                .zig_name = "p2",
                .desc = "Enable Propeller 2",
                .deps = &.{},
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "p1",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "p2",
                .features = &.{"p2"},
            },
        },
    },
    .{
        .zig_name = "spirv",
        .llvm = .{
            .name = "SPIRV",
            .td_name = "SPIRV",
        },
        .branch_quota = 2000,
        .extra_features = spirv_extra_features,
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "v1_0",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "v1_1",
                .features = &.{"v1_1"},
            },
            .{
                .llvm_name = null,
                .zig_name = "v1_2",
                .features = &.{"v1_2"},
            },
            .{
                .llvm_name = null,
                .zig_name = "v1_3",
                .features = &.{"v1_3"},
            },
            .{
                .llvm_name = null,
                .zig_name = "v1_4",
                .features = &.{"v1_4"},
            },
            .{
                .llvm_name = null,
                .zig_name = "v1_5",
                .features = &.{"v1_5"},
            },
            .{
                .llvm_name = null,
                .zig_name = "v1_6",
                .features = &.{"v1_6"},
            },
            .{
                .llvm_name = null,
                .zig_name = "opencl_v2_1",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "opencl_v2_2",
                .features = &.{"v1_2"},
            },
            .{
                .llvm_name = null,
                .zig_name = "opencl_v3_1",
                .features = &.{"v1_4"},
            },
            .{
                .llvm_name = null,
                .zig_name = "opengl_v4_6",
                .features = &.{"v1_4"},
            },
            .{
                .llvm_name = null,
                .zig_name = "vulkan_v1_0",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "vulkan_v1_1",
                .features = &.{"v1_3"},
            },
            .{
                .llvm_name = null,
                .zig_name = "vulkan_v1_2",
                .features = &.{"v1_5"},
            },
            .{
                .llvm_name = null,
                .zig_name = "vulkan_v1_3",
                .features = &.{"v1_6"},
            },
            .{
                .llvm_name = null,
                .zig_name = "vulkan_v1_4",
                .features = &.{"v1_6"},
            },
        },
    },
    .{
        .zig_name = "riscv",
        .llvm = .{
            .name = "RISCV",
            .td_name = "RISCV",
        },
        .branch_quota = 2000,
        .feature_overrides = &.{
            .{
                .llvm_name = "rva20s64",
                .flatten = true,
            },
            .{
                .llvm_name = "rva20u64",
                .flatten = true,
            },
            .{
                .llvm_name = "rva22s64",
                .flatten = true,
            },
            .{
                .llvm_name = "rva22u64",
                .flatten = true,
            },
            .{
                .llvm_name = "rva23s64",
                .flatten = true,
            },
            .{
                .llvm_name = "rva23u64",
                .flatten = true,
            },
            .{
                .llvm_name = "rvb23s64",
                .flatten = true,
            },
            .{
                .llvm_name = "rvb23u64",
                .flatten = true,
            },
            .{
                .llvm_name = "rvi20u32",
                .flatten = true,
            },
            .{
                .llvm_name = "rvi20u64",
                .flatten = true,
            },
            .{
                .llvm_name = "andes45",
                .flatten = true,
            },
            .{
                .llvm_name = "mips-p8700",
                .flatten = true,
            },
            .{
                .llvm_name = "sifive7",
                .flatten = true,
            },
            .{
                .llvm_name = "ventana-veyron",
                .flatten = true,
            },

            .{
                .llvm_name = "exact-asm",
                .omit = true,
            },
            .{
                .llvm_name = "disable-latency-sched-heuristic",
                .omit = true,
            },
            .{
                .llvm_name = "disable-misched-load-clustering",
                .omit = true,
            },
            .{
                .llvm_name = "disable-misched-store-clustering",
                .omit = true,
            },
            .{
                .llvm_name = "disable-postmisched-load-clustering",
                .omit = true,
            },
            .{
                .llvm_name = "disable-postmisched-store-clustering",
                .omit = true,
            },
            .{
                .llvm_name = "enable-select-opt",
                .omit = true,
            },
            .{
                .llvm_name = "enable-vsetvli-sched-heuristic",
                .omit = true,
            },
            .{
                .llvm_name = "forced-atomics",
                .omit = true,
            },
            .{
                .llvm_name = "no-default-unroll",
                .omit = true,
            },
            .{
                .llvm_name = "no-trailing-seq-cst-fence",
                .omit = true,
            },
            .{
                .llvm_name = "relax",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x2",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x3",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x4",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x5",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x6",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x7",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x8",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x9",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x10",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x11",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x12",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x13",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x14",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x15",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x16",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x17",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x18",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x19",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x20",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x21",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x22",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x23",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x24",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x25",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x26",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x27",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x28",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x29",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x30",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-x31",
                .omit = true,
            },
            .{
                .llvm_name = "save-restore",
                .omit = true,
            },
            .{
                .llvm_name = "tagged-globals",
                .omit = true,
            },
            .{
                .llvm_name = "use-postra-scheduler",
                .omit = true,
            },
            .{
                .llvm_name = "xcheriot",
                .omit = true,
            },

            // Future ISA extensions; omit until they are stabilized.
            .{
                .llvm_name = "experimental",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-p",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-rvm23u32",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-smpmpmt",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-svukte",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-xqccmt",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-xsfmclic",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-xsfsclic",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-y",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zibi",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zicfilp",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zicfiss",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvabd",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvbc32e",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvdot4a8i",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvfbdota32f",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvfbfa",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvfofp8min",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvfqwbdota8f",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvfqwdota8f",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvfwbdota16bf",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvfwdota16bf",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvkgs",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvqwbdota16i",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvqwbdota8i",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvqwdota16i",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvqwdota8i",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvvfmm",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvvmm",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvvmtls",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvvmttls",
                .omit = true,
            },
            .{
                .llvm_name = "experimental-zvzip",
                .omit = true,
            },

            .{
                .llvm_name = "unaligned-scalar-mem",
                .zig_name = "unaligned_access",
            },
            .{
                .llvm_name = "unaligned-vector-mem",
                .zig_name = "unaligned_vector_access",
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
            .{
                .llvm_name = "generic-ooo",
                .omit = true,
            },
            .{
                .llvm_name = "generic-rv32",
                .omit = true,
            },
            .{
                .llvm_name = "generic-rv64",
                .omit = true,
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "rv32e",
                .features = &.{ "32bit", "e" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv32ec",
                .features = &.{ "32bit", "c", "e" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv32emc",
                .features = &.{ "32bit", "c", "e", "m" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv32gc",
                .features = &.{ "32bit", "a", "c", "d", "i", "m", "zcd", "zcf", "zicsr", "zifencei" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv32i",
                .features = &.{ "32bit", "i" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv32im",
                .features = &.{ "32bit", "i", "m" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv32ima",
                .features = &.{ "32bit", "a", "i", "m" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv32imac",
                .features = &.{ "32bit", "a", "c", "i", "m" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv32imafc",
                .features = &.{ "32bit", "a", "c", "f", "i", "m", "zcf" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv32imafdc",
                .features = &.{ "32bit", "a", "c", "d", "i", "m", "zcd", "zcf" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv32imc",
                .features = &.{ "32bit", "c", "i", "m" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv64gc",
                .features = &.{ "64bit", "a", "c", "d", "i", "m", "zcd", "zicsr", "zifencei" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv64i",
                .features = &.{ "64bit", "i" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv64im",
                .features = &.{ "64bit", "i", "m" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv64ima",
                .features = &.{ "64bit", "a", "i", "m" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv64imac",
                .features = &.{ "64bit", "a", "c", "i", "m" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv64imafc",
                .features = &.{ "64bit", "a", "c", "f", "i", "m" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv64imafdc",
                .features = &.{ "64bit", "a", "c", "d", "i", "m", "zcd" },
            },
            .{
                .llvm_name = null,
                .zig_name = "rv64imc",
                .features = &.{ "64bit", "c", "i", "m" },
            },

            .{
                .llvm_name = null,
                .zig_name = "rva20s64",
                .features = &.{"rva20s64"},
            },
            .{
                .llvm_name = null,
                .zig_name = "rva20u64",
                .features = &.{"rva20u64"},
            },
            .{
                .llvm_name = null,
                .zig_name = "rva22s64",
                .features = &.{"rva22s64"},
            },
            .{
                .llvm_name = null,
                .zig_name = "rva22u64",
                .features = &.{"rva22u64"},
            },
            .{
                .llvm_name = null,
                .zig_name = "rva23s64",
                .features = &.{"rva23s64"},
            },
            .{
                .llvm_name = null,
                .zig_name = "rva23u64",
                .features = &.{"rva23u64"},
            },
            .{
                .llvm_name = null,
                .zig_name = "rvb23s64",
                .features = &.{"rvb23s64"},
            },
            .{
                .llvm_name = null,
                .zig_name = "rvb23u64",
                .features = &.{"rvb23u64"},
            },
            .{
                .llvm_name = null,
                .zig_name = "rvi20u32",
                .features = &.{"rvi20u32"},
            },
            .{
                .llvm_name = null,
                .zig_name = "rvi20u64",
                .features = &.{"rvi20u64"},
            },
        },
    },
    .{
        .zig_name = "sh",
        .llvm = null,
        .extra_features = &.{
            .{
                .zig_name = "cas",
                .desc = "Enable cas.l instruction",
                .deps = &.{},
            },
            .{
                .zig_name = "dsp",
                .desc = "Enable DSP extension",
                .deps = &.{},
            },
            .{
                .zig_name = "dynshift",
                .desc = "Enable dynamic shift instructions",
                .deps = &.{},
            },
            .{
                .zig_name = "fpu",
                .desc = "Enable floating point unit",
                .deps = &.{},
            },
            .{
                .zig_name = "fsca",
                .desc = "Enable fsca instruction",
                .deps = &.{"fpu"},
            },
            .{
                .zig_name = "fsrra",
                .desc = "Enable fsrra instruction",
                .deps = &.{"fpu"},
            },
            .{
                .zig_name = "mmu",
                .desc = "Enable memory management unit",
                .deps = &.{},
            },
            .{
                .zig_name = "sh2",
                .desc = "Enable SH-2 instructions",
                .deps = &.{},
            },
            .{
                .zig_name = "sh2a",
                .desc = "Enable SH-2A instructions",
                .deps = &.{ "dynshift", "sh2" },
            },
            .{
                .zig_name = "sh3",
                .desc = "Enable SH-3 instructions",
                .deps = &.{ "dynshift", "sh2" },
            },
            .{
                .zig_name = "sh4",
                .desc = "Enable SH-4 instructions",
                .deps = &.{"sh3"},
            },
            .{
                .zig_name = "sh4a",
                .desc = "Enable SH-4A instructions",
                .deps = &.{"sh4"},
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "j2",
                .features = &.{
                    "cas",
                    "dynshift",
                    "sh2",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh1",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "sh2",
                .features = &.{
                    "sh2",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh2_dsp",
                .features = &.{
                    "dsp",
                    "sh2",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh2a",
                .features = &.{
                    "fpu",
                    "sh2a",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh2a_nofpu",
                .features = &.{
                    "sh2a",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh2e",
                .features = &.{
                    "fpu",
                    "sh2",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh3",
                .features = &.{
                    "mmu",
                    "sh3",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh3_dsp",
                .features = &.{
                    "dsp",
                    "mmu",
                    "sh3",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh3e",
                .features = &.{
                    "fpu",
                    "mmu",
                    "sh3",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh4",
                .features = &.{
                    "fpu",
                    "mmu",
                    "sh4",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh4_nofpu",
                .features = &.{
                    "mmu",
                    "sh4",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh4a",
                .features = &.{
                    "fpu",
                    "fsca",
                    "fsrra",
                    "mmu",
                    "sh4a",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh4a_nofpu",
                .features = &.{
                    "mmu",
                    "sh4a",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "sh4al_dsp",
                .features = &.{
                    "dsp",
                    "mmu",
                    "sh4a",
                },
            },
        },
    },
    .{
        .zig_name = "sparc",
        .llvm = .{
            .name = "Sparc",
            .td_name = "Sparc",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "leon",
                .flatten = true,
            },

            .{
                .llvm_name = "64bit",
                .omit = true,
            },
            .{
                .llvm_name = "detectroundchange",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-g1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-g2",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-g3",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-g4",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-g5",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-g6",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-g7",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-i0",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-i1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-i2",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-i3",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-i4",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-i5",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-l0",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-l1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-l2",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-l3",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-l4",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-l5",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-l6",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-l7",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-o0",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-o1",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-o2",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-o3",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-o4",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-o5",
                .omit = true,
            },
            .{
                .llvm_name = "soft-float",
                .omit = true,
            },
            .{
                .llvm_name = "v8plus",
                .omit = true,
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
        },
    },
    .{
        .zig_name = "s390x",
        .llvm = .{
            .name = "SystemZ",
            .td_name = "SystemZ",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "backchain",
                .omit = true,
            },
            .{
                .llvm_name = "soft-float",
                .omit = true,
            },
            .{
                .llvm_name = "unaligned-symbols",
                .omit = true,
            },

            // LLVM's names for s390x features are horribly long.
            .{
                .llvm_name = "bear-enhancement",
                .zig_name = "beareh",
            },
            .{
                .llvm_name = "concurrent-functions",
                .zig_name = "ccf",
            },
            .{
                .llvm_name = "deflate-conversion",
                .zig_name = "deflate",
            },
            .{
                .llvm_name = "dfp-packed-conversion",
                .zig_name = "dfppc",
            },
            .{
                .llvm_name = "dfp-zoned-conversion",
                .zig_name = "dfpzc",
            },
            .{
                .llvm_name = "distinct-ops",
                .zig_name = "distops",
            },
            .{
                .llvm_name = "enhanced-dat-2",
                .zig_name = "edat2",
            },
            .{
                .llvm_name = "enhanced-sort",
                .zig_name = "esort",
            },
            .{
                .llvm_name = "execution-hint",
                .zig_name = "exhint",
            },
            .{
                .llvm_name = "fast-serialization",
                .zig_name = "fbcrs",
            },
            .{
                .llvm_name = "fp-extension",
                .zig_name = "fpe",
            },
            .{
                .llvm_name = "guarded-storage",
                .zig_name = "gs",
            },
            .{
                .llvm_name = "high-word",
                .zig_name = "hiword",
            },
            .{
                .llvm_name = "insert-reference-bits-multiple",
                .zig_name = "irbm",
            },
            .{
                .llvm_name = "interlocked-access1",
                .zig_name = "iacc",
            },
            .{
                .llvm_name = "load-and-trap",
                .zig_name = "lat",
            },
            .{
                .llvm_name = "load-and-zero-rightmost-byte",
                .zig_name = "lzr",
            },
            .{
                .llvm_name = "load-store-on-cond",
                .zig_name = "loc",
            },
            .{
                .llvm_name = "load-store-on-cond-2",
                .zig_name = "loc2",
            },
            .{
                .llvm_name = "message-security-assist-extension3",
                .zig_name = "msa3",
            },
            .{
                .llvm_name = "message-security-assist-extension4",
                .zig_name = "msa4",
            },
            .{
                .llvm_name = "message-security-assist-extension5",
                .zig_name = "msa5",
            },
            .{
                .llvm_name = "message-security-assist-extension7",
                .zig_name = "msa7",
                .extra_deps = &.{"msa5"},
            },
            .{
                .llvm_name = "message-security-assist-extension8",
                .zig_name = "msa8",
                .extra_deps = &.{"msa3"},
            },
            .{
                .llvm_name = "message-security-assist-extension9",
                .zig_name = "msa9",
                .extra_deps = &.{ "msa3", "msa4" },
            },
            .{
                .llvm_name = "message-security-assist-extension12",
                .zig_name = "msa12",
            },
            .{
                .llvm_name = "miscellaneous-extensions",
                .zig_name = "minste",
            },
            .{
                .llvm_name = "miscellaneous-extensions-2",
                .zig_name = "minste2",
            },
            .{
                .llvm_name = "miscellaneous-extensions-3",
                .zig_name = "minste3",
                .extra_deps = &.{ "distops", "fbcrs", "hiword", "iacc", "loc", "popcnt" },
            },
            .{
                .llvm_name = "miscellaneous-extensions-4",
                .zig_name = "minste4",
            },
            .{
                .llvm_name = "nnp-assist",
                .zig_name = "nnpa",
                .extra_deps = &.{"vx"},
            },
            .{
                .llvm_name = "population-count",
                .zig_name = "popcnt",
            },
            .{
                .llvm_name = "processor-activity-instrumentation",
                .zig_name = "pai",
            },
            .{
                .llvm_name = "processor-assist",
                .zig_name = "ppa",
            },
            .{
                .llvm_name = "reset-dat-protection",
                .zig_name = "rdp",
            },
            .{
                .llvm_name = "reset-reference-bits-multiple",
                .zig_name = "rrbm",
            },
            .{
                .llvm_name = "test-pending-external-interruption",
                .zig_name = "tpei",
            },
            .{
                .llvm_name = "transactional-execution",
                .zig_name = "te",
                .extra_deps = &.{ "exhint", "lat", "minste", "ppa" },
            },
            .{
                .llvm_name = "vector",
                .zig_name = "vx",
            },
            .{
                .llvm_name = "vector-enhancements-1",
                .zig_name = "vxeh",
                .extra_deps = &.{"vx"},
            },
            .{
                .llvm_name = "vector-enhancements-2",
                .zig_name = "vxeh2",
                .extra_deps = &.{"vxeh"},
            },
            .{
                .llvm_name = "vector-enhancements-3",
                .zig_name = "vxeh3",
                .extra_deps = &.{"vxeh2"},
            },
            .{
                .llvm_name = "vector-packed-decimal",
                .zig_name = "vxpd",
                .extra_deps = &.{"vx"},
            },
            .{
                .llvm_name = "vector-packed-decimal-enhancement",
                .zig_name = "vxpdeh",
                .extra_deps = &.{"vxpd"},
            },
            .{
                .llvm_name = "vector-packed-decimal-enhancement-2",
                .zig_name = "vxpdeh2",
                .extra_deps = &.{"vxpdeh"},
            },
            .{
                .llvm_name = "vector-packed-decimal-enhancement-3",
                .zig_name = "vxpdeh3",
                .extra_deps = &.{"vxpdeh2"},
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
        },
    },
    .{
        .zig_name = "ve",
        .llvm = .{
            .name = "VE",
            .td_name = "VE",
        },
        .extra_features = &.{
            .{
                .zig_name = "ve3",
                .desc = "Enable VE3 instructions",
                .deps = &.{},
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "ve10",
                .features = &.{"vpu"},
            },
            .{
                .llvm_name = null,
                .zig_name = "ve20",
                .features = &.{"vpu"},
            },
            .{
                .llvm_name = null,
                .zig_name = "ve30",
                .features = &.{ "ve3", "vpu" },
            },
        },
    },
    .{
        .zig_name = "wasm",
        .llvm = .{
            .name = "WebAssembly",
            .td_name = "WebAssembly",
        },
        // For whatever reason, LLVM's WebAssembly backend sets these implied features in code
        // rather than making them proper dependencies, so fix that here...
        .feature_overrides = &.{
            .{
                .llvm_name = "bulk-memory",
                .extra_deps = &.{"bulk_memory_opt"},
            },
            .{
                .llvm_name = "fp16",
                .extra_deps = &.{"simd128"},
            },
            .{
                .llvm_name = "reference-types",
                .extra_deps = &.{ "bulk_memory", "call_indirect_overlong" },
            },
            .{
                .llvm_name = "relaxed-atomics",
                .extra_deps = &.{"atomics"},
            },
            .{
                .llvm_name = "relaxed-simd",
                .extra_deps = &.{"simd128"},
            },
        },
        .extra_features = &.{
            .{
                .zig_name = "nontrapping_bulk_memory_len0",
                .desc = "Bulk memory operations with a zero length do not trap",
                .deps = &.{"bulk_memory_opt"},
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "bleeding-edge",
                .omit = true,
            },
            .{
                .llvm_name = "generic",
                .omit = true,
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "wasm_v1_0",
                .features = &.{
                    "mutable_globals",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "wasm_v2_0",
                .features = &.{
                    "multivalue",
                    "mutable_globals",
                    "nontrapping_fptoint",
                    "reference_types",
                    "sign_ext",
                    "simd128",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "wasm_v3_0",
                .features = &.{
                    "exception_handling",
                    "extended_const",
                    "gc",
                    "multimemory",
                    "mutable_globals",
                    "nontrapping_fptoint",
                    "relaxed_simd",
                    "sign_ext",
                    "tail_call",
                },
            },
        },
    },
    .{
        .zig_name = "x86",
        .llvm = .{
            .name = "X86",
            .td_name = "X86",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "64bit-mode",
                .omit = true,
            },
            .{
                .llvm_name = "avx10.1-512",
                .omit = true,
            },
            .{
                .llvm_name = "avx10.2-512",
                .omit = true,
            },
            .{
                .llvm_name = "evex512",
                .omit = true,
            },
            .{
                .llvm_name = "harden-sls-ijmp",
                .omit = true,
            },
            .{
                .llvm_name = "harden-sls-ret",
                .omit = true,
            },
            .{
                .llvm_name = "inline-asm-use-gpr32",
                .omit = true,
            },
            .{
                .llvm_name = "lvi-cfi",
                .omit = true,
            },
            .{
                .llvm_name = "lvi-load-hardening",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-edi",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r8",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r9",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r10",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r11",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r12",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r13",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r14",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r15",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r16",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r17",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r18",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r19",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r20",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r21",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r22",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r23",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r24",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r25",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r26",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r27",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r28",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r29",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r30",
                .omit = true,
            },
            .{
                .llvm_name = "reserve-r31",
                .omit = true,
            },
            .{
                .llvm_name = "retpoline",
                .omit = true,
            },
            .{
                .llvm_name = "retpoline-external-thunk",
                .omit = true,
            },
            .{
                .llvm_name = "retpoline-indirect-branches",
                .omit = true,
            },
            .{
                .llvm_name = "retpoline-indirect-calls",
                .omit = true,
            },
            .{
                .llvm_name = "seses",
                .omit = true,
            },
            .{
                .llvm_name = "soft-float",
                .omit = true,
            },
            .{
                .llvm_name = "tagged-globals",
                .omit = true,
            },
            .{
                .llvm_name = "x32",
                .omit = true,
            },

            .{
                .llvm_name = "sse4.2",
                .extra_deps = &.{"crc32"},
            },
            .{
                .llvm_name = "sse-unaligned-mem",
                .zig_name = "unaligned_sse_access",
            },
        },
        .extra_features = &.{
            // Features removed from LLVM
            .{
                .zig_name = "3dnow",
                .desc = "Enable 3DNow! instructions",
                .deps = &.{"mmx"},
            },
            .{
                .zig_name = "3dnowa",
                .desc = "Enable 3DNow! Athlon instructions",
                .deps = &.{"3dnow"},
            },
            .{
                .zig_name = "avx512er",
                .desc = "Enable AVX-512 Exponential and Reciprocal Instructions",
                .deps = &.{"avx512f"},
            },
            .{
                .zig_name = "avx512pf",
                .desc = "Enable AVX-512 PreFetch Instructions",
                .deps = &.{"avx512f"},
            },
            .{
                .zig_name = "prefetchwt1",
                .desc = "Prefetch with Intent to Write and T1 Hint",
                .deps = &.{},
            },

            // Custom Zig features
            .{
                .zig_name = "bsf_bsr_0_clobbers_result",
                .desc = "BSF/BSR may clobber the lower 32-bits of the result register when the source is zero",
                .deps = &.{},
            },
            .{
                .zig_name = "smap",
                .desc = "Enable Supervisor Mode Access Prevention",
                .deps = &.{},
            },
            .{
                .zig_name = "smep",
                .desc = "Enable Supervisor Mode Execution Prevention",
                .deps = &.{},
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
            // LLVM defines a bunch of dumb aliases with foreach loops in X86.td.
            .{
                .llvm_name = "pentium_mmx",
                .omit = true,
            },
            .{
                .llvm_name = "pentium_pro",
                .omit = true,
            },
            .{
                .llvm_name = "pentium_ii",
                .omit = true,
            },
            .{
                .llvm_name = "pentium_3m",
                .omit = true,
            },
            .{
                .llvm_name = "pentium_iii",
                .omit = true,
            },
            .{
                .llvm_name = "pentium_iii_no_xmm_regs",
                .omit = true,
            },
            .{
                .llvm_name = "pentium_m",
                .omit = true,
            },
            .{
                .llvm_name = "pentium4m",
                .omit = true,
            },
            .{
                .llvm_name = "pentium_4",
                .omit = true,
            },
            .{
                .llvm_name = "pentium_4_sse3",
                .omit = true,
            },
            .{
                .llvm_name = "core_2_duo_ssse3",
                .omit = true,
            },
            .{
                .llvm_name = "core_2_duo_sse4_1",
                .omit = true,
            },
            .{
                .llvm_name = "atom_sse4_2",
                .omit = true,
            },
            .{
                .llvm_name = "goldmont-plus",
                .omit = true,
            },
            .{
                .llvm_name = "core_i7_sse4_2",
                .omit = true,
            },
            .{
                .llvm_name = "core_aes_pclmulqdq",
                .omit = true,
            },
            .{
                .llvm_name = "corei7-avx",
                .omit = true,
            },
            .{
                .llvm_name = "core_2nd_gen_avx",
                .omit = true,
            },
            .{
                .llvm_name = "core-avx-i",
                .omit = true,
            },
            .{
                .llvm_name = "core_3rd_gen_avx",
                .omit = true,
            },
            .{
                .llvm_name = "core-avx2",
                .omit = true,
            },
            .{
                .llvm_name = "core_4th_gen_avx",
                .omit = true,
            },
            .{
                .llvm_name = "core_4th_gen_avx_tsx",
                .omit = true,
            },
            .{
                .llvm_name = "core_5th_gen_avx",
                .omit = true,
            },
            .{
                .llvm_name = "core_5th_gen_avx_tsx",
                .omit = true,
            },
            .{
                .llvm_name = "mic_avx512",
                .omit = true,
            },
            .{
                .llvm_name = "skylake-avx512",
                .omit = true,
            },
            .{
                .llvm_name = "icelake-client",
                .omit = true,
            },
            .{
                .llvm_name = "icelake-server",
                .omit = true,
            },
            .{
                .llvm_name = "graniterapids_d",
                .omit = true,
            },
            .{
                .llvm_name = "arrowlake_s",
                .omit = true,
            },

            .{
                .llvm_name = "amdfam10",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "athlon",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "athlon64",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "athlon64-sse3",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "athlon-4",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "athlon-fx",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "athlon-mp",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "athlon-tbird",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "athlon-xp",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "barcelona",
                .extra_deps = &.{ "3dnowa", "smap", "smep" },
            },
            .{
                .llvm_name = "broadwell",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "c3",
                .extra_deps = &.{"3dnow"},
            },
            .{
                .llvm_name = "cannonlake",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "cascadelake",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "emeraldrapids",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "geode",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "goldmont",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "goldmont_plus",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "haswell",
                .extra_deps = &.{"smep"},
            },
            .{
                .llvm_name = "i386",
                .extra_deps = &.{"bsf_bsr_0_clobbers_result"},
            },
            .{
                .llvm_name = "i486",
                .extra_deps = &.{"bsf_bsr_0_clobbers_result"},
            },
            .{
                .llvm_name = "icelake_client",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "icelake_server",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "ivybridge",
                .extra_deps = &.{"smep"},
            },
            .{
                .llvm_name = "k6-2",
                .extra_deps = &.{"3dnow"},
            },
            .{
                .llvm_name = "k6-3",
                .extra_deps = &.{"3dnow"},
            },
            .{
                .llvm_name = "k8",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "k8-sse3",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "knl",
                .extra_deps = &.{
                    "avx512er",
                    "avx512pf",
                    "prefetchwt1",
                },
            },
            .{
                .llvm_name = "knm",
                .extra_deps = &.{
                    "avx512er",
                    "avx512pf",
                    "prefetchwt1",
                },
            },
            .{
                .llvm_name = "meteorlake",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "opteron",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "opteron-sse3",
                .extra_deps = &.{"3dnowa"},
            },
            .{
                .llvm_name = "raptorlake",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "rocketlake",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "sapphirerapids",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "silvermont",
                .extra_deps = &.{"smep"},
            },
            .{
                .llvm_name = "skx",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "skylake",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "skylake_avx512",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "tigerlake",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "winchip2",
                .extra_deps = &.{"3dnow"},
            },
            .{
                .llvm_name = "alderlake",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "arrowlake",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "arrowlake-s",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "znver1",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "znver2",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "znver3",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "znver4",
                .extra_deps = &.{ "smap", "smep" },
            },
            .{
                .llvm_name = "znver5",
                .extra_deps = &.{ "smap", "smep" },
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "i86",
                .features = &.{"16bit_mode"},
            },
        },
    },
    .{
        .zig_name = "xcore",
        .llvm = .{
            .name = "XCore",
            .td_name = "XCore",
        },
        .extra_features = &.{
            .{
                .zig_name = "xs2",
                .desc = "Enable XS2 architecture",
                .deps = &.{},
            },
            .{
                .zig_name = "xs3",
                .desc = "Enable XS3 architecture",
                .deps = &.{"xs2"},
            },
            .{
                .zig_name = "fpu",
                .desc = "Enable single-precision floating point unit",
                .deps = &.{"xs3"},
            },
            .{
                .zig_name = "vpu",
                .desc = "Enable vector processing unit",
                .deps = &.{"xs3"},
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .omit = true,
            },
            .{
                .llvm_name = "xs1b-generic",
                .zig_name = "xs1b",
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "xs2a",
                .features = &.{"xs2"},
            },
            .{
                .llvm_name = null,
                .zig_name = "xs3a",
                .features = &.{ "fpu", "vpu" },
            },
        },
    },
    .{
        .zig_name = "xtensa",
        .llvm = .{
            .name = "Xtensa",
            .td_name = "Xtensa",
        },
        .feature_overrides = &.{
            .{
                .llvm_name = "forced-atomics",
                .omit = true,
            },
        },
        .model_overrides = &.{
            .{
                .llvm_name = "generic",
                .zig_name = "xtensa",
            },
        },
    },
    .{
        .zig_name = "z80",
        .llvm = null,
        .extra_features = &.{
            .{
                .zig_name = "epu",
                .desc = "Enable Extended Processing Unit instructions",
                .deps = &.{"z280"},
            },
            .{
                .zig_name = "ez80",
                .desc = "Enable eZ80 instructions including ADL mode",
                .deps = &.{ "xyhl", "z180" },
            },
            .{
                .zig_name = "infc",
                .desc = "Enable flags-only port input instruction",
                .deps = &.{},
            },
            .{
                .zig_name = "no_inirx",
                .desc = "Disable block I/O instructions that address the port with DE and count with BC",
                .deps = &.{},
            },
            .{
                .zig_name = "no_ld_i_hl",
                .desc = "Disable 16-bit I register transfers",
                .deps = &.{},
            },
            .{
                .zig_name = "r800",
                .desc = "Enable R800 instructions",
                .deps = &.{ "infc", "xyhl" },
            },
            .{
                .zig_name = "reliable_ld_a_i",
                .desc = "LD A,I and LD A,R reliably report the interrupt enable state",
                .deps = &.{},
            },
            .{
                .zig_name = "sli",
                .desc = "Enable shift left inserting 1 instructions",
                .deps = &.{},
            },
            .{
                .zig_name = "xdcb",
                .desc = "Enable indexed rotate, shift, RES, and SET instructions that also store to a register",
                .deps = &.{},
            },
            .{
                .zig_name = "xyhl",
                .desc = "Enable index register halves (IXH, IXL, IYH, IYL)",
                .deps = &.{},
            },
            .{
                .zig_name = "z180",
                .desc = "Enable Z180 instructions",
                .deps = &.{},
            },
            .{
                .zig_name = "z280",
                .desc = "Enable Z280 instructions",
                .deps = &.{ "infc", "xyhl" },
            },
            .{
                .zig_name = "z380",
                .desc = "Enable Z380 instructions including long mode",
                .deps = &.{ "xyhl", "z180" },
            },
            .{
                .zig_name = "z80n",
                .desc = "Enable Z80N instructions",
                .deps = &.{},
            },
        },
        .extra_models = &.{
            .{
                .llvm_name = null,
                .zig_name = "ez80",
                .features = &.{
                    "ez80",
                    "reliable_ld_a_i",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "ez80190",
                .features = &.{
                    "ez80",
                    "no_inirx",
                    "no_ld_i_hl",
                    "reliable_ld_a_i",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "ez80f92",
                .features = &.{
                    "ez80",
                    "no_ld_i_hl",
                    "reliable_ld_a_i",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "r800",
                .features = &.{
                    "r800",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "z180",
                .features = &.{
                    "z180",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "z280",
                .features = &.{
                    "reliable_ld_a_i",
                    "z280",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "z380",
                .features = &.{
                    "reliable_ld_a_i",
                    "z380",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "z80",
                .features = &.{},
            },
            .{
                .llvm_name = null,
                .zig_name = "z8400",
                .features = &.{
                    "infc",
                    "sli",
                    "xdcb",
                    "xyhl",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "z84c00",
                .features = &.{
                    "infc",
                    "reliable_ld_a_i",
                    "sli",
                    "xdcb",
                    "xyhl",
                },
            },
            .{
                .llvm_name = null,
                .zig_name = "z80n",
                .features = &.{
                    "infc",
                    "reliable_ld_a_i",
                    "sli",
                    "xdcb",
                    "xyhl",
                    "z80n",
                },
            },
        },
    },
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
            try std.fmt.allocPrint(arena, "{s}/llvm/lib/Target/{s}/{s}.td", .{
                job.llvm_src_root,
                llvm.name,
                llvm.td_name,
            }),
            try std.fmt.allocPrint(arena, "-I={s}/llvm/include", .{job.llvm_src_root}),
            try std.fmt.allocPrint(arena, "-I={s}/llvm/lib/Target/{s}", .{
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

        // So far, LLVM only has a few aliases for the same CPU.
        const Alias = struct {
            llvm: []const u8,
            zig: []const u8,
        };
        var model_aliases = std.StringHashMap(std.ArrayList(*Alias)).init(arena);

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

                    const gop = try model_aliases.getOrPut(try llvmNameToZigName(arena, llvm_name));

                    if (!gop.found_existing) {
                        gop.value_ptr.* = .empty;
                    }

                    const alias = try arena.create(Alias);
                    alias.* = .{
                        .llvm = llvm_alias,
                        .zig = try llvmNameToZigName(arena, llvm_alias),
                    };
                    try gop.value_ptr.append(arena, alias);
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
                    // This is used by AArch64.
                    if (kv.value_ptr.object.get("DefaultExts")) |exts_val| {
                        for (exts_val.array.items) |ext| {
                            const other_key = ext.object.get("def").?.string;
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

                    var zig_name = try llvmNameToZigName(arena, llvm_name);
                    var deps: std.ArrayList([]const u8) = .empty;
                    var omit_deps: []const []const u8 = &.{};
                    var extra_deps: []const []const u8 = &.{};
                    const omitted = for (target.model_overrides) |model_override| {
                        if (mem.eql(u8, llvm_name, model_override.llvm_name)) {
                            if (model_override.omit) {
                                break true;
                            }
                            if (model_override.zig_name) |override_name| {
                                zig_name = override_name;
                            }
                            omit_deps = model_override.omit_deps;
                            extra_deps = model_override.extra_deps;
                            break false;
                        }
                    } else false;
                    if (omitted) continue;
                    const features = kv.value_ptr.object.get("Features").?.array;
                    for (features.items) |feature| {
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
                    const tune_features = kv.value_ptr.object.get("TuneFeatures").?.array;
                    for (tune_features.items) |feature| {
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

                    if (model_aliases.get(zig_name)) |aliases| {
                        alias_it: for (aliases.items) |alias| {
                            for (target.model_overrides) |model_override| {
                                if (mem.eql(u8, alias.llvm, model_override.llvm_name) and model_override.omit) continue :alias_it;
                            }

                            try all_models.append(arena, .{
                                .llvm_name = alias.llvm,
                                .zig_name = alias.zig,
                                .features = deps.items,
                            });
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

    const zig_code_basename = try std.fmt.allocPrint(arena, "{s}.zig", .{target.zig_name});
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
        try deletion_set.put(dep, {});
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
