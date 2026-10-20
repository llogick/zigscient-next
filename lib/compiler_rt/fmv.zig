const builtin = @import("builtin");
const std = @import("std");

const compiler_rt = @import("../compiler_rt.zig");
const symbol = compiler_rt.symbol;

comptime {
    // This is just a conservative stub for now, to make sure that code that
    // uses function multiversioning will link correctly, albeit with the
    // slowest variant of a function.

    if (builtin.target.cpu.arch.isAarch64()) _ = aarch64;
    if (builtin.target.cpu.arch.isLoongarch()) _ = loongarch;
    if (builtin.target.cpu.arch.isRiscv()) _ = riscv;
    if (builtin.target.cpu.arch.isX86()) _ = x86;
}

const aarch64 = struct {
    comptime {
        symbol(&__aarch64_cpu_features, "__aarch64_cpu_features");
        symbol(&__init_cpu_features_resolver, "__init_cpu_features_resolver");
    }

    var __aarch64_cpu_features: extern struct {
        features: c_ulonglong,
    } = .{
        .features = 0,
    };

    const __init_cpu_features_resolver = switch (builtin.target.os.tag) {
        .fuchsia,
        .driverkit,
        .ios,
        .maccatalyst,
        .macos,
        .tvos,
        .visionos,
        .watchos,
        => struct {
            fn __init_cpu_features_resolver() callconv(.c) void {}
        },
        else => struct {
            fn __init_cpu_features_resolver(hwcap: c_ulong, arg: *const extern struct {
                _size: c_ulong,
                _hwcap: c_ulong,
                _hwcap2: c_ulong,
                _hwcap3: c_ulong,
                _hwcap4: c_ulong,
            }) callconv(.c) void {
                _ = hwcap;
                _ = arg;
            }
        },
    }.__init_cpu_features_resolver;
};

const loongarch = struct {
    comptime {
        symbol(&__loongarch_feature_bits, "__loongarch_feature_bits");
        symbol(&__init_loongarch_features_resolver, "__init_loongarch_features_resolver");
    }

    var __loongarch_feature_bits: extern struct {
        features: c_ulonglong,
    } = .{
        .features = 0,
    };

    var __riscv_cpu_model: extern struct {
        mvendorid: c_uint,
        marchid: c_ulonglong,
        mimpid: c_ulonglong,
    } = .{
        .mvendorid = 0,
        .marchid = 0,
        .mimpid = 0,
    };

    fn __init_loongarch_features_resolver() callconv(.c) void {}
};

const riscv = struct {
    comptime {
        symbol(&__riscv_feature_bits, "__riscv_feature_bits");
        symbol(&__riscv_cpu_model, "__riscv_cpu_model");
        symbol(&__init_riscv_feature_bits, "__init_riscv_feature_bits");
    }

    var __riscv_feature_bits: extern struct {
        length: c_uint,
        features: [2]c_ulonglong,
    } = .{
        .length = 2,
        .features = @splat(0),
    };

    var __riscv_cpu_model: extern struct {
        mvendorid: c_uint,
        marchid: c_ulonglong,
        mimpid: c_ulonglong,
    } = .{
        .mvendorid = 0,
        .marchid = 0,
        .mimpid = 0,
    };

    fn __init_riscv_feature_bits(args: ?*anyopaque) callconv(.c) void {
        _ = args;
    }
};

const x86 = struct {
    comptime {
        symbol(&__cpu_model, "__cpu_model");
        symbol(&__cpu_features2, "__cpu_features2");
        symbol(&__cpu_indicator_init, "__cpu_indicator_init");
    }

    var __cpu_model: extern struct {
        __cpu_vendor: c_uint,
        __cpu_type: c_uint,
        __cpu_subtype: c_uint,
        __cpu_features: [1]c_uint,
    } = .{
        .__cpu_vendor = 0,
        .__cpu_type = 0,
        .__cpu_subtype = 0,
        .__cpu_features = @splat(0),
    };

    var __cpu_features2: [4]c_uint = @splat(0);

    fn __cpu_indicator_init() callconv(.c) c_int {
        return 0;
    }
};
