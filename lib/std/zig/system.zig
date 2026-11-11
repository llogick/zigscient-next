const builtin = @import("builtin");
const native_endian = builtin.cpu.arch.endian();

const std = @import("../std.zig");
const mem = std.mem;
const elf = std.elf;
const fs = std.fs;
const assert = std.debug.assert;
const Target = std.Target;
const posix = std.posix;
const Io = std.Io;

pub const NativePaths = @import("system/NativePaths.zig");

pub const darwin = @import("system/darwin.zig");
pub const freebsd = @import("system/freebsd.zig");
pub const linux = @import("system/linux.zig");
pub const netbsd = @import("system/netbsd.zig");
pub const openbsd = @import("system/openbsd.zig");
pub const windows = @import("system/windows.zig");

pub const Executor = union(enum) {
    native,
    rosetta,
    qemu: []const u8,
    wine: []const u8,
    wasmtime: []const u8,
    darling: []const u8,
    bad_dl: []const u8,
    bad_os_or_cpu,
};

pub const GetExternalExecutorOptions = struct {
    host_cpu_arch: std.Target.Cpu.Arch,
    host_os_tag: std.Target.Os.Tag,
    qemu_fixes_dl: bool = false,
    link_mode: std.lang.LinkMode,
    link_libc: bool,
};

/// Return whether or not the given host is capable of running executables of
/// the other target.
pub fn getExternalExecutor(io: Io, candidate: *const std.Target, options: GetExternalExecutorOptions) Executor {
    const host_os_tag = options.host_os_tag;
    const host_cpu_arch = options.host_cpu_arch;
    const os_match = host_os_tag == candidate.os.tag or (host_os_tag == .macos and candidate.os.tag == .maccatalyst);
    const cpu_ok = cpu_ok: {
        if (host_cpu_arch == candidate.cpu.arch)
            break :cpu_ok true;

        if (host_cpu_arch == .x86_64 and candidate.cpu.arch == .x86)
            break :cpu_ok true;

        if (host_cpu_arch == .aarch64 and candidate.cpu.arch == .arm)
            break :cpu_ok true;

        if (host_cpu_arch == .aarch64_be and candidate.cpu.arch == .armeb)
            break :cpu_ok true;

        // TODO additionally detect incompatible CPU features.
        // Note that in some cases the OS kernel will emulate missing CPU features
        // when an illegal instruction is encountered.

        break :cpu_ok false;
    };

    var bad_result: Executor = .bad_os_or_cpu;

    if (os_match and cpu_ok) native: {
        if (options.link_libc and options.link_mode == .dynamic) {
            if (candidate.dynamic_linker.get()) |candidate_dl| {
                Io.Dir.cwd().access(io, candidate_dl, .{}) catch {
                    bad_result = .{ .bad_dl = candidate_dl };
                    break :native;
                };
            }
        }
        return .native;
    }

    // If the OS match and OS is macOS and CPU is arm64, we can use Rosetta 2
    // to emulate the foreign architecture.
    if (os_match and host_cpu_arch == .aarch64) {
        switch (candidate.cpu.arch) {
            .x86_64 => return .rosetta,
            else => return bad_result,
        }
    }

    // If the OS matches, we can use QEMU to emulate a foreign architecture.
    if (os_match and (!cpu_ok or options.qemu_fixes_dl)) {
        return switch (candidate.cpu.arch) {
            inline .aarch64,
            .arm,
            .riscv64,
            .x86,
            .x86_64,
            => |t| switch (candidate.os.tag) {
                .linux,
                .freebsd,
                => .{ .qemu = switch (t) {
                    .x86 => "qemu-i386",
                    .x86_64 => if (candidate.abi.isX32())
                        return bad_result
                    else
                        "qemu-x86_64",
                    else => "qemu-" ++ @tagName(t),
                } },
                else => bad_result,
            },
            inline .aarch64_be,
            .alpha,
            .armeb,
            .hexagon,
            .hppa,
            .loongarch64,
            .m68k,
            .microblaze,
            .microblazeel,
            .mips,
            .mipsel,
            .mipsf32,
            .mipsf32el,
            .mipsf64,
            .mipsf64el,
            .mipsf64n8,
            .mipsf64n8el,
            .mipsn8,
            .mipsn8el,
            .mipsr6,
            .mipsr6el,
            .mips64,
            .mips64el,
            .mips64n8,
            .mips64n8el,
            .mips64r6,
            .mips64r6el,
            .or1k,
            .powerpc,
            .powerpc64,
            .powerpc64le,
            .riscv32,
            .s390x,
            .sh,
            .sheb,
            .sparc,
            .sparc64,
            .thumb,
            .thumbeb,
            .xtensa,
            .xtensaeb,
            => |t| switch (candidate.os.tag) {
                .linux,
                => .{
                    .qemu = switch (t) {
                        .powerpc => "qemu-ppc",
                        .powerpc64 => "qemu-ppc64",
                        .powerpc64le => "qemu-ppc64le",
                        .mips,
                        .mipsel,
                        .mipsf32,
                        .mipsf32el,
                        .mipsf64,
                        .mipsf64el,
                        .mipsf64n8,
                        .mipsf64n8el,
                        .mipsn8,
                        .mipsn8el,
                        .mipsr6,
                        .mipsr6el,
                        => if (candidate.cpu.arch.endian() == .little) "qemu-mipsel" else "qemu-mips",
                        .mips64,
                        .mips64n8,
                        .mips64r6,
                        => if (candidate.abi.isN32()) "qemu-mipsn32" else "qemu-mips64",
                        .mips64el,
                        .mips64n8el,
                        .mips64r6el,
                        => if (candidate.abi.isN32()) "qemu-mipsn32el" else "qemu-mips64el",
                        .sh => "qemu-sh4",
                        .sheb => "qemu-sh4eb",
                        .sparc => "qemu-sparc32plus",
                        .thumb => "qemu-arm",
                        .thumbeb => "qemu-armeb",
                        else => "qemu-" ++ @tagName(t),
                    },
                },
                else => bad_result,
            },
            else => bad_result,
        };
    }

    if (candidate.cpu.arch.isWasm()) {
        return .{ .wasmtime = "wasmtime" };
    }

    switch (candidate.os.tag) {
        .windows => {
            const wine_supported = switch (candidate.cpu.arch) {
                .thumb => switch (host_cpu_arch) {
                    .arm, .thumb, .aarch64 => true,
                    else => false,
                },
                .aarch64 => host_cpu_arch == .aarch64,
                .x86 => host_cpu_arch.isX86(),
                .x86_64 => host_cpu_arch == .x86_64,
                else => false,
            };
            return if (wine_supported) .{ .wine = "wine" } else bad_result;
        },
        .driverkit, .macos => {
            // This check can be loosened once darling adds a QEMU-based emulation
            // layer for non-host architectures:
            // https://github.com/darlinghq/darling/issues/863
            if (candidate.cpu.arch != host_cpu_arch) {
                return bad_result;
            }
            return .{ .darling = "darling" };
        },
        else => return bad_result,
    }
}

pub const ResolveTargetError = error{
    FileSystem,
    SystemResources,
    SymLinkLoop,
    ProcessFdQuotaExceeded,
    SystemFdQuotaExceeded,
    DeviceBusy,
    OsVersionQueryFailed,
    /// Android-only. Querying API level through `getprop` failed.
    ApiLevelQueryFailed,
} || Io.Cancelable || Io.UnexpectedError;

/// Given a `Target.Query`, which specifies in detail which parts of the
/// target should be detected natively, which should be standard or default,
/// and which are provided explicitly, this function resolves the native
/// components by detecting the native system, and then resolves
/// standard/default parts relative to that.
pub fn resolveTargetQuery(io: Io, query: Target.Query) ResolveTargetError!Target {
    // Until https://github.com/ziglang/zig/issues/4592 is implemented (support detecting the
    // native CPU architecture as being different than the current target), we use this:
    const query_cpu_arch = query.cpu_arch orelse builtin.cpu.arch;
    const query_os_tag = query.os_tag orelse builtin.os.tag;
    const query_abi = query.abi orelse builtin.abi;
    var os = query_os_tag.defaultVersionRange(query_cpu_arch, query_abi);
    if (query.os_tag == null) {
        switch (builtin.target.os.tag) {
            .linux, .illumos => {
                const uts = posix.uname();
                const release = mem.sliceTo(&uts.release, 0);
                // The release field sometimes has a weird format,
                // `Version.parse` will attempt to find some meaningful interpretation.
                if (std.SemanticVersion.parse(release)) |ver| {
                    var stripped = ver;
                    stripped.pre = null;
                    stripped.build = null;
                    os.version_range.linux.range.min = stripped;
                    os.version_range.linux.range.max = stripped;
                } else |err| switch (err) {
                    error.Overflow => {},
                    error.InvalidVersion => {},
                }
            },
            .windows => {
                const detected_version = windows.detectRuntimeVersion();
                os.version_range.windows.min = detected_version;
                os.version_range.windows.max = detected_version;
            },
            .macos => try darwin.macos.detect(io, &os),
            .freebsd, .netbsd, .dragonfly => {
                const key = switch (builtin.target.os.tag) {
                    .freebsd => "kern.osreldate",
                    .netbsd, .dragonfly => "kern.osrevision",
                    else => unreachable,
                };
                var value: u32 = undefined;
                var len: usize = @sizeOf(@TypeOf(value));

                switch (posix.errno(posix.system.sysctlbyname(key, &value, &len, null, 0))) {
                    .SUCCESS => {},
                    .FAULT => unreachable,
                    .PERM => unreachable, // only when setting values,
                    .NOMEM => unreachable, // memory already on the stack
                    .NOENT => unreachable, // constant, known good value
                    else => return error.OsVersionQueryFailed,
                }

                switch (builtin.target.os.tag) {
                    .freebsd => {
                        // https://www.freebsd.org/doc/en_US.ISO8859-1/books/porters-handbook/versions.html
                        // Major * 100,000 has been convention since FreeBSD 2.2 (1997)
                        // Minor * 1(0),000 summed has been convention since FreeBSD 2.2 (1997)
                        // e.g. 492101 = 4.11-STABLE = 4.(9+2)
                        const major = value / 100_000;
                        const minor1 = (value % 100_000) / 10_000; // usually 0 since 5.1
                        const minor2 = (value % 10_000) / 1_000; // 0 before 5.1, minor version since
                        const patch = value % 1_000;
                        os.version_range.semver.min = .{ .major = major, .minor = minor1 + minor2, .patch = patch };
                        os.version_range.semver.max = os.version_range.semver.min;
                    },
                    .netbsd => {
                        // #define __NetBSD_Version__ MMmmrrpp00
                        //
                        // M = major version
                        // m = minor version; a minor number of 99 indicates current.
                        // r = 0 (*)
                        // p = patchlevel
                        const major = value / 100_000_000;
                        const minor = (value % 100_000_000) / 1_000_000;
                        const patch = (value % 10_000) / 100;
                        os.version_range.semver.min = .{ .major = major, .minor = minor, .patch = patch };
                        os.version_range.semver.max = os.version_range.semver.min;
                    },
                    .dragonfly => {
                        // https://github.com/DragonFlyBSD/DragonFlyBSD/blob/cb2cde83771754aeef9bb3251ee48959138dec87/Makefile.inc1#L15-L17
                        // flat base10 format: Mmmmpp
                        //   M = major
                        //   m = minor; odd-numbers indicate current dev branch
                        //   p = patch
                        const major = value / 100_000;
                        const minor = (value % 100_000) / 100;
                        const patch = value % 100;
                        os.version_range.semver.min = .{ .major = major, .minor = minor, .patch = patch };
                        os.version_range.semver.max = os.version_range.semver.min;
                    },
                    else => unreachable,
                }
            },
            .openbsd => {
                const mib: [2]c_int = [_]c_int{
                    posix.CTL.KERN,
                    posix.KERN.OSRELEASE,
                };
                var buf: [64:0]u8 = undefined;
                // consider that sysctl result includes null-termination
                var len: usize = buf.len + 1;

                posix.sysctl(&mib, &buf, &len, null, 0) catch |err| switch (err) {
                    error.NameTooLong => unreachable, // constant, known good value
                    error.PermissionDenied => unreachable, // only when setting values,
                    error.SystemResources => unreachable, // memory already on the stack
                    error.UnknownName => unreachable, // constant, known good value
                    error.Unexpected => return error.OsVersionQueryFailed,
                };

                if (Target.Query.parseVersion(buf[0 .. len - 1 :0])) |ver| {
                    assert(ver.build == null);
                    assert(ver.pre == null);
                    os.version_range.semver.min = ver;
                    os.version_range.semver.max = ver;
                } else |_| {
                    return error.OsVersionQueryFailed;
                }
            },
            else => {
                // Unimplemented, fall back to default version range.
            },
        }
    }

    if (query.os_version_min) |min| switch (min) {
        .none => {},
        .semver => |semver| switch (os.tag.versionRangeTag()) {
            inline .hurd, .linux => |t| @field(os.version_range, @tagName(t)).range.min = semver,
            else => os.version_range.semver.min = semver,
        },
        .windows => |win_ver| os.version_range.windows.min = win_ver,
    };

    if (query.os_version_max) |max| switch (max) {
        .none => {},
        .semver => |semver| switch (os.tag.versionRangeTag()) {
            inline .hurd, .linux => |t| @field(os.version_range, @tagName(t)).range.max = semver,
            else => os.version_range.semver.max = semver,
        },
        .windows => |win_ver| os.version_range.windows.max = win_ver,
    };

    if (query.glibc_version) |glibc| {
        switch (os.tag.versionRangeTag()) {
            inline .hurd, .linux => |t| @field(os.version_range, @tagName(t)).glibc = glibc,
            else => {},
        }
    }

    if (query.android_api_level) |android| {
        os.version_range.linux.android = android;
    }

    const maybe_cpu: ?Target.Cpu = switch (query.cpu_model) {
        .native => detectNativeCpuAndFeatures(io, query_cpu_arch),
        .baseline => null,
        .determined_by_arch_os => if (query.cpu_arch == null)
            detectNativeCpuAndFeatures(io, query_cpu_arch)
        else
            null,
        .explicit => |model| model.toCpu(query_cpu_arch),
    };

    const arch = if (maybe_cpu) |c| c.arch else query_cpu_arch;

    var detected = detectAbiAndDynamicLinker(io, arch, os, query) catch |err| switch (err) {
        error.Canceled => |e| return e,
        error.Unexpected => |e| return e,
        error.ApiLevelQueryFailed => |e| return e,
        error.WouldBlock => return error.Unexpected,
        error.ConnectionResetByPeer => return error.Unexpected,
        error.NotOpenForReading => return error.Unexpected,
        error.SocketUnconnected => return error.Unexpected,
        error.ReadOnlyFileSystem => return error.Unexpected,

        error.AccessDenied,
        error.SymLinkLoop,
        error.ProcessFdQuotaExceeded,
        error.SystemFdQuotaExceeded,
        error.SystemResources,
        error.IsDir,
        error.DeviceBusy,
        error.InputOutput,
        error.LockViolation,
        error.FileSystem,

        error.UnableToOpenElfFile,
        error.UnhelpfulFile,
        error.InvalidElfFile,
        error.RelativeShebang,
        => defaultAbiAndDynamicLinker(arch, os, query),
    };

    // It's possible that we detect the native ABI, but fail to detect the OS version or were told
    // to use the default OS version range. In that case, while we can't determine the exact native
    // OS version, we do at least know that some ABIs require a particular OS version (by way of
    // `std.zig.target.available_libcs`). So in this case, adjust the OS version to the minimum that
    // we know is required.
    if (detected.abi != query_abi and query.os_version_min == null) {
        const result_ver_range = &detected.os.version_range;
        const abi_ver_range = detected.os.tag.defaultVersionRange(arch, detected.abi).version_range;

        switch (detected.os.tag.versionRangeTag()) {
            .none => {},
            .semver => if (result_ver_range.semver.min.order(abi_ver_range.semver.min) == .lt) {
                result_ver_range.semver.min = abi_ver_range.semver.min;
            },
            inline .hurd, .linux => |t| {
                if (@field(result_ver_range, @tagName(t)).range.min.order(@field(abi_ver_range, @tagName(t)).range.min) == .lt) {
                    @field(result_ver_range, @tagName(t)).range.min = @field(abi_ver_range, @tagName(t)).range.min;
                }

                if (@field(result_ver_range, @tagName(t)).glibc.order(@field(abi_ver_range, @tagName(t)).glibc) == .lt and
                    query.glibc_version == null)
                {
                    @field(result_ver_range, @tagName(t)).glibc = @field(abi_ver_range, @tagName(t)).glibc;
                }
            },
            .windows => if (!result_ver_range.windows.min.isAtLeast(abi_ver_range.windows.min)) {
                result_ver_range.windows.min = abi_ver_range.windows.min;
            },
        }
    }

    var cpu: Target.Cpu = maybe_cpu orelse .baseline(arch, detected.os, detected.abi);

    resolveCpuFeatures(
        arch,
        &cpu.features,
        abiRemovedFeatures(arch, detected.abi),
        query.cpu_features_add,
        query.cpu_features_sub,
    );

    return .{
        .cpu = cpu,
        .os = detected.os,
        .abi = detected.abi,
        .ofmt = query.ofmt orelse .default(detected.os.tag, arch),
        .dynamic_linker = detected.dynamic_linker,
    };
}

fn resolveCpuFeatures(
    arch: Target.Cpu.Arch,
    set: *Target.Cpu.Feature.Set,
    abi_sub_set: Target.Cpu.Feature.Set,
    add_set: Target.Cpu.Feature.Set,
    sub_set: Target.Cpu.Feature.Set,
) void {
    const all = arch.allFeaturesList();

    var abi_sub = abi_sub_set;
    abi_sub.populateDependents(all);

    var sub = sub_set;
    sub.populateDependents(all);

    set.removeFeatureSet(abi_sub);
    set.removeFeatureSet(sub);
    set.addFeatureSet(add_set);
    set.populateDependencies(all);
    set.removeFeatureSet(sub);

    const blocked = sub;

    // This loop isn't necessary if derivation rules are ordered perfectly, but
    // that's far too easy to mess up and cause subtle bugs as a result. Still,
    // please do try to keep the rules ordered sensibly so derivation completes
    // in one iteration.
    while (true) {
        const old = set.*;

        switch (arch.family()) {
            .aarch64 => {
                const when = Target.aarch64.featureSetHas;
                const whenAll = Target.aarch64.featureSetHasAll;

                derive(set, blocked, .aarch64, .ete, when(set.*, .trbe));
                derive(set, blocked, .aarch64, .neon, when(set.*, .fp_armv8));
                derive(set, blocked, .aarch64, .rdm, whenAll(set.*, .{ .neon, .v8_1a }));
                derive(set, blocked, .aarch64, .fullfp16, whenAll(set.*, .{ .fp16fml, .v8_2a }));
                derive(set, blocked, .aarch64, .complxnum, whenAll(set.*, .{ .fp_armv8, .v8_3a }));
                derive(set, blocked, .aarch64, .jsconv, whenAll(set.*, .{ .fp_armv8, .v8_3a }));
                derive(set, blocked, .aarch64, .dotprod, whenAll(set.*, .{ .neon, .v8_4a }));
                derive(set, blocked, .aarch64, .fullfp16, whenAll(set.*, .{ .neon, .v9a }));
                derive(set, blocked, .aarch64, .fp16fml, whenAll(set.*, .{ .fullfp16, .v8_4a }));
                derive(set, blocked, .aarch64, .fptoint, whenAll(set.*, .{ .fp_armv8, .v8_5a }));
                derive(set, blocked, .aarch64, .bf16, whenAll(set.*, .{ .fp_armv8, .v8_6a }));
                derive(set, blocked, .aarch64, .i8mm, whenAll(set.*, .{ .neon, .v8_6a }));
                derive(set, blocked, .aarch64, .spe_eef, whenAll(set.*, .{ .spe, .v8_7a }));
                derive(set, blocked, .aarch64, .ete, whenAll(set.*, .{ .tracev8_4, .v9a }));
                derive(set, blocked, .aarch64, .sve2p1, whenAll(set.*, .{ .sve2, .v9_4a }));
                derive(set, blocked, .aarch64, .sme2p1, whenAll(set.*, .{ .sme2, .v9_4a }));
                derive(set, blocked, .aarch64, .faminmax, whenAll(set.*, .{ .fp_armv8, .v9_5a }));
                derive(set, blocked, .aarch64, .lut, whenAll(set.*, .{ .neon, .v9_5a }));
                derive(set, blocked, .aarch64, .sve_bfscale, whenAll(set.*, .{ .sve_b16b16, .v9_6a }));
                derive(set, blocked, .aarch64, .fprcvt, whenAll(set.*, .{ .fp_armv8, .v9_7a }));
                derive(set, blocked, .aarch64, .f16f32dot, whenAll(set.*, .{ .fp16fml, .v9_7a }));
                derive(set, blocked, .aarch64, .sve2p3, whenAll(set.*, .{ .sve2, .v9_7a }));
                derive(set, blocked, .aarch64, .sme2p3, whenAll(set.*, .{ .sme, .v9_7a }));
                derive(set, blocked, .aarch64, .lscp, whenAll(set.*, .{ .d128, .v9_7a }));
                derive(set, blocked, .aarch64, .f16f32mm, whenAll(set.*, .{ .sve_f16f32mm, .v9_7a }));
                derive(set, blocked, .aarch64, .sme2p1, whenAll(set.*, .{ .sme, .sve2p1 }));
                derive(set, blocked, .aarch64, .sme2p2, whenAll(set.*, .{ .sme, .sve2p2 }));
                derive(set, blocked, .aarch64, .sme2p3, whenAll(set.*, .{ .sme, .sve2p3 }));
                derive(set, blocked, .aarch64, .sve2p1, whenAll(set.*, .{ .sme2p1, .sve2 }));
                derive(set, blocked, .aarch64, .sve2p2, whenAll(set.*, .{ .sme2p2, .sve2 }));
                derive(set, blocked, .aarch64, .sve2p3, whenAll(set.*, .{ .sme2p3, .sve2 }));
                derive(set, blocked, .aarch64, .ssve_fp8fma, whenAll(set.*, .{ .fp8fma, .sme2 }));
                derive(set, blocked, .aarch64, .ssve_fp8dot2, whenAll(set.*, .{ .fp8dot2, .sme2 }));
                derive(set, blocked, .aarch64, .ssve_fp8dot4, whenAll(set.*, .{ .fp8dot4, .sme2 }));
                derive(set, blocked, .aarch64, .sve_sha3, whenAll(set.*, .{ .sve, .sve_aes }));
                derive(set, blocked, .aarch64, .sve_aes, whenAll(set.*, .{ .sve, .sve_sha3 }));
                derive(set, blocked, .aarch64, .sha2, whenAll(set.*, .{ .aes, .rme }));
                derive(set, blocked, .aarch64, .sha3, whenAll(set.*, .{ .aes, .rme }));
                derive(set, blocked, .aarch64, .aes, whenAll(set.*, .{ .rme, .sha2 }));
                derive(set, blocked, .aarch64, .sha3, whenAll(set.*, .{ .rme, .sha2 }));
                derive(set, blocked, .aarch64, .aes, whenAll(set.*, .{ .rme, .sm4 }));
                derive(set, blocked, .aarch64, .sha2, whenAll(set.*, .{ .rme, .sm4 }));
                derive(set, blocked, .aarch64, .sha3, whenAll(set.*, .{ .rme, .sm4 }));
                derive(set, blocked, .aarch64, .sve_aes, whenAll(set.*, .{ .aes, .rme, .sve }));
                derive(set, blocked, .aarch64, .sve_aes, whenAll(set.*, .{ .rme, .sha2, .sve }));
                derive(set, blocked, .aarch64, .sve_sha3, whenAll(set.*, .{ .aes, .rme, .sve }));
                derive(set, blocked, .aarch64, .sve_sha3, whenAll(set.*, .{ .rme, .sha2, .sve }));
                derive(set, blocked, .aarch64, .spe_eef, whenAll(set.*, .{ .rme, .spe }));
                derive(set, blocked, .aarch64, .sve2_aes, whenAll(set.*, .{ .sve_aes, .sve2 }));
                derive(set, blocked, .aarch64, .sve2_bitperm, whenAll(set.*, .{ .sve_bitperm, .sve2 }));
                derive(set, blocked, .aarch64, .sve2_sha3, whenAll(set.*, .{ .sve_sha3, .sve2 }));
                derive(set, blocked, .aarch64, .sve2_sm4, whenAll(set.*, .{ .sve_sm4, .sve2 }));
                derive(set, blocked, .aarch64, .crypto, whenAll(set.*, .{ .aes, .sha2 }));
                derive(set, blocked, .aarch64, .dotprod, whenAll(set.*, .{ .neon, .rclass }));
                derive(set, blocked, .aarch64, .rdm, whenAll(set.*, .{ .neon, .rclass }));
                derive(set, blocked, .aarch64, .complxnum, whenAll(set.*, .{ .fp_armv8, .rclass }));
                derive(set, blocked, .aarch64, .jsconv, whenAll(set.*, .{ .fp_armv8, .rclass }));
                derive(set, blocked, .aarch64, .fullfp16, whenAll(set.*, .{ .fp16fml, .rclass }));
                derive(set, blocked, .aarch64, .fp16fml, whenAll(set.*, .{ .fullfp16, .rclass }));
            },
            .arm => {
                const when = Target.arm.featureSetHas;
                const whenAll = Target.arm.featureSetHasAll;

                derive(set, blocked, .arm, .hwdiv, whenAll(set.*, .{ .mclass, .v8m }));
                derive(set, blocked, .arm, .v7clrex, whenAll(set.*, .{ .mclass, .v8m }));
                derive(set, blocked, .arm, .acquire_release, whenAll(set.*, .{ .mclass, .v8m }) and !when(set.*, .v7));
                derive(set, blocked, .arm, .no_unaligned_access, when(set.*, .mclass) and !when(set.*, .v7));
                derive(set, blocked, .arm, .no_unaligned_access, !when(set.*, .v6));
                derive(set, blocked, .arm, .dsp, !when(set.*, .mclass) and when(set.*, .v5te));
                derive(set, blocked, .arm, .trustzone, whenAll(set.*, .{ .aclass, .virtualization }) and !when(set.*, .v8));
                derive(set, blocked, .arm, .virtualization, whenAll(set.*, .{ .rclass, .v8 }));
                derive(set, blocked, .arm, .dfb, whenAll(set.*, .{ .rclass, .v8 }));
                derive(set, blocked, .arm, .neon, whenAll(set.*, .{ .fp64, .rclass, .v8 }));
                derive(set, blocked, .arm, .neon, whenAll(set.*, .{ .aclass, .v8, .vfp2sp }));
                derive(set, blocked, .arm, .fp_armv8, whenAll(set.*, .{ .aclass, .v8, .vfp2sp }));
                derive(set, blocked, .arm, .fp_armv8, whenAll(set.*, .{ .neon, .v8 }));
                derive(set, blocked, .arm, .fp_armv8d16sp, whenAll(set.*, .{ .fp64, .mclass }));
                derive(set, blocked, .arm, .vfp4d16sp, whenAll(set.*, .{ .mclass, .vfp2sp }));
                derive(set, blocked, .arm, .fp_armv8d16sp, whenAll(set.*, .{ .v8m_main, .vfp2sp }));
                derive(set, blocked, .arm, .fullfp16, whenAll(set.*, .{ .v8_1m_main, .vfp2sp }));
                derive(set, blocked, .arm, .vfp3d16sp, whenAll(set.*, .{ .v7, .vfp2sp }));
                derive(set, blocked, .arm, .fp_armv8d16sp, whenAll(set.*, .{ .v8, .vfp2sp }));
                derive(set, blocked, .arm, .fullfp16, whenAll(set.*, .{ .fp16fml, .v8_2a }));
                derive(set, blocked, .arm, .fullfp16, whenAll(set.*, .{ .neon, .v9a }));
                derive(set, blocked, .arm, .fp16fml, whenAll(set.*, .{ .fullfp16, .v8_4a }));
                derive(set, blocked, .arm, .dotprod, whenAll(set.*, .{ .neon, .v8_4a }));
                derive(set, blocked, .arm, .vfp2, whenAll(set.*, .{ .fp64, .vfp2sp }));
                derive(set, blocked, .arm, .vfp3d16, whenAll(set.*, .{ .fp64, .vfp3d16sp }));
                derive(set, blocked, .arm, .vfp3sp, whenAll(set.*, .{ .d32, .vfp3d16sp }));
                derive(set, blocked, .arm, .vfp3, whenAll(set.*, .{ .vfp3d16, .vfp3sp }));
                derive(set, blocked, .arm, .vfp4d16, whenAll(set.*, .{ .fp64, .vfp4d16sp }));
                derive(set, blocked, .arm, .vfp4sp, whenAll(set.*, .{ .d32, .vfp4d16sp }));
                derive(set, blocked, .arm, .vfp4, whenAll(set.*, .{ .vfp4d16, .vfp4sp }));
                derive(set, blocked, .arm, .fp_armv8d16, whenAll(set.*, .{ .fp_armv8d16sp, .fp64 }));
                derive(set, blocked, .arm, .fp_armv8sp, whenAll(set.*, .{ .d32, .fp_armv8d16sp }));
                derive(set, blocked, .arm, .fp_armv8, whenAll(set.*, .{ .fp_armv8d16, .fp_armv8sp }));
                derive(set, blocked, .arm, .crypto, whenAll(set.*, .{ .aes, .sha2 }));
            },
            .hexagon => {
                const when = Target.hexagon.featureSetHas;
                const whenAll = Target.hexagon.featureSetHasAll;

                derive(set, blocked, .hexagon, .hvxv60, whenAll(set.*, .{ .hvx, .v60 }));
                derive(set, blocked, .hexagon, .hvxv62, whenAll(set.*, .{ .hvx, .v62 }));
                derive(set, blocked, .hexagon, .hvxv65, whenAll(set.*, .{ .hvx, .v65 }));
                derive(set, blocked, .hexagon, .hvxv66, whenAll(set.*, .{ .hvx, .v66 }));
                derive(set, blocked, .hexagon, .hvxv67, whenAll(set.*, .{ .hvx, .v67 }));
                derive(set, blocked, .hexagon, .hvxv68, whenAll(set.*, .{ .hvx, .v68 }));
                derive(set, blocked, .hexagon, .hvxv69, whenAll(set.*, .{ .hvx, .v69 }));
                derive(set, blocked, .hexagon, .hvxv71, whenAll(set.*, .{ .hvx, .v71 }));
                derive(set, blocked, .hexagon, .hvxv73, whenAll(set.*, .{ .hvx, .v73 }));
                derive(set, blocked, .hexagon, .hvxv75, whenAll(set.*, .{ .hvx, .v75 }));
                derive(set, blocked, .hexagon, .hvxv79, whenAll(set.*, .{ .hvx, .v79 }));
                derive(set, blocked, .hexagon, .hvxv81, whenAll(set.*, .{ .hvx, .v81 }));
                derive(set, blocked, .hexagon, .hvx_length64b, when(set.*, .hvx) and !when(set.*, .hvx_length128b));
            },
            .mips => {
                const when = Target.mips.featureSetHas;
                const whenAll = Target.mips.featureSetHasAll;

                derive(set, blocked, .mips, .fpu_d, whenAll(set.*, .{ .fpu_s, .mips32 }) and !when(set.*, .mips32r6));
                derive(set, blocked, .mips, .fpu_d, whenAll(set.*, .{ .fpu_s, .msa }));
                derive(set, blocked, .mips, .abs2008, whenAll(set.*, .{ .fpu_s, .msa }));
                derive(set, blocked, .mips, .nan2008, whenAll(set.*, .{ .fpu_s, .msa }));
                derive(set, blocked, .mips, .mips64r2, whenAll(set.*, .{ .mips32r2, .mips64 }));
                derive(set, blocked, .mips, .mips64r3, whenAll(set.*, .{ .mips32r3, .mips64 }));
                derive(set, blocked, .mips, .mips64r5, whenAll(set.*, .{ .mips32r5, .mips64 }));
                derive(set, blocked, .mips, .mips64r6, whenAll(set.*, .{ .mips32r6, .mips64 }));
                derive(set, blocked, .mips, .dspr3, whenAll(set.*, .{ .dspr2, .mips32r6 }));
            },
            .powerpc => {
                const when = Target.powerpc.featureSetHas;
                const whenAll = Target.powerpc.featureSetHasAll;

                derive(set, blocked, .powerpc, .fsqrt, whenAll(set.*, .{ .fpu, .isa_v203_instructions }));
                derive(set, blocked, .powerpc, .stfiwx, whenAll(set.*, .{ .fpu, .isa_v203_instructions }));
                derive(set, blocked, .powerpc, .fcpsgn, whenAll(set.*, .{ .fpu, .isa_v205_instructions }));
                derive(set, blocked, .powerpc, .lfiwax, whenAll(set.*, .{ .fpu, .isa_v205_instructions }));
                derive(set, blocked, .powerpc, .fre, whenAll(set.*, .{ .fpu, .isa_v206_instructions }));
                derive(set, blocked, .powerpc, .fres, whenAll(set.*, .{ .fpu, .isa_v206_instructions }));
                derive(set, blocked, .powerpc, .fprnd, whenAll(set.*, .{ .fpu, .isa_v206_instructions }));
                derive(set, blocked, .powerpc, .frsqrte, whenAll(set.*, .{ .fpu, .isa_v206_instructions }));
                derive(set, blocked, .powerpc, .frsqrtes, whenAll(set.*, .{ .fpu, .isa_v206_instructions }));
                derive(set, blocked, .powerpc, .fpcvt, whenAll(set.*, .{ .fpu, .isa_v30_instructions }));
                derive(set, blocked, .powerpc, .ldbrx, whenAll(set.*, .{ .@"64bit", .isa_v206_instructions }));
                derive(set, blocked, .powerpc, .bpermd, whenAll(set.*, .{ .@"64bit", .isa_v30_instructions }));
                derive(set, blocked, .powerpc, .extdiv, whenAll(set.*, .{ .@"64bit", .isa_v30_instructions }));
                derive(set, blocked, .powerpc, .popcntd, whenAll(set.*, .{ .@"64bit", .isa_v30_instructions }));
                derive(set, blocked, .powerpc, .quadword_atomics, whenAll(set.*, .{ .@"64bit", .isa_v30_instructions }));
                derive(set, blocked, .powerpc, .vsx, whenAll(set.*, .{ .altivec, .isa_v30_instructions }));
                derive(set, blocked, .powerpc, .power8_altivec, whenAll(set.*, .{ .altivec, .isa_v207_instructions }));
                derive(set, blocked, .powerpc, .direct_move, whenAll(set.*, .{ .isa_v207_instructions, .vsx }));
                derive(set, blocked, .powerpc, .power8_vector, whenAll(set.*, .{ .isa_v207_instructions, .vsx }));
                derive(set, blocked, .powerpc, .crypto, whenAll(set.*, .{ .altivec, .isa_v30_instructions }));
                derive(set, blocked, .powerpc, .power9_altivec, whenAll(set.*, .{ .altivec, .isa_v30_instructions }));
                derive(set, blocked, .powerpc, .paired_vector_memops, whenAll(set.*, .{ .altivec, .isa_v31_instructions }));
                derive(set, blocked, .powerpc, .prefix_instrs, whenAll(set.*, .{ .@"64bit", .isa_v31_instructions }));
                derive(set, blocked, .powerpc, .pcrelative_memops, when(set.*, .prefix_instrs));
            },
            .riscv => {
                const when = Target.riscv.featureSetHas;
                const whenAll = Target.riscv.featureSetHasAll;

                derive(set, blocked, .riscv, .zcd, whenAll(set.*, .{ .c, .d }));
                derive(set, blocked, .riscv, .zcf, whenAll(set.*, .{ .@"32bit", .c, .f }));
                derive(set, blocked, .riscv, .zcf, whenAll(set.*, .{ .@"32bit", .f, .zce }));
                derive(set, blocked, .riscv, .c, when(set.*, .zca) and (when(set.*, .zcf) or !whenAll(set.*, .{ .@"32bit", .f })) and (when(set.*, .zcd) or !when(set.*, .d)));
                derive(set, blocked, .riscv, .zce, whenAll(set.*, .{ .zcb, .zcmp, .zcmt }) and (when(set.*, .zcf) or !whenAll(set.*, .{ .@"32bit", .f })));
                derive(set, blocked, .riscv, .a, whenAll(set.*, .{ .zaamo, .zalrsc }));
                derive(set, blocked, .riscv, .b, whenAll(set.*, .{ .zba, .zbb, .zbs }));
                derive(set, blocked, .riscv, .zkn, whenAll(set.*, .{ .zbkb, .zbkc, .zbkx, .zknd, .zkne, .zknh }));
                derive(set, blocked, .riscv, .zks, whenAll(set.*, .{ .zbkb, .zbkc, .zbkx, .zksed, .zksh }));
                derive(set, blocked, .riscv, .zk, whenAll(set.*, .{ .zkn, .zkr, .zkt }));
                derive(set, blocked, .riscv, .zvkn, whenAll(set.*, .{ .zvkb, .zvkned, .zvknhb, .zvkt }));
                derive(set, blocked, .riscv, .zvknc, whenAll(set.*, .{ .zvbc, .zvkn }));
                derive(set, blocked, .riscv, .zvkng, whenAll(set.*, .{ .zvkg, .zvkn }));
                derive(set, blocked, .riscv, .zvks, whenAll(set.*, .{ .zvkb, .zvksed, .zvksh, .zvkt }));
                derive(set, blocked, .riscv, .zvksc, whenAll(set.*, .{ .zvbc, .zvks }));
                derive(set, blocked, .riscv, .zvksg, whenAll(set.*, .{ .zvkg, .zvks }));
                derive(set, blocked, .riscv, .zve64f, whenAll(set.*, .{ .zve32f, .zve64x }));
                derive(set, blocked, .riscv, .sha, whenAll(set.*, .{ .h, .shcounterenw, .shgatpa, .shtvala, .shvsatpa, .shvstvala, .shvstvecd, .ssstateen }));
                derive(set, blocked, .riscv, .smcdeleg, when(set.*, .ssccfg));
            },
            .s390x => {
                const whenAny = Target.s390x.featureSetHasAny;

                // A single `stfle` bit covers multiple features:
                const b45 = .{ .distops, .fbcrs, .hiword, .iacc, .loc, .popcnt };
                const b49 = .{ .exhint, .lat, .minste, .ppa };
                const b53 = .{ .loc2, .lzr };

                inline for (b45) |feat| derive(set, blocked, .s390x, feat, whenAny(set.*, b45));
                inline for (b49) |feat| derive(set, blocked, .s390x, feat, whenAny(set.*, b49));
                inline for (b53) |feat| derive(set, blocked, .s390x, feat, whenAny(set.*, b53));
            },
            .sh => {
                const whenAll = Target.sh.featureSetHasAll;

                derive(set, blocked, .sh, .fsca, whenAll(set.*, .{ .fpu, .sh4 }));
                derive(set, blocked, .sh, .fsrra, whenAll(set.*, .{ .fpu, .sh4 }));
            },
            .x86 => {
                const whenAll = Target.x86.featureSetHasAll;
                const whenAny = Target.x86.featureSetHasAny;

                // What is LLVM smoking? It's literally a single `cpuid` bit...
                const apxf = .{ .ccmp, .cf, .egpr, .jmpabs, .ndd, .nf, .ppx, .push2pop2, .zu };

                derive(set, blocked, .x86, .@"287", whenAll(set.*, .{ .@"286", .@"8087" }));
                derive(set, blocked, .x86, .@"387", whenAll(set.*, .{ .@"486", .@"8087" }));
                inline for (apxf) |feat| derive(set, blocked, .x86, feat, whenAny(set.*, apxf));
            },
            else => {},
        }

        if (set.eql(old)) break;
    }
}

fn derive(
    set: *Target.Cpu.Feature.Set,
    blocked: Target.Cpu.Feature.Set,
    comptime family: Target.Cpu.Arch.Family,
    feature: @field(Target, @tagName(family)).Feature,
    condition: bool,
) void {
    if (!condition) return;

    var add: Target.Cpu.Feature.Set = @field(Target, @tagName(family)).featureSet(&.{feature});
    add.populateDependencies(&@field(Target, @tagName(family)).all_features);

    var unblocked = add;
    unblocked.removeFeatureSet(blocked);

    if (unblocked.eql(add)) set.addFeatureSet(add);
}

fn abiRemovedFeatures(arch: Target.Cpu.Arch, abi: Target.Abi) Target.Cpu.Feature.Set {
    return switch (abi.float()) {
        .soft => switch (arch.family()) {
            .aarch64 => Target.aarch64.featureSet(&.{
                .fp_armv8,
            }),
            .alpha => Target.alpha.featureSet(&.{
                .fpu,
            }),
            .arm => Target.arm.featureSet(&.{
                .fpregs,
            }),
            .csky => Target.csky.featureSet(&.{
                .floate1,
                .float7e60,
            }),
            .hppa => Target.hppa.featureSet(&.{
                .fpu,
            }),
            .loongarch => Target.loongarch.featureSet(&.{
                .f,
            }),
            .m68k => Target.m68k.featureSet(&.{
                .isa_68881,
            }),
            .mips => Target.mips.featureSet(&.{
                .fpu_s,
            }),
            .powerpc => Target.powerpc.featureSet(&.{
                .fpu,
            }),
            .riscv => Target.riscv.featureSet(&.{
                .f,
            }),
            .sh => Target.sh.featureSet(&.{
                .fpu,
            }),
            .sparc => Target.sparc.featureSet(&.{
                .fpu,
            }),
            .x86 => Target.x86.featureSet(&.{
                .@"8087",
            }),
            else => .empty,
        },
        .hard32 => switch (arch.family()) {
            .csky => Target.csky.featureSet(&.{
                .float1e2,
                .fpuv3_df,
            }),
            .loongarch => Target.loongarch.featureSet(&.{
                .d,
            }),
            .mips => Target.mips.featureSet(&.{
                .fpu_d,
            }),
            .riscv => Target.riscv.featureSet(&.{
                .d,
            }),
            else => .empty,
        },
        .hard64 => .empty,
    };
}

fn detectNativeCpuAndFeatures(io: Io, cpu_arch: Target.Cpu.Arch) ?Target.Cpu {
    const family = builtin.target.cpu.arch.family();
    assert(cpu_arch.family() == family);
    switch (family) {
        .loongarch => return @import("system/loongarch.zig").detectNativeCpuAndFeatures(cpu_arch),
        .x86 => return @import("system/x86.zig").detectNativeCpuAndFeatures(cpu_arch),
        else => {},
    }

    switch (builtin.os.tag) {
        .freebsd => return freebsd.detectNativeCpuAndFeatures(),
        .linux => return linux.detectNativeCpuAndFeatures(io),
        .macos => return darwin.macos.detectNativeCpuAndFeatures(),
        .netbsd => return netbsd.detectNativeCpuAndFeatures(),
        .openbsd => return openbsd.detectNativeCpuAndFeatures(),
        .windows => return windows.detectNativeCpuAndFeatures(),
        else => {},
    }

    // This architecture does not have CPU model & feature detection yet.
    return null;
}

const AbiAndDynamicLinkerFromFileError = error{
    Canceled,
    AccessDenied,
    Unexpected,
    Unseekable,
    ReadFailed,
    EndOfStream,
    NameTooLong,
    StaticElfFile,
    InvalidElfFile,
    StreamTooLong,
    Timeout,
    SymLinkLoop,
    SystemResources,
    ProcessFdQuotaExceeded,
    SystemFdQuotaExceeded,
    IsDir,
    WouldBlock,
    InputOutput,
    BrokenPipe,
    ConnectionResetByPeer,
    NotOpenForReading,
    SocketUnconnected,
    LockViolation,
    FileSystem,
};

fn abiAndDynamicLinkerFromFile(
    file_reader: *Io.File.Reader,
    header: *const elf.Header,
    arch: Target.Cpu.Arch,
    os: Target.Os,
    ld_info_list: []const LdInfo,
    query: Target.Query,
) AbiAndDynamicLinkerFromFileError!DetectedTarget {
    assert(query.abi == null); // See the early exit in `detectAbiAndDynamicLinker`.

    const io = file_reader.io;
    var result: DetectedTarget = .{
        .os = os,
        .abi = .default(arch, os.tag),
        .dynamic_linker = query.dynamic_linker orelse .none,
    };
    var rpath_offset: ?u64 = null; // Found inside PT.DYNAMIC
    const look_for_ld = query.dynamic_linker == null;

    var got_dyn_section: bool = false;
    {
        var it = header.iterateProgramHeaders(file_reader);
        while (try it.next()) |phdr| switch (phdr.type) {
            .INTERP => {
                got_dyn_section = true;

                var interp: Target.DynamicLinker = .none;
                const p_filesz = phdr.filesz;
                if (p_filesz < 2) return error.InvalidElfFile;
                if (p_filesz > interp.buffer.len) return error.NameTooLong;
                const filesz: usize = @intCast(p_filesz);
                try file_reader.seekTo(phdr.offset);
                try file_reader.interface.readSliceAll(interp.buffer[0..filesz]);
                // PT.INTERP includes a null byte in filesz.
                const len = filesz - 1;
                // dynamic_linker.max_byte is "max", not "len".
                // We know it will fit in u8 because we check against dynamic_linker.buffer.len above.
                interp.len = @intCast(len);

                // Use it to determine ABI.
                for (ld_info_list) |ld_info| {
                    if (std.mem.endsWith(u8, interp.get().?, fs.path.basename(ld_info.ld.get().?))) {
                        result.abi = ld_info.abi;
                        if (look_for_ld) result.dynamic_linker = interp;
                        break;
                    }
                }
            },
            // We only need this for detecting glibc version.
            .DYNAMIC => {
                got_dyn_section = true;

                if (builtin.target.os.tag == .linux and result.abi.isGnu() and query.glibc_version == null) {
                    var dyn_it = header.iterateDynamicSection(file_reader, phdr.offset, phdr.filesz);
                    while (try dyn_it.next()) |dyn| {
                        if (dyn.d_tag == elf.DT_RUNPATH) {
                            rpath_offset = dyn.d_val;
                            break;
                        }
                    }
                }
            },
            else => continue,
        };
    }

    if (look_for_ld and result.dynamic_linker.get() == null) {
        result.dynamic_linker = .standard(arch, os.tag, result.abi);
    }

    if (!got_dyn_section) {
        return error.StaticElfFile;
    }

    if (builtin.target.os.tag == .linux and result.abi.isGnu() and query.glibc_version == null) {
        const str_section_off = header.shoff + @as(u64, header.shentsize) * @as(u64, header.shstrndx);
        try file_reader.seekTo(str_section_off);
        const shstr = try elf.takeSectionHeader(&file_reader.interface, header.is_64, header.endian);
        var strtab_buf: [4096]u8 = undefined;
        const shstrtab = strtab_buf[0..@min(shstr.sh_size, strtab_buf.len)];
        try file_reader.seekTo(shstr.sh_offset);
        try file_reader.interface.readSliceAll(shstrtab);
        const dynstr: ?struct { offset: u64, size: u64 } = find_dyn_str: {
            var it = header.iterateSectionHeaders(file_reader);
            while (try it.next()) |shdr| {
                const end = mem.findScalarPos(u8, shstrtab, shdr.sh_name, 0) orelse continue;
                const sh_name = shstrtab[shdr.sh_name..end :0];
                if (mem.eql(u8, sh_name, ".dynstr")) break :find_dyn_str .{
                    .offset = shdr.sh_offset,
                    .size = shdr.sh_size,
                };
            } else break :find_dyn_str null;
        };
        if (dynstr) |ds| {
            if (rpath_offset) |rpoff| {
                if (rpoff > ds.size) return error.InvalidElfFile;
                const rpoff_file = ds.offset + rpoff;
                const rp_max_size = ds.size - rpoff;

                try file_reader.seekTo(rpoff_file);
                const rpath_list = try file_reader.interface.takeSentinel(0);
                if (rpath_list.len > rp_max_size) return error.StreamTooLong;

                var it = mem.tokenizeScalar(u8, rpath_list, ':');
                while (it.next()) |rpath| {
                    if (glibcVerFromRPath(io, rpath)) |ver| {
                        result.os.version_range.linux.glibc = ver;
                        return result;
                    } else |err| switch (err) {
                        error.GLibCNotFound => continue,
                        else => |e| return e,
                    }
                }
            }
        }

        if (result.dynamic_linker.get()) |dl_path| glibc_ver: {
            // There is no DT_RUNPATH so we try to find libc.so.6 inside the same
            // directory as the dynamic linker.
            if (fs.path.dirname(dl_path)) |rpath| {
                if (glibcVerFromRPath(io, rpath)) |ver| {
                    result.os.version_range.linux.glibc = ver;
                    return result;
                } else |err| switch (err) {
                    error.GLibCNotFound => {},
                    else => |e| return e,
                }
            }

            // So far, no luck. Next we try to see if the information is
            // present in the symlink data for the dynamic linker path.
            var link_buffer: [posix.PATH_MAX]u8 = undefined;
            const link_name = if (Io.Dir.readLinkAbsolute(io, dl_path, &link_buffer)) |n|
                link_buffer[0..n]
            else |err| switch (err) {
                error.NameTooLong => unreachable,
                error.BadPathName => unreachable, // Windows only
                error.UnsupportedReparsePointType => unreachable, // Windows only
                error.NetworkNotFound => unreachable, // Windows only
                error.AntivirusInterference => unreachable, // Windows only
                error.FileBusy => unreachable, // Windows only

                error.AccessDenied,
                error.PermissionDenied,
                error.FileNotFound,
                error.NotLink,
                error.NotDir,
                => break :glibc_ver,

                error.SystemResources,
                error.FileSystem,
                error.SymLinkLoop,
                error.Canceled,
                error.Unexpected,
                => |e| return e,
            };
            result.os.version_range.linux.glibc = glibcVerFromLinkName(
                fs.path.basename(link_name),
                "ld-",
            ) catch |err| switch (err) {
                error.UnrecognizedGnuLibCFileName,
                error.InvalidGnuLibCVersion,
                => break :glibc_ver,
            };
            return result;
        }

        // Nothing worked so far. Finally we fall back to hard-coded search paths.
        // Some distros such as Debian keep their libc.so.6 in `/lib/$triple/`.
        var path_buf: [posix.PATH_MAX]u8 = undefined;
        var index: usize = 0;
        const prefix = "/lib/";
        const cpu_arch = @tagName(arch);
        const os_tag = @tagName(result.os.tag);
        const abi = @tagName(result.abi);
        @memcpy(path_buf[index..][0..prefix.len], prefix);
        index += prefix.len;
        @memcpy(path_buf[index..][0..cpu_arch.len], cpu_arch);
        index += cpu_arch.len;
        path_buf[index] = '-';
        index += 1;
        @memcpy(path_buf[index..][0..os_tag.len], os_tag);
        index += os_tag.len;
        path_buf[index] = '-';
        index += 1;
        @memcpy(path_buf[index..][0..abi.len], abi);
        index += abi.len;
        const rpath = path_buf[0..index];
        if (glibcVerFromRPath(io, rpath)) |ver| {
            result.os.version_range.linux.glibc = ver;
            return result;
        } else |err| switch (err) {
            error.GLibCNotFound => {},
            else => |e| return e,
        }
    }

    return result;
}

fn glibcVerFromLinkName(link_name: []const u8, prefix: []const u8) error{ UnrecognizedGnuLibCFileName, InvalidGnuLibCVersion }!std.SemanticVersion {
    // example: "libc-2.3.4.so"
    // example: "libc-2.27.so"
    // example: "ld-2.33.so"
    const suffix = ".so";
    if (!mem.startsWith(u8, link_name, prefix) or !mem.endsWith(u8, link_name, suffix)) {
        return error.UnrecognizedGnuLibCFileName;
    }
    // chop off "libc-" and ".so"
    const link_name_chopped = link_name[prefix.len .. link_name.len - suffix.len];
    return Target.Query.parseVersion(link_name_chopped) catch |err| switch (err) {
        error.Overflow => return error.InvalidGnuLibCVersion,
        error.InvalidVersion => return error.InvalidGnuLibCVersion,
    };
}

test glibcVerFromLinkName {
    try std.testing.expectError(error.UnrecognizedGnuLibCFileName, glibcVerFromLinkName("ld-2.37.so", "this-prefix-does-not-exist"));
    try std.testing.expectError(error.UnrecognizedGnuLibCFileName, glibcVerFromLinkName("libc-2.37.so-is-not-end", "libc-"));

    try std.testing.expectError(error.InvalidGnuLibCVersion, glibcVerFromLinkName("ld-2.so", "ld-"));
    try std.testing.expectEqual(std.SemanticVersion{ .major = 2, .minor = 37, .patch = 0 }, try glibcVerFromLinkName("ld-2.37.so", "ld-"));
    try std.testing.expectEqual(std.SemanticVersion{ .major = 2, .minor = 37, .patch = 0 }, try glibcVerFromLinkName("ld-2.37.0.so", "ld-"));
    try std.testing.expectEqual(std.SemanticVersion{ .major = 2, .minor = 37, .patch = 1 }, try glibcVerFromLinkName("ld-2.37.1.so", "ld-"));
    try std.testing.expectError(error.InvalidGnuLibCVersion, glibcVerFromLinkName("ld-2.37.4.5.so", "ld-"));
}

fn glibcVerFromRPath(io: Io, rpath: []const u8) !std.SemanticVersion {
    const cwd: Io.Dir = .cwd();

    var dir = cwd.openDir(io, rpath, .{}) catch |err| switch (err) {
        error.NameTooLong => return error.Unexpected,
        error.BadPathName => return error.Unexpected,
        error.NetworkNotFound => return error.Unexpected, // Windows-only

        error.FileNotFound => return error.GLibCNotFound,
        error.NotDir => return error.GLibCNotFound,
        error.AccessDenied => return error.GLibCNotFound,
        error.PermissionDenied => return error.GLibCNotFound,
        error.NoDevice => return error.GLibCNotFound,

        error.ProcessFdQuotaExceeded => |e| return e,
        error.SystemFdQuotaExceeded => |e| return e,
        error.SystemResources => |e| return e,
        error.SymLinkLoop => |e| return e,
        error.Unexpected => |e| return e,
        error.Canceled => |e| return e,
    };
    defer dir.close(io);

    // Now we have a candidate for the path to libc shared object. In
    // the past, we used readlink() here because the link name would
    // reveal the glibc version. However, in more recent GNU/Linux
    // installations, there is no symlink. Thus we instead use a more
    // robust check of opening the libc shared object and looking at the
    // .dynstr section, and finding the max version number of symbols
    // that start with "GLIBC_2.".
    const glibc_so_basename = "libc.so.6";
    var file = dir.openFile(io, glibc_so_basename, .{}) catch |err| switch (err) {
        error.NameTooLong => return error.Unexpected,
        error.BadPathName => return error.Unexpected,
        error.PipeBusy => return error.Unexpected, // Windows-only
        error.NetworkNotFound => return error.Unexpected, // Windows-only
        error.AntivirusInterference => return error.Unexpected, // Windows-only
        error.FileLocksUnsupported => return error.Unexpected, // No lock requested.
        error.NoSpaceLeft => return error.Unexpected, // read-only
        error.PathAlreadyExists => return error.Unexpected, // read-only
        error.DeviceBusy => return error.Unexpected, // read-only
        error.FileBusy => return error.Unexpected, // read-only
        error.ReadOnlyFileSystem => return error.Unexpected, // read-only
        error.NoDevice => return error.Unexpected, // not asking for a special device
        error.FileTooBig => return error.Unexpected,
        error.WouldBlock => return error.Unexpected, // not opened in non-blocking

        error.AccessDenied => return error.GLibCNotFound,
        error.PermissionDenied => return error.GLibCNotFound,
        error.FileNotFound => return error.GLibCNotFound,
        error.NotDir => return error.GLibCNotFound,
        error.IsDir => return error.GLibCNotFound,

        error.ProcessFdQuotaExceeded => |e| return e,
        error.SystemFdQuotaExceeded => |e| return e,
        error.SystemResources => |e| return e,
        error.SymLinkLoop => |e| return e,
        error.Unexpected => |e| return e,
        error.Canceled => |e| return e,
    };
    defer file.close(io);

    // Empirically, glibc 2.34 libc.so .dynstr section is 32441 bytes on my system.
    var buffer: [8000]u8 = undefined;
    var file_reader: Io.File.Reader = .init(file, io, &buffer);

    return glibcVerFromSoFile(&file_reader) catch |err| switch (err) {
        error.InvalidElfMagic,
        error.InvalidElfEndian,
        error.InvalidElfClass,
        error.InvalidElfVersion,
        error.InvalidGnuLibCVersion,
        error.EndOfStream,
        => return error.GLibCNotFound,

        error.ReadFailed => return file_reader.err.?,
        else => |e| return e,
    };
}

fn glibcVerFromSoFile(file_reader: *Io.File.Reader) !std.SemanticVersion {
    const header = try elf.Header.read(&file_reader.interface);
    const str_section_off = header.shoff + @as(u64, header.shentsize) * @as(u64, header.shstrndx);
    try file_reader.seekTo(str_section_off);
    const shstr = try elf.takeSectionHeader(&file_reader.interface, header.is_64, header.endian);
    var strtab_buf: [4096]u8 = undefined;
    const shstrtab = strtab_buf[0..@min(shstr.sh_size, strtab_buf.len)];
    try file_reader.seekTo(shstr.sh_offset);
    try file_reader.interface.readSliceAll(shstrtab);
    const dynstr: struct { offset: u64, size: u64 } = find_dyn_str: {
        var it = header.iterateSectionHeaders(file_reader);
        while (try it.next()) |shdr| {
            const end = mem.findScalarPos(u8, shstrtab, shdr.sh_name, 0) orelse continue;
            const sh_name = shstrtab[shdr.sh_name..end :0];
            if (mem.eql(u8, sh_name, ".dynstr")) break :find_dyn_str .{
                .offset = shdr.sh_offset,
                .size = shdr.sh_size,
            };
        } else return error.InvalidGnuLibCVersion;
    };

    // Here we loop over all the strings in the dynstr string table, assuming that any
    // strings that start with "GLIBC_2." indicate the existence of such a glibc version,
    // and furthermore, that the system-installed glibc is at minimum that version.
    var max_ver: std.SemanticVersion = .{ .major = 2, .minor = 2, .patch = 5 };
    var offset: u64 = 0;
    try file_reader.seekTo(dynstr.offset);
    while (offset < dynstr.size) {
        if (file_reader.interface.takeSentinel(0)) |s| {
            if (mem.startsWith(u8, s, "GLIBC_2.")) {
                const chopped = s["GLIBC_".len..];
                const ver = Target.Query.parseVersion(chopped) catch |err| switch (err) {
                    error.Overflow => return error.InvalidGnuLibCVersion,
                    error.InvalidVersion => return error.InvalidGnuLibCVersion,
                };
                switch (ver.order(max_ver)) {
                    .gt => max_ver = ver,
                    .lt, .eq => continue,
                }
            }
            offset += s.len + 1;
        } else |err| switch (err) {
            error.EndOfStream, error.StreamTooLong => break,
            error.ReadFailed => |e| return e,
        }
    }

    return max_ver;
}

const DetectedTarget = struct {
    os: Target.Os,
    abi: Target.Abi,
    dynamic_linker: Target.DynamicLinker,
};

/// In the past, this function attempted to use the executable's own binary if it was dynamically
/// linked to answer both the C ABI question and the dynamic linker question. However, this
/// could be problematic on a system that uses a RUNPATH for the compiler binary, locking
/// it to an older glibc version, while system binaries such as /usr/bin/env use a newer glibc
/// version. The problem is that libc.so.6 glibc version will match that of the system while
/// the dynamic linker will match that of the compiler binary. Executables with these versions
/// mismatching will fail to run.
///
/// Therefore, this function works the same regardless of whether the compiler binary is
/// dynamically or statically linked. It inspects `/usr/bin/env` as an ELF file to find the
/// answer to these questions, or if there is a shebang line, then it chases the referenced
/// file recursively. If that does not provide the answer, then the function falls back to
/// defaults.
fn detectAbiAndDynamicLinker(io: Io, arch: Target.Cpu.Arch, os: Target.Os, query: Target.Query) !DetectedTarget {
    const native_os_uses_elf = comptime Target.ObjectFormat.default(builtin.target.os.tag, builtin.target.cpu.arch) == .elf;
    const native_ld_needs_abi = comptime Target.DynamicLinker.kind(builtin.target.os.tag) == .arch_os_abi;
    if (!native_os_uses_elf or !native_ld_needs_abi or query.os_tag != null or query.abi != null) {
        return defaultAbiAndDynamicLinker(arch, os, query);
    }

    assert(os.tag == builtin.target.os.tag);

    // The current target's ABI cannot be relied on for this. For example, we may build the zig
    // compiler for target riscv64-linux-musl and provide a tarball for users to download.
    // A user could then run that zig compiler on riscv64-linux-gnu. This use case is well-defined
    // and supported by Zig. But that means that we must detect the system ABI here rather than
    // relying on `builtin.target`.
    const all_abis = comptime blk: {
        assert(@backingInt(Target.Abi.none) == 0);
        const field_names = @typeInfo(Target.Abi).@"enum".field_names[1..];
        var array: [field_names.len]Target.Abi = undefined;
        for (field_names, 0..) |field_name, i| {
            array[i] = @field(Target.Abi, field_name);
        }
        break :blk array;
    };
    var ld_info_list_buffer: [all_abis.len]LdInfo = undefined;
    var ld_info_list_len: usize = 0;

    for (all_abis) |abi| {
        const ld: Target.DynamicLinker = .standard(arch, os.tag, abi);

        // Does the generated target triple actually have a standard dynamic linker path?
        if (ld.get() == null) continue;

        ld_info_list_buffer[ld_info_list_len] = .{
            .ld = ld,
            .abi = abi,
        };
        ld_info_list_len += 1;
    }

    const ld_info_list = ld_info_list_buffer[0..ld_info_list_len];

    var file_reader: Io.File.Reader = undefined;
    // According to `man 2 execve`:
    //
    // The kernel imposes a maximum length on the text
    // that follows the "#!" characters at the start of a script;
    // characters beyond the limit are ignored.
    // Before Linux 5.1, the limit is 127 characters.
    // Since Linux 5.1, the limit is 255 characters.
    //
    // Tests show that bash and zsh consider 255 as total limit,
    // *including* "#!" characters and ignoring newline.
    // For safety, we set max length as 255 + \n (1).
    const max_shebang_line_size = 256;
    var file_reader_buffer: [4096]u8 = undefined;
    comptime assert(file_reader_buffer.len >= max_shebang_line_size);

    // Best case scenario: the executable is dynamically linked, and we can iterate
    // over our own shared objects and find a dynamic linker.
    const header = elf_file: {
        // This block looks for a shebang line in "/usr/bin/env". If it finds
        // one, then instead of using "/usr/bin/env" as the ELF file to examine,
        // it uses the file it references instead, doing the same logic
        // recursively in case it finds another shebang line.

        // Since /usr/bin/env is hard-coded into the shebang line of many
        // portable scripts, it's a reasonably reliable path to start with.
        var file_name: []const u8 = "/usr/bin/env";

        while (true) {
            const file = Io.Dir.openFileAbsolute(io, file_name, .{}) catch |err| switch (err) {
                error.NoSpaceLeft => return error.Unexpected,
                error.NameTooLong => return error.Unexpected,
                error.PathAlreadyExists => return error.Unexpected,
                error.BadPathName => return error.Unexpected,
                error.PipeBusy => return error.Unexpected,
                error.FileLocksUnsupported => return error.Unexpected,
                error.FileBusy => return error.Unexpected, // opened without write permissions
                error.AntivirusInterference => return error.Unexpected, // Windows-only error

                error.IsDir,
                error.NotDir,
                error.AccessDenied,
                error.PermissionDenied,
                error.NoDevice,
                error.FileNotFound,
                error.NetworkNotFound,
                error.FileTooBig,
                error.Unexpected,
                => |e| if (builtin.target.os.tag == .linux and e == error.FileNotFound and mem.eql(u8, file_name, "/usr/bin/env")) {
                    // Android does not have a /usr directory, so try again
                    file_name = "/system/bin/env";
                    continue;
                } else return error.UnableToOpenElfFile,
                else => |e| return e,
            };
            var is_elf_file = false;
            defer if (!is_elf_file) file.close(io);

            file_reader = .init(file, io, &file_reader_buffer);
            file_name = undefined; // it aliases file_reader_buffer

            const header = elf.Header.read(&file_reader.interface) catch |hdr_err| switch (hdr_err) {
                error.EndOfStream,
                error.InvalidElfMagic,
                => {
                    const shebang_line = file_reader.interface.takeSentinel('\n') catch |err| switch (err) {
                        error.ReadFailed => return file_reader.err.?,
                        // It's neither an ELF file nor file with shebang line.
                        error.EndOfStream, error.StreamTooLong => return error.UnhelpfulFile,
                    };
                    if (!mem.startsWith(u8, shebang_line, "#!")) return error.UnhelpfulFile;
                    // We detected shebang, now parse entire line.

                    // Trim leading "#!", spaces and tabs.
                    const trimmed_line = mem.trimStart(u8, shebang_line[2..], &.{ ' ', '\t' });

                    // This line can have:
                    // * Interpreter path only,
                    // * Interpreter path and arguments, all separated by space, tab or NUL character.
                    // And optionally newline at the end.
                    const path_maybe_args = mem.trimEnd(u8, trimmed_line, "\n");

                    // Separate path and args.
                    const path_end = mem.findAny(u8, path_maybe_args, &.{ ' ', '\t', 0 }) orelse path_maybe_args.len;
                    const unvalidated_path = path_maybe_args[0..path_end];
                    file_name = if (fs.path.isAbsolute(unvalidated_path)) unvalidated_path else return error.RelativeShebang;
                    continue;
                },

                error.InvalidElfVersion,
                error.InvalidElfClass,
                error.InvalidElfEndian,
                => return error.InvalidElfFile,

                error.ReadFailed => return file_reader.err.?,
            };
            is_elf_file = true;
            break :elf_file header;
        }
    };
    defer file_reader.file.close(io);

    var result = abiAndDynamicLinkerFromFile(&file_reader, &header, arch, os, ld_info_list, query) catch |err| switch (err) {
        error.FileSystem,
        error.SystemResources,
        error.SymLinkLoop,
        error.ProcessFdQuotaExceeded,
        error.SystemFdQuotaExceeded,
        error.Canceled,
        => |e| return e,

        error.ReadFailed => return file_reader.err.?,

        else => |e| {
            std.log.warn("encountered {t}; falling back to default ABI and dynamic linker", .{e});
            return defaultAbiAndDynamicLinker(arch, os, query);
        },
    };

    if (builtin.target.os.tag == .linux and result.abi.isAndroid() and query.android_api_level == null) {
        result.os.version_range.linux.android = detectAndroidApiLevel(io) catch |err| return switch (err) {
            error.InvalidWtf8,
            error.InvalidBatchScriptArg,
            => unreachable, // Windows-only
            error.ApiLevelQueryFailed => |e| e,
            else => blk: {
                std.log.err("spawning or reading from getprop failed ({s})", .{@errorName(err)});
                switch (err) {
                    error.SystemResources,
                    error.FileSystem,
                    error.ProcessFdQuotaExceeded,
                    error.SystemFdQuotaExceeded,
                    error.SymLinkLoop,
                    => |e| break :blk e,
                    else => break :blk error.ApiLevelQueryFailed,
                }
            },
        };
    }

    return result;
}

fn defaultAbiAndDynamicLinker(arch: Target.Cpu.Arch, os: Target.Os, query: Target.Query) DetectedTarget {
    const abi: Target.Abi = query.abi orelse .default(arch, os.tag);
    return .{
        .os = os,
        .abi = abi,
        .dynamic_linker = query.dynamic_linker orelse .standard(arch, os.tag, abi),
    };
}

const LdInfo = struct {
    ld: Target.DynamicLinker,
    abi: Target.Abi,
};

fn detectAndroidApiLevel(io: Io) !u32 {
    comptime if (builtin.os.tag != .linux) unreachable;

    var child = try std.process.spawn(io, .{
        .argv = &.{
            "/system/bin/getprop",
            "ro.build.version.sdk",
        },
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .ignore,
    });
    errdefer child.kill(io);

    // PROP_VALUE_MAX is 92, output is value + newline.
    // Currently API levels are two-digit numbers, but we want to make sure we never read a partial value.
    var stdout_buf: [92 + 1]u8 = undefined;
    var reader = child.stdout.?.readerStreaming(io, &.{});
    const n = try reader.interface.readSliceShort(&stdout_buf);

    switch (try child.wait(io)) {
        .exited => |code| if (code != 0) {
            std.log.err("getprop terminated abnormally with exit code: {d}", .{code});
            return error.ApiLevelQueryFailed;
        },
        .signal => |sig| {
            std.log.err("getprop terminated abnormally with signal: {t}", .{sig});
            return error.ApiLevelQueryFailed;
        },
        .stopped => |sig| {
            std.log.err("getprop stopped abnormally with signal: {t}", .{sig});
            return error.ApiLevelQueryFailed;
        },
        .unknown => {
            std.log.err("getprop terminated abnormally", .{});
            return error.ApiLevelQueryFailed;
        },
    }

    const result = mem.trimEnd(u8, stdout_buf[0..n], "\n");
    return std.fmt.parseInt(u32, result, 10) catch |e| {
        std.log.err(
            "Could not parse API level, unexpected getprop output '{s}' ({s})",
            .{ result, @errorName(e) },
        );
        return error.ApiLevelQueryFailed;
    };
}

test {
    _ = NativePaths;

    _ = darwin;
    _ = freebsd;
    _ = linux;
    _ = netbsd;
    _ = openbsd;
    _ = windows;
}
