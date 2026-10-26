const builtin = @import("builtin");
const std = @import("std");

const Prid = packed struct(u32) {
    product: u12,
    series: enum(u4) {
        la132 = 0x8,
        la264 = 0xa,
        la364 = 0xb,
        la464 = 0xc,
        la664 = 0xd,
        _,
    },
    company: enum(u8) {
        loongson = 0x14,
        _,
    },
    reserved: u8,
};

const Variant = enum(u2) {
    la32r = 0b00,
    la32 = 0b01,
    la64 = 0b10,
    reserved = 0b11,
};

inline fn bit(input: u32, offset: u5) bool {
    return (input >> offset) & 1 != 0;
}

fn setFeature(cpu: *std.Target.Cpu, feature: std.Target.loongarch.Feature, enabled: bool) void {
    const idx = @as(std.Target.Cpu.Feature.Set.Index, @backingInt(feature));

    if (enabled) cpu.features.addFeature(idx) else cpu.features.removeFeature(idx);
}

pub fn detectNativeCpuAndFeatures(arch: std.Target.Cpu.Arch) ?std.Target.Cpu {
    const prid: Prid = @bitCast(cpucfg(0));

    var cpu: std.Target.Cpu = .{
        .arch = arch,
        .model = switch (arch) {
            .loongarch32 => m: {
                switch (prid.company) {
                    .loongson => switch (prid.series) {
                        .la132 => break :m &std.Target.loongarch.cpu.la132,
                        else => {},
                    },
                    else => {},
                }

                break :m &std.Target.loongarch.cpu.generic_la32;
            },
            .loongarch64 => m: {
                switch (prid.company) {
                    .loongson => switch (prid.series) {
                        .la464 => break :m &std.Target.loongarch.cpu.la464,
                        .la664 => break :m &std.Target.loongarch.cpu.la664,
                        else => {},
                    },
                    else => {},
                }

                break :m &std.Target.loongarch.cpu.generic_la64;
            },
            else => unreachable,
        },
        .features = .empty,
    };

    cpu.features.addFeatureSet(cpu.model.features);

    const variant: Variant = @fromBackingInt(@intCast(cpucfg(1) & 0b11));

    setFeature(&cpu, .@"32s", variant == .la32 or variant == .la64);

    const cfg2 = cpucfg(2);
    const cfg3 = cpucfg(3);

    if (builtin.os.tag == .linux) {
        const HWCAP = std.os.linux.HWCAP;
        const hwcap_bits: usize = if (builtin.link_libc)
            std.c.getauxval(std.elf.AT.HWCAP)
        else
            std.os.linux.getauxval(std.elf.AT.HWCAP);

        setFeature(&cpu, .unaligned_access, (hwcap_bits & HWCAP.UAL) != 0);

        const has_fpu = (hwcap_bits & HWCAP.FPU) != 0;
        setFeature(&cpu, .f, has_fpu and bit(cfg2, 1));
        setFeature(&cpu, .d, has_fpu and bit(cfg2, 2));
        setFeature(&cpu, .lsx, (hwcap_bits & HWCAP.LSX) != 0);
        setFeature(&cpu, .lasx, (hwcap_bits & HWCAP.LASX) != 0);

        setFeature(&cpu, .lvz, (hwcap_bits & HWCAP.LVZ) != 0);
        setFeature(&cpu, .lbt, (hwcap_bits & HWCAP.LBT_X86) != 0 and (hwcap_bits & HWCAP.LBT_ARM) != 0 and (hwcap_bits & HWCAP.LBT_MIPS) != 0);

        setFeature(&cpu, .lam_bh, (hwcap_bits & HWCAP.LAM_BH) != 0);
    } else {
        setFeature(&cpu, .unaligned_access, false);

        setFeature(&cpu, .f, false);
        setFeature(&cpu, .d, false);
        setFeature(&cpu, .lsx, false);
        setFeature(&cpu, .lasx, false);

        setFeature(&cpu, .lvz, false);
        setFeature(&cpu, .lbt, false);

        setFeature(&cpu, .lam_bh, false);
    }

    setFeature(&cpu, .frecipe, bit(cfg2, 25));
    setFeature(&cpu, .div32, bit(cfg2, 26));
    setFeature(&cpu, .lamcas, bit(cfg2, 28));
    setFeature(&cpu, .scq, bit(cfg2, 30));

    setFeature(&cpu, .ld_seq_sa, bit(cfg3, 23));

    cpu.features.populateDependencies(cpu.arch.allFeaturesList());

    return cpu;
}

fn cpucfg(word: u32) u32 {
    return asm ("cpucfg %[result], %[word]"
        : [result] "=r" (-> u32),
        : [word] "r" (word),
    );
}
