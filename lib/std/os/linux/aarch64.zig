const builtin = @import("builtin");
const std = @import("../../std.zig");
const SYS = std.os.linux.SYS;

pub const syscall_arg_t = u64;

/// Declared only with SVE, since a Z clobber also clobbers the V register within it.
const clobbers: std.lang.assembly.Clobbers = if (builtin.zig_backend != .stage2_aarch64 and
    builtin.cpu.has(.aarch64, .sve)) .{
    .memory = true,
    .p0 = true,
    .p1 = true,
    .p2 = true,
    .p3 = true,
    .p4 = true,
    .p5 = true,
    .p6 = true,
    .p7 = true,
    .p8 = true,
    .p9 = true,
    .p10 = true,
    .p11 = true,
    .p12 = true,
    .p13 = true,
    .p14 = true,
    .p15 = true,
    .z0 = true,
    .z1 = true,
    .z2 = true,
    .z3 = true,
    .z4 = true,
    .z5 = true,
    .z6 = true,
    .z7 = true,
    .z8 = true,
    .z9 = true,
    .z10 = true,
    .z11 = true,
    .z12 = true,
    .z13 = true,
    .z14 = true,
    .z15 = true,
    .z16 = true,
    .z17 = true,
    .z18 = true,
    .z19 = true,
    .z20 = true,
    .z21 = true,
    .z22 = true,
    .z23 = true,
    .z24 = true,
    .z25 = true,
    .z26 = true,
    .z27 = true,
    .z28 = true,
    .z29 = true,
    .z30 = true,
    .z31 = true,
} else .{ .memory = true };

pub fn syscall0(
    number: SYS,
) u64 {
    return asm volatile ("svc #0"
        : [ret] "={x0}" (-> u64),
        : [number] "{x8}" (@backingInt(number)),
        : clobbers);
}

pub fn syscall1(
    number: SYS,
    arg1: syscall_arg_t,
) u64 {
    return asm volatile ("svc #0"
        : [ret] "={x0}" (-> u64),
        : [number] "{x8}" (@backingInt(number)),
          [arg1] "{x0}" (arg1),
        : clobbers);
}

pub fn syscall2(
    number: SYS,
    arg1: syscall_arg_t,
    arg2: syscall_arg_t,
) u64 {
    return asm volatile ("svc #0"
        : [ret] "={x0}" (-> u64),
        : [number] "{x8}" (@backingInt(number)),
          [arg1] "{x0}" (arg1),
          [arg2] "{x1}" (arg2),
        : clobbers);
}

pub fn syscall3(
    number: SYS,
    arg1: syscall_arg_t,
    arg2: syscall_arg_t,
    arg3: syscall_arg_t,
) u64 {
    return asm volatile ("svc #0"
        : [ret] "={x0}" (-> u64),
        : [number] "{x8}" (@backingInt(number)),
          [arg1] "{x0}" (arg1),
          [arg2] "{x1}" (arg2),
          [arg3] "{x2}" (arg3),
        : clobbers);
}

pub fn syscall4(
    number: SYS,
    arg1: syscall_arg_t,
    arg2: syscall_arg_t,
    arg3: syscall_arg_t,
    arg4: syscall_arg_t,
) u64 {
    return asm volatile ("svc #0"
        : [ret] "={x0}" (-> u64),
        : [number] "{x8}" (@backingInt(number)),
          [arg1] "{x0}" (arg1),
          [arg2] "{x1}" (arg2),
          [arg3] "{x2}" (arg3),
          [arg4] "{x3}" (arg4),
        : clobbers);
}

pub fn syscall5(
    number: SYS,
    arg1: syscall_arg_t,
    arg2: syscall_arg_t,
    arg3: syscall_arg_t,
    arg4: syscall_arg_t,
    arg5: syscall_arg_t,
) u64 {
    return asm volatile ("svc #0"
        : [ret] "={x0}" (-> u64),
        : [number] "{x8}" (@backingInt(number)),
          [arg1] "{x0}" (arg1),
          [arg2] "{x1}" (arg2),
          [arg3] "{x2}" (arg3),
          [arg4] "{x3}" (arg4),
          [arg5] "{x4}" (arg5),
        : clobbers);
}

pub fn syscall6(
    number: SYS,
    arg1: syscall_arg_t,
    arg2: syscall_arg_t,
    arg3: syscall_arg_t,
    arg4: syscall_arg_t,
    arg5: syscall_arg_t,
    arg6: syscall_arg_t,
) u64 {
    return asm volatile ("svc #0"
        : [ret] "={x0}" (-> u64),
        : [number] "{x8}" (@backingInt(number)),
          [arg1] "{x0}" (arg1),
          [arg2] "{x1}" (arg2),
          [arg3] "{x2}" (arg3),
          [arg4] "{x3}" (arg4),
          [arg5] "{x4}" (arg5),
          [arg6] "{x5}" (arg6),
        : clobbers);
}

pub fn clone() callconv(.naked) u64 {
    // __clone(func, stack, flags, arg, ptid, tls, ctid)
    //         x0,   x1,    w2,    x3,  x4,   x5,  x6
    //
    // syscall(SYS_clone, flags, stack, ptid, tls, ctid)
    //         x8,        x0,    x1,    x2,   x3,  x4
    asm volatile (
        \\ // align stack and save func,arg
        \\ and x1,x1,#-16
        \\ stp x0,x3,[x1,#-16]!
        \\
        \\ // syscall
        \\ uxtw x0,w2
        \\ mov x2,x4
        \\ mov x3,x5
        \\ mov x4,x6
        \\ mov x8,#220 // SYS_clone
        \\ svc #0
        \\
        \\ cbz x0,1f
        \\ // parent
        \\ ret
        \\
        \\ // child
        \\1:
    );
    if (builtin.unwind_tables != .none or !builtin.strip_debug_info) asm volatile (
        \\ .cfi_undefined lr
    );
    asm volatile (
        \\ mov fp, 0
        \\ mov lr, 0
        \\
        \\ ldp x1,x0,[sp],#16
        \\ blr x1
        \\ mov x8,#93 // SYS_exit
        \\ svc #0
    );
}

pub const restore = restore_rt;

pub fn restore_rt() callconv(.naked) noreturn {
    asm volatile (
        \\ mov w8, %[number]
        \\ svc #0
        :
        : [number] "i" (@backingInt(SYS.rt_sigreturn)),
    );
}

pub const VDSO = struct {
    pub const CGT_SYM = "__kernel_clock_gettime";
    pub const CGT_VER = "LINUX_2.6.39";
};

pub const time_t = i64;
