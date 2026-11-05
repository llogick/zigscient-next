pub fn addCases(ctx: *LinkContext) void {
    if (ctx.includeTest("merge_visibility")) |case| {
        const obj = case.addObject(.{
            .name = "obj",
            .zig_source_bytes =
            \\// These all have default visibility:
            \\export var foo_0: u32 = 123;
            \\export var foo_1: u32 = 123;
            \\export var foo_2: u32 = 123;
            \\export var foo_3: u32 = 123;
            \\
            \\// ...but now we'll add some references which override the visibility:
            \\export const ptrs: [*]const *u32 = &.{
            \\    @extern(*u32, .{ .name = "foo_0", .visibility = .default }),
            \\
            \\    @extern(*u32, .{ .name = "foo_1", .visibility = .protected }),
            \\
            \\    @extern(*u32, .{ .name = "foo_2", .visibility = .hidden }),
            \\
            \\    @extern(*u32, .{ .name = "foo_3", .visibility = .default }),
            \\    @extern(*u32, .{ .name = "foo_3", .visibility = .protected }),
            \\    @extern(*u32, .{ .name = "foo_3", .visibility = .hidden }),
            \\    @extern(*u32, .{ .name = "foo_3", .visibility = .default }),
            \\    @extern(*u32, .{ .name = "foo_3", .visibility = .protected }),
            \\};
            \\
            ,
        });
        case.verifyObjdump(obj.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--relocs",
            "--only-symbol=foo_",
        }, .{ .sub_name = "obj" });

        const lib = case.addLibrary(.dynamic, .{
            .name = "lib",
        });
        lib.root_module.addObject(obj);
        case.verifyObjdump(lib.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--relocs",
            "--only-symbol=foo_",
        }, .{ .sub_name = "lib" });
    }

    if (ctx.includeTest("copy_reloc")) |case| case: {
        if (!ctx.link_libc) break :case;

        const lib = case.addLibrary(.dynamic, .{
            .name = "lib",
            .zig_source_bytes =
            \\export var my_favourite_number: u32 = 42;
            \\
            \\export const var_ptr_in_lib: *const u32 = &struct {
            \\    extern var my_favourite_number: u32;
            \\}.my_favourite_number;
            \\
            ,
        });

        const exe = case.addExecutable(.{
            .name = "test",
            .pic = false, // force PDC so that we need a copy reloc
            .zig_source_bytes =
            \\extern var my_favourite_number: u32;
            \\extern const var_ptr_in_lib: *u32;
            \\pub fn main() void {
            \\    if (var_ptr_in_lib != &my_favourite_number) {
            \\        std.process.fatal("pointers do not match\n", .{});
            \\    }
            \\    std.debug.print("number: {d}\n", .{my_favourite_number});
            \\}
            \\const std = @import("std");
            \\
            ,
        });
        exe.linkage = .dynamic;
        exe.root_module.linkLibrary(lib);

        case.verifyObjdump(exe.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--relocs",
            "--only-symbol=my_favourite_number",
            "--only-section=.dynsym",
            "--only-section=.rela.dyn",
        }, .{});

        const run = case.addRunArtifact(exe);
        run.expectStdErrEqual("number: 42\n");
    }

    if (ctx.includeTest("canonical_plt")) |case| case: {
        if (!ctx.link_libc) break :case;

        const lib = case.addLibrary(.dynamic, .{
            .name = "lib",
            .zig_source_bytes =
            \\export fn get_a_number() u32 {
            \\    return 1234;
            \\}
            \\
            \\export const fn_ptr_in_lib: *const fn () callconv(.c) u32 = &struct {
            \\    extern fn get_a_number() u32;
            \\}.get_a_number;
            \\
            ,
        });

        const exe = case.addExecutable(.{
            .name = "test",
            .pic = false, // force PDC so that we need a copy reloc
            .zig_source_bytes =
            \\extern fn get_a_number() u32;
            \\extern const fn_ptr_in_lib: *const fn () callconv(.c) u32;
            \\pub fn main() void {
            \\    if (fn_ptr_in_lib != &get_a_number) {
            \\        std.process.fatal("pointers do not match\n", .{});
            \\    }
            \\    std.debug.print("number: {d}\n", .{get_a_number()});
            \\}
            \\const std = @import("std");
            \\
            ,
        });
        exe.linkage = .dynamic;
        exe.root_module.linkLibrary(lib);

        case.verifyObjdump(exe.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--only-symbol=get_a_number",
            "--only-section=.dynsym",
        }, .{});

        const run = case.addRunArtifact(exe);
        run.expectStdErrEqual("number: 1234\n");
    }

    if (ctx.includeTest("textrel")) |case| {
        const foo = case.addLibrary(.dynamic, .{
            .name = "foo",
            .zig_source_bytes =
            \\export var my_favourite_number: u32 = 42;
            \\
            ,
        });

        const bar = case.addLibrary(.dynamic, .{
            .name = "bar",
            .asm_source_bytes =
            \\.section .rodata
            \\.global my_favourite_number
            \\.quad my_favourite_number
            \\
            ,
        });
        bar.root_module.linkLibrary(foo);

        case.verifyObjdump(bar.getEmittedBin(), &.{
            "-s",
            "--dynamic",
            "--relocs",
            "--only-symbol=my_favourite_number",
        }, .{ .abi = true });
    }

    if (ctx.includeTest("weak_extern")) |case| case: {
        // In a library, we get a dynamic reloc:
        const lib = case.addLibrary(.dynamic, .{
            .name = "test",
            .zig_source_bytes =
            \\export var ptr = @extern(*anyopaque, .{ .name = "interesting_symbol", .linkage = .weak });
            \\
            ,
        });
        case.verifyObjdump(lib.getEmittedBin(), &.{
            "-s",
            "--relocs",
            "--only-symbol=interesting_symbol",
        }, .{ .sub_name = "lib" });

        if (!ctx.link_libc) break :case;

        // ...but in an executable, we statically resolve the value to 0:
        const exe = case.addExecutable(.{
            .name = "test",
            .zig_source_bytes =
            \\export var ptr = @extern(*anyopaque, .{ .name = "interesting_symbol", .linkage = .weak });
            \\pub fn main() void {}
            \\
            ,
        });
        exe.linkage = .dynamic;
        case.verifyObjdump(exe.getEmittedBin(), &.{
            "-s",
            "--relocs",
            "--only-symbol=interesting_symbol",
        }, .{ .sub_name = "exe" });
    }

    if (ctx.includeTest("static_symbol_versions")) |case| case: {
        if (ctx.link_libc and ctx.target.result.abi.isGnu()) break :case;

        const libfoo = case.addLibrary(.static, .{
            .name = "foo",
            .zig_source_bytes =
            \\export fn @"do_stuff@@FOO_V3"() u32 { return 3; }
            \\export fn @"do_stuff@FOO_V2"() u32 { return 2; }
            \\export fn @"do_stuff@FOO_V1"() u32 { return 1; }
            \\
            ,
        });
        case.verifyObjdump(libfoo.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--only-symbol=do_stuff",
        }, .{ .sub_name = "lib" });

        const call = case.addExecutable(.{
            .name = "call",
            .zig_source_bytes =
            \\extern fn do_stuff() u32;
            \\extern fn @"do_stuff@FOO_V1"() u32;
            \\extern fn @"do_stuff@FOO_V2"() u32;
            \\extern fn @"do_stuff@FOO_V3"() u32;
            \\pub fn main() void {
            \\    std.debug.print("default: {d}\n", .{do_stuff()});
            \\    std.debug.print("explicit V1: {d}\n", .{@"do_stuff@FOO_V1"()});
            \\    std.debug.print("explicit V2: {d}\n", .{@"do_stuff@FOO_V2"()});
            \\    std.debug.print("explicit V3: {d}\n", .{@"do_stuff@FOO_V3"()});
            \\}
            \\const std = @import("std");
            \\
            ,
        });
        call.linkage = .static;
        call.root_module.linkLibrary(libfoo);

        const run_call = case.addRunArtifact(call);
        run_call.expectStdErrEqual(
            \\default: 3
            \\explicit V1: 1
            \\explicit V2: 2
            \\explicit V3: 3
            \\
        );
    }

    if (ctx.includeTest("dynamic_symbol_versions")) |case| case: {
        if (!ctx.link_libc) break :case;

        const libfoo = case.addLibrary(.dynamic, .{
            .name = "foo",
            .zig_source_bytes =
            \\export fn @"do_stuff@@FOO_V3"() u32 { return 3; }
            \\export fn @"do_stuff@FOO_V2"() u32 { return 2; }
            \\export fn @"do_stuff@FOO_V1"() u32 { return 1; }
            \\
            ,
        });
        case.verifyObjdump(libfoo.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--symbol-versions",
            "--only-section=.dynsym",
            "--only-symbol=do_stuff",
        }, .{ .sub_name = "lib" });

        const call_default = case.addExecutable(.{
            .name = "call_default",
            .zig_source_bytes =
            \\extern fn do_stuff() u32;
            \\pub fn main() void {
            \\    std.debug.print("default: {d}\n", .{do_stuff()});
            \\}
            \\const std = @import("std");
            \\
            ,
        });
        call_default.linkage = .dynamic;
        call_default.root_module.linkLibrary(libfoo);

        case.verifyObjdump(call_default.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--symbol-versions",
            "--only-section=.dynsym",
            "--only-symbol=do_stuff",
        }, .{ .sub_name = "call_default" });
        const run_call_default = case.addRunArtifact(call_default);
        run_call_default.expectStdErrEqual("default: 3\n");

        const call_explicit = case.addExecutable(.{
            .name = "call_explicit",
            .zig_source_bytes =
            \\extern fn @"do_stuff@FOO_V1"() u32;
            \\extern fn @"do_stuff@FOO_V2"() u32;
            \\extern fn @"do_stuff@FOO_V3"() u32;
            \\pub fn main() void {
            \\    std.debug.print("explicit V1: {d}\n", .{@"do_stuff@FOO_V1"()});
            \\    std.debug.print("explicit V2: {d}\n", .{@"do_stuff@FOO_V2"()});
            \\    std.debug.print("explicit V3: {d}\n", .{@"do_stuff@FOO_V3"()});
            \\}
            \\const std = @import("std");
            \\
            ,
        });
        call_explicit.linkage = .dynamic;
        call_explicit.root_module.linkLibrary(libfoo);

        case.verifyObjdump(call_explicit.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--symbol-versions",
            "--only-section=.dynsym",
            "--only-symbol=do_stuff",
        }, .{ .sub_name = "call_explicit" });
        const run_call_explicit = case.addRunArtifact(call_explicit);
        if (ctx.target.result.abi.isMusl()) {
            // musl's dynamic linker does not support GNU symbol versioning and will simply always
            // bind to the default version.
            run_call_explicit.expectStdErrEqual(
                \\explicit V1: 3
                \\explicit V2: 3
                \\explicit V3: 3
                \\
            );
        } else {
            run_call_explicit.expectStdErrEqual(
                \\explicit V1: 1
                \\explicit V2: 2
                \\explicit V3: 3
                \\
            );
        }

        const call_both = case.addExecutable(.{
            .name = "call_both",
            .zig_source_bytes =
            \\extern fn do_stuff() u32;
            \\extern fn @"do_stuff@FOO_V1"() u32;
            \\extern fn @"do_stuff@FOO_V2"() u32;
            \\extern fn @"do_stuff@FOO_V3"() u32;
            \\pub fn main() void {
            \\    std.debug.print("default: {d}\n", .{do_stuff()});
            \\    std.debug.print("explicit V1: {d}\n", .{@"do_stuff@FOO_V1"()});
            \\    std.debug.print("explicit V2: {d}\n", .{@"do_stuff@FOO_V2"()});
            \\    std.debug.print("explicit V3: {d}\n", .{@"do_stuff@FOO_V3"()});
            \\}
            \\const std = @import("std");
            \\
            ,
        });
        call_both.linkage = .dynamic;
        call_both.root_module.linkLibrary(libfoo);

        case.verifyObjdump(call_both.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--symbol-versions",
            "--only-section=.dynsym",
            "--only-symbol=do_stuff",
        }, .{ .sub_name = "call_both" });
        const run_call_both = case.addRunArtifact(call_both);
        if (ctx.target.result.abi.isMusl()) {
            // musl's dynamic linker does not support GNU symbol versioning and will simply always
            // bind to the default version.
            run_call_both.expectStdErrEqual(
                \\default: 3
                \\explicit V1: 3
                \\explicit V2: 3
                \\explicit V3: 3
                \\
            );
        } else {
            run_call_both.expectStdErrEqual(
                \\default: 3
                \\explicit V1: 1
                \\explicit V2: 2
                \\explicit V3: 3
                \\
            );
        }
    }

    if (ctx.includeTest("common_symbol")) |case| {
        const foo = case.addObject(.{
            .name = "foo",
            .asm_source_bytes =
            \\.comm something, 0x40, 0x10
            \\
            ,
        });
        const bar = case.addObject(.{
            .name = "bar",
            .asm_source_bytes =
            \\// This weak definition should be overriden by the common definition
            \\.weak something
            \\something:
            \\.word 0
            \\
            ,
        });
        const qux = case.addObject(.{
            .name = "qux",
            .asm_source_bytes =
            \\.comm something, 0x100, 0x4
            \\
            ,
        });

        const combined_obj = case.addObject(.{ .name = "combined" });
        combined_obj.root_module.addObject(foo);
        combined_obj.root_module.addObject(bar);
        combined_obj.root_module.addObject(qux);
        case.verifyObjdump(combined_obj.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--only-symbol=something",
        }, .{ .sub_name = "obj" });

        const combined_dynlib = case.addLibrary(.dynamic, .{ .name = "combined" });
        combined_dynlib.root_module.addObject(foo);
        combined_dynlib.root_module.addObject(bar);
        combined_dynlib.root_module.addObject(qux);
        case.verifyObjdump(combined_dynlib.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--only-symbol=something",
        }, .{ .sub_name = "dynlib" });
    }

    if (ctx.includeTest("override_common_symbol")) |case| {
        const foo = case.addObject(.{
            .name = "foo",
            .asm_source_bytes =
            \\.comm something, 0x40, 0x10
            \\
            ,
        });
        const bar = case.addObject(.{
            .name = "bar",
            .asm_source_bytes =
            \\// This strong definition should override the common definition
            \\.global something
            \\.set something, 0x1234
            \\
            ,
        });
        const qux = case.addObject(.{
            .name = "qux",
            .asm_source_bytes =
            \\.comm something, 0x100, 0x4
            \\
            ,
        });

        const combined_obj = case.addObject(.{ .name = "combined" });
        combined_obj.root_module.addObject(foo);
        combined_obj.root_module.addObject(bar);
        combined_obj.root_module.addObject(qux);
        case.verifyObjdump(combined_obj.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--only-symbol=something",
        }, .{ .sub_name = "obj" });

        const combined_dynlib = case.addLibrary(.dynamic, .{ .name = "combined" });
        combined_dynlib.root_module.addObject(foo);
        combined_dynlib.root_module.addObject(bar);
        combined_dynlib.root_module.addObject(qux);
        case.verifyObjdump(combined_dynlib.getEmittedBin(), &.{
            "-s",
            "--symbols",
            "--only-symbol=something",
        }, .{ .sub_name = "dynlib" });
    }

    if (ctx.includeTest("init_array_order")) |case| {
        const arrays = case.addExecutable(.{
            .name = "arrays",

            .zig_source_bytes =
            \\export fn init_00000() void { std.debug.print("init_00000\n", .{}); }
            \\export fn init_65535() void { std.debug.print("init_65535\n", .{}); }
            \\export fn init_default() void { std.debug.print("init_default\n", .{}); }
            \\export fn init_60000() void { std.debug.print("init_60000\n", .{}); }
            \\export fn init_00100() void { std.debug.print("init_00100\n", .{}); }
            \\
            \\export fn fini_00000() void { std.debug.print("fini_00000\n", .{}); }
            \\export fn fini_65535() void { std.debug.print("fini_65535\n", .{}); }
            \\export fn fini_default() void { std.debug.print("fini_default\n", .{}); }
            \\export fn fini_60000() void { std.debug.print("fini_60000\n", .{}); }
            \\export fn fini_00100() void { std.debug.print("fini_00100\n", .{}); }
            \\
            \\export fn preinit() void { std.debug.print("preinit\n", .{}); }
            \\
            \\pub fn main() void { std.debug.print("main\n", .{}); }
            \\
            \\const std = @import("std");
            \\
            ,

            .asm_source_bytes =
            \\.global init_00000
            \\.global init_65535
            \\.global init_default
            \\.global init_60000
            \\.global init_00100
            \\
            \\.global fini_00000
            \\.global fini_65535
            \\.global fini_default
            \\.global fini_60000
            \\.global fini_00100
            \\
            \\.global preinit
            \\
            \\.section .init_array.00000, "aw"
            \\.balign 8
            \\.quad init_00000
            \\.section .init_array.65535, "aw"
            \\.balign 8
            \\.quad init_65535
            \\.section .init_array, "aw"
            \\.balign 8
            \\.quad init_default
            \\.section .init_array.60000, "aw"
            \\.balign 8
            \\.quad init_60000
            \\.section .init_array.00100, "aw"
            \\.balign 8
            \\.quad init_00100
            \\
            \\.section .fini_array.00000, "aw"
            \\.balign 8
            \\.quad fini_00000
            \\.section .fini_array.65535, "aw"
            \\.balign 8
            \\.quad fini_65535
            \\.section .fini_array, "aw"
            \\.balign 8
            \\.quad fini_default
            \\.section .fini_array.60000, "aw"
            \\.balign 8
            \\.quad fini_60000
            \\.section .fini_array.00100, "aw"
            \\.balign 8
            \\.quad fini_00100
            \\
            \\.section .preinit_array, "aw"
            \\.balign 8
            \\.quad preinit
            \\
            ,
        });

        const run_arrays = case.addRunArtifact(arrays);
        if (!ctx.link_libc) {
            // Zig's freestanding entry point doesn't call preinit or fini.
            run_arrays.expectStdErrEqual(
                \\init_00000
                \\init_00100
                \\init_60000
                \\init_65535
                \\init_default
                \\main
                \\
            );
        } else if (ctx.target.result.abi.isMusl()) {
            // musl libc doesn't call preinit.
            run_arrays.expectStdErrEqual(
                \\init_00000
                \\init_00100
                \\init_60000
                \\init_65535
                \\init_default
                \\main
                \\fini_default
                \\fini_65535
                \\fini_60000
                \\fini_00100
                \\fini_00000
                \\
            );
        } else if (ctx.target.result.abi.isGnu()) {
            // glibc does everything.
            run_arrays.expectStdErrEqual(
                \\preinit
                \\init_00000
                \\init_00100
                \\init_60000
                \\init_65535
                \\init_default
                \\main
                \\fini_default
                \\fini_65535
                \\fini_60000
                \\fini_00100
                \\fini_00000
                \\
            );
        } else {
            unreachable;
        }
    }
}

const LinkContext = @import("../tests.zig").LinkContext;
const std = @import("std");
