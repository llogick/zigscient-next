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
        if (ctx.target.result.abi.isMusl()) {
            // TODO: Elf2 currently fails to link a dynamic executable on musl without PIC, citing
            // illegal dynamic relocations. This is most likely due to Elf2 not yet supporting
            // canonical PLT entries.
            break :case;
        }

        const lib = case.addLibrary(.dynamic, .{
            .name = "lib",
            .zig_source_bytes =
            \\export var my_favourite_number: u32 = 42;
            \\
            ,
        });

        const exe = case.addExecutable(.{
            .name = "test",
            .pic = false, // force PDC so that we need a copy reloc
            .zig_source_bytes =
            \\extern var my_favourite_number: u32;
            \\pub fn main() void {
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
        if (ctx.target.result.abi.isMusl()) {
            // TODO: Elf2 currently fails to link a dynamic executable on musl without PIC, citing
            // illegal dynamic relocations. This is most likely due to Elf2 not yet supporting
            // canonical PLT entries.
            break :case;
        }

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
        if (ctx.target.result.abi.isMusl()) {
            // TODO: Elf2 currently fails to link a dynamic executable on musl without PIC, citing
            // illegal dynamic relocations. This is most likely due to Elf2 not yet supporting
            // canonical PLT entries.
            break :case;
        }

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
        run_call_explicit.expectStdErrEqual(
            \\explicit V1: 1
            \\explicit V2: 2
            \\explicit V3: 3
            \\
        );

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
        run_call_both.expectStdErrEqual(
            \\default: 3
            \\explicit V1: 1
            \\explicit V2: 2
            \\explicit V3: 3
            \\
        );
    }
}

const LinkContext = @import("../tests.zig").LinkContext;
const std = @import("std");
