//! # Introduction
//! This is a test harness that should be used for tests that involve invoking the compiler to
//! compile code and checking how the compiler behaves.  This is different from unit tests which
//! test how code behaves thereby indirectly checking the compiler behavior.  Compiler CLI tests are
//! also currently excluded because this file exclusively uses the compiler protocol for invoking
//! the compiler, but that is a feature that could be added here if the build system becomes
//! insufficient for running these tests.
//!
//! # Anatomy of a Test Suite
//! At the top level, the `test` directory of this repo contains a set of test suites that make use
//! of this runner.  The main thing that distinguishes one test suite from another is the
//! `test/<suite>/matrix.zon` file which determines the set of "targets" on which each test in the
//! suite is run.  Note that in this case, target refers to more than just what is given to the
//! `-target` command line argument, but also any command line option that is desirable to test
//! different variations of in the same concrete test.
//!
//! Within a test suite, there are a collection of test cases.  Each file or directory inside the
//! test suite directory, ignoring `matrix.zon` and various editor temporary files, represents a
//! separate test case.  The main description of a test case resides within the manifest file, which
//! is located at either `test/<suite>/<case>` or `test/<suite>/<case>/manifest`, referred to as
//! file cases and directory cases respectively.  File cases are entirely self-contained within a
//! single file and will be more amenable to direct fuzzing of that file in the future.  Directory
//! cases have the same syntax as file cases, with an additional feature that allows splitting the
//! information about the test case among multiple files, all under the directory representing the
//! test case.  This is useful if those files are large or otherwise more difficult to understand
//! when embedded inside the manifest.  This gives us immediate `zig fmt` and editor highlighting
//! support of these separate files, although that benefit need not preclude adding support for
//! having these features for the manifest file format itself, for example with a tool that passes
//! fragments of the file to `zig fmt` and reassembles the results or with editor highlighting
//! support that uses different sub-highlighters for different regions of the file.
//!
//! # Format of the Test Manifest
//! The test manifest is an imperative sequence of instructions for this runner to execute.  This
//! makes it simple to conceptualize what exactly the runner is doing when reading a manifest file.
//! The outermost format of the file is a sequence of empty lines and commands.  Each command has
//! the format `#<cmd> <arg> <contents>`, where the line begins with a literal `#` character,
//! followed by `<cmd>`, the name of the command, followed by two optional space-separated
//! arguments.  The interpretation of `<arg>`, the first argument, depends on the command in
//! question. That, and whether it is required present, required absent, or optional, is documented
//! with each separate command.  Finally `<contents>`, the second argument, has a common syntax
//! shared with every command, and does not require `<arg>` to be present.  The syntax of
//! `<contents>` has 4 possible forms:
//!
//! 1. Empty: When missing it represents an empty file.
//! 2. String: When it is a valid zig string literal it represents a file with the contents of that
//!    string.
//! 3. Inline: When it matches `{`, it represents the rest of the manifest up to the next occurance
//!    of `#}`.  Newlines are manditory after the braces, and a newline before the `#}` indicates
//!    whether the represented file ends with a newline.
//! 4. File: When it uses the syntax `@path/to/file`, it requires a directory case and represents
//!    the file identified by the path which is interpreted relative to the directory representing
//!    the case.
//!
//! Here are some equivalent examples of a command having empty contents:
//!
//!     #cmd
//!
//!     #cmd ""
//!
//!     #cmd {
//!     #}
//!
//!     #cmd @empty
//!
//! Where the final example depends on the file created by the shell command
//! `truncate --size=0 empty` in the case directory.  The first example shows the preferred form for
//! representing empty contents.
//!
//! Here are some equivalent examples that represent `Hello, world!` without any newlines:
//!
//!     #cmd "Hello, world!"
//!
//!     #cmd {
//!     Hello, world!#}
//!
//!     #cmd @hello.txt
//!
//! Where the final example depends on the file created by the shell command
//! `printf %s "Hello, world!" > hello.txt` in the case directory.  The first example shows the
//! preferred form for representing short contents with no newlines.
//!
//! Here are some equivalent examples that represent 5 lines, with each line containing a sequence
//! of integers starting at 1:
//!
//!     #cmd {
//!     1
//!     2
//!     3
//!     4
//!     5
//!     #}
//!
//!     #cmd "1\n2\n3\n4\n5\n"
//!
//!     #cmd @numbers.seq
//!
//! Where the final example depends on the file created by the shell command `seq 5 > numbers.seq`
//! in the case directory.  The first example shows the preferred form for representing short to
//! medium contents containing newlines.
//!
//! Finally, the file form is preferred for representing long contents.
//!
//! # Manifest Command List
//! Manifests support the following commands:
//!
//! The empty command indicates a comment that can be used to describe other commands in the
//! manifest, for example:
//!
//!     # "the following targets are skipped until feature X is implemented"
//!
//! The skip command allows skipping a given target in the test matrix for the current case. These
//! must appear at the top of the file before any non-comment commands, for example:
//!
//!     #skip x86_64-linux-none-baseline-whole-selfhosted-new-pic
//!
//! The require command allows reducing the matrix for a given test based on the given condition.
//! These must appear at the top of the file before any non-comment commands, for example:
//!
//!     #require windows
//!
//! * `no-matrix` skips all but the first matrix entry.
//! * `safe-unwind` requires that a safe unwind strategy, like DWARF unwinding, is available.
//! * `wasm` requires a target cpu arch tag of `.wasm*`.
//! * `windows` requires a target os tag of `.uefi` or `.windows`.
//!
//! The zig command defines the compiler command line arguments that are specific to this case,
//! where the command argument chooses the compiler sub-command and the contents contains options
//! that can override options selected by the target, for example:
//!
//!     #zig build-exe {
//!     --name name -fomit-frame-pointer
//!     --dep mod1 -Mroot=root.zig
//!     --dep mod2 -Mmod1=mod1.zig
//!     -Mmod2=mod2.zig
//!     #}
//!
//! The write command creates a file in a per-case per-target temporary directory that is used as
//! the compiler process working directory, and so can be referenced by the commiler command line,
//! for example:
//!
//!     #write hello.zig {
//!     const std = @import("std");
//!
//!     pub fn main() void {
//!         std.debug.print("Hello, World!\n");
//!     }
//!     #}
//!
//! The delete command deletes a previously written file, which is different from writing an empty
//! file, for example:
//!
//!     #delete existing.zig
//!
//! The update command runs a compiler update with the current state of the written source files,
//! where the contents are an informative string shown by progress reporting and error messages, and
//! can be empty to let this runner choose one for you, for example:
//!
//!     #update "decl deleted"
//!
//! Finally, the check command actually tests the compiler behavior in some way, where the command
//! argument chooses what kind of check to run, for example:
//!
//!     #check errors {
//!     test.zig:4:5: error: use of undeclared identifier 'foo'
//!     test.zig:7:17: error: expected type 'type', found 'comptime_int'
//!     #}
//!
//! # Available checks
//! The errors check succeeds when the compiler produces the given compiler errors.  The contents
//! suppport some patterns that are not matched exactly:
//!
//! * `builtin.zig:?:?:` matches a path to a builtin source file and any line or column number.
//! * `lib/std/**.zig:?:?:` matches a path to a source file located in the standard library and any
//!   line or column number.
//! * `lib/std/**.zig:?:?: note: called inline here` matches zero or more `called inline here`
//!   errors in the standard library.
//!
//! Empty contents succeed when the compiler produces no compile errors.
//!
//! The stdout check succeeds when running the binary produced by the compiler, possibly with an
//! executor like qemu, exits with code 0 and produces the given standard output.  The contents are
//! a sequence of matches separated by `#...` that are expected to appear in the order given, with
//! arbitrary contents.
//!
//! The exit check succeeds when running the binary as the stdout check does exits with the given
//! exit code.  The contents are the decimal exit code in ascii with no trailing newline.
//!
//! The error-trace, panic, and stack-trace checks succeed when running the binary as the stdout
//! check does exits with 1, abort, or 0 respectively, produces no standard output, and produces the
//! given standard error.  The contents support a few useful patterns that are not matched exactly:
//!
//! * `thread ? panic: ` matches a panic message for any thread id.
//! * `test.zig:` matches a path ending with `test.zig` as the final component followed by a `:`.
//! * `lib/std/debug.zig:?:?:` matches a path ending with `lib/std/debug.zig` as the final three
//!   components, followed by any line or column number.
//! * `0x?` matches any hex address.
//! * `[__func_?]` matches either ``, or `__func_12345` for some decimal integer `12345`.
//! * ` (test*)` at the end of a line matches either ` (test)`, ` (test_zcu.obj)`, or ` (test.zig)`.
//! * ` (std*)`  at the end of a line matches either ` (test)`, ` (test_zcu.obj)`, or ` (std.zig)`.
//!
//! The llvm check succeeds when the llvm backend produces the given LLVM IR.  The contents support
//! a variety of useful patterns that are not matched exactly:
//!
//! * `...` at the end of a line matches any possibly empty sequence of lines.
//! * `... ` matches a possibly empty sequence of attributes.
//! * `..., ` matches a possibly empty sequence of parameters or operands.
//! * `  ...` at the end of a line matches any possibly empty sequence of lines not starting with
//!   `}`, which avoids accidentally escaping the current function.
//! * `, !...` at the end of a line matches any possibly empty sequence of debug attachments of the
//!   form `, !name !0`.
//! * `?` matches an unknown integer constant.
//! * `isize` matches any of the llvm types that Zig's `usize`/`isize` can lower to.
//! * `!!` matches a literal `!`.
//! * `!<id>` matches anything of the form `!0` the first time that id is used.  Later uses of
//!   the same id match exactly what the first id matched.
//! * `%%` matches a literal `%`.
//! * `%<id>` matches anything of the form `%"quoted name"` or `%name` or `%0` the first time that
//!   id is used.  Later uses of the same id match exactly what the first id matched.
//! * `^^` matches a literal `^`.
//! * `^<id>` matches anything of the form `^0` the first time that id is used.  Later uses of
//!   the same id match exactly what the first id matched.
//!
//! In the above descriptions, `<id>` uses the same syntax as non-quoted non-numeric llvm
//! identifiers. Numeric identifiers are excluded because they should not be reused in other
//! functions.
//!
//! The gdb and lldb checks succeed when running the binary under a debugger produces the given
//! standard output like the stdout check.  The commands to be executed by the debugger are
//! extracted from the contents by looking for lines starting with `(gdb) ` or `(lldb) `
//! respectively.
//!
//! # Running a Case Individually and Other Fun Tricks
//! This runner is primarily intended to be invoked by the build system, and uses the test protocol
//! for scheduling test runs and communicating test results.  However, sometimes it is much easier
//! to debug a failing test if it did not depend on the entire build system, and so this runner can
//! also be invoked directly, for example:
//!
//!     $ ./runner --zig=path/to/zig --lib=path/to/lib --keep-src -fqemu \
//!     --target aarch64-linux-none-baseline-whole-llvm-lld-pic \
//!     --target aarch64-linux-gnu-baseline-whole-llvm-lld-pic path/to/case
//!
//! The resulting binary can be examined or debugged by locating it in a `src_*` directory in the
//! current working directory after passing `--keep-src`.  Note also that when using the build
//! system, `-Dtest-filter` filters the case name and `-Dtest-target-filter` filters the target.

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
    targets: std.array_hash_map.Custom(u32, Target.State, void, true),
    target_bytes: std.ArrayList(u8),

    libc_runtimes_dir: ?std.Io.Dir,
    enable_darling: bool,
    enable_qemu: bool,
    enable_rosetta: bool,
    enable_wasmtime: bool,
    enable_wine: bool,

    gdb_exe: ?std.Io.File,
    lldb_exe: ?std.Io.File,

    const TargetAdapter = struct {
        bytes: []const u8,

        pub fn hash(_: @This(), key: []const u8) u32 {
            return @truncate(std.hash.Wyhash.hash(0, key));
        }

        pub fn eql(ctx: @This(), lhs_key: []const u8, rhs_key: u32, _: usize) bool {
            return std.mem.eql(u8, lhs_key, std.mem.sliceTo(ctx.bytes[rhs_key..], 0));
        }
    };

    fn deinit(args: *Args, gpa: std.mem.Allocator) void {
        args.targets.deinit(gpa);
        args.target_bytes.deinit(gpa);
        args.* = undefined;
    }

    fn addTarget(
        args: *Args,
        gpa: std.mem.Allocator,
        target: []const u8,
        skipped: bool,
    ) (std.mem.Allocator.Error || error{DuplicatedTarget})!void {
        try args.target_bytes.ensureUnusedCapacity(gpa, target.len + 1);
        const gop = try args.targets.getOrPutAdapted(gpa, target, TargetAdapter{
            .bytes = args.target_bytes.items,
        });
        if (gop.found_existing) return error.DuplicatedTarget;
        gop.key_ptr.* = @intCast(args.target_bytes.items.len);
        gop.value_ptr.* = .{ .skipped = skipped };
        args.target_bytes.appendSliceAssumeCapacity(target);
        args.target_bytes.appendAssumeCapacity(0);
    }

    fn getTargetIndex(args: *const Args, target: []const u8) ?usize {
        return args.targets.getIndexAdapted(target, TargetAdapter{ .bytes = args.target_bytes.items });
    }

    fn getTarget(args: *const Args, index: usize) []const u8 {
        return std.mem.sliceTo(args.target_bytes.items[args.targets.keys()[index]..], 0);
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

fn dirPath(
    runner: *Runner,
    dir: std.Io.Dir,
) (std.Io.Dir.RealPathError || std.mem.Allocator.Error)![]const u8 {
    const arena = runner.arena.allocator();
    const path_buffer = try arena.alloc(u8, std.Io.Dir.max_path_bytes);
    const path_len = try dir.realPath(runner.io, path_buffer);
    return arena.remap(path_buffer, path_len) orelse path_buffer[0..path_len];
}
fn filePath(
    runner: *Runner,
    file: std.Io.File,
) (std.Io.File.RealPathError || std.mem.Allocator.Error)![]const u8 {
    const arena = runner.arena.allocator();
    const path_buffer = try arena.alloc(u8, std.Io.Dir.max_path_bytes);
    const path_len = try file.realPath(runner.io, path_buffer);
    return arena.remap(path_buffer, path_len) orelse path_buffer[0..path_len];
}

pub const ProtocolError = error{
    ServerReadFailed,
    ServerWriteFailed,
    ServerEndOfStream,
} || FailError;
pub fn runServer(runner: *Runner) ProtocolError!void {
    runner.server.serveStringMessage(
        .zig_version,
        builtin.zig_version_string,
    ) catch |err| switch (err) {
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
            .exit => return std.process.cleanExit(runner.io),
            .args => {
                const args_body = try arena.alloc(u8, hdr.bytes_len);
                runner.server.in.readSliceAll(args_body) catch unreachable;
                const State = enum {
                    positional,
                    @"--log",
                    @"--zig",
                    @"--lib",
                    @"--src",
                    @"--target",
                    @"--skipped-target",
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
                    .@"--log" = .string,
                    .@"--zig" = .file,
                    .@"--lib" = .dir,
                    .@"--src" = .dir,
                    .@"--target" = .string,
                    .@"--skipped-target" = .string,
                    .@"--libc-runtimes" = .dir,
                    .@"--gdb" = .file,
                    .@"--lldb" = .file,
                });
                var duplicated_target = false;
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
                                .positional => if (std.mem.eql(u8, string, "--log")) {
                                    state = .@"--log";
                                    continue;
                                } else if (std.mem.cutPrefix(u8, string, "--log=")) |log_level| {
                                    global_log_level = std.meta.stringToEnum(
                                        std.log.Level,
                                        log_level,
                                    ) orelse return runner.fail("invalid log level {q}", .{log_level});
                                } else if (std.mem.eql(u8, string, "--target")) {
                                    state = .@"--target";
                                    continue;
                                } else if (std.mem.cutPrefix(u8, string, "--target=")) |target| {
                                    runner.args.addTarget(
                                        runner.gpa,
                                        target,
                                        false,
                                    ) catch |err| switch (err) {
                                        error.OutOfMemory => |e| return e,
                                        error.DuplicatedTarget => switch (runner.fail(
                                            "duplicated target {q}",
                                            .{target},
                                        )) {
                                            error.Canceled, error.OutOfMemory => |e| return e,
                                            error.AlreadyReported => duplicated_target = true,
                                        },
                                    };
                                } else if (std.mem.eql(u8, string, "--skipped-target")) {
                                    state = .@"--skipped-target";
                                    continue;
                                } else if (std.mem.cutPrefix(
                                    u8,
                                    string,
                                    "--skipped-target=",
                                )) |target| {
                                    runner.args.addTarget(
                                        runner.gpa,
                                        target,
                                        true,
                                    ) catch |err| switch (err) {
                                        error.OutOfMemory => |e| return e,
                                        error.DuplicatedTarget => switch (runner.fail(
                                            "duplicated target {q}",
                                            .{target},
                                        )) {
                                            error.Canceled, error.OutOfMemory => |e| return e,
                                            error.AlreadyReported => duplicated_target = true,
                                        },
                                    };
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
                                },
                                .@"--log" => {
                                    global_log_level = std.meta.stringToEnum(
                                        std.log.Level,
                                        string,
                                    ) orelse return runner.fail("invalid log level {q}", .{string});
                                },
                                .@"--target" => {
                                    runner.args.addTarget(
                                        runner.gpa,
                                        string,
                                        false,
                                    ) catch |err| switch (err) {
                                        error.OutOfMemory => |e| return e,
                                        error.DuplicatedTarget => switch (runner.fail(
                                            "duplicated target {q}",
                                            .{string},
                                        )) {
                                            error.Canceled, error.OutOfMemory => |e| return e,
                                            error.AlreadyReported => duplicated_target = true,
                                        },
                                    };
                                },
                                .@"--skipped-target" => {
                                    runner.args.addTarget(
                                        runner.gpa,
                                        string,
                                        true,
                                    ) catch |err| switch (err) {
                                        error.OutOfMemory => |e| return e,
                                        error.DuplicatedTarget => switch (runner.fail(
                                            "duplicated target {q}",
                                            .{string},
                                        )) {
                                            error.Canceled, error.OutOfMemory => |e| return e,
                                            error.AlreadyReported => duplicated_target = true,
                                        },
                                    };
                                },
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
                            debug: {
                                std.log.debug("inherited parent dir {t}={q}", .{
                                    state,
                                    runner.dirPath(dir) catch |err| switch (err) {
                                        error.Canceled, error.OutOfMemory => |e| return e,
                                        else => break :debug,
                                    },
                                });
                            }
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
                            debug: {
                                std.log.debug("inherited parent file {t}={q}", .{
                                    state,
                                    runner.filePath(file) catch |err| switch (err) {
                                        error.Canceled, error.OutOfMemory => |e| return e,
                                        else => break :debug,
                                    },
                                });
                            }
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
                if (duplicated_target) return error.AlreadyReported;
            },
            .query_test_metadata => runner.server.serveTestMetadata(.{
                .names = runner.args.targets.keys(),
                .string_bytes = runner.args.target_bytes.items,
            }) catch |err| switch (err) {
                error.WriteFailed => return error.ServerWriteFailed,
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
            const start = if (std.mem.findScalarLast(u8, written[0..s.interface.seek], '\n')) |nl_pos|
                nl_pos + 1
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
    fn srcInfo(s: *SourceReader, src_loc: [*]const u8) struct {
        line: u32,
        column: u32,
        source_line: []const u8,
    } {
        const buffer = s.interface.buffer[0..s.interface.end];
        const pos = src_loc - buffer.ptr;
        const end = std.mem.findScalarPos(u8, buffer, pos, '\n') orelse buffer.len;
        const start =
            if (std.mem.findScalarLast(u8, buffer[0..end], '\n')) |nl_pos| nl_pos + 1 else 0;
        return .{
            .line = @intCast(s.line + std.mem.countScalar(u8, buffer[0..start], '\n')),
            .column = @intCast(pos - start),
            .source_line = buffer[start..end],
        };
    }
    fn srcLoc(
        s: *SourceReader,
        src_loc: []const u8,
    ) std.mem.Allocator.Error!std.zig.ErrorBundle.SourceLocationIndex {
        const src_info = s.srcInfo(src_loc.ptr);
        return s.eb_wip.addSourceLocation(.{
            .src_path = s.src_path,
            .line = src_info.line,
            .column = src_info.column,
            .span_start = src_info.column,
            .span_main = src_info.column,
            .span_end = @intCast(src_info.column + src_loc.len),
            .source_line = try s.eb_wip.addString(src_info.source_line),
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

const newline: []const Pattern = &.{
    .{ .@"?" = &.{
        .{ .literal = "\r" },
    } },
    .{ .literal = "\n" },
};

const Update = struct {
    runner: *Runner,
    test_index: u32,
    target: Compiler.Target,
    skipped: ?union(enum) {
        no,
        yes_skip,
        yes_require: struct { tag: Requirement, negate: bool },
    },
    invalid_skip: bool,
    manifest_sr: SourceReader,
    compiler: ?Compiler,
    src_dir: std.Io.Dir,
    cmd_src_loc: std.zig.ErrorBundle.SourceLocationIndex,
    contents_src_loc: struct { path: std.zig.ErrorBundle.String, line: u32 = 0, column: u32 = 0 },
    src_loc: ?std.zig.ErrorBundle.SourceLocationIndex,
    num: usize,
    name: []const u8,
    prog_node: std.Progress.Node,
    mtime: std.Io.Timestamp,

    fn checkPattern(
        update: *const Update,
        what: []const u8,
        expected: []const u8,
        actual: []const u8,
        patterns: []const struct {
            []const u8,
            union(enum) { anon, name: *const Pattern.Compiled },
            []const Pattern,
        },
    ) FailError!void {
        const gpa = update.runner.gpa;
        var captures: std.array_hash_map.String(usize) = .empty;
        defer captures.deinit(gpa);
        var expected_pattern: Pattern.Compiled = .empty;
        defer expected_pattern.deinit(gpa);
        var literal_start: usize = 0;
        var literal_end: usize = 0;
        while (expected.len - literal_end > 0) {
            for (patterns) |pattern| {
                if (!std.mem.startsWith(u8, expected[literal_end..], pattern[0])) continue;
                const name_start = literal_end + pattern[0].len;
                const maybe_name, const name_end = name: switch (pattern[1]) {
                    .anon => .{ null, name_start },
                    .name => |name_pattern| {
                        switch (try name_pattern.matches(gpa, expected[name_start..], .start)) {
                            .expected => return update.failContents(.{
                                .context = expected,
                                .start = @intCast(literal_end),
                                .main = @intCast(name_start),
                            }, "{s} {q} pattern missing name", .{ what, pattern[0] }),
                            .matched => |name_len| break :name .{
                                expected[name_start..][0..name_len],
                                name_start + name_len,
                            },
                        }
                    },
                };
                if (literal_end - literal_start > 0) try expected_pattern.append(gpa, &.{
                    .{ .literal = expected[literal_start..literal_end] },
                });
                literal_end = name_end;
                if (maybe_name) |name| {
                    const capture_gop = try captures.getOrPut(gpa, name);
                    if (capture_gop.found_existing) try expected_pattern.append(gpa, &.{
                        .{ .reference = capture_gop.value_ptr.* },
                    }) else {
                        capture_gop.value_ptr.* = expected_pattern.captures.items.len;
                        try expected_pattern.append(gpa, &.{
                            .{ .capture = pattern[2] },
                        });
                    }
                } else try expected_pattern.append(gpa, pattern[2]);
                literal_start = literal_end;
                break;
            } else literal_end += 1;
        }
        if (literal_end - literal_start > 0) try expected_pattern.append(gpa, &.{
            .{ .literal = expected[literal_start..literal_end] },
        });
        switch (try expected_pattern.matches(gpa, actual, .all)) {
            .matched => {},
            .expected => |failure| {
                std.testing.failPrint("\n====== expected this output: =========\n", .{});
                std.testing.failPrintWithVisibleNewlines(expected);
                std.testing.failPrint("\n======== instead found this: =========\n", .{});
                std.testing.failPrintWithVisibleNewlines(actual);
                std.testing.failPrint("\n======================================\n", .{});

                const diff_line_number = 1 + std.mem.countScalar(u8, actual[0..failure.index], '\n');
                std.testing.failPrint("First difference occurs on line {d}:\n", .{diff_line_number});

                std.testing.failPrint("expected:\n", .{});
                switch (failure.what) {
                    .any, .none => |acs, tag| if (acs.eql(.empty)) std.testing.failPrint("/{s}/\n", .{
                        switch (tag) {
                            else => unreachable,
                            .any => "(?!)",
                            .none => ".",
                        },
                    }) else {
                        std.testing.failPrint("/[{s}", .{switch (tag) {
                            else => unreachable,
                            .any => "",
                            .none => "^",
                        }});
                        var remaining = acs;
                        while (remaining.findFirstSet()) |start| {
                            remaining.toggleAll();
                            remaining.setRangeValue(.{ .start = 0, .end = start }, false);
                            const end = remaining.findFirstSet() orelse
                                Pattern.AsciiCharSet.bit_length;
                            remaining.toggleAll();
                            remaining.setRangeValue(.{ .start = 0, .end = end }, false);
                            switch (end - start) {
                                0 => unreachable,
                                1 => std.testing.failPrint("{c}", .{@as(u8, @intCast(start))}),
                                else => std.testing.failPrint("{c}-{c}", .{
                                    @as(u8, @intCast(start)),
                                    @as(u8, @intCast(end - 1)),
                                }),
                            }
                        }
                        std.testing.failPrint("]/\n", .{});
                    },
                    .literal => |literal| std.testing.failPrint("{q}\n", .{literal}),
                }

                std.testing.failPrint("found:\n", .{});
                std.testing.failPrintIndicatorLine(actual, failure.index);

                return update.fail("{s} did not match expected", .{what});
            },
        }
    }

    fn fail(update: *const Update, comptime fmt: []const u8, args: anytype) FailError {
        return update.failString(try update.runner.eb_wip.printString(fmt, args));
    }
    fn failString(update: *const Update, msg: std.zig.ErrorBundle.String) FailError {
        const eb_wip = &update.runner.eb_wip;
        var notes_buffer: [1]std.zig.ErrorBundle.ErrorMessage = undefined;
        var notes_len: u32 = 0;
        if (update.src_loc) |src_loc| {
            notes_buffer[notes_len] = .{
                .msg = try eb_wip.addString("after this update"),
                .src_loc = src_loc,
            };
            notes_len += 1;
        }
        try eb_wip.addRootErrorMessageWithNotes(.{
            .msg = msg,
            .src_loc = update.cmd_src_loc,
            .notes_len = notes_len,
        }, notes_buffer[0..notes_len]);
        return error.AlreadyReported;
    }
    const ContentsSrc = struct {
        line: u32 = 0,
        context: []const u8,
        start: ?u32 = null,
        main: u32,
        end: ?u32 = null,
    };
    fn failContents(
        update: *const Update,
        src: ContentsSrc,
        comptime fmt: []const u8,
        args: anytype,
    ) FailError {
        return update.failContentsString(src, try update.runner.eb_wip.printString(fmt, args));
    }
    fn failContentsString(
        update: *const Update,
        src: ContentsSrc,
        msg: std.zig.ErrorBundle.String,
    ) FailError {
        const eb_wip = &update.runner.eb_wip;
        var notes_buffer: [2]std.zig.ErrorBundle.ErrorMessage = undefined;
        var notes_len: u32 = 0;
        notes_buffer[notes_len] = .{
            .msg = try eb_wip.addString("during this command"),
            .src_loc = update.cmd_src_loc,
        };
        notes_len += 1;
        if (update.src_loc) |src_loc| {
            notes_buffer[notes_len] = .{
                .msg = try eb_wip.addString("after this update"),
                .src_loc = src_loc,
            };
            notes_len += 1;
        }
        const src_loc = std.zig.findLineColumn(src.context, src.main);
        try eb_wip.addRootErrorMessageWithNotes(.{
            .msg = msg,
            .src_loc = try eb_wip.addSourceLocation(.{
                .src_path = update.contents_src_loc.path,
                .line = @intCast(update.contents_src_loc.line + src.line + src_loc.line),
                .column = @intCast(update.contents_src_loc.column + src_loc.column),
                .span_start = src.start orelse src.main,
                .span_main = src.main,
                .span_end = src.end orelse src.main,
                .source_line = try eb_wip.addString(src_loc.source_line),
            }),
            .notes_len = notes_len,
        }, notes_buffer[0..notes_len]);
        return error.AlreadyReported;
    }
};

const TestError = error{
    SkipTest,
} || FailError;
fn testOne(runner: *Runner, test_index: u32) (error{ServerWriteFailed} || TestError)!void {
    const target_query = runner.args.getTarget(test_index);
    std.log.info("testing target {q}...", .{target_query});
    if (runner.args.targets.values()[test_index].skipped) {
        std.log.info("skipped by the build system", .{});
        return error.SkipTest;
    }
    const manifest_file = runner.args.manifest_file orelse
        return runner.fail("missing \"path/to/test\" arg", .{});
    var manifest_buffer: [512]u8 = undefined;
    var manifest_fr = manifest_file.reader(runner.io, &manifest_buffer);
    var update: Update = .{
        .runner = runner,
        .test_index = test_index,
        .target = target: {
            var component_it = std.mem.splitBackwardsScalar(u8, target_query, '-');
            var component = component_it.first();
            const strip: Target.Strip =
                if (std.meta.stringToEnum(Target.Strip, component)) |strip| strip: {
                    component = component_it.next() orelse return runner.fail(
                        "target {q} missing components before strip",
                        .{target_query},
                    );
                    break :strip strip;
                } else .default;
            const link_libc: Target.LinkLibc =
                if (std.meta.stringToEnum(Target.LinkLibc, component)) |link_libc| link_libc: {
                    component = component_it.next() orelse return runner.fail(
                        "target {q} missing components before link libc",
                        .{target_query},
                    );
                    break :link_libc link_libc;
                } else .default;
            const linkage: Target.Linkage =
                if (std.meta.stringToEnum(Target.Linkage, component)) |linkage| linkage: {
                    component = component_it.next() orelse return runner.fail(
                        "target {q} missing components before linkage",
                        .{target_query},
                    );
                    break :linkage linkage;
                } else .default;
            const linker: Target.Linker =
                if (std.meta.stringToEnum(Target.Linker, component)) |linker| linker: {
                    component = component_it.next() orelse return runner.fail(
                        "target {q} missing components before linker",
                        .{target_query},
                    );
                    break :linker linker;
                } else .default;
            const unwind_tables: Target.UnwindTables = if (std.meta.stringToEnum(
                Target.UnwindTables,
                component,
            )) |unwind_tables| unwind_tables: {
                component = component_it.next() orelse return runner.fail(
                    "target {q} missing components before unwind tables",
                    .{target_query},
                );
                break :unwind_tables unwind_tables;
            } else .default;
            const frame_pointers: Target.FramePointers = if (std.meta.stringToEnum(
                Target.FramePointers,
                component,
            )) |frame_pointers| frame_pointers: {
                component = component_it.next() orelse return runner.fail(
                    "target {q} missing components before frame pointers",
                    .{target_query},
                );
                break :frame_pointers frame_pointers;
            } else .default;
            const pic: Target.Pic = if (std.meta.stringToEnum(Target.Pic, component)) |pic| pic: {
                component = component_it.next() orelse return runner.fail(
                    "target {q} missing components before pic",
                    .{target_query},
                );
                break :pic pic;
            } else .default;
            const optimize = std.meta.stringToEnum(std.lang.Optimize, component) orelse
                return runner.fail("target {q} unknown optimize {q}", .{ target_query, component });
            component = component_it.next() orelse return runner.fail(
                "target {q} missing components before optimize",
                .{target_query},
            );
            const backend = std.meta.stringToEnum(Target.Backend, component) orelse
                return runner.fail("target {q} unknown backend {q}", .{ target_query, component });
            component = component_it.next() orelse return runner.fail(
                "target {q} missing components before backend",
                .{target_query},
            );
            const cache_mode = std.meta.stringToEnum(Target.CacheMode, component) orelse
                return runner.fail("target {q} unknown cache mode {q}", .{
                    target_query, component,
                });
            const triple_cpu = component_it.rest();
            const triple, const cpu = triple_cpu: {
                var cpu_pos: usize = 0;
                for (0..3) |_| {
                    const component_start = cpu_pos;
                    const dash_pos = std.mem.findScalarPos(u8, triple_cpu, cpu_pos, '-') orelse
                        return runner.fail("target {q} missing cpu component", .{target_query});
                    cpu_pos = dash_pos + 1;
                    if (std.mem.eql(u8, triple_cpu[component_start..dash_pos], "native")) break;
                }
                break :triple_cpu .{ triple_cpu[0 .. cpu_pos - 1], triple_cpu[cpu_pos..] };
            };
            var query_parse_diags: std.Target.Query.ParseOptions.Diagnostics = undefined;
            const resolved = std.zig.system.resolveTargetQuery(
                runner.io,
                std.Target.Query.parse(.{
                    .arch_os_abi = triple,
                    .cpu_features = cpu,
                    .object_format = switch (backend) {
                        .selfhosted, .llvm => null,
                        .cbe => "c",
                    },
                    .diagnostics = &query_parse_diags,
                }) catch |err| switch (err) {
                    error.UnknownArchitecture => return runner.fail(
                        "unable to resolve target {q}: unknown architecture: {q}",
                        .{ target_query, query_parse_diags.unknown_architecture_name.? },
                    ),
                    error.MissingOperatingSystem => return runner.fail(
                        "unable to resolve target {q}: missing operating system",
                        .{target_query},
                    ),
                    error.UnknownOperatingSystem => return runner.fail(
                        "unable to resolve target {q}: unknown operating system: {q}",
                        .{ target_query, query_parse_diags.os_name.? },
                    ),
                    error.InvalidOperatingSystemVersion => return runner.fail(
                        "unable to resolve target {q}: invalid operating system version",
                        .{target_query},
                    ),
                    error.UnknownApplicationBinaryInterface => return runner.fail(
                        "unable to resolve target {q}: unknown abi",
                        .{target_query},
                    ),
                    error.InvalidAbiVersion => return runner.fail(
                        "unable to resolve target {q}: invalid abi version",
                        .{target_query},
                    ),
                    error.UnexpectedExtraField => unreachable, // validated above
                    error.UnknownCpuModel => return runner.fail(
                        "unable to resolve target {q}: unknown cpu: {q}",
                        .{ target_query, query_parse_diags.cpu_name.? },
                    ),
                    error.UnknownCpuFeature => return runner.fail(
                        "unable to resolve target {q}: unknown cpu feature: {q}",
                        .{ target_query, query_parse_diags.unknown_feature_name.? },
                    ),
                    error.UnknownObjectFormat => return runner.fail(
                        "unable to resolve target {q}: unknown object format",
                        .{target_query},
                    ),
                },
            ) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| return runner.fail(
                    "unable to resolve target {q}: {t}",
                    .{ triple, e },
                ),
            };
            break :target .{
                .triple = triple,
                .resolved = resolved,
                .cache_mode = cache_mode,
                .backend = backend,
                .optimize = optimize,
                .pic = pic,
                .default_frame_pointers = frame_pointers == .default,
                .frame_pointers = frame_pointers.resolve(&resolved, optimize),
                .default_unwind_tables = unwind_tables == .default,
                .unwind_tables = unwind_tables.resolve(&resolved),
                .linker = linker,
                .linkage = linkage,
                .link_libc = link_libc,
                .strip = strip,
            };
        },
        .skipped = null,
        .invalid_skip = false,
        .manifest_sr = .init(
            &manifest_fr.interface,
            runner.gpa,
            &runner.eb_wip,
            try runner.eb_wip.addString(src_path: {
                break :src_path runner.filePath(manifest_file) catch |err| switch (err) {
                    error.Canceled, error.OutOfMemory => |e| return e,
                    else => break :src_path "manifest",
                };
            }),
        ),
        .compiler = null,
        .src_dir = src_dir: {
            const src_root = runner.args.src_dir orelse
                return runner.fail("missing \"--src=path/to/src\" arg", .{});
            break :src_dir src_root.createDirPathOpen(
                runner.io,
                target_query,
                .{},
            ) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| return runner.fail("unable to create {q}: {t}", .{ target_query, e }),
            };
        },
        .cmd_src_loc = undefined,
        .contents_src_loc = undefined,
        .src_loc = null,
        .num = 0,
        .name = undefined,
        .prog_node = .none,
        .mtime = .now(runner.io, .real),
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
        const cmd_name = std.mem.cutPrefix(u8, cmd_src_loc, "#") orelse
            return update.manifest_sr.fail(cmd_src_loc, "expected command", .{});
        const cmd: Command = cmd: switch (cmd_name.len) {
            0 => .comment,
            else => switch (std.meta.stringToEnum(Command, cmd_name) orelse break :cmd null) {
                else => |cmd| cmd,
                .comment => null,
            },
        } orelse return update.manifest_sr.fail(cmd_src_loc, "unknown command", .{});

        const arg = switch (cmd) {
            .comment, .update => null,
            .skip, .require, .zig, .write, .delete, .check => line_it.next() orelse
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
        const contents_r: *std.Io.Reader = contents_r: {
            const contents = line_it.rest();
            if (contents.len == 0) {
                const src_info = update.manifest_sr.srcInfo(
                    update.manifest_sr.interface.buffered().ptr,
                );
                update.contents_src_loc = .{
                    .path = update.manifest_sr.src_path,
                    .line = src_info.line,
                    .column = src_info.column,
                };
                contents_impl = .{ .r = .ending_instance };
                break :contents_r &contents_impl.r;
            }
            switch (cmd) {
                .comment, .zig, .write, .update, .check => {},
                .skip, .require, .delete => return update.manifest_sr.fail(
                    contents,
                    "unexpected contents {q}",
                    .{contents},
                ),
            }
            contents: switch (contents[0]) {
                else => return update.manifest_sr.fail(contents, "invalid contents syntax", .{}),
                '"' => {
                    if (contents.len < 2 or contents[contents.len - 1] != '"') continue :contents 0;
                    update.contents_src_loc = .{
                        .path = try update.manifest_sr.eb_wip.addString("<string>"),
                    };
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

                    const src_info = update.manifest_sr.srcInfo(
                        update.manifest_sr.interface.buffered().ptr,
                    );
                    update.contents_src_loc = .{
                        .path = update.manifest_sr.src_path,
                        .line = src_info.line,
                        .column = src_info.column,
                    };
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

                    update.contents_src_loc = .{
                        .path = try update.manifest_sr.eb_wip.addString(contents_path),
                    };
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
                    try runner.discoverInput(args_test.input_dir, contents_path);
                    contents_impl = .{ .fr = contents_file.?.reader(runner.io, &contents_buffer) };
                    break :contents_r &contents_impl.fr.interface;
                },
            }
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
const Command = enum { comment, skip, require, zig, write, delete, update, check };
const Requirement = enum { @"no-matrix", @"safe-unwind", wasm, windows };
fn handleCommand(
    runner: *Runner,
    update: *Update,
    cmd_src_loc: []const u8,
    cmd: Command,
    arg: ?[]const u8,
    contents_r: *std.Io.Reader,
) (error{ContentsReadFailed} || TestError)!void {
    std.log.debug("handleCommand({}, {?q})", .{ cmd, arg });
    const arena = runner.arena.allocator();

    var name_buffer: [std.Progress.Node.max_name_len]u8 = undefined;
    const cmd_prog_node = runner.prog_node.start(std.mem.print(&name_buffer, "#{t}{s}{s}", .{
        cmd, if (arg) |_| " " else "", arg orelse "",
    }) catch &name_buffer, 0);
    defer cmd_prog_node.end();

    switch (cmd) {
        .comment => {},
        .skip, .require => if (update.skipped != null and update.skipped.? == .no) return update.manifest_sr.fail(
            cmd_src_loc,
            "\"#{t}\" must appear before other commands",
            .{cmd},
        ),
        else => {
            if (update.invalid_skip) return error.AlreadyReported;
            if (update.skipped) |s| switch (s) {
                .no => {},
                .yes_skip => {
                    std.log.info("skipped by a \"#skip\" command", .{});
                    return error.SkipTest;
                },
                .yes_require => |require| {
                    std.log.info("skipped due to unsatisfied requirement \"{s}{t}\"", .{
                        if (require.negate) "!" else "", require.tag,
                    });
                    return error.SkipTest;
                },
            } else {
                update.skipped = .no;
            }
        },
    }

    cmd: switch (cmd) {
        .comment => _ = contents_r.discardRemaining() catch |err| switch (err) {
            error.ReadFailed => {},
        },
        .skip => {
            const skip_target = arg.?;
            if (runner.args.getTargetIndex(skip_target)) |skip_target_index| {
                if (skip_target_index == update.test_index) update.skipped = .yes_skip;
            } else if (update.test_index == 0) {
                const eb_wip = update.manifest_sr.eb_wip;
                const ErrorMessage = std.zig.ErrorBundle.ErrorMessage;
                const notes: []const ErrorMessage = if (!update.invalid_skip) notes: {
                    const targets_count = runner.args.targets.count();
                    const notes = try arena.alloc(ErrorMessage, targets_count);
                    for (notes, 0..) |*note, target_index| note.* = .{
                        .msg = try eb_wip.printString("matrix contains {q}", .{
                            runner.args.getTarget(target_index),
                        }),
                    };
                    break :notes notes;
                } else &.{};
                try eb_wip.addRootErrorMessageWithNotes(.{
                    .msg = try eb_wip.addString("target not in matrix"),
                    .src_loc = try update.manifest_sr.srcLoc(skip_target),
                    .notes_len = @intCast(notes.len),
                }, notes);
                update.invalid_skip = true;
            } else return error.SkipTest;
        },
        .require => {
            const negate, const requirement_slice = if (std.mem.cutPrefix(u8, arg.?, "!")) |requirement|
                .{ true, requirement }
            else
                .{ false, arg.? };
            const requirement = std.meta.stringToEnum(Requirement, requirement_slice) orelse {
                return update.manifest_sr.fail(requirement_slice, "unknown requirement", .{});
            };
            const satisfied: bool = switch (requirement) {
                .@"no-matrix" => update.test_index == 0,
                .@"safe-unwind" => switch (update.target.unwind_tables) {
                    .async => true,
                    .none, .sync => switch (std.debug.FpUsability.of(&update.target.resolved)) {
                        .useless, .unsafe => false,
                        .safe, .ideal => switch (update.target.frame_pointers) {
                            .keep => true,
                            .omit => false,
                        },
                    },
                },
                .wasm => update.target.resolved.cpu.arch.isWasm(),
                .windows => switch (update.target.resolved.os.tag) {
                    else => false,
                    .uefi, .windows => true,
                },
            };
            if (switch (negate) {
                false => !satisfied,
                true => satisfied,
            }) {
                update.skipped = .{ .yes_require = .{ .tag = requirement, .negate = negate } };
            }
        },
        .zig => runner.spawnCompiler(
            &update.compiler,
            update.target,
            update.src_dir,
            std.meta.stringToEnum(Compiler.Command, arg.?) orelse
                return update.manifest_sr.fail(arg.?, "unknown compiler command", .{}),
            contents_r,
        ) catch |err| switch (err) {
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
                "missing \"#zig\" command",
                .{},
            ));
            update.prog_node.end();
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
                "missing \"#zig\" command",
                .{},
            ));
            if (comp.state != .update)
                return update.manifest_sr.fail(cmd_src_loc, "missing \"#update\"", .{});
            const stderr = comp.mr.reader(1);
            while (try comp.receiveMessage()) |message| switch (message.tag) {
                else => {}, // Ignore other messages
                .config => {
                    var body_r: std.Io.Reader = .fixed(message.body);
                    comp.config = body_r.takeStruct(std.zig.Server.Message.Config, .little) catch
                        unreachable;
                    try comp.updateEmitName(body_r.buffered());
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
                    const eb =
                        std.zig.Server.allocErrorBundle(arena, message.body) catch |err| switch (err) {
                            error.OutOfMemory => |e| return e,
                            error.EndOfStream => return update.fail("error bundle truncated", .{}),
                        };
                    if (check == .errors) {
                        var error_aw: std.Io.Writer.Allocating = .init(runner.gpa);
                        defer error_aw.deinit();
                        eb.renderToWriter(.{
                            .include_reference_trace = false,
                            .include_source_line = false,
                            .include_log_text = true,
                        }, &error_aw.writer) catch |err| switch (err) {
                            error.WriteFailed => return error.OutOfMemory,
                        };
                        try update.checkPattern("errors", expected, error_aw.written(), &.{
                            .{ "\n", .anon, newline },
                            .{ "/", .anon, &.{
                                .{ .any = &.{
                                    .{ '/', '/' },
                                    .{ '\\', '\\' },
                                } },
                            } },
                            .{ "builtin.zig:?:?:", .anon, &.{
                                .{ .none = &.{
                                    .{ '\n', '\n' },
                                    .{ '\r', '\r' },
                                    .{ ' ', ' ' },
                                    .{ ':', ':' },
                                } },
                                .{ .@"?" = &.{
                                    .{ .literal = ":" },
                                } },
                                .{ .@"*" = &.{
                                    .{ .none = &.{
                                        .{ '\n', '\n' },
                                        .{ '\r', '\r' },
                                        .{ ':', ':' },
                                    } },
                                } },
                                .{ .any = &.{
                                    .{ '/', '/' },
                                    .{ '\\', '\\' },
                                } },
                                .{ .literal = "b" },
                                .{ .any = &.{
                                    .{ '/', '/' },
                                    .{ '\\', '\\' },
                                } },
                                .{ .@"+" = &.{
                                    .{ .any = &.{
                                        .{ '0', '9' },
                                        .{ 'a', 'f' },
                                    } },
                                } },
                                .{ .any = &.{
                                    .{ '/', '/' },
                                    .{ '\\', '\\' },
                                } },
                                .{ .literal = "builtin.zig:" },
                                .{ .@"+" = &.{
                                    .{ .any = &.{
                                        .{ '0', '9' },
                                    } },
                                } },
                                .{ .literal = ":" },
                                .{ .@"+" = &.{
                                    .{ .any = &.{
                                        .{ '0', '9' },
                                    } },
                                } },
                                .{ .literal = ":" },
                            } },
                            .{ "lib/std/**.zig:?:?: note: called inline here\n", .anon, &.{
                                .{ .@"*" = &[_]Pattern{
                                    .{ .@"?" = &.{
                                        .{ .none = &.{
                                            .{ '\n', '\n' },
                                            .{ '\r', '\r' },
                                            .{ ' ', ' ' },
                                            .{ ':', ':' },
                                        } },
                                        .{ .@"?" = &.{
                                            .{ .literal = ":" },
                                        } },
                                        .{ .@"*" = &.{
                                            .{ .none = &.{
                                                .{ '\n', '\n' },
                                                .{ '\r', '\r' },
                                                .{ ':', ':' },
                                            } },
                                        } },
                                    } },
                                    .{ .literal = "lib" },
                                    .{ .any = &.{
                                        .{ '/', '/' },
                                        .{ '\\', '\\' },
                                    } },
                                    .{ .literal = "std" },
                                    .{ .any = &.{
                                        .{ '/', '/' },
                                        .{ '\\', '\\' },
                                    } },
                                    .{ .@"+" = &.{
                                        .{ .none = &.{
                                            .{ '\n', '\n' },
                                            .{ '\r', '\r' },
                                            .{ ':', ':' },
                                        } },
                                    } },
                                    .{ .literal = ".zig:" },
                                    .{ .@"+" = &.{
                                        .{ .any = &.{
                                            .{ '0', '9' },
                                        } },
                                    } },
                                    .{ .literal = ":" },
                                    .{ .@"+" = &.{
                                        .{ .any = &.{
                                            .{ '0', '9' },
                                        } },
                                    } },
                                    .{ .literal = ": note: called inline here" },
                                } ++ newline },
                            } },
                            .{ "lib/std/**.zig:?:?:", .anon, &.{
                                .{ .@"?" = &.{
                                    .{ .none = &.{
                                        .{ '\n', '\n' },
                                        .{ '\r', '\r' },
                                        .{ ' ', ' ' },
                                        .{ ':', ':' },
                                    } },
                                    .{ .@"?" = &.{
                                        .{ .literal = ":" },
                                    } },
                                    .{ .@"*" = &.{
                                        .{ .none = &.{
                                            .{ '\n', '\n' },
                                            .{ '\r', '\r' },
                                            .{ ':', ':' },
                                        } },
                                    } },
                                } },
                                .{ .literal = "lib" },
                                .{ .any = &.{
                                    .{ '/', '/' },
                                    .{ '\\', '\\' },
                                } },
                                .{ .literal = "std" },
                                .{ .any = &.{
                                    .{ '/', '/' },
                                    .{ '\\', '\\' },
                                } },
                                .{ .@"+" = &.{
                                    .{ .none = &.{
                                        .{ '\n', '\n' },
                                        .{ '\r', '\r' },
                                        .{ ':', ':' },
                                    } },
                                } },
                                .{ .literal = ".zig:" },
                                .{ .@"+" = &.{
                                    .{ .any = &.{
                                        .{ '0', '9' },
                                    } },
                                } },
                                .{ .literal = ":" },
                                .{ .@"+" = &.{
                                    .{ .any = &.{
                                        .{ '0', '9' },
                                    } },
                                } },
                                .{ .literal = ":" },
                            } },
                        });
                    } else if (eb.errorMessageCount() > 0) {
                        eb.renderToStderr(runner.io, .{}, .auto) catch |err| switch (err) {
                            error.Canceled => {},
                            else => {},
                        };
                        return update.fail("unexpected compile errors", .{});
                    }
                    comp.state = .idle;
                    break :cmd;
                },
            };

            const stderr_buffered = stderr.buffered();
            if (stderr_buffered.len > 0) {
                if (comp.allow_compiler_stderr) {
                    std.log.warn("#update {q} stderr:\n{s}", .{ update.name, stderr_buffered });
                } else return update.fail("unexpected stderr:\n{s}", .{stderr_buffered});
            }

            return update.fail("compiler failed to send terminating error_bundle", .{});
        },
    }
    assert((contents_r.discardRemaining() catch |err| switch (err) {
        error.ReadFailed => return error.ContentsReadFailed,
    }) == 0);
}

fn spawnCompiler(
    runner: *Runner,
    compiler: *?Compiler,
    target: Compiler.Target,
    src_dir: std.Io.Dir,
    command: Compiler.Command,
    args_r: *std.Io.Reader,
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
        .allow_compiler_stderr = false,
        .command = command,
        .state = .idle,
        .child = undefined,
        .fw = undefined,
        .mr_buffer = undefined,
        .mr = undefined,
        .client = undefined,
        .target = target,
        .emit = .bin,
        .emit_name = null,
        .prog_node = .none,
        .config = null,
    };
    const comp = &compiler.*.?;

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.appendSlice(gpa, &.{
        "zig",
        @tagName(command),
        "-target",
        comp.target.triple,
        "--cache-dir",
        ".zig-cache",
    });
    switch (comp.target.cache_mode) {
        .whole => try argv.append(gpa, "-fno-incremental"),
        .incremental => try argv.append(gpa, "-fincremental"),
    }
    if (comp.target.backend) |b| switch (b) {
        .selfhosted => try argv.append(gpa, "-fno-llvm"),
        .llvm => try argv.append(gpa, "-fllvm"),
        .cbe => try argv.append(gpa, "-ofmt=c"),
    } else try argv.appendSlice(gpa, &.{ "-I", lib_path: {
        const lib_dir = runner.args.lib_dir orelse
            return runner.fail("\"--lib=path/to/lib\" arg required for cbe", .{});
        break :lib_path runner.dirPath(lib_dir) catch |err| switch (err) {
            error.Canceled, error.OutOfMemory => |e| return e,
            else => |e| return runner.fail("unable to get lib path: {t}", .{e}),
        };
    } });
    try argv.appendSlice(gpa, &.{ "-O", @tagName(comp.target.optimize) });
    switch (comp.target.pic) {
        .default => {},
        .pdc => try argv.appendSlice(gpa, &.{ "-fno-PIC", "-fno-PIE" }),
        .pic => try argv.appendSlice(gpa, &.{ "-fPIC", "-fno-PIE" }),
        .pie => try argv.appendSlice(gpa, &.{ "-fPIC", "-fPIE" }),
    }
    if (!comp.target.default_frame_pointers) switch (comp.target.frame_pointers) {
        .keep => try argv.append(gpa, "-fno-omit-frame-pointer"),
        .omit => try argv.append(gpa, "-fomit-frame-pointer"),
    };
    if (!comp.target.default_unwind_tables) switch (comp.target.unwind_tables) {
        .none => try argv.append(gpa, "-fno-unwind-tables"),
        .sync => try argv.append(gpa, "-funwind-tables"),
        .async => try argv.append(gpa, "-fasync-unwind-tables"),
    };
    switch (comp.target.linker) {
        .default => {},
        .lld => try argv.appendSlice(gpa, &.{ "-flld", "-fno-new-linker" }),
        .old => try argv.appendSlice(gpa, &.{ "-fno-lld", "-fno-new-linker" }),
        .new => try argv.appendSlice(gpa, &.{ "-fno-lld", "-fnew-linker" }),
    }
    switch (comp.target.linkage) {
        .default => {},
        inline .static, .dynamic => |linkage| try argv.append(gpa, "-" ++ @tagName(linkage)),
    }
    switch (comp.target.link_libc) {
        .default => {},
        .lc => try argv.append(gpa, "-lc"),
    }
    switch (comp.target.strip) {
        .default => {},
        .nostrip => try argv.append(gpa, "-fno-strip"),
        .stripdebug => return runner.fail("compiler does not support stripping debug info", .{}),
        .strip => try argv.append(gpa, "-fstrip"),
    }

    var need: usize = 1;
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
            if (std.mem.cutPrefix(u8, arg, "-ofmt=")) |ofmt| {
                comp.target.resolved.ofmt =
                    std.meta.stringToEnum(std.Target.ObjectFormat, ofmt) orelse
                    return runner.fail("invalid object format {q}", .{ofmt});
            } else if (std.mem.eql(u8, arg, "-femit-asm")) {
                comp.emit = .@"asm";
            } else if (std.mem.eql(u8, arg, "-femit-bin")) {
                comp.emit = .bin;
            } else if (std.mem.eql(u8, arg, "-femit-docs")) {
                comp.emit = .docs;
            } else if (std.mem.eql(u8, arg, "-femit-h")) {
                comp.emit = .h;
            } else if (std.mem.eql(u8, arg, "-femit-implib")) {
                comp.emit = .implib;
            } else if (std.mem.eql(u8, arg, "-femit-llvm-bc")) {
                comp.emit = .llvm_bc;
            } else if (std.mem.eql(u8, arg, "-femit-llvm-ir")) {
                comp.emit = .llvm_ir;
            } else if (std.mem.eql(u8, arg, "-fno-emit-asm")) {
                if (comp.emit == .@"asm") comp.emit = null;
            } else if (std.mem.eql(u8, arg, "-fno-emit-bin")) {
                if (comp.emit == .bin) comp.emit = null;
            } else if (std.mem.eql(u8, arg, "-fno-emit-docs")) {
                if (comp.emit == .docs) comp.emit = null;
            } else if (std.mem.eql(u8, arg, "-fno-emit-h")) {
                if (comp.emit == .h) comp.emit = null;
            } else if (std.mem.eql(u8, arg, "-fno-emit-implib")) {
                if (comp.emit == .implib) comp.emit = null;
            } else if (std.mem.eql(u8, arg, "-fno-emit-llvm-bc")) {
                if (comp.emit == .llvm_bc) comp.emit = null;
            } else if (std.mem.eql(u8, arg, "-fno-emit-llvm-ir")) {
                if (comp.emit == .llvm_ir) comp.emit = null;
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

    std.log.info("{f}", .{std.zig.SubprocessCommand{
        .argv = argv.items,
        .cwd = cwd_path: {
            break :cwd_path runner.dirPath(src_dir) catch |err| switch (err) {
                error.Canceled, error.OutOfMemory => |e| return e,
                else => break :cwd_path null,
            };
        },
    }});

    comp.prog_node = runner.prog_node.start("compile", 0);
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
        error.OperationUnsupported => {
            std.log.info("skipped because spawning is not supported", .{});
            return error.SkipTest;
        },
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
    command: Compiler.Command,
    state: enum { idle, update },
    child: std.process.Child,
    fw: std.Io.File.Writer,
    mr_buffer: std.Io.File.MultiReader.Buffer(2),
    mr: std.Io.File.MultiReader,
    client: std.zig.Client,
    target: Compiler.Target,
    emit: ?std.zig.EmitArtifact,
    emit_name: ?[]const u8,
    prog_node: std.Progress.Node,
    config: ?std.zig.Server.Message.Config,

    const Command = enum {
        @"build-obj",
        @"build-exe",
        @"build-lib",
        @"test",
        @"test-obj",

        fn outputMode(cmd: Compiler.Command) std.lang.OutputMode {
            return switch (cmd) {
                .@"build-obj", .@"test-obj" => .Obj,
                .@"build-exe", .@"test" => .Exe,
                .@"build-lib" => .Lib,
            };
        }

        fn isTest(cmd: Compiler.Command) bool {
            return switch (cmd) {
                .@"build-obj", .@"build-exe", .@"build-lib" => false,
                .@"test", .@"test-obj" => true,
            };
        }
    };
    const Target = struct {
        triple: []const u8,
        resolved: std.Target,
        cache_mode: Runner.Target.CacheMode,
        /// `null` means c compiler.
        backend: ?Runner.Target.Backend,
        optimize: std.lang.Optimize,
        pic: Runner.Target.Pic,
        default_frame_pointers: bool,
        frame_pointers: Runner.Target.FramePointers.Resolved,
        default_unwind_tables: bool,
        unwind_tables: std.lang.UnwindTables,
        linker: Runner.Target.Linker,
        linkage: Runner.Target.Linkage,
        link_libc: Runner.Target.LinkLibc,
        strip: Runner.Target.Strip,
    };

    fn deinit(comp: *Compiler) void {
        comp.child.kill(comp.runner.io);
        comp.mr.deinit();
        comp.prog_node.end();
        comp.prog_node = .none;
    }

    fn receiveMessage(comp: *Compiler) FailError!?struct {
        tag: std.zig.Server.Message.Tag,
        body: []const u8,
    } {
        const maybe_header = comp.client.receiveMessageWithMultiReader(
            &comp.mr,
            .none,
        ) catch |err| switch (err) {
            error.Canceled, error.OutOfMemory => |e| return e,
            error.Timeout => unreachable,
            error.EndOfStream => null,
            else => |e| return comp.runner.fail("unable to receive message: {t}", .{e}),
        };
        const stderr = comp.mr.reader(1);
        const stderr_buffered = stderr.buffered();
        if (stderr_buffered.len > 0) {
            const tag_name = if (maybe_header) |header| @tagName(header.tag) else "exit";
            if (comp.allow_compiler_stderr) {
                std.log.warn("{s} stderr:\n{s}", .{ tag_name, stderr_buffered });
            } else return comp.runner.fail("{s} unexpected stderr:\n{s}", .{
                tag_name, stderr_buffered,
            });
            stderr.tossBuffered();
        }
        const header = maybe_header orelse return null;
        const body = comp.client.in.take(header.bytes_len) catch unreachable;
        return .{ .tag = header.tag, .body = body };
    }

    fn updateEmitName(comp: *Compiler, root_name: []const u8) !void {
        const emit = comp.emit orelse return;
        const emit_name = try emit.cacheName(comp.runner.arena.allocator(), .{
            .root_name = root_name,
            .cpu_arch = comp.target.resolved.cpu.arch,
            .os_tag = comp.target.resolved.os.tag,
            .ofmt = comp.target.resolved.ofmt,
            .abi = comp.target.resolved.abi,
            .output_mode = comp.command.outputMode(),
        });
        comp.emit_name = emit_name;
        var name_buffer: [std.Progress.Node.max_name_len]u8 = undefined;
        comp.prog_node.setName(
            std.mem.print(&name_buffer, "compile {s}", .{emit_name}) catch &name_buffer,
        );
    }

    const Check = enum { errors, stdout, exit, @"error-trace", panic, @"stack-trace", llvm, gdb, lldb };
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
        if (check == .errors) return; // expect no emitted file
        const emit_name = comp.emit_name orelse
            return update.fail("unable to perform {t} check without emitted file", .{check});
        const config = &comp.config.?;

        // After a failure while running a script, the debugger starts accepting commands from
        // stdin, and because it is empty, the debugger exits normally with status 0. Choose a
        // non-zero status to return from the debugger script instead to detect it running to
        // completion and indicate success.
        const debug_success: std.process.Child.Term = .{ .exited = 99 };

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
            .@"error-trace", .panic, .@"stack-trace" => {
                _ = environ_map.swapRemove("CLICOLOR_FORCE");
                try environ_map.put("NO_COLOR", "1"); // affects output formatting
                break :script .{ undefined, null, undefined };
            },
            .llvm => {
                var emit_fr = emit_file.reader(io, &.{});
                const actual = emit_fr.interface.readAllocAll(arena, @intCast(
                    emit_fr.getSize() catch |err| switch (err) {
                        error.Canceled => |e| return e,
                        else => |e| return runner.fail(
                            "unable to get size of {q}: {t}",
                            .{ emit_path, e },
                        ),
                    },
                )) catch |err| switch (switch (err) {
                    error.OutOfMemory => |e| return e,
                    error.EndOfStream => |e| e,
                    error.ReadFailed => emit_fr.err.?,
                }) {
                    error.Canceled => |e| return e,
                    else => |e| return runner.fail("unable to read {q}: {t}", .{ emit_path, e }),
                };
                const whitespace: Pattern = .{ .@"*" = &.{
                    .{ .any = &.{
                        .{ '\t', '\n' },
                        .{ '\r', '\r' },
                        .{ ' ', ' ' },
                    } },
                } };
                const int: []const Pattern = &.{
                    .{ .@"?" = &.{
                        .{ .@"?" = &.{
                            .{ .literal = "-" },
                        } },
                        .{ .literal = "0x" },
                    } },
                    .{ .@"+" = &.{
                        .{ .any = &.{
                            .{ '0', '9' },
                        } },
                    } },
                };
                const attr: []const Pattern = &.{
                    .{ .@"+" = &.{
                        .{ .@"|" = &.{ &.{
                            .{ .none = &.{
                                .{ '\t', '\n' },
                                .{ '\r', '\r' },
                                .{ ' ', ' ' },
                                .{ '"', '"' },
                                .{ '(', ')' },
                                .{ ',', ',' },
                            } },
                        }, &.{
                            .{ .literal = "\"" },
                            .{ .@"*" = &.{
                                .{ .none = &.{
                                    .{ '\t', '\n' },
                                    .{ '\r', '\r' },
                                    .{ '"', '"' },
                                } },
                            } },
                            .{ .literal = "\"" },
                        } } },
                    } },
                    .{ .@"?" = &.{
                        .{ .literal = "(" },
                        .{ .@"*" = &.{
                            .{ .none = &.{
                                .{ '\t', '\n' },
                                .{ '\r', '\r' },
                                .{ '(', ')' },
                            } },
                        } },
                        .{ .literal = ")" },
                    } },
                };
                var id: Pattern.Compiled = try .init(gpa, &.{
                    .{ .any = &.{
                        .{ '$', '$' },
                        .{ '-', '.' },
                        .{ 'A', 'Z' },
                        .{ '_', '_' },
                        .{ 'a', 'z' },
                    } },
                    .{ .@"*" = &.{
                        .{ .any = &.{
                            .{ '$', '$' },
                            .{ '-', '.' },
                            .{ '0', '9' },
                            .{ 'A', 'Z' },
                            .{ '_', '_' },
                            .{ 'a', 'z' },
                        } },
                    } },
                });
                defer id.deinit(gpa);
                return update.checkPattern("LLVM IR", expected, actual, &.{
                    .{ "\n", .anon, newline },
                    // Match zero or more metadata attachments at the end of a line.
                    .{ ", !...\n", .anon, &[_]Pattern{
                        .{ .@"*" = &[_]Pattern{
                            .{ .literal = "," },
                            whitespace,
                            .{ .literal = "!" },
                            whitespace,
                            .{ .any = &.{
                                .{ 'A', 'Z' },
                                .{ 'a', 'z' },
                                .{ '-', '-' },
                                .{ '$', '$' },
                                .{ '.', '.' },
                                .{ '_', '_' },
                                .{ '\\', '\\' },
                            } },
                            .{ .@"*" = &.{
                                .{ .any = &.{
                                    .{ 'A', 'Z' },
                                    .{ 'a', 'z' },
                                    .{ '0', '9' },
                                    .{ '-', '-' },
                                    .{ '$', '$' },
                                    .{ '.', '.' },
                                    .{ '_', '_' },
                                    .{ '\\', '\\' },
                                } },
                            } },
                            whitespace,
                            .{ .literal = "!" },
                            whitespace,
                        } ++ int },
                    } ++ newline },
                    // Match zero or more lines.
                    .{ "...\n", .anon, &.{
                        .{ .@"*" = &[_]Pattern{
                            .{ .@"*" = &.{
                                .{ .none = &.{
                                    .{ '\n', '\n' },
                                    .{ '\r', '\r' },
                                } },
                            } },
                        } ++ newline },
                    } },
                    // Match zero or more space separated attributes.
                    .{ "... ", .anon, &.{
                        .{ .@"*" = attr ++ &[_]Pattern{
                            .{ .literal = " " },
                        } },
                    } },
                    // Match zero or more comma separated parameters or operands.
                    .{ "..., ", .anon, &.{
                        .{ .@"*" = attr ++ &[_]Pattern{
                            .{ .@"*" = &[_]Pattern{
                                .{ .literal = " " },
                            } ++ attr },
                            .{ .literal = ", " },
                        } },
                    } },
                    // Match zero or more lines not starting with '}'.
                    .{ "  ...\n", .anon, &.{
                        .{ .@"*" = &[_]Pattern{
                            .{ .@"?" = &.{
                                .{ .none = &.{
                                    .{ '\n', '\n' },
                                    .{ '\r', '\r' },
                                    .{ '}', '}' },
                                } },
                                .{ .@"*" = &.{
                                    .{ .none = &.{
                                        .{ '\n', '\n' },
                                        .{ '\r', '\r' },
                                    } },
                                } },
                            } },
                        } ++ newline },
                    } },
                    // Matches a constant integer.
                    .{ "?", .anon, int },
                    // Matches an integer type corresponding to Zig `isize/usize`.
                    .{ "isize", .anon, &.{
                        .{ .literal = "i" },
                        .{ .@"|" = &.{ &.{
                            .{ .literal = "16" },
                        }, &.{
                            .{ .literal = "24" },
                        }, &.{
                            .{ .literal = "32" },
                        }, &.{
                            .{ .literal = "64" },
                        } } },
                    } },

                    // Matches literal `!`, escapes named syntax.
                    .{ "!!", .anon, &.{
                        .{ .literal = "!" },
                    } },
                    .{ "!", .{ .name = &id }, &[_]Pattern{
                        .{ .literal = "!" },
                        whitespace,
                    } ++ int },
                    // Matches literal `#`, escapes named syntax.
                    .{ "##", .anon, &.{
                        .{ .literal = "#" },
                    } },
                    //    AttrGrpID ::= #[0-9]+
                    .{ "#", .{ .name = &id }, &.{
                        .{ .literal = "#" },
                        .{ .@"+" = &.{
                            .{ .any = &.{
                                .{ '0', '9' },
                            } },
                        } },
                    } },
                    // Matches literal `#`, escapes named syntax.
                    .{ "%%", .anon, &.{
                        .{ .literal = "%" },
                    } },
                    //   LocalVar   ::= %\"[^\"]*\"
                    //   LocalVar   ::= %[-a-zA-Z$._][-a-zA-Z$._0-9]*
                    //   LocalVarID ::= %[0-9]+
                    .{ "%", .{ .name = &id }, &.{
                        .{ .literal = "%" },
                        .{ .@"|" = &.{ &.{
                            .{ .literal = "\"" },
                            .{ .@"*" = &.{
                                .{ .none = &.{
                                    .{ '\"', '\"' },
                                } },
                            } },
                            .{ .literal = "\"" },
                        }, &.{
                            .{ .any = &.{
                                .{ '-', '-' },
                                .{ 'a', 'z' },
                                .{ 'A', 'Z' },
                                .{ '$', '$' },
                                .{ '.', '.' },
                                .{ '_', '_' },
                            } },
                            .{ .@"*" = &.{
                                .{ .any = &.{
                                    .{ '-', '-' },
                                    .{ 'a', 'z' },
                                    .{ 'A', 'Z' },
                                    .{ '$', '$' },
                                    .{ '.', '.' },
                                    .{ '_', '_' },
                                    .{ '0', '9' },
                                } },
                            } },
                        }, &.{
                            .{ .@"+" = &.{
                                .{ .any = &.{
                                    .{ '0', '9' },
                                } },
                            } },
                        } } },
                    } },
                    // Matches literal `^`, escapes named syntax.
                    .{ "^^", .anon, &.{
                        .{ .literal = "^" },
                    } },
                    //    SummaryID ::= ^[0-9]+
                    .{ "^", .{ .name = &id }, &.{
                        .{ .literal = "^" },
                        .{ .@"+" = &.{
                            .{ .any = &.{
                                .{ '0', '9' },
                            } },
                        } },
                    } },
                });
            },
            .gdb => {
                const script_path = tmp_dir_path ++ std.Io.Dir.path.sep_str ++ "gdb";
                argv.appendSliceAssumeCapacity(&.{
                    "gdb",
                    "--batch",
                    "--command",
                    script_path,
                    "--args",
                });
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
                std.log.info("skipping execution because host {q} cannot " ++
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
                    argv.appendSliceAssumeCapacity(&.{ "-L", try std.Io.Dir.path.join(arena, &.{
                        runner.dirPath(runner.args.libc_runtimes_dir orelse
                            continue :use_executor .bad_os_or_cpu) catch |err| switch (err) {
                            error.Canceled, error.OutOfMemory => |e| return e,
                            else => |e| return update.fail(
                                "unable to get libc runtimes path: {t}",
                                .{e},
                            ),
                        },
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

        std.log.info("{f}", .{std.zig.SubprocessCommand{
            .argv = argv.items,
            .cwd = cwd_path: {
                break :cwd_path runner.dirPath(update.src_dir) catch |err| switch (err) {
                    error.Canceled, error.OutOfMemory => |e| return e,
                    else => break :cwd_path null,
                };
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
                .stdout, .exit, .@"error-trace", .panic, .@"stack-trace" => if (use_executor)
                    .search
                else
                    .{ .file = emit_file },
                .gdb, .lldb => {
                    if (use_executor) {
                        std.log.info(
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
                            else => |e| return update.fail(
                                "unable to write debugger script {q}: {t}",
                                .{ script_path, e },
                            ),
                        },
                    };
                    script_fw.interface.print(
                        \\
                        \\quit {d}
                        \\
                    , .{debug_success.exited}) catch |err| switch (err) {
                        error.WriteFailed => switch (script_fw.err.?) {
                            error.Canceled => |e| return e,
                            else => |e| return update.fail(
                                "unable to write debugger script {q}: {t}",
                                .{ script_path, e },
                            ),
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
                        std.log.info(
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
            .inherit_files = switch (check) {
                .errors, .llvm => unreachable,
                .stdout, .exit, .gdb, .lldb => &.{},
                .@"error-trace", .panic, .@"stack-trace" => &.{emit_file},
            },
        }) catch |err| if (use_executor) {
            // Chances are the foreign executor isn't available. Skip this evaluation.
            std.log.info(
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
        switch (check) {
            .errors, .llvm => unreachable,
            .stdout => {
                const expected_term: std.process.Child.Term = .{ .exited = 0 };
                if (!result.term.eql(expected_term)) return update.fail(
                    "{q} {f}, expected to have {f}",
                    .{ emit_path, result.term, expected_term },
                );
                try update.checkPattern("stdout", expected, result.stdout, &.{
                    .{ "\n", .anon, newline },
                });
            },
            .exit => switch (result.term) {
                else => return update.fail(
                    "{q} {f}, expected to have exited with code {s}",
                    .{ emit_path, result.term, expected },
                ),
                .exited => |code| {
                    var actual_buffer: [std.fmt.count("{d}", .{std.math.maxInt(u8)})]u8 = undefined;
                    const actual = std.mem.print(&actual_buffer, "{d}", .{code}) catch unreachable;
                    try update.checkPattern("exit code", expected, actual, &.{});
                },
            },
            .@"error-trace", .panic, .@"stack-trace" => {
                const expected_term: std.process.Child.Term = switch (check) {
                    else => unreachable,
                    .@"error-trace" => .{ .exited = 1 },
                    .panic => switch (comp.target.resolved.os.tag) {
                        else => .{ .signal = @fromBackingInt(@intCast(6)) },
                        .windows => .{ .exited = 3 },
                    },
                    .@"stack-trace" => .success,
                };
                if (!result.term.eql(expected_term)) return update.fail(
                    "{q} {f}, expected to have {f}",
                    .{ emit_path, result.term, expected_term },
                );
                try update.checkPattern("stderr", expected, result.stderr, &.{
                    .{ "\n", .anon, newline },
                    .{ "thread ? panic: ", .anon, &.{
                        .{ .literal = "thread " },
                        .{ .@"+" = &.{
                            .{ .any = &.{
                                .{ '0', '9' },
                            } },
                        } },
                        .{ .literal = " panic: " },
                    } },
                    .{ "test.zig:", .anon, &.{
                        .{ .none = &.{
                            .{ '\n', '\n' },
                            .{ '\r', '\r' },
                            .{ ' ', ' ' },
                            .{ ':', ':' },
                        } },
                        .{ .@"?" = &.{
                            .{ .literal = ":" },
                        } },
                        .{ .@"*" = &.{
                            .{ .none = &.{
                                .{ '\n', '\n' },
                                .{ '\r', '\r' },
                                .{ ':', ':' },
                            } },
                        } },
                        .{ .any = &.{
                            .{ '/', '/' },
                            .{ '\\', '\\' },
                        } },
                        .{ .literal = "test.zig:" },
                    } },
                    .{ "lib/std/debug.zig:?:?:", .anon, &.{
                        .{ .none = &.{
                            .{ '\n', '\n' },
                            .{ '\r', '\r' },
                            .{ ' ', ' ' },
                            .{ ':', ':' },
                        } },
                        .{ .@"?" = &.{
                            .{ .literal = ":" },
                        } },
                        .{ .@"*" = &.{
                            .{ .none = &.{
                                .{ '\n', '\n' },
                                .{ '\r', '\r' },
                                .{ ':', ':' },
                            } },
                        } },
                        .{ .any = &.{
                            .{ '/', '/' },
                            .{ '\\', '\\' },
                        } },
                        .{ .literal = "lib" },
                        .{ .any = &.{
                            .{ '/', '/' },
                            .{ '\\', '\\' },
                        } },
                        .{ .literal = "std" },
                        .{ .any = &.{
                            .{ '/', '/' },
                            .{ '\\', '\\' },
                        } },
                        .{ .literal = "debug.zig:" },
                        .{ .@"+" = &.{
                            .{ .any = &.{
                                .{ '0', '9' },
                            } },
                        } },
                        .{ .literal = ":" },
                        .{ .@"+" = &.{
                            .{ .any = &.{
                                .{ '0', '9' },
                            } },
                        } },
                        .{ .literal = ":" },
                    } },
                    .{ "0x?", .anon, &.{
                        .{ .literal = "0x" },
                        .{ .@"+" = &.{
                            .{ .any = &.{
                                .{ '0', '9' },
                                .{ 'a', 'f' },
                            } },
                        } },
                    } },
                    .{ "[__func_?]", .anon, &.{
                        .{ .@"?" = &.{
                            .{ .literal = "__func_" },
                            .{ .@"+" = &.{
                                .{ .any = &.{
                                    .{ '0', '9' },
                                } },
                            } },
                        } },
                    } },
                    .{ " (test*)\n", .anon, &[_]Pattern{
                        .{ .literal = " (test" },
                        .{ .@"?" = &.{
                            .{ .@"|" = &.{ &.{
                                .{ .literal = ".zig" },
                            }, &.{
                                .{ .literal = "_zcu.obj" },
                            } } },
                        } },
                        .{ .literal = ")" },
                    } ++ newline },
                    .{ " (std*)\n", .anon, &[_]Pattern{
                        .{ .literal = " (" },
                        .{ .@"|" = &.{ &.{
                            .{ .literal = "test" },
                        }, &.{
                            .{ .literal = "test_zcu.obj" },
                        }, &.{
                            .{ .literal = "std.zig" },
                        } } },
                        .{ .literal = ")" },
                    } ++ newline },
                    .{ "...\n", .anon, &.{
                        .{ .@"+" = &[_]Pattern{
                            .{ .@"*" = &.{
                                .{ .none = &.{
                                    .{ '\n', '\n' },
                                    .{ '\r', '\r' },
                                } },
                            } },
                        } ++ newline },
                    } },
                });
                return;
            },
            .gdb, .lldb => {
                if (!result.term.eql(debug_success)) return update.fail(
                    "{q} {f}, expected to have {f}",
                    .{ emit_path, result.term, debug_success },
                );
                try update.checkPattern("debugger stdout", expected, result.stdout, &.{
                    .{ "\n", .anon, newline },
                    .{ "#...", .anon, &.{
                        .{ .@"*" = &.{
                            .{ .none = &.{} },
                        } },
                    } },
                });
            },
        }
        if (result.stderr.len > 0) {
            // * If we are using an executor such as wine, ignore any stderr spam it may produce.
            // * Since we do not use the test protocol here yet, ignore stderr output from the test
            //   runner in terminal mode.
            if (use_executor or comp.command.isTest()) {
                std.log.warn(
                    \\#update {q} generated executable {q} had unexpected stderr:
                    \\{s}
                    \\
                , .{ update.name, emit_path, result.stderr });
            } else return runner.fail(
                \\#update {q} generated executable {q} had unexpected stderr:
                \\{s}
                \\
            , .{ update.name, emit_path, result.stderr });
        }
    }

    fn compileC(parent_comp: *Compiler, src_dir: std.Io.Dir, c_path: []const u8) TestError![]const u8 {
        const runner = parent_comp.runner;
        debug: {
            std.log.debug("compileC({q}, {q})", .{
                runner.dirPath(src_dir) catch |err| switch (err) {
                    error.Canceled, error.OutOfMemory => |e| return e,
                    else => break :debug,
                },
                c_path,
            });
        }
        const arena = runner.arena.allocator();

        var compiler: ?Compiler = null;
        var args_r: std.Io.Reader = .fixed(if (parent_comp.emit_name) |emit_name|
            try arena.print("--name {s} {s}", .{ emit_name, c_path })
        else
            c_path);
        parent_comp.runner.spawnCompiler(&compiler, target: {
            var target = parent_comp.target;
            target.resolved.ofmt = .default(target.resolved.os.tag, target.resolved.cpu.arch);
            target.cache_mode = .whole;
            target.backend = null;
            break :target target;
        }, src_dir, parent_comp.command, &args_r) catch |err| switch (err) {
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
            else => {}, // Ignore other messages
            .config => {
                var body_r: std.Io.Reader = .fixed(message.body);
                comp.config = body_r.takeStruct(std.zig.Server.Message.Config, .little) catch
                    unreachable;
                try comp.updateEmitName(body_r.buffered());
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
                return runner.fail("compiler failed with exit code {d}", .{code}),
            .signal => |sig| return runner.fail("compiler terminated with signal {t}", .{sig}),
            .stopped => |sig| return runner.fail("compiler stopped unexpectedly with signal {t}", .{
                sig,
            }),
            .unknown => return runner.fail("compiler terminated unexpectedly", .{}),
        }
    }
};

const ClientArgs = struct {
    log_level: ?std.log.Level,
    keep_src: bool,

    test_file: std.Io.File,
    test_kind: std.Io.File.Kind,
    zig_exe: std.Io.File,
    lib_dir: ?std.Io.Dir,
    src_dir: std.Io.Dir,

    libc_runtimes_dir: ?std.Io.Dir,
    enable_darling: bool,
    enable_qemu: bool,
    enable_rosetta: bool,
    enable_wine: bool,
    enable_wasmtime: bool,

    gdb_exe: ?std.Io.File,
    lldb_exe: ?std.Io.File,

    matrix: []const Target,
    targets: []const []const u8,
    target_filters: []const []const u8,
    skip_matrix: bool,
};
pub fn main(init: std.process.Init) (std.mem.Allocator.Error || std.Io.Cancelable)!u8 {
    @setEvalBranchQuota(4_000);
    const arena = init.arena.allocator();

    var args: ClientArgs = .{
        .log_level = null,
        .keep_src = false,

        .test_file = undefined,
        .test_kind = undefined,
        .zig_exe = undefined,
        .lib_dir = null,
        .src_dir = undefined,

        .libc_runtimes_dir = null,
        .enable_darling = false,
        .enable_qemu = false,
        .enable_rosetta = false,
        .enable_wine = false,
        .enable_wasmtime = false,

        .gdb_exe = null,
        .lldb_exe = null,

        .matrix = &.{},
        .targets = &.{},
        .target_filters = &.{},
        .skip_matrix = undefined,
    };

    var test_path_arg: ?[]const u8 = null;
    var zig_path_arg: ?[]const u8 = null;
    var lib_path_arg: ?[]const u8 = null;
    var libc_runtimes_path_arg: ?[]const u8 = null;
    var gdb_path_arg: ?[]const u8 = null;
    var lldb_path_arg: ?[]const u8 = null;
    var matrix_path_arg: ?[]const u8 = null;
    var target_args: std.ArrayList([]const u8) = .empty;
    defer target_args.deinit(init.gpa);
    var target_filter_args: std.ArrayList([]const u8) = .empty;
    defer target_filter_args.deinit(init.gpa);
    var skip_matrix_arg: ?bool = null;

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
                    else => |e| return fatal("unable to get environment: {t}", .{e}),
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
                },
                .host = std.zig.system.resolveTargetQuery(init.io, .{}) catch |err| switch (err) {
                    error.Canceled => |e| return e,
                    else => |e| return fatal("unable to resolve host target: {t}", .{e}),
                },
                .server = .{
                    .in = &stdin.interface,
                    .out = &stdout.interface,
                },
                .eb_wip = try .init(init.gpa),
            };
            defer runner.deinit();
            runner.runServer() catch |err| switch (err) {
                error.Canceled, error.OutOfMemory => |e| return e,
                error.ServerReadFailed => switch (stdin.err.?) {
                    error.Canceled => |e| return e,
                    else => |e| return fatal("unable to receive message: {t}", .{e}),
                },
                error.ServerWriteFailed => switch (stdout.err.?) {
                    error.Canceled => |e| return e,
                    else => |e| return fatal("unable to send message: {t}", .{e}),
                },
                error.ServerEndOfStream => return fatal("no more messages", .{}),
                error.AlreadyReported => {
                    var eb = try runner.eb_wip.toOwnedBundle("");
                    defer eb.deinit(init.gpa);
                    eb.renderToStderr(runner.io, .{}, .auto) catch |stderr_err| switch (stderr_err) {
                        error.Canceled => |e| return e,
                        else => {},
                    };
                    return 1;
                },
            };
            return 0;
        } else if (if (std.mem.eql(u8, arg, "--log"))
            arg_it.next() orelse return fatal("missing arg after {q}", .{arg})
        else
            std.mem.cutPrefix(u8, arg, "--log=")) |log_level|
        {
            args.log_level = std.meta.stringToEnum(std.log.Level, log_level) orelse
                return fatal("invalid arg {q} after {q}", .{ log_level, arg });
        } else if (std.mem.eql(u8, arg, "--keep-src")) {
            args.keep_src = true;
        } else if (if (std.mem.eql(u8, arg, "--zig"))
            arg_it.next() orelse return fatal("missing arg after {q}", .{arg})
        else
            std.mem.cutPrefix(u8, arg, "--zig=")) |zig_path|
        {
            zig_path_arg = zig_path;
        } else if (if (std.mem.eql(u8, arg, "--lib"))
            arg_it.next() orelse return fatal("missing arg after {q}", .{arg})
        else
            std.mem.cutPrefix(u8, arg, "--lib=")) |lib_path|
        {
            lib_path_arg = lib_path;
        } else if (if (std.mem.eql(u8, arg, "--libc-runtimes"))
            arg_it.next() orelse return fatal("missing arg after {q}", .{arg})
        else
            std.mem.cutPrefix(u8, arg, "--libc-runtimes=")) |libc_runtimes_path|
        {
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
        } else if (if (std.mem.eql(u8, arg, "--gdb"))
            arg_it.next() orelse return fatal("missing arg after {q}", .{arg})
        else
            std.mem.cutPrefix(u8, arg, "--gdb=")) |gdb_path|
        {
            gdb_path_arg = gdb_path;
        } else if (if (std.mem.eql(u8, arg, "--lldb"))
            arg_it.next() orelse return fatal("missing arg after {q}", .{arg})
        else
            std.mem.cutPrefix(u8, arg, "--lldb=")) |lldb_path|
        {
            lldb_path_arg = lldb_path;
        } else if (if (std.mem.eql(u8, arg, "--matrix"))
            arg_it.next() orelse return fatal("missing arg after {q}", .{arg})
        else
            std.mem.cutPrefix(u8, arg, "--matrix=")) |matrix_path|
        {
            matrix_path_arg = matrix_path;
        } else if (if (std.mem.eql(u8, arg, "--target"))
            arg_it.next() orelse return fatal("missing arg after {q}", .{arg})
        else
            std.mem.cutPrefix(u8, arg, "--target=")) |target|
        {
            try target_args.append(init.gpa, target);
        } else if (if (std.mem.eql(u8, arg, "--target-filter"))
            arg_it.next() orelse return fatal("missing arg after {q}", .{arg})
        else
            std.mem.cutPrefix(u8, arg, "--target-filter=")) |target_filter|
        {
            try target_filter_args.append(init.gpa, target_filter);
        } else if (std.mem.eql(u8, arg, "--keep-matrix")) {
            skip_matrix_arg = false;
        } else if (std.mem.eql(u8, arg, "--skip-matrix")) {
            skip_matrix_arg = true;
        } else {
            if (test_path_arg) |_| return fatal("unexpected arg {q}", .{arg});
            test_path_arg = arg;
        }
    }

    const test_path = test_path_arg orelse return fatal("missing \"path/to/test\" arg", .{});
    const zig_path = zig_path_arg orelse return fatal("missing \"--zig=path/to/zig\" arg", .{});

    const cwd: std.Io.Dir = .cwd();
    var inherit_dirs: std.ArrayList(std.Io.Dir) = try .initCapacity(arena, 3);
    var inherit_files: std.ArrayList(std.Io.File) = try .initCapacity(arena, 4);

    args.test_file = cwd.openFile(init.io, test_path, .{
        .allow_directory = true,
    }) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| return fatal("unable to open {q}: {t}", .{ test_path, e }),
    };
    inherit_files.appendAssumeCapacity(args.test_file);
    defer args.test_file.close(init.io);
    args.test_kind = (args.test_file.stat(init.io) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| return fatal("unable to stat {q}: {t}", .{ test_path, e }),
    }).kind;

    args.zig_exe = cwd.openFile(init.io, zig_path, .{}) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| return fatal("unable to open {q}: {t}", .{ zig_path, e }),
    };
    inherit_files.appendAssumeCapacity(args.zig_exe);
    defer args.zig_exe.close(init.io);

    if (lib_path_arg) |lib_path| {
        const lib_dir = cwd.openDir(init.io, lib_path, .{}) catch |err| switch (err) {
            error.Canceled => |e| return e,
            else => |e| return fatal("unable to open {q}: {t}", .{ lib_path, e }),
        };
        args.lib_dir = lib_dir;
        inherit_dirs.appendAssumeCapacity(lib_dir);
    }
    defer if (args.lib_dir) |lib_dir| lib_dir.close(init.io);

    if (libc_runtimes_path_arg) |libc_runtimes_path| {
        const libc_runtimes_dir = cwd.openDir(
            init.io,
            libc_runtimes_path,
            .{},
        ) catch |err| switch (err) {
            error.Canceled => |e| return e,
            else => |e| return fatal("unable to open {q}: {t}", .{ libc_runtimes_path, e }),
        };
        args.libc_runtimes_dir = libc_runtimes_dir;
        inherit_dirs.appendAssumeCapacity(libc_runtimes_dir);
    }
    defer if (args.libc_runtimes_dir) |libc_runtimes_dir| libc_runtimes_dir.close(init.io);

    if (gdb_path_arg) |gdb_path| {
        const gdb_exe = cwd.openFile(init.io, gdb_path, .{}) catch |err| switch (err) {
            error.Canceled => |e| return e,
            else => |e| return fatal("unable to open {q}: {t}", .{ gdb_path, e }),
        };
        args.gdb_exe = gdb_exe;
        inherit_files.appendAssumeCapacity(gdb_exe);
    }
    defer if (args.gdb_exe) |gdb_exe| gdb_exe.close(init.io);

    if (lldb_path_arg) |lldb_path| {
        const lldb_exe = cwd.openFile(init.io, lldb_path, .{}) catch |err| switch (err) {
            error.Canceled => |e| return e,
            else => |e| return fatal("unable to open {q}: {t}", .{ lldb_path, e }),
        };
        args.lldb_exe = lldb_exe;
        inherit_files.appendAssumeCapacity(lldb_exe);
    }
    defer if (args.lldb_exe) |lldb_exe| lldb_exe.close(init.io);

    const src_dir_path = "src_" ++ std.fmt.hex(rand_int: {
        var rand_int: u64 = undefined;
        init.io.random(@ptrCast(&rand_int));
        break :rand_int rand_int;
    });
    args.src_dir = cwd.createDirPathOpen(init.io, src_dir_path, .{}) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| return fatal("unable to create {q}: {t}", .{ src_dir_path, e }),
    };
    inherit_dirs.appendAssumeCapacity(args.src_dir);
    defer {
        args.src_dir.close(init.io);
        if (!args.keep_src) cwd.deleteTree(init.io, src_dir_path) catch |err| {
            std.log.err("unable to delete tree {q}: {t}", .{ src_dir_path, err });
        };
    }

    args.targets = target_args.items;
    args.target_filters = target_filter_args.items;
    matrix: {
        const matrix_path = matrix_path_arg orelse try std.Io.Dir.path.join(arena, &.{
            std.Io.Dir.path.dirname(test_path) orelse break :matrix, "matrix.zon",
        });
        const matrix_file = cwd.openFile(init.io, matrix_path, .{
            .allow_directory = true,
        }) catch |err| switch (err) {
            error.Canceled => |e| return e,
            else => |e| {
                switch (e) {
                    else => {},
                    error.FileNotFound => if (matrix_path_arg == null) break :matrix,
                }
                return fatal("unable to open {q}: {t}", .{ matrix_path, e });
            },
        };
        defer matrix_file.close(init.io);
        var matrix_fr = matrix_file.reader(init.io, &.{});
        const matrix_buffer = try arena.allocSentinel(u8, @intCast(
            matrix_fr.getSize() catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| return fatal("unable to get size of {q}: {t}", .{ matrix_path, e }),
            },
        ), 0);
        const matrix_len = matrix_fr.interface.readSliceShort(matrix_buffer) catch |err| switch (err) {
            error.ReadFailed => switch (matrix_fr.err.?) {
                error.Canceled => |e| return e,
                else => |e| return fatal("unable to read {q}: {t}", .{ matrix_path, e }),
            },
        };
        matrix_buffer[matrix_len] = 0;
        const matrix_source =
            (arena.remap(matrix_buffer, matrix_len + 1) orelse matrix_buffer)[0..matrix_len :0];
        var diagnostics: std.zon.parse.Diagnostics = undefined;
        args.matrix = std.zon.parse.fromSlice([]const Target, .{
            .gpa = init.gpa,
            .arena = arena,
            .source = matrix_source,
            .diagnostics = &diagnostics,
        }) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            error.ParseZon => return fatal("{f}", .{diagnostics.fmt(matrix_path)}),
        };
    }
    args.skip_matrix = skip_matrix_arg orelse (matrix_path_arg == null and args.targets.len > 0);

    var child = std.process.spawn(init.io, .{
        .argv = &.{ self, "--listen=-" },
        .stdin = .pipe,
        .stdout = .pipe,
        .inherit_dirs = inherit_dirs.items,
        .inherit_files = inherit_files.items,
    }) catch |err| switch (err) {
        error.Canceled, error.OutOfMemory => |e| return e,
        else => |e| return fatal("unable to spawn runner: {t}", .{e}),
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
            else => |e| return fatal("unable to receive message: {t}", .{e}),
        },
        error.WriteFailed => switch (child_writer.err.?) {
            error.Canceled => |e| return e,
            else => |e| return fatal("unable to send message: {t}", .{e}),
        },
        error.EndOfStream => return fatal("no more messages", .{}),
    };
    _ = child.wait(init.io) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => |e| return fatal("unable to wait for runner: {t}", .{e}),
    };
    return if (success) 0 else 1;
}
fn fatal(comptime fmt: []const u8, args: anytype) u8 {
    std.log.err(fmt, args);
    return 1;
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

    var matrix: std.array_hash_map.String(Target.State) = .empty;
    try matrix.ensureTotalCapacity(arena, args.matrix.len + args.targets.len);
    for (args.matrix) |target| {
        const target_string = try arena.print("{f}", .{target});
        matrix.putAssumeCapacity(target_string, .{
            .skipped = args.skip_matrix or for (args.target_filters) |target_filter| {
                if (std.mem.find(u8, target_string, target_filter)) |_| break false;
            } else args.target_filters.len > 0,
        });
    }
    for (args.targets) |target| matrix.putAssumeCapacity(target, .{ .skipped = false });

    var argv: std.ArrayList(u8) = .empty;
    defer argv.deinit(gpa);
    const Arg = std.zig.Client.Message.Arg;

    if (args.log_level) |log_level| {
        try argv.append(gpa, @backingInt(Arg.string));
        try argv.appendSlice(gpa, "--log=");
        try argv.appendSlice(gpa, @tagName(log_level));
        try argv.append(gpa, 0);
    }

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

    if (args.lib_dir) |lib_dir| {
        try argv.append(gpa, @backingInt(Arg.prefix));
        try argv.appendSlice(gpa, "--lib=");
        try argv.append(gpa, 0);
        try argv.append(gpa, @backingInt(Arg.input_dir));
        try argv.appendSlice(gpa, @ptrCast(&lib_dir.handle));
    }

    try argv.append(gpa, @backingInt(Arg.prefix));
    try argv.appendSlice(gpa, "--src=");
    try argv.append(gpa, 0);
    try argv.append(gpa, @backingInt(Arg.output_dir));
    try argv.appendSlice(gpa, @ptrCast(&args.src_dir.handle));

    for (matrix.keys(), matrix.values()) |target, state| {
        try argv.append(gpa, @backingInt(Arg.string));
        try argv.appendSlice(gpa, if (state.skipped) "--skipped-target=" else "--target=");
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
                    return false;
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
                        return false;
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

pub const std_options: std.Options = .{
    .logFn = log,
};
var global_log_level: std.log.Level = .info;
fn log(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    if (@backingInt(level) > @backingInt(global_log_level)) return;
    std.log.defaultLog(level, scope, format, args);
}

const assert = std.debug.assert;
const builtin = @import("builtin");
const std = @import("std");
const Pattern = @import("Runner/match.zig").Pattern;
const Runner = @This();
const Target = @import("Runner/Target.zig");
