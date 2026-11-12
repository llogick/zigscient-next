const std = @import("std");
const Io = std.Io;
const Dir = std.Io.Dir;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    const zig_src_lib_path = args[1];
    const mingw_src_path = args[2];
    const zig_src_libs_mingw_path = args[3];

    const dest_mingw_crt_path = try Dir.path.join(arena, &.{
        zig_src_lib_path, "libc", "mingw",
    });
    const src_mingw_crt_path = try Dir.path.join(arena, &.{
        mingw_src_path, "mingw-w64-crt",
    });

    // Update only the set of existing files we have already chosen to include
    // in zig's installation.

    var dest_crt_dir = Dir.cwd().openDir(io, dest_mingw_crt_path, .{ .iterate = true }) catch |err| {
        std.log.err("unable to open directory '{s}': {t}", .{ dest_mingw_crt_path, err });
        std.process.exit(1);
    };
    defer dest_crt_dir.close(io);

    var src_crt_dir = Dir.cwd().openDir(io, src_mingw_crt_path, .{ .iterate = true }) catch |err| {
        std.log.err("unable to open directory '{s}': {t}", .{ src_mingw_crt_path, err });
        std.process.exit(1);
    };
    defer src_crt_dir.close(io);

    {
        var walker = try dest_crt_dir.walk(arena);
        defer walker.deinit();

        var fail = false;

        while (try walker.next(io)) |entry| {
            if (entry.kind != .file) continue;

            src_crt_dir.copyFile(entry.path, dest_crt_dir, entry.path, io, .{}) catch |err| switch (err) {
                error.FileNotFound => {
                    const keep = for (kept_crt_files) |item| {
                        if (std.mem.eql(u8, entry.path, item)) break true;
                        if (std.mem.startsWith(u8, entry.path, "winpthreads/")) break true;
                    } else false;

                    if (!keep) {
                        std.log.warn("deleting {s}", .{entry.path});
                        try dest_crt_dir.deleteFile(io, entry.path);
                    }
                },
                else => {
                    std.log.err("unable to copy {s}: {t}", .{ entry.path, err });
                    fail = true;
                },
            };
        }

        if (fail) std.process.exit(1);
    }

    {
        const dest_mingw_winpthreads_path = try Dir.path.join(arena, &.{
            zig_src_lib_path, "libc", "mingw", "winpthreads",
        });
        const src_mingw_libraries_winpthreads_src_path = try Dir.path.join(arena, &.{
            mingw_src_path, "mingw-w64-libraries", "winpthreads", "src",
        });

        var dest_winpthreads_dir = Dir.cwd().openDir(io, dest_mingw_winpthreads_path, .{ .iterate = true }) catch |err| {
            std.log.err("unable to open directory '{s}': {s}", .{ dest_mingw_winpthreads_path, @errorName(err) });
            std.process.exit(1);
        };
        defer dest_winpthreads_dir.close(io);

        var src_winpthreads_dir = Dir.cwd().openDir(io, src_mingw_libraries_winpthreads_src_path, .{ .iterate = true }) catch |err| {
            std.log.err("unable to open directory '{s}': {s}", .{ src_mingw_libraries_winpthreads_src_path, @errorName(err) });
            std.process.exit(1);
        };
        defer src_winpthreads_dir.close(io);

        {
            var walker = try dest_winpthreads_dir.walk(arena);
            defer walker.deinit();

            var fail = false;

            while (try walker.next(io)) |entry| {
                if (entry.kind != .file) continue;

                src_winpthreads_dir.copyFile(entry.path, dest_winpthreads_dir, entry.path, io, .{}) catch |err| switch (err) {
                    error.FileNotFound => {
                        std.log.warn("deleting {s}", .{entry.path});
                        try dest_winpthreads_dir.deleteFile(io, entry.path);
                    },
                    else => {
                        std.log.err("unable to copy {s}: {t}", .{ entry.path, err });
                        fail = true;
                    },
                };
            }

            if (fail) std.process.exit(1);
        }
    }

    var libs_with_defs: std.StringHashMap(void) = .init(arena);

    {
        // Also add all new def and def.in files.
        var walker = try src_crt_dir.walkSelectively(arena);
        defer walker.deinit();

        var fail = false;

        while (try walker.next(io)) |entry| {
            switch (entry.kind) {
                .directory => {
                    switch (entry.depth()) {
                        1 => if (def_dirs.has(entry.basename)) {
                            try walker.enter(io, entry);
                            continue;
                        },
                        else => {
                            // The top-level directory was already validated
                            try walker.enter(io, entry);
                            continue;
                        },
                    }
                },
                .file => {},
                else => continue,
            }

            const ext = for (def_exts) |def_ext| {
                if (std.mem.endsWith(u8, entry.path, def_ext)) break def_ext;
            } else continue;

            const blacklisted = for (blacklisted_defs) |item| {
                if (std.mem.eql(u8, entry.basename, item)) break true;
            } else false;

            if (blacklisted) continue;

            if (std.mem.endsWith(u8, entry.basename, "_windowsapp.def"))
                continue;

            if (std.mem.endsWith(u8, entry.basename, "_onecore.def"))
                continue;

            src_crt_dir.copyFile(entry.path, dest_crt_dir, entry.path, io, .{}) catch |err| {
                std.log.err("unable to copy {s}: {t}", .{ entry.path, err });
                fail = true;
            };

            const dir_name = Dir.path.dirname(entry.path) orelse continue;
            if (def_dirs.has(dir_name) and !std.mem.eql(u8, dir_name, "def-include")) {
                try libs_with_defs.put(try arena.dupe(u8, entry.basename[0 .. entry.basename.len - ext.len]), {});
            }
        }
        if (fail) std.process.exit(1);
    }

    {
        const src_lib_common_path = try Dir.path.join(arena, &.{
            src_mingw_crt_path, "lib-common",
        });

        var src_lib_common_dir = Dir.cwd().openDir(io, src_lib_common_path, .{ .iterate = true }) catch |err| {
            std.log.err("unable to open directory '{s}': {t}", .{ src_lib_common_path, err });
            std.process.exit(1);
        };

        defer src_lib_common_dir.close(io);

        var merged_libs: std.ArrayList(MergedLib) = .empty;
        var it = src_lib_common_dir.iterate();

        outer: while (try it.next(io)) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".mri")) continue;

            for (skipped_mri_scripts) |item| {
                if (std.mem.eql(u8, entry.name, item)) continue :outer;
            }

            const script = try src_lib_common_dir.readFileAlloc(io, entry.name, arena, .unlimited);
            const merged_lib = try parseMri(arena, entry.name, script);

            if (libs_with_defs.contains(merged_lib.name)) continue;

            for (merged_lib.def_names) |def_name| {
                if (!libs_with_defs.contains(def_name)) {
                    std.log.err("unable to find def file '{s}.def' for {s}", .{ def_name, merged_lib.name });
                    std.process.exit(1);
                }
            }

            try merged_libs.append(arena, merged_lib);
        }

        std.mem.sortUnstable(MergedLib, merged_libs.items, {}, MergedLib.lessThan);

        var aw: Io.Writer.Allocating = .init(arena);
        const w = &aw.writer;

        try w.writeAll(
            \\// DO NOT MODIFY. This file is generated by tools/update_mingw.zig.
            \\.{
            \\
        );

        for (merged_libs.items) |merged_lib| {
            try w.print("    .{{ \"{f}\", .{{\n", .{std.zig.fmtString(merged_lib.name)});
            for (merged_lib.def_names) |def_name| {
                try w.print("        \"{f}\",\n", .{std.zig.fmtString(def_name)});
            }
            try w.writeAll("    } },\n");
        }

        try w.writeAll("}\n");

        var dest_libs_mingw_dir = Dir.cwd().openDir(io, zig_src_libs_mingw_path, .{}) catch |err| {
            std.log.err("unable to open directory '{s}': {t}", .{ zig_src_libs_mingw_path, err });
            std.process.exit(1);
        };

        defer dest_libs_mingw_dir.close(io);

        try dest_libs_mingw_dir.writeFile(io, .{ .sub_path = "merged_import_libs.zon", .data = aw.written() });
    }

    return std.process.cleanExit(io);
}

const MergedLib = struct {
    name: []const u8,
    def_names: []const []const u8,

    fn lessThan(_: void, a: MergedLib, b: MergedLib) bool {
        return std.mem.lessThan(u8, a.name, b.name);
    }
};

fn parseMri(arena: std.mem.Allocator, script_name: []const u8, script: []const u8) !MergedLib {
    var name: ?[]const u8 = null;
    var def_names: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.splitScalar(u8, script, '\n');

    while (lines.next()) |line_untrimmed| {
        const line = std.mem.trim(u8, line_untrimmed, " \t\r");

        if (line.len == 0 or line[0] == ';' or line[0] == '*') continue;

        var words = std.mem.tokenizeAny(u8, line, " \t");
        const command = words.next().?;
        const operand = words.next();

        if (words.next() != null) {
            std.log.err("too many operands for '{s}' in {s}", .{ command, script_name });
            std.process.exit(1);
        }

        if (std.ascii.eqlIgnoreCase(command, "CREATE")) {
            if (name != null) {
                std.log.err("multiple 'CREATE' commands in {s}", .{script_name});
                std.process.exit(1);
            }

            if (operand == null) {
                std.log.err("missing operand for 'CREATE' command in {s}", .{script_name});
                std.process.exit(1);
            }

            name = libName(operand.?);

            if (name == null) {
                std.log.err("bad lib name in {s}", .{script_name});
                std.process.exit(1);
            }
        } else if (std.ascii.eqlIgnoreCase(command, "ADDLIB")) {
            if (name == null) {
                std.log.err("ADDLIB command before CREATE command in {s}", .{script_name});
                std.process.exit(1);
            }

            if (operand == null) {
                std.log.err("missing operand for 'ADDLIB' command in {s}", .{script_name});
                std.process.exit(1);
            }

            try def_names.append(arena, libName(operand.?));
        } else if (!std.ascii.eqlIgnoreCase(command, "SAVE") and !std.ascii.eqlIgnoreCase(command, "END")) {
            std.log.err("unknown command '{s}' in {s}", .{ command, script_name });
            std.process.exit(1);
        }
    }

    if (def_names.items.len == 0) {
        std.log.err("mri script '{s}' added no libs", .{script_name});
        std.process.exit(1);
    }

    return .{
        .name = name.?,
        .def_names = def_names.items,
    };
}

fn libName(archive_name: []const u8) []const u8 {
    if (!std.mem.startsWith(u8, archive_name, "lib") or !std.mem.endsWith(u8, archive_name, ".a")) {
        std.log.err("mri script creates lib with bad name: {s}", .{archive_name});
        std.process.exit(1);
    }

    const name = archive_name["lib".len .. archive_name.len - ".a".len];

    if (name.len == 0 or std.mem.findAny(u8, name, "/\\") != null) {
        std.log.err("mri script creates lib with bad name: {s}", .{archive_name});
        std.process.exit(1);
    }

    return name;
}

const kept_crt_files = [_][]const u8{
    "COPYING",
    "include" ++ Dir.path.sep_str ++ "config.h",
};

const def_exts = [_][]const u8{
    ".def",
    ".def.in",
};

const def_dirs = std.StaticStringMap(void).initComptime(.{
    .{"lib32"},
    .{"lib64"},
    .{"libarm32"},
    .{"libarm64"},
    .{"lib-common"},
    .{"def-include"},
});

const blacklisted_defs = [_][]const u8{
    "crtdll.def.in",

    "msvcp60.def",
    "msvcp110.def",
    "msvcp120_app.def.in",
    "msvcp120_clr0400.def",

    "msvcr40d.def.in",
    "msvcr70.def.in",
    "msvcr70d.def.in",
    "msvcr71.def.in",
    "msvcr71d.def.in",
    "msvcr80.def.in",
    "msvcr80d.def.in",
    "msvcr90.def.in",
    "msvcr90d.def.in",
    "msvcr100.def.in",
    "msvcr100d.def.in",
    "msvcr110.def.in",
    "msvcr110d.def.in",
    "msvcr120.def.in",
    "msvcr120d.def.in",
    "msvcr120_app.def.in",

    "msvcrt.def.in",
    "msvcrtd.def.in",
    "msvcrt10.def.in",
    "msvcrt20.def.in",
    "msvcrt40.def.in",
};

const skipped_mri_scripts = [_][]const u8{
    "msvcr80.mri",
    "msvcr80d.mri",
    "msvcr90.mri",
    "msvcr90d.mri",
    "msvcr100.mri",
    "msvcr100d.mri",
    "msvcr110.mri",
    "msvcr110d.mri",
    "msvcr120.mri",
    "msvcr120d.mri",
    "msvcr120_app.mri",
    "msvcrt-os.mri",
    "ucrt.mri",
    "ucrtapp.mri",

    "onecore.mri",
    "windowsapp.mri",
};
