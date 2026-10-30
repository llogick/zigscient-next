//! Static libraries (archives) use more-or-less the same format on ELF, Mach-O, and COFF (just with
//! slightly different names for special files, and occasionally slightly different padding rules).
//!
//! This file therefore handles parsing for all of them, and delegates parsing of each contained
//! object file to the appropriate implementation for that object format.

pub fn identify(r: *Io.Reader) error{ReadFailed}!bool {
    // ELF, Mach-O, and COFF all use sort of the same archive format, so have the same magic bytes.
    const armag = "!<arch>\n";
    comptime assert(std.mem.eql(u8, armag, std.elf.ARMAG));
    comptime assert(std.mem.eql(u8, armag, std.macho.ARMAG));
    comptime assert(std.mem.eql(u8, armag, std.coff.archive_signature));

    if (r.take(armag.len)) |magic| {
        return std.mem.eql(u8, magic, armag);
    } else |err| switch (err) {
        error.EndOfStream => return false,
        else => |e| return e,
    }
}

pub fn dump(d: *Objdump) Objdump.Error!void {
    // The archive magic was already consumed by `identify`.
    const r = d.reader();
    const w = d.w;

    const archive_input_name = d.input_name;

    try w.print("{s}: ar archive (static library)\n\n", .{archive_input_name});

    var arena_state: std.heap.ArenaAllocator = .init(d.gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const Kind = union(enum) {
        bsd_symbols,
        sysv_symbols: enum { @"32", @"64" },
        windows_second_symbols,
        sysv_long_names,
        object: []const u8,
    };

    var seen_bsd_symbols: bool = false;
    var seen_sysv_symbols: bool = false;
    var seen_windows_second_symbols: bool = false;
    var seen_object: bool = false;

    var sysv_long_names: ?[]u8 = null;

    // Key is file offset of header. Allocated into `d.gpa`.
    var coff_sorted_members: std.array_hash_map.Auto(u64, CoffSortedMember) = .empty;
    defer coff_sorted_members.deinit(d.gpa);

    while (true) {
        const header_off = d.logicalPos();
        // Endianness is irrelevant here because this is a text format (so no multi-byte integers).
        const hdr = r.takeStruct(std.elf.ar_hdr, .native) catch |err| switch (err) {
            error.EndOfStream => {
                if (d.atEnd()) break;
                return d.fail("unexpected eof in archive header", .{});
            },
            else => |e| return e,
        };

        if (!std.mem.eql(u8, &hdr.ar_fmag, std.elf.ARFMAG)) {
            return d.fail("invalid trailing magic bytes in ar header", .{});
        }

        const name_in_header = std.mem.trimEnd(u8, &hdr.ar_name, " ");
        const kind: Kind = kind: {
            if (std.mem.eql(u8, name_in_header, std.macho.SYMDEF) or
                std.mem.eql(u8, name_in_header, std.macho.SYMDEF_SORTED) or
                std.mem.eql(u8, name_in_header, std.macho.SYMDEF64) or
                std.mem.eql(u8, name_in_header, std.macho.SYMDEF64_SORTED))
            {
                if (seen_sysv_symbols) {
                    try d.warn("conflicting symbol table types in archive", .{});
                } else if (seen_bsd_symbols) {
                    try d.warn("duplicate symbol table in archive", .{});
                } else if (seen_object) {
                    try d.warn("symbol table after object file in archive", .{});
                } else if (sysv_long_names != null) {
                    try d.warn("symbol table after file name table in archive", .{});
                }
                break :kind .bsd_symbols;
            }

            if (std.mem.eql(u8, name_in_header, "/")) {
                if (seen_bsd_symbols) {
                    try d.warn("conflicting symbol table types in archive", .{});
                } else if (seen_windows_second_symbols) {
                    try d.warn("duplicate symbol table in archive", .{});
                } else if (seen_object) {
                    try d.warn("symbol table after object file in archive", .{});
                } else if (sysv_long_names != null) {
                    try d.warn("symbol table after file name table in archive", .{});
                }

                if (seen_sysv_symbols) {
                    break :kind .windows_second_symbols;
                } else {
                    break :kind .{ .sysv_symbols = .@"32" };
                }
            }

            if (std.mem.eql(u8, name_in_header, "/SYM64/")) {
                if (seen_bsd_symbols) {
                    try d.warn("conflicting symbol table types in archive", .{});
                } else if (seen_sysv_symbols) {
                    try d.warn("duplicate symbol table in archive", .{});
                } else if (seen_object) {
                    try d.warn("symbol table after object file in archive", .{});
                } else if (sysv_long_names != null) {
                    try d.warn("symbol table after file name table in archive", .{});
                }

                break :kind .{ .sysv_symbols = .@"64" };
            }

            if (std.mem.eql(u8, name_in_header, "//")) {
                if (seen_bsd_symbols) {
                    return d.fail("file name table in BSD-style archive", .{});
                } else if (sysv_long_names != null) {
                    return d.fail("duplicate file name table in archive", .{});
                } else if (seen_object) {
                    return d.fail("file name table after object file in archive", .{});
                }
                break :kind .sysv_long_names;
            }

            if (std.mem.startsWith(u8, name_in_header, "#1/")) {
                if (std.fmt.parseUnsigned(u32, name_in_header[3..], 10)) |len| {
                    if (sysv_long_names != null) {
                        try d.warn("BSD-style long file name after string table in archive", .{});
                    }
                    const name = r.readAllocAll(arena, len) catch |err| switch (err) {
                        error.EndOfStream => return d.fail("unexpected eof in archive member name", .{}),
                        else => |e| return e,
                    };
                    break :kind .{ .object = std.mem.sliceTo(name, 0) };
                } else |err| switch (err) {
                    error.Overflow, error.InvalidCharacter => {},
                }
            }

            if (std.mem.startsWith(u8, name_in_header, "/")) {
                if (std.fmt.parseUnsigned(u32, name_in_header[1..], 10)) |offset| {
                    const strtab = sysv_long_names orelse return d.fail(
                        "archive member name {q} requires preceding \"//\" member",
                        .{name_in_header},
                    );
                    if (offset >= strtab.len) return d.fail(
                        "archive member name {q} exceeds length of \"//\"",
                        .{name_in_header},
                    );
                    if (std.mem.findPos(u8, strtab, offset, "/\n") orelse
                        std.mem.findScalarPos(u8, strtab, offset, 0)) |end|
                    {
                        break :kind .{ .object = strtab[offset..end] };
                    } else {
                        try d.warn("unterminated archive member name in \"//\"", .{});
                        break :kind .{ .object = strtab[offset..] };
                    }
                } else |err| switch (err) {
                    error.Overflow, error.InvalidCharacter => {},
                }
            }

            break :kind .{ .object = std.mem.sliceTo(name_in_header, '/') };
        };

        const member_off = d.logicalPos();

        // The size field needs to parse successfully so that we know how long the member is!
        const size = hdr.size() catch |err| switch (err) {
            error.Overflow, error.InvalidCharacter => |e| return d.fail(
                "invalid size {q} in ar header: {t}",
                .{ std.mem.trimEnd(u8, &hdr.ar_size, " "), e },
            ),
        };

        const full_object_name: []const u8 = switch (kind) {
            .bsd_symbols => {
                seen_bsd_symbols = true;
                if (d.opts.archive_symbols) {
                    try w.writeAll("TODO: dump symbols in BSD-style archive\n");
                }
                try d.seekTo(std.mem.alignForward(u64, member_off + size, 2));
                continue;
            },
            .sysv_symbols => |symtab_kind| {
                seen_sysv_symbols = true;
                if (d.opts.archive_symbols) {
                    assert(d.limited_reader == null);
                    assert(d.seek_base == 0);
                    defer {
                        d.limited_reader = null;
                        d.seek_base = 0;
                        d.input_name = archive_input_name;
                    }
                    var buf: [4096]u8 = undefined;
                    var limited_reader: Io.Reader.Limited = .init(
                        &d.raw_file_reader.interface,
                        .limited64(size),
                        &buf,
                    );
                    d.limited_reader = &limited_reader;
                    d.seek_base = member_off;
                    d.input_name = "archive symbol table";
                    const result = switch (symtab_kind) {
                        .@"32" => parseSysVSymbolTable(d, u32),
                        .@"64" => parseSysVSymbolTable(d, u64),
                    };
                    result catch |err| switch (err) {
                        error.AlreadyReported => {},
                        error.EndOfStream => switch (d.fail("contents truncated", .{})) {
                            error.AlreadyReported => {},
                            else => |e| return e,
                        },
                        else => |e| return e,
                    };
                }
                try d.seekTo(std.mem.alignForward(u64, member_off + size, 2));
                continue;
            },
            .windows_second_symbols => {
                seen_windows_second_symbols = true;
                if (d.opts.archive_symbols or d.opts.exports) {
                    assert(d.limited_reader == null);
                    assert(d.seek_base == 0);
                    defer {
                        d.limited_reader = null;
                        d.seek_base = 0;
                        d.input_name = archive_input_name;
                    }
                    var buf: [4096]u8 = undefined;
                    var limited_reader: Io.Reader.Limited = .init(
                        &d.raw_file_reader.interface,
                        .limited64(size),
                        &buf,
                    );
                    d.limited_reader = &limited_reader;
                    d.seek_base = member_off;
                    d.input_name = "second archive symbol table";
                    parseWindowsSecondSymbolTable(d, &coff_sorted_members) catch |err| switch (err) {
                        error.AlreadyReported => {},
                        error.EndOfStream => switch (d.fail("contents truncated", .{})) {
                            error.AlreadyReported => {},
                            else => |e| return e,
                        },
                        else => |e| return e,
                    };
                }
                try d.seekTo(std.mem.alignForward(u64, member_off + size, 2));
                continue;
            },
            .sysv_long_names => {
                sysv_long_names = r.readAllocAll(arena, size) catch |err| switch (err) {
                    error.EndOfStream => return d.fail("unexpected eof in archive file name table", .{}),
                    else => |e| return e,
                };
                try d.seekTo(std.mem.alignForward(u64, member_off + size, 2));
                continue;
            },
            .object => |name| name,
        };
        seen_object = true;

        for (d.opts.archive_member_filters) |filter| {
            if (std.mem.find(u8, full_object_name, filter) != null) break;
        } else if (d.opts.archive_member_filters.len > 0) {
            try d.seekTo(std.mem.alignForward(u64, member_off + size, 2));
            continue; // archive member name does not match filter
        }

        const object_name: []const u8 = switch (d.opts.basenames) {
            false => full_object_name,
            true => std.fs.path.basename(full_object_name),
        };

        const date_raw = std.mem.trimEnd(u8, &hdr.ar_date, " ");
        const uid_raw = std.mem.trimEnd(u8, &hdr.ar_uid, " ");
        const gid_raw = std.mem.trimEnd(u8, &hdr.ar_gid, " ");
        const mode_raw = std.mem.trimEnd(u8, &hdr.ar_mode, " ");

        if (d.opts.archive_member_headers) {
            try w.print(
                \\archive member header
                \\        off: {x}
                \\       name: {q}
                \\
            , .{
                d.fmtIntField(header_off, .offset),
                object_name,
            });

            if (parseUnsignedAllowEmpty(u32, date_raw, 8)) |timestamp| {
                const epoch_secs: std.time.epoch.EpochSeconds = .{ .secs = timestamp };
                const date = epoch_secs.getEpochDay().calculateYearDay();
                const time = epoch_secs.getDaySeconds();
                try w.print("       date: {d:0>4}-{d:0>2}-{d:0>2} {d:0>2}:{d:0>2}:{d:0>2}\n", .{
                    date.year,
                    date.calculateMonthDay().month.numeric(),
                    date.calculateMonthDay().day_index,
                    time.getHoursIntoDay(),
                    time.getMinutesIntoHour(),
                    time.getSecondsIntoMinute(),
                });
            } else |err| switch (err) {
                error.Overflow, error.InvalidCharacter => |e| try d.warn(
                    "invalid timestamp {q} in ar header: {t}",
                    .{ date_raw, e },
                ),
            }

            if (parseUnsignedAllowEmpty(u32, uid_raw, 8)) |uid| {
                try w.print("        uid: {d}\n", .{uid});
            } else |err| switch (err) {
                error.Overflow, error.InvalidCharacter => |e| try d.warn(
                    "invalid uid {q} in ar header: {t}",
                    .{ uid_raw, e },
                ),
            }

            if (parseUnsignedAllowEmpty(u32, gid_raw, 8)) |gid| {
                try w.print("        gid: {d}\n", .{gid});
            } else |err| switch (err) {
                error.Overflow, error.InvalidCharacter => |e| try d.warn(
                    "invalid gid {q} in ar header: {t}",
                    .{ gid_raw, e },
                ),
            }

            if (parseUnsignedAllowEmpty(u24, mode_raw, 8)) |mode| {
                try w.print("       mode: {o:0>3}\n", .{mode});
            } else |err| switch (err) {
                error.Overflow, error.InvalidCharacter => |e| try d.warn(
                    "invalid mode {q} in ar header: {t}",
                    .{ mode_raw, e },
                ),
            }

            try w.writeByte('\n');
        }

        if (coff_sorted_members.count() == 0) {
            assert(d.limited_reader == null);
            assert(d.seek_base == 0);
            defer {
                d.limited_reader = null;
                d.seek_base = 0;
                d.input_name = archive_input_name;
            }
            var buf: [4096]u8 = undefined;
            var limited_reader: Io.Reader.Limited = .init(
                &d.raw_file_reader.interface,
                .limited64(size),
                &buf,
            );
            d.limited_reader = &limited_reader;
            d.seek_base = member_off;
            d.input_name = try std.fmt.allocPrint(arena, "{s}({s})", .{
                archive_input_name,
                object_name,
            });
            d.dumpNonArchive() catch |err| switch (err) {
                else => |e| return e,
                // If parsing this archive member failed, that's fine; we can still do the next one.
                error.AlreadyReported => {},
            };
        } else {
            const ptr = coff_sorted_members.getPtr(header_off) orelse {
                return d.fail(
                    "member offset '{d}' does not exist in second archive symbol table",
                    .{header_off},
                );
            };
            ptr.* = .{
                .size = size,
                .name = try std.fmt.allocPrint(arena, "{s}({s})", .{
                    archive_input_name,
                    object_name,
                }),
            };
        }
        try d.seekTo(std.mem.alignForward(u64, member_off + size, 2));
    }

    for (coff_sorted_members.keys(), coff_sorted_members.values()) |header_offset, member| {
        const member_offset = header_offset + @sizeOf(std.elf.ar_hdr);
        try d.seekTo(member_offset);
        assert(d.limited_reader == null);
        assert(d.seek_base == 0);
        defer {
            d.limited_reader = null;
            d.seek_base = 0;
            d.input_name = archive_input_name;
        }
        var buf: [4096]u8 = undefined;
        var limited_reader: Io.Reader.Limited = .init(
            &d.raw_file_reader.interface,
            .limited64(member.size),
            &buf,
        );
        d.limited_reader = &limited_reader;
        d.seek_base = member_offset;
        d.input_name = member.name;
        d.dumpNonArchive() catch |err| switch (err) {
            else => |e| return e,
            // If parsing this archive member failed, that's fine; we can still do the next one.
            error.AlreadyReported => {},
        };
    }
}

fn parseUnsignedAllowEmpty(
    comptime T: type,
    buf: []const u8,
    base: u8,
) std.fmt.ParseIntError!T {
    if (buf.len == 0) return 0;
    return std.fmt.parseUnsigned(T, buf, base);
}

fn parseSysVSymbolTable(d: *Objdump, comptime Int: type) (Objdump.Error || error{EndOfStream})!void {
    const gpa = d.gpa;
    const r = d.reader();
    const w = d.w;

    const num_symbols = std.math.cast(
        usize,
        try r.takeInt(Int, .big),
    ) orelse return error.OutOfMemory;

    try w.print("{d} symbols in archive symbol table\n", .{
        d.fmtIntField(num_symbols, .size),
    });

    const offsets = try r.readSliceEndianAlloc(gpa, Int, num_symbols, .big);
    defer gpa.free(offsets);

    try w.print("  {s: ^18}  {s}\n", .{ "member offset", "symbol name" });
    for (offsets) |member_offset| {
        const name = r.takeSentinel(0) catch |err| switch (err) {
            error.StreamTooLong => return d.fail(
                "symbol name in archive symbol table exceeds maximum supported length",
                .{},
            ),
            else => |e| return e,
        };
        for (d.opts.symbol_filters) |filter| {
            if (std.mem.find(u8, name, filter) != null) break;
        } else if (d.opts.symbol_filters.len > 0) {
            continue;
        }
        try w.print("  {x:0>16}  {s}\n", .{
            d.fmtIntField(member_offset, .offset),
            name,
        });
    }

    try w.writeByte('\n');
}

fn parseWindowsSecondSymbolTable(
    d: *Objdump,
    sorted_members: *std.array_hash_map.Auto(u64, CoffSortedMember),
) (Objdump.Error || error{EndOfStream})!void {
    const gpa = d.gpa;
    const r = d.reader();
    const w = d.w;

    const num_members = try r.takeInt(u32, .little);

    const member_offsets = try r.readSliceEndianAlloc(gpa, u32, num_members, .little);
    defer gpa.free(member_offsets);

    const num_symbols = try r.takeInt(u32, .little);
    try w.print("{d} archive symbols, {d} archive members in second archive symbol table\n", .{
        d.fmtIntField(num_symbols, .size),
        d.fmtIntField(num_members, .size),
    });

    const symbol_member_indices = try r.readSliceEndianAlloc(gpa, u16, num_symbols, .little);
    defer gpa.free(symbol_member_indices);

    if (d.opts.exports_sort) {
        try sorted_members.ensureUnusedCapacity(gpa, member_offsets.len);
    }

    try w.print("  {s: >6}  {s: ^18}  {s}\n", .{ "member", "member offset", "symbol name" });
    for (symbol_member_indices) |member_index| {
        const name = r.takeSentinel(0) catch |err| switch (err) {
            error.StreamTooLong => return d.fail(
                "symbol name in archive symbol table exceeded maximum supported length",
                .{},
            ),
            else => |e| return e,
        };
        if (member_index == 0 or member_index > num_members) {
            return d.fail("symbol {q} has out-of-range member index '{d}'", .{ name, member_index });
        }
        if (d.opts.exports_sort) {
            sorted_members.putAssumeCapacity(member_offsets[member_index - 1], .init);
        }
        for (d.opts.symbol_filters) |filter| {
            if (std.mem.find(u8, name, filter) != null) break;
        } else if (d.opts.symbol_filters.len > 0) {
            continue;
        }
        try w.print("  {d: >6}  {x:0>16}  {s}\n", .{
            d.fmtIntField(member_index, .index),
            d.fmtIntField(member_offsets[member_index - 1], .offset),
            name,
        });
    }

    try w.writeByte('\n');

    if (d.opts.exports_sort) {
        // If any members were empty, they were never added to `sorted_members`; do that now.
        for (member_offsets) |member_offset| {
            sorted_members.putAssumeCapacity(member_offset, .init);
        }
    }
}

const CoffSortedMember = struct {
    size: u64,
    name: []const u8,
    const init: CoffSortedMember = .{ .size = 0, .name = "" };
};

const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;

const Objdump = @import("../Objdump.zig");
