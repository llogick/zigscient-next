objects: std.ArrayList(Object) = .empty,

pub fn deinit(self: *Archive, allocator: Allocator) void {
    self.objects.deinit(allocator);
}

pub fn unpack(self: *Archive, macho_file: *MachO, path: Path, handle_index: File.HandleIndex, fat_arch: ?fat.Arch) !void {
    const comp = macho_file.base.comp;
    const io = comp.io;
    const gpa = comp.gpa;
    const diags = &comp.link_diags;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    const handle = macho_file.getFileHandle(handle_index);
    const offset = if (fat_arch) |ar| ar.offset else 0;
    const end_pos = if (fat_arch) |ar| offset + ar.size else (try handle.stat(io)).size;

    var pos: usize = offset + macho.ARMAG.len;
    while (true) {
        if (pos >= end_pos) break;
        if (!mem.isAligned(pos, 2)) pos += 1;

        var hdr_buffer: [@sizeOf(macho.ar_hdr)]u8 = undefined;
        {
            const amt = try handle.readPositionalAll(io, &hdr_buffer, pos);
            if (amt != @sizeOf(macho.ar_hdr)) return error.InputOutput;
        }
        const hdr = @as(*align(1) const macho.ar_hdr, @ptrCast(&hdr_buffer)).*;
        pos += @sizeOf(macho.ar_hdr);

        if (!mem.eql(u8, &hdr.ar_fmag, macho.ARFMAG)) {
            return diags.failParse(path, "invalid header delimiter: expected '{f}', found '{f}'", .{
                std.ascii.hexEscape(macho.ARFMAG, .lower), std.ascii.hexEscape(&hdr.ar_fmag, .lower),
            });
        }

        var hdr_size = try hdr.size();
        const name = name: {
            const len = hdr.nameLength() orelse break :name hdr.name();
            hdr_size -= len;
            const buf = try arena.allocator().alloc(u8, len);
            const amt = try handle.readPositionalAll(io, buf, pos);
            if (amt != len) return error.InputOutput;
            pos += len;
            const actual_len = mem.findScalar(u8, buf, @as(u8, 0)) orelse len;
            break :name buf[0..actual_len];
        };
        defer pos += hdr_size;

        if (mem.eql(u8, name, macho.SYMDEF) or
            mem.eql(u8, name, macho.SYMDEF64) or
            mem.eql(u8, name, macho.SYMDEF_SORTED) or
            mem.eql(u8, name, macho.SYMDEF64_SORTED)) continue;

        const abs_path = try std.fs.path.resolvePosix(gpa, &.{
            comp.dirs.cwd,
            path.root_dir.path orelse ".",
            path.sub_path,
        });
        errdefer gpa.free(abs_path);

        const o_basename = try gpa.dupe(u8, name);
        errdefer gpa.free(o_basename);

        const object: Object = .{
            .offset = pos,
            .in_archive = .{
                .path = abs_path,
                .size = hdr_size,
            },
            .path = o_basename,
            .file_handle = handle_index,
            .index = undefined,
            .alive = false,
            .mtime = hdr.date() catch 0,
        };

        log.debug("extracting object '{s}' from archive '{f}'", .{ o_basename, path });

        try self.objects.append(gpa, object);
    }
}

pub fn writeHeader(
    object_name: []const u8,
    object_size: usize,
    writer: *Writer,
) !void {
    var hdr: macho.ar_hdr = .{
        .ar_name = @splat(' '),
        .ar_date = "0           ".*,
        .ar_uid = "0     ".*,
        .ar_gid = "0     ".*,
        .ar_mode = "0       ".*,
        .ar_size = @splat(' '),
        .ar_fmag = macho.ARFMAG.*,
    };

    const object_name_start = writer.end + @sizeOf(macho.ar_hdr);
    const object_start = mem.alignForward(usize, object_name_start + object_name.len + 1, 8);
    const object_name_len = object_start - object_name_start;
    const total_object_size = object_size + object_name_len;

    {
        var stream: Writer = .fixed(&hdr.ar_name);
        stream.print("#1/{d}", .{object_name_len}) catch unreachable;
    }
    {
        var stream: Writer = .fixed(&hdr.ar_size);
        stream.print("{d}", .{total_object_size}) catch unreachable;
    }

    try writer.writeAll(mem.asBytes(&hdr));
    try writer.print("{s}\x00", .{object_name});

    const padding = object_name_len - object_name.len - 1;
    if (padding > 0) {
        try writer.splatByteAll(0, padding);
    }
}

pub const ArSymtab = struct {
    entries: std.ArrayList(Entry) = .empty,
    strtab: StringTable = .{},

    pub fn deinit(ar: *ArSymtab, allocator: Allocator) void {
        ar.entries.deinit(allocator);
        ar.strtab.deinit(allocator);
    }

    pub fn sort(ar: *ArSymtab) void {
        mem.sort(Entry, ar.entries.items, {}, Entry.lessThan);
    }

    pub fn size(ar: ArSymtab, format: Format) usize {
        const ptr_width = ptrWidth(format);
        return ptr_width + ar.entries.items.len * 2 * ptr_width + ptr_width + mem.alignForward(usize, ar.strtab.buffer.items.len, ptr_width);
    }

    pub fn write(ar: ArSymtab, format: Format, macho_file: *MachO, writer: *Writer) !void {
        const ptr_width = ptrWidth(format);
        // Header
        try writeHeader(macho.SYMDEF, ar.size(format), writer);
        // Symtab size
        try writeInt(format, ar.entries.items.len * 2 * ptr_width, writer);
        // Symtab entries
        for (ar.entries.items) |entry| {
            const file_off = switch (macho_file.getFile(entry.file).?) {
                .zig_object => |x| x.output_ar_state.file_off,
                .object => |x| x.output_ar_state.file_off,
                else => unreachable,
            };
            // Name offset
            try writeInt(format, entry.off, writer);
            // File offset
            try writeInt(format, file_off, writer);
        }
        // Strtab size
        const strtab_size = mem.alignForward(usize, ar.strtab.buffer.items.len, ptr_width);
        const padding = strtab_size - ar.strtab.buffer.items.len;
        try writeInt(format, strtab_size, writer);
        // Strtab
        try writer.writeAll(ar.strtab.buffer.items);
        if (padding > 0) {
            try writer.splatByteAll(0, padding);
        }
    }

    const PrintFormat = struct {
        ar: ArSymtab,
        macho_file: *MachO,

        fn default(f: PrintFormat, bw: *Writer) Writer.Error!void {
            const ar = f.ar;
            const macho_file = f.macho_file;
            for (ar.entries.items, 0..) |entry, i| {
                const name = ar.strtab.getAssumeExists(entry.off);
                const file = macho_file.getFile(entry.file).?;
                try bw.print("  {d}: {s} in file({d})({f})\n", .{ i, name, entry.file, file.fmtPath() });
            }
        }
    };

    pub fn fmt(ar: ArSymtab, macho_file: *MachO) std.fmt.Alt(PrintFormat, PrintFormat.default) {
        return .{ .data = .{ .ar = ar, .macho_file = macho_file } };
    }

    const Entry = struct {
        /// Symbol name offset
        off: u32,
        /// Exporting file
        file: File.Index,

        pub fn lessThan(ctx: void, lhs: Entry, rhs: Entry) bool {
            _ = ctx;
            if (lhs.off == rhs.off) return lhs.file < rhs.file;
            return lhs.off < rhs.off;
        }
    };
};

pub const Format = enum {
    p32,
    p64,
};

pub fn ptrWidth(format: Format) usize {
    return switch (format) {
        .p32 => @as(usize, 4),
        .p64 => 8,
    };
}

pub fn writeInt(format: Format, value: u64, writer: *Writer) !void {
    switch (format) {
        .p32 => try writer.writeInt(u32, std.math.cast(u32, value) orelse return error.Overflow, .little),
        .p64 => try writer.writeInt(u64, value, .little),
    }
}

pub const ArState = struct {
    /// File offset of the ar_hdr describing the contributing
    /// object in the archive.
    file_off: u64 = 0,

    /// Total size of the contributing object (excludes ar_hdr and long name with padding).
    size: u64 = 0,
};

const fat = @import("fat.zig");
const link = @import("../../link.zig");
const log = std.log.scoped(.link);
const macho = std.macho;
const mem = std.mem;
const std = @import("std");
const Allocator = std.mem.Allocator;
const Path = std.Build.Cache.Path;
const Writer = std.Io.Writer;

const Archive = @This();
const File = @import("file.zig").File;
const MachO = @import("../MachO.zig");
const Object = @import("Object.zig");
const StringTable = @import("../StringTable.zig");
