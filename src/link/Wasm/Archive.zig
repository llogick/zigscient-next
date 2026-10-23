/// A list of long file names, delimited by a LF character (0x0a).
/// This is stored as a single slice of bytes, as the header-names
/// point to the character index of a file name, rather than the index
/// in the list.
/// Points into `file_contents`.
long_file_names: RelativeSlice,

/// Parsed table of contents.
/// Each symbol name points to a list of all definition
/// sites within the current static archive.
toc: Toc,

/// Key points into `LazyArchive` `file_contents`.
/// Value is allocated with gpa.
const Toc = std.array_hash_map.String(std.ArrayList(u32));

const ARMAG = std.elf.ARMAG;
const ARFMAG = std.elf.ARFMAG;

const RelativeSlice = struct {
    off: u32,
    len: u32,
};

const Header = extern struct {
    /// Member file name, sometimes / terminated.
    name: [16]u8,
    /// File date, decimal seconds since Epoch.
    date: [12]u8,
    /// User ID, in ASCII format.
    uid: [6]u8,
    /// Group ID, in ASCII format.
    gid: [6]u8,
    /// File mode, in ASCII octal.
    mode: [8]u8,
    /// File size, in ASCII decimal.
    size: [10]u8,
    /// Always contains ARFMAG.
    fmag: [2]u8,

    const NameOrIndex = union(enum) {
        name: []const u8,
        index: u32,
    };

    fn nameOrIndex(archive: Header) !NameOrIndex {
        const value = getValue(&archive.name);
        const slash_index = mem.findScalar(u8, value, '/') orelse return error.MalformedArchive;
        const len = value.len;
        if (slash_index == len - 1) {
            // Name stored directly
            return .{ .name = value };
        } else {
            // Name follows the header directly and its length is encoded in
            // the name field.
            const index = try std.fmt.parseInt(u32, value[slash_index + 1 ..], 10);
            return .{ .index = index };
        }
    }

    fn parsedSize(archive: Header) !u32 {
        const value = getValue(&archive.size);
        return std.fmt.parseInt(u32, value, 10);
    }

    fn getValue(raw: []const u8) []const u8 {
        return mem.trimEnd(u8, raw, &[_]u8{@as(u8, 0x20)});
    }
};

pub const Builder = struct {
    file_names: std.ArrayList(u8) = .empty,
    members: std.ArrayList(u8) = .empty,
    symbol_names: std.ArrayList(u8) = .empty,
    symbol_offsets: std.ArrayList(u32) = .empty,

    pub fn deinit(b: *Builder, gpa: Allocator) void {
        b.file_names.deinit(gpa);
        b.members.deinit(gpa);
        b.symbol_names.deinit(gpa);
        b.symbol_offsets.deinit(gpa);
        b.* = undefined;
    }

    pub fn append(b: *Builder, gpa: Allocator, name: []const u8, data: []const u8) !void {
        const member_offset: u32 = @intCast(b.members.items.len);
        const name_offset = b.file_names.items.len;

        try b.file_names.appendSlice(gpa, name);
        try b.file_names.appendSlice(gpa, "/\n");

        var name_buf: [16]u8 = undefined;
        const header_name = try std.mem.print(&name_buf, "/{d}", .{name_offset});
        const member_size = @sizeOf(Header) + mem.alignForward(usize, data.len, 2);
        var w: std.Io.Writer = .fixed(try b.members.addManyAsSlice(gpa, member_size));
        try writeHeader(&w, header_name, data.len);
        try w.writeAll(data);
        if (data.len % 2 != 0) try w.writeByte('\n');

        try b.collectSymbols(gpa, data, member_offset);
    }

    pub fn write(b: *Builder, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const symtab_size = 4 + 4 * b.symbol_offsets.items.len + b.symbol_names.items.len;
        const members_offset = ARMAG.len + @sizeOf(Header) + mem.alignForward(usize, symtab_size, 2) +
            @sizeOf(Header) + mem.alignForward(usize, b.file_names.items.len, 2);

        try w.writeAll(ARMAG);

        try writeHeader(w, "/", symtab_size);
        try w.writeInt(u32, @intCast(b.symbol_offsets.items.len), .big);
        for (b.symbol_offsets.items) |offset| try w.writeInt(u32, @intCast(members_offset + offset), .big);
        try w.writeAll(b.symbol_names.items);
        if (symtab_size % 2 != 0) try w.writeByte('\n');

        try writeHeader(w, "//", b.file_names.items.len);
        try w.writeAll(b.file_names.items);
        if (b.file_names.items.len % 2 != 0) try w.writeByte('\n');

        try w.writeAll(b.members.items);
    }

    fn collectSymbols(b: *Builder, gpa: Allocator, data: []const u8, member_offset: u32) !void {
        var reader: std.Io.Reader = .fixed(data);
        if (!mem.eql(u8, try reader.take(4), &std.wasm.magic)) return error.BadObjectMagic;
        if (!mem.eql(u8, try reader.take(4), &std.wasm.version)) return error.UnsupportedVersion;

        while (reader.seek < data.len) {
            const section_tag = try reader.takeEnum(std.wasm.Section, .little);
            const section_len = try reader.takeLeb128(u32);
            var section: std.Io.Reader = .fixed(try reader.take(section_len));
            if (section_tag != .custom) continue;

            const section_name = try readBytes(&section);
            if (mem.eql(u8, section_name, "linking")) {
                const section_version = try section.takeLeb128(u32);
                if (section_version != 2) return error.UnsupportedVersion;

                while (section.seek < section.buffer.len) {
                    const sub_type = try section.takeLeb128(u8);
                    const payload_len = try section.takeLeb128(u32);

                    var symtab: std.Io.Reader = .fixed(try section.take(payload_len));
                    if (sub_type != @backingInt(Object.SubsectionType.symbol_table)) continue;

                    const count = try symtab.takeLeb128(u32);
                    for (0..count) |_| {
                        const tag = try symtab.takeEnum(Object.Symbol.Tag, .little);
                        const flags = try symtab.takeLeb128(u32);
                        const symbol_flags: Wasm.SymbolFlags = @bitCast(flags);

                        var name: []const u8 = undefined;
                        switch (tag) {
                            .data => {
                                name = try readBytes(&symtab);
                                if (!symbol_flags.undefined) {
                                    _ = try symtab.takeLeb128(u32);
                                    _ = try symtab.takeLeb128(u64);
                                    _ = try symtab.takeLeb128(u64);
                                }
                            },
                            .section => {
                                _ = try symtab.takeLeb128(u32);
                                continue;
                            },
                            .function, .global, .event, .table => {
                                _ = try symtab.takeLeb128(u32);
                                if (!symbol_flags.undefined or symbol_flags.explicit_name) {
                                    name = try readBytes(&symtab);
                                }
                            },
                        }

                        if (!symbol_flags.undefined and symbol_flags.binding != .local) {
                            try b.symbol_names.appendSlice(gpa, name);
                            try b.symbol_names.append(gpa, 0);
                            try b.symbol_offsets.append(gpa, member_offset);
                        }
                    }
                }
                return;
            }
        }
        return error.MissingLinkingSection;
    }
};

fn writeHeader(w: *std.Io.Writer, name: []const u8, size: usize) std.Io.Writer.Error!void {
    var header: Header = .{
        .name = @splat(' '),
        .date = "0           ".*,
        .uid = "0     ".*,
        .gid = "0     ".*,
        .mode = "644     ".*,
        .size = @splat(' '),
        .fmag = ARFMAG.*,
    };
    @memcpy(header.name[0..name.len], name);
    _ = std.mem.print(&header.size, "{d}", .{size}) catch unreachable;
    try w.writeAll(mem.asBytes(&header));
}

fn readBytes(reader: *std.Io.Reader) ![]const u8 {
    const len = try reader.takeLeb128(u32);
    return reader.take(len);
}

pub fn deinit(archive: *Archive, gpa: Allocator) void {
    deinitToc(gpa, &archive.toc);
    archive.* = undefined;
}

fn deinitToc(gpa: Allocator, toc: *Toc) void {
    for (toc.values()) |*value| value.deinit(gpa);
    toc.deinit(gpa);
}

pub fn parse(gpa: Allocator, file_contents: []const u8) !Archive {
    var pos: usize = 0;

    if (!mem.eql(u8, file_contents[0..ARMAG.len], ARMAG)) return error.BadArchiveMagic;
    pos += ARMAG.len;

    const header = mem.bytesAsValue(Header, file_contents[pos..][0..@sizeOf(Header)]);
    if (!mem.eql(u8, &header.fmag, ARFMAG)) return error.BadHeaderDelimiter;
    pos += @sizeOf(Header);

    // The size field can have extra spaces padded in front as well as
    // the end, so we trim those first before parsing the ASCII value.
    const size_trimmed = mem.trim(u8, &header.size, " ");
    const sym_tab_size = try std.fmt.parseInt(u32, size_trimmed, 10);

    const num_symbols = mem.readInt(u32, file_contents[pos..][0..4], .big);
    pos += 4;

    const symbol_positions_size = @sizeOf(u32) * num_symbols;
    const symbol_positions_be = mem.bytesAsSlice(u32, file_contents[pos..][0..symbol_positions_size]);
    pos += symbol_positions_size;

    const sym_tab = file_contents[pos..][0 .. sym_tab_size - 4 - symbol_positions_size];
    pos += sym_tab.len;
    pos = mem.alignForward(usize, pos, 2);

    var toc: Toc = .empty;
    errdefer deinitToc(gpa, &toc);

    var sym_tab_pos: usize = 0;
    for (0..num_symbols) |i| {
        const name = mem.sliceTo(sym_tab[sym_tab_pos..], 0);
        sym_tab_pos += name.len + 1;
        if (name.len == 0) continue;

        const gop = try toc.getOrPut(gpa, name);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(gpa, switch (native_endian) {
            .big => symbol_positions_be[i],
            .little => @byteSwap(symbol_positions_be[i]),
        });
    }

    const long_file_names: RelativeSlice = s: {
        const sub_header = mem.bytesAsValue(Header, file_contents[pos..][0..@sizeOf(Header)]);
        pos += @sizeOf(Header);

        if (!mem.eql(u8, &header.fmag, ARFMAG)) return error.BadHeaderDelimiter;
        if (!mem.eql(u8, sub_header.name[0..2], "//")) return error.MissingTableName;
        const table_size = try sub_header.parsedSize();

        break :s .{
            .off = @intCast(pos),
            .len = table_size,
        };
    };

    return .{
        .toc = toc,
        .long_file_names = long_file_names,
    };
}

/// From a given file offset, starts reading for a file header.
/// When found, parses the object file into an `Object` and returns it.
pub fn parseObject(
    archive: Archive,
    wasm: *Wasm,
    file_contents: []const u8,
    object_offset: u32,
    path: Path,
    host_name: Wasm.OptionalString,
    scratch_space: *Object.ScratchSpace,
    must_link: bool,
    gc_sections: bool,
) !Object {
    const header = mem.bytesAsValue(Header, file_contents[object_offset..][0..@sizeOf(Header)]);
    if (!mem.eql(u8, &header.fmag, ARFMAG)) return error.BadHeaderDelimiter;

    const name_or_index = try header.nameOrIndex();
    const object_name = switch (name_or_index) {
        .name => |name| name,
        .index => |index| n: {
            const long_file_names = file_contents[archive.long_file_names.off..][0..archive.long_file_names.len];
            const name = mem.sliceTo(long_file_names[index..], 0x0a);
            break :n mem.trimEnd(u8, name, "/");
        },
    };

    const object_file_size = try header.parsedSize();
    const contents = file_contents[object_offset + @sizeOf(Header) ..][0..object_file_size];

    return Object.parse(wasm, contents, path, object_name, host_name, scratch_space, must_link, gc_sections);
}

const Archive = @This();

const builtin = @import("builtin");
const native_endian = builtin.cpu.arch.endian();

const std = @import("std");
const mem = std.mem;
const Allocator = std.mem.Allocator;
const Path = std.Build.Cache.Path;

const Wasm = @import("../Wasm.zig");
const Object = @import("Object.zig");
