//! Implements parsing, decoding, and caching of DWARF information.
//!
//! This API makes no assumptions about the relationship between the host and
//! the target being debugged. In other words, any DWARF information can be used
//! from any host via this API. Note, however, that the limits of 32-bit
//! addressing can cause very large 64-bit binaries to be impossible to open on
//! 32-bit hosts.
//!
//! For unopinionated types and bits, see `std.dwarf`.

const std = @import("../std.zig");
const Allocator = std.mem.Allocator;
const DW = std.dwarf;
const assert = std.debug.assert;
const Endian = std.builtin.Endian;
const Io = std.Io;
const Reader = Io.Reader;
const Error = std.debug.SelfInfoError;

const Dwarf = @This();

pub const expression = @import("Dwarf/expression.zig");
pub const Unwind = @import("Dwarf/Unwind.zig");
pub const SelfUnwinder = @import("Dwarf/SelfUnwinder.zig");

/// Useful to temporarily enable while working on this file.
const debug_debug_mode = false;

endian: Endian,
sections: Section.Array,
abbrev_tables: std.array_hash_map.Auto(u64, Abbrev.Table),
compile_units: std.MultiArrayList(struct { end_offset: u64, compile_unit: CompileUnit }),
compile_units_by_pc: std.MultiArrayList(struct {
    low_pc: u64,
    high_pc: u64,
    compile_unit: CompileUnit.Index,
}),
funcs: std.MultiArrayList(Func),

pub const Section = struct {
    slice: ?[]const u8,

    pub const Id = enum {
        debug_info,
        debug_abbrev,
        debug_str,
        debug_str_offsets,
        debug_line,
        debug_line_str,
        debug_ranges,
        debug_loclists,
        debug_rnglists,
        debug_addr,
        debug_names,
    };
    pub const Array = std.enums.EnumArray(Section.Id, Section);
};

const Abbrev = struct {
    code: u64,
    tag_id: u64,
    has_children: bool,
    attrs: []const Attr,

    const Attr = struct {
        id: u64,
        form_id: u64,
        /// Only valid if form_id is .implicit_const
        payload: i64,
    };

    const Table = struct {
        abbrevs: []const Abbrev,
        max_attrs: usize,

        const Index = enum(usize) {
            _,

            fn ptr(abbrev_table: Abbrev.Table.Index, dwarf: *const Dwarf) *const Abbrev.Table {
                return &dwarf.abbrev_tables.values()[@backingInt(abbrev_table)];
            }
        };

        fn deinit(table: *Table, gpa: Allocator) void {
            for (table.abbrevs) |abbrev| gpa.free(abbrev.attrs);
            gpa.free(table.abbrevs);
            table.* = undefined;
        }

        fn parse(dwarf: *Dwarf, gpa: Allocator, offset: u64) !Abbrev.Table {
            var fr: Reader =
                .fixed(dwarf.sections.get(.debug_abbrev).slice orelse return missing());
            fr.seek = std.math.cast(usize, offset) orelse return bad();

            var abbrevs: std.ArrayList(Abbrev) = .empty;
            defer {
                for (abbrevs.items) |abbrev| gpa.free(abbrev.attrs);
                abbrevs.deinit(gpa);
            }

            var attrs: std.ArrayList(Abbrev.Attr) = .empty;
            defer attrs.deinit(gpa);
            var max_attrs: usize = 0;

            while (true) {
                const code = try fr.takeLeb128(u64);
                if (code == 0) break;
                const tag_id = try fr.takeLeb128(u64);
                const has_children = (try fr.takeByte()) == DW.CHILDREN.yes;

                while (true) {
                    const attr_id = try fr.takeLeb128(u64);
                    const form_id = try fr.takeLeb128(u64);
                    if (attr_id == 0 and form_id == 0) break;
                    try attrs.append(gpa, .{
                        .id = attr_id,
                        .form_id = form_id,
                        .payload = switch (form_id) {
                            DW.FORM.implicit_const => try fr.takeLeb128(i64),
                            else => undefined,
                        },
                    });
                }
                max_attrs = @max(max_attrs, attrs.items.len);
                try abbrevs.append(gpa, .{
                    .code = code,
                    .tag_id = tag_id,
                    .has_children = has_children,
                    .attrs = try attrs.toOwnedSlice(gpa),
                });
            }

            return .{ .abbrevs = try abbrevs.toOwnedSlice(gpa), .max_attrs = max_attrs };
        }

        fn get(table: *const Table, abbrev_code: u64) ?*const Abbrev {
            return for (table.abbrevs) |*abbrev| {
                if (abbrev.code == abbrev_code) break abbrev;
            } else null;
        }
    };
};

pub const CompileUnit = struct {
    format: DW.Format,
    version: u16,
    address_size: u8,
    abbrev_table: Abbrev.Table.Index,
    header_end_offset: u8,

    low_pc: ?u64,
    high_pc: ?u64,
    ranges: ?FormValue,
    name: ?[]const u8,
    language: ?u16,
    stmt_list: ?usize,
    comp_dir: ?[]const u8,
    producer: ?[]const u8,
    str_offsets_base: ?usize,
    addr_base: ?usize,
    rnglists_base: ?usize,
    loclists_base: ?usize,

    src_loc_cache: ?SrcLocCache,

    pub const Index = enum(usize) {
        _,

        fn offset(compile_unit: CompileUnit.Index, dwarf: *const Dwarf) u64 {
            const prev_index, const overflow = @subWithOverflow(@backingInt(compile_unit), 1);
            return switch (overflow) {
                0 => dwarf.compile_units.items(.end_offset)[prev_index],
                1 => 0,
            };
        }

        fn endOffset(compile_unit: CompileUnit.Index, dwarf: *const Dwarf) u64 {
            return dwarf.compile_units.items(.end_offset)[@backingInt(compile_unit)];
        }

        fn length(compile_unit: CompileUnit.Index, dwarf: *const Dwarf) u64 {
            return compile_unit.endOffset(dwarf) - compile_unit.offset(dwarf);
        }

        pub fn ptr(compile_unit: CompileUnit.Index, dwarf: *const Dwarf) *CompileUnit {
            return &dwarf.compile_units.items(.compile_unit)[@backingInt(compile_unit)];
        }
    };

    pub const SrcLocCache = struct {
        line_table: LineTable,
        directories: []const FileEntry,
        files: []FileEntry,
        version: u16,

        pub const LineTable = std.array_hash_map.Auto(u64, LineEntry);

        pub const LineEntry = struct {
            line: u32,
            column: u32,
            /// Offset by 1 depending on whether Dwarf version is >= 5.
            file: u32,

            pub const invalid: LineEntry = .{
                .line = undefined,
                .column = undefined,
                .file = std.math.maxInt(u32),
            };

            pub fn isInvalid(le: LineEntry) bool {
                return le.file == invalid.file;
            }
        };

        pub fn findSource(slc: *const SrcLocCache, pc: u64) !LineEntry {
            const index = std.sort.upperBound(u64, slc.line_table.keys(), pc, order_u64);
            if (index == 0) return missing();
            return slc.line_table.values()[index - 1];
        }
    };

    fn parseDie(
        compile_unit: *const CompileUnit,
        dwarf: *const Dwarf,
        fr: *Reader,
        attrs_buf: []Die.Attr,
    ) OpenError!?Die {
        const abbrev_code = try fr.takeLeb128(u64);
        if (abbrev_code == 0) return null;
        const table_entry = compile_unit.abbrev_table.ptr(dwarf).get(abbrev_code) orelse return bad();

        const attrs = attrs_buf[0..table_entry.attrs.len];
        for (attrs, table_entry.attrs) |*result_attr, attr| result_attr.* = .{
            .id = attr.id,
            .value = try .parse(
                dwarf,
                compile_unit.format,
                compile_unit.version,
                compile_unit.address_size,
                fr,
                attr.form_id,
                attr.payload,
            ),
        };
        return .{
            .tag_id = table_entry.tag_id,
            .has_children = table_entry.has_children,
            .attrs = attrs,
        };
    }
};

const Func = struct {
    low_pc: u64,
    high_pc: u64,
    name: ?[]const u8,
};

pub const FormValue = union(enum) {
    addr: u64,
    addrx: u64,
    block: []const u8,
    udata: u64,
    data16: *const [16]u8,
    sdata: i64,
    exprloc: []const u8,
    flag: bool,
    sec_offset: u64,
    ref: u64,
    ref_addr: u64,
    string: [:0]const u8,
    strp: u64,
    strx: u64,
    line_strp: u64,
    loclistx: u64,
    rnglistx: u64,

    fn parse(
        dwarf: *const Dwarf,
        format: DW.Format,
        version: u16,
        address_size: u8,
        r: *Reader,
        form_id: u64,
        implicit_const: ?i64,
    ) OpenError!FormValue {
        return switch (form_id) {
            // DWARF5.pdf page 213: the size of this value is encoded in the
            // compilation unit header as address size.
            DW.FORM.addr => .{ .addr = try readAddress(r, dwarf.endian, address_size) },
            DW.FORM.addrx1 => .{ .addrx = try r.takeByte() },
            DW.FORM.addrx2 => .{ .addrx = try r.takeInt(u16, dwarf.endian) },
            DW.FORM.addrx3 => .{ .addrx = try r.takeInt(u24, dwarf.endian) },
            DW.FORM.addrx4 => .{ .addrx = try r.takeInt(u32, dwarf.endian) },
            DW.FORM.addrx => .{ .addrx = try r.takeLeb128(u64) },

            DW.FORM.block1 => .{ .block = try r.take(try r.takeByte()) },
            DW.FORM.block2 => .{ .block = try r.take(try r.takeInt(u16, dwarf.endian)) },
            DW.FORM.block4 => .{ .block = try r.take(try r.takeInt(u32, dwarf.endian)) },
            DW.FORM.block => .{ .block = try r.take(try r.takeLeb128(usize)) },

            DW.FORM.data1 => .{ .udata = try r.takeByte() },
            DW.FORM.data2 => .{ .udata = try r.takeInt(u16, dwarf.endian) },
            DW.FORM.data4 => .{ .udata = try r.takeInt(u32, dwarf.endian) },
            DW.FORM.data8 => .{ .udata = try r.takeInt(u64, dwarf.endian) },
            DW.FORM.data16 => .{ .data16 = try r.takeArray(16) },
            DW.FORM.udata => .{ .udata = try r.takeLeb128(u64) },
            DW.FORM.sdata => .{ .sdata = try r.takeLeb128(i64) },
            DW.FORM.exprloc => .{ .exprloc = try r.take(try r.takeLeb128(usize)) },
            DW.FORM.flag => .{ .flag = (try r.takeByte()) != 0 },
            DW.FORM.flag_present => .{ .flag = true },
            DW.FORM.sec_offset => .{ .sec_offset = try readFormatSizedInt(r, format, dwarf.endian) },

            DW.FORM.ref1 => .{ .ref = try r.takeByte() },
            DW.FORM.ref2 => .{ .ref = try r.takeInt(u16, dwarf.endian) },
            DW.FORM.ref4 => .{ .ref = try r.takeInt(u32, dwarf.endian) },
            DW.FORM.ref8 => .{ .ref = try r.takeInt(u64, dwarf.endian) },
            DW.FORM.ref_udata => .{ .ref = try r.takeLeb128(u64) },

            DW.FORM.ref_addr => .{
                .ref_addr = switch (version) {
                    2 => try readAddress(r, dwarf.endian, address_size),
                    else => try readFormatSizedInt(r, format, dwarf.endian),
                },
            },
            DW.FORM.ref_sig8 => .{ .ref = try r.takeInt(u64, dwarf.endian) },

            DW.FORM.string => .{ .string = try r.takeSentinel(0) },
            DW.FORM.strp => .{ .strp = try readFormatSizedInt(r, format, dwarf.endian) },
            DW.FORM.strx1 => .{ .strx = try r.takeByte() },
            DW.FORM.strx2 => .{ .strx = try r.takeInt(u16, dwarf.endian) },
            DW.FORM.strx3 => .{ .strx = try r.takeInt(u24, dwarf.endian) },
            DW.FORM.strx4 => .{ .strx = try r.takeInt(u32, dwarf.endian) },
            DW.FORM.strx => .{ .strx = try r.takeLeb128(usize) },
            DW.FORM.line_strp => .{ .line_strp = try readFormatSizedInt(r, format, dwarf.endian) },
            DW.FORM.indirect => parse(dwarf, format, version, address_size, r, try r.takeLeb128(u64), null),
            DW.FORM.implicit_const => .{ .sdata = implicit_const orelse return bad() },
            DW.FORM.loclistx => .{ .loclistx = try r.takeLeb128(u64) },
            DW.FORM.rnglistx => .{ .rnglistx = try r.takeLeb128(u64) },
            else => {
                //debug.print("unrecognized form id: {x}\n", .{form_id});
                return bad();
            },
        };
    }

    fn getAddr(
        form_value: *const FormValue,
        dwarf: *const Dwarf,
        compile_unit: *const CompileUnit,
    ) !u64 {
        return switch (form_value.*) {
            else => bad(),
            .addr => |value| value,
            .addrx => |index| dwarf.readDebugAddr(compile_unit, index),
        };
    }

    fn getHighPc(
        form_value: *const FormValue,
        dwarf: *const Dwarf,
        compile_unit: *const CompileUnit,
        low_pc: u64,
    ) !u64 {
        return switch (form_value.*) {
            else => try form_value.getAddr(dwarf, compile_unit),
            .udata => |offset| std.math.add(u64, low_pc, offset) catch |err| switch (err) {
                error.Overflow => return bad(),
            },
        };
    }

    fn getUInt(form_value: *const FormValue, comptime U: type) !U {
        return switch (form_value.*) {
            else => bad(),
            inline .udata, .sdata => |c| std.math.cast(U, c) orelse bad(),
        };
    }

    fn getSecOffset(form_value: *const FormValue) !usize {
        return switch (form_value.*) {
            else => bad(),
            .sec_offset => |sec_offset| std.math.cast(usize, sec_offset) orelse bad(),
        };
    }

    fn getRef(
        form_value: *const FormValue,
        dwarf: *const Dwarf,
        compile_unit: CompileUnit.Index,
    ) !struct { CompileUnit.Index, u64 } {
        switch (form_value.*) {
            else => return bad(),
            .ref => |offset| {
                if (offset >= compile_unit.length(dwarf)) return bad();
                return .{ compile_unit, compile_unit.offset(dwarf) + offset };
            },
            .ref_addr => |addr| {
                const ref_compile_unit: CompileUnit.Index = @fromBackingInt(std.sort.upperBound(
                    u64,
                    dwarf.compile_units.items(.end_offset),
                    addr,
                    order_u64,
                ));
                if (dwarf.compile_units.len - @backingInt(ref_compile_unit) == 0) return bad();
                if (addr >= ref_compile_unit.endOffset(dwarf)) return bad();
                if (ref_compile_unit.ptr(dwarf).version == 0) return bad();
                return .{ ref_compile_unit, addr };
            },
        }
    }

    fn getString(
        form_value: *const FormValue,
        dwarf: *const Dwarf,
        maybe_compile_unit: ?*const CompileUnit,
    ) ![]const u8 {
        return switch (form_value.*) {
            else => bad(),
            .string => |value| value,
            .strp => |offset| try dwarf.getString(offset),
            .line_strp => |offset| try dwarf.getLineString(offset),
            .strx => |index| {
                const debug_str_offsets =
                    dwarf.sections.get(.debug_str_offsets).slice orelse return missing();
                const compile_unit = maybe_compile_unit orelse return bad();
                const str_offsets_base = compile_unit.str_offsets_base orelse return bad();
                return try dwarf.getString(switch (compile_unit.format) {
                    .@"32" => offset: {
                        const byte_offset = str_offsets_base + 4 * index;
                        if (byte_offset + 4 > debug_str_offsets.len) return bad();
                        break :offset std.mem.readInt(
                            u32,
                            debug_str_offsets[@intCast(byte_offset)..][0..4],
                            dwarf.endian,
                        );
                    },
                    .@"64" => offset: {
                        const byte_offset = str_offsets_base + 8 * index;
                        if (byte_offset + 8 > debug_str_offsets.len) return bad();
                        break :offset std.mem.readInt(
                            u64,
                            debug_str_offsets[@intCast(byte_offset)..][0..8],
                            dwarf.endian,
                        );
                    },
                });
            },
        };
    }
};

pub const Die = struct {
    tag_id: u64,
    has_children: bool,
    attrs: []Attr,

    const Attr = struct {
        id: u64,
        value: FormValue,
    };

    fn getAttr(self: *const Die, id: u64) ?*const FormValue {
        for (self.attrs) |*attr| {
            if (attr.id == id) return &attr.value;
        }
        return null;
    }
};

pub const empty: Dwarf = .{
    .endian = .native,
    .sections = .initFill(.{ .slice = null }),
    .abbrev_tables = .empty,
    .compile_units = .empty,
    .compile_units_by_pc = .empty,
    .funcs = .empty,
};

pub const OpenError = error{
    InvalidDebugInfo,
    MissingDebugInfo,
    ReadFailed,
    EndOfStream,
    Overflow,
    StreamTooLong,
} || Allocator.Error;

/// Initialize DWARF info. The caller has the responsibility to initialize most
/// the `Dwarf` fields before calling. `binary_mem` is the raw bytes of the
/// main binary file (not the secondary debug info file).
pub fn open(dwarf: *Dwarf, gpa: Allocator, opts: struct {
    populate_ranges: bool = true,
}) OpenError!void {
    var fr: Reader = .fixed(dwarf.sections.get(.debug_info).slice orelse return missing());

    var attrs_buf: std.ArrayList(Die.Attr) = .empty;
    defer attrs_buf.deinit(gpa);

    var next_unit_offset: u64 = 0;
    while (fr.buffer.len - next_unit_offset > 0) {
        const unit_offset = next_unit_offset;
        fr.seek = @intCast(unit_offset);
        const unit_header = try readUnitHeader(&fr, dwarf.endian);
        const unit_length = unit_header.header_length + unit_header.unit_length;
        next_unit_offset += unit_length;
        if (next_unit_offset > fr.buffer.len) return bad();

        const compile_unit_index: CompileUnit.Index =
            @fromBackingInt(try dwarf.compile_units.addOne(gpa));
        dwarf.compile_units.items(.end_offset)[@backingInt(compile_unit_index)] = next_unit_offset;
        const compile_unit = compile_unit_index.ptr(dwarf);
        compile_unit.* = .{
            .format = unit_header.format,
            .version = 0,
            .address_size = undefined,
            .abbrev_table = undefined,
            .header_end_offset = undefined,

            .low_pc = null,
            .high_pc = null,
            .ranges = null,
            .name = null,
            .language = null,
            .stmt_list = null,
            .comp_dir = null,
            .producer = null,
            .str_offsets_base = null,
            .addr_base = null,
            .rnglists_base = null,
            .loclists_base = null,

            .src_loc_cache = null,
        };

        compile_unit.version = try fr.takeInt(u16, dwarf.endian);
        const debug_abbrev_offset = if (compile_unit.version == 5) debug_abbrev_offset: {
            const unit_type = try fr.takeByte();
            if (unit_type != DW.UT.compile) return bad();
            compile_unit.address_size = try fr.takeByte();
            break :debug_abbrev_offset try readFormatSizedInt(&fr, unit_header.format, dwarf.endian);
        } else if (compile_unit.version >= 2 and compile_unit.version < 5) debug_abbrev_offset: {
            const debug_abbrev_offset = try readFormatSizedInt(&fr, unit_header.format, dwarf.endian);
            compile_unit.address_size = try fr.takeByte();
            break :debug_abbrev_offset debug_abbrev_offset;
        } else {
            compile_unit.version = 0;
            continue;
        };
        compile_unit.header_end_offset = @intCast(fr.seek - unit_offset);

        const abbrev_table_gop = try dwarf.abbrev_tables.getOrPut(gpa, debug_abbrev_offset);
        const abbrev_table = abbrev_table_gop.value_ptr;
        if (!abbrev_table_gop.found_existing) {
            errdefer _ = dwarf.abbrev_tables.pop().?;
            abbrev_table.* = try .parse(dwarf, gpa, debug_abbrev_offset);
        }
        compile_unit.abbrev_table = @fromBackingInt(abbrev_table_gop.index);

        attrs_buf.clearRetainingCapacity();
        try attrs_buf.ensureTotalCapacity(gpa, 2 * abbrev_table.max_attrs);
        attrs_buf.resizeAssumeCapacity(abbrev_table.max_attrs);
        const die = try compile_unit.parseDie(dwarf, &fr, attrs_buf.items) orelse {
            compile_unit.version = 0;
            continue;
        };
        if (die.tag_id != DW.TAG.compile_unit) {
            compile_unit.version = 0;
            continue;
        }
        for (die.attrs) |*attr| switch (attr.id) {
            else => {},
            DW.AT.low_pc => compile_unit.low_pc = try attr.value.getAddr(dwarf, compile_unit),
            DW.AT.stmt_list => compile_unit.stmt_list = try attr.value.getSecOffset(),
            DW.AT.str_offsets_base => compile_unit.str_offsets_base = try attr.value.getSecOffset(),
            DW.AT.addr_base => compile_unit.addr_base = try attr.value.getSecOffset(),
            DW.AT.rnglists_base => compile_unit.rnglists_base = try attr.value.getSecOffset(),
            DW.AT.loclists_base => compile_unit.loclists_base = try attr.value.getSecOffset(),
        };
        for (die.attrs) |*attr| switch (attr.id) {
            else => {},
            DW.AT.high_pc => if (compile_unit.low_pc) |low_pc| {
                compile_unit.high_pc = try attr.value.getHighPc(dwarf, compile_unit, low_pc);
            },
            DW.AT.ranges => compile_unit.ranges = attr.value,
            DW.AT.name => compile_unit.name = try attr.value.getString(dwarf, compile_unit),
            DW.AT.language => compile_unit.language = try attr.value.getUInt(u16),
            DW.AT.comp_dir => compile_unit.comp_dir = try attr.value.getString(dwarf, compile_unit),
            DW.AT.producer => compile_unit.producer = try attr.value.getString(dwarf, compile_unit),
        };
    }

    fr.seek = 0;
    for (
        dwarf.compile_units.items(.compile_unit),
        0..,
        dwarf.compile_units.items(.end_offset),
    ) |*compile_unit, compile_unit_index, end_offset| {
        if (compile_unit.version == 0) {
            fr.seek = @intCast(end_offset);
            continue;
        }

        const max_attrs = compile_unit.abbrev_table.ptr(dwarf).max_attrs;
        fr.seek += compile_unit.header_end_offset;
        while (fr.seek < end_offset) {
            attrs_buf.clearRetainingCapacity();
            attrs_buf.resizeAssumeCapacity(max_attrs);
            var die = try compile_unit.parseDie(dwarf, &fr, attrs_buf.items) orelse continue;

            switch (die.tag_id) {
                DW.TAG.subprogram, DW.TAG.inlined_subroutine, DW.TAG.subroutine, DW.TAG.entry_point => {
                    const name = name: {
                        var cur_compile_unit: CompileUnit.Index = @fromBackingInt(compile_unit_index);
                        var cur_die = die;
                        // Prevent endless loops
                        for (0..3) |_| {
                            if (cur_die.getAttr(DW.AT.name)) |name| {
                                break :name try name.getString(dwarf, compile_unit);
                            } else if (cur_die.getAttr(DW.AT.abstract_origin)) |abstract_origin| {
                                const after_die_offset = fr.seek;
                                defer fr.seek = after_die_offset;

                                // Follow the DIE it points to and repeat
                                cur_compile_unit, const ref_offset =
                                    try abstract_origin.getRef(dwarf, cur_compile_unit);
                                fr.seek = @intCast(ref_offset);
                                cur_die = try cur_compile_unit.ptr(dwarf).parseDie(
                                    dwarf,
                                    &fr,
                                    attrs_buf.unusedCapacitySlice(),
                                ) orelse return bad();
                            } else if (cur_die.getAttr(DW.AT.specification)) |specification| {
                                const after_die_offset = fr.seek;
                                defer fr.seek = after_die_offset;

                                // Follow the DIE it points to and repeat
                                cur_compile_unit, const ref_offset =
                                    try specification.getRef(dwarf, cur_compile_unit);
                                fr.seek = @intCast(ref_offset);
                                cur_die = try cur_compile_unit.ptr(dwarf).parseDie(
                                    dwarf,
                                    &fr,
                                    attrs_buf.unusedCapacitySlice(),
                                ) orelse return bad();
                            } else break :name null;
                        }
                        break :name null;
                    };

                    range: {
                        const low_pc_attr = die.getAttr(DW.AT.low_pc) orelse break :range;
                        const high_pc_attr = die.getAttr(DW.AT.high_pc) orelse break :range;
                        const low_pc = try low_pc_attr.getAddr(dwarf, compile_unit);
                        const high_pc = try high_pc_attr.getHighPc(dwarf, compile_unit, low_pc);
                        try dwarf.funcs.append(gpa, .{
                            .low_pc = low_pc,
                            .high_pc = high_pc,
                            .name = name,
                        });
                    }

                    if (die.getAttr(DW.AT.ranges)) |ranges| ranges: {
                        var iter = DebugRangeIterator.init(
                            ranges,
                            dwarf,
                            compile_unit,
                        ) catch |err| switch (err) {
                            else => |e| return e,
                            error.MissingDebugInfo => break :ranges,
                        };
                        while (try iter.next()) |pc_range| try dwarf.funcs.append(gpa, .{
                            .low_pc = pc_range.low_pc,
                            .high_pc = pc_range.high_pc,
                            .name = name,
                        });
                    }
                },
                else => {},
            }
        }
        fr.seek = @intCast(end_offset);
    }

    if (opts.populate_ranges) try dwarf.populateRanges(gpa);
}

pub fn deinit(dwarf: *Dwarf, gpa: Allocator) void {
    for (dwarf.abbrev_tables.values()) |*abbrev| abbrev.deinit(gpa);
    dwarf.abbrev_tables.deinit(gpa);
    for (dwarf.compile_units.items(.compile_unit)) |*compile_unit| {
        if (compile_unit.src_loc_cache) |*slc| {
            slc.line_table.deinit(gpa);
            gpa.free(slc.directories);
            gpa.free(slc.files);
        }
    }
    dwarf.compile_units.deinit(gpa);
    dwarf.compile_units_by_pc.deinit(gpa);
    dwarf.funcs.deinit(gpa);
    dwarf.* = undefined;
}

pub fn getSymbolName(dwarf: *const Dwarf, pc: u64) ?[]const u8 {
    // Iterate the function list backwards so that we see child DIEs before their parents. This is
    // important because `DW_TAG_inlined_subroutine` DIEs will have a range which is a sub-range of
    // their caller, and we want to return the callee's name, not the caller's.
    var func_index: usize = dwarf.funcs.len;
    while (func_index > 0) {
        func_index -= 1;
        const func = dwarf.funcs.get(func_index);
        if (func.low_pc <= pc and pc < func.high_pc) return func.name;
    }
    return null;
}

pub fn populateRanges(dwarf: *Dwarf, gpa: Allocator) OpenError!void {
    assert(dwarf.compile_units_by_pc.len == 0);

    for (dwarf.compile_units.items(.compile_unit), 0..) |*compile_unit, compile_unit_index| {
        if (compile_unit.version == 0) continue;
        range: {
            const low_pc = compile_unit.low_pc orelse break :range;
            const high_pc = compile_unit.high_pc orelse break :range;
            if (low_pc >= high_pc) continue;
            try dwarf.compile_units_by_pc.append(gpa, .{
                .low_pc = low_pc,
                .high_pc = high_pc,
                .compile_unit = @fromBackingInt(compile_unit_index),
            });
        }
        if (compile_unit.ranges) |*ranges| ranges: {
            var iter = DebugRangeIterator.init(ranges, dwarf, compile_unit) catch |err| switch (err) {
                else => |e| return e,
                error.MissingDebugInfo => break :ranges,
            };
            while (try iter.next()) |pc_range| {
                if (pc_range.low_pc >= pc_range.high_pc) continue;
                try dwarf.compile_units_by_pc.append(gpa, .{
                    .low_pc = pc_range.low_pc,
                    .high_pc = pc_range.high_pc,
                    .compile_unit = @fromBackingInt(compile_unit_index),
                });
            }
        }
    }

    dwarf.compile_units_by_pc.sortUnstable(struct {
        high_pcs: []const u64,
        pub fn lessThan(ctx: @This(), lhs_index: usize, rhs_index: usize) bool {
            return ctx.high_pcs[lhs_index] < ctx.high_pcs[rhs_index];
        }
    }{ .high_pcs = dwarf.compile_units_by_pc.items(.high_pc) });
}

const DebugRangeIterator = struct {
    base_address: u64,
    section_id: Section.Id,
    dwarf: *const Dwarf,
    compile_unit: *const CompileUnit,
    fr: Reader,

    fn init(
        ranges_value: *const FormValue,
        dwarf: *const Dwarf,
        compile_unit: *const CompileUnit,
    ) !DebugRangeIterator {
        const section_id: Section.Id =
            if (compile_unit.version >= 5) .debug_rnglists else .debug_ranges;
        const debug_ranges = dwarf.sections.get(section_id).slice orelse return missing();

        const ranges_offset = switch (ranges_value.*) {
            .sec_offset, .udata => |off| off,
            .rnglistx => |idx| off: {
                const rnglists_base = compile_unit.rnglists_base orelse return bad();
                switch (compile_unit.format) {
                    .@"32" => {
                        const offset_loc = rnglists_base + 4 * idx;
                        if (offset_loc + 4 > debug_ranges.len) return bad();
                        const offset = std.mem.readInt(
                            u32,
                            debug_ranges[@intCast(offset_loc)..][0..4],
                            dwarf.endian,
                        );
                        break :off rnglists_base + offset;
                    },
                    .@"64" => {
                        const offset_loc = rnglists_base + 8 * idx;
                        if (offset_loc + 8 > debug_ranges.len) return bad();
                        const offset = std.mem.readInt(
                            u64,
                            debug_ranges[@intCast(offset_loc)..][0..8],
                            dwarf.endian,
                        );
                        break :off rnglists_base + offset;
                    },
                }
            },
            else => return bad(),
        };

        // All the addresses in the list are relative to the value
        // specified by DW_AT.low_pc or to some other value encoded
        // in the list itself.
        // If no starting value is specified use zero.
        const base_address = compile_unit.low_pc orelse 0;

        var fr: Reader = .fixed(debug_ranges);
        fr.seek = std.math.cast(usize, ranges_offset) orelse return bad();

        return .{
            .base_address = base_address,
            .section_id = section_id,
            .dwarf = dwarf,
            .compile_unit = compile_unit,
            .fr = fr,
        };
    }

    // Returns the next range in the list, or null if the end was reached.
    fn next(it: *DebugRangeIterator) !?struct { low_pc: u64, high_pc: u64 } {
        switch (it.section_id) {
            .debug_rnglists => {
                const kind = try it.fr.takeByte();
                switch (kind) {
                    DW.RLE.end_of_list => return null,
                    DW.RLE.base_addressx => {
                        const index = try it.fr.takeLeb128(u64);
                        it.base_address = try it.dwarf.readDebugAddr(it.compile_unit, index);
                        return try it.next();
                    },
                    DW.RLE.startx_endx => {
                        const start_index = try it.fr.takeLeb128(u64);
                        const start_addr = try it.dwarf.readDebugAddr(it.compile_unit, start_index);

                        const end_index = try it.fr.takeLeb128(u64);
                        const end_addr = try it.dwarf.readDebugAddr(it.compile_unit, end_index);

                        return .{ .low_pc = start_addr, .high_pc = end_addr };
                    },
                    DW.RLE.startx_length => {
                        const start_index = try it.fr.takeLeb128(u64);
                        const start_addr = try it.dwarf.readDebugAddr(it.compile_unit, start_index);

                        const len = try it.fr.takeLeb128(u64);
                        const end_addr = start_addr + len;

                        return .{ .low_pc = start_addr, .high_pc = end_addr };
                    },
                    DW.RLE.offset_pair => {
                        const start_addr = try it.fr.takeLeb128(u64);
                        const end_addr = try it.fr.takeLeb128(u64);

                        return .{
                            .low_pc = it.base_address + start_addr,
                            .high_pc = it.base_address + end_addr,
                        };
                    },
                    DW.RLE.base_address => {
                        it.base_address = try readAddress(&it.fr, it.dwarf.endian, it.compile_unit.address_size);
                        return try it.next();
                    },
                    DW.RLE.start_end => {
                        const start_addr = try readAddress(&it.fr, it.dwarf.endian, it.compile_unit.address_size);
                        const end_addr = try readAddress(&it.fr, it.dwarf.endian, it.compile_unit.address_size);

                        return .{ .low_pc = start_addr, .high_pc = end_addr };
                    },
                    DW.RLE.start_length => {
                        const start_addr = try readAddress(&it.fr, it.dwarf.endian, it.compile_unit.address_size);
                        const len = try it.fr.takeLeb128(u64);
                        const end_addr = start_addr + len;

                        return .{ .low_pc = start_addr, .high_pc = end_addr };
                    },
                    else => return bad(),
                }
            },
            .debug_ranges => {
                const start_addr = try readAddress(&it.fr, it.dwarf.endian, it.compile_unit.address_size);
                const end_addr = try readAddress(&it.fr, it.dwarf.endian, it.compile_unit.address_size);
                if (start_addr == 0 and end_addr == 0) return null;

                // The entry with start_addr = max_representable_address selects a new value for the base address
                const max_representable_address = ~@as(u64, 0) >> @intCast(64 - it.compile_unit.address_size);
                if (start_addr == max_representable_address) {
                    it.base_address = end_addr;
                    return try it.next();
                }

                return .{
                    .low_pc = it.base_address + start_addr,
                    .high_pc = it.base_address + end_addr,
                };
            },
            else => unreachable,
        }
    }
};

pub fn findCompileUnit(dwarf: *const Dwarf, pc: u64) ?CompileUnit.Index {
    assert(dwarf.compile_units_by_pc.len > 0); // call `populateRanges` first.
    const index = std.sort.upperBound(u64, dwarf.compile_units_by_pc.items(.high_pc), pc, order_u64);
    if (dwarf.compile_units_by_pc.len - index == 0) return null;
    if (pc < dwarf.compile_units_by_pc.items(.low_pc)[index]) return null;
    const compile_unit = dwarf.compile_units_by_pc.items(.compile_unit)[index];
    if (compile_unit.ptr(dwarf).version == 0) return null;
    return compile_unit;
}

/// Gets an already existing AbbrevTable given the abbrev_offset, or if not found,
/// seeks in the stream and parses it.
fn getAbbrevTable(dwarf: *Dwarf, gpa: Allocator, abbrev_offset: u64) !*const Abbrev.Table {
    for (dwarf.abbrev_tables.items) |*table| {
        if (table.offset == abbrev_offset) {
            return table;
        }
    }
    const abbrev_table = try dwarf.abbrev_tables.addOne(gpa);
    errdefer _ = dwarf.abbrev_tables.pop().?;
    abbrev_table.* = try dwarf.parseAbbrevTable(gpa, abbrev_offset);
    return abbrev_table;
}

/// Ensures that addresses in the returned LineTable are monotonically increasing.
fn runLineNumberProgram(
    dwarf: *const Dwarf,
    gpa: Allocator,
    compile_unit_index: CompileUnit.Index,
) !CompileUnit.SrcLocCache {
    const compile_unit = compile_unit_index.ptr(dwarf);
    const line_info_offset = compile_unit.stmt_list orelse return missing();

    var fr: Reader = .fixed(dwarf.sections.get(.debug_line).slice orelse return missing());
    fr.seek = @intCast(line_info_offset);

    const unit_header = try readUnitHeader(&fr, dwarf.endian);
    if (unit_header.unit_length == 0) return missing();

    const unit_length = unit_header.header_length + unit_header.unit_length;

    const version = try fr.takeInt(u16, dwarf.endian);
    if (version < 2) return bad();

    const address_size: u8, const seg_size: u8 = if (version >= 5) .{
        try fr.takeByte(),
        try fr.takeByte(),
    } else .{ compile_unit.address_size, 0 };
    if (seg_size != 0) return bad(); // unsupported

    const prologue_length = try readFormatSizedInt(&fr, unit_header.format, dwarf.endian);
    const prog_start_offset = fr.seek + prologue_length;

    const minimum_instruction_length = try fr.takeByte();
    if (minimum_instruction_length == 0) return bad();

    if (version >= 4) {
        const maximum_operations_per_instruction = try fr.takeByte();
        _ = maximum_operations_per_instruction;
    }

    const default_is_stmt = (try fr.takeByte()) != 0;
    const line_base = try fr.takeByteSigned();

    const line_range = try fr.takeByte();
    if (line_range == 0) return bad();

    const opcode_base = try fr.takeByte();

    const standard_opcode_lengths = try fr.take(opcode_base - 1);

    var directories: std.ArrayList(FileEntry) = .empty;
    defer directories.deinit(gpa);
    var file_entries: std.ArrayList(FileEntry) = .empty;
    defer file_entries.deinit(gpa);

    if (version < 5) {
        try directories.append(gpa, .{
            .path = compile_unit.comp_dir orelse return missing(),
        });

        while (true) {
            const dir = try fr.takeSentinel(0);
            if (dir.len == 0) break;
            try directories.append(gpa, .{ .path = dir });
        }

        while (true) {
            const file_name = try fr.takeSentinel(0);
            if (file_name.len == 0) break;
            const dir_index = try fr.takeLeb128(u32);
            const mtime = try fr.takeLeb128(u64);
            const size = try fr.takeLeb128(u64);
            try file_entries.append(gpa, .{
                .path = file_name,
                .dir_index = dir_index,
                .mtime = mtime,
                .size = size,
            });
        }
    } else {
        const FileEntFmt = struct {
            content_type_code: u16,
            form_code: u16,
        };
        {
            var dir_ent_fmt_buf: [10]FileEntFmt = undefined;
            const directory_entry_format_count = try fr.takeByte();
            if (directory_entry_format_count > dir_ent_fmt_buf.len) return bad();
            for (dir_ent_fmt_buf[0..directory_entry_format_count]) |*ent_fmt| {
                ent_fmt.* = .{
                    .content_type_code = try fr.takeLeb128(u8),
                    .form_code = try fr.takeLeb128(u16),
                };
            }

            const directories_count = try fr.takeLeb128(usize);

            for (try directories.addManyAsSlice(gpa, directories_count)) |*e| {
                e.* = .{ .path = &.{} };
                for (dir_ent_fmt_buf[0..directory_entry_format_count]) |ent_fmt| {
                    const form_value: FormValue = try .parse(
                        dwarf,
                        unit_header.format,
                        version,
                        address_size,
                        &fr,
                        ent_fmt.form_code,
                        null,
                    );
                    switch (ent_fmt.content_type_code) {
                        DW.LNCT.path => e.path = try form_value.getString(dwarf, compile_unit),
                        DW.LNCT.directory_index => e.dir_index = try form_value.getUInt(u32),
                        DW.LNCT.timestamp => e.mtime = try form_value.getUInt(u64),
                        DW.LNCT.size => e.size = try form_value.getUInt(u64),
                        DW.LNCT.MD5 => e.md5 = switch (form_value) {
                            .data16 => |data16| data16.*,
                            else => return bad(),
                        },
                        else => continue,
                    }
                }
            }
        }

        var file_ent_fmt_buf: [10]FileEntFmt = undefined;
        const file_name_entry_format_count = try fr.takeByte();
        if (file_name_entry_format_count > file_ent_fmt_buf.len) return bad();
        for (file_ent_fmt_buf[0..file_name_entry_format_count]) |*ent_fmt| {
            ent_fmt.* = .{
                .content_type_code = try fr.takeLeb128(u16),
                .form_code = try fr.takeLeb128(u16),
            };
        }

        const file_names_count = try fr.takeLeb128(usize);
        try file_entries.ensureUnusedCapacity(gpa, file_names_count);

        for (try file_entries.addManyAsSlice(gpa, file_names_count)) |*e| {
            e.* = .{ .path = &.{} };
            for (file_ent_fmt_buf[0..file_name_entry_format_count]) |ent_fmt| {
                const form_value: FormValue = try .parse(
                    dwarf,
                    unit_header.format,
                    version,
                    address_size,
                    &fr,
                    ent_fmt.form_code,
                    null,
                );
                switch (ent_fmt.content_type_code) {
                    DW.LNCT.path => e.path = try form_value.getString(dwarf, null),
                    DW.LNCT.directory_index => e.dir_index = try form_value.getUInt(u32),
                    DW.LNCT.timestamp => e.mtime = try form_value.getUInt(u64),
                    DW.LNCT.size => e.size = try form_value.getUInt(u64),
                    DW.LNCT.MD5 => e.md5 = switch (form_value) {
                        .data16 => |data16| data16.*,
                        else => return bad(),
                    },
                    else => continue,
                }
            }
        }
    }

    const abbrev_table = compile_unit.abbrev_table.ptr(dwarf);
    const attrs_buf = try gpa.alloc(Die.Attr, max_attrs: {
        var max_attrs: usize = 0;
        for (abbrev_table.abbrevs) |abbrev| {
            max_attrs = @max(max_attrs, abbrev.attrs.len);
        }
        break :max_attrs max_attrs;
    });
    defer gpa.free(attrs_buf);

    var prog = LineNumberProgram.init(default_is_stmt, version);
    var line_table: CompileUnit.SrcLocCache.LineTable = .{};
    errdefer line_table.deinit(gpa);

    fr.seek = @intCast(prog_start_offset);

    const next_unit_pos = line_info_offset + unit_length;

    while (fr.seek < next_unit_pos) {
        const opcode = try fr.takeByte();

        if (opcode == DW.LNS.extended_op) {
            const op_size = try fr.takeLeb128(u64);
            if (op_size < 1) return bad();
            const extended_op = try fr.takeByte();
            switch (extended_op) {
                DW.LNE.end_sequence => {
                    // The row being added here is an "end" address, meaning
                    // that it does not map to the source location here -
                    // rather it marks the previous address as the last address
                    // that maps to this source location.

                    // In this implementation we don't mark end of addresses.
                    // This is a performance optimization based on the fact
                    // that we don't need to know if an address is missing
                    // source location info; we are only interested in being
                    // able to look up source location info for addresses that
                    // are known to have debug info.
                    //if (debug_debug_mode) assert(!line_table.contains(prog.address));
                    //try line_table.put(gpa, prog.address, CompileUnit.SrcLocCache.LineEntry.invalid);
                    prog.reset();
                },
                DW.LNE.set_address => {
                    prog.address = try readAddress(&fr, dwarf.endian, address_size);
                },
                DW.LNE.define_file => {
                    const path = try fr.takeSentinel(0);
                    const dir_index = try fr.takeLeb128(u32);
                    const mtime = try fr.takeLeb128(u64);
                    const size = try fr.takeLeb128(u64);
                    try file_entries.append(gpa, .{
                        .path = path,
                        .dir_index = dir_index,
                        .mtime = mtime,
                        .size = size,
                    });
                },
                DW.LNE.ZIG_set_decl_ref,
                DW.LNE.ZIG_set_source_decl_ref,
                DW.LNE.ZIG_set_decl,
                DW.LNE.ZIG_set_source_decl,
                => {
                    const decl_die_offset = switch (op_size) {
                        else => return bad(),
                        1 + 1 => try fr.takeByte(),
                        1 + 2 => try fr.takeInt(u16, dwarf.endian),
                        1 + 4 => try fr.takeInt(u32, dwarf.endian),
                        1 + 8 => try fr.takeInt(u64, dwarf.endian),
                    };
                    var dwarf_fr: Reader =
                        .fixed(dwarf.sections.get(.debug_info).slice orelse return missing());
                    var cur_compile_unit: CompileUnit.Index = compile_unit_index;
                    dwarf_fr.seek = @intCast(switch (extended_op) {
                        else => unreachable,
                        DW.LNE.ZIG_set_decl_ref, DW.LNE.ZIG_set_source_decl_ref => decl_die_offset,
                        DW.LNE.ZIG_set_decl,
                        DW.LNE.ZIG_set_source_decl,
                        => cur_compile_unit.offset(dwarf) + decl_die_offset,
                    });
                    var cur_die = try cur_compile_unit.ptr(dwarf)
                        .parseDie(dwarf, &dwarf_fr, attrs_buf) orelse return bad();
                    switch (extended_op) {
                        else => unreachable,
                        DW.LNE.ZIG_set_decl_ref,
                        DW.LNE.ZIG_set_decl,
                        => if (cur_die.getAttr(DW.AT.low_pc)) |low_pc| {
                            prog.address = try low_pc.getAddr(dwarf, cur_compile_unit.ptr(dwarf));
                        },
                        DW.LNE.ZIG_set_source_decl_ref, DW.LNE.ZIG_set_source_decl => {},
                    }
                    while (true) {
                        if (cur_die.getAttr(DW.AT.decl_line)) |decl_line| {
                            prog.line = try decl_line.getUInt(i64);
                            break;
                        }
                        if (cur_die.getAttr(DW.AT.abstract_origin)) |abstract_origin| {
                            cur_compile_unit, const ref_offset =
                                try abstract_origin.getRef(dwarf, cur_compile_unit);
                            dwarf_fr.seek = @intCast(ref_offset);
                        } else if (cur_die.getAttr(DW.AT.specification)) |specification| {
                            cur_compile_unit, const ref_offset =
                                try specification.getRef(dwarf, cur_compile_unit);
                            dwarf_fr.seek = @intCast(ref_offset);
                        } else {
                            // no specification, so we can't find DW_AT_decl_line
                            break;
                        }
                        cur_die = try cur_compile_unit.ptr(dwarf)
                            .parseDie(dwarf, &dwarf_fr, attrs_buf) orelse break;
                    }

                    if (cur_die.getAttr(DW.AT.decl_column)) |decl_column| {
                        prog.column = try decl_column.getUInt(u64);
                    }

                    while (true) {
                        if (cur_die.getAttr(DW.AT.decl_file)) |decl_file| {
                            prog.file = try decl_file.getUInt(usize);
                            break;
                        }
                        if (cur_die.getAttr(DW.AT.abstract_origin)) |abstract_origin| {
                            cur_compile_unit, const ref_offset =
                                try abstract_origin.getRef(dwarf, cur_compile_unit);
                            dwarf_fr.seek = @intCast(ref_offset);
                        } else if (cur_die.getAttr(DW.AT.specification)) |specification| {
                            cur_compile_unit, const ref_offset =
                                try specification.getRef(dwarf, cur_compile_unit);
                            dwarf_fr.seek = @intCast(ref_offset);
                        } else if (cur_die.getAttr(DW.AT.ZIG_parent)) |parent| {
                            cur_compile_unit, const ref_offset =
                                try parent.getRef(dwarf, cur_compile_unit);
                            dwarf_fr.seek = @intCast(ref_offset);
                        } else {
                            // no parent, so we can't find DW_AT_decl_file
                            break;
                        }
                        cur_die = try cur_compile_unit.ptr(dwarf)
                            .parseDie(dwarf, &dwarf_fr, attrs_buf) orelse break;
                    }
                },
                else => try fr.discardAll64(op_size - 1),
            }
        } else if (opcode >= opcode_base) {
            // special opcodes
            const adjusted_opcode = opcode - opcode_base;
            const inc_addr = minimum_instruction_length * (adjusted_opcode / line_range);
            const inc_line = @as(i32, line_base) + @as(i32, adjusted_opcode % line_range);
            prog.line += inc_line;
            prog.address += inc_addr;
            try prog.addRow(gpa, &line_table);
            prog.basic_block = false;
        } else {
            switch (opcode) {
                DW.LNS.copy => {
                    try prog.addRow(gpa, &line_table);
                    prog.basic_block = false;
                },
                DW.LNS.advance_pc => {
                    const arg = try fr.takeLeb128(u64);
                    prog.address += arg * minimum_instruction_length;
                },
                DW.LNS.advance_line => {
                    const arg = try fr.takeLeb128(i64);
                    prog.line += arg;
                },
                DW.LNS.set_file => {
                    const arg = try fr.takeLeb128(usize);
                    prog.file = arg;
                },
                DW.LNS.set_column => {
                    const arg = try fr.takeLeb128(u64);
                    prog.column = arg;
                },
                DW.LNS.negate_stmt => {
                    prog.is_stmt = !prog.is_stmt;
                },
                DW.LNS.set_basic_block => {
                    prog.basic_block = true;
                },
                DW.LNS.const_add_pc => {
                    const inc_addr = minimum_instruction_length * ((255 - opcode_base) / line_range);
                    prog.address += inc_addr;
                },
                DW.LNS.fixed_advance_pc => {
                    const arg = try fr.takeInt(u16, dwarf.endian);
                    prog.address += arg;
                },
                DW.LNS.set_prologue_end => {},
                else => {
                    if (opcode - 1 >= standard_opcode_lengths.len) return bad();
                    try fr.discardAll(standard_opcode_lengths[opcode - 1]);
                },
            }
        }
    }

    // Dwarf standard v5, 6.2.5 says
    // > Within a sequence, addresses and operation pointers may only increase.
    // However, this is empirically not the case in reality, so we sort here.
    line_table.sortUnstable(struct {
        keys: []const u64,

        pub fn lessThan(ctx: @This(), a_index: usize, b_index: usize) bool {
            return ctx.keys[a_index] < ctx.keys[b_index];
        }
    }{ .keys = line_table.keys() });

    try directories.shrinkToLen(gpa);
    try file_entries.shrinkToLen(gpa);

    return .{
        .line_table = line_table,
        .directories = directories.toOwnedSliceAssert(),
        .files = file_entries.toOwnedSliceAssert(),
        .version = version,
    };
}

pub fn populateSrcLocCache(
    dwarf: *const Dwarf,
    gpa: Allocator,
    compile_unit: CompileUnit.Index,
) OpenError!void {
    assert(compile_unit.ptr(dwarf).src_loc_cache == null);
    const src_loc_cache = try dwarf.runLineNumberProgram(gpa, compile_unit);
    compile_unit.ptr(dwarf).src_loc_cache = src_loc_cache;
}

pub fn getLineNumberInfo(
    dwarf: *Dwarf,
    gpa: Allocator,
    text_arena: Allocator,
    compile_unit: CompileUnit.Index,
    pc: u64,
) !std.debug.SourceLocation {
    if (compile_unit.ptr(dwarf).src_loc_cache == null) try dwarf.populateSrcLocCache(gpa, compile_unit);
    const slc = &compile_unit.ptr(dwarf).src_loc_cache.?;
    const entry = try slc.findSource(pc);
    const file_index = entry.file - @intFromBool(slc.version < 5);
    if (file_index >= slc.files.len) return bad();
    const file_entry = &slc.files[file_index];
    if (file_entry.dir_index >= slc.directories.len) return bad();
    const dir_name = slc.directories[file_entry.dir_index].path;
    const file_name = try std.fs.path.join(text_arena, &.{ dir_name, file_entry.path });
    return .{ .line = entry.line, .column = entry.column, .file_name = file_name };
}

fn getString(dwarf: *const Dwarf, offset: u64) ![:0]const u8 {
    return getStringGeneric(dwarf.sections.get(.debug_str).slice orelse return missing(), offset);
}

fn getLineString(dwarf: *const Dwarf, offset: u64) ![:0]const u8 {
    return getStringGeneric(dwarf.sections.get(.debug_line_str).slice orelse return missing(), offset);
}

fn readDebugAddr(dwarf: *const Dwarf, compile_unit: *const CompileUnit, index: u64) !u64 {
    const addr_base = compile_unit.addr_base orelse return bad();
    const debug_addr = dwarf.sections.get(.debug_addr).slice orelse return missing();

    // addr_base points to the first item after the header, however we
    // need to read the header to know the size of each item. Empirically,
    // it may disagree with is_64 on the compile unit.
    // The header is 8 or 12 bytes depending on is_64.
    if (addr_base < 8) return bad();

    const version = std.mem.readInt(u16, debug_addr[@intCast(addr_base - 4)..][0..2], dwarf.endian);
    if (version != 5) return bad();

    const addr_size = debug_addr[addr_base - 2];
    const seg_size = debug_addr[addr_base - 1];

    const byte_offset = addr_base + (addr_size + seg_size) * index;
    if (byte_offset + addr_size > debug_addr.len) return bad();
    return switch (addr_size) {
        1 => debug_addr[@intCast(byte_offset)],
        2 => std.mem.readInt(u16, debug_addr[@intCast(byte_offset)..][0..2], dwarf.endian),
        4 => std.mem.readInt(u32, debug_addr[@intCast(byte_offset)..][0..4], dwarf.endian),
        8 => std.mem.readInt(u64, debug_addr[@intCast(byte_offset)..][0..8], dwarf.endian),
        else => bad(),
    };
}

const FileEntry = struct {
    path: []const u8,
    dir_index: u32 = 0,
    mtime: u64 = 0,
    size: u64 = 0,
    md5: [16]u8 = @splat(0),
};

const LineNumberProgram = struct {
    address: u64,
    file: usize,
    line: i64,
    column: u64,
    version: u16,
    is_stmt: bool,
    basic_block: bool,

    default_is_stmt: bool,

    // Reset the state machine following the DWARF specification
    pub fn reset(self: *LineNumberProgram) void {
        self.address = 0;
        self.file = 1;
        self.line = 1;
        self.column = 0;
        self.is_stmt = self.default_is_stmt;
        self.basic_block = false;
    }

    pub fn init(is_stmt: bool, version: u16) LineNumberProgram {
        return .{
            .address = 0,
            .file = 1,
            .line = 1,
            .column = 0,
            .version = version,
            .is_stmt = is_stmt,
            .basic_block = false,
            .default_is_stmt = is_stmt,
        };
    }

    pub fn addRow(prog: *LineNumberProgram, gpa: Allocator, table: *CompileUnit.SrcLocCache.LineTable) !void {
        if (prog.line == 0) {
            //if (debug_debug_mode) @panic("garbage line data");
            return;
        }
        if (debug_debug_mode) assert(!table.contains(prog.address));
        try table.put(gpa, prog.address, .{
            .line = std.math.cast(u32, prog.line) orelse std.math.maxInt(u32),
            .column = std.math.cast(u32, prog.column) orelse std.math.maxInt(u32),
            .file = std.math.cast(u32, prog.file) orelse return bad(),
        });
    }
};

const UnitHeader = struct {
    format: DW.Format,
    header_length: u4,
    unit_length: u64,
};

pub fn readUnitHeader(r: *Reader, endian: Endian) OpenError!UnitHeader {
    return switch (try r.takeInt(u32, endian)) {
        0...0xfffffff0 - 1 => |unit_length| .{
            .format = .@"32",
            .header_length = 4,
            .unit_length = unit_length,
        },
        0xfffffff0...0xffffffff - 1 => bad(),
        0xffffffff => .{
            .format = .@"64",
            .header_length = 12,
            .unit_length = try r.takeInt(u64, endian),
        },
    };
}

/// Returns the DWARF register number for an x86_64 register number found in compact unwind info
pub fn compactUnwindToDwarfRegNumber(unwind_reg_number: u3) !u16 {
    return switch (unwind_reg_number) {
        1 => 3, // RBX
        2 => 12, // R12
        3 => 13, // R13
        4 => 14, // R14
        5 => 15, // R15
        6 => 6, // RBP
        else => error.InvalidRegister,
    };
}

/// Returns `null` for CPU architectures without an instruction pointer register.
pub fn ipRegNum(arch: std.Target.Cpu.Arch) ?u16 {
    return switch (arch) {
        .aarch64, .aarch64be => 32,
        .alpha => 64,
        .arc, .arceb => 160,
        .arm, .armeb, .thumb, .thumbeb => 15,
        .csky => 64,
        .hexagon => 76,
        .kvx => 64,
        .lanai => 2,
        .loongarch32, .loongarch64 => 64,
        .m68k => 26,
        .m88k => 64,
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
        => 66,
        .or1k => 35,
        .powerpc, .powerpcle, .powerpc64, .powerpc64le => 67,
        .riscv32, .riscv32be, .riscv64, .riscv64be => 65,
        .s390x => 65,
        .sparc, .sparc64 => 65,
        .ve => 144,
        .x86, .x86c16 => 8,
        .x86_64 => 16,
        else => null,
    };
}

pub fn fpRegNum(arch: std.Target.Cpu.Arch) u16 {
    return switch (arch) {
        .aarch64, .aarch64be => 29,
        .alpha => 15,
        .arc, .arceb => 27,
        .arm, .armeb => 11,
        .thumb, .thumbeb => 7,
        .csky => 14,
        .hexagon => 30,
        .kvx => 14,
        .lanai => 5,
        .loongarch32, .loongarch64 => 22,
        .m68k => 14,
        .m88k => 30,
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
        => 30,
        .or1k => 2,
        .powerpc, .powerpcle, .powerpc64, .powerpc64le => 1,
        .riscv32, .riscv32be, .riscv64, .riscv64be => 8,
        .s390x => 11,
        .sparc, .sparc64 => 30,
        .ve => 9,
        .x86, .x86c16 => 5,
        .x86_64 => 6,
        else => unreachable,
    };
}

pub fn spRegNum(arch: std.Target.Cpu.Arch) u16 {
    return switch (arch) {
        .aarch64, .aarch64be => 31,
        .alpha => 30,
        .arc, .arceb => 28,
        .arm, .armeb, .thumb, .thumbeb => 13,
        .csky => 14,
        .hexagon => 29,
        .kvx => 12,
        .lanai => 4,
        .loongarch32, .loongarch64 => 3,
        .m68k => 15,
        .m88k => 31,
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
        => 29,
        .or1k => 1,
        .powerpc, .powerpcle, .powerpc64, .powerpc64le => 1,
        .riscv32, .riscv32be, .riscv64, .riscv64be => 2,
        .s390x => 15,
        .sparc, .sparc64 => 14,
        .ve => 11,
        .x86, .x86c16 => 4,
        .x86_64 => 7,
        else => unreachable,
    };
}

/// This function is to make it handy to comment out the return and make it
/// into a crash when working on this file.
pub fn bad() error{InvalidDebugInfo} {
    if (debug_debug_mode) @panic("bad dwarf");
    return error.InvalidDebugInfo;
}

pub fn missing() error{MissingDebugInfo} {
    if (debug_debug_mode) @panic("missing dwarf");
    return error.MissingDebugInfo;
}

fn getStringGeneric(opt_str: ?[]const u8, offset: u64) ![:0]const u8 {
    const str = opt_str orelse return bad();
    if (offset > str.len) return bad();
    const casted_offset = std.math.cast(usize, offset) orelse return bad();
    // Valid strings always have a terminating zero byte
    const last = std.mem.findScalarPos(u8, str, casted_offset, 0) orelse return bad();
    return str[casted_offset..last :0];
}

pub fn getSymbols(
    dwarf: *Dwarf,
    symbol_allocator: Allocator,
    text_arena: Allocator,
    pc: u64,
    resolve_inline_callers: bool,
    symbols: *std.ArrayList(std.debug.Symbol),
) std.debug.SelfInfoError!void {
    _ = resolve_inline_callers;
    const gpa = std.debug.getDebugInfoAllocator();

    const compile_unit = dwarf.findCompileUnit(pc) orelse return missing();
    try symbols.append(symbol_allocator, .{
        .name = dwarf.getSymbolName(pc),
        .compile_unit_name = compile_unit.ptr(dwarf).name,
        .source_location = dwarf.getLineNumberInfo(
            gpa,
            text_arena,
            compile_unit,
            pc,
        ) catch |err| switch (err) {
            error.MissingDebugInfo, error.InvalidDebugInfo => null,
            error.ReadFailed,
            error.EndOfStream,
            error.Overflow,
            error.StreamTooLong,
            => return bad(),
            else => |e| return e,
        },
    });
}

/// DWARF5 7.4: "In the 32-bit DWARF format, all values that represent lengths of DWARF sections and
/// offsets relative to the beginning of DWARF sections are represented using four bytes. In the
/// 64-bit DWARF format, all values that represent lengths of DWARF sections and offsets relative to
/// the beginning of DWARF sections are represented using eight bytes".
///
/// This function is for reading such values.
fn readFormatSizedInt(r: *Reader, format: std.dwarf.Format, endian: Endian) !u64 {
    return switch (format) {
        .@"32" => try r.takeInt(u32, endian),
        .@"64" => try r.takeInt(u64, endian),
    };
}

fn readAddress(r: *Reader, endian: Endian, addr_size_bytes: u8) !u64 {
    return switch (addr_size_bytes) {
        2 => try r.takeInt(u16, endian),
        4 => try r.takeInt(u32, endian),
        8 => try r.takeInt(u64, endian),
        else => return bad(),
    };
}

pub fn order_u64(lhs: u64, rhs: u64) std.math.Order {
    return std.math.order(lhs, rhs);
}
