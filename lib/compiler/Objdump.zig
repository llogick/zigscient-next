const Objdump = @This();

const std = @import("std");
const assert = std.debug.assert;
const fatal = std.process.fatal;
const mem = std.mem;
const Allocator = mem.Allocator;
const Io = std.Io;

const archive = @import("Objdump/archive.zig");
const elf = @import("Objdump/elf.zig");
const macho = @import("Objdump/macho.zig");
const wasm = @import("Objdump/wasm.zig");
const coff = @import("Objdump/coff.zig");

const Expr = @import("Objdump/Expr.zig");

gpa: Allocator,
terminal_mode: Io.Terminal.Mode,
opts: Options,
w: *Io.Writer,

input_name: []const u8,

seek_base: u64,
raw_file_reader: *Io.File.Reader,
limited_reader: ?*Io.Reader.Limited,

any_errors: bool,

/// Prevents `reportOfmtErrors` from repeating the same error for every file in an archive.
reported_ofmt_errors: std.enums.EnumSet(std.meta.Tag(Table)),

pub fn reader(d: *const Objdump) *Io.Reader {
    if (d.limited_reader) |lr| {
        return &lr.interface;
    } else {
        return &d.raw_file_reader.interface;
    }
}

/// Intended for use from `format` functions printing to `d.w`.
pub fn setColor(d: *const Objdump, w: *Io.Writer, color: Io.Terminal.Color) Io.Writer.Error!void {
    assert(w == d.w);
    const term: Io.Terminal = .{ .writer = w, .mode = d.terminal_mode };
    term.setColor(color) catch |err| switch (err) {
        error.Unexpected => {},
        error.Canceled => term.mode.windows_api.io.recancel(),
        else => |e| return e,
    };
}

pub fn logicalPos(d: *const Objdump) u64 {
    if (d.limited_reader) |lr| {
        return d.raw_file_reader.logicalPos() - lr.interface.bufferedLen() - d.seek_base;
    } else {
        assert(d.seek_base == 0);
        return d.raw_file_reader.logicalPos();
    }
}

pub fn atEnd(d: *const Objdump) bool {
    if (d.limited_reader) |lr| {
        if (lr.interface.bufferedLen() != 0) return false;
        if (lr.remaining == .nothing) return true;
    } else {
        assert(d.seek_base == 0);
    }
    return d.raw_file_reader.atEnd();
}

pub fn seekTo(d: *const Objdump, new_pos: u64) (Io.Cancelable || Io.UnexpectedError || error{ReadFailed})!void {
    if (d.limited_reader) |lr| {
        const old_pos = d.logicalPos();
        if (new_pos >= old_pos and new_pos <= old_pos + lr.interface.bufferedLen()) {
            lr.interface.toss(@intCast(new_pos - old_pos));
            return;
        }
        if (lr.remaining.toInt64()) |old_remaining| {
            const total_available = old_pos + lr.interface.bufferedLen() + old_remaining;
            lr.remaining = .limited64(total_available -| new_pos);
        }
        lr.interface.tossBuffered();
    } else {
        assert(d.seek_base == 0);
    }
    d.raw_file_reader.seekTo(d.seek_base + new_pos) catch |err| switch (err) {
        error.Unseekable,
        error.AccessDenied,
        => |e| fatal("failed to seek input file: {t}", .{e}),

        error.EndOfStream => {}, // swallow and let the caller re-trigger

        else => |e| return e,
    };
}

pub const Error = Allocator.Error ||
    error{ReadFailed} ||
    Io.Writer.Error ||
    Io.Cancelable ||
    Io.UnexpectedError ||
    error{AlreadyReported};

pub const Options = struct {
    archive_symbol_tables: bool,
    archive_symbol_filters: []const []const u8,
    archive_member_headers: bool,
    archive_member_filters: []const []const u8,

    file_headers: bool,

    basenames: bool,
    redact: std.enums.EnumSet(FieldKind),

    ofmt_error: std.enums.EnumArray(std.meta.Tag(Table), ?[]const u8),

    tables: Tables,

    pub const Tables = struct {
        elf: std.enums.EnumArray(elf.Table, ?Table.Config),
        macho: std.enums.EnumArray(macho.Table, ?Table.Config),
        coff: std.enums.EnumArray(coff.Table, ?Table.Config),
        wasm: std.enums.EnumArray(wasm.Table, ?Table.Config),
    };
};

fn reportOfmtErrors(d: *Objdump, ofmt: std.meta.Tag(Table)) Objdump.Error!void {
    if (d.opts.ofmt_error.get(ofmt)) |msg| {
        try d.w.flush();
        if (!d.reported_ofmt_errors.contains(ofmt)) {
            d.any_errors = true;
            d.reported_ofmt_errors.insert(ofmt);
            std.log.err("cannot dump {t} file: {s}", .{ ofmt, msg });
            try d.w.writeByte('\n'); // trailing blank line for this input
        }
        assert(d.any_errors);
        return error.AlreadyReported;
    }
}

/// Returns a formatter which supports numeric formatting ("{d}", "{x:0>10}", etc), with the
/// following differences from the default integer formatting logic:
///
/// * If the given `kind` is to be redacted, the value is replaced with '?' bytes.
///
/// * If the value is formatted as binary, hexadecimal, or octal, the appropriate base prefix (one
///   of "0b", "0o", "0x") is automatically prepended. It is written before any "fill" bytes, so it
///   is generally expected that the caller will use '0' as the fill byte if specifying a width.
///
/// * If the value is formatted with a width specifier, then insignificant zeroes (and the base
///   prefix described above) are rendered to the output in grey (if color is enabled).
pub fn fmtIntField(d: *const Objdump, val: u64, kind: ?FieldKind) FormatIntField {
    return .{ .d = d, .val = val, .kind = kind };
}
pub const FormatIntField = struct {
    d: *const Objdump,
    val: u64,
    kind: ?FieldKind,
    pub fn formatNumber(ctx: FormatIntField, w: *Io.Writer, number: std.fmt.Number) Io.Writer.Error!void {
        const prefix: []const u8 = switch (number.mode) {
            .decimal => "",
            .binary => "0b",
            .octal => "0o",
            .hex => "0x",
            .scientific => unreachable, // not supported
        };

        var buffer: [128]u8 = undefined;
        const significant_digits: []const u8, const fill_digit: u8 = digits: {
            if (ctx.kind) |k| {
                if (ctx.d.opts.redact.contains(k)) {
                    if (number.width == null) {
                        // One "?" looks weird, use a few.
                        break :digits .{ "???", '?' };
                    } else {
                        break :digits .{ "", '?' };
                    }
                }
            }

            if (number.width != null and number.fill == '0' and ctx.val == 0) {
                // We want all of the zeroes to be highlighted as insignificant instead of the last
                // zero appearing as significant.
                break :digits .{ "", number.fill };
            }

            var buf_w: std.Io.Writer = .fixed(&buffer);
            buf_w.printInt(
                ctx.val,
                number.mode.base().?,
                number.case,
                .{},
            ) catch |err| switch (err) {
                // The value is a `u64`, so the worst-case output is 64 bits when base = 2.
                error.WriteFailed => unreachable,
            };
            break :digits .{ buf_w.buffered(), number.fill };
        };

        if (number.width) |width| {
            try ctx.d.setColor(w, .dim);
            try w.writeAll(prefix);
            try w.splatByteAll(fill_digit, width -| significant_digits.len);
            try ctx.d.setColor(w, .reset);
            try w.writeAll(significant_digits);
        } else {
            assert(significant_digits.len > 0);
            try w.writeAll(prefix);
            try w.writeAll(significant_digits);
        }
    }
};

pub fn fail(
    d: *Objdump,
    comptime fmt: []const u8,
    args: anytype,
) (Io.Writer.Error || error{AlreadyReported}) {
    d.any_errors = true;
    try d.w.flush();
    std.log.err("failed to parse {s}: " ++ fmt, .{d.input_name} ++ args);
    try d.w.writeByte('\n'); // trailing blank line for this input
    return error.AlreadyReported;
}

pub fn warn(
    d: *Objdump,
    comptime fmt: []const u8,
    args: anytype,
) Error!void {
    d.any_errors = true;
    try d.w.flush();
    std.log.warn(fmt, args);
}

pub const FieldKind = enum {
    address,
    offset,
    index,
    size,
    @"align",
};

const Element = enum {
    @"input-path",
    @"member-path",
};

pub fn main(init: std.process.Init) (Io.UnexpectedError || Io.Cancelable || Allocator.Error)!u8 {
    const io = init.io;
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);

    var input_paths: std.ArrayList([]const u8) = .empty;

    var archive_symbol_tables: bool = false;
    var archive_symbol_filters: std.ArrayList([]const u8) = .empty;
    var archive_member_headers: bool = false;
    var archive_member_filters: std.ArrayList([]const u8) = .empty;

    var file_headers: bool = false;

    var basenames: bool = false;
    var redact: std.enums.EnumSet(FieldKind) = .empty;

    var tables: Options.Tables = .{
        .elf = .initFill(null),
        .macho = .initFill(null),
        .coff = .initFill(null),
        .wasm = .initFill(null),
    };

    var wip_tables_buf: [@typeInfo(Table).@"union".field_names.len]Table = undefined;

    var ofmt_error: std.enums.EnumArray(std.meta.Tag(Table), ?[]const u8) = .initFill(null);

    // A single `--[table name]` argument may match multiple ofmts, which means we are sometimes
    // populating the configuration of multiple tables at the same time. In that case this variable
    // is a slice of all tables we are populating.
    //
    // A length of 0 means we have not seen any `--[table name]` argument yet.
    var wip_tables: []Table = &.{};

    var seen_sort: bool = undefined;
    var seen_filter: bool = undefined;

    for (args[1..]) |arg| {
        if (!mem.startsWith(u8, arg, "-")) {
            try input_paths.append(arena, arg);
            continue;
        }

        if (mem.eql(u8, arg, "-h") or mem.eql(u8, arg, "--help")) {
            Io.File.stdout().writeStreamingAll(io, usage) catch |err|
                fatal("failed to write stdout: {t}", .{err});
            return 0;
        } else if (mem.eql(u8, arg, "--all")) {
            archive_symbol_tables = true;
            archive_member_headers = true;
            file_headers = true;
            wip_tables = &.{};
            tables = .{
                .elf = .initFill(.{ .sort = null, .filter = null }),
                .macho = .initFill(.{ .sort = null, .filter = null }),
                .coff = .initFill(.{ .sort = null, .filter = null }),
                .wasm = .initFill(.{ .sort = null, .filter = null }),
            };
        } else if (mem.eql(u8, arg, "--archive-symbol-tables")) {
            archive_symbol_tables = true;
        } else if (mem.cutPrefix(u8, arg, "--archive-symbol-filter=")) |symbol_name| {
            try archive_symbol_filters.append(arena, symbol_name);
        } else if (mem.eql(u8, arg, "--archive-member-headers")) {
            archive_member_headers = true;
        } else if (mem.cutPrefix(u8, arg, "--archive-member-filter=")) |member_name| {
            try archive_member_filters.append(arena, member_name);
        } else if (mem.eql(u8, arg, "--file-headers")) {
            file_headers = true;
        } else if (mem.eql(u8, arg, "-s") or mem.eql(u8, arg, "--snapshot")) {
            basenames = true;
            redact = .full;
        } else if (mem.eql(u8, arg, "--basenames")) {
            basenames = true;
        } else if (mem.cutPrefix(u8, arg, "--redact=")) |kind| {
            if (std.meta.stringToEnum(FieldKind, kind)) |field_kind| {
                redact.insert(field_kind);
            } else if (mem.eql(u8, kind, "all")) {
                redact = .full;
            } else {
                fatal("unrecognized redaction kind: {s}", .{kind});
            }
        } else if (mem.cutPrefix(u8, arg, "--filter=")) |expr_src| {
            if (wip_tables.len == 0) {
                fatal("'--filter' before any table", .{});
            }
            if (seen_filter) {
                fatal("multiple '--filter' options for the same table", .{});
            }
            seen_filter = true;
            var err_msg: []const u8 = undefined;
            const unresolved_expr = Expr.parse(arena, expr_src, &err_msg) catch |err| switch (err) {
                error.OutOfMemory => |e| return e,
                error.ParseFail => std.process.fatal(
                    "failed to parse filter expression {q}: {s}",
                    .{ expr_src, err_msg },
                ),
            };
            for (wip_tables) |wip_table| {
                var expr = try unresolved_expr.clone(arena);
                if (expr.resolve(
                    arena,
                    wip_table.propertyNames(),
                    wip_table.propertyTypes(),
                    .bool,
                    &err_msg,
                )) {
                    const config_ptr: *?Table.Config = switch (wip_table) {
                        inline else => |ofmt_table, ofmt| @field(tables, @tagName(ofmt)).getPtr(ofmt_table),
                    };
                    config_ptr.*.?.filter = expr;
                } else |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    error.ResolveFail => if (ofmt_error.get(wip_table) == null) {
                        ofmt_error.set(wip_table, try std.fmt.allocPrint(
                            arena,
                            "invalid filter expression {q}: {s}",
                            .{ expr_src, err_msg },
                        ));
                    },
                }
            }
        } else if (mem.cutPrefix(u8, arg, "--sort=")) |expr_src| {
            if (wip_tables.len == 0) {
                fatal("'--sort' before any table", .{});
            }
            if (seen_sort) {
                fatal("multiple '--sort' options for the same table", .{});
            }
            seen_sort = true;
            var err_msg: []const u8 = undefined;
            const unresolved_expr = Expr.parse(arena, expr_src, &err_msg) catch |err| switch (err) {
                error.OutOfMemory => |e| return e,
                error.ParseFail => std.process.fatal(
                    "failed to parse sort expression {q}: {s}",
                    .{ expr_src, err_msg },
                ),
            };
            for (wip_tables) |wip_table| {
                var expr = try unresolved_expr.clone(arena);
                if (expr.resolve(
                    arena,
                    wip_table.propertyNames(),
                    wip_table.propertyTypes(),
                    null, // any type can be used to sort
                    &err_msg,
                )) {
                    const config_ptr: *?Table.Config = switch (wip_table) {
                        inline else => |ofmt_table, ofmt| @field(tables, @tagName(ofmt)).getPtr(ofmt_table),
                    };
                    config_ptr.*.?.sort = expr;
                } else |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    error.ResolveFail => if (ofmt_error.get(wip_table) == null) {
                        ofmt_error.set(wip_table, try std.fmt.allocPrint(
                            arena,
                            "invalid sort expression {q}: {s}",
                            .{ expr_src, err_msg },
                        ));
                    },
                }
            }
        } else {
            if (std.mem.cutPrefix(u8, arg, "--")) |table_name| {
                // As the very last thing we do, assume this argument is a table name (e.g.
                // `--sections`) and update `wip_tables`. This must be the very last thing we check
                // because the way we check for this invalidates the old contents of `wip_tables`.

                wip_tables = wip_tables_buf[0..0];
                seen_sort = false;
                seen_filter = false;

                inline for (@typeInfo(Options.Tables).@"struct".field_names) |ofmt_name| {
                    const ofmt = @field(std.meta.Tag(Table), ofmt_name);
                    const OfmtTable = @FieldType(Table, ofmt_name);
                    if (std.meta.stringToEnum(OfmtTable, table_name)) |table| {
                        if (@field(tables, ofmt_name).get(table) != null) {
                            fatal("table {q} specified twice", .{table_name});
                        }
                        @field(tables, ofmt_name).set(table, .{ .sort = null, .filter = null });
                        wip_tables_buf[wip_tables.len] = @unionInit(Table, ofmt_name, table);
                        wip_tables.len += 1;
                    } else if (ofmt_error.get(ofmt) == null) {
                        const msg = try std.fmt.allocPrint(arena, "table {q} does not exist in this format", .{table_name});
                        ofmt_error.set(ofmt, msg);
                    }
                }
                if (wip_tables.len > 0) {
                    // This was indeed a valid table argument for at least one ofmt.
                    continue;
                }
            }
            fatal("unrecognized argument: {s}", .{arg});
        }
    }

    if (input_paths.items.len == 0) {
        fatal("missing input file path positional argument", .{});
    }

    const opts: Options = .{
        .archive_symbol_tables = archive_symbol_tables,
        .archive_symbol_filters = archive_symbol_filters.items,
        .archive_member_headers = archive_member_headers,
        .archive_member_filters = archive_member_filters.items,

        .file_headers = file_headers,

        .basenames = basenames,
        .redact = redact,

        .tables = tables,

        .ofmt_error = ofmt_error,
    };

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);

    const terminal_mode: Io.Terminal.Mode = try .detect(
        io,
        .stdout(),
        if (init.environ_map.get("NO_COLOR")) |s| s.len > 0 else false,
        if (init.environ_map.get("CLICOLOR_FORCE")) |s| s.len > 0 else false,
    );

    var any_errors: bool = false;

    for (input_paths.items) |input_path| {
        var file = std.Io.Dir.cwd().openFile(io, input_path, .{}) catch |err| switch (err) {
            error.Canceled => |e| return e,
            else => |open_err| {
                std.log.err("failed to open {q}: {t}", .{ input_path, open_err });
                continue;
            },
        };
        defer file.close(io);

        var file_buffer: [4096]u8 = undefined;
        var file_reader = file.reader(io, &file_buffer);

        var d: Objdump = .{
            .gpa = init.gpa,
            .terminal_mode = terminal_mode,
            .opts = opts,
            .w = &stdout_writer.interface,
            .input_name = switch (opts.basenames) {
                false => input_path,
                true => std.fs.path.basename(input_path),
            },

            .seek_base = 0,
            .raw_file_reader = &file_reader,
            .limited_reader = null,

            .any_errors = false,

            .reported_ofmt_errors = .empty,
        };

        d.dumpOne(input_path) catch |err| switch (err) {
            error.WriteFailed => switch (stdout_writer.err.?) {
                error.Canceled => |e| return e,
                else => |e| fatal("failed to write stdout: {t}", .{e}),
            },
            else => |e| return e,
        };
        stdout_writer.flush() catch |err| switch (err) {
            error.Canceled => |e| return e,
            else => |e| fatal("failed to write stdout: {t}", .{e}),
        };

        if (d.any_errors) any_errors = true;
    }

    if (any_errors) {
        return 1;
    } else {
        return 0;
    }
}

const DumpOneError = Allocator.Error ||
    Io.Writer.Error ||
    Io.Cancelable ||
    Io.UnexpectedError;

fn dumpOne(d: *Objdump, input_path: []const u8) DumpOneError!void {
    d.dumpOneInner() catch |err| switch (err) {
        error.ReadFailed => switch (d.raw_file_reader.err.?) {
            error.Canceled => |e| return e,
            else => |e| {
                d.any_errors = true;
                try d.w.flush();
                std.log.err("failed to read {q}: {t}", .{ input_path, e });
            },
        },
        error.AlreadyReported => assert(d.any_errors),
        else => |e| return e,
    };
}
fn dumpOneInner(d: *Objdump) Objdump.Error!void {
    const r = d.reader();

    if (try archive.identify(r)) {
        return archive.dump(d);
    }
    try d.seekTo(0);

    return d.dumpNonArchive();
}
pub fn dumpNonArchive(d: *Objdump) Objdump.Error!void {
    const r = d.reader();

    if (try elf.identify(r)) {
        try d.reportOfmtErrors(.elf);
        return elf.dump(d);
    }
    try d.seekTo(0);

    if (try macho.identify(r)) {
        try d.reportOfmtErrors(.macho);
        return macho.dump(d);
    }
    try d.seekTo(0);

    if (try wasm.identify(r)) {
        try d.reportOfmtErrors(.wasm);
        return wasm.dump(d);
    }
    try d.seekTo(0);

    if (try coff.identifyImage(r)) {
        try d.reportOfmtErrors(.coff);
        return coff.dumpImage(d);
    }
    try d.seekTo(0);

    if (try coff.identifyObject(r)) {
        try d.reportOfmtErrors(.coff);
        return coff.dumpObject(d);
    }
    try d.seekTo(0);

    if (try coff.identifyImport(r)) {
        try d.reportOfmtErrors(.coff);
        return coff.dumpImport(d);
    }
    try d.seekTo(0);

    return d.fail("unknown file format", .{});
}

const usage =
    \\Usage: zig objdump [options] file... ([table] [table options])...
    \\
    \\Options:
    \\  -h, --help                        Print this help and exit
    \\  --all                             Display all available information, including file-format-specific
    \\  --archive-symbol-tables           Display contents of archive symbol tables
    \\  --archive-symbol-filter=<name>    Only consider symbols in archives whose names contain <anem>.
    \\                                    Can be specified multiple times.
    \\  --archive-member-headers          Display archive member headers
    \\  --archive-member-filter=<name>    Only consider archive members names that contain <name>.
    \\                                    Can be specified multiple times.
    \\  --file-headers                    Display each file's format-specific header
    \\  -s, --snapshot                    Alias for --basenames --redact=all
    \\  --basenames                       Only show the basenames of input files and archive members instead of their full paths.
    \\  --redact=[kind]                   Redact the specified field kind. Intended for snapshot testing.
    \\      address                       Virtual addresses
    \\      offset                        File offsets
    \\      index                         Indices and ordinals
    \\      size                          Sizes and lengths
    \\      align                         Alignments
    \\      all                           All of the above
    \\
    \\Per-Table Options:
    \\  --filter=<expr>                   Filter the table to rows where <expr> evaluates to 'true'
    \\  --sort=<expr>                     Sort the table by the result of evaluating <expr>
    \\
    \\  <expr> is an expression with the following syntax:
    \\    Comparison Operators: ==, !=, >=, <=, >, <
    \\    Arithmetic Operators: +, -, *, /, %
    \\    Boolean Operators: and, or, !
    \\    String Operators: `haystack ~ needle` is true if haystack contains needle
    \\    Literals: "string", 0x1234, true, false (Zig syntax)
    \\
    \\ELF Tables:
    \\
++ usageTables(elf.Table) ++
    \\
    \\COFF Tables:
    \\
++ usageTables(coff.Table) ++
    \\
    \\Mach-O Tables:
    \\
++ usageTables(macho.Table) ++
    \\
    \\WASM Tables:
    \\
++ usageTables(wasm.Table) ++
    \\
    \\Examples:
    \\  zig objdump --basenames a.out \
    \\      --sections --filter='name ~ ".debug_"' --sort=name \
    \\      --symbols --filter='size > 0x10'
    \\    Reads and dumps the file `a.out`. Sections whose names contain ".debug_" are
    \\    dumped sorted by name. Symbols whose sizes exceed 16 bytes are dumped sorted
    \\    by index (the default). Any file paths are replaced by their basenames.
    \\
;

fn usageTables(comptime OfmtTable: type) []const u8 {
    var str: []const u8 = "";
    for (@typeInfo(OfmtTable).@"enum".field_names) |table_name| {
        str = str ++ std.fmt.comptimePrint("  --{s: <32}{s}\n", .{
            table_name, @field(OfmtTable, table_name).cliDescription(),
        });
    }
    return str;
}

pub const Table = union(enum) {
    elf: elf.Table,
    macho: macho.Table,
    coff: coff.Table,
    wasm: wasm.Table,

    pub const PropertyType = enum {
        bool,
        int,
        string,
        pub fn ZigType(comptime t: PropertyType) type {
            return switch (t) {
                .bool => bool,
                .int => u64,
                .string => []const u8,
            };
        }
    };

    fn propertyNames(table: Table) []const []const u8 {
        return switch (table) {
            inline else => |ofmt_specific_table| switch (ofmt_specific_table) {
                inline else => |t| @typeInfo(t.Property()).@"enum".field_names,
            },
        };
    }
    fn propertyTypes(table: Table) []const PropertyType {
        return switch (table) {
            inline else => |ofmt_specific_table| switch (ofmt_specific_table) {
                inline else => |t| &t.Property().types.values,
            },
        };
    }

    pub const PropertyIndex = enum(u32) { _ };

    pub const Config = struct {
        sort: ?Expr,
        filter: ?Expr,

        pub fn iterateRows(
            conf: Config,
            d: *const Objdump,
            comptime table: Table,
            len: u64,
            comptime Context: type,
            context: *const Context,
        ) Allocator.Error!RowIterator {
            const static = struct {
                pub fn getPropertyTypeErased(
                    type_erased_ctx: *const anyopaque,
                    entry_index: u64,
                    property_index: PropertyIndex,
                ) Expr.Value {
                    const ctx: *const Context = @ptrCast(@alignCast(type_erased_ctx));
                    const Property = @field(table, @tagName(table)).Property();
                    return switch (@as(Property, @bitCast(property_index))) {
                        inline else => |prop| @unionInit(
                            Expr.Value,
                            @tagName(Property.types.get(prop)),
                            ctx.property(entry_index, prop),
                        ),
                    };
                }
            };

            return conf.iterateRowsInner(
                d,
                len,
                &static.getPropertyTypeErased,
                context,
            );
        }
        /// `iterateRows` is a generic interface wrapping a concrete implementation, so it calls
        /// this function for all the non-generic logic.
        fn iterateRowsInner(
            conf: Config,
            d: *const Objdump,
            len: u64,
            getProperty: *const fn (*const anyopaque, u64, PropertyIndex) Expr.Value,
            get_property_context: *const anyopaque,
        ) Allocator.Error!RowIterator {
            const gpa = d.gpa;

            const eval_stack_buf = try gpa.alloc(Expr.Value, @max(
                if (conf.sort) |expr| expr.max_stack_len.? else 0,
                if (conf.filter) |expr| expr.max_stack_len.? else 0,
            ));
            errdefer gpa.free(eval_stack_buf);

            const sort_expr = conf.sort orelse return .{
                .getProperty = getProperty,
                .get_property_context = get_property_context,
                .filter = conf.filter,
                .eval_stack_buf = eval_stack_buf,
                .len = len,
                .sorted_entry_indices = null,
                .index = 0,
            };

            const entry_indices = try gpa.alloc(u64, len);
            errdefer gpa.free(entry_indices);
            for (entry_indices, 0..) |*index_out, i| index_out.* = i;
            const SortContext = struct {
                expr: Expr,
                eval_stack_buf: []Expr.Value,
                getProperty: *const fn (*const anyopaque, u64, PropertyIndex) Expr.Value,
                get_property_context: *const anyopaque,
                pub fn lessThan(sort_ctx: @This(), lhs: u64, rhs: u64) bool {
                    const lhs_result = sort_ctx.expr.eval(
                        sort_ctx.eval_stack_buf,
                        sort_ctx.getProperty,
                        sort_ctx.get_property_context,
                        lhs,
                    );
                    const rhs_result = sort_ctx.expr.eval(
                        sort_ctx.eval_stack_buf,
                        sort_ctx.getProperty,
                        sort_ctx.get_property_context,
                        rhs,
                    );
                    return switch (lhs_result) {
                        .bool => |lhs_bool| @intFromBool(lhs_bool) < @intFromBool(rhs_result.bool),
                        .int => |lhs_int| lhs_int < rhs_result.int,
                        .string => |lhs_string| std.mem.order(u8, lhs_string, rhs_result.string).compare(.lt),
                    };
                }
            };
            std.mem.sort(u64, entry_indices, @as(SortContext, .{
                .expr = sort_expr,
                .eval_stack_buf = eval_stack_buf,
                .getProperty = getProperty,
                .get_property_context = get_property_context,
            }), SortContext.lessThan);
            return .{
                .getProperty = getProperty,
                .get_property_context = get_property_context,
                .filter = conf.filter,
                .eval_stack_buf = eval_stack_buf,
                .len = len,
                .sorted_entry_indices = entry_indices,
                .index = 0,
            };
        }
        pub const RowIterator = struct {
            getProperty: *const fn (*const anyopaque, u64, PropertyIndex) Expr.Value,
            get_property_context: *const anyopaque,

            filter: ?Expr,
            eval_stack_buf: []Expr.Value,
            len: u64,
            sorted_entry_indices: ?[]u64,

            index: u64,

            pub fn deinit(it: *@This(), d: *const Objdump) void {
                const gpa = d.gpa;

                gpa.free(it.eval_stack_buf);
                if (it.sorted_entry_indices) |indices| {
                    gpa.free(indices);
                }

                it.* = undefined;
            }

            pub fn reset(it: *@This()) void {
                it.index = 0;
            }

            pub fn next(it: *@This()) ?u64 {
                while (true) {
                    if (it.index == it.len) {
                        return null;
                    }
                    const entry_index: u64 = if (it.sorted_entry_indices) |indices| index: {
                        break :index indices[it.index];
                    } else it.index;
                    it.index += 1;
                    if (it.filter) |filter| {
                        const filter_result = filter.eval(
                            it.eval_stack_buf,
                            it.getProperty,
                            it.get_property_context,
                            entry_index,
                        );
                        if (!filter_result.bool) {
                            continue;
                        }
                    }
                    return entry_index;
                }
            }

            pub fn empty(it: *@This()) bool {
                assert(it.index == 0);
                const filter = it.filter orelse return it.len == 0;
                while (it.index < it.len) : (it.index += 1) {
                    const entry_index: u64 = if (it.sorted_entry_indices) |indices| index: {
                        break :index indices[it.index];
                    } else it.index;
                    const filter_result = filter.eval(
                        it.eval_stack_buf,
                        it.getProperty,
                        it.get_property_context,
                        entry_index,
                    );
                    if (filter_result.bool) {
                        return false;
                    }
                }
                return true;
            }
        };
    };
};

test {
    _ = Expr;
}
