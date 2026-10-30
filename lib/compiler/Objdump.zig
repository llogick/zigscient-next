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

gpa: Allocator,
terminal_mode: Io.Terminal.Mode,
opts: Options,
w: *Io.Writer,

input_name: []const u8,

seek_base: u64,
raw_file_reader: *Io.File.Reader,
limited_reader: ?*Io.Reader.Limited,

any_errors: bool,

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
    archive_member_headers: bool,
    archive_symbols: bool,
    file_headers: bool,
    sections: bool,
    symbols: bool,
    relocs: bool,

    archive_member_filters: []const []const u8,
    section_filters: []const []const u8,
    symbol_filters: []const []const u8,

    basenames: bool,
    redact: std.enums.EnumSet(FieldKind),

    phdrs: bool,
    dynamic: bool,
    symbol_versions: bool,
    version_definitions: bool,

    imports: bool,
    exports: bool,
    exports_sort: bool,
    strings: bool,
    tls: bool,
};

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

    var archive_member_headers: bool = false;
    var archive_symbols: bool = false;
    var file_headers: bool = false;
    var sections: bool = false;
    var symbols: bool = false;
    var relocs: bool = false;

    var archive_member_filters: std.ArrayList([]const u8) = .empty;
    var section_filters: std.ArrayList([]const u8) = .empty;
    var symbol_filters: std.ArrayList([]const u8) = .empty;

    var basenames: bool = false;
    var redact: std.enums.EnumSet(FieldKind) = .empty;

    var phdrs: bool = false;
    var dynamic: bool = false;
    var symbol_versions: bool = false;
    var version_definitions: bool = false;

    var imports: bool = false;
    var exports: bool = false;
    var exports_sort: bool = false;
    var strings: bool = false;
    var tls: bool = false;

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
            archive_member_headers = true;
            archive_symbols = true;
            file_headers = true;
            sections = true;
            symbols = true;
            relocs = true;

            phdrs = true;
            dynamic = true;
            symbol_versions = true;
            version_definitions = true;

            imports = true;
            exports = true;
            strings = true;
            tls = true;
        } else if (mem.eql(u8, arg, "--archive-member-headers")) {
            archive_member_headers = true;
        } else if (mem.eql(u8, arg, "--archive-symbols")) {
            archive_symbols = true;
        } else if (mem.eql(u8, arg, "--file-headers")) {
            file_headers = true;
        } else if (mem.eql(u8, arg, "--sections")) {
            sections = true;
        } else if (mem.eql(u8, arg, "--symbols")) {
            symbols = true;
        } else if (mem.eql(u8, arg, "--relocs")) {
            relocs = true;
        } else if (mem.startsWith(u8, arg, "--only-archive-member=")) {
            (try archive_member_filters.addOne(arena)).* = try arena.dupe(u8, arg["--only-archive-member=".len..]);
        } else if (mem.startsWith(u8, arg, "--only-section=")) {
            (try section_filters.addOne(arena)).* = try arena.dupe(u8, arg["--only-section=".len..]);
        } else if (mem.startsWith(u8, arg, "--only-symbol=")) {
            (try symbol_filters.addOne(arena)).* = try arena.dupe(u8, arg["--only-symbol=".len..]);
        } else if (mem.eql(u8, arg, "-s") or mem.eql(u8, arg, "--snapshot")) {
            basenames = true;
            redact = .full;
        } else if (mem.eql(u8, arg, "--basenames")) {
            basenames = true;
        } else if (mem.startsWith(u8, arg, "--redact=")) {
            const kind = arg["--redact=".len..];
            if (std.meta.stringToEnum(FieldKind, kind)) |field_kind| {
                redact.insert(field_kind);
            } else if (mem.eql(u8, kind, "all")) {
                redact = .full;
            } else {
                fatal("unrecognized redaction kind: {s}", .{kind});
            }
        } else if (mem.eql(u8, arg, "--phdrs")) {
            phdrs = true;
        } else if (mem.eql(u8, arg, "--dynamic")) {
            dynamic = true;
        } else if (mem.eql(u8, arg, "--symbol-versions")) {
            symbol_versions = true;
        } else if (mem.eql(u8, arg, "--version-definitions")) {
            version_definitions = true;
        } else if (mem.eql(u8, arg, "--imports")) {
            imports = true;
        } else if (mem.eql(u8, arg, "--exports")) {
            exports = true;
        } else if (mem.eql(u8, arg, "--exports=sort")) {
            exports = true;
            exports_sort = true;
        } else if (mem.eql(u8, arg, "--strings")) {
            strings = true;
        } else if (mem.eql(u8, arg, "--tls")) {
            tls = true;
        } else {
            fatal("unrecognized argument: {s}", .{arg});
        }
    }

    if (input_paths.items.len == 0) {
        fatal("missing input file path positional argument", .{});
    }

    const opts: Options = .{
        .archive_member_headers = archive_member_headers,
        .archive_symbols = archive_symbols,
        .file_headers = file_headers,
        .sections = sections,
        .symbols = symbols,
        .relocs = relocs,

        .archive_member_filters = archive_member_filters.items,
        .section_filters = section_filters.items,
        .symbol_filters = symbol_filters.items,

        .basenames = basenames,
        .redact = redact,

        .phdrs = phdrs,
        .dynamic = dynamic,
        .symbol_versions = symbol_versions,
        .version_definitions = version_definitions,

        .imports = imports,
        .exports = exports,
        .exports_sort = exports_sort,
        .strings = strings,
        .tls = tls,
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
        return elf.dump(d);
    }
    try d.seekTo(0);

    if (try macho.identify(r)) {
        return macho.dump(d);
    }
    try d.seekTo(0);

    if (try wasm.identify(r)) {
        return wasm.dump(d);
    }
    try d.seekTo(0);

    if (try coff.identifyImage(r)) {
        return coff.dumpImage(d);
    }
    try d.seekTo(0);

    if (try coff.identifyObject(r)) {
        return coff.dumpObject(d);
    }
    try d.seekTo(0);

    if (try coff.identifyImport(r)) {
        return coff.dumpImport(d);
    }
    try d.seekTo(0);

    return d.fail("unknown file format", .{});
}

const usage =
    \\Usage: zig objdump [options] file
    \\
    \\Options:
    \\  -h, --help                         Print this help and exit
    \\  --all                              Display all available information, including file-format-specific
    \\  --archive-member-headers           Display archive member headers
    \\  --archive-symbols                  Display contents of archive symbol tables
    \\  --file-headers                     Display file-format specific headers
    \\  --sections                         Display section headers
    \\  --symbols                          Display symbol tables
    \\  --relocs                           Display relocations
    \\  --only-archive-member=[name]       Only consider archive members names that contain [name]. Can be specified multiple times.
    \\  --only-section=[name]              Only consider section names that contain [name]. Can be specified multiple times.
    \\  --only-symbol=[name]               Only consider symbol names that contain [name]. Can be specified multiple times.
    \\  -s, --snapshot                     Alias for --basenames --redact=all
    \\  --basenames                        Only show the basenames of input files and archive members instead of their full paths.
    \\  --redact=[kind]                    Redact the specified field kind. Intended for snapshot testing.
    \\      address                        Virtual addresses
    \\      offset                         File offsets
    \\      index                          Indices and ordinals
    \\      size                           Sizes and lengths
    \\      align                          Alignments
    \\      all                            All of the above
    \\
    \\ELF-specific options:
    \\  --phdrs                            Display program headers
    \\  --dynamic                          Display dynamic entries
    \\  --symbol-versions                  Display GNU symbol version tables
    \\  --version-definitions              Display GNU symbol version definitions and requirements
    \\
    \\COFF-specific options:
    \\  --imports                          Display imported symbols
    \\  --exports[=sort]                   Display exported symbols.
    \\                                     For import libraries, displays the symbol list and import headers.
    \\                                     Specify =sort to sort the import headers by symbol name.
    \\  --strings                          Display string tables
    \\  --tls                              Display TLS information
    \\
;
