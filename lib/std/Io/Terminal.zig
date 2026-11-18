/// Abstraction for writing to a stream that might support terminal escape
/// codes.
const Terminal = @This();

const builtin = @import("builtin");
const is_windows = builtin.os.tag == .windows;

const std = @import("std");
const Io = std.Io;
const File = std.Io.File;

writer: *Io.Writer,
mode: Mode,

pub const Mode = union(enum) {
    no_color,
    escape_codes,
    /// Deprecated, and only supported by the deprecated `setColor` API.
    ///
    /// Windows 10 RS1 introduced support for ANSI escape codes.
    windows_api: WindowsApi,

    pub const WindowsApi = if (!is_windows) noreturn else struct {
        io: Io,
        file: File,
        reset_attributes: u16,
    };

    /// Detect suitable TTY configuration options for the given file (commonly
    /// stdout/stderr).
    ///
    /// Will attempt to enable ANSI escape code support if necessary/possible.
    ///
    /// * `NO_COLOR` indicates whether "NO_COLOR" environment variable is
    ///   present and non-empty.
    /// * `CLICOLOR_FORCE` indicates whether "CLICOLOR_FORCE" environment
    ///   variable is present and non-empty.
    pub fn detect(io: Io, file: File, NO_COLOR: bool, CLICOLOR_FORCE: bool) Io.Cancelable!Mode {
        const force_color: ?bool = if (NO_COLOR) false else if (CLICOLOR_FORCE) true else null;
        if (force_color == false) return .no_color;

        if (file.enableAnsiEscapeCodes(io)) |_| {
            return .escape_codes;
        } else |err| switch (err) {
            error.Canceled => |e| return e,
            error.NotTerminalDevice, error.Unexpected => {},
        }

        if (is_windows and try file.isTty(io)) {
            var get_console_info = std.os.windows.CONSOLE.USER_IO.GET_SCREEN_BUFFER_INFO;
            switch (try get_console_info.operate(io, file)) {
                .SUCCESS => return .{ .windows_api = .{
                    .io = io,
                    .file = file,
                    .reset_attributes = get_console_info.Data.wAttributes,
                } },
                else => {},
            }
        }
        return if (force_color == true) .escape_codes else .no_color;
    }
};

/// Terminal style state, including foreground and background color and font attributes.
pub const Style = struct {
    color: Style.Color,
    background_color: Style.Color,
    weight: Weight,
    italic: bool,
    underline: bool,
    strikethrough: bool,

    /// Backing int value is the ANSI foreground color code for the given color.
    pub const Color = enum(u8) {
        default = 39,
        black = 30,
        red = 31,
        green = 32,
        yellow = 33,
        blue = 34,
        magenta = 35,
        cyan = 36,
        white = 37,
        bright_black = 90,
        bright_red = 91,
        bright_green = 92,
        bright_yellow = 93,
        bright_blue = 94,
        bright_magenta = 95,
        bright_cyan = 96,
        bright_white = 97,
    };

    /// Backing int value is the ANSI code to switch to the given font weight.
    pub const Weight = enum(u8) { default = 22, bold = 1, dim = 2 };

    /// The terminal's initial style, before any `Style.update`/`Terminal.updateStyle` calls.
    pub const reset: Style = .{
        .color = .default,
        .background_color = .default,
        .weight = .default,
        .italic = false,
        .underline = false,
        .strikethrough = false,
    };

    pub fn eql(a: *const Style, b: *const Style) bool {
        return a.color == b.color and
            a.background_color == b.background_color and
            a.weight == b.weight and
            a.italic == b.italic and
            a.underline == b.underline and
            a.strikethrough == b.strikethrough;
    }

    /// Any field may be `null` to indicate that the existing state should not be modified.
    pub const UpdateOptions = struct {
        pub const reset: UpdateOptions = .{
            .color = .default,
            .background_color = .default,
            .weight = .default,
            .italic = false,
            .underline = false,
            .strikethrough = false,
        };

        color: ?Style.Color = null,
        background_color: ?Style.Color = null,
        weight: ?Weight = null,
        italic: ?bool = null,
        underline: ?bool = null,
        strikethrough: ?bool = null,
    };

    /// Like `Terminal.updateStyle`, but tracks the current style in `s` to avoid writing redundant
    /// escape codes setting style properties which are already correctly configured.
    pub fn update(s: *Style, term: Terminal, opts: UpdateOptions) Io.Writer.Error!void {
        const new: Style = .{
            // zig fmt: off
            .color            = opts.color            orelse s.color,
            .background_color = opts.background_color orelse s.background_color,
            .weight           = opts.weight           orelse s.weight,
            .italic           = opts.italic           orelse s.italic,
            .underline        = opts.underline        orelse s.underline,
            .strikethrough    = opts.strikethrough    orelse s.strikethrough,
            // zig fmt: on
        };

        if (s.eql(&new)) return;

        if (new.eql(&.reset)) {
            try term.resetStyle();
        } else try term.updateStyle(.{
            // zig fmt: off
            .color            = if (new.color            != s.color)            new.color            else null,
            .background_color = if (new.background_color != s.background_color) new.background_color else null,
            .weight           = if (new.weight           != s.weight)           new.weight           else null,
            .italic           = if (new.italic           != s.italic)           new.italic           else null,
            .underline        = if (new.underline        != s.underline)        new.underline        else null,
            .strikethrough    = if (new.strikethrough    != s.strikethrough)    new.strikethrough    else null,
            // zig fmt: on
        });
        s.* = new;
    }
};

/// This function writes redundant escape sequences when a field of `opts` matches the terminal's
/// current style. Consider instead using `Style.update` to prevent redundant work.
pub fn updateStyle(term: Terminal, opts: Style.UpdateOptions) Io.Writer.Error!void {
    switch (term.mode) {
        .no_color => {},
        .windows_api => {}, // deprecated; only supported by the legacy `setColor` interface
        .escape_codes => {
            // We use our own buffer for this instead of writing pieces to `w` so that we can remove
            // the trailing ';' once we're done. This is easy because we have a small fixed bound on
            // the size of the escape code. Currently, each of our ANSI escape parameters is no more
            // than 3 decimal bytes, followed by the ';' separator, so 4 bytes per parameter.
            var ansi_params_buf: [@typeInfo(Style).@"struct".field_names.len * 4]u8 = undefined;
            var ansi_params: std.ArrayList(u8) = .initBuffer(&ansi_params_buf);

            if (opts.color) |color| {
                ansi_params.printAssumeCapacity("{d};", .{@backingInt(color)});
            }
            if (opts.background_color) |background_color| {
                // Background color codes are offset by 10 from foreground color codes.
                ansi_params.printAssumeCapacity("{d};", .{@backingInt(background_color) + 10});
            }
            if (opts.weight) |weight| {
                ansi_params.printAssumeCapacity("{d};", .{@backingInt(weight)});
            }
            if (opts.italic) |italic| {
                ansi_params.appendSliceAssumeCapacity(if (italic) "3;" else "23;");
            }
            if (opts.underline) |underline| {
                ansi_params.appendSliceAssumeCapacity(if (underline) "4;" else "24;");
            }
            if (opts.strikethrough) |strikethrough| {
                ansi_params.appendSliceAssumeCapacity(if (strikethrough) "9;" else "29;");
            }

            if (ansi_params.items.len == 0) {
                // All properties unchanged.
            } else {
                try term.writer.print("\x1b[{s}m", .{ansi_params.items[0 .. ansi_params.items.len - 1]});
            }
        },
    }
}

/// Equivalent to `term.updateStyle(.reset)`, but may use a more efficient escape sequence.
///
/// This function writes redundant escape sequences when the terminal's style is already in the
/// reset state. Consider instead using `Style.update(.reset)` to prevent redundant work.
pub fn resetStyle(term: Terminal) Io.Writer.Error!void {
    switch (term.mode) {
        .no_color => {},
        .windows_api => {}, // deprecated; only supported by the legacy `setColor` interface
        .escape_codes => try term.writer.writeAll("\x1b[0m"),
    }
}

/// Deprecated; use `Style.Color` instead.
///
/// To be deleted before Zig 0.19.0 alongside `setColor`.
pub const Color = enum {
    black,
    red,
    green,
    yellow,
    blue,
    magenta,
    cyan,
    white,
    bright_black,
    bright_red,
    bright_green,
    bright_yellow,
    bright_blue,
    bright_magenta,
    bright_cyan,
    bright_white,
    dim,
    bold,
    reset,
};
/// Deprecated; see doc comment on `setColor`.
///
/// To be deleted before Zig 0.19.0 alongside `setColor`.
pub const SetColorError = Io.Cancelable || Io.UnexpectedError || Io.Writer.Error;
/// Deprecated; use `Style.update` or `Terminal.updateStyle` instead.
///
/// To be deleted before Zig 0.19.0.
pub fn setColor(t: Terminal, color: Color) SetColorError!void {
    switch (t.mode) {
        .no_color => return,
        .escape_codes => {
            const color_string = switch (color) {
                .black => "\x1b[30m",
                .red => "\x1b[31m",
                .green => "\x1b[32m",
                .yellow => "\x1b[33m",
                .blue => "\x1b[34m",
                .magenta => "\x1b[35m",
                .cyan => "\x1b[36m",
                .white => "\x1b[37m",
                .bright_black => "\x1b[90m",
                .bright_red => "\x1b[91m",
                .bright_green => "\x1b[92m",
                .bright_yellow => "\x1b[93m",
                .bright_blue => "\x1b[94m",
                .bright_magenta => "\x1b[95m",
                .bright_cyan => "\x1b[96m",
                .bright_white => "\x1b[97m",
                .bold => "\x1b[1m",
                .dim => "\x1b[2m",
                .reset => "\x1b[0m",
            };
            try t.writer.writeAll(color_string);
        },
        .windows_api => |wa| {
            const windows = std.os.windows;
            const attributes: windows.WORD = switch (color) {
                .black => 0,
                .red => windows.FOREGROUND_RED,
                .green => windows.FOREGROUND_GREEN,
                .yellow => windows.FOREGROUND_RED | windows.FOREGROUND_GREEN,
                .blue => windows.FOREGROUND_BLUE,
                .magenta => windows.FOREGROUND_RED | windows.FOREGROUND_BLUE,
                .cyan => windows.FOREGROUND_GREEN | windows.FOREGROUND_BLUE,
                .white => windows.FOREGROUND_RED | windows.FOREGROUND_GREEN | windows.FOREGROUND_BLUE,
                .bright_black => windows.FOREGROUND_INTENSITY,
                .bright_red => windows.FOREGROUND_RED | windows.FOREGROUND_INTENSITY,
                .bright_green => windows.FOREGROUND_GREEN | windows.FOREGROUND_INTENSITY,
                .bright_yellow => windows.FOREGROUND_RED | windows.FOREGROUND_GREEN | windows.FOREGROUND_INTENSITY,
                .bright_blue => windows.FOREGROUND_BLUE | windows.FOREGROUND_INTENSITY,
                .bright_magenta => windows.FOREGROUND_RED | windows.FOREGROUND_BLUE | windows.FOREGROUND_INTENSITY,
                .bright_cyan => windows.FOREGROUND_GREEN | windows.FOREGROUND_BLUE | windows.FOREGROUND_INTENSITY,
                .bright_white, .bold => windows.FOREGROUND_RED | windows.FOREGROUND_GREEN | windows.FOREGROUND_BLUE | windows.FOREGROUND_INTENSITY,
                // "dim" is not supported using basic character attributes, but let's still make it do *something*.
                // This matches the old behavior of TTY.Color before the bright variants were added.
                .dim => windows.FOREGROUND_INTENSITY,
                .reset => wa.reset_attributes,
            };
            try t.writer.flush();
            var set_text_attribute = windows.CONSOLE.USER_IO.SET_TEXT_ATTRIBUTE(attributes);
            switch (try set_text_attribute.operate(wa.io, wa.file)) {
                .SUCCESS => {},
                else => |status| return windows.unexpectedStatus(status),
            }
        },
    }
}
