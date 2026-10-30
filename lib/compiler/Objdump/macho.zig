pub fn identify(r: *Io.Reader) error{ReadFailed}!bool {
    const magic = r.peekInt(u32, .little) catch |err| switch (err) {
        error.EndOfStream => return false,
        else => |e| return e,
    };
    return switch (magic) {
        std.macho.MH_MAGIC,
        std.macho.MH_MAGIC_64,
        std.macho.FAT_MAGIC,
        std.macho.FAT_MAGIC_64,
        => true, // little-endian

        std.macho.MH_CIGAM,
        std.macho.MH_CIGAM_64,
        std.macho.FAT_CIGAM,
        std.macho.FAT_CIGAM_64,
        => true, // big-endian

        else => false,
    };
}

pub fn dump(d: *Objdump) Objdump.Error!void {
    return d.fail("TODO: dump Mach-O object", .{});
}

const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;

const Objdump = @import("../Objdump.zig");
