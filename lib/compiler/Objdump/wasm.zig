pub fn identify(r: *Io.Reader) error{ReadFailed}!bool {
    const magic = r.peek(4) catch |err| switch (err) {
        error.EndOfStream => return false,
        else => |e| return e,
    };
    return std.mem.eql(u8, magic, "\x00asm");
}

pub fn dump(d: *Objdump) Objdump.Error!void {
    return d.fail("TODO: dump WASM module", .{});
}

const std = @import("std");
const assert = std.debug.assert;
const Io = std.Io;

const Objdump = @import("../Objdump.zig");
