const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    try io.sleep(.fromSeconds(5), .awake);
}
