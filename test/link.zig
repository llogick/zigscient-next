pub fn addCases(ctx: *LinkContext) void {
    switch (ctx.target.result.os.tag) {
        .windows => @import("link/windows.zig").addCases(ctx),
        .linux => @import("link/linux.zig").addCases(ctx),
        else => unreachable, // nothing else in the matrix yet
    }
}

const LinkContext = @import("tests.zig").LinkContext;
const std = @import("std");
