pub const completions = @import("completions.zig");
pub const InternPool = @import("InternPool.zig");
pub const StringPool = @import("string_pool.zig").StringPool;
pub const degibberish = @import("degibberish.zig");

comptime {
    const std = @import("std");
    std.testing.refAllDeclsRecursive(@This());
}
