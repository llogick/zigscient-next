const E = enum {};
const U = union {
    foo: struct { a: u8, b: noreturn },
    bar: E,
};
export fn loadSliceUninstantiableEnum() void {
    const bytes: [32]u8 = @splat(0);
    const ptr: *const [1]E = @ptrCast(&bytes);
    const slice: []const E = ptr;
    _ = comptime slice[0];
    comptime unreachable;
}
export fn iterateSliceUninstantiableEnum() void {
    const bytes: [32]u8 = @splat(0);
    const ptr: *const [1]E = @ptrCast(&bytes);
    const slice: []const E = ptr;
    comptime for (slice) |elem| {
        _ = elem;
    };
    comptime unreachable;
}
export fn loadSliceUninstantiableUnion() void {
    const bytes: [32]u8 = @splat(0);
    const ptr: *const [1]U = @ptrCast(&bytes);
    const slice: []const U = ptr;
    _ = comptime slice[0];
    comptime unreachable;
}
export fn iterateSliceUninstantiableUnion() void {
    const bytes: [32]u8 = @splat(0);
    const ptr: *const [1]U = @ptrCast(&bytes);
    const slice: []const U = ptr;
    comptime for (slice) |elem| {
        _ = elem;
    };
    comptime unreachable;
}

// error
//
// :10:23: error: cannot load uninstantiable type 'tmp.E'
// :1:11: note: enum declared here
// :17:19: error: cannot load uninstantiable type 'tmp.E'
// :1:11: note: enum declared here
// :26:23: error: cannot load uninstantiable type 'tmp.U'
// :2:11: note: union declared here
// :33:19: error: cannot load uninstantiable type 'tmp.U'
// :2:11: note: union declared here
