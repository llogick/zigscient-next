const Foo = union(enum) { a: u32, b: u32 };
export fn entry() void {
    const foo: Foo = .{ .a = 123 };
    switch (foo) {
        else => |payload, tag| {
            _ = payload;
            _ = tag;
        },
    }
}

// error
//
// :5:27: error: cannot capture tag of 'else' prong
// :5:18: note: first capture in 'else' prong always captures 'switch' operand
