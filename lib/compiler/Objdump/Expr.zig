steps: []Step,
/// Populated iff the `Expr` is resolved.
max_stack_len: ?usize,

pub const Step = union(enum) {
    int_lit: u64,
    int_add,
    int_sub,
    int_mul,
    int_div,
    int_mod,

    bool_lit: bool,
    bool_and,
    bool_or,
    bool_not,

    string_lit: []const u8,

    cmp_eq,
    cmp_neq,

    cmp_string_contains,

    cmp_lte,
    cmp_gte,
    cmp_lt,
    cmp_gt,

    property_unresolved: []const u8,
    property_resolved: Objdump.Table.PropertyIndex,
};

pub fn parse(
    arena: Allocator,
    src: []const u8,
    /// If `error.ParseFail` is returned, this is populated with a slice allocated into `arena`.
    err_msg_out: *[]const u8,
) (Allocator.Error || error{ParseFail})!Expr {
    var offset: usize = 0;
    var steps: std.ArrayList(Step) = .empty;
    // No need for a `defer steps.deinit(...)` because it's arena-allocated.
    try parseAppendExpr(arena, src, &offset, &steps, 0, err_msg_out);
    offset = std.mem.findNonePos(u8, src, offset, whitespace_bytes) orelse src.len;
    if (offset < src.len) {
        err_msg_out.* = try std.fmt.allocPrint(
            arena,
            "column {d}: expected eof, found '{c}'",
            .{ offset + 1, src[offset] },
        );
        return error.ParseFail;
    }
    // No need for a `toOwnedSlice` because it's arena-allocated.
    return .{
        .steps = steps.items,
        .max_stack_len = null,
    };
}

pub fn clone(expr: *const Expr, arena: Allocator) Allocator.Error!Expr {
    return .{
        .steps = try arena.dupe(Step, expr.steps),
        .max_stack_len = expr.max_stack_len,
    };
}

/// Replaces all `.property_unresolved` steps in `expr` with `.property_resolved` steps, and
/// performs type checking on the expression.
///
/// If any property name is unknown, or the expression fails type checking, `error.ResolveFail` is
/// returned and `err_msg_out.*` is populated with an error message allocated into `arena`.
pub fn resolve(
    expr: *Expr,
    arena: Allocator,
    prop_names: []const []const u8,
    prop_types: []const Objdump.Table.PropertyType,
    /// If `null`, the result of the expression may be any type. Otherwise, an error is returned if
    /// the result of the expression does not have this type.
    maybe_want_type: ?Objdump.Table.PropertyType,
    err_msg_out: *[]const u8,
) (Allocator.Error || error{ResolveFail})!void {
    assert(expr.max_stack_len == null); // input is unresolved
    const FormatAvailableProps = struct {
        names: []const []const u8,
        types: []const Objdump.Table.PropertyType,
        pub fn format(props: @This(), w: *std.Io.Writer) std.Io.Writer.Error!void {
            for (props.names, props.types) |name, @"type"| {
                try w.print("\n  {q} ({t})", .{ name, @"type" });
            }
        }
    };
    var type_stack: std.ArrayList(Objdump.Table.PropertyType) = .empty;
    var max_stack_len: u32 = 0;
    // No need for a `defer type_stack.deinit(...)` because it's arena-allocated.
    for (expr.steps) |*s| {
        switch (s.*) {
            .property_resolved => unreachable, // input is unresolved
            .property_unresolved => |target_prop_name| {
                for (prop_names, prop_types, 0..) |prop_name, prop_type, prop_index| {
                    if (std.mem.eql(u8, prop_name, target_prop_name)) {
                        s.* = .{ .property_resolved = @fromBackingInt(@intCast(prop_index)) };
                        try type_stack.append(arena, prop_type);
                        break;
                    }
                } else {
                    err_msg_out.* = try std.fmt.allocPrint(
                        arena,
                        "unknown property {q}\navailable properties:{f}",
                        .{
                            target_prop_name,
                            @as(FormatAvailableProps, .{ .names = prop_names, .types = prop_types }),
                        },
                    );
                    return error.ResolveFail;
                }
            },

            .int_lit => try type_stack.append(arena, .int),

            .int_add,
            .int_sub,
            .int_mul,
            .int_div,
            .int_mod,
            => {
                const rhs = type_stack.pop().?;
                const lhs = type_stack.pop().?;
                if (lhs != .int) {
                    err_msg_out.* = try std.fmt.allocPrint(
                        arena,
                        "lhs type mismatch: expected 'int', got '{t}'",
                        .{lhs},
                    );
                    return error.ResolveFail;
                }
                if (rhs != .int) {
                    err_msg_out.* = try std.fmt.allocPrint(
                        arena,
                        "rhs type mismatch: expected 'int', got '{t}'",
                        .{rhs},
                    );
                    return error.ResolveFail;
                }
                type_stack.appendAssumeCapacity(.int);
            },

            .bool_lit => try type_stack.append(arena, .bool),

            .bool_and,
            .bool_or,
            => {
                const rhs = type_stack.pop().?;
                const lhs = type_stack.pop().?;
                if (lhs != .bool) {
                    err_msg_out.* = try std.fmt.allocPrint(
                        arena,
                        "invalid lhs: expected 'bool', got '{t}'",
                        .{lhs},
                    );
                    return error.ResolveFail;
                }
                if (rhs != .bool) {
                    err_msg_out.* = try std.fmt.allocPrint(
                        arena,
                        "invalid rhs: expected 'bool', got '{t}'",
                        .{rhs},
                    );
                    return error.ResolveFail;
                }
                type_stack.appendAssumeCapacity(.bool);
            },

            .bool_not => {
                const operand = type_stack.pop().?;
                if (operand != .bool) {
                    err_msg_out.* = try std.fmt.allocPrint(
                        arena,
                        "invalid operand: expected 'bool', got '{t}'",
                        .{operand},
                    );
                    return error.ResolveFail;
                }
                type_stack.appendAssumeCapacity(.bool);
            },

            .string_lit => try type_stack.append(arena, .string),

            .cmp_eq, .cmp_neq => {
                const rhs = type_stack.pop().?;
                const lhs = type_stack.pop().?;
                if (lhs != rhs) {
                    err_msg_out.* = try std.fmt.allocPrint(
                        arena,
                        "type mismatch: lhs is '{t}', rhs is '{t}'",
                        .{ lhs, rhs },
                    );
                    return error.ResolveFail;
                }
                type_stack.appendAssumeCapacity(.bool);
            },

            .cmp_string_contains => {
                const rhs = type_stack.pop().?;
                const lhs = type_stack.pop().?;
                if (lhs != .string) {
                    err_msg_out.* = try std.fmt.allocPrint(
                        arena,
                        "invalid lhs: expected 'string', got '{t}'",
                        .{lhs},
                    );
                    return error.ResolveFail;
                }
                if (rhs != .string) {
                    err_msg_out.* = try std.fmt.allocPrint(
                        arena,
                        "invalid rhs: expected 'string', got '{t}'",
                        .{rhs},
                    );
                    return error.ResolveFail;
                }
                type_stack.appendAssumeCapacity(.bool);
            },

            .cmp_lte,
            .cmp_gte,
            .cmp_lt,
            .cmp_gt,
            => {
                const rhs = type_stack.pop().?;
                const lhs = type_stack.pop().?;
                if (lhs != .int) {
                    err_msg_out.* = try std.fmt.allocPrint(
                        arena,
                        "invalid lhs: expected 'int', got '{t}'",
                        .{lhs},
                    );
                    return error.ResolveFail;
                }
                if (rhs != .int) {
                    err_msg_out.* = try std.fmt.allocPrint(
                        arena,
                        "invalid rhs: expected 'int', got '{t}'",
                        .{rhs},
                    );
                    return error.ResolveFail;
                }
                type_stack.appendAssumeCapacity(.bool);
            },
        }
        max_stack_len = @intCast(@max(max_stack_len, type_stack.items.len));
    }
    assert(type_stack.items.len == 1);
    if (maybe_want_type) |want_type| {
        if (type_stack.items[0] != want_type) {
            err_msg_out.* = try std.fmt.allocPrint(
                arena,
                "expected '{t}', got '{t}'",
                .{ want_type, type_stack.items[0] },
            );
            return error.ResolveFail;
        }
    }
    expr.max_stack_len = max_stack_len;
}

pub const Value = union(Objdump.Table.PropertyType) {
    bool: bool,
    int: u64,
    string: []const u8,
};

pub fn eval(
    expr: *const Expr,
    stack_buf: []Value,
    getProperty: *const fn (*const anyopaque, u64, Objdump.Table.PropertyIndex) Value,
    get_property_context: *const anyopaque,
    get_property_entry_index: u64,
) Value {
    {
        const max_stack_len = expr.max_stack_len.?; // input is resolved
        assert(stack_buf.len >= max_stack_len);
    }
    var stack: std.ArrayList(Value) = .initBuffer(stack_buf);
    for (expr.steps) |s| switch (s) {
        .property_unresolved => unreachable, // input is resolved

        .int_lit => |x| stack.appendAssumeCapacity(.{ .int = x }),
        .bool_lit => |x| stack.appendAssumeCapacity(.{ .bool = x }),
        .string_lit => |x| stack.appendAssumeCapacity(.{ .string = x }),

        .property_resolved => |property_index| {
            stack.appendAssumeCapacity(getProperty(
                get_property_context,
                get_property_entry_index,
                property_index,
            ));
        },

        .int_add => {
            const rhs = stack.pop().?.int;
            const lhs = stack.pop().?.int;
            stack.appendAssumeCapacity(.{ .int = lhs +% rhs });
        },
        .int_sub => {
            const rhs = stack.pop().?.int;
            const lhs = stack.pop().?.int;
            stack.appendAssumeCapacity(.{ .int = lhs -% rhs });
        },
        .int_mul => {
            const rhs = stack.pop().?.int;
            const lhs = stack.pop().?.int;
            stack.appendAssumeCapacity(.{ .int = lhs *% rhs });
        },
        .int_div => {
            const rhs = stack.pop().?.int;
            const lhs = stack.pop().?.int;
            const result: u64 = switch (rhs) {
                0 => 0,
                else => lhs / rhs,
            };
            stack.appendAssumeCapacity(.{ .int = result });
        },
        .int_mod => {
            const rhs = stack.pop().?.int;
            const lhs = stack.pop().?.int;
            const result: u64 = switch (rhs) {
                0 => 0,
                else => lhs % rhs,
            };
            stack.appendAssumeCapacity(.{ .int = result });
        },

        .bool_and => {
            const rhs = stack.pop().?.bool;
            const lhs = stack.pop().?.bool;
            stack.appendAssumeCapacity(.{ .bool = lhs & rhs });
        },
        .bool_or => {
            const rhs = stack.pop().?.bool;
            const lhs = stack.pop().?.bool;
            stack.appendAssumeCapacity(.{ .bool = lhs | rhs });
        },
        .bool_not => {
            const operand = stack.pop().?.bool;
            stack.appendAssumeCapacity(.{ .bool = !operand });
        },

        .cmp_neq => {
            const rhs = stack.pop().?;
            const lhs = stack.pop().?;
            const result: bool = switch (lhs) {
                .bool => |lhs_bool| lhs_bool != rhs.bool,
                .int => |lhs_int| lhs_int != rhs.int,
                .string => |lhs_string| !std.mem.eql(u8, lhs_string, rhs.string),
            };
            stack.appendAssumeCapacity(.{ .bool = result });
        },
        .cmp_eq => {
            const rhs = stack.pop().?;
            const lhs = stack.pop().?;
            const result: bool = switch (lhs) {
                .bool => |lhs_bool| lhs_bool == rhs.bool,
                .int => |lhs_int| lhs_int == rhs.int,
                .string => |lhs_string| std.mem.eql(u8, lhs_string, rhs.string),
            };
            stack.appendAssumeCapacity(.{ .bool = result });
        },

        .cmp_string_contains => {
            const rhs = stack.pop().?.string;
            const lhs = stack.pop().?.string;
            stack.appendAssumeCapacity(.{ .bool = std.mem.find(u8, lhs, rhs) != null });
        },

        .cmp_lte => {
            const rhs = stack.pop().?.int;
            const lhs = stack.pop().?.int;
            stack.appendAssumeCapacity(.{ .bool = lhs <= rhs });
        },
        .cmp_gte => {
            const rhs = stack.pop().?.int;
            const lhs = stack.pop().?.int;
            stack.appendAssumeCapacity(.{ .bool = lhs >= rhs });
        },
        .cmp_lt => {
            const rhs = stack.pop().?.int;
            const lhs = stack.pop().?.int;
            stack.appendAssumeCapacity(.{ .bool = lhs < rhs });
        },
        .cmp_gt => {
            const rhs = stack.pop().?.int;
            const lhs = stack.pop().?.int;
            stack.appendAssumeCapacity(.{ .bool = lhs > rhs });
        },
    };
    assert(stack.items.len == 1);
    return stack.items[0];
}

const ident_bytes: []const u8 = std.ascii.letters ++ "_.";
const whitespace_bytes: []const u8 = &std.ascii.whitespace;

fn parseAppendExpr(
    arena: Allocator,
    src: []const u8,
    offset: *usize,
    steps: *std.ArrayList(Step),
    min_bind_power: u8,
    err_msg_out: *[]const u8,
) (Allocator.Error || error{ParseFail})!void {
    try parseAppendPrefixedAtom(arena, src, offset, steps, err_msg_out);
    while (peekInfix(src, offset)) |op| {
        if (op.bind_power[0] < min_bind_power) {
            break;
        }
        offset.* = op.next_offset; // consume the operator
        try parseAppendExpr(arena, src, offset, steps, op.bind_power[1], err_msg_out);
        try steps.append(arena, op.step);
    }
}

fn parseAppendPrefixedAtom(
    arena: Allocator,
    src: []const u8,
    offset: *usize,
    steps: *std.ArrayList(Step),
    err_msg_out: *[]const u8,
) (Allocator.Error || error{ParseFail})!void {
    offset.* = std.mem.findNonePos(u8, src, offset.*, whitespace_bytes) orelse src.len;

    if (std.mem.startsWith(u8, src[offset.*..], "!") and
        !std.mem.startsWith(u8, src[offset.*..], "!="))
    {
        offset.* += 1;
        try parseAppendPrefixedAtom(arena, src, offset, steps, err_msg_out);
        try steps.append(arena, .bool_not);
        return;
    }

    return parseAppendAtom(arena, src, offset, steps, err_msg_out);
}

fn parseAppendAtom(
    arena: Allocator,
    src: []const u8,
    offset: *usize,
    steps: *std.ArrayList(Step),
    err_msg_out: *[]const u8,
) (Allocator.Error || error{ParseFail})!void {
    const start = std.mem.findNonePos(u8, src, offset.*, whitespace_bytes) orelse src.len;

    if (std.mem.startsWith(u8, src[start..], "(")) {
        offset.* = start + 1;
        try parseAppendExpr(arena, src, offset, steps, 0, err_msg_out);
        const rparen_off = std.mem.findNonePos(u8, src, offset.*, whitespace_bytes) orelse src.len;
        if (rparen_off == src.len) {
            err_msg_out.* = try std.fmt.allocPrint(
                arena,
                "column {d}: expected ')', found eof",
                .{rparen_off + 1},
            );
            return error.ParseFail;
        } else if (src[rparen_off] != ')') {
            err_msg_out.* = try std.fmt.allocPrint(
                arena,
                "column {d}: expected ')', found '{c}'",
                .{ rparen_off + 1, src[rparen_off] },
            );
            return error.ParseFail;
        }
        offset.* = rparen_off + 1;
        return;
    }

    ident: {
        const end = std.mem.findNonePos(u8, src, start, ident_bytes) orelse src.len;
        if (end == start) break :ident;
        const ident = src[start..end];
        if (std.mem.eql(u8, ident, "and") or
            std.mem.eql(u8, ident, "or"))
        {
            err_msg_out.* = try std.fmt.allocPrint(
                arena,
                "column {d}: expected identifier, literal, or '(', found '{s}'",
                .{ start + 1, ident },
            );
            return error.ParseFail;
        }
        offset.* = end;
        if (std.mem.eql(u8, ident, "true")) {
            try steps.append(arena, .{ .bool_lit = true });
        } else if (std.mem.eql(u8, ident, "false")) {
            try steps.append(arena, .{ .bool_lit = false });
        } else {
            try steps.append(arena, .{ .property_unresolved = ident });
        }
        return;
    }

    int_lit: {
        if (start == src.len) break :int_lit;
        if (src[start] < '0') break :int_lit;
        if (src[start] > '9') break :int_lit;
        const end = std.mem.findNonePos(u8, src, start, "0123456789_" ++ std.ascii.letters) orelse src.len;
        if (end == start) break :int_lit;
        const int_val = std.fmt.parseInt(u64, src[start..end], 0) catch |err| switch (err) {
            error.InvalidCharacter => break :int_lit,
            error.Overflow => {
                err_msg_out.* = try std.fmt.allocPrint(
                    arena,
                    "column {d}: int literal {q} does not fit in a u64",
                    .{ start + 1, src[start..end] },
                );
                return error.ParseFail;
            },
        };
        offset.* = end;
        try steps.append(arena, .{ .int_lit = int_val });
        return;
    }

    string_lit: {
        if (!std.mem.startsWith(u8, src[start..], "\"")) break :string_lit;
        var end: usize = start + 1;
        while (true) {
            if (end == src.len) {
                err_msg_out.* = try std.fmt.allocPrint(
                    arena,
                    "column {d}: expected '\"', found eof",
                    .{start + 1},
                );
                return error.ParseFail;
            }
            switch (src[end]) {
                '"' => {
                    end += 1; // include ending quote
                    break;
                },
                '\\' => switch (std.zig.string_literal.parseEscapeSequence(src, &end)) {
                    .success => {}, // `end` was updated for us
                    .failure => |parse_err| {
                        err_msg_out.* = try std.fmt.allocPrint(
                            arena,
                            "column {d}: {f}",
                            .{ start + 1, parse_err.fmt(src) },
                        );
                        return error.ParseFail;
                    },
                },
                else => end += 1,
            }
        }
        const slice = std.zig.string_literal.parseAlloc(arena, src[start..end]) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            error.InvalidLiteral => unreachable, // we already validated it
        };
        offset.* = end;
        try steps.append(arena, .{ .string_lit = slice });
        return;
    }

    if (start == src.len) {
        err_msg_out.* = try std.fmt.allocPrint(
            arena,
            "column {d}: expected identifier, literal, or '(', found eof",
            .{start + 1},
        );
        return error.ParseFail;
    } else {
        err_msg_out.* = try std.fmt.allocPrint(
            arena,
            "column {d}: expected identifier, literal, or '(', found '{c}'",
            .{ start + 1, src[start] },
        );
        return error.ParseFail;
    }
}

fn peekInfix(src: []const u8, offset: *usize) ?struct {
    step: Step,
    next_offset: usize,
    bind_power: [2]u8,
} {
    const start = std.mem.findNonePos(u8, src, offset.*, whitespace_bytes) orelse src.len;
    offset.* = start;

    // identifier-like operators
    {
        const ident_end = std.mem.findNonePos(u8, src, start, ident_bytes) orelse src.len;
        const ident = src[start..ident_end];
        if (std.mem.eql(u8, ident, "and")) return .{
            .step = .bool_and,
            .next_offset = start + "and".len,
            .bind_power = .{ 1, 2 },
        };
        if (std.mem.eql(u8, ident, "or")) return .{
            .step = .bool_or,
            .next_offset = start + "or".len,
            .bind_power = .{ 1, 2 },
        };
    }

    // multi-character operators
    if (std.mem.startsWith(u8, src[start..], "==")) return .{
        .step = .cmp_eq,
        .next_offset = start + "==".len,
        .bind_power = .{ 3, 4 },
    };
    if (std.mem.startsWith(u8, src[start..], "!=")) return .{
        .step = .cmp_neq,
        .next_offset = start + "!=".len,
        .bind_power = .{ 3, 4 },
    };
    if (std.mem.startsWith(u8, src[start..], "<=")) return .{
        .step = .cmp_lte,
        .next_offset = start + "<=".len,
        .bind_power = .{ 3, 4 },
    };
    if (std.mem.startsWith(u8, src[start..], ">=")) return .{
        .step = .cmp_gte,
        .next_offset = start + ">=".len,
        .bind_power = .{ 3, 4 },
    };

    // single-character operators
    if (start < src.len) switch (src[start]) {
        '<' => return .{
            .step = .cmp_lt,
            .next_offset = start + 1,
            .bind_power = .{ 3, 4 },
        },

        '>' => return .{
            .step = .cmp_gt,
            .next_offset = start + 1,
            .bind_power = .{ 3, 4 },
        },

        '~' => return .{
            .step = .cmp_string_contains,
            .next_offset = start + 1,
            .bind_power = .{ 3, 4 },
        },

        '+' => return .{
            .step = .int_add,
            .next_offset = start + 1,
            .bind_power = .{ 5, 6 },
        },

        '-' => return .{
            .step = .int_sub,
            .next_offset = start + 1,
            .bind_power = .{ 5, 6 },
        },

        '*' => return .{
            .step = .int_mul,
            .next_offset = start + 1,
            .bind_power = .{ 7, 8 },
        },

        '/' => return .{
            .step = .int_div,
            .next_offset = start + 1,
            .bind_power = .{ 7, 8 },
        },

        '%' => return .{
            .step = .int_mod,
            .next_offset = start + 1,
            .bind_power = .{ 7, 8 },
        },

        else => {},
    };

    return null;
}

test parse {
    const gpa = std.testing.allocator;

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var err_msg: []const u8 = undefined;
    const expr = Expr.parse(
        arena,
        \\(1 * 2 + 3 / 4 % 5) * foo == 20 and (true or false) and bar ~ "hello\x00world\n"
    ,
        &err_msg,
    ) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.ParseFail => {
            std.log.err("failed to parse: {s}", .{err_msg});
            return error.TestUnexpectedError;
        },
    };

    try std.testing.expectEqualDeep(@as([]const Expr.Step, &.{
        .{ .int_lit = 1 },
        .{ .int_lit = 2 },
        .int_mul,
        .{ .int_lit = 3 },
        .{ .int_lit = 4 },
        .int_div,
        .{ .int_lit = 5 },
        .int_mod,
        .int_add,
        .{ .property_unresolved = "foo" },
        .int_mul,
        .{ .int_lit = 20 },
        .cmp_eq,
        .{ .bool_lit = true },
        .{ .bool_lit = false },
        .bool_or,
        .bool_and,
        .{ .property_unresolved = "bar" },
        .{ .string_lit = "hello\x00world\n" },
        .cmp_string_contains,
        .bool_and,
    }), expr.steps);
}

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const Objdump = @import("../Objdump.zig");

const Expr = @This();
