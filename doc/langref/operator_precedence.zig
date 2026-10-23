const std = @import("std");

test "operator precedence" {
    // Arithmetic operators follow the conventional order of operations.
    {
        const a = 1 + 2 * 3 - 4 / 2;
        const b = 1 + (2 * 3) - (4 / 2);
        try std.testing.expectEqual(a, b);
    }

    // Remainder division is a special case. It still has a higher precedence than
    // addition and subtraction, but it has ambiguous precedence with itself and
    // other multiplication/division operators.
    {
        const a = 1 + 2 % 3;
        const b = 1 + (2 % 3);
        try std.testing.expectEqual(a, b);

        // `1 % 2 % 3`, `1 % 2 / 3`, etc. would be a compile error,
        // and parentheses would be required to disambiguate.
    }

    // Bitwise operators have ambiguous precedence with other bitwise operators.
    // However, non-shifting bitwise operators (&, |, ^) can be chained with themselves.
    {
        const a = 1 ^ 2 ^ 3;
        const b = (1 ^ 2) ^ 3;
        try std.testing.expectEqual(a, b);

        // `1 ^ 2 | 3`, `1 << 2 << 3`, etc. would be a compile error,
        // and parentheses would be required to disambiguate.

        // To mix bitwise operators, parentheses must be used.
        try std.testing.expectEqual(0x10101, (1 << 16) | (1 << 8) | 1);
    }

    // Comparison operators have lower precedence than both arithmetic and bitwise
    // operators. Comparisons cannot be chained.
    {
        const a = 1 | 4 <= 1 + 4;
        const b = (1 | 4) <= (1 + 4);
        try std.testing.expectEqual(a, b);

        // `1 == 2 >= 3`, etc. would be a compile error,
        // and parentheses would be required to disambiguate.
    }

    // Logical operators have lower precedence than comparison operators,
    // and may only be chained homogeneously (`and` with `and`, but not
    // with `or`, and vice versa).
    {
        const false_expr: bool = 7 < 2;
        const true_expr: bool = 1 + 1 == 2;

        const a = false_expr or false_expr or true_expr;
        const b = 7 < 2 or 7 < 2 or 1 + 1 == 2;
        const c = ((7 < 2) or (7 < 2)) or (1 + 1 == 2);
        try std.testing.expect(a and b and c);

        // `a or b and c`, etc. would be a compile error,
        // and parentheses would be required to disambiguate.
    }

    // `orelse` and `catch` may be chained with themselves, but not each other.
    {
        const some_null: ?u8 = null;

        const a = some_null orelse some_null orelse 2;
        const b = (some_null orelse some_null) orelse 2;
        try std.testing.expectEqual(a, b);

        // `a catch b orelse c`, etc. would be a compile error,
        // and parentheses would be required to disambiguate.

        // Prefix operators like `try` have higher precedence than `orelse`/`catch`.
        const c = try maybe() orelse 1;
        const d = (try maybe()) orelse 1;
        try std.testing.expectEqual(c, d);
    }
}

fn maybe() error{Fail}!?u8 {
    return null;
}

// test
