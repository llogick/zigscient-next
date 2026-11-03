pub const Pattern = union(enum) {
    any: []const AsciiCharRange,
    none: []const AsciiCharRange,
    literal: []const u8,
    @"?": []const Pattern,
    @"*": []const Pattern,
    @"+": []const Pattern,
    @"|": []const []const Pattern,
    capture: []const Pattern,
    reference: usize,

    pub const AsciiCharRange = struct { u7, u7 };
    pub const AsciiCharSet = std.bit_set.Static(0x80);
    pub fn parseAsciiCharRanges(acrs: []const AsciiCharRange) AsciiCharSet {
        var acs: AsciiCharSet = .empty;
        for (acrs) |acr| acs.setRangeValue(.{ .start = acr[0], .end = @as(u8, acr[1]) + 1 }, true);
        return acs;
    }

    pub const Compiled = struct {
        nodes: std.MultiArrayList(Node),
        stack: std.ArrayList(usize),
        captures: std.ArrayList(Node.Capture),

        pub const empty: Compiled = .{
            .nodes = .empty,
            .stack = .empty,
            .captures = .empty,
        };

        const Node = union(enum) {
            any: AsciiCharSet,
            none: AsciiCharSet,
            literal: []const u8,
            choice: usize,
            goto: usize,
            capture_start: usize,
            capture_end: usize,
            reference: Capture,

            const Capture = struct { start_ni: usize, end_ni: usize };
        };

        pub fn init(gpa: Allocator, patterns: []const Pattern) Allocator.Error!Compiled {
            var c: Compiled = .empty;
            try c.append(gpa, patterns);
            return c;
        }

        pub fn deinit(c: *Compiled, gpa: Allocator) void {
            c.nodes.deinit(gpa);
            c.stack.deinit(gpa);
            c.captures.deinit(gpa);
            c.* = undefined;
        }

        pub fn append(c: *Compiled, gpa: Allocator, patterns: []const Pattern) Allocator.Error!void {
            std.debug.assert(c.stack.items.len == 0);
            defer std.debug.assert(c.stack.items.len == 0);
            try c.appendInner(gpa, patterns);
        }

        pub const FindResult = union(enum) {
            matched: Range,
            expected: Expected,

            pub const Range = struct { start: usize, end: usize };

            pub const Expected = struct {
                range: Range,
                what: What,

                pub const What = MatchResult.Expected.What;
            };
        };
        pub fn find(c: *const Compiled, gpa: Allocator, text: []const u8) Allocator.Error!FindResult {
            var arena: std.heap.ArenaAllocator = .init(gpa);
            defer arena.deinit();
            var matcher: Pattern.Matcher = .{
                .gpa = gpa,
                .arena = arena.allocator(),
                .nodes = &c.nodes,
                .text = undefined,
                .states = .empty,
                .best = undefined,
            };
            defer matcher.deinit();
            var best: struct { ni: usize, expected: FindResult.Expected } = .{
                .ni = 0,
                .expected = .{
                    .range = .{ .start = 0, .end = 0 },
                    .what = .{ .any = .empty },
                },
            };
            for (0..text.len) |start| {
                _ = arena.reset(.retain_capacity);
                matcher.text = text[start..];
                matcher.states.clearRetainingCapacity();
                matcher.best = .{
                    .ni = 0,
                    .expected = .{
                        .index = 0,
                        .what = .{ .any = .empty },
                    },
                };
                switch (try matcher.run(.start)) {
                    .matched => |len| return .{ .matched = .{
                        .start = start,
                        .end = start + len,
                    } },
                    .expected => |expected| if ((std.math.order(
                        matcher.best.ni,
                        best.ni,
                    ).differ() orelse std.math.order(
                        expected.index,
                        best.expected.range.end - best.expected.range.start,
                    )).compare(.gt)) {
                        best = .{
                            .ni = matcher.best.ni,
                            .expected = .{
                                .range = .{
                                    .start = start,
                                    .end = start + expected.index,
                                },
                                .what = expected.what,
                            },
                        };
                    },
                }
            }
            return .{ .expected = best.expected };
        }

        pub const MatchAnchor = Matcher.Anchor;
        pub const MatchResult = Matcher.Result;
        pub fn matches(
            c: *const Compiled,
            gpa: Allocator,
            text: []const u8,
            anchor: MatchAnchor,
        ) Allocator.Error!MatchResult {
            var arena: std.heap.ArenaAllocator = .init(gpa);
            defer arena.deinit();
            var matcher: Pattern.Matcher = .{
                .gpa = gpa,
                .arena = arena.allocator(),
                .nodes = &c.nodes,
                .text = text,
                .states = .empty,
                .best = .{
                    .ni = 0,
                    .expected = .{
                        .index = 0,
                        .what = .{ .any = .empty },
                    },
                },
            };
            defer matcher.deinit();
            return matcher.run(anchor);
        }

        pub const Print = struct {
            gpa: Allocator,
            pattern: *const Compiled,
            err: ?Allocator.Error,

            pub fn format(p: *Print, w: *std.Io.Writer) std.Io.Writer.Error!void {
                print(p.gpa, p.pattern, w) catch |err| switch (err) {
                    error.WriteFailed => |e| return e,
                    error.OutOfMemory => |e| {
                        p.err = e;
                        return error.WriteFailed;
                    },
                };
            }

            pub fn print(
                gpa: Allocator,
                c: *const Compiled,
                w: *std.Io.Writer,
            ) (Allocator.Error || std.Io.Writer.Error)!void {
                var deferred: std.ArrayList(struct { usize, []const u8 }) = .empty;
                defer deferred.deinit(gpa);
                {
                    var ni = c.nodes.len;
                    while (ni > 0) {
                        ni -= 1;
                        switch (c.nodes.get(ni)) {
                            else => {},
                            .choice => |alt_ni| if (alt_ni <= ni)
                                try deferred.append(gpa, .{ alt_ni, "(?:" }),
                        }
                    }
                }
                var captures: std.array_hash_map.Auto(usize, void) = .empty;
                defer captures.deinit(gpa);
                try w.writeByte('/');
                for (0..c.nodes.len) |ni| {
                    node: switch (c.nodes.get(ni)) {
                        .any, .none => |acs, tag| if (acs.eql(.empty)) try w.writeAll(switch (tag) {
                            else => unreachable,
                            .any => "(?!)",
                            .none => ".",
                        }) else {
                            try w.writeAll(switch (tag) {
                                else => unreachable,
                                .any => "[",
                                .none => "[^",
                            });
                            var ac_it = acs.iterator(.{});
                            while (ac_it.next()) |ac| try writeChar(@as(u7, @intCast(ac)), w);
                            try w.writeByte(']');
                        },
                        .literal => |literal| {
                            const literal_utf8 = std.unicode.Utf8View.init(literal) catch
                                return for (literal) |b| try writeChar(b, w);
                            var literal_utf8_it = literal_utf8.iterator();
                            while (literal_utf8_it.nextCodepoint()) |cp| if (std.math.cast(u7, cp)) |ac|
                                try writeChar(ac, w)
                            else
                                try w.print("\\u{{{x}}}", .{cp});
                        },
                        .goto => try w.writeByte('|'),
                        .choice => |alt_ni| if (alt_ni <= ni) {
                            if (alt_ni > 0) switch (c.nodes.get(alt_ni - 1)) {
                                else => {},
                                .choice => |end_ni| if (ni + 1 == end_ni)
                                    break :node try w.writeAll(")*"),
                            };
                            try w.writeAll(")+");
                        } else switch (c.nodes.get(alt_ni - 1)) {
                            else => {
                                try w.writeAll("(?:");
                                try deferred.append(gpa, .{ alt_ni, ")?" });
                            },
                            .choice => |end_ni| if (ni + 1 == end_ni) break :node,
                            .goto => |end_ni| {
                                if (deferred.last()) |end| {
                                    if (end_ni == end[0] and std.mem.eql(u8, end[1], ")")) break :node;
                                }
                                try w.writeAll("(?:");
                                try deferred.append(gpa, .{ end_ni, ")" });
                            },
                        },
                        .capture_start => {
                            try captures.putNoClobber(gpa, ni, {});
                            try w.writeByte('(');
                        },
                        .capture_end => try w.writeByte(')'),
                        .reference => |capture| try w.print("\\{d}", .{
                            1 + captures.getIndex(capture.start_ni).?,
                        }),
                    }
                    while (deferred.last()) |end| {
                        if (ni + 1 != end[0]) break;
                        try w.writeAll(end[1]);
                        _ = deferred.pop().?;
                    }
                }
                try w.writeByte('/');
            }

            fn writeChar(c: u8, w: *std.Io.Writer) std.Io.Writer.Error!void {
                switch (c) {
                    0...'\t' - 1,
                    '\n' + 1...'\r' - 1,
                    '\r' + 1...' ' - 1,
                    '~' + 1...std.math.maxInt(u8),
                    => try w.print("\\x{x:02}", .{c}),
                    '\t' => try w.writeAll("\\t"),
                    '\n' => try w.writeAll("\\n"),
                    '\r' => try w.writeAll("\\r"),
                    ' '...'(' - 1,
                    '+' + 1...'-' - 1,
                    '.' + 1...'[' - 1,
                    '^' + 1...'|' - 1,
                    '|' + 1...'~',
                    => try w.writeByte(c),
                    '(', ')', '*', '+', '-', '.', '[', '\\', ']', '^', '|' => try w.print("\\{c}", .{c}),
                }
            }
        };

        fn appendInner(c: *Compiled, gpa: Allocator, patterns: []const Pattern) Allocator.Error!void {
            for (patterns) |pattern| switch (pattern) {
                .any => |acrs| try c.nodes.append(gpa, .{ .any = parseAsciiCharRanges(acrs) }),
                .none => |acrs| try c.nodes.append(gpa, .{ .none = parseAsciiCharRanges(acrs) }),
                .literal => |literal| try c.nodes.append(gpa, .{ .literal = literal }),
                .@"?" => |sub_patterns| {
                    const choice_ni = try c.nodes.addOne(gpa);
                    try c.appendInner(gpa, sub_patterns);
                    c.nodes.set(choice_ni, .{ .choice = c.nodes.len });
                },
                .@"*" => |sub_patterns| {
                    const choice_ni = try c.nodes.addOne(gpa);
                    const loop_ni = c.nodes.len;
                    try c.appendInner(gpa, sub_patterns);
                    try c.nodes.append(gpa, .{ .choice = loop_ni });
                    c.nodes.set(choice_ni, .{ .choice = c.nodes.len });
                },
                .@"+" => |sub_patterns| {
                    const loop_ni = c.nodes.len;
                    try c.appendInner(gpa, sub_patterns);
                    try c.nodes.append(gpa, .{ .choice = loop_ni });
                },
                .@"|" => |choices| switch (choices.len) {
                    0 => try c.nodes.append(gpa, .{ .any = .empty }),
                    else => {
                        const stack_start = c.stack.items.len;
                        defer c.stack.shrinkRetainingCapacity(stack_start);
                        for (choices[0 .. choices.len - 1]) |choice| {
                            const choice_ni = try c.nodes.addOne(gpa);
                            try c.appendInner(gpa, choice);
                            try c.stack.append(gpa, try c.nodes.addOne(gpa));
                            c.nodes.set(choice_ni, .{ .choice = c.nodes.len });
                        }
                        try c.appendInner(gpa, choices[choices.len - 1]);
                        for (c.stack.items[stack_start..]) |goto_ni|
                            c.nodes.set(goto_ni, .{ .goto = c.nodes.len });
                    },
                },
                .capture => |sub_patterns| {
                    const capture = try c.captures.addOne(gpa);
                    capture.start_ni = try c.nodes.addOne(gpa);
                    try c.appendInner(gpa, sub_patterns);
                    capture.end_ni = c.nodes.len;
                    c.nodes.set(capture.start_ni, .{ .capture_start = capture.end_ni });
                    try c.nodes.append(gpa, .{ .capture_end = capture.start_ni });
                },
                .reference => |ci| try c.nodes.append(gpa, .{ .reference = c.captures.items[ci] }),
            };
        }
    };

    const Matcher = struct {
        gpa: Allocator,
        arena: Allocator,
        nodes: *const std.MultiArrayList(Compiled.Node),
        text: []const u8,
        states: std.ArrayList(State),
        best: Failure,

        const Result = union(enum) {
            matched: usize,
            expected: Expected,

            pub const Expected = struct {
                index: usize,
                what: What,

                pub const What = union(enum) {
                    any: AsciiCharSet,
                    none: AsciiCharSet,
                    literal: []const u8,

                    pub fn merge(lhs: What, rhs: What) What {
                        switch (lhs) {
                            .any => |lhs_acs| if (lhs_acs.eql(.empty)) return rhs else switch (rhs) {
                                .any => |rhs_acs| return .{ .any = lhs_acs.unionWith(rhs_acs) },
                                .none => |rhs_acs| return .{ .none = rhs_acs.differenceWith(lhs_acs) },
                                .literal => |rhs_literal| {
                                    var acs = lhs_acs;
                                    acs.set(rhs_literal[0]);
                                    return .{ .any = acs };
                                },
                            },
                            .none => |lhs_acs| switch (rhs) {
                                .any => |rhs_acs| return .{ .none = lhs_acs.differenceWith(rhs_acs) },
                                .none => |rhs_acs| return .{ .none = lhs_acs.intersectWith(rhs_acs) },
                                .literal => |rhs_literal| {
                                    var acs = lhs_acs;
                                    acs.unset(rhs_literal[0]);
                                    return .{ .none = acs };
                                },
                            },
                            .literal => |lhs_literal| switch (rhs) {
                                .any => |rhs_acs| if (rhs_acs.eql(.empty)) return lhs else {
                                    var acs = rhs_acs;
                                    acs.set(lhs_literal[0]);
                                    return .{ .any = acs };
                                },
                                .none => |rhs_acs| {
                                    var acs = rhs_acs;
                                    acs.unset(lhs_literal[0]);
                                    return .{ .none = acs };
                                },
                                .literal => |rhs_literal| {
                                    var acs: AsciiCharSet = .empty;
                                    acs.set(lhs_literal[0]);
                                    acs.set(rhs_literal[0]);
                                    return .{ .any = acs };
                                },
                            },
                        }
                    }
                };
            };
        };

        const State = struct {
            ni: usize,
            ti: usize,
            captures: ?*const Captures,

            const Captures = struct {
                ni: usize,
                ti: usize,
                next: ?*const Captures,
            };
        };

        const Anchor = enum { start, all };

        const Failure = struct { ni: usize, expected: Result.Expected };

        fn deinit(m: *Matcher) void {
            m.states.deinit(m.gpa);
            m.* = undefined;
        }

        fn run(m: *Matcher, anchor: Anchor) Allocator.Error!Result {
            try m.states.append(m.gpa, .{ .ni = 0, .ti = 0, .captures = null });
            while (m.states.pop()) |state| {
                if (m.nodes.len - state.ni == 0) {
                    switch (anchor) {
                        .start => {},
                        .all => if (m.text.len - state.ti > 0) {
                            m.fail(.{
                                .ni = state.ni,
                                .expected = .{
                                    .index = state.ti,
                                    .what = .{ .any = .empty },
                                },
                            });
                            continue;
                        },
                    }
                    return .{ .matched = state.ti };
                }
                node: switch (m.nodes.get(state.ni)) {
                    .any => |acs| {
                        if (fail: {
                            if (m.text.len - state.ti == 0) break :fail true;
                            const c = m.text[state.ti];
                            if (c >= AsciiCharSet.bit_length) break :fail true;
                            break :fail !acs.isSet(c);
                        }) {
                            m.fail(.{
                                .ni = state.ni,
                                .expected = .{
                                    .index = state.ti,
                                    .what = .{ .any = acs },
                                },
                            });
                            continue;
                        }
                        m.states.appendAssumeCapacity(.{
                            .ni = state.ni + 1,
                            .ti = state.ti + 1,
                            .captures = state.captures,
                        });
                    },
                    .none => |acs| {
                        if (fail: {
                            if (m.text.len - state.ti == 0) break :fail true;
                            const c = m.text[state.ti];
                            if (c >= AsciiCharSet.bit_length) break :fail false;
                            break :fail acs.isSet(c);
                        }) {
                            m.fail(.{
                                .ni = state.ni,
                                .expected = .{
                                    .index = state.ti,
                                    .what = .{ .any = acs },
                                },
                            });
                            continue;
                        }
                        m.states.appendAssumeCapacity(.{
                            .ni = state.ni + 1,
                            .ti = state.ti + 1,
                            .captures = state.captures,
                        });
                    },
                    .literal => |literal| {
                        fail: {
                            const diff_index = std.mem.findDiff(u8, m.text[state.ti..], literal) orelse
                                break :fail;
                            if (literal.len - diff_index == 0) break :fail;
                            m.fail(.{
                                .ni = state.ni,
                                .expected = .{
                                    .index = state.ti + diff_index,
                                    .what = .{ .literal = literal[diff_index..] },
                                },
                            });
                            continue;
                        }
                        m.states.appendAssumeCapacity(.{
                            .ni = state.ni + 1,
                            .ti = state.ti + literal.len,
                            .captures = state.captures,
                        });
                    },
                    .choice => |alt_ni| {
                        const states = try m.states.addManyAsArray(m.gpa, 2);
                        states.* = .{
                            .{ .ni = alt_ni, .ti = state.ti, .captures = state.captures },
                            .{ .ni = state.ni + 1, .ti = state.ti, .captures = state.captures },
                        };
                        if (states[0].ni < states[1].ni) std.mem.swap(State, &states[0], &states[1]);
                    },
                    .goto => |target_ni| m.states.appendAssumeCapacity(.{
                        .ni = target_ni,
                        .ti = state.ti,
                        .captures = state.captures,
                    }),
                    .capture_start, .capture_end => {
                        const captures = try m.arena.create(State.Captures);
                        captures.* = .{
                            .ni = state.ni,
                            .ti = state.ti,
                            .next = state.captures,
                        };
                        m.states.appendAssumeCapacity(.{
                            .ni = state.ni + 1,
                            .ti = state.ti,
                            .captures = captures,
                        });
                    },
                    .reference => |reference| {
                        var capture_it = state.captures;
                        const end = while (capture_it) |capture| {
                            capture_it = capture.next;
                            if (capture.ni == reference.end_ni) break capture.ti;
                        } else unreachable;
                        const start = while (capture_it) |capture| {
                            capture_it = capture.next;
                            if (capture.ni == reference.start_ni) break capture.ti;
                        } else unreachable;
                        continue :node .{ .literal = m.text[start..end] };
                    },
                }
            }
            return .{ .expected = m.best.expected };
        }

        fn fail(m: *Matcher, failure: Failure) void {
            switch (std.math.order(failure.ni, m.best.ni).differ() orelse
                std.math.order(failure.expected.index, m.best.expected.index)) {
                .lt => {},
                .eq => m.best.expected.what = m.best.expected.what.merge(failure.expected.what),
                .gt => m.best = failure,
            }
        }
    };
};

test "literal" {
    const gpa = std.testing.allocator;
    var pattern: Pattern.Compiled = try .init(gpa, &.{
        .{ .literal = "foo" },
    });
    defer pattern.deinit(gpa);
    try std.testing.expectEqual(Pattern.Matcher.Result{ .expected = .{
        .index = "f".len,
        .what = .{ .literal = "foo"["f".len..] },
    } }, try pattern.matches(gpa, "f"));
    try std.testing.expectEqual(Pattern.Matcher.Result{ .expected = .{
        .index = "fo".len,
        .what = .{ .literal = "foo"["fo".len..] },
    } }, try pattern.matches(gpa, "fo"));
    try std.testing.expectEqual(Pattern.Matcher.Result{
        .matched = "foo".len,
    }, try pattern.matches(gpa, "foo"));
    try std.testing.expectEqual(Pattern.Matcher.Result{
        .matched = "foo".len,
    }, try pattern.matches(gpa, "foobar"));
}

test "word" {
    const gpa = std.testing.allocator;
    var pattern: Pattern.Compiled = try .init(gpa, &.{
        .{ .any = &.{
            .{ 'A', 'Z' },
            .{ '_', '_' },
            .{ 'a', 'z' },
        } },
        .{ .@"*" = &.{
            .{ .any = &.{
                .{ '0', '9' },
                .{ 'A', 'Z' },
                .{ '_', '_' },
                .{ 'a', 'z' },
            } },
        } },
    });
    defer pattern.deinit(gpa);
    try std.testing.expectEqual(Pattern.Matcher.Result{
        .matched = "This".len,
    }, try pattern.matches(gpa, "This is a sentence."));
    try std.testing.expectEqual(Pattern.Matcher.Result{ .expected = .{
        .index = "".len,
        .what = .{ .any = Pattern.parseAsciiCharRanges(&.{
            .{ 'A', 'Z' },
            .{ '_', '_' },
            .{ 'a', 'z' },
        }) },
    } }, try pattern.matches(gpa, "3.141592653"));
}

test "alternation" {
    const gpa = std.testing.allocator;
    var pattern: Pattern.Compiled = try .init(gpa, &.{
        .{ .@"|" = &.{ &.{
            .{ .literal = "foo" },
        }, &.{
            .{ .literal = "bar" },
        } } },
    });
    defer pattern.deinit(gpa);
    try std.testing.expectEqual(Pattern.Matcher.Result{
        .matched = "foo".len,
    }, try pattern.matches(gpa, "foobar"));
    try std.testing.expectEqual(Pattern.Matcher.Result{
        .matched = "bar".len,
    }, try pattern.matches(gpa, "barfoo"));
    try std.testing.expectEqual(Pattern.Matcher.Result{ .expected = .{
        .index = "ba".len,
        .what = .{ .literal = "bar"["ba".len..] },
    } }, try pattern.matches(gpa, "baz"));
}

test "capture" {
    const gpa = std.testing.allocator;
    var pattern: Pattern.Compiled = try .init(gpa, &.{
        .{ .capture = &.{
            .{ .none = &.{} },
        } },
        .{ .reference = 0 },
    });
    defer pattern.deinit(gpa);
    try std.testing.expectEqual(Pattern.Matcher.Result{
        .matched = "00".len,
    }, try pattern.matches(gpa, "001122"));
    try std.testing.expectEqual(Pattern.Matcher.Result{
        .matched = "aa".len,
    }, try pattern.matches(gpa, "aabbcc"));
    try std.testing.expectEqual(Pattern.Matcher.Result{ .expected = .{
        .index = "a".len,
        .what = .{ .literal = "abc"[0.."a".len] },
    } }, try pattern.matches(gpa, "abc"));
}

const Allocator = std.mem.Allocator;
const std = @import("std");
