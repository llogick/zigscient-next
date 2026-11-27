const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const ArrayList = std.ArrayList;

const Scanner = @import("Scanner.zig");
const Token = Scanner.Token;
const AllocWhen = Scanner.AllocWhen;
const default_max_value_len = Scanner.default_max_value_len;
const isNumberFormattedLikeAnInteger = Scanner.isNumberFormattedLikeAnInteger;
const Diagnostics = Scanner.Diagnostics;
const TokenType = Scanner.TokenType;

const Value = @import("./dynamic.zig").Value;
const Array = @import("./dynamic.zig").Array;

/// Controls how to deal with various inconsistencies between the JSON document and the Zig struct type passed in.
/// For duplicate fields or unknown fields, set options in this struct.
/// For missing fields, give the Zig struct fields default values.
pub const ParseOptions = struct {
    /// Behaviour when a duplicate field is encountered.
    /// The default is to return `error.DuplicateField`.
    duplicate_field_behavior: enum {
        use_first,
        @"error",
        use_last,
    } = .@"error",

    /// If false, finding an unknown field returns `error.UnknownField`.
    ignore_unknown_fields: bool = false,

    /// Passed to `std.json.Scanner.nextAllocMax` or `std.json.Reader.nextAllocMax`.
    /// The default for `parseFromSlice` or `parseFromTokenSource` with a `*std.json.Scanner` input
    /// is the length of the input slice, which means `error.ValueTooLong` will never be returned.
    /// The default for `parseFromTokenSource` with a `*std.json.Reader` is `std.json.default_max_value_len`.
    /// Ignored for values that don't need allocation or are not copied (see `allocate`).
    /// Ignored for `parseFromValue` and `parseFromValueLeaky`.
    max_value_len: ?usize = null,

    /// This determines whether strings should always be copied,
    /// or if a reference to the given buffer should be preferred if possible.
    /// The default for `parseFromSlice` or `parseFromTokenSource` with a `*std.json.Scanner` input
    /// is `.alloc_if_needed`.
    /// The default with a `*std.json.Reader` input is `.alloc_always`.
    /// Ignored for `parseFromValue` and `parseFromValueLeaky`.
    allocate: ?AllocWhen = null,

    /// When parsing to a `std.json.Value`, set this option to false to always emit
    /// JSON numbers as unparsed `std.json.Value.number_string`.
    /// Otherwise, JSON numbers are parsed as either `std.json.Value.integer`,
    /// `std.json.Value.float` or left as unparsed `std.json.Value.number_string`
    /// depending on the format and value of the JSON number.
    /// When this option is true, JSON numbers encoded as floats (see `std.json.isNumberFormattedLikeAnInteger`)
    /// may lose precision when being parsed into `std.json.Value.float`.
    parse_numbers: bool = true,

    /// Enable diagnostics with `var diagnostics: json.Diagnostics = .{};`,
    /// and set this option to `&diagnostics`.
    diagnostics: ?*Diagnostics = null,
};

pub fn Parsed(comptime T: type) type {
    return struct {
        arena: *ArenaAllocator,
        value: T,

        pub fn deinit(self: @This()) void {
            const allocator = self.arena.child_allocator;
            self.arena.deinit();
            allocator.destroy(self.arena);
        }
    };
}

/// Parses the json document from `s` and returns the result packaged in a `std.json.Parsed`.
/// Created resources are not managed by this function.
/// Function will not return `error.BufferUnderrun`.
pub fn parseFromSlice(
    comptime T: type,
    gpa: Allocator,
    s: []const u8,
    options: ParseOptions,
) ParseError(Scanner)!Parsed(T) {
    var scanner = Scanner.initCompleteInput(gpa, s);
    defer scanner.deinit();

    return parseFromTokenSource(T, gpa, &scanner, options);
}

/// Parses the json document from `s` and returns the result.
/// Allocations made during this operation are not carefully tracked and may not be possible to individually clean up.
/// It is recommended to use a `std.heap.ArenaAllocator` or similar.
pub fn parseFromSliceLeaky(
    comptime T: type,
    arena: Allocator,
    s: []const u8,
    options: ParseOptions,
) ParseError(Scanner)!T {
    var scanner = Scanner.initCompleteInput(arena, s);
    defer scanner.deinit();

    return parseFromTokenSourceLeaky(T, arena, &scanner, options);
}

/// `scanner_or_reader` must be either a `*std.json.Scanner` with complete input or a `*std.json.Reader`.
/// Function will not return `error.BufferUnderrun`.
pub fn parseFromTokenSource(
    comptime T: type,
    gpa: Allocator,
    scanner_or_reader: anytype,
    options: ParseOptions,
) ParseError(@TypeOf(scanner_or_reader.*))!Parsed(T) {
    errdefer scanner_or_reader.saveDiagnostics();
    var parsed = Parsed(T){
        .arena = try gpa.create(ArenaAllocator),
        .value = undefined,
    };
    errdefer gpa.destroy(parsed.arena);
    parsed.arena.* = ArenaAllocator.init(gpa);
    errdefer parsed.arena.deinit();

    parsed.value = try parseFromTokenSourceLeaky(T, parsed.arena.allocator(), scanner_or_reader, options);

    return parsed;
}

/// `scanner_or_reader` must be either a `*std.json.Scanner` with complete input or a `*std.json.Reader`.
/// Allocations made during this operation are not carefully tracked and may not be possible to individually clean up.
/// It is recommended to use a `std.heap.ArenaAllocator` or similar.
pub fn parseFromTokenSourceLeaky(
    comptime T: type,
    arena: Allocator,
    scanner_or_reader: anytype,
    options: ParseOptions,
) ParseError(@TypeOf(scanner_or_reader.*))!T {
    errdefer scanner_or_reader.saveDiagnostics();
    if (@TypeOf(scanner_or_reader.*) == Scanner) {
        assert(scanner_or_reader.is_end_of_input);
    }
    var resolved_options = options;
    if (resolved_options.max_value_len == null) {
        if (@TypeOf(scanner_or_reader.*) == Scanner) {
            resolved_options.max_value_len = scanner_or_reader.input.len;
        } else {
            resolved_options.max_value_len = default_max_value_len;
        }
    }
    if (resolved_options.allocate == null) {
        if (@TypeOf(scanner_or_reader.*) == Scanner) {
            resolved_options.allocate = .alloc_if_needed;
        } else {
            resolved_options.allocate = .alloc_always;
        }
    }
    if (resolved_options.diagnostics) |diag| {
        scanner_or_reader.enableDiagnostics(diag);
    }

    const value = try innerParse(T, arena, scanner_or_reader, resolved_options);

    assert(.end_of_document == try scanner_or_reader.next());

    return value;
}

/// Like `parseFromSlice`, but the input is an already-parsed `std.json.Value` object.
/// Only `options.ignore_unknown_fields` is used from `options`.
pub fn parseFromValue(
    comptime T: type,
    gpa: Allocator,
    source: Value,
    options: ParseOptions,
) ParseFromValueError!Parsed(T) {
    var parsed = Parsed(T){
        .arena = try gpa.create(ArenaAllocator),
        .value = undefined,
    };
    errdefer gpa.destroy(parsed.arena);
    parsed.arena.* = ArenaAllocator.init(gpa);
    errdefer parsed.arena.deinit();

    parsed.value = try parseFromValueLeaky(T, parsed.arena.allocator(), source, options);

    return parsed;
}

/// Parses the json document from `source` and returns the result.
/// Allocations made during this operation are not carefully tracked and may not be possible to individually clean up.
/// It is recommended to use a `std.heap.ArenaAllocator` or similar.
pub fn parseFromValueLeaky(
    comptime T: type,
    arena: Allocator,
    source: Value,
    options: ParseOptions,
) ParseFromValueError!T {
    // I guess this function doesn't need to exist,
    // but the flow of the sourcecode is easy to follow and grouped nicely with
    // this pub redirect function near the top and the implementation near the bottom.
    return innerParseFromValue(T, arena, source, options);
}

/// The error set that will be returned when parsing from `*Source`.
/// Note that this may contain `error.BufferUnderrun`, but that error will never actually be returned.
pub fn ParseError(comptime Source: type) type {
    // A few of these will either always be present or present enough of the time that
    // omitting them is more confusing than always including them.
    return ParseFromValueError || Source.NextError || Source.PeekError || Source.AllocError;
}

pub const ParseFromValueError = std.fmt.ParseIntError || std.fmt.ParseFloatError || Allocator.Error || error{
    UnexpectedToken,
    InvalidNumber,
    Overflow,
    InvalidEnumTag,
    DuplicateField,
    UnknownField,
    MissingField,
    LengthMismatch,
};

/// This is an internal function called recursively
/// during the implementation of `parseFromTokenSourceLeaky` and similar.
/// It is exposed primarily to enable custom `jsonParse()` methods to call back into the `parseFrom*` system,
/// such as when implementing a custom container of type `T`;
/// `innerParse(T, ...)` can be called for each of the container's items.
/// `null` fields are not allowed on the `options` when calling this function.
/// (The `options` in the `jsonParse` method have no `null` fields.)
pub fn innerParse(
    comptime T: type,
    arena: Allocator,
    source: anytype,
    options: ParseOptions,
) ParseError(@TypeOf(source.*))!T {
    errdefer if (options.diagnostics) |diag| {
        source.saveDiagnostics();
        diag.recordContext(@typeName(T));
    };
    switch (@typeInfo(T)) {
        .bool => {
            return switch (try source.next()) {
                .true => true,
                .false => false,
                else => |t| return typeError(options.diagnostics, t, "bool"),
            };
        },
        .float, .comptime_float => {
            const token = try source.nextAllocMax(arena, .alloc_if_needed, options.max_value_len.?);
            defer freeAllocated(arena, token);
            const slice = switch (token) {
                inline .number, .allocated_number, .string, .allocated_string => |slice| slice,
                else => |t| return typeError(options.diagnostics, t, "float"),
            };
            return try std.fmt.parseFloat(T, slice);
        },
        .int, .comptime_int => {
            const token = try source.nextAllocMax(arena, .alloc_if_needed, options.max_value_len.?);
            defer freeAllocated(arena, token);
            const slice = switch (token) {
                inline .number, .allocated_number, .string, .allocated_string => |slice| slice,
                else => |t| return typeError(options.diagnostics, t, "int"),
            };
            return sliceToInt(T, slice, options);
        },
        .optional => |optionalInfo| {
            switch (try source.peekNextTokenType()) {
                .null => {
                    _ = try source.next();
                    return null;
                },
                else => {
                    return try innerParse(optionalInfo.child, arena, source, options);
                },
            }
        },
        .@"enum" => {
            if (std.meta.hasFn(T, "jsonParse")) {
                return T.jsonParse(arena, source, options);
            }

            const token = try source.nextAllocMax(arena, .alloc_if_needed, options.max_value_len.?);
            defer freeAllocated(arena, token);
            const slice = switch (token) {
                inline .number, .allocated_number, .string, .allocated_string => |slice| slice,
                else => |t| return typeError(options.diagnostics, t, "enum (number or string)"),
            };
            return sliceToEnum(T, slice, options);
        },
        .@"union" => |unionInfo| {
            if (std.meta.hasFn(T, "jsonParse")) {
                return T.jsonParse(arena, source, options);
            }

            if (unionInfo.tag_type == null) @compileError("Unable to parse into untagged union '" ++ @typeName(T) ++ "'");

            switch (try source.next()) {
                .object_begin => {},
                else => |t| return typeError(options.diagnostics, t, "union (object with one field)"),
            }

            var result: ?T = null;
            var name_token: ?Token = try source.nextAllocMax(arena, .alloc_if_needed, options.max_value_len.?);
            const field_name = switch (name_token.?) {
                inline .string, .allocated_string => |slice| slice,
                else => {
                    if (options.diagnostics) |diag| diag.setMessage("tagged union requires a field to identify the active tag");
                    return error.UnexpectedToken;
                },
            };

            inline for (unionInfo.field_names, unionInfo.field_types) |u_field_name, u_field_type| {
                if (std.mem.eql(u8, u_field_name, field_name)) {
                    // Free the name token now in case we're using an allocator that optimizes freeing the last allocated object.
                    // (Recursing into innerParse() might trigger more allocations.)
                    freeAllocated(arena, name_token.?);
                    name_token = null;
                    if (u_field_type == void) {
                        // void isn't really a json type, but we can support void payload union tags with {} as a value.
                        switch (try source.next()) {
                            .object_begin => {},
                            else => |t| return typeError(options.diagnostics, t, "void payload ('{}')"),
                        }
                        if (.object_end != try source.next()) {
                            if (options.diagnostics) |diag| diag.setMessage("void payload '{}' should have no fields");
                            return error.UnexpectedToken;
                        }
                        result = @unionInit(T, u_field_name, {});
                    } else {
                        // Recurse.
                        result = @unionInit(T, u_field_name, try innerParse(u_field_type, arena, source, options));
                    }
                    break;
                }
            } else {
                // Didn't match anything.
                if (options.diagnostics) |diag| diag.setMessage("unrecognized tag name");
                return error.UnknownField;
            }

            if (.object_end != try source.next()) {
                if (options.diagnostics) |diag| diag.setMessage("tagged union requires only one field or object is badly formatted json");
                return error.UnexpectedToken;
            }

            return result.?;
        },

        .@"struct" => |structInfo| {
            if (structInfo.is_tuple) {
                switch (try source.next()) {
                    .array_begin => {},
                    else => |t| return typeError(options.diagnostics, t, "tuple (array of values)"),
                }

                var r: T = undefined;
                inline for (structInfo.field_types, 0..) |field_type, i| {
                    if (.array_end == try source.peekNextTokenType()) {
                        if (options.diagnostics) |diag| diag.setMessage(std.fmt.comptimePrint(
                            "tuple too short. expected length: {}",
                            .{structInfo.field_types.len},
                        ));
                        return error.LengthMismatch;
                    }
                    r[i] = try innerParse(field_type, arena, source, options);
                }

                if (.array_end != try source.next()) {
                    if (options.diagnostics) |diag| diag.setMessage(std.fmt.comptimePrint(
                        "tuple too long. expected length: {}",
                        .{structInfo.field_types.len},
                    ));
                    return error.UnexpectedToken;
                }

                return r;
            }

            if (std.meta.hasFn(T, "jsonParse")) {
                return T.jsonParse(arena, source, options);
            }

            switch (try source.next()) {
                .object_begin => {},
                else => |t| return typeError(options.diagnostics, t, "struct ({...})"),
            }

            var r: T = undefined;
            var fields_seen: [structInfo.field_names.len]bool = @splat(false);

            while (true) {
                var name_token: ?Token = try source.nextAllocMax(arena, .alloc_if_needed, options.max_value_len.?);
                const field_name = switch (name_token.?) {
                    inline .string, .allocated_string => |slice| slice,
                    .object_end => { // No more fields.
                        break;
                    },
                    else => |t| return typeError(options.diagnostics, t, "struct ({...})"),
                };

                inline for (
                    structInfo.field_names,
                    structInfo.field_types,
                    structInfo.field_attrs,
                    0..,
                ) |f_name, f_type, f_attrs, i| {
                    if (f_attrs.@"comptime") @compileError("comptime fields are not supported: " ++ @typeName(T) ++ "." ++ f_name);
                    if (std.mem.eql(u8, f_name, field_name)) {
                        // Free the name token now in case we're using an allocator that optimizes freeing the last allocated object.
                        // (Recursing into innerParse() might trigger more allocations.)
                        freeAllocated(arena, name_token.?);
                        errdefer if (options.diagnostics) |diag| diag.recordContext(@typeName(T) ++ "." ++ f_name);
                        name_token = null;
                        if (fields_seen[i]) {
                            switch (options.duplicate_field_behavior) {
                                .use_first => {
                                    // Parse and ignore the redundant value.
                                    // We don't want to skip the value, because we want type checking.
                                    _ = try innerParse(f_type, arena, source, options);
                                    break;
                                },
                                .@"error" => {
                                    if (options.diagnostics) |diag| diag.setMessage("duplicate field name: " ++ f_name);
                                    return error.DuplicateField;
                                },
                                .use_last => {},
                            }
                        }
                        @field(r, f_name) = try innerParse(f_type, arena, source, options);
                        fields_seen[i] = true;
                        break;
                    }
                } else {
                    // Didn't match anything.
                    freeAllocated(arena, name_token.?);
                    if (options.ignore_unknown_fields) {
                        try source.skipValue();
                    } else {
                        if (options.diagnostics) |diag| diag.setMessageFmt("found unknown field. found: {s}", .{field_name});
                        return error.UnknownField;
                    }
                }
            }
            try fillDefaultStructValues(T, options, &r, &fields_seen);
            return r;
        },

        .array => |arrayInfo| {
            switch (try source.peekNextTokenType()) {
                .array_begin => {
                    // Typical array.
                    return internalParseArray(T, arrayInfo.child, arena, source, options);
                },
                .string => {
                    if (arrayInfo.child != u8) return typeError(options.diagnostics, .string, "string");
                    // Fixed-length string.

                    var r: T = undefined;
                    var i: usize = 0;
                    while (true) {
                        switch (try source.next()) {
                            .string => |slice| {
                                if (i + slice.len > r.len) {
                                    if (options.diagnostics) |diag| diag.setMessage(std.fmt.comptimePrint("string too long for fixed length array. expected length: {}", .{r.len}));
                                    return error.LengthMismatch;
                                }
                                if (i + slice.len < r.len) {
                                    if (options.diagnostics) |diag| diag.setMessage(std.fmt.comptimePrint("string too short for fixed length array. expected length: {}", .{r.len}));
                                    return error.LengthMismatch;
                                }
                                @memcpy(r[i..][0..slice.len], slice);
                                break;
                            },
                            .partial_string => |slice| {
                                if (i + slice.len > r.len) {
                                    if (options.diagnostics) |diag| diag.setMessage(std.fmt.comptimePrint("string too long for fixed length array. expected length: {}", .{r.len}));
                                    return error.LengthMismatch;
                                }
                                @memcpy(r[i..][0..slice.len], slice);
                                i += slice.len;
                            },
                            .partial_string_escaped_1 => |arr| {
                                if (i + arr.len > r.len) {
                                    if (options.diagnostics) |diag| diag.setMessage(std.fmt.comptimePrint("string too long for fixed length array. expected length: {}", .{r.len}));
                                    return error.LengthMismatch;
                                }
                                @memcpy(r[i..][0..arr.len], arr[0..]);
                                i += arr.len;
                            },
                            .partial_string_escaped_2 => |arr| {
                                if (i + arr.len > r.len) {
                                    if (options.diagnostics) |diag| diag.setMessage(std.fmt.comptimePrint("string too long for fixed length array. expected length: {}", .{r.len}));
                                    return error.LengthMismatch;
                                }
                                @memcpy(r[i..][0..arr.len], arr[0..]);
                                i += arr.len;
                            },
                            .partial_string_escaped_3 => |arr| {
                                if (i + arr.len > r.len) {
                                    if (options.diagnostics) |diag| diag.setMessage(std.fmt.comptimePrint("string too long for fixed length array. expected length: {}", .{r.len}));
                                    return error.LengthMismatch;
                                }
                                @memcpy(r[i..][0..arr.len], arr[0..]);
                                i += arr.len;
                            },
                            .partial_string_escaped_4 => |arr| {
                                if (i + arr.len > r.len) {
                                    if (options.diagnostics) |diag| diag.setMessage(std.fmt.comptimePrint("string too long for fixed length array. expected length: {}", .{r.len}));
                                    return error.LengthMismatch;
                                }
                                @memcpy(r[i..][0..arr.len], arr[0..]);
                                i += arr.len;
                            },
                            else => unreachable,
                        }
                    }

                    return r;
                },

                else => |t| return typeError(options.diagnostics, t, if (arrayInfo.child != u8) "array" else "array or string"),
            }
        },

        .vector => |vector_info| {
            switch (try source.peekNextTokenType()) {
                .array_begin => {
                    const A = [vector_info.len]vector_info.child;
                    return try internalParseArray(A, vector_info.child, arena, source, options);
                },
                else => |t| return typeError(options.diagnostics, t, "array"),
            }
        },

        .pointer => |ptrInfo| {
            switch (ptrInfo.size) {
                .one => {
                    const r: *ptrInfo.child = try arena.create(ptrInfo.child);
                    r.* = try innerParse(ptrInfo.child, arena, source, options);
                    return r;
                },
                .slice => {
                    switch (try source.peekNextTokenType()) {
                        .array_begin => {
                            _ = try source.next();

                            // Typical array.
                            var arraylist: ArrayList(ptrInfo.child) = .empty;
                            while (true) {
                                switch (try source.peekNextTokenType()) {
                                    .array_end => {
                                        _ = try source.next();
                                        break;
                                    },
                                    else => {},
                                }

                                try arraylist.ensureUnusedCapacity(arena, 1);
                                arraylist.appendAssumeCapacity(try innerParse(ptrInfo.child, arena, source, options));
                            }

                            if (ptrInfo.sentinel()) |s| {
                                return try arraylist.toOwnedSliceSentinel(arena, s);
                            }

                            return try arraylist.toOwnedSlice(arena);
                        },
                        .string => {
                            if (ptrInfo.child != u8) {
                                if (options.diagnostics) |diag| diag.setMessageFmt("expected array of {}, found string", .{ptrInfo.child});
                                return error.UnexpectedToken;
                            }

                            // Dynamic length string.
                            if (ptrInfo.sentinel()) |s| {
                                // Use our own array list so we can append the sentinel.
                                var value_list: ArrayList(u8) = .empty;
                                _ = try source.allocNextIntoArrayListMax(arena, &value_list, .alloc_always, options.max_value_len.?);
                                return try value_list.toOwnedSliceSentinel(arena, s);
                            }
                            if (ptrInfo.attrs.@"const") {
                                switch (try source.nextAllocMax(arena, options.allocate.?, options.max_value_len.?)) {
                                    inline .string, .allocated_string => |slice| return slice,
                                    else => unreachable,
                                }
                            } else {
                                // Have to allocate to get a mutable copy.
                                switch (try source.nextAllocMax(arena, .alloc_always, options.max_value_len.?)) {
                                    .allocated_string => |slice| return slice,
                                    else => unreachable,
                                }
                            }
                        },
                        else => |t| return typeError(options.diagnostics, t, if (ptrInfo.child == u8) "string" else "array"),
                    }
                },
                else => @compileError("Unable to parse into type '" ++ @typeName(T) ++ "'"),
            }
        },
        else => @compileError("Unable to parse into type '" ++ @typeName(T) ++ "'"),
    }
    unreachable;
}

fn internalParseArray(
    comptime T: type,
    comptime Child: type,
    arena: Allocator,
    source: anytype,
    options: ParseOptions,
) !T {
    assert(.array_begin == try source.next());

    var r: T = undefined;
    for (&r) |*elem| {
        if (.array_end == try source.peekNextTokenType()) {
            if (options.diagnostics) |diag| diag.setMessage(std.fmt.comptimePrint("not enough items for fixed length array. expected length: {}", .{r.len}));
            return error.LengthMismatch;
        }
        elem.* = try innerParse(Child, arena, source, options);
    }

    if (.array_end != try source.next()) {
        if (options.diagnostics) |diag| diag.setMessageFmt("too many items for fixed length array. expected length: {}", .{r.len});
        return error.UnexpectedToken;
    }

    return r;
}

fn coerceToStringDescriptor(token: anytype, comptime prefix: []const u8) []const u8 {
    if (@TypeOf(token) == TokenType) {
        return switch (token) {
            .object_begin => prefix ++ "object",
            .array_begin => prefix ++ "array",
            .true, .false => prefix ++ "bool",
            .null => prefix ++ "null",
            .string => prefix ++ "string",
            .number => prefix ++ "number",
            .object_end => prefix ++ "object",
            .array_end => prefix ++ "array",
            .end_of_document => prefix ++ " end of document",
        };
    }
    if (@TypeOf(token) == Value) {
        return switch (@as(std.meta.Tag(Value), token)) {
            .null => prefix ++ "null",
            .array => prefix ++ "array",
            .integer, .float, .number_string => prefix ++ "number",
            .string => prefix ++ "string",
            .object => prefix ++ "object",
            .bool => prefix ++ "bool",
        };
    }
    return switch (@as(std.meta.Tag(Token), token)) {
        .object_begin => prefix ++ "object",
        .array_begin => prefix ++ "array",
        .true, .false => prefix ++ "bool",
        .null => prefix ++ "null",

        .number,
        .partial_number,
        .allocated_number,
        => prefix ++ "number",

        .string,
        .partial_string,
        .partial_string_escaped_1,
        .partial_string_escaped_2,
        .partial_string_escaped_3,
        .partial_string_escaped_4,
        .allocated_string,
        => prefix ++ "string",

        .object_end => prefix ++ "object",
        .array_end => prefix ++ "array",
        .end_of_document => prefix ++ " end of document",
    };
}

pub fn typeError(
    diagnostics: ?*Diagnostics,
    token: anytype,
    comptime expected: []const u8,
) error{UnexpectedToken} {
    if (diagnostics) |diag| {
        const prefix = "expected: " ++ expected ++ ", found: ";
        const s = coerceToStringDescriptor(token, prefix);
        diag.setMessage(s);
    }
    return error.UnexpectedToken;
}

/// This is an internal function called recursively
/// during the implementation of `parseFromValueLeaky`.
/// It is exposed primarily to enable custom `jsonParseFromValue()` methods to call back into the `parseFromValue*` system,
/// such as when implementing a custom container of type `T`;
/// `innerParse(T, ...)` can be called for each of the container's items.
pub fn innerParseFromValue(
    comptime T: type,
    arena: Allocator,
    source: Value,
    options: ParseOptions,
) ParseFromValueError!T {
    errdefer if (options.diagnostics) |diag| diag.recordContext(@typeName(T));
    switch (@typeInfo(T)) {
        .bool => {
            switch (source) {
                .bool => |b| return b,
                else => |t| return typeError(options.diagnostics, t, "bool"),
            }
        },
        .float, .comptime_float => {
            switch (source) {
                .float => |f| return @as(T, @floatCast(f)),
                .integer => |i| return @as(T, @floatFromInt(i)),
                .number_string, .string => |s| return std.fmt.parseFloat(T, s),
                else => |t| return typeError(options.diagnostics, t, "float"),
            }
        },
        .int, .comptime_int => {
            switch (source) {
                .float => |f| {
                    errdefer if (options.diagnostics) |diag| diag.setMessageFmt("failed to parse. expected: {s} found: {d}", .{ @typeName(T), f });
                    if (@round(f) != f) {
                        return error.InvalidNumber;
                    }
                    if (f > @as(@TypeOf(f), @floatFromInt(std.math.maxInt(T)))) return error.Overflow;
                    if (f < @as(@TypeOf(f), @floatFromInt(std.math.minInt(T)))) return error.Overflow;
                    return @intFromFloat(f);
                },
                .integer => |i| {
                    errdefer if (options.diagnostics) |diag| diag.setMessageFmt("failed to parse. expected: {s} found: {d}", .{ @typeName(T), i });
                    if (i > std.math.maxInt(T)) return error.Overflow;
                    if (i < std.math.minInt(T)) return error.Overflow;
                    return @intCast(i);
                },
                .number_string, .string => |s| {
                    return sliceToInt(T, s, options);
                },
                else => |t| return typeError(options.diagnostics, t, "int"),
            }
        },
        .optional => |optionalInfo| {
            switch (source) {
                .null => return null,
                else => return try innerParseFromValue(optionalInfo.child, arena, source, options),
            }
        },
        .@"enum" => {
            if (std.meta.hasFn(T, "jsonParseFromValue")) {
                return T.jsonParseFromValue(arena, source, options);
            }

            switch (source) {
                .float => {
                    if (options.diagnostics) |diag| diag.setMessageFmt("invalid enum tag, float: {}", .{source});
                    return error.InvalidEnumTag;
                },
                .integer => |i| return std.enums.fromInt(T, i) orelse {
                    if (options.diagnostics) |diag| diag.setMessageFmt("invalid enum tag, integer: {}", .{source});
                    return error.InvalidEnumTag;
                },
                .number_string, .string => |s| return sliceToEnum(T, s, options),
                else => |t| return typeError(options.diagnostics, t, "enum (int or string)"),
            }
        },
        .@"union" => |unionInfo| {
            if (std.meta.hasFn(T, "jsonParseFromValue")) {
                return T.jsonParseFromValue(arena, source, options);
            }

            if (unionInfo.tag_type == null) @compileError("Unable to parse into untagged union '" ++ @typeName(T) ++ "'");

            if (source != .object) {
                if (options.diagnostics) |diag| diag.setMessageFmt("union (object with one field), found: {s}", .{@tagName(source)});
                return error.UnexpectedToken;
            }
            if (source.object.count() != 1) {
                if (options.diagnostics) |diag| diag.setMessage("tagged union requires exactly one field");
                return error.UnexpectedToken;
            }

            var it = source.object.iterator();
            const kv = it.next().?;
            const field_name = kv.key_ptr.*;

            inline for (unionInfo.field_names, unionInfo.field_types) |u_field_name, u_field_type| {
                if (std.mem.eql(u8, u_field_name, field_name)) {
                    if (u_field_type == void) {
                        // void isn't really a json type, but we can support void payload union tags with {} as a value.

                        switch (kv.value_ptr.*) {
                            .object => {},
                            else => |t| return typeError(options.diagnostics, t, "void payload ('{}')"),
                        }

                        if (kv.value_ptr.*.object.count() != 0) {
                            if (options.diagnostics) |diag| diag.setMessage("void payload '{}' should have no fields");
                            return error.UnexpectedToken;
                        }
                        return @unionInit(T, u_field_name, {});
                    }
                    // Recurse.
                    return @unionInit(T, u_field_name, try innerParseFromValue(u_field_type, arena, kv.value_ptr.*, options));
                }
            }
            // Didn't match anything.
            if (options.diagnostics) |diag| diag.setMessage("unrecognized tag name");
            return error.UnknownField;
        },

        .@"struct" => |structInfo| {
            if (structInfo.is_tuple) {
                switch (source) {
                    .array => {},
                    else => |t| return typeError(options.diagnostics, t, "tuple (array of values)"),
                }

                if (source.array.items.len != structInfo.field_names.len) {
                    if (options.diagnostics) |diag| diag.setMessageFmt("tuple length doesn't match expected length. expected: {} found: {}", .{ structInfo.field_names.len, source.array.items.len });
                    return error.UnexpectedToken;
                }

                var r: T = undefined;
                inline for (0..structInfo.field_names.len, source.array.items) |i, item| {
                    r[i] = try innerParseFromValue(structInfo.field_types[i], arena, item, options);
                }

                return r;
            }

            if (std.meta.hasFn(T, "jsonParseFromValue")) {
                return T.jsonParseFromValue(arena, source, options);
            }

            switch (source) {
                .object => {},
                else => |t| return typeError(options.diagnostics, t, "struct"),
            }

            var r: T = undefined;
            var fields_seen: [structInfo.field_names.len]bool = @splat(false);

            var it = source.object.iterator();
            while (it.next()) |kv| {
                const field_name = kv.key_ptr.*;

                inline for (
                    structInfo.field_names,
                    structInfo.field_types,
                    structInfo.field_attrs,
                    0..,
                ) |f_name, f_type, f_attrs, i| {
                    if (f_attrs.@"comptime") @compileError("comptime fields are not supported: " ++ @typeName(T) ++ "." ++ f_name);
                    if (std.mem.eql(u8, f_name, field_name)) {
                        errdefer if (options.diagnostics) |diag| diag.recordContext(@typeName(T) ++ "." ++ f_name);
                        assert(!fields_seen[i]); // Can't have duplicate keys in a Value.object.
                        @field(r, f_name) = try innerParseFromValue(f_type, arena, kv.value_ptr.*, options);
                        fields_seen[i] = true;
                        break;
                    }
                } else {
                    // Didn't match anything.
                    if (!options.ignore_unknown_fields) {
                        if (options.diagnostics) |diag| diag.setMessageFmt("found unknown field. found: {s}", .{field_name});
                        return error.UnknownField;
                    }
                }
            }
            try fillDefaultStructValues(T, options, &r, &fields_seen);
            return r;
        },

        .array => |arrayInfo| {
            switch (source) {
                .array => |array| {
                    // Typical array.
                    return innerParseArrayFromArrayValue(T, arrayInfo.child, arrayInfo.len, arena, array, options);
                },
                .string => |s| {
                    if (arrayInfo.child != u8) return typeError(options.diagnostics, source, "array");
                    // Fixed-length string.

                    if (s.len != arrayInfo.len) {
                        if (options.diagnostics) |diag| diag.setMessageFmt("Length mismatch for fixed-length string. Found: {}. Expected: {}", .{ s.len, arrayInfo.len });
                        return error.LengthMismatch;
                    }

                    var r: T = undefined;
                    @memcpy(r[0..], s);
                    return r;
                },
                else => |t| return typeError(options.diagnostics, t, if (arrayInfo.child == u8) "array or fixed-length string" else "array"),
            }
        },

        .vector => |vecInfo| {
            switch (source) {
                .array => |array| {
                    return innerParseArrayFromArrayValue(T, vecInfo.child, vecInfo.len, arena, array, options);
                },
                else => |t| return typeError(options.diagnostics, t, "array"),
            }
        },

        .pointer => |ptrInfo| {
            switch (ptrInfo.size) {
                .one => {
                    const r: *ptrInfo.child = try arena.create(ptrInfo.child);
                    r.* = try innerParseFromValue(ptrInfo.child, arena, source, options);
                    return r;
                },
                .slice => {
                    switch (source) {
                        .array => |array| {
                            const r = if (ptrInfo.sentinel()) |sentinel|
                                try arena.allocSentinel(ptrInfo.child, array.items.len, sentinel)
                            else
                                try arena.alloc(ptrInfo.child, array.items.len);

                            for (array.items, r) |item, *dest| {
                                dest.* = try innerParseFromValue(ptrInfo.child, arena, item, options);
                            }

                            return r;
                        },
                        .string => |s| {
                            if (ptrInfo.child != u8) return typeError(options.diagnostics, source, "array");
                            // Dynamic length string.

                            const r = if (ptrInfo.sentinel()) |sentinel|
                                try arena.allocSentinel(ptrInfo.child, s.len, sentinel)
                            else
                                try arena.alloc(ptrInfo.child, s.len);
                            @memcpy(r[0..], s);

                            return r;
                        },
                        else => |t| return typeError(options.diagnostics, t, if (ptrInfo.child == u8) "string" else "array"),
                    }
                },
                else => @compileError("Unable to parse into type '" ++ @typeName(T) ++ "'"),
            }
        },
        else => @compileError("Unable to parse into type '" ++ @typeName(T) ++ "'"),
    }
}

fn innerParseArrayFromArrayValue(
    comptime T: type,
    comptime Child: type,
    comptime len: comptime_int,
    arena: Allocator,
    array: Array,
    options: ParseOptions,
) !T {
    if (array.items.len != len) {
        if (options.diagnostics) |diag| diag.setMessageFmt("array length mismatch. expected: {}, found: {}", .{ len, array.items.len });
        return error.LengthMismatch;
    }

    var r: T = undefined;
    for (array.items, 0..) |item, i| {
        r[i] = try innerParseFromValue(Child, arena, item, options);
    }

    return r;
}

fn sliceToInt(comptime T: type, slice: []const u8, options: ParseOptions) !T {
    if (isNumberFormattedLikeAnInteger(slice))
        return std.fmt.parseInt(T, slice, 10) catch |err| {
            if (options.diagnostics) |diag| diag.setMessageFmt("failed to parse. expected: {s} found: {s}", .{ @typeName(T), slice });
            return err;
        };
    // Try to coerce a float to an integer.
    const float = std.fmt.parseFloat(f128, slice) catch |err| {
        if (options.diagnostics) |diag| diag.setMessageFmt("failed to parse. expected: {s} found: {s}", .{ @typeName(T), slice });
        return err;
    };
    if (@round(float) != float) {
        if (options.diagnostics) |diag| diag.setMessageFmt("failed to parse. expected: {s} found: {s}", .{ @typeName(T), slice });
        return error.InvalidNumber;
    }
    if (float > @as(f128, @floatFromInt(std.math.maxInt(T))) or float < @as(f128, @floatFromInt(std.math.minInt(T)))) {
        if (options.diagnostics) |diag| diag.setMessageFmt("failed to parse. expected: {s} found: {s}", .{ @typeName(T), slice });
        return error.Overflow;
    }
    return @as(T, @intCast(@as(i128, @intFromFloat(float))));
}

fn sliceToEnum(comptime T: type, slice: []const u8, options: ParseOptions) !T {
    // Check for a named value.
    if (std.meta.stringToEnum(T, slice)) |value| return value;
    // Check for a numeric value.
    if (!isNumberFormattedLikeAnInteger(slice)) {
        if (options.diagnostics) |diag| diag.setMessageFmt("invalid enum tag, number not formatted as integer, slice: {s}", .{slice});
        return error.InvalidEnumTag;
    }
    const n = std.fmt.parseInt(@typeInfo(T).@"enum".tag_type, slice, 10) catch {
        if (options.diagnostics) |diag| diag.setMessageFmt("invalid enum tag, failed to parse int from enum tag type, slice: {s}", .{slice});
        return error.InvalidEnumTag;
    };
    return std.enums.fromInt(T, n) orelse {
        if (options.diagnostics) |diag| diag.setMessageFmt("invalid enum tag, failed to parse int from enum, slice: {s}", .{slice});
        return error.InvalidEnumTag;
    };
}

fn fillDefaultStructValues(comptime T: type, options: ParseOptions, r: *T, fields_seen: *[@typeInfo(T).@"struct".field_names.len]bool) !void {
    const info = @typeInfo(T).@"struct";
    inline for (
        info.field_names,
        info.field_types,
        info.field_attrs,
        0..,
    ) |field_name, field_type, field_attrs, i| {
        if (!fields_seen[i]) {
            if (field_attrs.defaultValue(field_type)) |default| {
                @field(r, field_name) = default;
            } else {
                if (options.diagnostics) |diag| diag.setMessage("missing field: " ++ @typeName(T) ++ "." ++ field_name);
                return error.MissingField;
            }
        }
    }
}

fn freeAllocated(allocator: Allocator, token: Token) void {
    switch (token) {
        .allocated_number, .allocated_string => |slice| {
            allocator.free(slice);
        },
        else => {},
    }
}

test {
    _ = @import("./static_test.zig");
}
