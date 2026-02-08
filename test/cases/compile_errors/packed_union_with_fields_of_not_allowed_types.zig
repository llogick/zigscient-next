export fn entry0() void {
    _ = @sizeOf(packed union {
        foo: struct { a: u32 },
        bar: bool,
    });
}
export fn entry1() void {
    _ = @sizeOf(packed union {
        x: *const u32,
    });
}

// error
//
// :3:14: error: packed unions cannot contain fields of type 'packed_union_with_fields_of_not_allowed_types.entry0__union_180__struct_182'
// :3:14: note: non-packed structs do not have a bit-packed representation
// :3:14: note: struct declared here
// :9:12: error: packed unions cannot contain fields of type '*const u32'
// :9:12: note: pointers cannot be directly bitpacked
// :9:12: note: consider using 'usize' and '@intFromPtr'
