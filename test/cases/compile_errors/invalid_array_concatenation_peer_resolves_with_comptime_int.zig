comptime {
    const lhs: [2]void = .{ {}, {} };
    const rhs: [2]comptime_int = .{ 2, 3 };
    _ = lhs ++ rhs;
}

// error
//
// :4:13: error: incompatible types: 'void' and 'comptime_int'
// :4:9: note: type 'void' here
// :4:16: note: type 'comptime_int' here
