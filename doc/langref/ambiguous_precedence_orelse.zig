test "ambiguous precedence" {
    const x: ?bool = true;
    const y: bool = false;
    const z: bool = false;

    // The following expression is always considered ambiguous,
    // and, in this particular case, `(x orelse y) == z` would
    // produce a different result than `x orelse (y == z)`.
    _ = x orelse y == z;
}

// obj=ambiguous operator precedence with operators ... orelse ... == ...; use parentheses to disambiguate
