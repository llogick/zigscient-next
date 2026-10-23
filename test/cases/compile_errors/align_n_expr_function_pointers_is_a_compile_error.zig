export fn foo() align(2) void {
    return;
}

// error
// target=wasm32-freestanding-none
//
// :1:23: error: target does not support function alignment greater than 1
