const builtin = @import("builtin");
const std = @import("../../std.zig");

const assert = std.debug.assert;

pub const getprop_path = "/system/bin/getprop";

pub fn detectApiLevel(
    io: std.Io,
    arena: std.mem.Allocator,
    diags: *std.zig.system.ResolveTargetDiagnostics,
) std.zig.system.ResolveTargetError!?u32 {
    assert(builtin.target.os.tag == .linux);

    var child = std.process.spawn(io, .{
        .argv = &.{
            getprop_path,
            "ro.build.version.sdk",
        },
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .ignore,
    }) catch |err| switch (err) {
        error.ProcessAlreadyExec,
        error.InvalidUserId,
        error.InvalidProcessGroupId,
        => unreachable,

        error.InvalidWtf8,
        error.InvalidBatchScriptArg,
        error.UnrecognizedVolume,
        error.InvalidName,
        error.PipeBusy,
        error.NetworkNotFound,
        error.AntivirusInterference,
        => unreachable, // Windows-only

        error.OutOfMemory,
        error.SystemResources,
        error.SystemFdQuotaExceeded,
        error.ProcessFdQuotaExceeded,
        error.ResourceLimitReached,
        error.Canceled,
        => |e| return e,

        error.IsDir,
        error.WouldBlock,
        error.AccessDenied,
        error.FileTooBig,
        error.NoSpaceLeft,
        error.DeviceBusy,
        error.PermissionDenied,
        error.NoDevice,
        error.FileBusy,
        error.PathAlreadyExists,
        error.SymLinkLoop,
        error.FileNotFound,
        error.NotDir,
        error.ReadOnlyFileSystem,
        error.NameTooLong,
        error.BadPathName,
        error.FileLocksUnsupported,
        error.OperationUnsupported,
        error.FileSystem,
        error.InvalidExe,
        error.Unexpected,
        => |e| {
            diags.android_api_level = .{ .spawn_failed = e };
            return null;
        },
    };
    defer child.kill(io);

    // `PROP_VALUE_MAX` is 92, output is value + newline, plus one byte so we can tell if there is
    // more output. Currently API levels are two-digit numbers, but we want to make sure we never
    // read a partial value.
    var stdout_buf: [92 + 1 + 1]u8 = undefined;
    var reader = child.stdout.?.readerStreaming(io, &.{});
    const n = reader.interface.readSliceShort(&stdout_buf) catch |err| switch (err) {
        error.ReadFailed => switch (reader.err.?) {
            error.LockViolation => unreachable, // Windows-only

            error.SystemResources,
            error.Canceled,
            => |e| return e,

            error.InputOutput,
            error.IsDir,
            error.AccessDenied,
            error.WouldBlock,
            error.ConnectionResetByPeer,
            error.NotOpenForReading,
            error.SocketUnconnected,
            error.Unexpected,
            => |e| {
                diags.android_api_level = .{ .read_error = e };
                return null;
            },
        },
    };

    // Anything longer is definitely not a property value, so just give up.
    if (n == stdout_buf.len) {
        diags.android_api_level = .{ .invalid_output = try arena.dupe(u8, &stdout_buf) };
        return null;
    }

    const term = child.wait(io) catch |err| switch (err) {
        error.Canceled => |e| return e,
        error.Unexpected => |e| {
            diags.android_api_level = .{ .wait_failed = e };
            return null;
        },
    };

    if (!term.eql(.success)) {
        diags.android_api_level = .{ .bad_term = term };
        return null;
    }

    const output = std.mem.trimEnd(u8, stdout_buf[0..n], "\n");
    return std.fmt.parseInt(u32, output, 10) catch |err| switch (err) {
        error.Overflow,
        error.InvalidCharacter,
        => {
            diags.android_api_level = .{ .invalid_output = try arena.dupe(u8, output) };
            return null;
        },
    };
}
