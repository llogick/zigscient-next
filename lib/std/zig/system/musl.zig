const builtin = @import("builtin");
const std = @import("../../std.zig");

const assert = std.debug.assert;

pub fn detectVersion(
    io: std.Io,
    arena: std.mem.Allocator,
    dynamic_linker: []const u8,
    diags: *std.zig.system.ResolveTargetDiagnostics,
) std.zig.system.ResolveTargetError!?std.SemanticVersion {
    assert(builtin.target.os.tag == .linux);
    assert(dynamic_linker.len <= std.Target.DynamicLinker.max_len);

    var child = std.process.spawn(io, .{
        .argv = &.{dynamic_linker},
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .pipe,
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
            diags.libc_version = .{ .spawn_failed = .{ .path = try arena.dupe(u8, dynamic_linker), .err = e } };
            return null;
        },
    };
    defer child.kill(io);

    var stderr_buf: [256 + std.Target.DynamicLinker.max_len]u8 = undefined; // usage text + executable name
    var reader = child.stderr.?.readerStreaming(io, &.{});
    const n = reader.interface.readSliceShort(&stderr_buf) catch |err| switch (err) {
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
                diags.libc_version = .{ .read_error = .{ .path = try arena.dupe(u8, dynamic_linker), .err = e } };
                return null;
            },
        },
    };

    // More output being available suggests that this is not actually a musl dynamic linker, in
    // which case we should just give up rather than wait for it to exit gracefully.
    if (n == stderr_buf.len) {
        diags.libc_version = .{ .invalid_output = .{ .path = try arena.dupe(u8, dynamic_linker), .output = try arena.dupe(u8, &stderr_buf) } };
        return null;
    }

    const term = child.wait(io) catch |err| switch (err) {
        error.Canceled => |e| return e,
        error.Unexpected => |e| {
            diags.libc_version = .{ .wait_failed = .{ .path = try arena.dupe(u8, dynamic_linker), .err = e } };
            return null;
        },
    };

    // The exit code will be 1 for this usage, but we don't actually care about the exact value.
    if (term != .exited) {
        diags.libc_version = .{ .bad_term = .{ .path = try arena.dupe(u8, dynamic_linker), .term = term } };
        return null;
    }

    const usage = stderr_buf[0..n];
    return b: {
        var lines = std.mem.splitScalar(u8, usage, '\n');
        if (!std.mem.startsWith(u8, lines.first(), "musl libc (")) break :b null;

        const prefix = "Version ";
        const version_line = lines.next() orelse break :b null;
        if (!std.mem.startsWith(u8, version_line, prefix)) break :b null;

        // Examples:
        //
        // * `1.2.5-git-12-gabcdef01`
        // * `1.2.5-git`
        // * `1.2.5`
        const version_text = version_line[prefix.len..];
        const end = std.mem.findNone(u8, version_text, "0123456789.") orelse version_text.len;
        break :b std.Target.Query.parseVersion(version_text[0..end]) catch |err| switch (err) {
            error.Overflow,
            error.InvalidVersion,
            => null,
        };
    } orelse {
        diags.libc_version = .{ .invalid_output = .{ .path = try arena.dupe(u8, dynamic_linker), .output = try arena.dupe(u8, usage) } };
        return null;
    };
}
