const std = @import("std");
const Ast = std.zig.Ast;

const compiler = @import("compiler");

const Asta = @import("analysis.zig");
const ZigDoc = @import("ZigDoc.zig");

const ErrSet = error{OutOfMemory};

pub fn resolveVarDecl(
    arena: std.mem.Allocator,
    io: std.Io,
    zdoc: *ZigDoc,
    node: Ast.Node.Index,
) ErrSet!?[]const u8 {
    const tree = zdoc.tree;

    switch (tree.nodeTag(node)) {
        .global_var_decl,
        .local_var_decl,
        .aligned_var_decl,
        .simple_var_decl,
        => {},
        else => return null,
    }

    const full_var_decl = tree.fullVarDecl(node).?;

    std.log.err("hovering over: {} with init node {}", .{ node, full_var_decl.ast.init_node.unwrap().? });
    const doc_scope = try zdoc.getDocumentScope();
    const fn_scope = Asta.innermostScopeAtIndexWithTag(doc_scope, tree.tokens.items(.start)[full_var_decl.firstToken()], .initOne(.function)).unwrap() orelse return null;
    const fn_node = doc_scope.getScopeAstNode(fn_scope) orelse return null;

    var buf: [1]Ast.Node.Index = undefined;
    const fn_proto_node = tree.fullFnProto(&buf, fn_node).?.ast.proto_node;

    zdoc.computed_data.lock.lockSharedUncancelable(io);
    defer zdoc.computed_data.lock.unlockShared(io);

    const build = zdoc.computed_data.build orelse return null;

    if (!build.mutex.tryLock()) return null;
    defer build.mutex.unlock(io);

    if (!build.has_completed_once) return null;

    const args = zdoc.computed_data.air.get(fn_proto_node) orelse return null;
    const zcu = build.compilation.?.zcu orelse return null;

    const active = zcu.activate(args.tid);
    defer active.deactivate();
    const pt = active.pt;

    for (args.air.getMainBody()) |inst| {
        const tag = args.air.instructions.items(.tag)[@backingInt(inst)];
        switch (tag) {
            else => continue,
            .dbg_var_ptr,
            .dbg_var_val,
            // .dbg_arg_inline,
            => {
                const pl_op = args.air.instructions.items(.data)[@backingInt(inst)].pl_op;
                // TODO Check that the identifier matches as well
                if (pl_op.tree_data_index == @backingInt(full_var_decl.ast.init_node.unwrap().?)) {
                    std.log.err("found a match! {}", .{pl_op.tree_data_index});
                    return try resolveInst(arena, args.air, inst, pt);
                }
            },
        }
    }

    return null;
}

fn resolveInst(
    arena: std.mem.Allocator,
    air: compiler.Compilation.Air,
    instruction: compiler.Compilation.Air.Inst.Index,
    pt: compiler.Compilation.Zcu.PerThread,
) ErrSet!?[]const u8 {
    const tags = air.instructions.items(.tag);
    const datas = air.instructions.items(.data);

    var aw: std.Io.Writer.Allocating = .init(arena);
    defer aw.deinit();

    var inst = instruction;
    while (true) {
        switch (tags[@backingInt(inst)]) {
            else => return null,
            .dbg_var_ptr,
            .dbg_var_val,
            // .dbg_arg_inline,
            => {
                const pl_op = datas[@backingInt(inst)].pl_op;
                if (@backingInt(pl_op.operand) < compiler.Compilation.InternPool.static_len) {
                    std.log.err("@{}", .{pl_op.operand});
                } else if (pl_op.operand.toInterned()) |ip_index| {
                    const ty = compiler.Compilation.Type.fromInterned(pt.zcu.intern_pool.indexToKey(ip_index).typeOf());
                    std.log.err("<{f}, {f}>", .{
                        ty.fmt(pt),
                        compiler.Compilation.Value.fromInterned(ip_index).fmtValue(pt),
                    });
                } else {
                    if (pl_op.operand.toIndex()) |sub_inst| {
                        inst = sub_inst;
                        continue;
                    } else {
                        std.log.err("unknown", .{});
                        return null;
                    }
                }
            },
            .alloc => {
                const ty = datas[@backingInt(inst)].ty;
                ty.print(&aw.writer, pt, null) catch return null;
                std.log.err("{s}", .{aw.written()[1..]});
                return (try aw.toOwnedSlice())[1..];
            },
            .load => {
                const ty_op = datas[@backingInt(inst)].ty_op;
                ty_op.ty.print(&aw.writer, pt, null) catch return null;
                std.log.err("{s}", .{aw.written()});
                return try aw.toOwnedSlice();
            },
            .call => {
                const data = datas[@backingInt(inst)];
                std.log.err("data: {}", .{data.pl_op.operand});
                if (data.pl_op.operand.toInterned()) |ip_index| {
                    const ty = compiler.Compilation.Type.fromInterned(pt.zcu.intern_pool.indexToKey(ip_index).typeOf());
                    switch (pt.zcu.intern_pool.indexToKey(ty.ip_index)) {
                        else => {},
                        .func_type => |ft| {
                            std.log.err("fn ret ty itk: {}", .{pt.zcu.intern_pool.indexToKey(ft.return_type)});
                            switch (pt.zcu.intern_pool.indexToKey(ft.return_type)) {
                                else => {},
                                .struct_type => {
                                    const let = pt.zcu.intern_pool.loadStructType(ft.return_type);
                                    for (let.field_names.get(&pt.zcu.intern_pool)) |field_name| {
                                        std.log.err("name: {q}", .{field_name.toSlice(&pt.zcu.intern_pool)});
                                    }
                                },
                                .ptr_type => |pty| {
                                    std.log.err("pty child itk: {}", .{pt.zcu.intern_pool.indexToKey(pty.child)});
                                    switch (pt.zcu.intern_pool.indexToKey(pty.child)) {
                                        else => {},
                                        .struct_type => {
                                            const let = pt.zcu.intern_pool.loadStructType(pty.child);
                                            for (let.field_names.get(&pt.zcu.intern_pool)) |field_name| {
                                                std.log.err("name: {q}", .{field_name.toSlice(&pt.zcu.intern_pool)});
                                            }
                                        },
                                        .enum_type => {
                                            const let = pt.zcu.intern_pool.loadEnumType(pty.child);
                                            for (let.field_names.get(&pt.zcu.intern_pool)) |field_name| {
                                                std.log.err("name: {q}", .{field_name.toSlice(&pt.zcu.intern_pool)});
                                            }
                                        },
                                    }
                                },
                            }
                            aw.writer.print("{f}", .{compiler.Compilation.Type.fromInterned(ft.return_type).fmt(pt)}) catch return null;
                            return try aw.toOwnedSlice();
                        },
                    }
                    std.log.err("<{f}, {f}>", .{
                        ty.fmt(pt),
                        compiler.Compilation.Value.fromInterned(ip_index).fmtValue(pt),
                    });
                }
                return null;
            },
        }
    }
}
