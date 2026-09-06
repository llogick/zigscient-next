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

    // std.log.err("hovering over: {} with init node {}", .{ node, full_var_decl.ast.init_node.unwrap().? });
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

    if (matchVarDecl(full_var_decl.ast.init_node.unwrap().?, args.air, args.air.getMainBody())) |inst| return try resolveInst(arena, args.air, inst, pt);
    return null;
}

fn matchVarDecl(
    node: std.zig.Ast.Node.Index,
    air: compiler.Compilation.Air,
    instructions: []const compiler.Compilation.Air.Inst.Index,
) ?compiler.Compilation.Air.Inst.Index {
    for (instructions) |inst| {
        const tag = air.instructions.items(.tag)[@backingInt(inst)];
        switch (tag) {
            else => continue,
            .dbg_var_ptr,
            .dbg_var_val,
            // .dbg_arg_inline,
            => {
                const pl_op = air.instructions.items(.data)[@backingInt(inst)].pl_op;
                // TODO Check that the identifier matches as well
                // std.log.err("comparing: {} with {}", .{ pl_op.tree_data_index, @backingInt(node) });
                if (pl_op.tree_data_index == @backingInt(node)) {
                    std.log.err("found a match! {}", .{pl_op.tree_data_index});
                    return inst;
                }
            },
            .loop,
            .block,
            => {
                if (matchVarDecl(node, air, air.unwrapBlock(inst).body)) |i| return i;
            },
            .dbg_inline_block => {
                if (matchVarDecl(node, air, air.unwrapDbgBlock(inst).body)) |i| return i;
            },
            .cond_br => {
                const cond_br = air.unwrapCondBr(inst);
                if (matchVarDecl(node, air, cond_br.then_body)) |i| return i;
                if (matchVarDecl(node, air, cond_br.else_body)) |i| return i;
            },
            .loop_switch_br,
            .switch_br,
            => {
                const switch_br = air.unwrapSwitch(inst);
                var it = switch_br.iterateCases();
                while (it.next()) |case| if (matchVarDecl(node, air, case.body)) |i| return i;
                if (matchVarDecl(node, air, it.elseBody())) |i| return i;
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
        const tag = tags[@backingInt(inst)];
        std.log.err("tag: {t}", .{tags[@backingInt(inst)]});
        switch (tag) {
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
                    dumpFields(pt, ty.ip_index);
                    aw.writer.print("{f}", .{ty.fmt(pt)}) catch return null;
                    return try aw.toOwnedSlice();
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
                                .struct_type => dumpFields(pt, ft.return_type),
                                .ptr_type => |pty| {
                                    std.log.err("pty child itk: {}", .{pt.zcu.intern_pool.indexToKey(pty.child)});
                                    switch (pt.zcu.intern_pool.indexToKey(pty.child)) {
                                        else => {},
                                        .enum_type,
                                        .struct_type,
                                        => dumpFields(pt, pty.child),
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
            .block,
            .dbg_inline_block,
            => {
                const ty_pl = datas[@backingInt(inst)].ty_pl;
                ty_pl.ty.print(&aw.writer, pt, null) catch return null;
                std.log.err("{s}", .{aw.written()});
                return try aw.toOwnedSlice();
            },
        }
    }
}

fn dumpFields(
    pt: compiler.Compilation.Zcu.PerThread,
    index: compiler.Compilation.InternPool.Index,
) void {
    var idx = index;
    sw: switch (pt.zcu.intern_pool.indexToKey(idx)) {
        else => {},
        .struct_type => {
            const let = pt.zcu.intern_pool.loadStructType(idx);
            for (let.field_names.get(&pt.zcu.intern_pool), let.field_types.get(&pt.zcu.intern_pool)) |field_name, field_type_index| {
                const ty = compiler.Compilation.Type.fromInterned(field_type_index);
                std.log.err("{s} : {f}", .{ field_name.toSlice(&pt.zcu.intern_pool), ty.fmt(pt) });
            }
        },
        .enum_type => {
            const let = pt.zcu.intern_pool.loadEnumType(idx);
            for (let.field_names.get(&pt.zcu.intern_pool), 0..) |field_name, i| {
                const field_values = let.field_values.get(&pt.zcu.intern_pool);
                if (i < field_values.len) {
                    const val = compiler.Compilation.Value.fromInterned(field_values[i]);
                    std.log.err("{s} = {f}", .{ field_name.toSlice(&pt.zcu.intern_pool), val.fmtValue(pt) });
                } else std.log.err("{s}", .{field_name.toSlice(&pt.zcu.intern_pool)});
            }
        },
        .union_type => {
            const let = pt.zcu.intern_pool.loadUnionType(idx);
            idx = let.enum_tag_type;
            continue :sw pt.zcu.intern_pool.indexToKey(let.enum_tag_type);
        },
    }
}
