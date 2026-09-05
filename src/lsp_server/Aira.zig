const std = @import("std");
const Ast = std.zig.Ast;

const compiler = @import("compiler");

const Asta = @import("analysis.zig");
const ZigDoc = @import("ZigDoc.zig");

const ErrSet = error{OutOfMemory};

pub fn resolveVarDecl(io: std.Io, zdoc: *ZigDoc, node: Ast.Node.Index) ErrSet!?void {
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

    for (args.air.getMainBody()) |inst| {
        const tag = args.air.instructions.items(.tag)[@backingInt(inst)];
        switch (tag) {
            else => continue,
            .dbg_var_ptr,
            .dbg_var_val,
            // .dbg_arg_inline,
            => {
                const pl_op = args.air.instructions.items(.data)[@backingInt(inst)].pl_op;
                if (pl_op.tree_data_index == @backingInt(full_var_decl.ast.init_node.unwrap().?)) {
                    std.log.err("found a match! {}", .{pl_op.tree_data_index});
                    if (@backingInt(pl_op.operand) < compiler.Compilation.InternPool.static_len) {
                        std.log.err("@{}", .{pl_op.operand});
                    } else if (pl_op.operand.toInterned()) |ip_index| {
                        const active = zcu.activate(args.tid);
                        defer active.deactivate();
                        const pt = active.pt;
                        const ty = compiler.Compilation.Type.fromInterned(pt.zcu.intern_pool.indexToKey(ip_index).typeOf());
                        std.log.err("<{f}, {f}>", .{
                            ty.fmt(pt),
                            compiler.Compilation.Value.fromInterned(ip_index).fmtValue(pt),
                        });
                    } else {
                        if (pl_op.operand.toIndex()) |sub_inst| {
                            const sub_inst_tag = args.air.instructions.items(.tag)[@backingInt(sub_inst)];
                            std.log.err("tag: {}", .{sub_inst_tag});
                            switch (sub_inst_tag) {
                                else => {},
                                .alloc => {
                                    const active = zcu.activate(args.tid);
                                    defer active.deactivate();
                                    const pt = active.pt;
                                    const ty = args.air.instructions.items(.data)[@backingInt(sub_inst)].ty;
                                    var buffer: [4096]u8 = undefined;
                                    var aw: std.Io.Writer = .fixed(&buffer);
                                    ty.print(&aw, pt, null) catch @memset(&buffer, 'a');
                                    std.log.err("{s}", .{aw.buffered()[1..]});
                                },
                                .load => {
                                    const active = zcu.activate(args.tid);
                                    defer active.deactivate();
                                    const pt = active.pt;
                                    const ty_op = args.air.instructions.items(.data)[@backingInt(sub_inst)].ty_op;
                                    var buffer: [4096]u8 = undefined;
                                    var aw: std.Io.Writer = .fixed(&buffer);
                                    ty_op.ty.print(&aw, pt, null) catch @memset(&buffer, 'a');
                                    std.log.err("{s}", .{aw.buffered()});
                                },
                                .call => {
                                    const data = args.air.instructions.items(.data)[@backingInt(sub_inst)];
                                    std.log.err("data: {}", .{data.pl_op.operand});
                                    if (data.pl_op.operand.toInterned()) |ip_index| {
                                        const active = zcu.activate(args.tid);
                                        defer active.deactivate();
                                        const pt = active.pt;
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
                                            },
                                        }
                                        std.log.err("<{f}, {f}>", .{
                                            ty.fmt(pt),
                                            compiler.Compilation.Value.fromInterned(ip_index).fmtValue(pt),
                                        });
                                    }
                                    //
                                },
                            }
                        } else std.log.err("unknonw", .{});
                    }
                    break;
                }
            },
        }
    }
}
