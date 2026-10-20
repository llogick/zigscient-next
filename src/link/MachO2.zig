const MachO = @This();

base: link.File,
options: link.File.OpenOptions,

const_prog_node: std.Progress.Node,
synth_prog_node: std.Progress.Node,
input_prog_node: std.Progress.Node,

mf: MappedFile,
nodes: std.MultiArrayList(Node),

node: struct {
    macho: MappedFile.Node.Index,
    header: MappedFile.Node.Index,
    symtab: MappedFile.Node.Index,
    strtab: MappedFile.Node.Index,
    got_indirect_syms: MappedFile.Node.Index,
    stubs_indirect_syms: MappedFile.Node.Index,
    fixup_chains_header: MappedFile.Node.Index,
    fixup_chains_imports: MappedFile.Node.Index,
    fixup_chains_starts_in_image: MappedFile.Node.Index,
    fixup_chains_symbol_names: MappedFile.Node.Index,
},

lc: struct {
    offsets: std.ArrayList(u32),
    pagezero: LoadCommand,
    symtab: LoadCommand,
    dysymtab: LoadCommand,
    entry_point: ?LoadCommand,
},
known_sections: struct {
    got: Section,
    stubs: Section,
},
/// Does not contain an entry for the `__PAGEZERO` segment.
segments: std.MultiArrayList(struct {
    lc: LoadCommand,
    node: MappedFile.Node.Index,
    fixup_chain_starts: MappedFile.Node.Index.Optional,
}),
sections: std.MultiArrayList(struct {
    lc: LoadCommand,
    node: MappedFile.Node.Index,
    segment: Segment,
    index: u8,
}),
/// Accessed with `Section.NameAdapter` using `Section.NameAdapter.SegmentAndSection`.
sections_by_name: std.array_hash_map.Custom(void, void, void, true),

dirty_fixup_chain_pages: std.array_hash_map.Auto(struct { seg: Segment, page: u16 }, void),

input_atoms: std.ArrayList(InputAtom),

/// Accessed with `String.Adapter` using `[]const u8` keys.
strtab: std.array_hash_map.Custom(String, void, void, true),

symtab: std.ArrayList(Symbol),

/// Contains all symbols which can be referenced by name. Mach-O generally calls these "external"
/// symbols, but I'm avoiding that name in favour of ELF's term "global symbol", because the Mach-O
/// term is needlessly confusing (they may be defined locally!).
///
/// Accessed with `Symbol.GlobalAdapter` using `[]const u8` keys.
global_symbols: std.array_hash_map.Custom(Symbol.Index, void, void, true),

relocs: std.ArrayList(Reloc),
overflowed_reloc_count: u32,
misaligned_reloc_count: u32,

imports: std.array_hash_map.Auto(Symbol.Index, void),
fixup_chain_symbol_names_len: u32,
got: std.array_hash_map.Auto(Symbol.Index, void),
stubs: std.array_hash_map.Auto(Symbol.Index, void),

sections_changed_index: std.array_hash_map.Auto(Section, void),

/// The file offset of the `__got` section relative to its parent segment at the time that fixups
/// for the GOT were last marked dirty. We track this so that when the GOT moves, we know its
/// previous location and can invalidate those fixup pages too (to clear the old fixup entries).
prev_got_offset: u32,

const Error = link.Error || error{MappedFileIo};

/// Stable reference to a particular load command (or `std.macho.section_64`, despite those not
/// technically being load commands). Underlying value is an index into `MachO.lc.offsets`.
const LoadCommand = enum(u32) {
    _,
    fn offset(lc: LoadCommand, macho: *const MachO) u32 {
        return macho.lc.offsets.items[@backingInt(lc)];
    }
    fn ptrConst(lc: LoadCommand, macho: *const MachO, comptime Cmd: type) *align(8) const Cmd {
        return @ptrCast(@alignCast(
            macho.node.header.sliceConst(&macho.mf)[lc.offset(macho)..][0..@sizeOf(Cmd)],
        ));
    }
    fn ptr(lc: LoadCommand, macho: *MachO, comptime Cmd: type) *align(8) Cmd {
        return @ptrCast(@alignCast(
            macho.node.header.slice(&macho.mf)[lc.offset(macho)..][0..@sizeOf(Cmd)],
        ));
    }
};

/// Stable reference to a segment defined by a `std.macho.segment_command_64` load command.
///
/// This type is backed by `u8` instead of a larger type because Mach-O itself uses 8-bit indices to
/// refer to sections, so the limitation exists regardless.
const Segment = enum(u8) {
    /// Aside from `__PAGEZERO` (which we don't assign a `Segment`), the `__TEXT`  segment must come
    /// first in memory. This segment is special: it must contain the Mach header, and some offsets
    /// are relative to its vmaddr. So, as it's special and always exists, we give it a named tag!
    text,
    _,

    fn lc(seg: Segment, macho: *const MachO) LoadCommand {
        return macho.segments.items(.lc)[@backingInt(seg)];
    }
    fn lcPtrConst(seg: Segment, macho: *const MachO) *const std.macho.segment_command_64 {
        return seg.lc(macho).ptrConst(macho, std.macho.segment_command_64);
    }
    fn lcPtr(seg: Segment, macho: *MachO) *std.macho.segment_command_64 {
        return seg.lc(macho).ptr(macho, std.macho.segment_command_64);
    }

    fn name(seg: Segment, macho: *const MachO) []const u8 {
        return std.mem.sliceTo(&seg.lcPtrConst(macho).segname, 0);
    }

    fn vaddr(seg: Segment, macho: *const MachO) u64 {
        return macho.targetLoad(&seg.lcPtrConst(macho).vmaddr);
    }

    fn node(seg: Segment, macho: *const MachO) MappedFile.Node.Index {
        return macho.segments.items(.node)[@backingInt(seg)];
    }

    fn fixupChainStartsNode(seg: Segment, macho: *const MachO) MappedFile.Node.Index.Optional {
        return macho.segments.items(.fixup_chain_starts)[@backingInt(seg)];
    }
};

/// Stable reference to a section defined by a `std.macho.section_64` in the load commands.
///
/// This type is backed by `u8` instead of a larger type because Mach-O itself uses 8-bit indices to
/// refer to sections, so the limitation exists regardless.
///
/// However, the backing integer of a `Section` is *not* the same as the section index, because
/// section indices can change. For the index of a section, see `Section.index`.
const Section = enum(u8) {
    _,

    fn lc(sect: Section, macho: *const MachO) LoadCommand {
        return macho.sections.items(.lc)[@backingInt(sect)];
    }
    fn lcPtrConst(sect: Section, macho: *const MachO) *const std.macho.section_64 {
        return sect.lc(macho).ptrConst(macho, std.macho.section_64);
    }
    fn lcPtr(sect: Section, macho: *MachO) *std.macho.section_64 {
        return sect.lc(macho).ptr(macho, std.macho.section_64);
    }

    fn name(sect: Section, macho: *const MachO) []const u8 {
        return std.mem.sliceTo(&sect.lcPtrConst(macho).sectname, 0);
    }

    fn vaddr(sect: Section, macho: *const MachO) u64 {
        return macho.targetLoad(&sect.lcPtrConst(macho).addr);
    }

    fn segmentName(sect: Section, macho: *const MachO) []const u8 {
        return std.mem.sliceTo(&sect.lcPtrConst(macho).segname, 0);
    }

    fn segment(sect: Section, macho: *const MachO) Segment {
        return macho.sections.items(.segment)[@backingInt(sect)];
    }

    fn node(sect: Section, macho: *const MachO) MappedFile.Node.Index {
        return macho.sections.items(.node)[@backingInt(sect)];
    }

    fn alignment(sect: Section, macho: *const MachO) MappedFile.Alignment {
        return .fromLog2Units(@intCast(macho.targetLoad(&sect.lcPtrConst(macho).@"align")));
    }

    /// Returns this section's index in the Mach-O file. Section indices are implicitly assigned
    /// based on the order of load commands, so this value may change in response to segments or
    /// sections being added. When this value changes for a  given `Section`, it is added to
    /// `MachO.sections_changed_index`, so that references to the section index can be updated.
    fn index(sect: Section, macho: *const MachO) u8 {
        return macho.sections.items(.index)[@backingInt(sect)];
    }

    const NameAdapter = struct {
        macho: *const MachO,
        const SegmentAndSection = struct {
            segment: []const u8,
            section: []const u8,
        };
        pub fn eql(ctx: NameAdapter, a_name: SegmentAndSection, _: void, b_sect_raw: usize) bool {
            const b_sect: Section = @fromBackingInt(@intCast(b_sect_raw));
            return std.mem.eql(u8, a_name.segment, b_sect.segmentName(ctx.macho)) and
                std.mem.eql(u8, a_name.section, b_sect.name(ctx.macho));
        }
        pub fn hash(ctx: NameAdapter, a_name: SegmentAndSection) u32 {
            _ = ctx;
            var h: std.hash.Wyhash = .init(a_name.section.len);
            h.update(a_name.section);
            h.update(a_name.segment);
            return @truncate(h.final());
        }
    };

    /// Matches the layout of `std.macho.section_64.flags`.
    ///
    /// TODO: move to `std.macho` and use as the type of `std.macho.section_64.flags`.
    const Flags = packed struct(u32) {
        type: enum(u8) {
            regular = 0x00,
            zerofill = 0x01,
            cstring_literals = 0x02,
            @"4byte_literals" = 0x03,
            @"8byte_literals" = 0x04,
            literal_pointers = 0x05,
            non_lazy_symbol_pointers = 0x06,
            lazy_symbol_pointers = 0x07,
            symbol_stubs = 0x08,
            mod_init_func_pointers = 0x09,
            mod_term_func_pointers = 0x0a,
            coalesced = 0x0b,
            gb_zerofill = 0x0c,
            interposing = 0x0d,
            @"16byte_literals" = 0x0e,
            dtrace_dof = 0x0f,
            lazy_dylib_symbol_pointers = 0x10,
            thread_local_regular = 0x11,
            thread_local_zerofill = 0x12,
            thread_local_variables = 0x13,
            thread_local_variable_pointers = 0x14,
            thread_local_init_function_pointers = 0x15,
            init_func_offsets = 0x16,
            _,
        },
        loc_reloc: bool = false,
        ext_reloc: bool = false,
        some_instructions: bool = false,
        _unused: u15 = 0,
        self_modifying_code: bool = false,
        live_support: bool = false,
        no_dead_strip: bool = false,
        strip_static_syms: bool = false,
        no_toc: bool = false,
        pure_instructions: bool = false,
    };
};

const InputAtom = struct {
    node: MappedFile.Node.Index,
    first_symbol: Symbol.Index.Optional,
    first_reloc: Reloc.Index,
    vaddr: u64,
    /// `true` if there are any relocations *in* this atom which are runtime fixups.
    ///
    /// TODO: storing a bool here is bad for memory utilization, but there's no point in me finding
    /// somewhere else to put this state before we finish figuring out this linker's design. It
    /// should definitely be moved at some point though!
    any_fixups: bool,

    const Index = enum(u32) {
        _,
        fn ptr(iai: InputAtom.Index, macho: *MachO) *InputAtom {
            return &macho.input_atoms.items[@backingInt(iai)];
        }
        fn ptrConst(iai: InputAtom.Index, macho: *const MachO) *const InputAtom {
            return &macho.input_atoms.items[@backingInt(iai)];
        }
        fn node(iai: InputAtom.Index, macho: *const MachO) MappedFile.Node.Index {
            return iai.ptrConst(macho).node;
        }
    };
};

const Symbol = struct {
    node: MappedFile.Node.Index.Optional,
    first_target_reloc: Reloc.Index.Optional,
    prev_in_node: Symbol.Index.Optional,
    next_in_node: Symbol.Index.Optional,

    const Index = enum(u32) {
        _,
        fn ptr(i: Index, macho: *MachO) *Symbol {
            return &macho.symtab.items[@backingInt(i)];
        }
        fn ptrConst(i: Index, macho: *const MachO) *const Symbol {
            return &macho.symtab.items[@backingInt(i)];
        }
        fn nlist(i: Index, macho: *MachO) *std.macho.nlist_64 {
            const nsyms = macho.targetLoad(
                &macho.lc.symtab.ptrConst(macho, std.macho.symtab_command).nsyms,
            );
            const slice: []std.macho.nlist_64 = @ptrCast(@alignCast(
                macho.node.symtab.slice(&macho.mf)[0 .. nsyms * @sizeOf(std.macho.nlist_64)],
            ));
            return &slice[@backingInt(i)];
        }
        fn nlistConst(i: Index, macho: *const MachO) *const std.macho.nlist_64 {
            const nsyms = macho.targetLoad(
                &macho.lc.symtab.ptrConst(macho, std.macho.symtab_command).nsyms,
            );
            const slice: []const std.macho.nlist_64 = @ptrCast(@alignCast(
                macho.node.symtab.slice(&macho.mf)[0 .. nsyms * @sizeOf(std.macho.nlist_64)],
            ));
            return &slice[@backingInt(i)];
        }
        fn name(i: Index, macho: *const MachO) []const u8 {
            const n_strx = macho.targetLoad(&i.nlistConst(macho).n_strx);
            return std.mem.sliceTo(macho.node.strtab.sliceConst(&macho.mf)[n_strx..], 0);
        }
        fn value(i: Index, macho: *const MachO) u64 {
            return macho.targetLoad(&i.nlistConst(macho).n_value);
        }
        fn flushMoved(symbol: Index, macho: *MachO, new_value: u64) Allocator.Error!void {
            macho.targetStore(&symbol.nlist(macho).n_value, new_value);
            try symbol.applyTargetRelocs(macho, .init(.{
                .symbol = true,
                .branch = true,
                .got = false,
                .tlv = false,
            }));
        }
        fn applyTargetRelocs(
            i: Index,
            macho: *MachO,
            targets: std.enums.EnumSet(Reloc.Target),
        ) Allocator.Error!void {
            var reloc_it = i.ptr(macho).first_target_reloc;
            while (reloc_it.unwrap()) |ri| : (reloc_it = ri.ptr(macho).next_with_same_target_sym) {
                if (targets.contains(ri.ptr(macho).bits.target)) {
                    try ri.ptr(macho).apply(macho);
                }
            }
        }
        const Optional = enum(u32) {
            none = std.math.maxInt(u32),
            _,
            fn unwrap(o: Optional) ?Index {
                return switch (o) {
                    .none => null,
                    _ => @fromBackingInt(@backingInt(o)),
                };
            }
            fn wrap(i: Index) Optional {
                return @bitCast(i);
            }
        };
    };

    const InitValue = union(enum) {
        abs: u64,
        addr_in_node: struct {
            /// Must support `getNodeVAddr`, `getNodeSection`, and `nodeFirstSymbolPtr`.
            node: MappedFile.Node.Index,
            offset: u64,
        },
    };

    const GlobalAdapter = struct {
        macho: *MachO,
        pub fn eql(ctx: GlobalAdapter, lhs_slice: []const u8, rhs_symbol: Symbol.Index, _: usize) bool {
            const macho = ctx.macho;
            return std.mem.eql(u8, lhs_slice, rhs_symbol.name(macho));
        }
        pub fn hash(ctx: GlobalAdapter, s: []const u8) u32 {
            _ = ctx;
            return std.array_hash_map.hashString(s);
        }
    };
};

const Reloc = struct {
    node: MappedFile.Node.Index,
    offset: u32,
    target_symbol: Symbol.Index,
    addend: u64,
    bits: Bits,
    next_with_same_target_sym: Reloc.Index.Optional,

    const Bits = packed struct(u8) {
        target: Target,
        type: Type,
        result: enum(u2) {
            ok,
            overflowed,
            misaligned,
            /// This is a special value indicating that this relocation is lowered to a runtime
            /// fixup rather than being resolved statically. TODO: I'm not entirely sure if this is
            /// a good idea, particularly since the question of "is this a runtime fixup" can, I
            /// believe, be fully recovered based on the `Bits.target` and `Bits.type`...?
            fixup,
        },
    };
    const Target = enum(u2) {
        symbol,
        branch,
        got,
        tlv,
    };
    const Type = enum(u4) {
        abs8,
        abs16,
        abs32,
        abs64,
        rel32,
        arm64_branchrel26,
        arm64_pagerel21,
        arm64_pageoff12,

        /// If the given relocation requires a particular machine instruction to already exist in
        /// the destination bytes, this function checks if that requirement is upheld and returns
        /// `false` if not.
        ///
        /// This function does not perform length checks on `dest_overlong`; the caller guarantees
        /// that it will be at least long enough to support this relocation type.
        fn validateInst(t: Type, dest_overlong: []const u8) bool {
            switch (t) {
                .abs8,
                .abs16,
                .abs32,
                .abs64,
                .rel32,
                => return true,

                .arm64_branchrel26 => {
                    const inst: aarch64.encoding.Instruction = .read(dest_overlong[0..4]);
                    return switch (inst.decode()) {
                        .branch_exception_generating_system => |begs| switch (begs.decode()) {
                            .unconditional_branch_immediate => true,
                            else => false,
                        },
                        else => false,
                    };
                },
                .arm64_pagerel21 => {
                    const inst: aarch64.encoding.Instruction = .read(dest_overlong[0..4]);
                    return switch (inst.decode()) {
                        .data_processing_immediate => |dpi| switch (dpi.decode()) {
                            .pc_relative_addressing => |pra| switch (pra.decode()) {
                                .adrp => true,
                                else => false,
                            },
                            else => false,
                        },
                        else => false,
                    };
                },
                .arm64_pageoff12 => {
                    const inst: aarch64.encoding.Instruction = .read(dest_overlong[0..4]);
                    return switch (inst.decode()) {
                        .data_processing_immediate => |dpi| switch (dpi.decode()) {
                            .add_subtract_immediate => true,
                            else => false,
                        },
                        .load_store => |ls| switch (ls.decode()) {
                            .register_unsigned_immediate => true,
                            else => false,
                        },
                        else => false,
                    };
                },
            }
        }
    };

    fn apply(reloc: *Reloc, macho: *MachO) Allocator.Error!void {
        // Don't bother applying the reloc if the bytes we're writing to have a pending move event:
        if (reloc.node.hasMoved(&macho.mf)) {
            return;
        }

        // Also don't bother if a pending move will change the target value:
        {
            const target_node: MappedFile.Node.Index.Optional = switch (reloc.bits.target) {
                .symbol => reloc.target_symbol.ptr(macho).node,
                .branch => switch (macho.stubs.contains(reloc.target_symbol)) {
                    false => reloc.target_symbol.ptr(macho).node,
                    true => .wrap(macho.known_sections.stubs.node(macho)),
                },
                .got => .wrap(macho.known_sections.got.node(macho)),
                .tlv => @panic("TODO(MachO2): apply TLV relocation"),
            };
            if (target_node.unwrap()) |node| {
                if (node.hasMoved(&macho.mf)) {
                    return;
                }
            }
        }

        switch (reloc.bits.result) {
            .fixup => {
                const segment = macho.getNodeSection(reloc.node).segment(macho);
                const offset_in_segment: u64 = reloc.node.fileLocation(&macho.mf, false).offset -
                    segment.node(macho).fileLocation(&macho.mf, false).offset +
                    reloc.offset;
                const page_align = macho.targetPageAlign();
                const page_start: u32 = @intCast(@divFloor(offset_in_segment, page_align.toByteUnits()));
                const page_end: u32 = @intCast(@divCeil(offset_in_segment + 8, page_align.toByteUnits()));
                for (page_start..page_end) |page_index| {
                    try macho.dirty_fixup_chain_pages.put(macho.base.comp.gpa, .{
                        .seg = segment,
                        .page = @intCast(page_index),
                    }, {});
                }
                return;
            },
            .ok => {},
            .overflowed => macho.overflowed_reloc_count -= 1,
            .misaligned => macho.misaligned_reloc_count -= 1,
        }

        if (reloc.applyInner(macho)) {
            @branchHint(.likely);
            reloc.bits.result = .ok;
        } else |err| switch (err) {
            error.RelocationOverflow => {
                log.debug("relocation failed (overflow)", .{});
                reloc.bits.result = .overflowed;
                macho.overflowed_reloc_count += 1;
            },
            error.RelocationMisaligned => {
                log.debug("relocation failed (misaligned)", .{});
                reloc.bits.result = .misaligned;
                macho.misaligned_reloc_count += 1;
            },
        }
    }
    fn applyInner(r: Reloc, macho: *MachO) error{ RelocationMisaligned, RelocationOverflow }!void {
        log.debug("applying relocation: nodes[{d}][{d}..] = {t}({t}({q}) + {d})", .{
            r.node, r.offset, r.bits.type, r.bits.target, r.target_symbol.name(macho), r.addend,
        });

        const target_endian = macho.targetEndian();

        const dest_slice = r.node.slice(&macho.mf)[r.offset..];
        const dest_vaddr: u64 = macho.getNodeVAddr(r.node) + r.offset;

        const target_val: u64 = switch (r.bits.target) {
            .symbol => r.target_symbol.value(macho) +% r.addend,
            .branch => if (macho.stubs.getIndex(r.target_symbol)) |stub_index| val: {
                const stub_offset = @as(u64, stub_index) * macho.targetStubSize();
                const stub_vaddr = macho.known_sections.stubs.vaddr(macho) + stub_offset;
                break :val stub_vaddr +% r.addend;
            } else r.target_symbol.value(macho) +% r.addend,
            .got => val: {
                const got_entry_index = macho.got.getIndex(r.target_symbol).?;
                const got_entry_offset = got_entry_index * macho.targetPtrSize();
                const got_entry_vaddr = macho.known_sections.got.vaddr(macho) + got_entry_offset;
                break :val got_entry_vaddr +% r.addend;
            },
            .tlv => @panic("TODO(MachO2): apply TLV relocation"),
        };

        switch (r.bits.type) {
            .abs8 => std.mem.writeInt(
                u8,
                dest_slice[0..1],
                std.math.cast(u8, target_val) orelse return error.RelocationOverflow,
                target_endian,
            ),
            .abs16 => std.mem.writeInt(
                u16,
                dest_slice[0..2],
                std.math.cast(u16, target_val) orelse return error.RelocationOverflow,
                target_endian,
            ),
            .abs32 => std.mem.writeInt(
                u32,
                dest_slice[0..4],
                std.math.cast(u32, target_val) orelse return error.RelocationOverflow,
                target_endian,
            ),
            .abs64 => std.mem.writeInt(
                u64,
                dest_slice[0..8],
                std.math.cast(u64, target_val) orelse return error.RelocationOverflow,
                target_endian,
            ),
            .rel32 => std.mem.writeInt(
                i32,
                dest_slice[0..4],
                std.math.cast(
                    i32,
                    @as(i64, @bitCast(target_val -% dest_vaddr)),
                ) orelse return error.RelocationOverflow,
                target_endian,
            ),
            .arm64_branchrel26 => {
                const off: i64 = @bitCast(target_val -% dest_vaddr);
                const shifted: i64 = off >> 2;
                if (shifted << 2 != off) {
                    return error.RelocationMisaligned;
                }
                const casted = std.math.cast(i26, shifted) orelse return error.RelocationOverflow;

                var inst: aarch64.encoding.Instruction = .read(dest_slice[0..4]);
                inst.branch_exception_generating_system.unconditional_branch_immediate.group.imm26 = casted;
                inst.write(dest_slice[0..4]);
            },
            .arm64_pagerel21 => {
                const cur_page: u53 = @intCast(dest_vaddr >> 12);
                const target_page: u53 = @intCast(target_val >> 12);
                const page_off: i53 = @bitCast(target_page -% cur_page);
                const split: packed struct(i21) { lo: u2, hi: i19 } = @bitCast(
                    std.math.cast(i21, page_off) orelse return error.RelocationOverflow,
                );

                var inst: aarch64.encoding.Instruction = .read(dest_slice[0..4]);
                inst.data_processing_immediate.pc_relative_addressing.adrp.immhi = split.hi;
                inst.data_processing_immediate.pc_relative_addressing.adrp.immlo = split.lo;
                inst.write(dest_slice[0..4]);
            },
            .arm64_pageoff12 => {
                const page_off: u12 = @truncate(target_val);

                // This could be one of several instructions: an `ADD`, an `LDR`, or an `STR`. The
                // `LDR` and `STR` instructions have scale factors which we need to compensate for,
                // so we need to decode the instruction to figure out which case we're in.
                var inst: aarch64.encoding.Instruction = .read(dest_slice[0..4]);
                switch (inst.decode()) {
                    .data_processing_immediate => {
                        inst.data_processing_immediate.add_subtract_immediate.group.imm12 = page_off;
                    },
                    .load_store => |ls| {
                        const scale: u3 = switch (ls.register_unsigned_immediate.decode()) {
                            .integer => |i| @backingInt(i.group.size), // 1/2/4/8-byte load/store
                            .vector => |v| @backingInt(v.group.opc1.decode(v.group.size)), // 1/2/4/8/16-byte vector load/store
                        };
                        const shifted = page_off >> scale;
                        if (shifted << scale != page_off) {
                            return error.RelocationMisaligned;
                        }
                        inst.load_store.register_unsigned_immediate.group.imm12 = shifted;
                    },
                    else => unreachable,
                }
                inst.write(dest_slice[0..4]);
            },
        }
    }

    const Index = enum(u32) {
        _,

        const Optional = enum(u32) {
            none = std.math.maxInt(u32),
            _,

            fn unwrap(o: Optional) ?Index {
                return switch (o) {
                    .none => null,
                    _ => @fromBackingInt(@backingInt(o)),
                };
            }
            fn wrap(i: Index) Optional {
                return @bitCast(i);
            }
        };

        fn ptr(ri: Reloc.Index, macho: *MachO) *Reloc {
            return &macho.relocs.items[@backingInt(ri)];
        }
    };
};

const Node = union(enum) {
    macho,
    /// Contains the `mach_header_64` structure and all load commands. Inside of the `.segment` node for the `__TEXT` segment.
    macho_header,

    symtab,
    strtab,

    indirect_syms,
    section_indirect_syms: Section,

    linkedit_data: LoadCommand,

    /// Header of the `LC_DYLD_CHAINED_FIXUPS` load command's `.linkedit_data` node, containing the
    /// `std.macho.dyld_chained_fixups_header` struct.
    fixup_chains_header,
    fixup_chains_imports,
    fixup_chains_symbol_names,
    fixup_chains_starts,
    fixup_chains_starts_in_image,
    fixup_chains_starts_in_segment: Segment,

    segment: Segment,
    section: Section,
    section_manual_size: Section,

    input_atom: InputAtom.Index,
};
fn getNode(macho: *const MachO, ni: MappedFile.Node.Index) Node {
    return macho.nodes.get(@backingInt(ni));
}

pub fn open(
    arena: Allocator,
    comp: *Compilation,
    path: std.Build.Cache.Path,
    options: link.File.OpenOptions,
) !*MachO {
    return create(arena, comp, path, options);
}
pub fn createEmpty(
    arena: Allocator,
    comp: *Compilation,
    path: std.Build.Cache.Path,
    options: link.File.OpenOptions,
) !*MachO {
    return create(arena, comp, path, options);
}
fn create(
    arena: Allocator,
    comp: *Compilation,
    path: std.Build.Cache.Path,
    options: link.File.OpenOptions,
) !*MachO {
    const io = comp.io;

    const macho = try arena.create(MachO);
    const file = try path.root_dir.handle.createFile(io, path.sub_path, .{
        .read = true,
        .permissions = link.File.determinePermissions(comp.config.output_mode, comp.config.link_mode),
    });

    macho.* = .{
        .base = .{
            .tag = .macho2,

            .comp = comp,
            .emit = path,

            .file = file,
            .gc_sections = false,
            .print_gc_sections = false,
            .build_id = .none,
            .allow_shlib_undefined = false,
            .stack_size = 0,
        },
        .options = options,

        .const_prog_node = .none,
        .synth_prog_node = .none,
        .input_prog_node = .none,

        .mf = try .init(file, comp.gpa, io),
        .nodes = .empty,

        .node = undefined,
        .lc = .{
            .offsets = .empty,
            .pagezero = undefined,
            .symtab = undefined,
            .dysymtab = undefined,
            .entry_point = null,
        },
        .known_sections = undefined,
        .segments = .empty,
        .sections = .empty,
        .sections_by_name = .empty,

        .dirty_fixup_chain_pages = .empty,

        .input_atoms = .empty,

        .strtab = .empty,
        .symtab = .empty,
        .global_symbols = .empty,

        .relocs = .empty,
        .overflowed_reloc_count = 0,
        .misaligned_reloc_count = 0,

        .fixup_chain_symbol_names_len = 0,
        .imports = .empty,
        .got = .empty,
        .stubs = .empty,

        .sections_changed_index = .empty,

        .prev_got_offset = 0,
    };
    errdefer macho.deinit();

    try macho.initInner();

    return macho;
}
fn initInner(macho: *MachO) !void {
    const comp = macho.base.comp;
    const gpa = comp.gpa;

    macho.node.macho = .root;
    try macho.nodes.append(gpa, .macho);

    const arch: enum { aarch64, x86_64 } = switch (comp.getTarget().cpu.arch) {
        .aarch64 => .aarch64,
        .x86_64 => .x86_64,
        else => |arch| std.debug.panic("TODO: unsupported target arch {t}", .{arch}),
    };

    const mach_header_64 = std.macho.mach_header_64;

    // The `__TEXT` segment is special because it contains the mach header and load commands, so we
    // must create the text segment node before we can actually call `addSegment`.
    const text_seg_node = try macho.node.macho.addOnlyHeaderChild(gpa, &macho.mf, .{
        .alignment = macho.mf.flags.block_size.max(macho.targetPageAlign()),
        .moved = true, // required by `addSegment`
        .bubbles_moved = false,
    });
    try macho.nodes.append(gpa, .{ .segment = undefined });

    macho.node.header = try text_seg_node.addOnlyHeaderChild(gpa, &macho.mf, .{
        .size = @sizeOf(mach_header_64),
        .alignment = .@"8",
    });
    try macho.nodes.append(gpa, .macho_header);

    const mach_header: *mach_header_64 = @ptrCast(@alignCast(
        macho.node.header.slice(&macho.mf)[0..@sizeOf(mach_header_64)],
    ));
    mach_header.* = .{
        .cputype = switch (arch) {
            .aarch64 => std.macho.CPU_TYPE_ARM64,
            .x86_64 => std.macho.CPU_TYPE_X86_64,
        },
        .cpusubtype = switch (arch) {
            .aarch64 => std.macho.CPU_SUBTYPE_ARM_ALL,
            .x86_64 => std.macho.CPU_SUBTYPE_X86_64_ALL,
        },
        .filetype = switch (comp.config.output_mode) {
            .Exe => std.macho.MH_EXECUTE,
            .Lib => switch (comp.config.link_mode) {
                .static => std.macho.MH_OBJECT,
                .dynamic => std.macho.MH_DYLIB,
            },
            .Obj => std.macho.MH_OBJECT,
        },
        .ncmds = 0,
        .sizeofcmds = 0,
        .flags = std.macho.MH_DYLDLINK | std.macho.MH_PIE,
    };
    if (macho.targetEndian() != std.lang.Endian.native) {
        std.mem.byteSwapAllFields(mach_header_64, mach_header);
    }

    // The "__LINKEDIT" segment is special in that it needs to be at the end of the Mach-O file, so
    // like "__TEXT", we'll construct its node here instead of in `addSegment`.
    const linkedit_seg_node = try macho.node.macho.addFooterChildBefore(gpa, &macho.mf, .none, .{
        .alignment = macho.mf.flags.block_size.max(macho.targetPageAlign()),
        .moved = true, // required by `addSegment`
        .bubbles_moved = false,
    });
    try macho.nodes.append(gpa, .{ .segment = undefined });

    // The "__PAGEZERO" segment must come before any other segment load commands. It is inaccessible
    // and occupies 4 GiB (apparently to catch invalid 32-bit pointer accesses).
    macho.lc.pagezero = try macho.appendSimpleLoadCommand(std.macho.segment_command_64, .{
        .cmdsize = @sizeOf(std.macho.segment_command_64),
        .segname = "__PAGEZERO\x00\x00\x00\x00\x00\x00".*,
        .vmaddr = 0,
        .vmsize = 0x1_0000_0000,
        .fileoff = 0,
        .filesize = 0,
        .maxprot = .{},
        .initprot = .{},
        .nsects = 0,
        .flags = 0,
    });
    assert(try macho.addSegment(.{
        .name = "__TEXT",
        .prot = .{ .READ = true, .EXEC = true },
        .existing_node = text_seg_node,
        .flags = .{},
    }) == .text);
    const data_seg = try macho.addSegment(.{
        .name = "__DATA",
        .prot = .{ .READ = true, .WRITE = true },
        .flags = .{},
    });
    // TODO: we should also create a `__DATA_CONST` segment with `.{ .flags = .read_only = true }`,
    // and when we do, `loadObject` should use it for `InputSection.Usage.data_const_fixups`.
    // Unfortunately, doing that now introduces a second floating segment node, which reveals that
    // dyld has an annoying ordering requirement:
    //
    //     dyld[77415]: segment '__DATA' file offset out of order
    //
    // We need to decide the best way to resolve this. Reordering the segments in the load commands
    // and in virtual memory is probably too expensive to risk doing often, so we need to preserve
    // the segment order somehow, e.g. by having all but one segment be a header or footer.
    const linkedit_seg = try macho.addSegment(.{
        .name = "__LINKEDIT",
        .prot = .{ .READ = true },
        .existing_node = linkedit_seg_node,
        .flags = .{},
    });

    _ = try macho.addSection(.{
        .segment = .text,
        .name = "__text",
        .@"align" = .@"4",
        .flags = .{ .type = .regular, .some_instructions = true },
    });
    _ = try macho.addSection(.{
        .segment = .text,
        .name = "__const",
        .@"align" = .@"1",
        .flags = .{
            .type = .regular,
        },
    });
    _ = try macho.addSection(.{
        .segment = data_seg,
        .name = "__data",
        .@"align" = .@"1",
        .flags = .{
            .type = .regular,
        },
    });

    macho.known_sections.got = try macho.addSection(.{
        .segment = data_seg,
        .name = "__got",
        .@"align" = .@"8",
        .manual_size = true,
        .flags = .{ .type = .non_lazy_symbol_pointers },
        // This field holds an offset into the "indirect symbol table" at which this section's
        // indirect symbol references begin. It will be populated by `flushMoved` of a
        // `.section_indirect_syms` node we'll create shortly.
        .reserved1 = 0,
    });

    macho.known_sections.stubs = try macho.addSection(.{
        .segment = .text,
        .name = "__stubs",
        .@"align" = .@"4",
        .manual_size = true,
        .flags = .{
            .type = .symbol_stubs,
            .some_instructions = true,
            .pure_instructions = true,
        },
        // As with `__got`, will be populated by `flushMoved` of a `.section_indirect_syms` node.
        .reserved1 = 0,
        // `reserved2` holds the size of each stub entry.
        .reserved2 = macho.targetStubSize(),
    });

    macho.node.symtab = try linkedit_seg.node(macho).addFloatingChild(gpa, &macho.mf, .{
        .alignment = macho.mf.flags.block_size.max(.of(std.macho.nlist_64)),
        .moved = true, // so that `symoff` in the symtab load command will be set
    });
    try macho.nodes.append(gpa, .symtab);

    macho.node.strtab = try linkedit_seg.node(macho).addFloatingChild(gpa, &macho.mf, .{
        .alignment = macho.mf.flags.block_size,
        .moved = true, // so that `stroff` in the symtab load command will be set
    });
    try macho.nodes.append(gpa, .strtab);

    try macho.node.strtab.ensureMinimumSize(gpa, &macho.mf, 1);
    macho.node.strtab.slice(&macho.mf)[0] = 0;

    const indirect_syms_node = try linkedit_seg.node(macho).addFloatingChild(gpa, &macho.mf, .{
        .alignment = .@"4",
        .moved = true, // so that `indirectsymoff` in the dysymtab load command will be set
    });
    try macho.nodes.append(gpa, .indirect_syms);

    macho.node.got_indirect_syms = try indirect_syms_node.addFloatingChild(gpa, &macho.mf, .{
        .alignment = .@"4",
        .moved = true, // so that `reserved1` in the section will be set
    });
    try macho.nodes.append(gpa, .{ .section_indirect_syms = macho.known_sections.got });

    macho.node.stubs_indirect_syms = try indirect_syms_node.addFloatingChild(gpa, &macho.mf, .{
        .alignment = .@"4",
        .moved = true, // so that `reserved1` in the section will be set
    });
    try macho.nodes.append(gpa, .{ .section_indirect_syms = macho.known_sections.stubs });

    const chained_fixups_lc = try macho.appendSimpleLoadCommand(std.macho.linkedit_data_command, .{
        .cmd = .DYLD_CHAINED_FIXUPS,
        .dataoff = 0,
        .datasize = 0,
    });
    const chained_fixups_node = try linkedit_seg.node(macho).addFloatingChild(gpa, &macho.mf, .{
        .alignment = macho.mf.flags.block_size.max(.@"8"),
        .moved = true, // so that `dataoff` will be set
        .bubbles_moved = false,
    });
    try macho.nodes.append(gpa, .{ .linkedit_data = chained_fixups_lc });

    macho.node.fixup_chains_header = try chained_fixups_node.addOnlyHeaderChild(gpa, &macho.mf, .{
        .size = @sizeOf(std.macho.dyld_chained_fixups_header),
        .alignment = .@"4",
    });
    try macho.nodes.append(gpa, .fixup_chains_header);

    {
        const chained_fixups_hdr: *std.macho.dyld_chained_fixups_header = @ptrCast(@alignCast(
            macho.node.fixup_chains_header.slice(&macho.mf)[0..@sizeOf(std.macho.dyld_chained_fixups_header)],
        ));
        chained_fixups_hdr.* = .{
            .fixups_version = 0,
            .starts_offset = 0,
            .imports_offset = 0,
            .symbols_offset = 0,
            .imports_count = 0,
            .imports_format = .addend_64,
            .symbols_format = .uncompressed,
        };
        if (macho.targetEndian() != std.lang.Endian.native) {
            std.mem.byteSwapAllFields(std.macho.dyld_chained_fixups_header, chained_fixups_hdr);
        }
    }

    // We are now going to create the `std.macho.dyld_chained_starts_in_segment` nodes for every
    // segment, which means we are assuming that no more segments will be created. Therefore, all
    // `addSegment` calls must come *before* this point.

    const fixup_chains_starts = try chained_fixups_node.addFloatingChild(gpa, &macho.mf, .{
        .size = 0,
        .alignment = .@"8",
        .moved = true, // so that `starts_offset` is populated
    });
    try macho.nodes.append(gpa, .fixup_chains_starts);

    macho.node.fixup_chains_starts_in_image = try fixup_chains_starts.addOnlyHeaderChild(gpa, &macho.mf, .{
        .size = 8 + 4 * macho.segments.len,
        .alignment = .@"4",
    });
    try macho.nodes.append(gpa, .fixup_chains_starts_in_image);

    {
        const chained_starts_in_image: []u32 = @ptrCast(@alignCast(
            macho.node.fixup_chains_starts_in_image.slice(&macho.mf),
        ));
        chained_starts_in_image[0] = @intCast(macho.segments.len + 1); // +1 for `__PAGEZERO`
        chained_starts_in_image[1] = 0; // `__PAGEZERO`
        chained_starts_in_image[macho.segments.len + 1] = 0; // `__LINKEDIT`
        for (
            macho.segments.items(.fixup_chain_starts)[0 .. macho.segments.len - 1],
            0..,
        ) |*starts_in_segment, segment_usize| {
            const segment: Segment = @fromBackingInt(@intCast(segment_usize));
            assert(!std.mem.eql(u8, segment.name(macho), "__LINKEDIT"));
            starts_in_segment.* = .wrap(try fixup_chains_starts.addFloatingChild(gpa, &macho.mf, .{
                .alignment = .@"8",
                .size = @sizeOf(std.macho.dyld_chained_starts_in_segment),
                .moved = true,
            }));
            try macho.nodes.append(gpa, .{ .fixup_chains_starts_in_segment = segment });
            const starts_hdr: *std.macho.dyld_chained_starts_in_segment = @ptrCast(@alignCast(
                starts_in_segment.unwrap().?.slice(&macho.mf)[0..@sizeOf(std.macho.dyld_chained_starts_in_segment)],
            ));
            starts_hdr.* = .{
                .size = @sizeOf(std.macho.dyld_chained_starts_in_segment),
                .page_size = @intCast(macho.targetPageAlign().toByteUnits()),
                .pointer_format = .@"64",
                .segment_offset = 0, // populated by `flushMoved` of the corresponding segment
                .max_valid_pointer = 0,
                .page_count = 0,
            };
            if (macho.targetEndian() != std.lang.Endian.native) {
                std.mem.byteSwapAllFields(std.macho.dyld_chained_starts_in_segment, starts_hdr);
            }
        }
    }

    // This is a footer because dyld has an undocumented requirement that the imports must be
    // ordered after the starts.
    macho.node.fixup_chains_imports = try chained_fixups_node.addOnlyFooterChild(gpa, &macho.mf, .{
        .alignment = .@"8",
        .moved = true, // so that `imports_offset` is populated
    });
    try macho.nodes.append(gpa, .fixup_chains_imports);

    // This is a footer because `dyld_info` complains if the imports appear after the symbol name
    // string table. Since there's no length associated with that string table, `dyld_info` can only
    // be sure they don't overlap if the string table is the last thing in the chained fixups data.
    macho.node.fixup_chains_symbol_names = try chained_fixups_node.addFooterChildBefore(gpa, &macho.mf, .none, .{
        .alignment = .@"8", // so that the alignment of `macho.node.fixup_chains_imports` is not violated
        .moved = true, // so that `symbols_offset` is populated
    });
    try macho.nodes.append(gpa, .fixup_chains_symbol_names);

    macho.lc.symtab = try macho.appendSimpleLoadCommand(std.macho.symtab_command, .{
        .symoff = 0,
        .nsyms = 0,
        .stroff = 0,
        .strsize = 1, // empty string (null terminator)
    });

    // Nowadays, `LC_DYSYMTAB` is redundant with chained fixups and the symbol export trie. However,
    // if we omit it, some macOS tooling does get annoyed at us. Therefore, we try to emit a minimum
    // viable `LC_DYSYMTAB` command, just to keep the tooling happy.
    macho.lc.dysymtab = try macho.appendSimpleLoadCommand(std.macho.dysymtab_command, .{
        // Some tooling will check that the number of symbols according to the `LC_DYSYMTAB` command
        // matches the actual length of the symbol table in `LC_SYMTAB`. However, we don't want to
        // need to partition our symbol table into local/extdef/undef symbols, because that makes
        // incremental linking less efficient. Luckily, since this information is basically useless
        // nowadays, nothing seems to mind if we just say that every symbol is local. So, whenever
        // we add any symbol to the symtab, we will increase `ilocalsym` in lockstep with the symbol
        // count. We also need to increment `iextdefsym` and `iundefsym`, because some tooling may
        // check that local symbols come before extdef and undef symbols here. I don't even know why
        // they *have* the `iextdefsym` and `iundefsym` fields if they're only going to allow them
        // to have specific values...
        .ilocalsym = 0, // always 0
        .nlocalsym = 0, // equal to symbol count
        .iextdefsym = 0, // equal to symbol count
        .nextdefsym = 0, // always 0
        .iundefsym = 0, // equal to symbol count
        .nundefsym = 0, // always 0

        // We *do* need to emit a valid indirect symbol table, which is an array of symbol indices
        // indicating which symbols are being targeted by entries in `__got` and `__stubs`. Although
        // not required for fixups, some debug tooling will still try to read this table without any
        // validation, so will just crash if it's absent or invalid. Luckily, the indirection here
        // actually makes this pretty straightforward for us to emit.
        .indirectsymoff = 0, // populated by `flushMachOFileOffset` of the `.indirect_syms` node
        .nindirectsyms = 0, // populated by `flushResized` of the `.indirect_syms` node

        // These fields are *required* to be zeroed in modern Mach-O binaries.
        .tocoff = 0,
        .ntoc = 0,
        .modtaboff = 0,
        .nmodtab = 0,
        .extrefsymoff = 0,
        .nextrefsyms = 0,
        .extreloff = 0,
        .nextrel = 0,
        .locreloff = 0,
        .nlocrel = 0,
    });

    const want_dyld_path: bool = switch (comp.config.output_mode) {
        .Obj => false,
        .Lib => switch (comp.config.link_mode) {
            .static => false,
            .dynamic => comp.root_mod.resolved_target.is_explicit_dynamic_linker,
        },
        .Exe => true,
    };
    if (want_dyld_path) {
        if (comp.getTarget().dynamic_linker.get()) |dyld_path| {
            const cmd = try macho.appendLoadCommand(std.macho.dylinker_command, .{
                .cmd = .LOAD_DYLINKER,
                .cmdsize = std.mem.alignForward(u32, @intCast(
                    @sizeOf(std.macho.dylinker_command) + dyld_path.len + 1,
                ), 8),
                .name = @sizeOf(std.macho.dylinker_command),
            });
            @memcpy(cmd.trailing[0..dyld_path.len], dyld_path);
            @memset(cmd.trailing[dyld_path.len..], 0);
        }
    }

    entry: switch (macho.options.entry) {
        .default => switch (comp.config.output_mode) {
            .Exe => continue :entry .enabled,
            .Lib, .Obj => continue :entry .disabled,
        },
        .disabled => {},
        .enabled, .named => {
            macho.lc.entry_point = try macho.appendSimpleLoadCommand(
                std.macho.entry_point_command,
                .{ .entryoff = 0, .stacksize = macho.options.stack_size orelse 0 },
            );
        },
    }

    _ = try macho.appendSimpleLoadCommand(std.macho.uuid_command, .{
        .uuid = @splat(0),
    });

    try macho.updateSegmentLoadAddresses();
}
fn addInternalSymbol(macho: *MachO, opts: struct {
    name: []const u8,
    value: Symbol.InitValue,
}) Error!Symbol.Index {
    const gpa = macho.base.comp.gpa;

    try macho.symtab.ensureUnusedCapacity(gpa, 1);
    try macho.node.symtab.ensureMinimumSize(gpa, &macho.mf, (macho.symtab.items.len + 1) * @sizeOf(std.macho.nlist_64));

    const strx = try macho.string(opts.name);

    const symtab_ptr = macho.lc.symtab.ptr(macho, std.macho.symtab_command);
    assert(macho.targetLoad(&symtab_ptr.nsyms) == macho.symtab.items.len);
    macho.targetStore(&symtab_ptr.nsyms, @intCast(macho.symtab.items.len + 1));

    {
        const dysymtab_cmd = macho.lc.dysymtab.ptr(macho, std.macho.dysymtab_command);
        // See comment in `initInner` (around the creation of this `dysymtab_command`) for details
        // on why we're doing this.
        macho.targetStore(&dysymtab_cmd.nlocalsym, @intCast(macho.symtab.items.len + 1));
        macho.targetStore(&dysymtab_cmd.iextdefsym, @intCast(macho.symtab.items.len + 1));
        macho.targetStore(&dysymtab_cmd.iundefsym, @intCast(macho.symtab.items.len + 1));
    }

    const sym_index: Symbol.Index = @fromBackingInt(@intCast(macho.symtab.items.len));
    macho.symtab.appendAssumeCapacity(.{
        .node = switch (opts.value) {
            .abs => .none,
            .addr_in_node => |addr| .wrap(addr.node),
        },
        .prev_in_node = .none,
        .next_in_node = .none,
        .first_target_reloc = .none,
    });
    if (sym_index.ptr(macho).node.unwrap()) |node| {
        const head_ptr = macho.nodeFirstSymbolPtr(node);
        sym_index.ptr(macho).next_in_node = head_ptr.*;
        if (head_ptr.*.unwrap()) |next_symbol| {
            next_symbol.ptr(macho).prev_in_node = .wrap(sym_index);
        }
        head_ptr.* = .wrap(sym_index);
    }

    sym_index.nlist(macho).* = .{
        .n_strx = @backingInt(strx),
        .n_type = .{ .bits = .{
            .ext = false,
            .type = switch (opts.value) {
                .abs => .abs,
                .addr_in_node => .sect,
            },
            .pext = false,
            .is_stab = 0,
        } },
        .n_sect = switch (opts.value) {
            .abs => 0,
            .addr_in_node => |addr| macho.getNodeSection(addr.node).index(macho),
        },
        .n_desc = .{
            .arm_thumb_def = false,
            .referenced_dynamically = false,
            .discarded_or_no_dead_strip = false,
            .weak_ref = false,
            .weak_def_or_ref_to_weak = false,
            .symbol_resolver = false,
            .alt_entry = false,
        },
        .n_value = switch (opts.value) {
            .abs => |x| x,
            .addr_in_node => |addr| macho.getNodeVAddr(addr.node) + addr.offset,
        },
    };
    if (macho.targetEndian() != std.lang.Endian.native) {
        std.mem.byteSwapAllFields(std.macho.nlist_64, sym_index.nlist(macho));
    }

    return sym_index;
}
fn addGlobalSymbol(macho: *MachO, opts: struct {
    name: []const u8,
    linkage: std.lang.GlobalLinkage,
    visibility: enum { default, hidden },
    /// `null` means the symbol is undefined.
    value: ?Symbol.InitValue,
}) (Error || error{MultipleDefinitions})!Symbol.Index {
    const gpa = macho.base.comp.gpa;

    try macho.symtab.ensureUnusedCapacity(gpa, 1);
    try macho.node.symtab.ensureMinimumSize(gpa, &macho.mf, (macho.symtab.items.len + 1) * @sizeOf(std.macho.nlist_64));

    const adapter: Symbol.GlobalAdapter = .{ .macho = macho };
    const gop = try macho.global_symbols.getOrPutAdapted(gpa, opts.name, adapter);
    if (gop.found_existing) {
        const symbol = gop.key_ptr.*;

        switch (opts.visibility) {
            .default => {},
            .hidden => {
                var n_type = macho.targetLoad(&symbol.nlist(macho).n_type);
                n_type.bits.pext = true;
                switch (macho.machFiletype()) {
                    .OBJECT => {},
                    .EXECUTE, .DYLIB => n_type.bits.ext = false,
                }
                macho.targetStore(&symbol.nlist(macho).n_type, n_type);
                // TODO: once the exports trie is implemented, remove `symbol` from it here
            },
        }

        const existing_defined: bool, const existing_linkage: std.lang.GlobalLinkage = existing: {
            const n_type = macho.targetLoad(&symbol.nlist(macho).n_type);
            const n_desc = macho.targetLoad(&symbol.nlist(macho).n_desc);
            break :existing switch (n_type.bits.type) {
                .undf => .{ false, if (n_desc.weak_ref) .weak else .strong },
                else => .{ true, if (n_desc.weak_def_or_ref_to_weak) .weak else .strong },
            };
        };

        const override: bool = if (opts.value != null) override: {
            if (!existing_defined) break :override true;
            // Two definitions of the same name!
            break :override switch (opts.linkage) {
                .strong => switch (existing_linkage) {
                    .strong => return error.MultipleDefinitions,
                    .weak => true,
                },
                .weak => false,
            };
        } else override: {
            if (existing_defined) break :override false;
            break :override switch (opts.linkage) {
                .strong => switch (existing_linkage) {
                    .strong => false,
                    .weak => true,
                },
                .weak => false,
            };
        };

        if (override) {
            try macho.setGlobalSymbolValue(symbol, .{
                .linkage = opts.linkage,
                .value = opts.value,
            });
        }

        return symbol;
    }
    errdefer _ = macho.global_symbols.pop().?;

    const strx = try macho.string(opts.name);

    const symtab_ptr = macho.lc.symtab.ptr(macho, std.macho.symtab_command);
    assert(macho.targetLoad(&symtab_ptr.nsyms) == macho.symtab.items.len);
    macho.targetStore(&symtab_ptr.nsyms, @intCast(macho.symtab.items.len + 1));

    {
        const dysymtab_cmd = macho.lc.dysymtab.ptr(macho, std.macho.dysymtab_command);
        // See comment in `initInner` (around the creation of this `dysymtab_command`) for details
        // on why we're doing this.
        macho.targetStore(&dysymtab_cmd.nlocalsym, @intCast(macho.symtab.items.len + 1));
        macho.targetStore(&dysymtab_cmd.iextdefsym, @intCast(macho.symtab.items.len + 1));
        macho.targetStore(&dysymtab_cmd.iundefsym, @intCast(macho.symtab.items.len + 1));
    }

    const symbol: Symbol.Index = @fromBackingInt(@intCast(macho.symtab.items.len));
    gop.key_ptr.* = symbol;
    macho.symtab.appendAssumeCapacity(.{
        .node = .none,
        .prev_in_node = .none,
        .next_in_node = .none,
        .first_target_reloc = .none,
    });
    symbol.nlist(macho).* = .{
        .n_strx = @backingInt(strx),
        .n_type = .{ .bits = .{
            .ext = switch (macho.machFiletype()) {
                .OBJECT => true,
                .EXECUTE, .DYLIB => switch (opts.visibility) {
                    .default => true,
                    .hidden => false,
                },
            },
            .pext = switch (opts.visibility) {
                .default => false,
                .hidden => true,
            },
            .type = .undf,
            .is_stab = 0,
        } },
        .n_sect = 0,
        .n_desc = .{
            .arm_thumb_def = false,
            .referenced_dynamically = false,
            .discarded_or_no_dead_strip = false,
            .weak_ref = false,
            .weak_def_or_ref_to_weak = false,
            .symbol_resolver = false,
            .alt_entry = false,
        },
        .n_value = 0,
    };
    if (macho.targetEndian() != std.lang.Endian.native) {
        std.mem.byteSwapAllFields(std.macho.nlist_64, symbol.nlist(macho));
    }

    try macho.setGlobalSymbolValue(symbol, .{
        .linkage = opts.linkage,
        .value = opts.value,
    });

    return symbol;
}
fn setGlobalSymbolValue(macho: *MachO, symbol: Symbol.Index, opts: struct {
    linkage: std.lang.GlobalLinkage,
    value: ?Symbol.InitValue,
}) Allocator.Error!void {
    const gpa = macho.base.comp.gpa;

    log.debug("setGlobalSymbolValue({q}, {?any})", .{ symbol.name(macho), opts.value });

    const nlist = symbol.nlist(macho);

    {
        const weak: bool = switch (opts.linkage) {
            .strong => false,
            .weak => true,
        };
        var n_desc = macho.targetLoad(&nlist.n_desc);
        n_desc.weak_ref = opts.value == null and weak;
        n_desc.weak_def_or_ref_to_weak = opts.value != null and weak;
        macho.targetStore(&nlist.n_desc, n_desc);
    }

    // Remove this symbol from its old node.
    if (symbol.ptr(macho).node.unwrap()) |old_node| {
        if (symbol.ptr(macho).next_in_node.unwrap()) |next_symbol| {
            assert(next_symbol.ptr(macho).prev_in_node == Symbol.Index.Optional.wrap(symbol));
            next_symbol.ptr(macho).prev_in_node = symbol.ptr(macho).prev_in_node;
        }
        if (symbol.ptr(macho).prev_in_node.unwrap()) |prev_symbol| {
            assert(prev_symbol.ptr(macho).next_in_node == Symbol.Index.Optional.wrap(symbol));
            prev_symbol.ptr(macho).next_in_node = symbol.ptr(macho).next_in_node;
        } else {
            // We're the head of the list.
            macho.nodeFirstSymbolPtr(old_node).* = symbol.ptr(macho).next_in_node;
        }
    }

    if (opts.value) |value| switch (value) {
        .abs => |val| {
            var n_type = macho.targetLoad(&nlist.n_type);
            n_type.bits.type = .abs;
            macho.targetStore(&nlist.n_type, n_type);

            macho.targetStore(&nlist.n_sect, 0);
            macho.targetStore(&nlist.n_value, val);

            symbol.ptr(macho).node = .none;
            symbol.ptr(macho).prev_in_node = .none;
            symbol.ptr(macho).next_in_node = .none;
        },
        .addr_in_node => |addr| {
            var n_type = macho.targetLoad(&nlist.n_type);
            n_type.bits.type = .sect;
            macho.targetStore(&nlist.n_type, n_type);

            macho.targetStore(&nlist.n_sect, macho.getNodeSection(addr.node).index(macho));
            macho.targetStore(&nlist.n_value, macho.getNodeVAddr(addr.node) + addr.offset);

            const head_ptr = macho.nodeFirstSymbolPtr(addr.node);
            symbol.ptr(macho).node = .wrap(addr.node);
            symbol.ptr(macho).prev_in_node = .none;
            symbol.ptr(macho).next_in_node = head_ptr.*;
            if (head_ptr.*.unwrap()) |next_symbol| {
                next_symbol.ptr(macho).prev_in_node = .wrap(symbol);
            }
            head_ptr.* = .wrap(symbol);
        },
    } else {
        var n_type = macho.targetLoad(&nlist.n_type);
        n_type.bits.type = .undf;
        macho.targetStore(&nlist.n_type, n_type);

        macho.targetStore(&nlist.n_sect, 0);
        macho.targetStore(&nlist.n_value, 0);

        symbol.ptr(macho).node = .none;
        symbol.ptr(macho).prev_in_node = .none;
        symbol.ptr(macho).next_in_node = .none;
    }

    // If `symbol` has a GOT entry, it needs regenerating.
    if (macho.got.getIndex(symbol)) |got_index| {
        if (!macho.known_sections.got.node(macho).hasMoved(&macho.mf)) {
            const segment = macho.known_sections.got.segment(macho);
            const page_index: u16 = @intCast(@divFloor(
                macho.prev_got_offset + got_index * macho.targetPtrSize(),
                macho.targetPageAlign().toByteUnits(),
            ));
            try macho.dirty_fixup_chain_pages.put(gpa, .{ .seg = segment, .page = page_index }, {});
        }
    }

    // If `symbol` has an `imports` entry, it should not exist.
    if (macho.imports.getIndex(symbol)) |import_index| {
        // TODO: delete the import. For now we just mark it as weak so that dyld doesn't complain.
        const imports: []std.macho.dyld_chained_import_addend64 = @ptrCast(@alignCast(
            macho.node.fixup_chains_imports.slice(&macho.mf)[0 .. macho.imports.count() * @sizeOf(std.macho.dyld_chained_import_addend64)],
        ));
        macho.targetStore(&imports[import_index].bits, .{
            .lib_ordinal = macho.targetLoad(&imports[import_index].bits).lib_ordinal,
            .name_offset = macho.targetLoad(&imports[import_index].bits).name_offset,
            .weak_import = true,
        });
    }
}
fn globalByName(macho: *MachO, name: []const u8) Symbol.Index.Optional {
    const adapter: Symbol.GlobalAdapter = .{ .macho = macho };
    if (macho.global_symbols.getKeyAdapted(name, adapter)) |symbol| {
        return .wrap(symbol);
    } else {
        return .none;
    }
}

/// All calls to this function must occur in `initInner`, and must be before we initialize fixup
/// chains, as fixup chains requires some per-segment tracking and the logic to initialize them
/// assumes that no more segments will be created.
fn addSegment(macho: *MachO, opts: struct {
    name: []const u8,
    prot: std.macho.vm_prot_t,
    /// The caller is allowed to create the section's node themselves, in which case it is passed in
    /// here. The caller must initialize the `MachO.nodes` entry to `.{ .segment = undefined }`, and
    /// the node must be created with the option `.moved = true`.
    existing_node: ?MappedFile.Node.Index = null,
    /// This matches the layout of `std.macho.segment_command_64.flags`.
    flags: packed struct(u32) {
        _unused0: u4 = 0,
        read_only: bool = false,
        _unused1: u27 = 0,
    },
}) Error!Segment {
    const gpa = macho.base.comp.gpa;

    const segment_command_64 = std.macho.segment_command_64;

    // We limit the number of segments to 255, because there can only be 255 *sections*, so there's
    // no need to allow more. See corresponding check in `addSection`.
    if (macho.segments.len == 255) {
        return macho.base.comp.link_diags.fail("maximum segment count exceeded", .{});
    }

    try macho.nodes.ensureUnusedCapacity(gpa, 1);
    try macho.segments.ensureUnusedCapacity(gpa, 1);

    const root_ni: MappedFile.Node.Index = .root;

    const lc = try macho.appendSimpleLoadCommand(std.macho.segment_command_64, .{
        .cmdsize = @sizeOf(segment_command_64),
        .segname = name: {
            var name: [16]u8 = undefined;
            @memcpy(name[0..opts.name.len], opts.name);
            @memset(name[opts.name.len..], 0);
            break :name name;
        },
        .vmaddr = 0,
        .vmsize = 0,
        .fileoff = 0, // to be set by `flushMoved`
        .filesize = 0,
        .maxprot = opts.prot,
        .initprot = opts.prot,
        .nsects = 0,
        .flags = @bitCast(opts.flags),
    });

    const segment: Segment = @fromBackingInt(@intCast(macho.segments.len));
    const segment_node: MappedFile.Node.Index = node: {
        if (opts.existing_node) |node| {
            assert(node.hasMoved(&macho.mf)); // so that `fileoff` will be set
            assert(macho.nodes.items(.tags)[@backingInt(node)] == .segment);
            macho.nodes.set(@backingInt(node), .{ .segment = segment });
            break :node node;
        }
        const node = try root_ni.addFloatingChild(gpa, &macho.mf, .{
            .alignment = macho.mf.flags.block_size.max(macho.targetPageAlign()),
            .moved = true, // so that `fileoff` will be set
            .bubbles_moved = false,
        });
        macho.nodes.appendAssumeCapacity(.{ .segment = segment });
        break :node node;
    };
    macho.segments.appendAssumeCapacity(.{
        .lc = lc,
        .node = segment_node,
        .fixup_chain_starts = .none,
    });

    return segment;
}

const String = enum(u32) {
    empty = 0,
    _,
    fn slice(s: String, macho: *const MachO) [:0]const u8 {
        const overlong = macho.node.strtab.slice(&macho.mf)[@backingInt(s)..];
        return overlong[0..std.mem.findScalar(u8, overlong, 0).? :0];
    }

    const Adapter = struct {
        macho: *const MachO,
        pub fn eql(ctx: Adapter, lhs_slice: []const u8, rhs: String, _: usize) bool {
            const macho = ctx.macho;
            return std.mem.eql(u8, lhs_slice, rhs.slice(macho));
        }
        pub fn hash(ctx: Adapter, s: []const u8) u32 {
            _ = ctx;
            return std.array_hash_map.hashString(s);
        }
    };
};

fn string(macho: *MachO, slice: []const u8) Error!String {
    const gpa = macho.base.comp.gpa;

    const old_strsize = macho.targetLoad(
        &macho.lc.symtab.ptr(macho, std.macho.symtab_command).strsize,
    );
    try macho.node.strtab.ensureMinimumSize(gpa, &macho.mf, old_strsize + slice.len + 1);

    try macho.strtab.ensureUnusedCapacity(gpa, 1);

    errdefer comptime unreachable;

    const adapter: String.Adapter = .{ .macho = macho };
    const gop = macho.strtab.getOrPutAssumeCapacityAdapted(slice, adapter);
    if (!gop.found_existing) {
        macho.targetStore(
            &macho.lc.symtab.ptr(macho, std.macho.symtab_command).strsize,
            @intCast(old_strsize + slice.len + 1),
        );
        gop.key_ptr.* = @fromBackingInt(old_strsize);
        const dest_slice = macho.node.strtab.slice(&macho.mf)[old_strsize..];
        @memcpy(dest_slice[0..slice.len], slice);
        dest_slice[slice.len] = 0;
    }
    return gop.key_ptr.*;
}

fn updateSegmentLoadAddresses(macho: *MachO) Error!void {
    const gpa = macho.base.comp.gpa;

    // Grow all `.fixup_chains_starts_in_segment` node to the necessary length. We need to do this
    // before updating the load addresses because this could cause `__LINKEDIT` to grow.
    for (0..macho.segments.len) |segment_raw| {
        const segment: Segment = @fromBackingInt(@intCast(segment_raw));

        const starts_node = segment.fixupChainStartsNode(macho).unwrap() orelse {
            assert(std.mem.eql(u8, segment.name(macho), "__LINKEDIT"));
            continue;
        };

        const size = segment.node(macho).location(&macho.mf).resolve(&macho.mf)[1];

        const page_count: u16 = @intCast(@divExact(size, macho.targetPageAlign().toByteUnits()));
        try starts_node.ensureMinimumSize(gpa, &macho.mf, MappedFile.Alignment.@"8".forward(
            @sizeOf(std.macho.dyld_chained_starts_in_segment) + page_count * 2,
        ));
        const starts_hdr: *std.macho.dyld_chained_starts_in_segment = @ptrCast(@alignCast(
            starts_node.slice(&macho.mf)[0..@sizeOf(std.macho.dyld_chained_starts_in_segment)],
        ));
        const old_page_count = macho.targetLoad(&starts_hdr.page_count);
        macho.targetStore(&starts_hdr.page_count, page_count);
        macho.targetStore(&starts_hdr.size, @sizeOf(std.macho.dyld_chained_starts_in_segment) + 2 * page_count);
        const page_starts: []u16 = @ptrCast(@alignCast(
            starts_node.slice(&macho.mf)[@sizeOf(std.macho.dyld_chained_starts_in_segment) - 2 ..],
        ));
        @memset(page_starts[old_page_count..page_count], 0xFFFF);
    }

    // TODO: this segment load address allocation logic is bad because it moves segments quite
    // frequently. We should avoid that by introducing padding between segments so that they have
    // space to grow in-place. We could also consider allowing a segment which can't fit to jump
    // over another segment, like in `Elf2.allocateSegmentLoadAddress`.
    var cur_addr: u64 = start_addr: {
        const pagezero_ptr = macho.lc.pagezero.ptr(macho, std.macho.segment_command_64);
        assert(macho.targetLoad(&pagezero_ptr.vmaddr) == 0);
        break :start_addr macho.targetLoad(&pagezero_ptr.vmsize);
    };

    for (0..macho.segments.len) |segment_raw| {
        const segment: Segment = @fromBackingInt(@intCast(segment_raw));

        _, const size: u64 = segment.node(macho).location(&macho.mf).resolve(&macho.mf);
        const cmd_ptr = segment.lcPtr(macho);

        macho.targetStore(&cmd_ptr.vmsize, size);

        const old_addr = macho.targetLoad(&cmd_ptr.vmaddr);
        if (old_addr >= cur_addr) {
            cur_addr = old_addr;
        } else {
            macho.targetStore(&cmd_ptr.vmaddr, cur_addr);
            try segment.node(macho).childrenMoved(gpa, &macho.mf);
        }

        cur_addr = macho.targetPageAlign().forward(cur_addr + size);
    }
}

/// Like `appendLoadCommand`, but asserts that `cmd.cmdsize` exactly equals `@sizeOf(Cmd)` and does
/// not return the (empty) trailing slice.
fn appendSimpleLoadCommand(
    macho: *MachO,
    comptime Cmd: type,
    cmd: Cmd,
) Error!LoadCommand {
    comptime assert(@sizeOf(Cmd) % 8 == 0);
    assert(cmd.cmdsize == @sizeOf(Cmd));
    return (try macho.appendLoadCommand(Cmd, cmd)).lc;
}

/// Asserts that the first two fields of `Cmd` are `cmd: LC` and `cmdsize: u32`, and that the value
/// of `lc.cmdsize` is 8-byte aligned.
fn appendLoadCommand(
    macho: *MachO,
    comptime Cmd: type,
    cmd: Cmd,
) Error!struct {
    lc: LoadCommand,
    trailing: []u8,
} {
    const gpa = macho.base.comp.gpa;

    try macho.lc.offsets.ensureUnusedCapacity(gpa, 1);

    comptime {
        const s = @typeInfo(Cmd).@"struct";
        assert(s.layout == .@"extern");
        // First field of load command should be `cmd: LC`
        assert(std.mem.eql(u8, s.field_names[0], "cmd"));
        assert(s.field_types[0] == std.macho.LC);
        // Second field of load command should be `cmdsize: u32`
        assert(std.mem.eql(u8, s.field_names[1], "cmdsize"));
        assert(s.field_types[1] == u32);
    }

    const cmd_size = @as(*const std.macho.load_command, @ptrCast(&cmd)).cmdsize;
    assert(cmd_size % 8 == 0);

    const mach_header_64 = std.macho.mach_header_64;
    const old_cmds_size: u32 = size: {
        const mach_header: *const mach_header_64 = @ptrCast(@alignCast(
            macho.node.header.sliceConst(&macho.mf)[0..@sizeOf(mach_header_64)],
        ));
        break :size macho.targetLoad(&mach_header.sizeofcmds);
    };
    const new_cmds_size = old_cmds_size + cmd_size;
    try macho.node.header.ensureMinimumSize(gpa, &macho.mf, @sizeOf(mach_header_64) + new_cmds_size);

    const header_slice = macho.node.header.slice(&macho.mf);
    const mach_header: *mach_header_64 = @ptrCast(@alignCast(
        header_slice[0..@sizeOf(mach_header_64)],
    ));
    macho.targetStore(&mach_header.sizeofcmds, new_cmds_size);
    macho.targetStore(&mach_header.ncmds, macho.targetLoad(&mach_header.ncmds) + 1);

    const cmd_slice: []align(8) u8 = @alignCast(
        header_slice[@sizeOf(mach_header_64) + old_cmds_size ..][0..cmd_size],
    );

    const cmd_ptr: *Cmd = @ptrCast(cmd_slice[0..@sizeOf(Cmd)]);
    cmd_ptr.* = cmd;
    if (macho.targetEndian() != std.lang.Endian.native) {
        std.mem.byteSwapAllFields(Cmd, cmd_ptr);
    }

    const lc: LoadCommand = @fromBackingInt(@intCast(macho.lc.offsets.items.len));
    macho.lc.offsets.appendAssumeCapacity(@sizeOf(mach_header_64) + old_cmds_size);

    return .{ .lc = lc, .trailing = cmd_slice[@sizeOf(Cmd)..] };
}

fn addSection(macho: *MachO, opts: struct {
    name: []const u8,
    segment: Segment,
    @"align": MappedFile.Alignment,
    manual_size: bool = false,
    flags: Section.Flags,
    reserved1: u32 = 0,
    reserved2: u32 = 0,
}) Error!Section {
    const gpa = macho.base.comp.gpa;

    const mach_header_64 = std.macho.mach_header_64;
    const segment_command_64 = std.macho.segment_command_64;
    const section_64 = std.macho.section_64;

    // Check if we can actually fit another section. The limit is 255 because Mach-O implicitly
    // assigns 8-bit indices to sections starting from 1 (0 is reserved).
    if (macho.sections.len == 255) {
        return macho.base.comp.link_diags.fail("maximum section count exceeded", .{});
    }

    const old_cmds_size: u32 = size: {
        const mach_header: *const mach_header_64 = @ptrCast(@alignCast(
            macho.node.header.sliceConst(&macho.mf)[0..@sizeOf(mach_header_64)],
        ));
        break :size macho.targetLoad(&mach_header.sizeofcmds);
    };
    const new_cmds_size = old_cmds_size + @sizeOf(section_64);
    try macho.node.header.ensureMinimumSize(gpa, &macho.mf, @sizeOf(mach_header_64) + new_cmds_size);

    try macho.nodes.ensureUnusedCapacity(gpa, 1);
    try macho.lc.offsets.ensureUnusedCapacity(gpa, 1);
    try macho.sections.ensureUnusedCapacity(gpa, 1);
    try macho.sections_by_name.ensureUnusedCapacity(gpa, 1);

    const section: Section = @fromBackingInt(@intCast(macho.sections.len));
    const section_node = try opts.segment.node(macho).addFloatingChild(gpa, &macho.mf, .{
        .alignment = macho.mf.flags.block_size.max(opts.@"align"),
        .moved = true, // so that `offset` will be set
    });
    if (opts.manual_size) {
        macho.nodes.appendAssumeCapacity(.{ .section_manual_size = section });
    } else {
        macho.nodes.appendAssumeCapacity(.{ .section = section });
    }
    macho.sections.appendAssumeCapacity(.{
        .lc = @fromBackingInt(@intCast(macho.lc.offsets.items.len)),
        .node = section_node,
        .segment = opts.segment,
        .index = undefined, // populated later
    });

    {
        const adapter: Section.NameAdapter = .{ .macho = macho };
        const key: Section.NameAdapter.SegmentAndSection = .{
            .segment = opts.segment.name(macho),
            .section = opts.name,
        };
        const gop = macho.sections_by_name.getOrPutAssumeCapacityAdapted(key, adapter);
        assert(!gop.found_existing);
        assert(gop.index == @backingInt(section));
    }

    const header_slice = macho.node.header.slice(&macho.mf);

    {
        const mach_header: *mach_header_64 = @ptrCast(@alignCast(
            header_slice[0..@sizeOf(mach_header_64)],
        ));
        macho.targetStore(&mach_header.sizeofcmds, new_cmds_size);
    }

    const seg_cmd = opts.segment.lcPtr(macho);
    const old_nsects = macho.targetLoad(&seg_cmd.nsects);
    const old_cmdsize = macho.targetLoad(&seg_cmd.cmdsize);
    assert(old_cmdsize == @sizeOf(segment_command_64) + old_nsects * @sizeOf(section_64));
    const new_cmdsize = old_cmdsize + @sizeOf(section_64);
    macho.targetStore(&seg_cmd.cmdsize, new_cmdsize);
    macho.targetStore(&seg_cmd.nsects, old_nsects + 1);

    const seg_cmd_off = opts.segment.lc(macho).offset(macho);
    const offset = seg_cmd_off + old_cmdsize;

    // Shift forward every load command which comes after this segment command, to make space
    // for the new `section_64` (and update the offsets in `macho.lc.offsets` accordingly).
    @memmove(
        header_slice[seg_cmd_off + new_cmdsize .. @sizeOf(mach_header_64) + new_cmds_size],
        header_slice[seg_cmd_off + old_cmdsize .. @sizeOf(mach_header_64) + old_cmds_size],
    );
    for (macho.lc.offsets.items) |*other_off| {
        if (other_off.* >= offset) {
            other_off.* += @sizeOf(section_64);
        }
    }
    // Only now do we append our *own* offset (otherwise the above loop would have changed it!).
    macho.lc.offsets.appendAssumeCapacity(offset);

    // Increment the index of every section after where we've inserted ourselves.
    {
        var new_section_index: u8 = 1;
        try macho.sections_changed_index.ensureUnusedCapacity(gpa, macho.sections.len);
        for (0..macho.sections.len) |other_section_raw| {
            const other_section: Section = @fromBackingInt(@intCast(other_section_raw));
            if (other_section == section) continue;
            if (other_section.lc(macho).offset(macho) > offset) {
                macho.sections.items(.index)[@backingInt(other_section)] += 1;
                macho.sections_changed_index.putAssumeCapacity(other_section, {});
            } else {
                new_section_index += 1;
            }
        }
        macho.sections.items(.index)[@backingInt(section)] = new_section_index;
    }

    // Write the new `section_64`.
    const section_cmd: *section_64 = @ptrCast(@alignCast(
        header_slice[offset..][0..@sizeOf(section_64)],
    ));
    section_cmd.* = .{
        .sectname = name: {
            var name: [16]u8 = undefined;
            @memcpy(name[0..opts.name.len], opts.name);
            @memset(name[opts.name.len..], 0);
            break :name name;
        },
        .segname = seg_cmd.segname,
        .addr = 0,
        .size = 0,
        .offset = 0,
        .@"align" = opts.@"align".toLog2Units(),
        .reloff = 0,
        .nreloc = 0,
        .flags = @bitCast(opts.flags),
        .reserved1 = opts.reserved1,
        .reserved2 = opts.reserved2,
    };
    if (macho.targetEndian() != std.lang.Endian.native) {
        std.mem.byteSwapAllFields(section_64, section_cmd);
    }

    return section;
}

fn sectionByName(macho: *const MachO, segment_name: []const u8, section_name: []const u8) ?Section {
    const adapter: Section.NameAdapter = .{ .macho = macho };
    const key: Section.NameAdapter.SegmentAndSection = .{
        .segment = segment_name,
        .section = section_name,
    };
    if (macho.sections_by_name.getIndexAdapted(key, adapter)) |section_raw| {
        return @fromBackingInt(@intCast(section_raw));
    } else {
        return null;
    }
}

fn machFiletype(macho: *const MachO) enum(u32) {
    OBJECT = std.macho.MH_OBJECT,
    DYLIB = std.macho.MH_DYLIB,
    EXECUTE = std.macho.MH_EXECUTE,
} {
    const hdr: *const std.macho.mach_header_64 = @ptrCast(@alignCast(
        macho.node.header.sliceConst(&macho.mf)[0..@sizeOf(std.macho.mach_header_64)],
    ));
    return @fromBackingInt(macho.targetLoad(&hdr.filetype));
}

fn targetEndian(macho: *const MachO) std.lang.Endian {
    _ = macho;
    return .little;
}
fn targetPtrSize(macho: *const MachO) u8 {
    _ = macho;
    return 8;
}
fn targetPageAlign(macho: *const MachO) MappedFile.Alignment {
    return switch (macho.base.comp.getTarget().cpu.arch) {
        else => unreachable,
        .aarch64 => .fromByteUnits(0x4000),
        .x86_64 => .fromByteUnits(0x1000),
    };
}
fn targetLoad(macho: *const MachO, ptr: anytype) @typeInfo(@TypeOf(ptr)).pointer.child {
    const pointer_ty = @typeInfo(@TypeOf(ptr)).pointer;
    const Child = pointer_ty.child;
    const alignment = pointer_ty.attrs.@"align" orelse @alignOf(Child);
    return switch (@typeInfo(Child)) {
        else => @compileError(@typeName(Child)),
        .int => std.mem.toNative(Child, ptr.*, macho.targetEndian()),
        .@"enum" => |@"enum"| @fromBackingInt(macho.targetLoad(@as(*align(alignment) const @"enum".tag_type, @ptrCast(ptr)))),
        inline .@"struct", .@"union" => |bitpack| @bitCast(
            macho.targetLoad(@as(*align(alignment) const bitpack.backing_integer.?, @ptrCast(ptr))),
        ),
    };
}
fn targetStore(macho: *const MachO, ptr: anytype, val: @typeInfo(@TypeOf(ptr)).pointer.child) void {
    const pointer_ty = @typeInfo(@TypeOf(ptr)).pointer;
    const Child = pointer_ty.child;
    const alignment = pointer_ty.attrs.@"align" orelse @alignOf(Child);
    return switch (@typeInfo(Child)) {
        else => @compileError(@typeName(Child)),
        .int => ptr.* = std.mem.nativeTo(Child, val, macho.targetEndian()),
        .@"enum" => |@"enum"| macho.targetStore(
            @as(*align(alignment) @"enum".tag_type, @ptrCast(ptr)),
            @backingInt(val),
        ),
        inline .@"struct", .@"union" => |bitpack| macho.targetStore(
            @as(*align(alignment) bitpack.backing_integer.?, @ptrCast(ptr)),
            @bitCast(val),
        ),
    };
}

pub fn deinit(macho: *MachO) void {
    const gpa = macho.base.comp.gpa;
    macho.mf.deinit(gpa);
    macho.nodes.deinit(gpa);
    macho.segments.deinit(gpa);
    macho.sections.deinit(gpa);
    macho.sections_by_name.deinit(gpa);
    macho.dirty_fixup_chain_pages.deinit(gpa);
    macho.input_atoms.deinit(gpa);
    macho.strtab.deinit(gpa);
    macho.symtab.deinit(gpa);
    macho.global_symbols.deinit(gpa);
    macho.relocs.deinit(gpa);
    macho.imports.deinit(gpa);
    macho.got.deinit(gpa);
    macho.stubs.deinit(gpa);
    macho.sections_changed_index.deinit(gpa);
    macho.* = undefined;
}

pub fn startProgress(macho: *MachO, prog_node: std.Progress.Node) void {
    prog_node.increaseEstimatedTotalItems(4);
    macho.const_prog_node = prog_node.start("Constants", 0);
    macho.synth_prog_node = prog_node.start("Synthetics", 0);
    macho.input_prog_node = prog_node.start("Inputs", 0);
    macho.mf.update_prog_node = prog_node.start("Relocations", macho.mf.updates.items.len);
}
pub fn endProgress(macho: *MachO) void {
    macho.const_prog_node.end();
    macho.const_prog_node = .none;
    macho.synth_prog_node.end();
    macho.synth_prog_node = .none;
    macho.input_prog_node.end();
    macho.input_prog_node = .none;
    macho.mf.update_prog_node.end();
    macho.mf.update_prog_node = .none;
}

pub fn navSymbol(macho: *MachO, nav_id: InternPool.Nav.Index) link.Error!link.File.SymbolId {
    _ = macho;
    _ = nav_id;
    @panic("TODO");
}
pub fn uavSymbol(
    macho: *MachO,
    pt: Zcu.PerThread,
    uav_val: InternPool.Index,
    uav_align: InternPool.Alignment,
) link.Error!link.File.SymbolId {
    _ = macho;
    _ = pt;
    _ = uav_val;
    _ = uav_align;
    @panic("TODO");
}
pub fn relocSymAddr(
    macho: *MachO,
    reloc_info: link.File.RelocInfo,
) link.Error!void {
    _ = macho;
    _ = reloc_info;
    @panic("TODO");
}

pub fn loadInput(macho: *MachO, input: link.Input) link.Error!void {
    const diags = &macho.base.comp.link_diags;
    macho.loadInputInner(input) catch |err| switch (err) {
        error.MappedFileIo => return diags.fail("failed to write output file: {t}", .{macho.mf.io_err.?}),
        else => |e| return e,
    };
    macho.updateSegmentLoadAddresses() catch |err| switch (err) {
        error.MappedFileIo => return diags.fail("failed to write output file: {t}", .{macho.mf.io_err.?}),
        else => |e| return e,
    };
}
fn loadInputInner(macho: *MachO, input: link.Input) Error!void {
    const comp = macho.base.comp;
    const diags = &comp.link_diags;
    const gpa = comp.gpa;
    const io = comp.io;
    switch (input) {
        .res => unreachable,
        .object => |object| {
            var buf: [4096]u8 = undefined;
            var fr = object.file.reader(io, &buf);
            macho.loadObject(object.path, null, &fr, .{
                .offset = fr.logicalPos(),
                .size = fr.getSize() catch |err| switch (err) {
                    error.Canceled => |e| return e,
                    else => |e| return diags.fail(
                        "failed to stat \"{f}\": {t}",
                        .{ object.path.fmtEscapeString(), e },
                    ),
                },
            }) catch |err| switch (err) {
                else => |e| return e,
                error.EndOfStream => return diags.failParse(
                    object.path,
                    "unexpected eof",
                    .{},
                ),
                error.AccessDenied, error.Unexpected, error.Unseekable => |e| return diags.fail(
                    "failed to read \"{f}\": {t}",
                    .{ object.path.fmtEscapeString(), e },
                ),
                error.ReadFailed => switch (fr.err.?) {
                    error.Canceled => |e| return e,
                    else => |e| return diags.fail(
                        "failed to read \"{f}\": {t}",
                        .{ object.path.fmtEscapeString(), e },
                    ),
                },
            };
        },
        .archive => |archive| {
            var buf: [4096]u8 = undefined;
            var fr = archive.file.reader(io, &buf);
            macho.loadArchive(archive.path, &fr) catch |err| switch (err) {
                else => |e| return e,
                error.EndOfStream => return diags.failParse(
                    archive.path,
                    "unexpected eof",
                    .{},
                ),
                error.AccessDenied, error.Unexpected, error.Unseekable => |e| return diags.fail(
                    "failed to read \"{f}\": {t}",
                    .{ archive.path.fmtEscapeString(), e },
                ),
                error.ReadFailed => switch (fr.err.?) {
                    error.Canceled => |e| return e,
                    else => |e| return diags.fail(
                        "failed to read \"{f}\": {t}",
                        .{ archive.path.fmtEscapeString(), e },
                    ),
                },
            };
        },
        .dso => @panic("TODO(MachO2): load dso"),
        .tbd => |tbd| {
            log.debug("load tbd {qf}", .{tbd.path.fmtEscapeString()});

            var lib_stub = tapi.LibStub.loadFromFile(gpa, io, tbd.file) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| return diags.failParse(tbd.path, "failed to parse tbd: {t}", .{e}),
            };
            defer lib_stub.deinit();

            const umbrella_lib = lib_stub.inner[0];

            const dylib_name = umbrella_lib.installName();
            const cmd = try macho.appendLoadCommand(std.macho.dylib_command, .{
                .cmd = .LOAD_DYLIB,
                .cmdsize = std.mem.alignForward(u32, @intCast(
                    @sizeOf(std.macho.dylib_command) + dylib_name.len + 1,
                ), 8),
                .dylib = .{
                    .name = @sizeOf(std.macho.dylib_command),
                    .timestamp = 0,
                    .current_version = 0, // TODO
                    .compatibility_version = 0, // TODO
                },
            });
            @memcpy(cmd.trailing[0..dylib_name.len], dylib_name);
            @memset(cmd.trailing[dylib_name.len..], 0);
        },
    }
}

/// Internal error set used by input parsing functions `loadObject`, `loadArchive`, `loadDso`.
const LoadParseInputError = Error || Io.File.SeekError || Io.Reader.Error;

fn loadArchive(macho: *MachO, path: std.Build.Cache.Path, fr: *Io.File.Reader) LoadParseInputError!void {
    const comp = macho.base.comp;
    const gpa = comp.gpa;
    const diags = &comp.link_diags;
    const r = &fr.interface;

    log.debug("loadArchive({f})", .{path.fmtEscapeString()});

    {
        const magic = r.take(std.macho.ARMAG.len) catch |err| switch (err) {
            error.ReadFailed => |e| return e,
            error.EndOfStream => return diags.failParse(path, "invalid archive file", .{}),
        };
        if (!std.mem.eql(u8, magic, std.macho.ARMAG)) {
            return diags.failParse(path, "invalid archive file", .{});
        }
    }

    var long_name_buf: std.ArrayList(u8) = .empty;
    defer long_name_buf.deinit(gpa);

    while (r.takeStruct(std.macho.ar_hdr, .native)) |header| {
        if (!std.mem.eql(u8, &header.ar_fmag, std.macho.ARFMAG))
            return diags.failParse(path, "bad file magic", .{});
        const offset = fr.logicalPos();
        const size = header.size() catch
            return diags.failParse(path, "bad member size", .{});
        const member_name: []const u8, const member_offset: u64, const member_size: u64 = member: {
            const name_len = header.nameLength() orelse {
                break :member .{ header.name(), offset, size };
            };
            try long_name_buf.resize(gpa, name_len);
            try fr.interface.readSliceAll(long_name_buf.items);
            break :member .{
                std.mem.sliceTo(long_name_buf.items, 0),
                offset + name_len,
                size - name_len,
            };
        };
        if (std.mem.eql(u8, member_name, std.macho.SYMDEF) or
            std.mem.eql(u8, member_name, std.macho.SYMDEF64) or
            std.mem.eql(u8, member_name, std.macho.SYMDEF_SORTED) or
            std.mem.eql(u8, member_name, std.macho.SYMDEF64_SORTED))
        {
            // Ignore these special members.
        } else {
            try macho.loadObject(path, member_name, fr, .{
                .offset = member_offset,
                .size = member_size,
            });
        }
        try fr.seekTo(std.mem.alignForward(u64, offset + size, 2));
    } else |err| switch (err) {
        else => |e| return e,
        error.EndOfStream => if (!fr.atEnd()) return error.EndOfStream,
    }
}
fn loadObject(
    macho: *MachO,
    path: std.Build.Cache.Path,
    member: ?[]const u8,
    fr: *Io.File.Reader,
    fl: MappedFile.Node.FileLocation,
) LoadParseInputError!void {
    const comp = macho.base.comp;
    const diags = &comp.link_diags;
    const gpa = comp.gpa;

    log.debug("loadObject({f}{f})", .{ path.fmtEscapeString(), fmtMemberString(member) });

    const target_endian = macho.targetEndian();

    switch (try fr.interface.peekInt(u32, target_endian)) {
        std.macho.MH_MAGIC_64 => {},
        std.macho.MH_MAGIC => {
            return diags.failParse(path, "incorrect pointer bit width (expected 64)", .{});
        },
        std.macho.MH_CIGAM_64, std.macho.MH_CIGAM => {
            return diags.failParse(path, "incorrect endian (expected {t})", .{target_endian});
        },
        else => {
            return diags.failParse(path, "invalid Mach-O object file", .{});
        },
    }

    const header = try fr.interface.takeStruct(std.macho.mach_header_64, target_endian);
    assert(header.magic == std.macho.MH_MAGIC_64); // checked above

    if (header.filetype != std.macho.MH_OBJECT) return diags.failParse(
        path,
        "not a relocatable",
        .{},
    );

    {
        const output_header: *const std.macho.mach_header_64 = @ptrCast(@alignCast(
            macho.node.header.sliceConst(&macho.mf)[0..@sizeOf(std.macho.mach_header_64)],
        ));
        const output_cputype = macho.targetLoad(&output_header.cputype);
        const output_cpusubtype = macho.targetLoad(&output_header.cpusubtype);
        if (header.cputype != output_cputype) return diags.failParse(
            path,
            "cputype '{d}' does not match target cputype '{d}'",
            .{ header.cputype, output_cputype },
        );
        if (header.cpusubtype != output_cpusubtype) return diags.failParse(
            path,
            "cpusubtype '{d}' does not match target cpusubtype '{d}'",
            .{ header.cpusubtype, output_cpusubtype },
        );
    }

    if (@sizeOf(std.macho.mach_header_64) + header.sizeofcmds > fl.size) {
        return diags.failParse(path, "load commands exceed file size", .{});
    }

    var maybe_symtab_lc: ?std.macho.symtab_command = null;

    const InputSection = struct {
        segname: [16]u8,
        sectname: [16]u8,

        reloff: u32,
        nreloc: u32,

        addr: u64,
        size: u64,
        @"align": MappedFile.Alignment,

        usage: Usage,
        file_offset: u32,

        atoms_start: usize,
        atoms_end: usize,

        const Usage = enum {
            text,
            data_const,
            data_const_split_4,
            data_const_split_8,
            data_const_split_16,
            data_const_split_c_strings,
            data_const_fixups,
            data_mut,
            bss,
            fn parentSection(usage: Usage, m: *const MachO) Section {
                return switch (usage) {
                    .text => m.sectionByName("__TEXT", "__text").?,

                    .data_const,
                    .data_const_split_4,
                    .data_const_split_8,
                    .data_const_split_16,
                    .data_const_split_c_strings,
                    => m.sectionByName("__TEXT", "__const").?,

                    // TODO: this should really be "__DATA_CONST,__data". See comment in `initInner`
                    .data_const_fixups => m.sectionByName("__DATA", "__data").?,

                    .data_mut => m.sectionByName("__DATA", "__data").?,
                    .bss => m.sectionByName("__DATA", "__data").?,
                };
            }
        };
    };
    var input_sections: std.ArrayList(InputSection) = .empty;
    defer input_sections.deinit(gpa);

    var cmd_offset: u64 = 0;
    for (0..header.ncmds) |_| {
        const lc = try fr.interface.peekStruct(std.macho.load_command, target_endian);
        if (lc.cmdsize < @sizeOf(std.macho.load_command)) return diags.failParse(
            path,
            "load command at offset 0x{X} has invalid size",
            .{cmd_offset + @sizeOf(std.macho.mach_header_64)},
        );
        if (cmd_offset + lc.cmdsize > header.sizeofcmds) return diags.failParse(
            path,
            "load command at offset 0x{X} exceeds 'sizeofcmds'",
            .{cmd_offset + @sizeOf(std.macho.mach_header_64)},
        );
        cmd_offset += lc.cmdsize;
        switch (lc.cmd) {
            else => if (std.enums.tagName(std.macho.LC, lc.cmd)) |cmd_name| {
                return diags.failParse(path, "unhandled load command 'LC_{s}'", .{cmd_name});
            } else {
                return diags.failParse(path, "unknown load command 0x{X}", .{lc.cmd});
            },

            .NONE,
            .DYSYMTAB,
            .NOTE,
            .BUILD_VERSION,
            .DATA_IN_CODE,
            .LINKER_OPTIMIZATION_HINT,
            => try fr.interface.discardAll(lc.cmdsize), // ignore

            .SEGMENT_64 => {
                if (lc.cmdsize < @sizeOf(std.macho.segment_command_64)) {
                    return diags.failParse(path, "bad cmdsize in 'LC_SEGMENT_64' load command", .{});
                }
                const segment_lc = try fr.interface.takeStruct(std.macho.segment_command_64, target_endian);
                if (lc.cmdsize - @sizeOf(std.macho.segment_command_64) < segment_lc.nsects * @sizeOf(std.macho.section_64)) {
                    return diags.failParse(path, "nsects in 'LC_SEGMENT_64' load command exceeds cmdsize", .{});
                }
                try input_sections.ensureUnusedCapacity(gpa, segment_lc.nsects);
                for (0..segment_lc.nsects) |_| {
                    const section = try fr.interface.takeStruct(std.macho.section_64, target_endian);

                    const flags: Section.Flags = @bitCast(section.flags);

                    const segname = std.mem.sliceTo(&section.segname, 0);
                    const sectname = std.mem.sliceTo(&section.sectname, 0);

                    const usage: InputSection.Usage = switch (flags.type) {
                        _ => return diags.failParse(
                            path,
                            "section '{s},{s}' has unknown type '{d}'",
                            .{ segname, sectname, flags.type },
                        ),

                        else => return diags.failParse(
                            path,
                            "section '{s},{s}' has unsupported type '{t}'",
                            .{ segname, sectname, flags.type },
                        ),

                        .zerofill => .bss,

                        .@"4byte_literals" => .data_const_split_4,
                        .@"8byte_literals" => .data_const_split_8,
                        .@"16byte_literals" => .data_const_split_16,
                        .cstring_literals => .data_const_split_c_strings,

                        .regular => usage: {
                            if (flags.some_instructions or flags.pure_instructions) {
                                break :usage .text;
                            }

                            // "__DATA,__const" is constant data with fixups.
                            if (std.mem.eql(u8, segname, "__DATA") and
                                std.mem.eql(u8, sectname, "__const"))
                            {
                                break :usage .data_const_fixups;
                            }

                            // Otherwise, the decision is based solely on segment name.
                            if (std.mem.eql(u8, segname, "__TEXT")) {
                                break :usage .data_const;
                            }
                            if (std.mem.eql(u8, segname, "__DATA_CONST")) {
                                break :usage .data_const_fixups;
                            }
                            if (std.mem.eql(u8, segname, "__DATA")) {
                                break :usage .data_mut;
                            }

                            return diags.failParse(path, "unknown segment name '{s}'", .{segname});
                        },
                    };

                    if (usage != .bss and section.offset + section.size > fl.size) {
                        return diags.failParse(path, "section '{s},{s}' exceeds file size", .{
                            segname, sectname,
                        });
                    }

                    const @"align": MappedFile.Alignment = .fromLog2Units(std.math.cast(u6, section.@"align") orelse {
                        return diags.failParse(path, "section '{s},{s}' has unsupported alignment '1 << {d}'", .{
                            segname, sectname, section.@"align",
                        });
                    });

                    // Before creating the input section nodes, make sure the parent section is
                    // suitably aligned. Unlike ELF, Mach-O does not allow customizing segment
                    // alignment, so section alignments greater than the page size cannot be
                    // fulfilled and must trigger an error.
                    if (@"align".compare(.gt, macho.targetPageAlign())) {
                        return diags.failParse(path, "section '{s},{s}' has alignment '{d}' which exceeds page alignment '{d}'", .{
                            segname, sectname, @"align".toByteUnits(), macho.targetPageAlign().toByteUnits(),
                        });
                    }
                    const parent_section = usage.parentSection(macho);
                    if (@"align".compare(.gt, parent_section.alignment(macho))) {
                        macho.targetStore(&parent_section.lcPtr(macho).@"align", @"align".toLog2Units());
                        if (@"align".compare(.gt, parent_section.node(macho).alignment(&macho.mf))) {
                            try parent_section.node(macho).realign(gpa, &macho.mf, @"align");
                        }
                    }

                    input_sections.appendAssumeCapacity(.{
                        .segname = section.segname,
                        .sectname = section.sectname,
                        .reloff = section.reloff,
                        .nreloc = section.nreloc,
                        .addr = section.addr,
                        .size = section.size,
                        .@"align" = @"align",
                        .usage = usage,
                        .file_offset = section.offset,
                        .atoms_start = undefined,
                        .atoms_end = undefined,
                    });
                }
                try fr.interface.discardAll(lc.cmdsize -
                    @sizeOf(std.macho.segment_command_64) -
                    segment_lc.nsects * @sizeOf(std.macho.section_64));
            },

            .SYMTAB => {
                if (lc.cmdsize < @sizeOf(std.macho.symtab_command)) {
                    return diags.failParse(path, "bad cmdsize in 'LC_SYMTAB' load command", .{});
                }
                if (maybe_symtab_lc != null) {
                    return diags.failParse(path, "duplicate 'LC_SYMTAB' load command", .{});
                }
                maybe_symtab_lc = try fr.interface.peekStruct(std.macho.symtab_command, target_endian);
                try fr.interface.discardAll(lc.cmdsize);
            },
        }
    }

    const symtab_lc = maybe_symtab_lc orelse return diags.failParse(
        path,
        "object file does not contain any 'LC_SYMTAB' load command",
        .{},
    );
    if (symtab_lc.symoff + symtab_lc.nsyms * @sizeOf(std.macho.nlist_64) > fl.size) {
        return diags.failParse(path, "symbol table in 'LC_SYMTAB' load command exceeds file size", .{});
    }
    if (symtab_lc.stroff + symtab_lc.strsize > fl.size) {
        return diags.failParse(path, "string table in 'LC_SYMTAB' load command exceeds file size", .{});
    }

    const symbols = try gpa.alloc(Symbol.Index, symtab_lc.nsyms);
    defer gpa.free(symbols);

    const Atom = struct {
        offset_in_section: u64,
        index: InputAtom.Index,
    };
    var atoms: std.ArrayList(Atom) = .empty;
    defer atoms.deinit(gpa);

    // This block processes symbols and atoms (populating `symbols` and `atoms` in the process).
    // After the block exits, the only work remining is relocation lowering.
    {
        try fr.seekTo(fl.offset + symtab_lc.stroff);
        const strtab = try fr.interface.readAllocAll(gpa, symtab_lc.strsize);
        defer gpa.free(strtab);

        try fr.seekTo(fl.offset + symtab_lc.symoff);
        const symtab = try fr.interface.readSliceEndianAlloc(gpa, std.macho.nlist_64, symtab_lc.nsyms, target_endian);
        defer gpa.free(symtab);

        // Sort the symbols so that we can do subsection splitting efficiently. This is okay because
        // there is no significance to the order of the symbol table in an object file. We'll put
        // any symbols not in a section (undef, absolute, etc) first, and then we'll sort the
        // remaining symbols by section index and address.
        //
        // Small caveat: when we lower symbols into the `symbols` array we allocated earlier, we do
        // need to use the *original* indices rather than our sorted indices, so that we can look up
        // relocation target symbols by index later. To help us do that, we'll also build a mapping
        // from our sorted indices to the original indices.
        const orig_sym_indices = try gpa.alloc(u32, symtab.len);
        defer gpa.free(orig_sym_indices);
        for (orig_sym_indices, 0..) |*orig_index, i| {
            orig_index.* = @intCast(i);
        }
        const SymtabSortContext = struct {
            symtab: []std.macho.nlist_64,
            map: []u32,
            pub fn lessThan(ctx: @This(), lhs_index: usize, rhs_index: usize) bool {
                const lhs = ctx.symtab[lhs_index];
                const rhs = ctx.symtab[rhs_index];
                // Don't need to check `n_type.bits.is_stab` here, because valid object files do not
                // contain stabs symbols, so if any are present we'll error out later anyway.
                const lhs_sect: u8, const lhs_addr: u64 = switch (lhs.n_type.bits.type) {
                    .sect => .{ lhs.n_sect, lhs.n_value },
                    else => .{ 0, 0 },
                };
                const rhs_sect: u8, const rhs_addr: u64 = switch (rhs.n_type.bits.type) {
                    .sect => .{ rhs.n_sect, rhs.n_value },
                    else => .{ 0, 0 },
                };
                const order = std.math.order(lhs_sect, rhs_sect).differ() orelse std.math.order(lhs_addr, rhs_addr);
                return order.compare(.lt);
            }
            pub fn swap(ctx: @This(), lhs_index: usize, rhs_index: usize) void {
                std.mem.swap(std.macho.nlist_64, &ctx.symtab[lhs_index], &ctx.symtab[rhs_index]);
                std.mem.swap(u32, &ctx.map[lhs_index], &ctx.map[rhs_index]);
            }
        };
        std.mem.sortUnstableContext(0, symtab.len, @as(SymtabSortContext, .{
            .symtab = symtab,
            .map = orig_sym_indices,
        }));

        var sym_index: u32 = 0;

        // This initial loop processes all of the non-section symbols (undef, absolute, etc). When
        // it exits, `sym_index` points to the first symbol with `n_type.bits.type == .sect`, which
        // is guaranteed to have `n_sect > 0`.
        while (sym_index < symtab.len) : (sym_index += 1) {
            const nlist = &symtab[sym_index];

            if (nlist.n_type.bits.is_stab != 0) return diags.failParse(
                path,
                "invalid symtab entry at index {d}: object file cannot contain stabs",
                .{sym_index},
            );
            if (nlist.n_strx > strtab.len) return diags.failParse(
                path,
                "invalid symtab entry at index {d}: 'n_strx' value '{d}' out of range",
                .{ sym_index, nlist.n_strx },
            );
            const sym_name = std.mem.sliceTo(strtab[nlist.n_strx..], 0);

            symbols[orig_sym_indices[sym_index]] = switch (nlist.n_type.bits.type) {
                .sect => if (nlist.n_sect == 0) return diags.failParse(
                    path,
                    "symbol '{s}' of type 'N_SECT' has invalid section index '{d}'",
                    .{ sym_name, nlist.n_sect },
                ) else {
                    // All done with the non-section symbols!
                    break;
                },

                .undf => if (!nlist.n_type.bits.ext) return diags.failParse(
                    path,
                    "internal symbol '{s}' cannot be undefined",
                    .{sym_name},
                ) else macho.addGlobalSymbol(.{
                    .name = sym_name,
                    .linkage = switch (nlist.n_desc.weak_ref) {
                        false => .strong,
                        true => .weak,
                    },
                    .visibility = switch (nlist.n_type.bits.pext) {
                        false => .default,
                        true => .hidden,
                    },
                    .value = null,
                }) catch |err| switch (err) {
                    error.MultipleDefinitions => unreachable, // not a definition
                    else => |e| return e,
                },

                .abs => if (!nlist.n_type.bits.ext) try macho.addInternalSymbol(.{
                    .name = sym_name,
                    .value = .{ .abs = nlist.n_value },
                }) else macho.addGlobalSymbol(.{
                    .name = sym_name,
                    .linkage = switch (nlist.n_desc.weak_def_or_ref_to_weak) {
                        false => .strong,
                        true => .weak,
                    },
                    .visibility = switch (nlist.n_type.bits.pext) {
                        false => .default,
                        true => .hidden,
                    },
                    .value = null,
                }) catch |err| switch (err) {
                    error.MultipleDefinitions => return diags.failParse(
                        path,
                        "multiple definitions of '{s}'",
                        .{sym_name},
                    ),
                    else => |e| return e,
                },

                .pbud, .indr => |sym_type| return diags.failParse(
                    path,
                    "symbol '{s}' has unsupported symbol type '{t}",
                    .{ sym_name, sym_type },
                ),

                _ => |sym_type| return diags.failParse(
                    path,
                    "symbol '{s}' has unknown symbol type '0x{X}'",
                    .{ sym_name, sym_type },
                ),
            };
        }

        // Now we need to process the symbols defined in sections. We're going to do this by looping
        // over *sections*, just because subsection splitting turns out to be much easier this way.
        for (input_sections.items, 0..) |*section, section_index| {
            if (section.usage != .bss) {
                try fr.seekTo(fl.offset + section.file_offset);
            }

            const parent_section = section.usage.parentSection(macho);

            section.atoms_start = atoms.items.len;
            var cur_atom_end: u64 = 0;
            while (true) {
                const sym_in_section: bool = sym_in_section: {
                    if (sym_index == symtab.len) break :sym_in_section false;

                    const nlist = &symtab[sym_index];

                    assert(nlist.n_sect - 1 >= section_index); // we sorted the symtab by section
                    if (nlist.n_sect - 1 != section_index) break :sym_in_section false;

                    if (nlist.n_type.bits.is_stab != 0) return diags.failParse(
                        path,
                        "invalid symtab entry at index {d}: object file cannot contain stabs",
                        .{sym_index},
                    );
                    if (nlist.n_strx > strtab.len) return diags.failParse(
                        path,
                        "invalid symtab entry at index {d}: 'n_strx' value '{d}' out of range",
                        .{ sym_index, nlist.n_strx },
                    );

                    if (nlist.n_value < section.addr or
                        // This comparison intentionally uses '>=' to disallow defining a symbol at
                        // the exclusive end of a section. This is because of an annoying quirk of
                        // Mach-O: when using `MH_SUBSECTIONS_VIA_SYMBOLS`, it is impossible to
                        // define a symbol at the exclusive end of any subsection, because such a
                        // symbol would be considered a part of the *next* subsection.
                        nlist.n_value >= section.addr + section.size)
                    {
                        const sym_name = std.mem.sliceTo(strtab[nlist.n_strx..], 0);
                        const segname = std.mem.sliceTo(&section.segname, 0);
                        const sectname = std.mem.sliceTo(&section.sectname, 0);
                        return diags.failParse(
                            path,
                            "symbol '{s}' address '0x{X}' exceeds range of containing section '{s},{s}'",
                            .{ sym_name, nlist.n_value, segname, sectname },
                        );
                    }

                    break :sym_in_section true;
                };

                const target_atom_offset: u64 = target_atom_offset: {
                    if (sym_in_section) {
                        break :target_atom_offset symtab[sym_index].n_value - section.addr;
                    }
                    if (cur_atom_end != section.size) {
                        // There were no more symbols in this section, but we still need to create
                        // the atoms for the rest of the section! So just process everything up to
                        // and including the last byte of the section.
                        break :target_atom_offset section.size - 1;
                    }
                    // There are no symbols or atoms left to process, so we're done.
                    break;
                };
                assert(target_atom_offset < section.size);

                while (cur_atom_end <= target_atom_offset) {
                    // We need to start a new atom at `cur_atom_end`.
                    const atom_offset: u64 = cur_atom_end;
                    const atom_size: u64 = usage: switch (section.usage) {
                        // TODO: we should also deduplicate these
                        .data_const_split_4 => 4,
                        .data_const_split_8 => 8,
                        .data_const_split_16 => 16,

                        // TODO: split C string sections (although `MH_SUBSECTIONS_VIA_SYMBOLS` is
                        // usually sufficient anyway); and deduplicate those too
                        .data_const_split_c_strings => continue :usage .data_const,

                        .text,
                        .data_const,
                        .data_const_fixups,
                        .data_mut,
                        .bss,
                        => atom_size: {
                            if (header.flags & std.macho.MH_SUBSECTIONS_VIA_SYMBOLS == 0) {
                                // We aren't splitting the section into multiple atoms, so this must
                                // mean that we haven't created *any* atom for this section yet.
                                assert(atom_offset == 0);
                                break :atom_size section.size;
                            }

                            // Find the next symbol which splits this section. Usually this is just
                            // the next symbol, but there could be some alt entry or alias symbols.
                            var other_sym_index = sym_index + 1;
                            const new_atom_end: u64 = while (other_sym_index < symtab.len and
                                symtab[other_sym_index].n_sect - 1 == section_index) : (other_sym_index += 1)
                            {
                                if (symtab[other_sym_index].n_value == section.addr + atom_offset) continue;
                                if (symtab[other_sym_index].n_desc.alt_entry) continue;
                                // This symbol splits up the subsections. We haven't validated the
                                // value of the symbol, so it might exceed the section bounds. We
                                // will check for that error case later when the outer loop reaches
                                // that symbol, so we don't need to check now, but we should limit
                                // the size of the atom so that we don't allocate a potentially huge
                                // node just before failing.
                                break @min(
                                    symtab[other_sym_index].n_value - section.addr,
                                    section.size,
                                );
                            } else new_atom_end: {
                                // There is no other symbol which splits this section, so the new
                                // atom must cover the entire remainder of the section.
                                break :new_atom_end section.size;
                            };

                            break :atom_size new_atom_end - atom_offset;
                        },
                    };
                    assert(atom_size > 0);
                    assert(atom_offset + atom_size <= section.size);

                    const atom_align = section.@"align".offset(atom_offset);

                    try macho.nodes.ensureUnusedCapacity(gpa, 1);
                    try macho.input_atoms.ensureUnusedCapacity(gpa, 1);
                    const atom_node = try parent_section.node(macho).addFloatingChild(gpa, &macho.mf, .{
                        .size = atom_align.forward(atom_size),
                        .alignment = atom_align,
                        .moved = true, // `flushMoved` will set up vaddrs and apply relocs
                    });
                    const iai: InputAtom.Index = @fromBackingInt(@intCast(macho.input_atoms.items.len));
                    macho.input_atoms.appendAssumeCapacity(.{
                        .node = atom_node,
                        .first_symbol = .none,
                        .first_reloc = @fromBackingInt(0), // populated for real later
                        .vaddr = 0,
                        .any_fixups = false,
                    });
                    macho.nodes.appendAssumeCapacity(.{ .input_atom = iai });

                    if (section.usage != .bss) {
                        const slice = atom_node.slice(&macho.mf);
                        try fr.interface.readSliceAll(slice[0..@intCast(atom_size)]);
                    }

                    try atoms.append(gpa, .{
                        .offset_in_section = atom_offset,
                        .index = iai,
                    });
                    cur_atom_end = atom_offset + atom_size;
                }

                if (sym_in_section) {
                    // We just advanced to the atom containing this symbol, so now we can add it.
                    const atom = atoms.lastPtr().?;
                    const nlist = &symtab[sym_index];
                    const sym_name = std.mem.sliceTo(strtab[nlist.n_strx..], 0);
                    const symbol_offset_in_section = symtab[sym_index].n_value - section.addr;
                    symbols[orig_sym_indices[sym_index]] = if (!nlist.n_type.bits.ext) try macho.addInternalSymbol(.{
                        .name = sym_name,
                        .value = .{ .addr_in_node = .{
                            .node = atom.index.node(macho),
                            .offset = symbol_offset_in_section - atom.offset_in_section,
                        } },
                    }) else macho.addGlobalSymbol(.{
                        .name = sym_name,
                        .linkage = switch (nlist.n_desc.weak_def_or_ref_to_weak) {
                            false => .strong,
                            true => .weak,
                        },
                        .visibility = switch (nlist.n_type.bits.pext) {
                            false => .default,
                            true => .hidden,
                        },
                        .value = .{ .addr_in_node = .{
                            .node = atom.index.node(macho),
                            .offset = symbol_offset_in_section - atom.offset_in_section,
                        } },
                    }) catch |err| switch (err) {
                        error.MultipleDefinitions => return diags.failParse(
                            path,
                            "multiple definitions of '{s}'",
                            .{sym_name},
                        ),
                        else => |e| return e,
                    };

                    sym_index += 1;
                }
            }
            section.atoms_end = atoms.items.len;
        }

        // If, after handling all of the input sections, there is still a symbol remaining, then
        // that symbol must have an out-of-range `n_sect` value.
        if (sym_index != symtab.len) {
            const nlist = &symtab[sym_index];
            if (nlist.n_type.bits.is_stab != 0) return diags.failParse(
                path,
                "invalid symtab entry at index {d}: object file cannot contain stabs",
                .{sym_index},
            );
            if (nlist.n_strx > strtab.len) return diags.failParse(
                path,
                "invalid symtab entry at index {d}: 'n_strx' value '{d}' out of range",
                .{ sym_index, nlist.n_strx },
            );
            const sym_name = std.mem.sliceTo(strtab[nlist.n_strx..], 0);
            assert(nlist.n_type.bits.type == .sect);
            assert(nlist.n_sect - 1 > input_sections.items.len);
            return diags.failParse(
                path,
                "symbol '{s}' of type 'N_SECT' has invalid section index '{d}'",
                .{ sym_name, nlist.n_sect },
            );
        }
    }

    // Now that the symbols are done, we can process relocations.

    var relocs_buf: std.ArrayList(std.macho.relocation_info) = .empty;
    defer relocs_buf.deinit(gpa);

    for (input_sections.items) |*section| {
        if (section.reloff + section.nreloc > fl.size) {
            return diags.failParse(path, "relocations for '{s},{s}' exceed file size", .{
                std.mem.sliceTo(&section.segname, 0),
                std.mem.sliceTo(&section.sectname, 0),
            });
        }

        try relocs_buf.resize(gpa, section.nreloc);
        try fr.seekTo(fl.offset + section.reloff);
        try fr.interface.readSliceEndian(std.macho.relocation_info, relocs_buf.items, target_endian);

        // We need to sort the section's relocations by address for two reasons:
        //
        // * We need the relocations in each atom to be contiguous. This allows us to use sub-arrays
        //   of `macho.relocs` for the relocations in an atom, rather than e.g. linked lists.
        //
        // * We need the relocations *within* each atom to be sorted, so that we can efficiently
        //   generate runtime fixup chains if necessary.
        //
        // This sort must be stable so that `*_RELOC_ADDEND` and `*_RELOC_SUBTRACTOR` relocations
        // remain correctly paired.
        const SortRelocsContext = struct {
            fn lessThan(_: @This(), lhs: std.macho.relocation_info, rhs: std.macho.relocation_info) bool {
                return lhs.r_address < rhs.r_address;
            }
        };
        std.mem.sort(
            std.macho.relocation_info,
            relocs_buf.items,
            @as(SortRelocsContext, .{}),
            SortRelocsContext.lessThan,
        );

        try macho.relocs.ensureUnusedCapacity(gpa, relocs_buf.items.len);

        assert(atoms.items[section.atoms_start].offset_in_section == 0);

        var atom_index = section.atoms_start;
        atoms.items[atom_index].index.ptr(macho).first_reloc = @fromBackingInt(@intCast(
            macho.relocs.items.len,
        ));

        var paired_addend: ?u64 = null;
        for (relocs_buf.items) |reloc| {
            if (reloc.r_address < 0) {
                return diags.failParse(path, "scattered relocations are not supported", .{});
            }
            const offset_in_section: u32 = @intCast(reloc.r_address);

            // Find the atom containing `reloc`. (This is why we sorted the relocations!)
            while (atom_index + 1 < section.atoms_end and
                offset_in_section >= atoms.items[atom_index + 1].offset_in_section)
            {
                atom_index += 1;
                atoms.items[atom_index].index.ptr(macho).first_reloc = @fromBackingInt(@intCast(
                    macho.relocs.items.len,
                ));
            }

            {
                const atom_end: u64 = atom_end: {
                    if (atom_index == section.atoms_end - 1) {
                        break :atom_end section.size;
                    }
                    const next_atom = &atoms.items[atom_index + 1];
                    break :atom_end next_atom.offset_in_section;
                };
                if (offset_in_section + (@as(u8, 1) << reloc.r_length) > atom_end) {
                    const segname = std.mem.sliceTo(&section.segname, 0);
                    const sectname = std.mem.sliceTo(&section.sectname, 0);
                    return diags.failParse(
                        path,
                        "relocation at offset '{d}' in '{s},{s}' spans multiple atoms",
                        .{ offset_in_section, segname, sectname },
                    );
                }
            }

            const offset_in_atom: u32 = @intCast(offset_in_section - atoms.items[atom_index].offset_in_section);

            const length_types: [4]?Reloc.Type, const target: Reloc.Target, const pcrel: bool, const implicit_negative_addend: u64 = switch (header.cputype) {
                else => unreachable,

                std.macho.CPU_TYPE_X86_64 => reloc_type: switch (std.enums.fromInt(
                    std.macho.reloc_type_x86_64,
                    reloc.r_type,
                ) orelse {
                    return diags.failParse(path, "unknown relocation type '{d}'", .{reloc.r_type});
                }) {
                    .X86_64_RELOC_SUBTRACTOR => @panic("TODO(MachO2): SUBTRACTOR relocation"),

                    .X86_64_RELOC_UNSIGNED => .{ .{ .abs8, .abs16, .abs32, .abs64 }, .symbol, false, 0 },
                    .X86_64_RELOC_SIGNED => .{ .{ null, null, .rel32, null }, .symbol, true, 4 },
                    .X86_64_RELOC_SIGNED_1 => .{ .{ null, null, .rel32, null }, .symbol, true, if (reloc.r_extern == 0) 5 else 4 },
                    .X86_64_RELOC_SIGNED_2 => .{ .{ null, null, .rel32, null }, .symbol, true, if (reloc.r_extern == 0) 6 else 4 },
                    .X86_64_RELOC_SIGNED_4 => .{ .{ null, null, .rel32, null }, .symbol, true, if (reloc.r_extern == 0) 8 else 4 },

                    .X86_64_RELOC_BRANCH => .{ .{ null, null, .rel32, null }, .branch, true, 4 },

                    .X86_64_RELOC_TLV => .{ .{ null, null, .rel32, null }, .tlv, true, 4 },

                    .X86_64_RELOC_GOT_LOAD => continue :reloc_type .X86_64_RELOC_GOT, // relaxable
                    .X86_64_RELOC_GOT => .{ .{ null, null, .rel32, null }, .got, true, 4 },
                },

                std.macho.CPU_TYPE_ARM64 => switch (std.enums.fromInt(
                    std.macho.reloc_type_arm64,
                    reloc.r_type,
                ) orelse {
                    return diags.failParse(path, "unknown relocation type '{d}'", .{reloc.r_type});
                }) {
                    .ARM64_RELOC_SUBTRACTOR => @panic("TODO(MachO2): SUBTRACTOR relocation"),

                    .ARM64_RELOC_ADDEND => {
                        const addend: i24 = @bitCast(reloc.r_symbolnum);
                        const extended: i64 = addend;
                        paired_addend = @bitCast(extended);
                        continue;
                    },

                    .ARM64_RELOC_UNSIGNED => .{ .{ .abs8, .abs16, .abs32, .abs64 }, .symbol, false, 0 },
                    .ARM64_RELOC_BRANCH26 => .{ .{ null, null, .arm64_branchrel26, null }, .branch, true, 0 },
                    .ARM64_RELOC_PAGE21 => .{ .{ null, null, .arm64_pagerel21, null }, .symbol, true, 0 },
                    .ARM64_RELOC_PAGEOFF12 => .{ .{ null, null, .arm64_pageoff12, null }, .symbol, false, 0 },

                    .ARM64_RELOC_TLVP_LOAD_PAGE21 => .{ .{ null, null, .arm64_pagerel21, null }, .tlv, true, 0 },
                    .ARM64_RELOC_TLVP_LOAD_PAGEOFF12 => .{ .{ null, null, .arm64_pageoff12, null }, .tlv, false, 0 },

                    .ARM64_RELOC_GOT_LOAD_PAGE21 => .{ .{ null, null, .arm64_pagerel21, null }, .got, true, 0 },
                    .ARM64_RELOC_GOT_LOAD_PAGEOFF12 => .{ .{ null, null, .arm64_pageoff12, null }, .got, false, 0 },

                    // Apple's ld_classic supported an 8-byte non-pcrel version of this relocation,
                    // but it had very weird semantics which didn't actually involve the GOT at all.
                    // My guess is that they intended it to do the obvious thing but shipped with a
                    // bug. In any case, it is not supported in Apple's new linker, and LLD does not
                    // support it either, so therefore we should follow suit and only support the
                    // usual form of this relocation (4-byte pcrel).
                    .ARM64_RELOC_POINTER_TO_GOT => .{ .{ null, null, .rel32, null }, .got, true, 0 },
                },
            };
            const reloc_type_str: []const u8 = switch (header.cputype) {
                else => unreachable,
                std.macho.CPU_TYPE_ARM64 => @tagName(@as(
                    std.macho.reloc_type_arm64,
                    @fromBackingInt(reloc.r_type),
                )),
                std.macho.CPU_TYPE_X86_64 => @tagName(@as(
                    std.macho.reloc_type_x86_64,
                    @fromBackingInt(reloc.r_type),
                )),
            };
            if (@intFromBool(pcrel) != reloc.r_pcrel) {
                return diags.failParse(path, "relocation '{s}' {s} be pc-relative", .{
                    reloc_type_str,
                    if (pcrel) "must" else "cannot",
                });
            }
            const internal_type: Reloc.Type = length_types[reloc.r_length] orelse {
                return diags.failParse(path, "relocation '{s}' does not support length '{d}'", .{
                    reloc_type_str,
                    @as(u8, 1) << reloc.r_length,
                });
            };

            const dest_slice = atoms.items[atom_index].index.node(macho).sliceConst(&macho.mf)[offset_in_atom..];
            if (!internal_type.validateInst(dest_slice)) {
                return diags.failParse(path, "relocation '{s}' against invalid instruction", .{
                    reloc_type_str,
                });
            }
            const stored_addend: u64 = switch (internal_type) {
                .abs8 => std.mem.readInt(u8, dest_slice[0..1], target_endian),
                .abs16 => std.mem.readInt(u16, dest_slice[0..2], target_endian),
                .abs32 => std.mem.readInt(u32, dest_slice[0..4], target_endian),
                .abs64 => std.mem.readInt(u64, dest_slice[0..8], target_endian),
                .rel32 => a: {
                    const extended: i64 = std.mem.readInt(i32, dest_slice[0..4], target_endian);
                    break :a @bitCast(extended);
                },

                .arm64_branchrel26,
                .arm64_pagerel21,
                .arm64_pageoff12,
                => a: {
                    defer paired_addend = null;
                    break :a paired_addend orelse 0;
                },
            };
            if (paired_addend != null) {
                return diags.failParse(path, "relocation '{s}' does not support paired addend", .{reloc_type_str});
            }
            const target_symbol: Symbol.Index, const final_addend: u64 = switch (reloc.r_extern) {
                1 => target: {
                    // The relocation targets a symbol. This is the nice and reasonable case.
                    if (reloc.r_symbolnum >= symbols.len) {
                        return diags.failParse(path, "relocation target symbol '{d}' is invalid", .{reloc.r_symbolnum});
                    }
                    const target_sym = symbols[reloc.r_symbolnum];
                    break :target .{ target_sym, stored_addend -% implicit_negative_addend };
                },
                0 => target: {
                    // In this case, the relocation targets an offset into a *section*. I think the
                    // best way to interpret this is to say that the "stored addend" value contains
                    // what the relocated value *would* be if the linked image's section addresses
                    // matched those of the relocatable itself (unlike ELF, Mach-O uses a flat
                    // virtual address space even in relocatables). So, the best way to handle this
                    // in the pcrel case is to reverse the computation to figure out what the actual
                    // target address is.
                    //
                    // Unfortunately, this is further complicated by the existence of subsections.
                    // Subsections split up the target section's virtual address space, and the
                    // subsection being targeted by a relocation is determined implicitly based on
                    // the target address. Therefore, to ensure we target the correct subsection, it
                    // is necessary to compute the *exact* target address (no off-by-N). This Mach-O
                    // quirk is why we have `implicit_negative_addend` on x86_64: it indicates that
                    // the encoded pc-relative value is relative not to the exact address of the
                    // relocation, but rather to an address a few bytes later, allowing us to
                    // compute the exact target address instead of being 4--8 bytes short.
                    //
                    // Mach-O is a tremendously well-designed object format with no flaws whatsoever.
                    if (reloc.r_symbolnum > input_sections.items.len) {
                        return diags.failParse(path, "relocation target section '{d}' is invalid", .{reloc.r_symbolnum});
                    }
                    const target_addr_in_object: u64 = switch (pcrel) {
                        false => stored_addend +% implicit_negative_addend,
                        true => target_addr: {
                            const pcrel_base: u64 = section.addr + @as(u32, @intCast(reloc.r_address));
                            break :target_addr pcrel_base +% stored_addend +% implicit_negative_addend;
                        },
                    };
                    const target_section = &input_sections.items[reloc.r_symbolnum - 1];
                    const target_offset_in_section = target_addr_in_object -% target_section.addr;
                    if (target_offset_in_section >= target_section.size) {
                        const segname = std.mem.sliceTo(&target_section.segname, 0);
                        const sectname = std.mem.sliceTo(&target_section.sectname, 0);
                        return diags.failParse(
                            path,
                            "relocation target offset '{d}' exceeds range of target section '{s},{s}'",
                            .{ target_offset_in_section, segname, sectname },
                        );
                    }
                    // We need to know which atom we're targeting.
                    const FindAtomContext = struct {
                        offset_in_section: u64,
                        fn order(lhs: @This(), rhs: Atom) std.math.Order {
                            return std.math.order(lhs.offset_in_section, rhs.offset_in_section);
                        }
                    };
                    assert(target_section.atoms_start < target_section.atoms_end);
                    assert(atoms.items[target_section.atoms_start].offset_in_section == 0);
                    const target_atom_index = target_section.atoms_start + std.sort.upperBound(
                        Atom,
                        atoms.items[target_section.atoms_start..target_section.atoms_end],
                        @as(FindAtomContext, .{ .offset_in_section = target_offset_in_section }),
                        FindAtomContext.order,
                    ) - 1; // safe because atom 0 is at offset 0 so never >rhs
                    assert(target_offset_in_section >= atoms.items[target_atom_index].offset_in_section);
                    if (target_atom_index != target_section.atoms_end - 1) {
                        assert(target_offset_in_section < atoms.items[target_atom_index + 1].offset_in_section);
                    }
                    // To preserve my own sanity, I'm just creating symbols to represent these
                    // relocation targets. It's fine, I'm sure nobody will notice.
                    var sym_name_buf: [64]u8 = undefined; // sectname is capped at 16 bytes
                    const sym_name = std.mem.print(
                        &sym_name_buf,
                        "l_{s}_{s}_{x}",
                        .{
                            std.mem.trim(u8, std.mem.sliceTo(&target_section.segname, 0), "_"),
                            std.mem.trim(u8, std.mem.sliceTo(&target_section.sectname, 0), "_"),
                            target_offset_in_section,
                        },
                    ) catch unreachable;
                    const target_sym = try macho.addInternalSymbol(.{
                        .name = sym_name,
                        .value = .{ .addr_in_node = .{
                            .node = atoms.items[target_atom_index].index.node(macho),
                            .offset = target_offset_in_section - atoms.items[target_atom_index].offset_in_section,
                        } },
                    });
                    // Lovely, we have an actual symbol at the correct address which we can target.
                    // Just remember to add the implicit addend back in so that the relocated
                    // PC-relative address is relative to the correct base address!
                    break :target .{ target_sym, -%implicit_negative_addend };
                },
            };
            const target_is_undef = macho.targetLoad(&target_symbol.nlist(macho).n_type).bits.type == .undf;
            if (target_is_undef) {
                const imports_gop = try macho.imports.getOrPut(gpa, target_symbol);
                if (!imports_gop.found_existing) {
                    assert(imports_gop.index == macho.imports.count() - 1);
                    try macho.appendImportEntry(macho.imports.count() - 1, target_symbol);
                }
            }
            if (target == .got or (target_is_undef and target == .branch)) {
                const got_gop = try macho.got.getOrPut(gpa, target_symbol);
                if (!got_gop.found_existing) {
                    assert(got_gop.index == macho.got.count() - 1);
                    try macho.appendGotEntry(macho.got.count() - 1, target_symbol);
                }
            }
            if (target_is_undef and target == .branch) {
                const stubs_gop = try macho.stubs.getOrPut(gpa, target_symbol);
                if (!stubs_gop.found_existing) {
                    assert(stubs_gop.index == macho.stubs.count() - 1);
                    const got_index = macho.got.getIndex(target_symbol).?; // added above
                    try macho.appendStubEntry(macho.stubs.count() - 1, got_index, target_symbol);
                }
            }
            const ri: Reloc.Index = @fromBackingInt(@intCast(macho.relocs.items.len));
            macho.relocs.appendAssumeCapacity(.{
                .node = atoms.items[atom_index].index.node(macho),
                .offset = offset_in_atom,
                .target_symbol = target_symbol,
                .addend = final_addend,
                .bits = .{ .target = target, .type = internal_type, .result = .ok },
                .next_with_same_target_sym = target_symbol.ptr(macho).first_target_reloc,
            });
            target_symbol.ptr(macho).first_target_reloc = .wrap(ri);

            if (target == .symbol and internal_type == .abs64) {
                // In this case the relocation needs a runtime fixup. This is the case even if the
                // target symbol is internal, because we still need to rebase the address.
                ri.ptr(macho).bits.result = .fixup;
                atoms.items[atom_index].index.ptr(macho).any_fixups = true;
            }
        }
    }
}
fn fmtMemberString(member: ?[]const u8) std.fmt.Alt(?[]const u8, memberStringEscape) {
    return .{ .data = member };
}
fn memberStringEscape(member: ?[]const u8, w: *Io.Writer) Io.Writer.Error!void {
    try w.print("({f})", .{std.zig.fmtString(member orelse return)});
}

pub fn setDarwinSdkVersion(macho: *MachO, sdk_version: link.DarwinSdkVersion) link.Error!void {
    const comp = macho.base.comp;
    const diags = &comp.link_diags;
    const target = comp.getTarget();

    const os_min_version: link.DarwinSdkVersion = .{
        .major = @intCast(target.os.version_range.semver.min.major),
        .minor = @intCast(target.os.version_range.semver.min.minor),
        .patch = @intCast(target.os.version_range.semver.min.patch),
    };

    _ = macho.appendSimpleLoadCommand(std.macho.version_min_command, .{
        .cmd = switch (target.os.tag) {
            .macos => .VERSION_MIN_MACOSX,
            .ios => .VERSION_MIN_IPHONEOS,
            .tvos => .VERSION_MIN_TVOS,
            .watchos => .VERSION_MIN_WATCHOS,
            .maccatalyst, .driverkit, .visionos => |os_tag| std.debug.panic(
                "TODO(MachO2): sdk version for '{t}'",
                .{os_tag},
            ),
            else => unreachable,
        },
        .version = @backingInt(os_min_version),
        .sdk = @backingInt(sdk_version),
    }) catch |err| switch (err) {
        error.MappedFileIo => return diags.fail("failed to write output file: {t}", .{macho.mf.io_err.?}),
        else => |e| return e,
    };
}
pub fn prelink(macho: *MachO, prog_node: std.Progress.Node) link.Error!void {
    _ = macho;
    _ = prog_node;
}
pub fn flush(
    macho: *MachO,
    arena: Allocator,
    tid: Zcu.PerThread.Id,
    prog_node: std.Progress.Node,
) link.Error!void {
    const comp = macho.base.comp;
    const diags = &comp.link_diags;
    _ = arena;
    _ = tid;
    _ = prog_node;

    macho.updateSegmentLoadAddresses() catch |err| switch (err) {
        error.MappedFileIo => return diags.fail("failed to write output file: {t}", .{macho.mf.io_err.?}),
        else => |e| return e,
    };

    while (try macho.idle()) {}

    if (macho.overflowed_reloc_count > 0) {
        diags.addError("failed to apply {d} relocations: overflow", .{macho.overflowed_reloc_count});
    }
    if (macho.misaligned_reloc_count > 0) {
        diags.addError("failed to apply {d} relocations: misaligned value", .{macho.misaligned_reloc_count});
    }

    if (macho.lc.entry_point) |entry_point_lc| {
        const sym_name: []const u8 = name: switch (macho.options.entry) {
            .default => switch (comp.config.output_mode) {
                .Exe => continue :name .enabled,
                .Lib, .Obj => continue :name .disabled,
            },
            .disabled => unreachable, // `macho.lc.entry_point` is only set if entry is enabled
            .enabled => "_main",
            .named => |named| named,
        };
        const symbol = macho.globalByName(sym_name).unwrap() orelse {
            return diags.fail("no definition of entry point symbol {q}", .{sym_name});
        };
        if (macho.targetLoad(&symbol.nlist(macho).n_type).bits.type == .undf) {
            return diags.fail("no definition of entry point symbol {q}", .{sym_name});
        }

        // Documentation says that this field should be the offset of the entry point in the file,
        // but that's a blatant lie; if you read the dyld source code, it's actually the vaddr minus
        // the text base vaddr. Because, like, obviously---what the fuck would they do with the
        // offset in the Mach-O file?
        const entry_point_ptr = entry_point_lc.ptr(macho, std.macho.entry_point_command);
        macho.targetStore(&entry_point_ptr.entryoff, std.math.cast(
            u32,
            symbol.value(macho) - Segment.text.vaddr(macho),
        ) orelse {
            return diags.fail("entry point address too big", .{});
        });
    }

    macho.mf.flush() catch |err| switch (err) {
        error.MappedFileIo => return diags.fail("failed to write output file: {t}", .{macho.mf.io_err.?}),
        else => |e| return e,
    };

    if (macho.options.enable_link_snapshots)
        macho.dumpStderr() catch |err|
            return diags.fail("dumping link snapshot failed: {t}", .{err});
}
pub fn updateErrorData(macho: *MachO, pt: Zcu.PerThread) link.Error!void {
    _ = macho;
    _ = pt;
    @panic("TODO");
}
pub fn idle(macho: *MachO) link.Error!bool {
    macho.mf.nodes_lock.lock();
    defer macho.mf.nodes_lock.unlock();

    task: {
        if (macho.sections_changed_index.pop()) |kv| {
            const section = kv.key;
            var child_it = section.node(macho).first(&macho.mf);
            while (child_it.unwrap()) |child| : (child_it = child.next(&macho.mf)) {
                switch (macho.getNode(child)) {
                    .macho => unreachable,
                    .macho_header => unreachable,
                    .symtab => unreachable,
                    .strtab => unreachable,
                    .indirect_syms => unreachable,
                    .section_indirect_syms => unreachable,
                    .linkedit_data => unreachable,
                    .fixup_chains_header => unreachable,
                    .fixup_chains_imports => unreachable,
                    .fixup_chains_symbol_names => unreachable,
                    .fixup_chains_starts => unreachable,
                    .fixup_chains_starts_in_image => unreachable,
                    .fixup_chains_starts_in_segment => unreachable,
                    .segment => unreachable,
                    .section => unreachable,
                    .section_manual_size => unreachable,
                    .input_atom => |iai| {
                        var sym_it = iai.ptrConst(macho).first_symbol;
                        while (sym_it.unwrap()) |sym| : (sym_it = sym.ptr(macho).next_in_node) {
                            macho.targetStore(&sym.nlist(macho).n_sect, section.index(macho));
                        }
                    },
                }
            }
            break :task;
        }

        while (macho.mf.updates.pop()) |ni| {
            ni.cleanUpdated(&macho.mf);
            const clean_moved = ni.cleanMoved(&macho.mf);
            const clean_resized = ni.cleanResized(&macho.mf);
            const clean_next_moved = ni.cleanNextMoved(&macho.mf);
            if (!clean_moved and !clean_resized and !clean_next_moved) continue;

            const sub_prog_node = macho.mf.update_prog_node.start(@tagName(macho.getNode(ni)), 0);
            defer sub_prog_node.end();

            if (clean_moved) try macho.flushMoved(ni);
            if (clean_resized) try macho.flushResized(ni);

            break :task;
        }

        if (macho.dirty_fixup_chain_pages.pop()) |kv| {
            const sub_prog_node = macho.mf.update_prog_node.start("Fixup Chains", 0);
            defer sub_prog_node.end();
            macho.flushFixupChains(kv.key.seg, kv.key.page);
            break :task;
        }
    }

    return macho.sections_changed_index.count() > 0 or
        macho.mf.updates.items.len > 0 or
        macho.dirty_fixup_chain_pages.count() > 0;
}
fn flushMoved(macho: *MachO, ni: MappedFile.Node.Index) link.Error!void {
    macho.flushMachOFileOffset(ni);
    switch (macho.getNode(ni)) {
        .macho => unreachable,
        .macho_header => {},
        .indirect_syms => {},
        .section_indirect_syms => |section| {
            const offset, _ = ni.location(&macho.mf).resolve(&macho.mf);
            macho.targetStore(&section.lcPtr(macho).reserved1, @intCast(
                @divExact(offset, 4), // valid due to alignment
            ));
        },
        .segment => |segment| {
            if (segment.fixupChainStartsNode(macho).unwrap()) |starts_in_segment_node| {
                const starts_in_segment: *std.macho.dyld_chained_starts_in_segment = @ptrCast(@alignCast(
                    starts_in_segment_node.slice(&macho.mf)[0..@sizeOf(std.macho.dyld_chained_starts_in_segment)],
                ));
                macho.targetStore(
                    &starts_in_segment.segment_offset,
                    segment.vaddr(macho) - Segment.text.vaddr(macho),
                );
            }
            return; // segments don't bubble moved
        },
        .section, .section_manual_size => |section| {
            const segment_vaddr: u64 = section.segment(macho).vaddr(macho);
            const section_offset = ni.location(&macho.mf).resolve(&macho.mf)[0];
            macho.targetStore(&section.lcPtr(macho).addr, segment_vaddr + section_offset);

            if (section == macho.known_sections.stubs) {
                for (macho.stubs.keys(), 0..) |symbol, stub_index| {
                    const got_index = macho.got.getIndex(symbol).?;
                    macho.writeStubEntry(stub_index, got_index);
                }
                for (macho.got.keys()) |symbol| {
                    try symbol.applyTargetRelocs(macho, .init(.{
                        .symbol = false,
                        .branch = true,
                        .got = false,
                        .tlv = false,
                    }));
                }
            }

            if (section == macho.known_sections.got) {
                for (macho.stubs.keys(), 0..) |symbol, stub_index| {
                    const got_index = macho.got.getIndex(symbol).?;
                    macho.writeStubEntry(stub_index, got_index);
                }
                for (macho.got.keys()) |symbol| {
                    try symbol.applyTargetRelocs(macho, .init(.{
                        .symbol = false,
                        .branch = false,
                        .got = true,
                        .tlv = false,
                    }));
                }
            }

            // If it is the GOT which has moved, its runtime fixup need to be flushed.
            if (section == macho.known_sections.got and macho.got.count() > 0) {
                const segment = section.segment(macho);

                const old_start = macho.prev_got_offset;
                const new_start: u32 = off: {
                    const got_file_off = ni.fileLocation(&macho.mf, false).offset;
                    const seg_file_off = segment.node(macho).fileLocation(&macho.mf, false).offset;
                    break :off @intCast(got_file_off - seg_file_off);
                };

                const page_align = macho.targetPageAlign();
                const got_size = @as(u64, macho.got.count()) * macho.targetPtrSize();

                const old_page_start: u32 = @intCast(@divFloor(old_start, page_align.toByteUnits()));
                const old_page_end: u32 = @intCast(@divCeil(old_start + got_size, page_align.toByteUnits()));
                try macho.dirty_fixup_chain_pages.ensureUnusedCapacity(
                    macho.base.comp.gpa,
                    old_page_end - old_page_start,
                );
                for (old_page_start..old_page_end) |page_index| {
                    macho.dirty_fixup_chain_pages.putAssumeCapacity(.{
                        .seg = segment,
                        .page = @intCast(page_index),
                    }, {});
                }

                const new_page_start: u32 = @intCast(@divFloor(new_start, page_align.toByteUnits()));
                const new_page_end: u32 = @intCast(@divCeil(new_start + got_size, page_align.toByteUnits()));
                try macho.dirty_fixup_chain_pages.ensureUnusedCapacity(
                    macho.base.comp.gpa,
                    new_page_end - new_page_start,
                );
                for (new_page_start..new_page_end) |page_index| {
                    macho.dirty_fixup_chain_pages.putAssumeCapacity(.{
                        .seg = segment,
                        .page = @intCast(page_index),
                    }, {});
                }

                macho.prev_got_offset = new_start;
            }
        },
        .symtab => {},
        .strtab => {},
        .linkedit_data => {
            // No need to bubble moved.
            return;
        },
        .fixup_chains_header => {},
        .fixup_chains_imports => {
            const offset: u64, _ = ni.location(&macho.mf).resolve(&macho.mf);
            const hdr: *std.macho.dyld_chained_fixups_header = @ptrCast(@alignCast(
                macho.node.fixup_chains_header.slice(&macho.mf),
            ));
            macho.targetStore(&hdr.imports_offset, @intCast(offset));
        },
        .fixup_chains_symbol_names => {
            const offset: u64, _ = ni.location(&macho.mf).resolve(&macho.mf);
            const hdr: *std.macho.dyld_chained_fixups_header = @ptrCast(@alignCast(
                macho.node.fixup_chains_header.slice(&macho.mf),
            ));
            macho.targetStore(&hdr.symbols_offset, @intCast(offset));
        },
        .fixup_chains_starts => {
            const offset: u64, _ = ni.location(&macho.mf).resolve(&macho.mf);
            const hdr: *std.macho.dyld_chained_fixups_header = @ptrCast(@alignCast(
                macho.node.fixup_chains_header.slice(&macho.mf),
            ));
            macho.targetStore(&hdr.starts_offset, @intCast(offset));
        },
        .fixup_chains_starts_in_image => {},
        .fixup_chains_starts_in_segment => |segment| {
            const starts_in_image_off = macho.node.fixup_chains_starts_in_image.location(&macho.mf).resolve(&macho.mf)[0];
            const starts_in_segment_off = ni.location(&macho.mf).resolve(&macho.mf)[0];

            const chained_starts_in_image: []u32 = @ptrCast(@alignCast(
                macho.node.fixup_chains_starts_in_image.slice(&macho.mf),
            ));
            macho.targetStore(
                &chained_starts_in_image[@backingInt(segment) + 2],
                @intCast(starts_in_segment_off - starts_in_image_off),
            );
        },
        .input_atom => |iai| {
            const input_atom_offset: u64, _ = ni.location(&macho.mf).resolve(&macho.mf);
            const section = macho.getNode(iai.node(macho).parent(&macho.mf).unwrap().?).section;

            const old_atom_addr = iai.ptrConst(macho).vaddr;
            const new_atom_addr = macho.targetLoad(&section.lcPtrConst(macho).addr) + input_atom_offset;
            iai.ptr(macho).vaddr = new_atom_addr;

            const first_reloc = iai.ptr(macho).first_reloc;
            for (macho.relocs.items[@backingInt(first_reloc)..]) |*reloc| {
                if (reloc.node != ni) break;
                try reloc.apply(macho);
            }

            var sym_it = iai.ptrConst(macho).first_symbol;
            while (sym_it.unwrap()) |sym| : (sym_it = sym.ptr(macho).next_in_node) {
                const new_addr = sym.value(macho) - old_atom_addr + new_atom_addr;
                try sym.flushMoved(macho, new_addr);
            }

            // If the atom which has moved contains runtime fixups, they need to be flushed.
            if (iai.ptr(macho).any_fixups) {
                const segment = section.segment(macho);

                const page_align = macho.targetPageAlign();

                const new_start: u32 = off: {
                    const ni_file_off = ni.fileLocation(&macho.mf, false).offset;
                    const seg_file_off = segment.node(macho).fileLocation(&macho.mf, false).offset;
                    break :off @intCast(ni_file_off - seg_file_off);
                };
                const new_end = new_start + ni.location(&macho.mf).resolve(&macho.mf)[1];

                const page_start: u32 = @intCast(@divFloor(new_start, page_align.toByteUnits()));
                const page_end: u32 = @intCast(@divCeil(new_end, page_align.toByteUnits()));
                try macho.dirty_fixup_chain_pages.ensureUnusedCapacity(
                    macho.base.comp.gpa,
                    page_end - page_start,
                );
                for (page_start..page_end) |page_index| {
                    macho.dirty_fixup_chain_pages.putAssumeCapacity(.{
                        .seg = segment,
                        .page = @intCast(page_index),
                    }, {});
                }
            }
        },
    }
    try ni.childrenMoved(macho.base.comp.gpa, &macho.mf);
}
fn flushMachOFileOffset(macho: *MachO, ni: MappedFile.Node.Index) void {
    const offset = ni.fileLocation(&macho.mf, false).offset;
    switch (macho.getNode(ni)) {
        .macho => unreachable,
        .macho_header => {
            assert(offset == 0);
        },
        .symtab => {
            const symtab_cmd = macho.lc.symtab.ptr(macho, std.macho.symtab_command);
            macho.targetStore(&symtab_cmd.symoff, @intCast(offset));
        },
        .strtab => {
            const symtab_cmd = macho.lc.symtab.ptr(macho, std.macho.symtab_command);
            macho.targetStore(&symtab_cmd.stroff, @intCast(offset));
        },
        .indirect_syms => {
            const dysymtab_cmd = macho.lc.dysymtab.ptr(macho, std.macho.dysymtab_command);
            macho.targetStore(&dysymtab_cmd.indirectsymoff, @intCast(offset));
        },
        .section_indirect_syms => {},
        .linkedit_data => |lc| {
            const linkedit_data_cmd = lc.ptr(macho, std.macho.linkedit_data_command);
            macho.targetStore(&linkedit_data_cmd.dataoff, @intCast(offset));
        },
        .fixup_chains_header => {},
        .fixup_chains_imports => {},
        .fixup_chains_symbol_names => {},
        .fixup_chains_starts => {},
        .fixup_chains_starts_in_image => {},
        .fixup_chains_starts_in_segment => {},
        .segment => |segment| {
            macho.targetStore(&segment.lcPtr(macho).fileoff, @intCast(offset));
            var child_oni = ni.first(&macho.mf);
            while (child_oni.unwrap()) |child_ni| : (child_oni = child_ni.next(&macho.mf)) {
                macho.flushMachOFileOffset(child_ni);
            }
        },
        .section, .section_manual_size => |section| {
            macho.targetStore(&section.lcPtr(macho).offset, @intCast(offset));
        },
        .input_atom => {},
    }
}
fn flushResized(macho: *MachO, ni: MappedFile.Node.Index) link.Error!void {
    const size = ni.location(&macho.mf).resolve(&macho.mf)[1];
    switch (macho.getNode(ni)) {
        .macho => {},
        .macho_header => {},
        .symtab, .strtab => {
            // Don't modify the symtab load command's `nsyms` or `strsize` field here; we only do
            // that when we actually add strings or symbols.
        },
        .indirect_syms => {
            const dysymtab_cmd = macho.lc.dysymtab.ptr(macho, std.macho.dysymtab_command);
            macho.targetStore(&dysymtab_cmd.nindirectsyms, @intCast(@divExact(size, 4)));
        },
        .section_indirect_syms => {},
        .linkedit_data => |lc| {
            const linkedit_data_cmd = lc.ptr(macho, std.macho.linkedit_data_command);
            macho.targetStore(&linkedit_data_cmd.datasize, @intCast(size));
        },
        .fixup_chains_header => {},
        .fixup_chains_imports => {},
        .fixup_chains_symbol_names => {},
        .fixup_chains_starts => {},
        .fixup_chains_starts_in_image => {},
        .fixup_chains_starts_in_segment => {},
        .segment => |segment| {
            macho.targetStore(&segment.lcPtr(macho).filesize, size);
        },
        .section => |section| {
            macho.targetStore(&section.lcPtr(macho).size, size);
        },
        .section_manual_size => {},
        .input_atom => {},
    }
}

fn getNodeVAddr(macho: *MachO, ni: MappedFile.Node.Index) u64 {
    return switch (macho.getNode(ni)) {
        .macho => unreachable,
        .macho_header => unreachable,
        .symtab => unreachable,
        .strtab => unreachable,
        .indirect_syms => unreachable,
        .section_indirect_syms => unreachable,
        .linkedit_data => unreachable,
        .fixup_chains_header => unreachable,
        .fixup_chains_imports => unreachable,
        .fixup_chains_symbol_names => unreachable,
        .fixup_chains_starts => unreachable,
        .fixup_chains_starts_in_image => unreachable,
        .fixup_chains_starts_in_segment => unreachable,
        .segment => |segment| segment.vaddr(macho),
        .section, .section_manual_size => |section| section.vaddr(macho),
        .input_atom => |iai| iai.ptr(macho).vaddr,
    };
}
fn getNodeSection(macho: *MachO, ni: MappedFile.Node.Index) Section {
    var cur_ni = ni;
    while (true) {
        switch (macho.getNode(cur_ni)) {
            .section, .section_manual_size => |section| return section,
            else => cur_ni = cur_ni.parent(&macho.mf).unwrap().?,
        }
    }
}
fn nodeFirstSymbolPtr(macho: *MachO, ni: MappedFile.Node.Index) *Symbol.Index.Optional {
    return switch (macho.getNode(ni)) {
        .input_atom => |iai| &iai.ptr(macho).first_symbol,
        else => unreachable,
    };
}

fn targetStubSize(macho: *const MachO) u8 {
    return switch (macho.base.comp.getTarget().cpu.arch) {
        else => unreachable,
        .aarch64 => 12,
        .x86_64 => 6,
    };
}
fn appendStubEntry(macho: *MachO, old_len: u64, got_index: u64, target_symbol: Symbol.Index) Error!void {
    const gpa = macho.base.comp.gpa;

    const entry_size = macho.targetStubSize();
    const stub_offset = entry_size * old_len;
    assert(macho.targetLoad(&macho.known_sections.stubs.lcPtrConst(macho).size) == stub_offset);

    const new_size = stub_offset + entry_size;
    try macho.known_sections.stubs.node(macho).ensureMinimumSize(gpa, &macho.mf, new_size);
    macho.targetStore(&macho.known_sections.stubs.lcPtr(macho).size, new_size);

    macho.writeStubEntry(old_len, got_index);

    try macho.node.stubs_indirect_syms.ensureMinimumSize(gpa, &macho.mf, (old_len + 1) * 4);
    const indirect_syms: []Symbol.Index = @ptrCast(@alignCast(
        macho.node.stubs_indirect_syms.slice(&macho.mf),
    ));
    indirect_syms[@intCast(old_len)] = target_symbol;
}
fn writeStubEntry(macho: *MachO, stub_index: u64, got_index: u64) void {
    if (macho.known_sections.stubs.node(macho).hasMoved(&macho.mf)) return;
    if (macho.known_sections.got.node(macho).hasMoved(&macho.mf)) return;

    const entry_size = macho.targetStubSize();
    const stub_offset = entry_size * stub_index;

    const got_entry_vaddr = macho.known_sections.got.vaddr(macho) + got_index * macho.targetPtrSize();
    const new_stub_vaddr = macho.known_sections.stubs.vaddr(macho) + stub_offset;

    switch (macho.base.comp.getTarget().cpu.arch) {
        else => unreachable,
        .aarch64 => {
            // The first instruction of the stub will be an ADRP for which we need to compute the
            // offset from that instruction's 4k page to the GOT entry's 4k page.
            const cur_page_base = (new_stub_vaddr >> 12) << 12;
            const target_page_base = (got_entry_vaddr >> 12) << 12;
            const cur_page_to_target_page: i64 = @bitCast(target_page_base - cur_page_base);

            // Then we will use an LDR to reference the GOT entry on that page.
            const target_page_off: u12 = @truncate(got_entry_vaddr);

            const insts: []align(4) aarch64.encoding.Instruction = @ptrCast(@alignCast(
                macho.known_sections.stubs.node(macho).slice(&macho.mf)[@intCast(stub_offset)..][0..entry_size],
            ));
            @memcpy(insts, @as([]const aarch64.encoding.Instruction, &.{
                .adrp(.x16, @intCast(cur_page_to_target_page)),
                .ldr(.x16, .{ .unsigned_offset = .{ .base = .x16, .offset = target_page_off } }),
                .br(.x16),
            }));
            if (.little != std.lang.Endian.native) { // aarch64 instructions are little-endian
                std.mem.byteSwapAllElements(aarch64.encoding.Instruction, insts);
            }
        },
        .x86_64 => {
            const new_stub_slice: []align(2) u8 = @ptrCast(@alignCast(
                macho.known_sections.stubs.node(macho).slice(&macho.mf)[@intCast(stub_offset)..][0..entry_size],
            ));
            // jmpq *got_entry(%rip)
            @memcpy(new_stub_slice[0..2], @as([]const u8, &.{ 0xff, 0x25 }));
            const offset_ptr: *align(2) i32 = @ptrCast(new_stub_slice[2..]);
            const offset: i64 = @bitCast(got_entry_vaddr -% (new_stub_vaddr + 6));
            macho.targetStore(offset_ptr, @intCast(offset));
        },
    }
}

fn appendGotEntry(macho: *MachO, old_len: u64, target_symbol: Symbol.Index) Error!void {
    const gpa = macho.base.comp.gpa;

    const ptr_size = macho.targetPtrSize();
    const entry_offset = ptr_size * old_len;
    assert(macho.targetLoad(&macho.known_sections.got.lcPtrConst(macho).size) == entry_offset);

    const new_size = entry_offset + ptr_size;
    try macho.known_sections.got.node(macho).ensureMinimumSize(gpa, &macho.mf, new_size);
    macho.targetStore(&macho.known_sections.got.lcPtr(macho).size, new_size);

    const new_entry = macho.known_sections.got.node(macho).slice(&macho.mf)[@intCast(entry_offset)..][0..ptr_size];
    @memset(new_entry, 0);

    if (!macho.known_sections.got.node(macho).hasMoved(&macho.mf)) {
        const segment = macho.known_sections.got.segment(macho);
        const page_index: u16 = @intCast(@divFloor(
            macho.prev_got_offset + entry_offset,
            macho.targetPageAlign().toByteUnits(),
        ));
        try macho.dirty_fixup_chain_pages.put(gpa, .{ .seg = segment, .page = page_index }, {});
    }

    try macho.node.got_indirect_syms.ensureMinimumSize(gpa, &macho.mf, (old_len + 1) * 4);
    const indirect_syms: []Symbol.Index = @ptrCast(@alignCast(
        macho.node.got_indirect_syms.slice(&macho.mf),
    ));
    indirect_syms[@intCast(old_len)] = target_symbol;
}

fn appendImportEntry(macho: *MachO, old_len: u64, symbol_index: Symbol.Index) Error!void {
    const gpa = macho.base.comp.gpa;

    const new_size = @sizeOf(std.macho.dyld_chained_import_addend64) * (old_len + 1);
    try macho.node.fixup_chains_imports.ensureMinimumSize(gpa, &macho.mf, new_size);

    const symbol_name_off = macho.fixup_chain_symbol_names_len;
    macho.fixup_chain_symbol_names_len += @intCast(symbol_index.name(macho).len + 1);
    try macho.node.fixup_chains_symbol_names.ensureMinimumSize(gpa, &macho.mf, macho.fixup_chain_symbol_names_len);
    {
        const sym_name = symbol_index.name(macho);
        const name_slice = macho.node.fixup_chains_symbol_names.slice(&macho.mf)[symbol_name_off..];
        @memcpy(name_slice[0..sym_name.len], sym_name);
        name_slice[sym_name.len] = 0;
    }

    {
        const hdr: *std.macho.dyld_chained_fixups_header = @ptrCast(@alignCast(
            macho.node.fixup_chains_header.slice(&macho.mf)[0..@sizeOf(std.macho.dyld_chained_fixups_header)],
        ));
        assert(macho.targetLoad(&hdr.imports_count) == old_len);
        macho.targetStore(&hdr.imports_count, @intCast(old_len + 1));
    }

    const imports: []std.macho.dyld_chained_import_addend64 = @ptrCast(@alignCast(
        macho.node.fixup_chains_imports.slice(&macho.mf)[0..@intCast(new_size)],
    ));
    imports[@intCast(old_len)] = .{
        .bits = .{
            .lib_ordinal = 0xFFFE,
            .weak_import = false,
            .name_offset = symbol_name_off,
        },
        .addend = 0,
    };
    if (macho.targetEndian() != std.lang.Endian.native) {
        std.mem.byteSwapAllFields(std.macho.dyld_chained_import_addend64, &imports[@intCast(old_len)]);
    }
}

pub fn updateFunc(
    macho: *MachO,
    pt: Zcu.PerThread,
    func_index: InternPool.Index,
    mir: *const codegen.AnyMir,
) link.Error!void {
    _ = macho;
    _ = pt;
    _ = func_index;
    _ = mir;
    @panic("TODO");
}
pub fn updateNav(macho: *MachO, pt: Zcu.PerThread, nav_index: InternPool.Nav.Index) link.Error!void {
    _ = macho;
    _ = pt;
    _ = nav_index;
    @panic("TODO");
}
pub fn updateExports(
    macho: *MachO,
    pt: Zcu.PerThread,
    export_indices: []const Zcu.Export.Index,
) link.Error!void {
    _ = macho;
    _ = pt;
    _ = export_indices;
    @panic("TODO");
}

fn dumpStderr(macho: *MachO) Io.File.Writer.Error!void {
    const comp = macho.base.comp;
    const io = comp.io;
    var buffer: [512]u8 = undefined;
    const stderr = try io.lockStderr(&buffer, null);
    defer io.unlockStderr();
    const w = &stderr.file_writer.interface;
    _ = macho.dump(w) catch |err| switch (err) {
        error.WriteFailed => return stderr.file_writer.err.?,
    };
}

pub fn dump(macho: *MachO, w: *Io.Writer) !link.File.DumpResult {
    if (macho.options.enable_link_snapshots) {
        try macho.printNode(w, .root, 0);
        return .enabled;
    }
    return .disabled;
}
fn printNode(
    macho: *const MachO,
    w: *Io.Writer,
    ni: MappedFile.Node.Index,
    indent: usize,
) Io.Writer.Error!void {
    try w.splatByteAll(' ', indent);
    switch (macho.getNode(ni)) {
        .macho => try w.writeAll("macho"),
        .macho_header => try w.writeAll("macho_header"),
        .symtab => try w.writeAll("symtab"),
        .strtab => try w.writeAll("strtab"),
        .indirect_syms => try w.writeAll("indirect_syms"),
        .section_indirect_syms => |section| try w.print("section_indirect_syms({s},{s})", .{
            section.segmentName(macho),
            section.name(macho),
        }),
        .linkedit_data => |lc| try w.print("linkedit_data(LC_{t})", .{
            macho.targetLoad(&lc.ptrConst(macho, std.macho.load_command).cmd),
        }),
        .fixup_chains_header => try w.writeAll("fixup_chains_header"),
        .fixup_chains_imports => try w.writeAll("fixup_chains_imports"),
        .fixup_chains_symbol_names => try w.writeAll("fixup_chains_symbol_names"),
        .fixup_chains_starts => try w.writeAll("fixup_chains_starts"),
        .fixup_chains_starts_in_image => try w.writeAll("fixup_chains_starts_in_image"),
        .fixup_chains_starts_in_segment => |seg| try w.print("fixup_chains_starts_in_segment({s})", .{seg.name(macho)}),
        .section => |section| try w.print("section({s},{s})", .{
            section.segmentName(macho),
            section.name(macho),
        }),
        .section_manual_size => |section| try w.print("section_manual_size({s},{s})", .{
            section.segmentName(macho),
            section.name(macho),
        }),
        .segment => |segment| try w.print("segment({s})", .{segment.name(macho)}),
        .input_atom => |iai| {
            // It's useful to try and give a name to the input atom, but input atoms don't actually
            // have names. However, a decent approach if the input atom only contains one symbol is
            // to print that symbol's name. We also accept the input atom containing many global
            // symbols but only one internal symbol (since that's how Zig exports work).
            if (iai.ptrConst(macho).first_symbol.unwrap()) |first_symbol| {
                if (first_symbol.ptrConst(macho).next_in_node == .none) {
                    // There is only one symbol in the section, so it's a good identifier.
                    try w.print("input_atom({q})", .{first_symbol.name(macho)});
                } else {
                    // Also, if it has N global symbols but only one internal symbol, go with the
                    // internal name---this works well for how Zig handles exports.
                    var prev_local_sym: Symbol.Index.Optional = .none;
                    var sym_it: Symbol.Index.Optional = .wrap(first_symbol);
                    const only_local_sym: Symbol.Index.Optional = while (sym_it.unwrap()) |sym| : (sym_it = sym.ptrConst(macho).next_in_node) {
                        const n_type = macho.targetLoad(&sym.nlistConst(macho).n_type);
                        if (!n_type.bits.ext and !n_type.bits.pext) {
                            // this is an internal symbol
                            if (prev_local_sym != .none) break .none;
                            prev_local_sym = .wrap(sym);
                        }
                    } else prev_local_sym;

                    if (only_local_sym.unwrap()) |local_sym| {
                        try w.print("input_atom({q})", .{local_sym.name(macho)});
                    } else {
                        try w.writeAll("input_atom([multiple symbols])");
                    }
                }
            } else {
                try w.writeAll("input_atom([no symbols])");
            }
        },
    }
    {
        const mf_node = &macho.mf.nodes.items[@backingInt(ni)];
        const off, const size = mf_node.location().resolve(&macho.mf);
        try w.print(" index={d} offset=0x{x} size=0x{x} align=0x{x} {t}{s}{s}{s}{s}{s}{s}\n", .{
            @backingInt(ni),
            off,
            size,
            mf_node.flags.alignment.toByteUnits(),
            mf_node.flags.position,
            if (mf_node.flags.bubbles_moved) " bubbles_moved" else "",
            if (mf_node.flags.moved) " moved" else "",
            if (mf_node.flags.resized) " resized" else "",
            if (mf_node.flags.enable_next_moved) " enable_next_moved" else "",
            if (mf_node.flags.next_moved) " next_moved" else "",
            if (mf_node.flags.has_content) " has_content" else "",
        });
    }
    if (ni.first(&macho.mf).unwrap()) |first_ni| {
        // non-leaf, just print children
        var child_ni = first_ni;
        while (true) {
            try macho.printNode(w, child_ni, indent + 1);
            child_ni = child_ni.next(&macho.mf).unwrap() orelse break;
        }
        return;
    }
    const start_address: usize, const end_address: usize = file_loc: {
        const file_loc = ni.fileLocation(&macho.mf, false);
        break :file_loc .{ @intCast(file_loc.offset), @intCast(file_loc.offset + file_loc.size) };
    };
    var address = start_address;
    const line_len = 0x10;
    while (true) : (address = @min(std.mem.alignForward(usize, address + 1, line_len), end_address)) {
        try w.splatByteAll(' ', indent + 1);
        try w.print("{x:0>8}", .{address});
        if (address == end_address) break try w.writeByte('\n');
        try w.splatByteAll(' ', 2);
        const start_byte_address = std.mem.alignBackward(usize, address, line_len);
        const end_byte_address = start_byte_address + line_len;
        for (start_byte_address..end_byte_address) |byte_address|
            if (byte_address < start_address or byte_address >= end_address)
                try w.splatByteAll(' ', 3)
            else
                try w.print("{x:0>2} ", .{macho.mf.memory_map.memory[byte_address]});
        try w.writeByte(' ');
        for (start_byte_address..@min(end_address, end_byte_address)) |byte_address|
            try w.writeByte(if (byte_address < start_address or byte_address >= end_address) ' ' else char: {
                const byte = macho.mf.memory_map.memory[byte_address];
                break :char if (std.ascii.isPrint(byte)) byte else '.';
            });
        try w.writeByte('\n');
    }
}

fn flushFixupChains(
    macho: *MachO,
    segment: Segment,
    page: u16,
) void {
    assert(!std.mem.eql(u8, segment.name(macho), "__LINKEDIT"));

    const page_size = macho.targetPageAlign().toByteUnits();
    const start_vaddr = segment.vaddr(macho) + @as(u64, page) * page_size;
    const end_vaddr = start_vaddr + page_size;

    var next_vaddr: u64 = 0;
    var node_it = segment.node(macho).last(&macho.mf);
    while (node_it.unwrap()) |node| : (node_it = node.prev(&macho.mf)) {
        // Other than `__LINKEDIT`, segments can only contain sections.
        const sect: Section = switch (macho.getNode(node)) {
            .section, .section_manual_size => |sect| sect,
            .macho => unreachable,
            .macho_header => continue, // the Mach header and load commands are in the __TEXT section
            .symtab => unreachable,
            .strtab => unreachable,
            .indirect_syms => unreachable,
            .section_indirect_syms => unreachable,
            .linkedit_data => unreachable,
            .fixup_chains_header => unreachable,
            .fixup_chains_imports => unreachable,
            .fixup_chains_symbol_names => unreachable,
            .fixup_chains_starts => unreachable,
            .fixup_chains_starts_in_image => unreachable,
            .fixup_chains_starts_in_segment => unreachable,
            .segment => unreachable,
            .input_atom => unreachable,
        };
        if (sect.vaddr(macho) >= end_vaddr) {
            // This section is after the page we're interested in. We're iterating backwards, so
            // that means we've not yet found the page.
            continue;
        }
        if (sect.vaddr(macho) + macho.targetLoad(&sect.lcPtrConst(macho).size) < start_vaddr) {
            // This section is before the page we're interested in. We're iterating backwards, so
            // that means we've finished scanning the page.
            break;
        }
        if (sect == macho.known_sections.got) {
            macho.flushFixupChainsGot(
                &next_vaddr,
                start_vaddr,
                start_vaddr + page_size,
            );
        } else {
            var sub_node_it = node.last(&macho.mf);
            while (sub_node_it.unwrap()) |sub_node| : (sub_node_it = sub_node.prev(&macho.mf)) {
                macho.flushFixupChainsNonSpecial(
                    sub_node,
                    &next_vaddr,
                    start_vaddr,
                    start_vaddr + page_size,
                );
            }
        }
    }

    const page_starts: []u16 = @ptrCast(@alignCast(
        segment.fixupChainStartsNode(macho).unwrap().?.slice(&macho.mf)[@sizeOf(std.macho.dyld_chained_starts_in_segment) - 2 ..],
    ));
    if (next_vaddr == 0) {
        macho.targetStore(&page_starts[page], 0xFFFF);
    } else {
        macho.targetStore(&page_starts[page], @intCast(next_vaddr - start_vaddr));
    }
}
fn flushFixupChainsGot(
    macho: *MachO,
    next_vaddr: *u64,
    start_vaddr: u64,
    end_vaddr: u64,
) void {
    const ptr_size = macho.targetPtrSize();
    const got_start_vaddr = macho.known_sections.got.vaddr(macho);
    const got_end_vaddr = got_start_vaddr + @as(u64, macho.got.count()) * ptr_size;
    if (got_start_vaddr >= end_vaddr) return;
    if (got_end_vaddr <= start_vaddr) return;
    // There is some overlap between the GOT and the page range.
    const start_index = @divExact(start_vaddr -| got_start_vaddr, ptr_size);
    const end_index = @min(@divExact(end_vaddr -| got_start_vaddr, ptr_size), macho.got.count());

    const got_node = macho.known_sections.got.node(macho);
    const got_syms = macho.got.keys();

    var got_index = end_index;
    while (got_index > start_index) {
        got_index -= 1;
        macho.writeOneFixup(
            got_syms[got_index],
            0,
            got_node,
            got_index * ptr_size,
            start_vaddr,
            end_vaddr,
            next_vaddr,
        );
    }
}
fn flushFixupChainsNonSpecial(
    macho: *MachO,
    node: MappedFile.Node.Index,
    next_vaddr: *u64,
    start_vaddr: u64,
    end_vaddr: u64,
) void {
    switch (macho.getNode(node)) {
        .macho => unreachable,
        .macho_header => unreachable,
        .symtab => unreachable,
        .strtab => unreachable,
        .indirect_syms => unreachable,
        .section_indirect_syms => unreachable,
        .linkedit_data => unreachable,
        .fixup_chains_header => unreachable,
        .fixup_chains_imports => unreachable,
        .fixup_chains_symbol_names => unreachable,
        .fixup_chains_starts => unreachable,
        .fixup_chains_starts_in_image => unreachable,
        .fixup_chains_starts_in_segment => unreachable,
        .segment => unreachable,
        .section => unreachable,
        .section_manual_size => unreachable,
        .input_atom => |iai| {
            const first_reloc = iai.ptr(macho).first_reloc;

            var reloc = first_reloc;
            var prev_reloc_off: ?u32 = null;
            while (@backingInt(reloc) < macho.relocs.items.len and
                reloc.ptr(macho).node == node)
            {
                if (prev_reloc_off) |prev_off| {
                    assert(reloc.ptr(macho).offset > prev_off);
                }
                prev_reloc_off = reloc.ptr(macho).offset;
                reloc = @fromBackingInt(@backingInt(reloc) + 1);
            }

            while (reloc != first_reloc) {
                reloc = @fromBackingInt(@backingInt(reloc) - 1);
                if (reloc.ptr(macho).bits.result == .fixup) {
                    macho.writeOneFixup(
                        reloc.ptr(macho).target_symbol,
                        reloc.ptr(macho).addend,
                        reloc.ptr(macho).node,
                        reloc.ptr(macho).offset,
                        start_vaddr,
                        end_vaddr,
                        next_vaddr,
                    );
                }
            }
        },
    }
}
fn writeOneFixup(
    macho: *MachO,
    target: Symbol.Index,
    addend: u64,
    dest_node: MappedFile.Node.Index,
    dest_offset: u64,
    start_vaddr: u64,
    end_vaddr: u64,
    next_vaddr: *u64,
) void {
    assert(macho.targetPtrSize() == 8);
    const dest_vaddr = macho.getNodeVAddr(dest_node) + dest_offset;
    if (dest_vaddr < start_vaddr or dest_vaddr >= end_vaddr) return;
    const dest_ptr: *align(4) std.macho.dyld_chained_ptr_64 = @ptrCast(@alignCast(
        dest_node.slice(&macho.mf)[@intCast(dest_offset)..][0..8],
    ));
    switch (macho.targetLoad(&target.nlist(macho).n_type).bits.type) {
        .undf => {
            const import_index = macho.imports.getIndex(target).?;
            macho.targetStore(dest_ptr, .{
                .kind = .bind,
                .next = if (next_vaddr.* != 0) @intCast(
                    @divExact(next_vaddr.* - dest_vaddr, 4),
                ) else 0,
                .per_kind = .{ .bind = .{
                    .ordinal = @intCast(import_index),
                    .addend = std.math.cast(u8, addend) orelse {
                        std.debug.panic("TODO(MachO2): large fixup addend (target={q}, addend={d})", .{
                            target.name(macho), addend,
                        });
                    },
                } },
            });
            next_vaddr.* = dest_vaddr;
        },
        .sect => {
            const target_val: packed struct(u64) {
                low36: u36,
                mid: u20,
                high8: u8,
            } = @bitCast(target.value(macho) +% addend);
            if (target_val.mid != 0) {
                @panic("TODO(MachO2): report non-zero mid bits in rebase fixup (vmaddrs too large?)");
            }
            macho.targetStore(dest_ptr, .{
                .kind = .rebase,
                .next = if (next_vaddr.* != 0) @intCast(
                    @divExact(next_vaddr.* - dest_vaddr, 4),
                ) else 0,
                .per_kind = .{ .rebase = .{
                    .target = target_val.low36,
                    .high8 = target_val.high8,
                } },
            });
            next_vaddr.* = dest_vaddr;
        },
        .abs => {
            // We actually *don't* need a fixup---this is just a static relocation with extra steps!
            const target_val = target.value(macho) +% addend;
            macho.targetStore(dest_ptr, @bitCast(target_val));
        },
        else => unreachable,
    }
}

const std = @import("std");
const assert = std.debug.assert;
const log = std.log.scoped(.link);
const Allocator = std.mem.Allocator;
const Io = std.Io;

const aarch64 = @import("../codegen/aarch64.zig");
const link = @import("../link.zig");
const tapi = @import("tapi.zig");
const codegen = @import("../codegen.zig");
const Compilation = @import("../Compilation.zig");
const InternPool = @import("../InternPool.zig");
const MappedFile = @import("MappedFile.zig");
const Zcu = @import("../Zcu.zig");
