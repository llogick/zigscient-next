pub fn identify(r: *Io.Reader) error{ReadFailed}!bool {
    const magic = r.peek(std.elf.MAGIC.len) catch |err| switch (err) {
        error.EndOfStream => return false,
        else => |e| return e,
    };
    return std.mem.eql(u8, magic, std.elf.MAGIC);
}

pub fn dump(d: *Objdump) Objdump.Error!void {
    var arena_state: std.heap.ArenaAllocator = .init(d.gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const r = d.reader();
    const w = d.w;

    // Endian is irrelevant in this read because `std.elf.Ident` has no multi-byte fields.
    const ident = r.peekStruct(std.elf.Ident, .little) catch |err| switch (err) {
        error.EndOfStream => return d.fail("truncated ELF header", .{}),
        else => |e| return e,
    };
    assert(std.mem.eql(u8, &ident.magic, std.elf.MAGIC)); // checked by `identify`
    switch (ident.class) {
        .NONE => return d.fail("unsupported EI_CLASS value 'ELFCLASSNONE'", .{}),
        _ => return d.fail("unknown EI_CLASS value '{d}'", .{ident.class}),
        .@"32", .@"64" => {},
    }
    const endian: std.lang.Endian = switch (ident.data) {
        .NONE => return d.fail("unsupported EI_DATA value 'ELFDATANONE'", .{}),
        _ => return d.fail("unknown EI_DATA value '{d}'", .{ident.data}),
        .@"2LSB" => .little,
        .@"2MSB" => .big,
    };
    if (ident.version != 1) {
        return d.fail("unknown EI_VERSION value '{d}'", .{ident.version});
    }

    switch (ident.class) {
        .NONE, _ => unreachable,
        inline else => |class| {
            const ehdr = r.takeStruct(class.ElfN().Ehdr, endian) catch |err| switch (err) {
                error.EndOfStream => return d.fail("truncated ELF header", .{}),
                else => |e| return e,
            };
            if (ehdr.version != 1) {
                return d.fail("unknown ehdr version '{d}'", .{ehdr.version});
            }
            const file_type_str: []const u8 = switch (ehdr.type) {
                .NONE, _ => "file",
                .REL => "relocatable",
                // ET_EXEC vs ET_DYN differentiates position-dependent modules (which are always
                // executables) from position-independent ones (which may be executables or dynamic
                // libraries). There's no concise way to explain that difference here, so just write
                // the same thing for both.
                .EXEC, .DYN => "image",
                .CORE => "core dump",
            };
            try w.print("{s}: ELF {s}\n\n", .{ d.input_name, file_type_str });

            if (d.opts.file_headers) try w.print(
                \\ELF header
                \\       type: {s}
                \\    machine: {s}
                \\      entry: {x}
                \\      phoff: {x}
                \\      shoff: {x}
                \\      flags: {x}
                \\     ehsize: {x}
                \\  phentsize: {x}
                \\      phnum: {d}
                \\  shentsize: {x}
                \\      shnum: {d}
                \\   shstrndx: {d}
                \\
                \\
            , .{
                std.enums.tagName(std.elf.ET, ehdr.type) orelse "(unknown)",
                std.enums.tagName(std.elf.EM, ehdr.machine) orelse "(unknown)",
                d.fmtIntField(ehdr.entry, .address),
                d.fmtIntField(ehdr.phoff, .address),
                d.fmtIntField(ehdr.shoff, .address),
                d.fmtIntField(ehdr.flags.int, null),
                d.fmtIntField(ehdr.ehsize, .size),
                d.fmtIntField(ehdr.phentsize, .size),
                d.fmtIntField(ehdr.phnum, .size),
                d.fmtIntField(ehdr.shentsize, .size),
                d.fmtIntField(ehdr.shnum, .size),
                d.fmtIntField(ehdr.shstrndx, .index),
            });

            const sections: []const Section = scanSections(
                d,
                arena,
                class,
                endian,
                &ehdr,
            ) catch |err| switch (err) {
                error.EndOfStream => sections: {
                    try d.warn("failed to read shdrs: unexpected eof", .{});
                    break :sections &.{};
                },
                else => |e| return e,
            };

            const versions: []const ?AvailableSymbolVersion = try collectVersions(d, arena, sections);
            try validateSymbolVersions(d, sections, versions);

            if (d.opts.sections) {
                try dumpSections(d, sections);
            }
            scanAndDumpPhdrs(d, sections, class, endian, &ehdr) catch |err| switch (err) {
                error.EndOfStream => try d.warn("failed to read phdrs: unexpected eof", .{}),
                else => |e| return e,
            };
            if (d.opts.symbols) {
                try dumpSymbols(d, sections);
            }
            if (d.opts.symbol_versions) {
                try dumpSymbolVersions(d, sections, versions);
            }
            if (d.opts.version_definitions) {
                try dumpVersionDefinitions(d, sections);
            }
            if (d.opts.relocs) {
                try dumpRelocations(d, ehdr.machine, sections);
            }
            if (d.opts.dynamic) {
                try dumpDynamic(d, sections);
            }
        },
    }
}

const Section = struct {
    name: []const u8,
    name_offset: u32,
    type: std.elf.SHT,
    flags: std.elf.SHF,
    addr: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    addralign: u64,
    entsize: u64,

    /// Even if this section has an applicable type like `.STRTAB`, this field may be `.none` in the
    /// event that the section header or contents could not be parsed.
    contents: Contents,

    const Contents = union(enum) {
        none,
        strtab: []const u8,
        symtab: []const std.elf.Elf64.Sym,
        relocs: []const std.elf.Elf64.Rela,
        relr: []const u64,
        dynamic: []const std.elf.Elf64.Dyn,
        versym: []const std.elf.Versym,
        verdef: []const Verdef,
        verneed: []const Verneed,

        const Verdef = struct {
            ver_ndx: std.elf.VER_NDX,
            aux_cnt: u16,
            name: ?u32,
            hash: u32,
        };
        const Verneed = struct {
            ver_ndx: std.elf.VER_NDX,
            file: u32,
            name: u32,
            hash: u32,
            weak: bool,
        };
    };

    fn linkedContents(
        s: *const Section,
        comptime tag: std.meta.Tag(Contents),
        all_sections: []const Section,
    ) ?@FieldType(Contents, @tagName(tag)) {
        if (s.link == 0) return null;
        if (s.link >= all_sections.len) return null;
        return switch (all_sections[s.link].contents) {
            tag => |contents| contents,
            else => null,
        };
    }
};
const AvailableSymbolVersion = struct {
    name: []const u8,
    kind: Kind,
    const Kind = enum { defined, required };
};

/// Reads the shdrs of an ELF file into a slice of `Section` values.
///
/// Sections with special meanings, such as symbol tables and relocation sections, are validated,
/// parsed, and (if successfully parsed) will have their `contents` fields populated.
fn scanSections(
    d: *Objdump,
    arena: Allocator,
    comptime class: std.elf.CLASS,
    endian: std.lang.Endian,
    ehdr: *const class.ElfN().Ehdr,
) (Objdump.Error || error{EndOfStream})![]const Section {
    const ElfN = class.ElfN();
    const r = d.reader();

    if (ehdr.shnum == 0) return &.{};

    if (ehdr.shentsize < @sizeOf(ElfN.Shdr)) {
        try d.warn("failed to read shdrs: invalid shentsize '0x{x}'", .{ehdr.shentsize});
        return &.{};
    }

    const sections = try arena.alloc(Section, ehdr.shnum);

    try d.seekTo(ehdr.shoff);
    for (sections, 0..) |*section, shndx| {
        const shdr = try r.peekStruct(ElfN.Shdr, endian);
        try r.discardAll64(ehdr.shentsize);
        section.* = .{
            .name = try std.fmt.allocPrint(arena, "(section {d})", .{shndx}),
            .name_offset = shdr.name,
            .type = shdr.type,
            .flags = shdr.flags.shf,
            .addr = shdr.addr,
            .offset = shdr.offset,
            .size = shdr.size,
            .link = shdr.link,
            .info = shdr.info,
            .addralign = shdr.addralign,
            .entsize = shdr.entsize,
            .contents = .none,
        };
    }

    // Load the contents of all `SHT_STRTAB` sections. We do this first so that we can load the
    // section names from the shstrtab section.
    for (sections, 0..) |*section, shndx| {
        if (section.type != .STRTAB) continue;
        try d.seekTo(section.offset);
        if (r.readAllocAll(
            arena,
            std.math.cast(usize, section.size) orelse return error.OutOfMemory,
        )) |contents| {
            section.contents = .{ .strtab = contents };
        } else |err| switch (err) {
            error.EndOfStream => try d.warn(
                "unexpected eof reading {t} section at index '{d}'",
                .{ section.type, shndx },
            ),
            else => |e| return e,
        }
    }

    // Now populate section names (or warn if we can't).
    if (ehdr.shstrndx >= sections.len) {
        try d.warn("failed to populate section names: shstrndx '{d}' >= shnum '{d}'", .{
            ehdr.shstrndx, sections.len,
        });
    } else if (sections[ehdr.shstrndx].type != .STRTAB) {
        try d.warn("failed to populate section names: shstrndx is not a STRTAB section", .{});
    } else switch (sections[ehdr.shstrndx].contents) {
        else => unreachable,
        .none => try d.warn("failed to populate section names", .{}),
        .strtab => |shstrtab| for (sections, 0..) |*section, shndx| {
            if (section.name_offset >= shstrtab.len) {
                try d.warn("failed to populate name of section at index '{d}': bad offset '{d}' into string table", .{
                    shndx, section.name_offset,
                });
            } else {
                section.name = std.mem.sliceTo(shstrtab[section.name_offset..], 0);
            }
        },
    }

    // Now scan symbols, relocations, etc; validate the sections and populate `contents`.
    for (sections) |*section| {
        if (section.addralign != 0 and !std.math.isPowerOfTwo(section.addralign)) {
            try d.warn(
                "section {q} addralign is neither zero nor a power of two",
                .{section.name},
            );
        }

        switch (section.type) {
            .NULL,
            .PROGBITS,
            .NOBITS,
            .STRTAB,
            => {},

            .SYMTAB, .DYNSYM => scanSymtabSection(
                d,
                arena,
                class,
                ehdr.type,
                endian,
                sections,
                section,
            ) catch |err| switch (err) {
                error.EndOfStream => try d.warn("unexpected eof reading {t} section {q}", .{
                    section.type, section.name,
                }),
                else => |e| return e,
            },

            .REL => scanRelocationSection(
                d,
                arena,
                ElfN.Rel,
                ehdr.type,
                endian,
                sections,
                section,
            ) catch |err| switch (err) {
                error.EndOfStream => try d.warn("unexpected eof reading {t} section {q}", .{
                    section.type, section.name,
                }),
                else => |e| return e,
            },
            .RELA => scanRelocationSection(
                d,
                arena,
                ElfN.Rela,
                ehdr.type,
                endian,
                sections,
                section,
            ) catch |err| switch (err) {
                error.EndOfStream => try d.warn("unexpected eof reading {t} section {q}", .{
                    section.type, section.name,
                }),
                else => |e| return e,
            },
            .RELR => scanRelrSection(
                d,
                arena,
                class,
                ehdr.type,
                endian,
                sections,
                section,
            ) catch |err| switch (err) {
                error.EndOfStream => try d.warn("unexpected eof reading {t} section {q}", .{
                    section.type, section.name,
                }),
                else => |e| return e,
            },

            .INIT_ARRAY,
            .FINI_ARRAY,
            .PREINIT_ARRAY,
            => if (section.entsize != class.size()) {
                try d.warn("{t} section {q} has entsize '{d}' (need exactly '{d}')", .{
                    section.type, section.name, section.entsize, class.size(),
                });
            } else if (section.size % section.entsize != 0) {
                try d.warn("{t} section {q} has size '{d}' which is not divisible by entsize '{d}'", .{
                    section.type, section.name, section.size, section.entsize,
                });
            },

            .DYNAMIC => scanDynamicSection(
                d,
                arena,
                class,
                ehdr.type,
                endian,
                sections,
                section,
            ) catch |err| switch (err) {
                error.EndOfStream => try d.warn("unexpected eof reading {t} section {q}", .{
                    section.type, section.name,
                }),
                else => |e| return e,
            },

            // TODO: validate/parse dynamic symbol hash table
            .HASH => {},

            // TODO: do we need to validate or parse anything from these?
            .NOTE => {},
            .GROUP => {},
            .SYMTAB_SHNDX => {},
            .SHLIB => {},

            _ => switch (section.type) {
                .GNU_VERSYM => scanVersymSection(
                    d,
                    arena,
                    ehdr.type,
                    endian,
                    sections,
                    section,
                ) catch |err| switch (err) {
                    error.EndOfStream => try d.warn("unexpected eof reading {t} section {q}", .{
                        section.type, section.name,
                    }),
                    else => |e| return e,
                },
                .GNU_VERDEF => scanVerdefSection(
                    d,
                    arena,
                    ehdr.type,
                    endian,
                    sections,
                    section,
                ) catch |err| switch (err) {
                    error.EndOfStream => try d.warn("unexpected eof reading {t} section {q}", .{
                        section.type, section.name,
                    }),
                    else => |e| return e,
                },
                .GNU_VERNEED => scanVerneedSection(
                    d,
                    arena,
                    ehdr.type,
                    endian,
                    sections,
                    section,
                ) catch |err| switch (err) {
                    error.EndOfStream => try d.warn("unexpected eof reading {t} section {q}", .{
                        section.type, section.name,
                    }),
                    else => |e| return e,
                },

                // TODO: validate/parse dynamic symbol hash table
                .GNU_HASH => {},

                // TODO: do we need to validate or parse anything from these?
                .LLVM_ADDRSIG => {},
                .X86_64_UNWIND => {},

                else => {
                    try d.warn("section {q} has unknown type '{d}'", .{ section.name, section.type });
                },
            },
        }
    }

    // Finally, validate any indices into other sections (e.g. relocation entries contain indices
    // into a symtab section). We couldn't do this in the previous loop because it's possible for an
    // earlier section to reference a later one.
    for (sections) |*section| {
        switch (section.contents) {
            .none => {},
            .strtab => {},
            .symtab => |symbols| if (section.linkedContents(.strtab, sections)) |strtab| {
                for (symbols, 0..) |*symbol, symbol_index| {
                    if (symbol.name >= strtab.len) {
                        try d.warn("symbol at index '{d}' in {q} has invalid offset '{d}' into {q}", .{
                            symbol_index, section.name, symbol.name, sections[section.link].name,
                        });
                    }
                }
            },
            .relr => {},
            .relocs => |relocs| if (section.linkedContents(.symtab, sections)) |symtab| {
                for (relocs, 0..) |*reloc, reloc_index| {
                    if (reloc.info.sym >= symtab.len) {
                        try d.warn("relocation at index '{d}' in {q} targets invalid symbol index '{d}'", .{
                            reloc_index, section.name, reloc.info.sym,
                        });
                    }
                }
            },
            .dynamic => |dynamic| if (section.linkedContents(.strtab, sections)) |strtab| {
                for (dynamic, 0..) |*dyn, dyn_index| {
                    if (dyn.tag.dt.valueType() == .string and dyn.val >= strtab.len) {
                        try d.warn("dynamic entry at index '{d}' in {q} has invalid offset '{d}' into {q}", .{
                            dyn_index, section.name, dyn.val, sections[section.link].name,
                        });
                    }
                }
            },
            .versym => |versym| if (section.linkedContents(.symtab, sections)) |symtab| {
                if (versym.len != symtab.len) {
                    try d.warn("length of symbol version table {q} does not match length of corresponding symbol table {q}", .{
                        section.name, sections[section.link].name,
                    });
                }
            },
            .verdef => |verdef| if (section.linkedContents(.strtab, sections)) |strtab| {
                for (verdef) |def| {
                    if (def.name) |name_off| {
                        if (name_off >= strtab.len) try d.warn(
                            "version definition '{d}' in {q} has invalid offset '{d}' into {q}",
                            .{ def.ver_ndx, section.name, name_off, sections[section.link].name },
                        ) else {
                            const name_slice = std.mem.sliceTo(strtab[name_off..], 0);
                            const expect_hash = std.elf.hash.calculate(name_slice);
                            if (def.hash != expect_hash) try d.warn(
                                "incorrect hash 0x{x:0>8} for version {q} in {q} (expected 0x{x:0>8})",
                                .{ def.hash, name_slice, section.name, expect_hash },
                            );
                        }
                    }
                }
            },
            .verneed => |verneed| if (section.linkedContents(.strtab, sections)) |strtab| {
                for (verneed) |need| {
                    if (need.file >= strtab.len) try d.warn(
                        "file name for required version '{d}' in {q} has invalid offset '{d}' into {q}",
                        .{ need.ver_ndx, section.name, need.file, sections[section.link].name },
                    );
                    if (need.name >= strtab.len) try d.warn(
                        "required version '{d}' in {q} has invalid offset '{d}' into {q}",
                        .{ need.ver_ndx, section.name, need.name, sections[section.link].name },
                    ) else {
                        const name_slice = std.mem.sliceTo(strtab[need.name..], 0);
                        const expect_hash = std.elf.hash.calculate(name_slice);
                        if (need.hash != expect_hash) try d.warn(
                            "incorrect hash 0x{x:0>8} for version {q} in {q} (expected 0x{x:0>8})",
                            .{ need.hash, name_slice, section.name, expect_hash },
                        );
                    }
                }
            },
        }
    }

    return sections;
}

fn scanSymtabSection(
    d: *Objdump,
    arena: Allocator,
    comptime class: std.elf.CLASS,
    elf_type: std.elf.ET,
    endian: std.lang.Endian,
    all_sections: []const Section,
    section: *Section,
) (Objdump.Error || error{EndOfStream})!void {
    const ElfN = class.ElfN();
    const r = d.reader();

    if (section.type == .DYNSYM and elf_type == .REL) {
        try d.warn("{t} section {q} in relocatable", .{
            section.type, section.name,
        });
    }

    if (section.entsize < @sizeOf(ElfN.Sym)) {
        try d.warn("{t} section {q} has bad entsize '{d}' (need at least '{d}')", .{
            section.type, section.name, section.entsize, @sizeOf(ElfN.Sym),
        });
        return;
    }

    if (section.size % section.entsize != 0) {
        try d.warn("{t} section {q} has size '{d}' which is not divisible by entsize '{d}'", .{
            section.type, section.name, section.size, section.entsize,
        });
        return;
    }

    // Problems with `section.link` are non-fatal.
    if (section.link == 0 or section.link >= all_sections.len) {
        try d.warn("{t} section {q} links to invalid section index '{d}'", .{
            section.type, section.name, section.link,
        });
    } else if (all_sections[section.link].type != .STRTAB) {
        try d.warn("{t} section {q} links to non-strtab section {q}", .{
            section.type, section.name, all_sections[section.link].name,
        });
    }

    const nsyms = @divExact(section.size, section.entsize);
    const symbols = try arena.alloc(
        std.elf.Elf64.Sym,
        std.math.cast(usize, nsyms) orelse return error.OutOfMemory,
    );
    try d.seekTo(section.offset);
    for (symbols) |*symbol| {
        const sym = try r.peekStruct(ElfN.Sym, endian);
        try r.discardAll64(section.entsize);
        symbol.* = .{
            .name = sym.name,
            .info = sym.info,
            .other = sym.other,
            .shndx = sym.shndx,
            .value = sym.value,
            .size = sym.size,
        };
    }
    section.contents = .{ .symtab = symbols };
}

fn scanRelocationSection(
    d: *Objdump,
    arena: Allocator,
    comptime ElfReloc: type,
    elf_type: std.elf.ET,
    endian: std.lang.Endian,
    all_sections: []const Section,
    section: *Section,
) (Objdump.Error || error{EndOfStream})!void {
    const r = d.reader();

    if (section.entsize < @sizeOf(ElfReloc)) {
        try d.warn("{t} section {q} has bad entsize '{d}' (need at least '{d}')", .{
            section.type, section.name, section.entsize, @sizeOf(ElfReloc),
        });
        return;
    }

    if (section.size % section.entsize != 0) {
        try d.warn("{t} section {q} has size '{d}' which is not divisible by entsize '{d}'", .{
            section.type, section.name, section.size, section.entsize,
        });
        return;
    }

    // Problems with `section.link` are non-fatal.
    if (section.link == 0 or section.link >= all_sections.len) {
        try d.warn("{t} section {q} links to invalid section index '{d}'", .{
            section.type, section.name, section.link,
        });
    } else switch (all_sections[section.link].type) {
        .SYMTAB, .DYNSYM => {},
        else => try d.warn("{t} section {q} links to non-symtab section {q}", .{
            section.type, section.name, all_sections[section.link].name,
        }),
    }

    // Problems with `section.info` are non-fatal.
    if (section.info >= all_sections.len) {
        try d.warn("{t} section {q} against invalid section index '{d}'", .{
            section.type, section.name, section.info,
        });
    } else if (section.info == 0 and elf_type == .REL) {
        try d.warn("dynamic {t} section {q} in relocatable", .{
            section.type, section.name,
        });
    }

    const nrelocs = @divExact(section.size, section.entsize);
    const relocs = try arena.alloc(
        std.elf.Elf64.Rela,
        std.math.cast(usize, nrelocs) orelse return error.OutOfMemory,
    );
    try d.seekTo(section.offset);
    for (relocs) |*reloc| {
        const elf_reloc = try r.peekStruct(ElfReloc, endian);
        try r.discardAll64(section.entsize);
        reloc.* = .{
            .offset = elf_reloc.offset,
            .info = .{
                .type = elf_reloc.info.type,
                .sym = elf_reloc.info.sym,
            },
            .addend = elf_reloc.addend,
        };
    }
    section.contents = .{ .relocs = relocs };
}

fn scanRelrSection(
    d: *Objdump,
    arena: Allocator,
    comptime class: std.elf.CLASS,
    elf_type: std.elf.ET,
    endian: std.lang.Endian,
    all_sections: []const Section,
    section: *Section,
) (Objdump.Error || error{EndOfStream})!void {
    const r = d.reader();

    switch (elf_type) {
        .REL => try d.warn("{t} section {q} in relocatable", .{
            section.type, section.name,
        }),
        .EXEC => try d.warn("{t} section {q} in position-dependent executable", .{
            section.type, section.name,
        }),
        .DYN => {},
        .NONE, .CORE, _ => {},
    }

    if (section.entsize != class.size()) {
        try d.warn("{t} section {q} has entsize '{d}' (need exactly '{d}')", .{
            section.type, section.name, section.entsize, class.size() * 2,
        });
        return;
    }

    if (section.size % section.entsize != 0) {
        try d.warn("{t} section {q} has size '{d}' which is not divisible by entsize '{d}'", .{
            section.type, section.name, section.size, section.entsize,
        });
        return;
    }

    // Problems with `section.info` are fatal here, because we can't read the "addends" without them.
    if (section.info >= all_sections.len) {
        try d.warn("{t} section {q} against invalid section index '{d}'", .{
            section.type, section.name, section.info,
        });
    }

    const num_entries = std.math.cast(
        usize,
        @divExact(section.size, section.entsize),
    ) orelse return error.OutOfMemory;

    var reloc_addrs: std.ArrayList(u64) = .empty;
    try reloc_addrs.ensureUnusedCapacity(arena, @intCast(num_entries)); // not sufficient, but a minimum
    try d.seekTo(section.offset);
    var next_address: ?class.ElfN().Addr = null;
    for (0..num_entries) |_| {
        const relr: class.ElfN().Relr = @fromBackingInt(
            try r.takeInt(class.ElfN().Addr, endian),
        );
        switch (relr.info.tag) {
            .address => {
                try reloc_addrs.append(arena, relr.address);
                next_address = relr.address +% class.size();
            },
            .chain => {
                const chain = relr.info.chain;
                const start_address = next_address orelse {
                    try d.warn("{t} section {q} missing initial address", .{
                        section.type, section.name,
                    });
                    return;
                };
                const bitmap_size: u32 = @bitSizeOf(@TypeOf(chain));
                try reloc_addrs.ensureUnusedCapacity(arena, @popCount(chain));
                for (0..bitmap_size) |bit_index| {
                    const bit: u1 = @truncate(chain >> @intCast(bit_index));
                    switch (bit) {
                        0 => {},
                        1 => reloc_addrs.appendAssumeCapacity(start_address + bit_index * class.size()),
                    }
                }
                next_address = start_address +% bitmap_size * class.size();
            },
        }
    }
    section.contents = .{ .relr = reloc_addrs.items };
}

fn scanDynamicSection(
    d: *Objdump,
    arena: Allocator,
    comptime class: std.elf.CLASS,
    elf_type: std.elf.ET,
    endian: std.lang.Endian,
    all_sections: []const Section,
    section: *Section,
) (Objdump.Error || error{EndOfStream})!void {
    const r = d.reader();

    if (elf_type == .REL) {
        try d.warn("{t} section {q} in relocatable", .{
            section.type, section.name,
        });
    }

    if (section.entsize != class.size() * 2) {
        try d.warn("{t} section {q} has entsize '{d}' (need exactly '{d}')", .{
            section.type, section.name, section.entsize, class.size() * 2,
        });
        return;
    }

    if (section.size % section.entsize != 0) {
        try d.warn("{t} section {q} has size '{d}' which is not divisible by entsize '{d}'", .{
            section.type, section.name, section.size, section.entsize,
        });
        return;
    }

    // Problems with `section.link` are non-fatal.
    if (section.link == 0 or section.link >= all_sections.len) {
        try d.warn("{t} section {q} links to invalid section index '{d}'", .{
            section.type, section.name, section.link,
        });
    } else if (all_sections[section.link].type != .STRTAB) {
        try d.warn("{t} section {q} links to non-strtab section {q}", .{
            section.type, section.name, all_sections[section.link].name,
        });
    }

    const max_entries = std.math.cast(
        usize,
        @divExact(section.size, section.entsize),
    ) orelse return error.OutOfMemory;
    const entries = try arena.alloc(std.elf.Elf64.Dyn, max_entries);
    try d.seekTo(section.offset);
    const num_entries: usize = for (entries, 0..) |*entry, entry_index| {
        const dyn = try r.takeStruct(class.ElfN().Dyn, endian);
        entry.* = .{ .tag = .{ .dt = dyn.tag.dt }, .val = dyn.val };
        switch (dyn.tag.dt) {
            .NULL => break entry_index + 1,

            .STRTAB => if (section.link != 0 and
                section.link < all_sections.len and
                all_sections[section.link].type == .STRTAB and
                dyn.val != all_sections[section.link].addr)
            {
                try d.warn("DT_STRTAB entry at index '{d}' in {t} section {q} does not match address of linked {t} section {q}", .{
                    entry_index,
                    section.type,
                    section.name,
                    all_sections[section.link].type,
                    all_sections[section.link].name,
                });
            },

            else => {},
        }
    } else n: {
        try d.warn("{t} section {q} has no terminating DT_NULL entry", .{
            section.type, section.name,
        });
        break :n max_entries;
    };
    section.contents = .{ .dynamic = entries[0..num_entries] };

    // If the section has space for more entries, they should all be `DT_NULL` entries.
    for (num_entries..max_entries) |_| {
        const trailing_dyn = try r.takeStruct(class.ElfN().Dyn, endian);
        if (trailing_dyn.tag.dt != .NULL) {
            try d.warn("trailing entry in {t} section {q} is not DT_NULL", .{
                section.type, section.name,
            });
            break;
        }
    }
}

fn scanVersymSection(
    d: *Objdump,
    arena: Allocator,
    elf_type: std.elf.ET,
    endian: std.lang.Endian,
    all_sections: []const Section,
    section: *Section,
) (Objdump.Error || error{EndOfStream})!void {
    const r = d.reader();

    if (elf_type == .REL) {
        try d.warn("{t} section {q} in relocatable", .{
            section.type, section.name,
        });
    }

    if (section.entsize != @sizeOf(u16)) {
        try d.warn("{t} section {q} has entsize '{d}' (need exactly '{d}')", .{
            section.type, section.name, section.entsize, @sizeOf(u16),
        });
        return;
    }

    if (section.size % section.entsize != 0) {
        try d.warn("{t} section {q} has size '{d}' which is not divisible by entsize '{d}'", .{
            section.type, section.name, section.size, section.entsize,
        });
        return;
    }

    // Problems with `section.link` are non-fatal.
    if (section.link == 0 or section.link >= all_sections.len) {
        try d.warn("{t} section {q} links to invalid section index '{d}'", .{
            section.type, section.name, section.link,
        });
    } else if (all_sections[section.link].type != .DYNSYM) {
        try d.warn("{t} section {q} links to non-dynsym section {q}", .{
            section.type, section.name, all_sections[section.link].name,
        });
    }

    const num_versions = std.math.cast(
        usize,
        @divExact(section.size, section.entsize),
    ) orelse return error.OutOfMemory;
    try d.seekTo(section.offset);
    const versions = try r.readSliceEndianAlloc(arena, std.elf.Versym, num_versions, endian);
    section.contents = .{ .versym = versions };
}

fn scanVerdefSection(
    d: *Objdump,
    arena: Allocator,
    elf_type: std.elf.ET,
    endian: std.lang.Endian,
    all_sections: []const Section,
    section: *Section,
) (Objdump.Error || error{EndOfStream})!void {
    const r = d.reader();

    if (elf_type == .REL) {
        try d.warn("{t} section {q} in relocatable", .{
            section.type, section.name,
        });
    }

    // Problems with `section.link` are non-fatal.
    if (section.link == 0 or section.link >= all_sections.len) {
        try d.warn("{t} section {q} links to invalid section index '{d}'", .{
            section.type, section.name, section.link,
        });
    } else if (all_sections[section.link].type != .STRTAB) {
        try d.warn("{t} section {q} links to non-strtab section {q}", .{
            section.type, section.name, all_sections[section.link].name,
        });
    }

    // `section.info` is the number of verdef entries, but some of them might have `cnt == 0` and
    // therefore not actually define any versions, so track the actual length with an arraylist.
    var versions: std.ArrayList(Section.Contents.Verdef) = .empty;
    try versions.ensureUnusedCapacity(arena, section.info);
    var verdef_offset: u64 = 0;
    for (0..section.info) |actual_verdef_index| {
        if (verdef_offset +| @sizeOf(std.elf.Verdef) > section.size) {
            return error.EndOfStream;
        }
        try d.seekTo(section.offset +| verdef_offset);
        const def = try r.peekStruct(std.elf.Verdef, endian);
        if (def.version != 1) {
            try d.warn("unsupported GNU symbol versioning version '{d}' in {q}", .{
                def.version, section.name,
            });
            return;
        }

        if (def.cnt == 0) {
            try d.warn("symbol version '{d}' defined in {q} with aux count of 0", .{
                def.ndx, section.name,
            });
        } else if (def.aux == 0) {
            try d.warn("symbol version '{d}' defined in {q} with aux offset 0 despite non-zero aux count", .{
                def.ndx, section.name,
            });
        } else {
            if (def.ndx == .LOCAL) {
                try d.warn("LOCAL symbol version defined in {q}", .{section.name});
            } else if (def.ndx == .GLOBAL) {
                if ((def.flags & std.elf.VER_FLG_BASE) == 0) try d.warn(
                    "symbol version GLOBAL defined in {q} without VER_FLG_BASE",
                    .{section.name},
                );
            } else {
                if ((def.flags & std.elf.VER_FLG_BASE) != 0) try d.warn(
                    "symbol version '{d}' defined in {q} with VER_FLG_BASE",
                    .{ def.ndx, section.name },
                );
            }
            if (verdef_offset +| def.aux +| @sizeOf(std.elf.Verdaux) > section.size) {
                return error.EndOfStream;
            }
            try d.seekTo(section.offset +| verdef_offset +| def.aux);
            const aux = try r.peekStruct(std.elf.Verdaux, endian);
            versions.appendAssumeCapacity(.{
                .ver_ndx = def.ndx,
                .aux_cnt = def.cnt,
                .name = aux.name,
                .hash = def.hash,
            });
        }

        if (def.next == 0) {
            if (actual_verdef_index + 1 != section.info) {
                try d.warn("version definition list in {q} has length '{d}' but section info specified '{d}'", .{
                    section.name, actual_verdef_index + 1, section.info,
                });
            }
            break;
        }

        if (actual_verdef_index + 1 == section.info) {
            try d.warn("version definition list in {q} continues beyond length '{d}' specified by section info", .{
                section.name, section.info,
            });
        } else {
            verdef_offset +|= def.next;
        }
    }

    section.contents = .{ .verdef = versions.items };

    for (versions.items) |version| {
        // In the `SHT_GNU_VERSYM` sections which actually assign versions to symbols, the top bit
        // of the version index is repurposed as a flag, so if a version index has that bit set, it
        // cannot be referenced.
        if (@as(std.elf.Versym, @bitCast(version.ver_ndx)).HIDDEN) try d.warn(
            "required version '{d}' in {q} is inaccessible because its index contains the HIDDEN flag",
            .{ version.ver_ndx, section.name },
        );
    }
}

fn scanVerneedSection(
    d: *Objdump,
    arena: Allocator,
    elf_type: std.elf.ET,
    endian: std.lang.Endian,
    all_sections: []const Section,
    section: *Section,
) (Objdump.Error || error{EndOfStream})!void {
    const r = d.reader();

    if (elf_type == .REL) {
        try d.warn("{t} section {q} in relocatable", .{
            section.type, section.name,
        });
    }

    // Problems with `section.link` are non-fatal.
    if (section.link == 0 or section.link >= all_sections.len) {
        try d.warn("{t} section {q} links to invalid section index '{d}'", .{
            section.type, section.name, section.link,
        });
    } else if (all_sections[section.link].type != .STRTAB) {
        try d.warn("{t} section {q} links to non-strtab section {q}", .{
            section.type, section.name, all_sections[section.link].name,
        });
    }

    var versions: std.ArrayList(Section.Contents.Verneed) = .empty;
    var verneed_offset: u64 = 0;
    for (0..section.info) |actual_verneed_index| {
        if (verneed_offset +| @sizeOf(std.elf.Verneed) > section.size) {
            return error.EndOfStream;
        }
        try d.seekTo(section.offset +| verneed_offset);
        const need = try r.peekStruct(std.elf.Verneed, endian);
        if (need.version != 1) {
            try d.warn("unsupported GNU symbol versioning version '{d}' in {q}", .{
                need.version, section.name,
            });
            return;
        }

        if (need.aux == 0) {
            if (need.cnt != 0) try d.warn(
                "verneed entry in {q} has aux offset 0 despite non-zero aux count",
                .{section.name},
            );
        } else if (need.cnt == 0) {
            try d.warn(
                "verneed entry in {q} has non-zero aux offset despite aux count of 0",
                .{section.name},
            );
        } else {
            var vernaux_offset = verneed_offset + need.aux;
            for (0..need.cnt) |actual_vernaux_index| {
                if (vernaux_offset +| @sizeOf(std.elf.Verdaux) > section.size) {
                    return error.EndOfStream;
                }
                try d.seekTo(section.offset +| vernaux_offset);
                const aux = try r.peekStruct(std.elf.Vernaux, endian);
                const ver_ndx: std.elf.VER_NDX = @fromBackingInt(aux.other);
                if (ver_ndx == .LOCAL or ver_ndx == .GLOBAL) try d.warn(
                    "symbol version {t} imported in {q} should only ever be defined",
                    .{ ver_ndx, section.name },
                );
                try versions.append(arena, .{
                    .ver_ndx = ver_ndx,
                    .file = need.file,
                    .name = aux.name,
                    .hash = aux.hash,
                    .weak = (aux.flags & std.elf.VER_FLG_WEAK) != 0,
                });
                if (aux.next == 0) {
                    if (actual_vernaux_index + 1 != need.cnt) {
                        try d.warn("vernaux list in {q} has length '{d}' but verneed.cnt is '{d}'", .{
                            section.name, actual_vernaux_index + 1, need.cnt,
                        });
                    }
                    break;
                }
                if (actual_vernaux_index + 1 == need.cnt) {
                    try d.warn("vernaux list in {q} continues beyond length '{d}' specified by verneed.cnt", .{
                        section.name, section.info,
                    });
                } else {
                    vernaux_offset +|= aux.next;
                }
            }
        }

        if (need.next == 0) {
            if (actual_verneed_index + 1 != section.info) {
                try d.warn("version requirement list in {q} has length '{d}' but section info specified '{d}'", .{
                    section.name, actual_verneed_index + 1, section.info,
                });
            }
            break;
        }

        if (actual_verneed_index + 1 == section.info) {
            try d.warn("version requirement list in {q} continues beyond length '{d}' specified by section info", .{
                section.name, section.info,
            });
        } else {
            verneed_offset +|= need.next;
        }
    }

    section.contents = .{ .verneed = versions.items };

    for (versions.items) |version| {
        // In the `SHT_GNU_VERSYM` sections which actually assign versions to symbols, the top bit
        // of the version index is repurposed as a flag, so if a version index has that bit set, it
        // cannot be referenced.
        if (@as(std.elf.Versym, @bitCast(version.ver_ndx)).HIDDEN) try d.warn(
            "version definition '{d}' in {q} is inaccessible because its index contains the HIDDEN flag",
            .{ version.ver_ndx, section.name },
        );
    }
}

fn collectVersions(
    d: *Objdump,
    arena: Allocator,
    sections: []const Section,
) Objdump.Error![]const ?AvailableSymbolVersion {
    var versions: std.ArrayList(?AvailableSymbolVersion) = .empty;
    for (sections) |section| switch (section.contents) {
        else => {},
        .verneed => |verneeds| for (verneeds) |verneed| {
            const name: []const u8 = name: {
                const strtab = section.linkedContents(.strtab, sections) orelse break :name "";
                if (verneed.name >= strtab.len) break :name "";
                break :name std.mem.sliceTo(strtab[verneed.name..], 0);
            };
            const need_len = @backingInt(verneed.ver_ndx) + 1;
            try versions.appendNTimes(arena, null, need_len -| versions.items.len);
            if (versions.items[@backingInt(verneed.ver_ndx)]) |existing| {
                try d.warn("version '{d}' has duplicate definitions {q} and {q}", .{
                    verneed.ver_ndx, existing.name, name,
                });
            } else {
                versions.items[@backingInt(verneed.ver_ndx)] = .{
                    .name = name,
                    .kind = .required,
                };
            }
        },
        .verdef => |verdefs| for (verdefs) |verdef| {
            const name: []const u8 = name: {
                const name_off = verdef.name orelse break :name "";
                const strtab = section.linkedContents(.strtab, sections) orelse break :name "";
                if (name_off >= strtab.len) break :name "";
                break :name std.mem.sliceTo(strtab[name_off..], 0);
            };
            const need_len = @backingInt(verdef.ver_ndx) + 1;
            try versions.appendNTimes(arena, null, need_len -| versions.items.len);
            if (versions.items[@backingInt(verdef.ver_ndx)]) |existing| {
                try d.warn("version '{d}' has duplicate definitions {q} and {q}", .{
                    verdef.ver_ndx, existing.name, name,
                });
            } else {
                versions.items[@backingInt(verdef.ver_ndx)] = .{
                    .name = name,
                    .kind = .defined,
                };
            }
        },
    };
    return versions.items;
}
fn validateSymbolVersions(
    d: *Objdump,
    sections: []const Section,
    versions: []const ?AvailableSymbolVersion,
) Objdump.Error!void {
    for (sections) |versym_section| {
        const versym = switch (versym_section.contents) {
            .versym => |versym| versym,
            else => continue,
        };
        const symtab = versym_section.linkedContents(.symtab, sections) orelse continue;
        if (symtab.len != versym.len) continue;

        for (symtab, versym, 0..) |symbol, version, symbol_index| {
            const ver_ndx: std.elf.VER_NDX = @fromBackingInt(version.VERSION);
            switch (symbol.info.bind) {
                .LOCAL => {
                    if (ver_ndx != .LOCAL) {
                        try d.warn("{s} symbol at index '{d}' in {q} has version index '{d}' in {q}", .{
                            symbolBindString(symbol.info.bind),
                            symbol_index,
                            sections[versym_section.link].name,
                            fmtVerNdx(d, ver_ndx),
                            versym_section.name,
                        });
                    }
                    continue;
                },
                .GLOBAL, .WEAK, .GNU_UNIQUE => {}, // handled below
                else => continue, // invalid bind already reported
            }

            // In theory, undefined unversioned symbols should use VER_NDX_LOCAL, and defined
            // unversioned symbols should use VER_NDX_GLOBAL. However, GNU ld versions 2.35--2.45
            // had a bug where GLOBAL was used for both cases (https://sourceware.org/PR33577). This
            // is ultimately pretty harmless, and those binaries are widespread, so we allow GLOBAL
            // for any symbol (but LOCAL for undefined symbols only).
            switch (ver_ndx) {
                .GLOBAL => continue,
                .LOCAL => {
                    if (symbol.shndx != std.elf.SHN_UNDEF) {
                        try d.warn("defined symbol at index '{d}' in {q} has version index '{d}' in {q}", .{
                            symbol_index,
                            sections[versym_section.link].name,
                            fmtVerNdx(d, ver_ndx),
                            versym_section.name,
                        });
                    }
                    continue;
                },
                else => {}, // handled below
            }
            const maybe_version_info: ?AvailableSymbolVersion = info: {
                if (@backingInt(ver_ndx) >= versions.len) break :info null;
                break :info versions[@backingInt(ver_ndx)];
            };
            const version_info = maybe_version_info orelse {
                try d.warn("symbol at index '{d}' in {q} has invalid version index '{d}' in {q}", .{
                    symbol_index,
                    sections[versym_section.link].name,
                    fmtVerNdx(d, ver_ndx),
                    versym_section.name,
                });
                continue;
            };
            switch (version_info.kind) {
                .defined => if (symbol.shndx == std.elf.SHN_UNDEF) {
                    try d.warn("undefined symbol at index '{d}' in {q} has locally defined version {q} in {q}", .{
                        symbol_index,
                        sections[versym_section.link].name,
                        version_info.name,
                        versym_section.name,
                    });
                },
                // In theory it makes sense to error if the symbol is defined but has a "required"
                // version; but this can actually happen if a copy relocation exists for the symbol.
                // We could try to check for that, but this could cause confusing symbol versioning
                // errors if e.g. the relocation section is corrupted; so instead, just allow this.
                .required => {},
            }
        }
    }
}

fn dumpSections(d: *const Objdump, sections: []const Section) Objdump.Error!void {
    const w = d.w;

    if (sections.len == 0) return;

    try w.print("{d} section headers\n", .{d.fmtIntField(sections.len, .size)});
    try w.print("  {[index]s: >5}  {[type]s: <13}  {[flags]s: ^12}  {[offset]s: ^10}  {[address]s: ^18}  {[size]s: ^18}  {[entsize]s: ^10}  {[addralign]s: ^10}  {[link]s: >6}  {[info]s: >6}  {[name]s}\n", .{
        .index = "index",
        .type = "type",
        .flags = "flags",
        .offset = "offset",
        .address = "address",
        .size = "size",
        .entsize = "entsize",
        .addralign = "addralign",
        .link = "link",
        .info = "info",
        .name = "name",
    });
    for (sections, 0..) |section, shndx| {
        for (d.opts.section_filters) |filter| {
            if (std.mem.find(u8, section.name, filter) != null) break;
        } else if (d.opts.section_filters.len > 0) {
            continue;
        }
        try w.print("  {d: >5}  {s: <13}  {f}  {x:0>8}  {x:0>16}  {x:0>16}  {x:0>8}  {x:0>8}  {d: >6}  {d: >6}  {s}\n", .{
            d.fmtIntField(shndx, .index),
            sectionTypeString(section.type),
            fmtSectionFlags(d, section.flags),
            d.fmtIntField(section.offset, .offset),
            d.fmtIntField(section.addr, .address),
            d.fmtIntField(section.size, .size),
            d.fmtIntField(section.entsize, .size),
            d.fmtIntField(section.addralign, .@"align"),
            d.fmtIntField(section.link, .index),
            d.fmtIntField(section.info, .index),
            section.name,
        });
    }

    try w.writeAll(
        \\
        \\  flags:  [A] ALLOC        [M] MERGE        [I] INFO_LINK
        \\          [W] WRITE        [S] STRINGS      [L] LINK_ORDER
        \\          [X] EXECINSTR    [G] GROUP        [O] OS_NONCONFORMING
        \\          [T] TLS          [C] COMPRESSED   [?] unknown
        \\
        \\
    );
}

/// Because we don't need to store phdrs for anything, we don't bother allocating a buffer for
/// them. Instead, if `--phdrs` was passed, we dump phdrs *while* scanning them for errors.
fn scanAndDumpPhdrs(
    d: *Objdump,
    sections: []const Section,
    comptime class: std.elf.CLASS,
    endian: std.lang.Endian,
    ehdr: *const class.ElfN().Ehdr,
) (Objdump.Error || error{EndOfStream})!void {
    const ElfN = class.ElfN();
    const r = d.reader();
    const w = d.w;

    if (ehdr.phnum == 0) return;

    if (ehdr.phentsize < @sizeOf(ElfN.Phdr)) {
        try d.warn("failed to read phdrs: invalid phentsize '0x{x}'", .{ehdr.phentsize});
        return;
    }

    try d.seekTo(ehdr.phoff);
    if (d.opts.phdrs) {
        try w.print("{d} program headers\n", .{d.fmtIntField(ehdr.phnum, .size)});
        try w.print("  {[index]s: >5}  {[type]s: <13}  {[flags]s: ^5}  {[offset]s: ^10}  {[filesz]s: ^18}  {[paddr]s: ^18}  {[vaddr]s: ^18}  {[memsz]s: ^18}  {[align]s: ^10}  {[sections]s}\n", .{
            .index = "index",
            .type = "type",
            .flags = "flags",
            .offset = "offset",
            .filesz = "filesz",
            .paddr = "paddr",
            .vaddr = "vaddr",
            .memsz = "memsz",
            .@"align" = "align",
            .sections = "sections",
        });
    }
    for (0..ehdr.phnum) |phndx| {
        const phdr = try r.peekStruct(ElfN.Phdr, endian);
        try r.discardAll64(ehdr.phentsize);

        if (phdr.@"align" != 0) {
            if (!std.math.isPowerOfTwo(phdr.@"align")) {
                try d.warn(
                    "phdr '{d}' alignment is neither zero nor a power of two",
                    .{phndx},
                );
            } else if (phdr.offset % phdr.@"align" != phdr.vaddr % phdr.@"align") {
                try d.warn(
                    "phdr '{d}' vaddr != offset modulo alignment",
                    .{phndx},
                );
            }
        }

        if (d.opts.phdrs) {
            try w.print("  {d: >5}  {s: <13}  {f}   {x:0>8}  {x:0>16}  {x:0>16}  {x:0>16}  {x:0>16}  {x:0>8} ", .{
                d.fmtIntField(phndx, .index),
                phdrTypeString(phdr.type),
                fmtPhdrFlags(d, phdr.flags),
                d.fmtIntField(phdr.offset, .offset),
                d.fmtIntField(phdr.filesz, .size),
                d.fmtIntField(phdr.paddr, .address),
                d.fmtIntField(phdr.vaddr, .address),
                d.fmtIntField(phdr.memsz, .size),
                d.fmtIntField(phdr.@"align", .@"align"),
            });
            switch (phdr.type) {
                // GNU_STACK segments don't correspond to any address range so do not have any
                // corresponding sections; showing any would be bogus.
                .GNU_STACK => {},
                else => for (sections) |section| {
                    if (!section.flags.ALLOC) continue;
                    // In TLS phdrs, `memsz` is actually the size of each dynamically-initialized
                    // TLS block, so when computing segment overlap, consider the `filesz` instead.
                    // An exception is made for TLS sections, since they are supposed to exist in
                    // each dynamically-initialized TLS block.
                    // don't exist in memory normally, and are only populated when the dynamic TLS
                    // blocks are initialized, so we actually care about those fake addresses, not
                    // the address of the TLS template.
                    const effective_memsz: u64 = switch (phdr.type) {
                        else => phdr.memsz,
                        .TLS => if (section.flags.TLS) phdr.memsz else phdr.filesz,
                    };
                    if (section.type == .NOBITS) {
                        // NOBITS sections are meaningful if they appear in the range [filesz,memsz)
                        // in the segment (i.e. in the segment's "zero-fill" range).
                        if (section.addr < phdr.vaddr + phdr.filesz) continue;
                        if (section.addr >= phdr.vaddr + effective_memsz) continue;
                    } else {
                        // Other segments are meaningful if they appear in the range (0,memsz) in
                        // the segment (i.e. anywhere within it, but not in the boundary).
                        if (section.addr + section.size <= phdr.vaddr) continue;
                        if (section.addr >= phdr.vaddr + effective_memsz) continue;
                    }
                    try w.print(" {s}", .{section.name});
                },
            }
            try w.writeByte('\n');
        }
    }

    if (d.opts.phdrs) try w.writeByte('\n');
}

fn dumpSymbols(
    d: *const Objdump,
    sections: []const Section,
) Objdump.Error!void {
    const w = d.w;

    for (sections) |symtab_section| {
        const symbols = switch (symtab_section.contents) {
            .symtab => |symtab| symtab,
            else => continue,
        };

        for (d.opts.section_filters) |filter| {
            if (std.mem.find(u8, symtab_section.name, filter) != null) break;
        } else if (d.opts.section_filters.len > 0) {
            continue;
        }

        const symbol_kind_str: []const u8 = switch (symtab_section.type) {
            .SYMTAB => "symbols",
            .DYNSYM => "dynamic symbols",
            else => unreachable,
        };

        try w.print("{d} {s} in section {q}\n", .{
            d.fmtIntField(symbols.len, .size),
            symbol_kind_str,
            symtab_section.name,
        });
        try w.print("  {[index]s: >5}  {[type]s: <10}  {[bind]s: <10}  {[visibility]s: <10}  {[shndx]s: >5}  {[value]s: ^18}  {[size]s: ^18}  {[name]s}\n", .{
            .index = "index",
            .type = "type",
            .bind = "bind",
            .visibility = "visibility",
            .shndx = "shndx",
            .value = "value",
            .size = "size",
            .name = "name",
        });
        for (symbols, 0..) |symbol, symbol_index| {
            const sym_name: []const u8 = name: {
                const strtab = symtab_section.linkedContents(.strtab, sections) orelse break :name "";
                if (symbol.name >= strtab.len) break :name "";
                break :name std.mem.sliceTo(strtab[symbol.name..], 0);
            };
            for (d.opts.symbol_filters) |filter| {
                if (std.mem.find(u8, sym_name, filter) != null) break;
            } else if (d.opts.symbol_filters.len > 0) {
                continue;
            }
            try w.print("  {d: >5}  {s: <10}  {s: <10}  {t: <10}  {f}  {x:0>16}  {x:0>16}  {s}\n", .{
                d.fmtIntField(symbol_index, .index),
                symbolTypeString(symbol.info.type),
                symbolBindString(symbol.info.bind),
                symbol.other.visibility,
                fmtShndx(d, symbol.shndx),
                d.fmtIntField(symbol.value, .address),
                d.fmtIntField(symbol.size, .size),
                sym_name,
            });
        }
        try w.writeByte('\n');
    }
}

fn dumpSymbolVersions(
    d: *const Objdump,
    sections: []const Section,
    available_versions: []const ?AvailableSymbolVersion,
) Objdump.Error!void {
    const w = d.w;

    for (sections) |versym_section| {
        const versym = switch (versym_section.contents) {
            .versym => |versym| versym,
            else => continue,
        };
        const symtab = versym_section.linkedContents(.symtab, sections) orelse continue;
        if (symtab.len != versym.len) continue;

        try w.print("{d} symbol versions in section {q}\n", .{
            d.fmtIntField(versym.len, .size),
            versym_section.name,
        });
        try w.print("  {[sym]s: >7}  {[ver_ndx]s: >7}  {[flags]s: ^6}  {[ver_name]s: <24}  {[sym_name]s}\n", .{
            .sym = "sym",
            .ver_ndx = "ver_ndx",
            .flags = "flags",
            .ver_name = "ver_name",
            .sym_name = "sym_name",
        });
        for (symtab, versym, 0..) |symbol, version, symbol_index| {
            const sym_name: []const u8 = name: {
                const strtab = sections[versym_section.link].linkedContents(
                    .strtab,
                    sections,
                ) orelse break :name "";
                if (symbol.name >= strtab.len) break :name "";
                break :name std.mem.sliceTo(strtab[symbol.name..], 0);
            };
            for (d.opts.symbol_filters) |filter| {
                if (std.mem.find(u8, sym_name, filter) != null) break;
            } else if (d.opts.symbol_filters.len > 0) {
                continue;
            }
            const ver_ndx: std.elf.VER_NDX = @fromBackingInt(version.VERSION);
            const ver_name: []const u8 = switch (ver_ndx) {
                .LOCAL, .GLOBAL => "",
                else => name: {
                    if (@backingInt(ver_ndx) >= available_versions.len) break :name "<invalid>";
                    if (available_versions[@backingInt(ver_ndx)]) |version_info| {
                        break :name version_info.name;
                    } else {
                        break :name "<invalid>";
                    }
                },
            };
            try w.print("  {d: >7}  {d: >7}  {s: ^6}  {s: <24}  {s}\n", .{
                d.fmtIntField(symbol_index, .index),
                fmtVerNdx(d, ver_ndx),
                if (version.HIDDEN) "HIDDEN" else "",
                ver_name,
                sym_name,
            });
        }
        try w.writeByte('\n');
    }
}

fn dumpVersionDefinitions(
    d: *const Objdump,
    sections: []const Section,
) Objdump.Error!void {
    const w = d.w;

    for (sections) |section| switch (section.contents) {
        else => {},
        .verdef => |versions| {
            for (d.opts.section_filters) |filter| {
                if (std.mem.find(u8, section.name, filter) != null) break;
            } else if (d.opts.section_filters.len > 0) {
                continue;
            }
            try w.print("{d} version definitions in section {q}\n", .{
                d.fmtIntField(versions.len, .size),
                section.name,
            });
            try w.print("  {[ver_ndx]s: >7}  {[aux_cnt]s: >7}  {[name]s}\n", .{
                .ver_ndx = "ver_ndx",
                .aux_cnt = "aux_cnt",
                .name = "name",
            });
            for (versions) |version| {
                const name: []const u8 = name: {
                    const name_off = version.name orelse break :name "";
                    const strtab = section.linkedContents(.strtab, sections) orelse break :name "";
                    if (name_off >= strtab.len) break :name "";
                    break :name std.mem.sliceTo(strtab[name_off..], 0);
                };
                try w.print("  {d: >7}  {d: >7}  {s}\n", .{
                    fmtVerNdx(d, version.ver_ndx),
                    version.aux_cnt,
                    name,
                });
            }
            try w.writeByte('\n');
        },
        .verneed => |versions| {
            for (d.opts.section_filters) |filter| {
                if (std.mem.find(u8, section.name, filter) != null) break;
            } else if (d.opts.section_filters.len > 0) {
                continue;
            }
            try w.print("{d} version requirements in section {q}\n", .{
                d.fmtIntField(versions.len, .size),
                section.name,
            });
            try w.print("  {[ver_ndx]s: >7}  {[flags]s: <5}  {[name]s: <24}  {[file]s}\n", .{
                .ver_ndx = "ver_ndx",
                .flags = "flags",
                .name = "name",
                .file = "file",
            });
            for (versions) |version| {
                const file_name: []const u8 = name: {
                    const strtab = section.linkedContents(.strtab, sections) orelse break :name "";
                    if (version.file >= strtab.len) break :name "";
                    break :name std.mem.sliceTo(strtab[version.file..], 0);
                };
                const ver_name: []const u8 = name: {
                    const strtab = section.linkedContents(.strtab, sections) orelse break :name "";
                    if (version.name >= strtab.len) break :name "";
                    break :name std.mem.sliceTo(strtab[version.name..], 0);
                };
                try w.print("  {d: >7}  {s: <5}  {s: <24}  {s}\n", .{
                    fmtVerNdx(d, version.ver_ndx),
                    if (version.weak) "WEAK" else "",
                    ver_name,
                    file_name,
                });
            }
            try w.writeByte('\n');
        },
    };
}

fn dumpRelocations(
    d: *const Objdump,
    machine: std.elf.EM,
    sections: []const Section,
) Objdump.Error!void {
    const w = d.w;

    for (sections) |relocs_section| {
        const relocs = switch (relocs_section.contents) {
            .relocs => |relocs| relocs,
            .relr => |reloc_addrs| {
                for (d.opts.section_filters) |filter| {
                    if (std.mem.find(u8, relocs_section.name, filter) != null) break;
                } else if (d.opts.section_filters.len > 0) {
                    continue;
                }

                try w.print("{d} relative relocations in section {q}\n", .{
                    d.fmtIntField(reloc_addrs.len, .size),
                    relocs_section.name,
                });
                try w.print("  {[index]s: >5}  {[offset]s: ^18}\n", .{
                    .index = "index",
                    .offset = "offset",
                });
                for (reloc_addrs, 0..) |reloc_addr, reloc_index| {
                    try w.print("  {d: >5}  {x:0>16}\n", .{
                        reloc_index,
                        d.fmtIntField(reloc_addr, .address),
                    });
                }
                try w.writeByte('\n');
                continue;
            },
            else => continue,
        };

        for (d.opts.section_filters) |filter| {
            if (std.mem.find(u8, relocs_section.name, filter) != null) break;
        } else if (d.opts.section_filters.len > 0) {
            continue;
        }

        var type_col_width: usize = 4;
        for (relocs) |reloc| {
            if (d.opts.symbol_filters.len > 0) {
                if (reloc.info.sym == 0) continue;
                const symtab = relocs_section.linkedContents(.symtab, sections) orelse continue;
                if (reloc.info.sym >= symtab.len) continue;
                const sym = &symtab[reloc.info.sym];
                const sym_name: []const u8 = if (sym.info.type == .SECTION and
                    sym.shndx != std.elf.SHN_UNDEF and
                    sym.shndx < sections.len)
                name: {
                    break :name sections[sym.shndx].name;
                } else name: {
                    const strtab = sections[relocs_section.link].linkedContents(.strtab, sections) orelse continue;
                    if (sym.name >= strtab.len) continue;
                    break :name std.mem.sliceTo(strtab[sym.name..], 0);
                };
                if (sym_name.len == 0) continue;
                for (d.opts.symbol_filters) |filter| {
                    if (std.mem.find(u8, sym_name, filter) != null) break;
                } else continue;
            }
            const reloc_type_str = relocTypeString(machine, reloc.info.type);
            type_col_width = @max(type_col_width, reloc_type_str.len);
        }

        if (relocs_section.info != 0 and relocs_section.info < sections.len) {
            try w.print("{d} relocations against section {q} in section {q}\n", .{
                d.fmtIntField(relocs.len, .size),
                sections[relocs_section.info].name,
                relocs_section.name,
            });
        } else {
            try w.print("{d} relocations in section {q}\n", .{
                d.fmtIntField(relocs.len, .size),
                relocs_section.name,
            });
        }
        try w.print("  {[index]s: >5}  {[offset]s: ^18}  {[type]s: <[type_width]}  {[sym]s: >6}  {[target]s}\n", .{
            .type_width = type_col_width,
            .index = "index",
            .offset = "offset",
            .type = "type",
            .sym = "sym",
            .target = "target",
        });

        for (relocs, 0..) |reloc, reloc_index| {
            const target_sym_name: []const u8 = target_name: {
                if (reloc.info.sym == 0) break :target_name "";
                const symtab = relocs_section.linkedContents(.symtab, sections) orelse {
                    break :target_name "<invalid>";
                };
                if (reloc.info.sym >= symtab.len) break :target_name "<invalid>";
                const sym = &symtab[reloc.info.sym];
                if (sym.info.type == .SECTION and
                    sym.shndx != std.elf.SHN_UNDEF and
                    sym.shndx < sections.len)
                {
                    break :target_name sections[sym.shndx].name;
                }
                const strtab = sections[relocs_section.link].linkedContents(.strtab, sections) orelse {
                    break :target_name "<invalid>";
                };
                if (sym.name >= strtab.len) break :target_name "<invalid>";
                const name = std.mem.sliceTo(strtab[sym.name..], 0);
                if (name.len == 0) break :target_name "<unnamed>";
                break :target_name name;
            };
            for (d.opts.symbol_filters) |filter| {
                if (std.mem.find(u8, target_sym_name, filter) != null) break;
            } else if (d.opts.symbol_filters.len > 0) {
                continue;
            }
            try w.print("  {[index]d: >5}  {[offset]x:0>16}  {[type]s: <[type_col_width]}  {[sym]d: >6}  ", .{
                .index = d.fmtIntField(reloc_index, .index),
                .offset = d.fmtIntField(reloc.offset, .offset),
                .type = relocTypeString(machine, reloc.info.type),
                .type_col_width = type_col_width,
                .sym = d.fmtIntField(reloc.info.sym, .index),
            });
            if (reloc.info.sym == 0) {
                if (reloc.addend < 0) try w.writeByte('-');
                try w.print("{x}", .{d.fmtIntField(@abs(reloc.addend), null)});
            } else {
                try w.writeAll(target_sym_name);
                if (reloc.addend != 0) {
                    const sign: u8 = if (reloc.addend < 0) '-' else '+';
                    try w.print(" {c} {x}", .{ sign, d.fmtIntField(@abs(reloc.addend), null) });
                }
            }
            try w.writeByte('\n');
        }
        try w.writeByte('\n');
    }
}

fn dumpDynamic(d: *const Objdump, sections: []const Section) Objdump.Error!void {
    const w = d.w;

    for (sections) |dynamic_section| {
        const dynamic = switch (dynamic_section.contents) {
            .dynamic => |dynamic| dynamic,
            else => continue,
        };

        for (d.opts.section_filters) |filter| {
            if (std.mem.find(u8, dynamic_section.name, filter) != null) break;
        } else if (d.opts.section_filters.len > 0) {
            continue;
        }

        try w.print("{d} dynamic entries in section {q}\n", .{
            d.fmtIntField(dynamic.len, .size),
            dynamic_section.name,
        });
        try w.print("  {[index]s: >5}  {[type]s: <16}  {[value]s: ^18}\n", .{
            .index = "index",
            .type = "type",
            .value = "value",
        });
        for (dynamic, 0..) |entry, entry_index| {
            const field_kind: ?Objdump.FieldKind = switch (entry.tag.dt.valueType()) {
                .none => null,
                .flags => null,
                .flags_1 => null,
                .reloc_type => null,
                .unknown => null,
                .address => .address,
                .string => .offset,
                .size => .size,
            };
            try w.print("  {d: >5}  {s: <16}  {x:0>16}", .{
                d.fmtIntField(entry_index, .index),
                std.enums.tagName(std.elf.DT, entry.tag.dt) orelse "(unknown)",
                d.fmtIntField(entry.val, field_kind),
            });
            switch (entry.tag.dt.valueType()) {
                .none, .unknown, .size => {},

                .address => {
                    // This address will typically match the address of a section.
                    for (sections) |referenced_section| {
                        if (!referenced_section.flags.ALLOC or
                            (referenced_section.flags.TLS and referenced_section.type == .NOBITS))
                        {
                            // These sections don't actually exist in memory at the given addresses,
                            // so don't consider them for this.
                            continue;
                        }
                        if (entry.val == referenced_section.addr) {
                            try w.print("  {s}", .{referenced_section.name});
                            break;
                        }
                    }
                },

                .string => if (dynamic_section.linkedContents(.strtab, sections)) |dynstr| {
                    if (entry.val < dynstr.len) {
                        const str_val = std.mem.sliceTo(dynstr[@intCast(entry.val)..], 0);
                        if (d.opts.basenames and (entry.tag.dt == .RUNPATH or entry.tag.dt == .RPATH)) {
                            // The rpath string contains paths to directories which in practice tend
                            // to be in the Zig cache directory; best to just omit the whole string.
                        } else {
                            try w.print("  {q}", .{str_val});
                        }
                    }
                },

                .reloc_type => if (std.enums.tagName(
                    std.elf.DT.RelocType,
                    @fromBackingInt(@truncate(entry.val)),
                )) |name| {
                    try w.print("  [{s}]", .{name});
                },

                .flags => {
                    const flags: std.elf.DT.Flags = @fromBackingInt(@truncate(entry.val));
                    var any_flags: bool = false;
                    inline for (@typeInfo(std.elf.DT.Flags).@"struct".field_names) |flag_name| {
                        if (comptime std.mem.eql(u8, flag_name, "unused")) continue;
                        if (@field(flags, flag_name)) {
                            try w.writeAll(if (!any_flags) "  [" else " | ");
                            try w.writeAll(flag_name);
                            any_flags = true;
                        }
                    }
                    if (any_flags) try w.writeByte(']');
                },

                .flags_1 => {
                    const flags: std.elf.DT.Flags1 = @fromBackingInt(@truncate(entry.val));
                    var any_flags: bool = false;
                    inline for (@typeInfo(std.elf.DT.Flags1).@"struct".field_names) |flag_name| {
                        if (comptime std.mem.eql(u8, flag_name, "unused")) continue;
                        if (@field(flags, flag_name)) {
                            try w.writeAll(if (!any_flags) "  [" else " | ");
                            try w.writeAll(flag_name);
                            any_flags = true;
                        }
                    }
                    if (any_flags) try w.writeByte(']');
                },
            }
            try w.writeByte('\n');
        }
        try w.writeByte('\n');
    }
}

fn sectionTypeString(sht: std.elf.SHT) []const u8 {
    return switch (sht) {
        .LLVM_ADDRSIG => "LLVM_ADDRSIG",
        .GNU_HASH => "GNU_HASH",
        .GNU_VERDEF => "GNU_VERDEF",
        .GNU_VERNEED => "GNU_VERNEED",
        .GNU_VERSYM => "GNU_VERSYM",
        .X86_64_UNWIND => "X86_64_UNWIND",
        else => std.enums.tagName(std.elf.SHT, sht) orelse "(unknown)",
    };
}
fn fmtSectionFlags(d: *const Objdump, shf: std.elf.SHF) FormatSectionFlags {
    return .{ .d = d, .shf = shf };
}
const FormatSectionFlags = struct {
    d: *const Objdump,
    shf: std.elf.SHF,
    pub fn format(ctx: FormatSectionFlags, w: *Io.Writer) Io.Writer.Error!void {
        const d = ctx.d;
        const shf = ctx.shf;

        var is_dim = false;

        // The order in which we print flags does not match the field order. Instead, this order is
        // chosen to put the most important & common flags near the start, which in turn means the
        // key we print in `dumpSections` (which matches this order) is easier to understand.
        try writeFlag(d, w, &is_dim, 'A', shf.ALLOC);
        try writeFlag(d, w, &is_dim, 'W', shf.WRITE);
        try writeFlag(d, w, &is_dim, 'X', shf.EXECINSTR);
        try writeFlag(d, w, &is_dim, 'T', shf.TLS);
        try writeFlag(d, w, &is_dim, 'M', shf.MERGE);
        try writeFlag(d, w, &is_dim, 'S', shf.STRINGS);
        try writeFlag(d, w, &is_dim, 'G', shf.GROUP);
        try writeFlag(d, w, &is_dim, 'C', shf.COMPRESSED);
        try writeFlag(d, w, &is_dim, 'I', shf.INFO_LINK);
        try writeFlag(d, w, &is_dim, 'L', shf.LINK_ORDER);
        try writeFlag(d, w, &is_dim, 'O', shf.OS_NONCONFORMING);

        const any_unknown = shf.unused3 != 0 or
            shf.unused12 != 0 or
            shf.OS.MASK != 0 or
            shf.PROC.MASK != 0;

        try writeFlag(d, w, &is_dim, '?', any_unknown);

        if (is_dim) try d.setColor(w, .reset);
    }
    fn writeFlag(d: *const Objdump, w: *Io.Writer, is_dim: *bool, flag_char: u8, flag_set: bool) Io.Writer.Error!void {
        if (flag_set) {
            if (is_dim.*) try d.setColor(w, .reset);
            is_dim.* = false;
            try w.writeByte(flag_char);
        } else {
            if (!is_dim.*) try d.setColor(w, .dim);
            is_dim.* = true;
            try w.writeByte('-');
        }
    }
};
fn fmtVerNdx(d: *const Objdump, ver_ndx: std.elf.VER_NDX) FormatVerNdx {
    return .{ .d = d, .ver_ndx = ver_ndx };
}
const FormatVerNdx = struct {
    d: *const Objdump,
    ver_ndx: std.elf.VER_NDX,
    pub fn formatNumber(
        ctx: FormatVerNdx,
        w: *Io.Writer,
        number: std.fmt.Number,
    ) Io.Writer.Error!void {
        switch (ctx.ver_ndx) {
            .LOCAL, .GLOBAL => |ver_ndx| try w.alignBuffer(
                @tagName(ver_ndx),
                number.width orelse 0,
                number.alignment,
                number.fill,
            ),
            else => |ver_ndx| try Objdump.FormatIntField.formatNumber(.{
                .d = ctx.d,
                .val = @backingInt(ver_ndx),
                .kind = .index,
            }, w, number),
        }
    }
};

fn phdrTypeString(pt: std.elf.PT) []const u8 {
    return switch (pt) {
        .GNU_EH_FRAME => "GNU_EH_FRAME",
        .GNU_PROPERTY => "GNU_PROPERTY",
        .GNU_STACK => "GNU_STACK",
        .GNU_RELRO => "GNU_RELRO",
        .SUNWBSS => "SUNWBSS",
        .SUNWSTACK => "SUNWSTACK",
        else => std.enums.tagName(std.elf.PT, pt) orelse "(unknown)",
    };
}
fn fmtPhdrFlags(d: *const Objdump, pf: std.elf.PF) FormatPhdrFlags {
    return .{ .d = d, .pf = pf };
}
const FormatPhdrFlags = struct {
    d: *const Objdump,
    pf: std.elf.PF,
    pub fn format(ctx: FormatPhdrFlags, w: *Io.Writer) Io.Writer.Error!void {
        const d = ctx.d;
        const pf = ctx.pf;

        var is_dim = false;

        try FormatSectionFlags.writeFlag(d, w, &is_dim, 'R', pf.R);
        try FormatSectionFlags.writeFlag(d, w, &is_dim, 'W', pf.W);
        try FormatSectionFlags.writeFlag(d, w, &is_dim, 'X', pf.X);

        const any_unknown = pf.unused3 != 0 or
            pf.OS.MASK != 0 or
            pf.PROC.MASK != 0;

        try FormatSectionFlags.writeFlag(d, w, &is_dim, '?', any_unknown);

        if (is_dim) try d.setColor(w, .reset);
    }
};

fn symbolTypeString(stt: std.elf.STT) []const u8 {
    return switch (stt) {
        .GNU_IFUNC => "GNU_IFUNC",
        else => std.enums.tagName(std.elf.STT, stt) orelse "(unknown)",
    };
}
fn symbolBindString(stb: std.elf.STB) []const u8 {
    return switch (stb) {
        .GNU_UNIQUE => "GNU_UNIQUE",
        else => std.enums.tagName(std.elf.STB, stb) orelse "(unknown)",
    };
}

fn relocTypeString(machine: std.elf.EM, reloc_type: u32) []const u8 {
    return switch (machine) {
        else => switch (reloc_type) {
            0 => "NONE",
            else => "(unknown)",
        },
        .X86_64 => std.enums.tagName(std.elf.R_X86_64, @fromBackingInt(reloc_type)) orelse "(unknown)",
        .AARCH64 => std.enums.tagName(std.elf.R_AARCH64, @fromBackingInt(reloc_type)) orelse "(unknown)",
        .RISCV => std.enums.tagName(std.elf.R_RISCV, @fromBackingInt(reloc_type)) orelse "(unknown)",
        .PPC64 => std.enums.tagName(std.elf.R_PPC64, @fromBackingInt(reloc_type)) orelse "(unknown)",
        .SPARCV9 => std.enums.tagName(std.elf.R_SPARC, @fromBackingInt(reloc_type)) orelse "(unknown)",
        .LOONGARCH => std.enums.tagName(std.elf.R_LARCH, @fromBackingInt(reloc_type)) orelse "(unknown)",
    };
}

fn fmtShndx(d: *const Objdump, shndx: u16) FormatShndx {
    return .{ .d = d, .shndx = shndx };
}
const FormatShndx = struct {
    d: *const Objdump,
    shndx: u16,
    pub fn format(ctx: FormatShndx, w: *Io.Writer) Io.Writer.Error!void {
        switch (ctx.shndx) {
            std.elf.SHN_UNDEF => try w.writeAll("UNDEF"),
            std.elf.SHN_ABS => try w.writeAll("  ABS"),
            else => |shndx| try w.print("{d: >5}", .{ctx.d.fmtIntField(shndx, .index)}),
        }
    }
};

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Io = std.Io;

const Objdump = @import("../Objdump.zig");
